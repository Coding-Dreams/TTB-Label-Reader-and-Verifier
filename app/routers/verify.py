import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image

from thefuzz import fuzz
from app.models.label import LabelFields
from app.services import log_bus
from app.services.comparator import compare_label, _CANONICAL_WARNING
from app.services.compliance import check_compliance
from app.services.db import save_verification
from app.services.ollama import extract_label_fields

router = APIRouter()
logger = logging.getLogger(__name__)

# Reuse the debug trace logger from ollama.py
_dbg = logging.getLogger("debug.trace")

_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}
_MAX_UPLOAD_MB = 20
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024


def _save_upload(upload: UploadFile, prefix: str) -> Path:
    """Read, validate (type + size), then write to disk. Raises HTTPException on bad input."""
    data = upload.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — maximum {_MAX_UPLOAD_MB} MB per image",
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
        if fmt not in _ALLOWED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format '{fmt}'. Allowed: JPEG, PNG, WebP, TIFF, BMP",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=415, detail="File does not appear to be a valid image")
    safe_name = Path(upload.filename or "upload").name
    dest = _UPLOAD_DIR / f"{prefix}_{uuid.uuid4()}_{safe_name}"
    dest.write_bytes(data)
    return dest


@router.post("/extract")
async def extract(
    image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(default=None),
    verbose: bool = Query(default=False),
):
    _t0 = time.monotonic()
    fname = image.filename or "image"
    _dbg.debug("[/extract] >>>>>>>>>> REQUEST START file=%s", fname)
    tmp = _save_upload(image, "tmp")
    tmp_back: Optional[Path] = None
    try:
        if back_image and back_image.filename:
            tmp_back = _save_upload(back_image, "tmp_back")
        _dbg.debug("[/extract] Files saved, calling extract_label_fields")
        await log_bus.emit(f"Extract request received — {fname}")
        debug_info: Optional[dict] = {} if verbose else None
        meta: dict = {}
        fields = await extract_label_fields(tmp, tmp_back, debug_info=debug_info, metadata=meta)
        _dbg.debug("[/extract] extract_label_fields returned, building response")
        result = fields.model_dump()
        gw = result.get("government_warning")
        result["government_warning_confidence"] = (
            fuzz.ratio(gw.lower(), _CANONICAL_WARNING.lower())
            if gw and gw != "GOVERNMENT WARNING"
            else None
        )
        result["government_warning_prefix_bold"] = meta.get("government_warning_prefix_bold")
        result["compliance"] = check_compliance(result)
        if verbose and debug_info is not None:
            result["_debug"] = debug_info
        _dbg.debug("[/extract] Returning JSON response (%.1fs total)", time.monotonic() - _t0)
        return result
    except httpx.TimeoutException:
        _dbg.debug("[/extract] TIMEOUT EXCEPTION (%.1fs)", time.monotonic() - _t0)
        await log_bus.emit("Error: model timed out")
        raise HTTPException(status_code=504, detail="Model took too long — try again")
    except httpx.ConnectError:
        _dbg.debug("[/extract] CONNECT ERROR (%.1fs)", time.monotonic() - _t0)
        await log_bus.emit("Error: Ollama service unavailable")
        raise HTTPException(status_code=503, detail="Verification service unavailable — is Ollama running?")
    except Exception as e:
        _dbg.debug("[/extract] EXCEPTION: %s: %s (%.1fs)", type(e).__name__, e, time.monotonic() - _t0)
        logger.exception("Extraction failed")
        await log_bus.emit("Error: extraction failed — check server logs")
        raise HTTPException(status_code=422, detail="Extraction failed — please try again")
    finally:
        _dbg.debug("[/extract] finally: deleting temp files")
        tmp.unlink(missing_ok=True)
        if tmp_back:
            tmp_back.unlink(missing_ok=True)
        _dbg.debug("[/extract] <<<<<<<<<< REQUEST END (%.1fs total)", time.monotonic() - _t0)


@router.post("/verify-fields")
async def verify_fields(
    brand_name: str = Form(default="", max_length=512),
    class_type: str = Form(default="", max_length=128),
    alcohol_content: str = Form(default="", max_length=128),
    net_contents: str = Form(default="", max_length=128),
    producer_name_address: str = Form(default="", max_length=1024),
    country_of_origin: str = Form(default="", max_length=128),
    government_warning: str = Form(default="", max_length=512),
    contains_sulfites: str = Form(default="", max_length=256),
):
    """Verify user-confirmed field values without re-extracting from the image.
    Treats whatever is in the form as ground truth; only compliance is checked."""
    fields = LabelFields(
        brand_name=brand_name or None,
        class_type=class_type or None,
        alcohol_content=alcohol_content or None,
        net_contents=net_contents or None,
        producer_name_address=producer_name_address or None,
        country_of_origin=country_of_origin or None,
        government_warning=government_warning or None,
        contains_sulfites=contains_sulfites or None,
    )
    compliance = check_compliance(fields.model_dump())

    _FIELD_NAMES = [
        "brand_name", "class_type", "alcohol_content", "net_contents",
        "producer_name_address", "country_of_origin", "government_warning",
        "contains_sulfites",
    ]
    field_results = [
        {
            "field": f,
            "status": "pass" if getattr(fields, f) else "not_detected",
            "extracted_value": getattr(fields, f),
            "submitted_value": getattr(fields, f),
        }
        for f in _FIELD_NAMES
    ]
    form_match = {
        "overall_pass": True,
        "fields": field_results,
    }
    overall_pass = compliance["compliant"]
    response = {
        "overall_pass": overall_pass,
        "form_match": form_match,
        "compliance": compliance,
    }
    save_verification(
        image_filename="(manual entry)",
        form_data=fields.model_dump(),
        extracted=fields.model_dump(),
        results=response,
        overall_pass=overall_pass,
    )
    return response


@router.post("/verify")
async def verify(
    image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(default=None),
    brand_name: str = Form(default="", max_length=512),
    class_type: str = Form(default="", max_length=128),
    alcohol_content: str = Form(default="", max_length=128),
    net_contents: str = Form(default="", max_length=128),
    producer_name_address: str = Form(default="", max_length=1024),
    country_of_origin: str = Form(default="", max_length=128),
    government_warning: str = Form(default="", max_length=512),
    contains_sulfites: str = Form(default="", max_length=256),
):
    tmp = _save_upload(image, "tmp")
    tmp_back: Optional[Path] = None
    try:
        if back_image and back_image.filename:
            tmp_back = _save_upload(back_image, "tmp_back")

        await log_bus.emit(f"Verify request received — {image.filename or 'image'}")
        extracted = await extract_label_fields(tmp, tmp_back)
        await log_bus.emit("Comparing extracted fields against submitted form")
        form_data = LabelFields(
            brand_name=brand_name or None,
            class_type=class_type or None,
            alcohol_content=alcohol_content or None,
            net_contents=net_contents or None,
            producer_name_address=producer_name_address or None,
            country_of_origin=country_of_origin or None,
            government_warning=government_warning or None,
            contains_sulfites=contains_sulfites or None,
        )
        match_result = compare_label(extracted, form_data)
        compliance = check_compliance(extracted.model_dump())

        # Verification passes only when both the application matches the extraction
        # AND the label itself satisfies TTB rules. Either failing makes the whole
        # submission a fail.
        overall_pass = match_result.overall_pass and compliance["compliant"]
        response = {
            "overall_pass": overall_pass,
            "form_match": match_result.model_dump(),
            "compliance": compliance,
        }

        save_verification(
            image_filename=image.filename,
            form_data=form_data.model_dump(),
            extracted=extracted.model_dump(),
            results=response,
            overall_pass=overall_pass,
        )
        status = "PASS" if overall_pass else "FAIL"
        await log_bus.emit(f"Verification complete — {status}")
        return response
    except httpx.TimeoutException:
        await log_bus.emit("Error: model timed out")
        raise HTTPException(status_code=504, detail="Model took too long — try again")
    except httpx.ConnectError:
        await log_bus.emit("Error: Ollama service unavailable")
        raise HTTPException(status_code=503, detail="Verification service unavailable — is Ollama running?")
    except Exception as e:
        logger.exception("Verification failed")
        await log_bus.emit("Error: verification failed — check server logs")
        raise HTTPException(status_code=422, detail="Verification failed — please try again")
    finally:
        tmp.unlink(missing_ok=True)
        if tmp_back:
            tmp_back.unlink(missing_ok=True)
