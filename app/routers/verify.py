import shutil
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.models.label import LabelFields
from app.services.comparator import compare_label
from app.services.db import save_verification
from app.services.ollama import extract_label_fields

router = APIRouter()

_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(upload: UploadFile, prefix: str) -> Path:
    dest = _UPLOAD_DIR / f"{prefix}_{uuid.uuid4()}_{upload.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@router.post("/extract")
async def extract(
    image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(default=None),
    verbose: bool = Query(default=False),
):
    tmp = _save_upload(image, "tmp")
    tmp_back: Optional[Path] = None
    try:
        if back_image and back_image.filename:
            tmp_back = _save_upload(back_image, "tmp_back")
        debug_info: Optional[dict] = {} if verbose else None
        fields = await extract_label_fields(tmp, tmp_back, debug_info=debug_info)
        result = fields.model_dump()
        if verbose and debug_info is not None:
            result["_debug"] = debug_info
        return result
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model took too long — try again")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Verification service unavailable — is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {str(e)}")
    finally:
        tmp.unlink(missing_ok=True)
        if tmp_back:
            tmp_back.unlink(missing_ok=True)


@router.post("/verify")
async def verify(
    image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(default=None),
    brand_name: str = Form(default=""),
    class_type: str = Form(default=""),
    alcohol_content: str = Form(default=""),
    net_contents: str = Form(default=""),
    producer_name_address: str = Form(default=""),
    country_of_origin: str = Form(default=""),
    government_warning: str = Form(default=""),
    contains_sulfites: str = Form(default=""),
):
    tmp = _save_upload(image, "tmp")
    tmp_back: Optional[Path] = None
    try:
        if back_image and back_image.filename:
            tmp_back = _save_upload(back_image, "tmp_back")

        extracted = await extract_label_fields(tmp, tmp_back)
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
        result = compare_label(extracted, form_data)
        save_verification(
            image_filename=image.filename,
            form_data=form_data.model_dump(),
            extracted=extracted.model_dump(),
            results=result.model_dump(),
            overall_pass=result.overall_pass,
        )
        return result.model_dump()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model took too long — try again")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Verification service unavailable — is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Verification failed: {str(e)}")
    finally:
        tmp.unlink(missing_ok=True)
        if tmp_back:
            tmp_back.unlink(missing_ok=True)
