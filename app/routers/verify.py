import shutil
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.label import LabelFields
from app.services.comparator import compare_label
from app.services.db import save_verification
from app.services.ollama import extract_label_fields

router = APIRouter()

_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/extract")
async def extract(image: UploadFile = File(...)):
    tmp = _UPLOAD_DIR / f"tmp_{uuid.uuid4()}_{image.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(image.file, f)
        fields = await extract_label_fields(tmp)
        return fields.model_dump()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model took too long — try again")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Verification service unavailable — is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {str(e)}")
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/verify")
async def verify(
    image: UploadFile = File(...),
    brand_name: str = Form(default=""),
    class_type: str = Form(default=""),
    alcohol_content: str = Form(default=""),
    net_contents: str = Form(default=""),
    producer_name_address: str = Form(default=""),
    country_of_origin: str = Form(default=""),
    government_warning: str = Form(default=""),
    contains_sulfites: str = Form(default=""),
):
    tmp = _UPLOAD_DIR / f"tmp_{uuid.uuid4()}_{image.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(image.file, f)

        extracted = await extract_label_fields(tmp)
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
    finally:
        tmp.unlink(missing_ok=True)
