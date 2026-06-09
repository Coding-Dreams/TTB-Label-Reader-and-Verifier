from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.db import save_verification

router = APIRouter()


class SaveGroupRequest(BaseModel):
    image_filename: str
    back_image_filename: Optional[str] = None
    extracted: dict
    overall_pass: bool
    compliance: Optional[dict] = None
    batch_id: Optional[str] = None


@router.post("/batch/save-group")
def save_group(body: SaveGroupRequest):
    filename = body.image_filename
    if body.back_image_filename:
        filename = f"{body.image_filename} + {body.back_image_filename}"
    results = {"overall_pass": body.overall_pass, "compliance": body.compliance or {}}
    save_verification(
        image_filename=filename,
        form_data=body.extracted,
        extracted=body.extracted,
        results=results,
        overall_pass=body.overall_pass,
        batch_id=body.batch_id,
    )
    return {"ok": True}
