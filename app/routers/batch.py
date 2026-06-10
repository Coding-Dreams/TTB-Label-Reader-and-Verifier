import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.services.compliance import check_compliance
from app.services.db import save_verification

router = APIRouter()
logger = logging.getLogger(__name__)

_VALID_EXTRACTED_KEYS = frozenset({
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "country_of_origin", "government_warning", "contains_sulfites",
})


class SaveGroupRequest(BaseModel):
    image_filename: str = Field(max_length=512)
    back_image_filename: Optional[str] = Field(default=None, max_length=512)
    extracted: dict
    compliance: Optional[dict] = None
    batch_id: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_extracted(self) -> "SaveGroupRequest":
        unknown = set(self.extracted.keys()) - _VALID_EXTRACTED_KEYS
        if unknown:
            raise ValueError(f"Unexpected keys in extracted: {unknown}")
        for key, value in self.extracted.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"extracted.{key} must be a string or null")
            if isinstance(value, str) and len(value) > 2048:
                raise ValueError(f"extracted.{key} exceeds maximum length")
        return self


@router.post("/batch/save-group")
def save_group(body: SaveGroupRequest):
    filename = body.image_filename
    if body.back_image_filename:
        filename = f"{body.image_filename} + {body.back_image_filename}"
    server_compliance = check_compliance(body.extracted)
    overall_pass = server_compliance["compliant"]
    # Use server-computed compliance — never trust the client-supplied value
    results = {"overall_pass": overall_pass, "compliance": server_compliance}
    try:
        save_verification(
            image_filename=filename,
            form_data=body.extracted,
            extracted=body.extracted,
            results=results,
            overall_pass=overall_pass,
            batch_id=body.batch_id,
        )
    except Exception as exc:
        logger.exception("Failed to save batch group")
        raise HTTPException(status_code=500, detail="Failed to save — please try again")
    return {"ok": True}
