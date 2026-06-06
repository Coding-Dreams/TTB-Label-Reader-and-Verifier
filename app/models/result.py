from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class FieldStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    NOT_DETECTED = "not_detected"


class FieldResult(BaseModel):
    field: str
    status: FieldStatus
    extracted_value: Optional[str] = None
    submitted_value: Optional[str] = None
    score: Optional[float] = None


class VerificationResult(BaseModel):
    overall_pass: bool
    fields: List[FieldResult]
