import re
from typing import Optional
from thefuzz import fuzz
from app.models.label import LabelFields
from app.models.result import FieldResult, FieldStatus, VerificationResult

_REQUIRED_WARNING_PREFIX = "GOVERNMENT WARNING:"


def check_government_warning(
    extracted: Optional[str], submitted: Optional[str]
) -> FieldResult:
    if extracted is None:
        return FieldResult(
            field="government_warning",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=None,
            submitted_value=submitted,
        )
    passes = _REQUIRED_WARNING_PREFIX in extracted
    return FieldResult(
        field="government_warning",
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def check_fuzzy_field(
    field: str,
    extracted: Optional[str],
    submitted: Optional[str],
    threshold: int,
) -> FieldResult:
    if extracted is None:
        return FieldResult(
            field=field,
            status=FieldStatus.NOT_DETECTED,
            extracted_value=None,
            submitted_value=submitted,
        )
    if submitted is None:
        return FieldResult(
            field=field,
            status=FieldStatus.FAIL,
            extracted_value=extracted,
            submitted_value=None,
        )
    score = fuzz.ratio(extracted.lower().strip(), submitted.lower().strip())
    if score >= threshold:
        status = FieldStatus.PASS
    elif score >= 70:
        status = FieldStatus.WARN
    else:
        status = FieldStatus.FAIL
    return FieldResult(
        field=field,
        status=status,
        extracted_value=extracted,
        submitted_value=submitted,
        score=float(score),
    )
