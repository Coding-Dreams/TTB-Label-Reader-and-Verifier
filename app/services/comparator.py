import re
from typing import Optional
from thefuzz import fuzz
from app.models.label import LabelFields
from app.models.result import FieldResult, FieldStatus, VerificationResult

_REQUIRED_WARNING = "GOVERNMENT WARNING"

# Matches negation words that turn a sulfite mention into a "no sulfites" statement
_SULFITE_ABSENT_RE = re.compile(
    r'\b(free|no\b|none|without|not\s+detected|undetectable)\b', re.IGNORECASE
)


def _sulfite_absent(text: str) -> bool:
    return bool(_SULFITE_ABSENT_RE.search(text))


def check_government_warning(
    extracted: Optional[str], submitted: Optional[str]
) -> FieldResult:
    if not submitted:  # not provided — skip verification
        return FieldResult(
            field="government_warning",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # warning not found on label — cannot verify claim
        return FieldResult(
            field="government_warning",
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    passes = extracted == _REQUIRED_WARNING
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
    partial: bool = False,
) -> FieldResult:
    if not submitted:  # None or empty string — field not provided, skip verification
        return FieldResult(
            field=field,
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # claimed value not found on label — cannot verify
        return FieldResult(
            field=field,
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    fn = fuzz.partial_ratio if partial else fuzz.ratio
    score = fn(extracted.lower().strip(), submitted.lower().strip())
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


def _parse_abv(value: str) -> Optional[float]:
    match = re.search(r"(\d+\.?\d*)\s*%", value)
    return float(match.group(1)) if match else None


def check_abv(extracted: Optional[str], submitted: Optional[str]) -> FieldResult:
    if not submitted:  # not provided — skip verification
        return FieldResult(
            field="alcohol_content",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # not found on label — cannot verify claim
        return FieldResult(
            field="alcohol_content",
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    ext_val = _parse_abv(extracted)
    sub_val = _parse_abv(submitted)
    if ext_val is None or sub_val is None:
        return check_fuzzy_field("alcohol_content", extracted, submitted, threshold=90)
    passes = abs(ext_val - sub_val) <= 0.1
    return FieldResult(
        field="alcohol_content",
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def _parse_volume_ml(value: str) -> Optional[float]:
    v = value.lower().strip()
    m = re.search(r"(\d+\.?\d*)\s*ml", v)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*l(?:iter|itre)?s?\b", v)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r"(\d+\.?\d*)\s*(?:fl\.?\s*oz|fluid\s*oz)", v)
    if m:
        return float(m.group(1)) * 29.5735
    return None


def check_net_contents(extracted: Optional[str], submitted: Optional[str]) -> FieldResult:
    if not submitted:  # not provided — skip verification
        return FieldResult(
            field="net_contents",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # not found on label — cannot verify claim
        return FieldResult(
            field="net_contents",
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    ext_ml = _parse_volume_ml(extracted)
    sub_ml = _parse_volume_ml(submitted)
    if ext_ml is None or sub_ml is None:
        return check_fuzzy_field("net_contents", extracted, submitted, threshold=90)
    passes = abs(ext_ml - sub_ml) <= 1.0
    return FieldResult(
        field="net_contents",
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def check_exact_field(
    field: str, extracted: Optional[str], submitted: Optional[str]
) -> FieldResult:
    if not submitted:  # None or empty string — field not provided, skip verification
        return FieldResult(
            field=field,
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # not found on label — cannot verify claim
        return FieldResult(
            field=field,
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    passes = extracted.strip().lower() == submitted.strip().lower()
    return FieldResult(
        field=field,
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def check_sulfites(extracted: Optional[str], submitted: Optional[str]) -> FieldResult:
    if not submitted:  # None or empty string — not provided, skip verification
        return FieldResult(
            field="contains_sulfites",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=extracted,
            submitted_value=None,
        )
    if extracted is None:  # no sulfite declaration found on label — cannot verify claim
        return FieldResult(
            field="contains_sulfites",
            status=FieldStatus.FAIL,
            extracted_value=None,
            submitted_value=submitted,
        )
    if "sulfite" not in submitted.lower():
        return FieldResult(
            field="contains_sulfites",
            status=FieldStatus.FAIL,
            extracted_value=extracted,
            submitted_value=submitted,
        )
    # Both sides mention sulfites — verify they agree on presence vs absence
    passes = _sulfite_absent(extracted) == _sulfite_absent(submitted)
    return FieldResult(
        field="contains_sulfites",
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def compare_label(extracted: LabelFields, form_data: LabelFields) -> VerificationResult:
    results = [
        check_fuzzy_field("brand_name", extracted.brand_name, form_data.brand_name, threshold=90, partial=True),
        check_fuzzy_field("class_type", extracted.class_type, form_data.class_type, threshold=85, partial=True),
        check_abv(extracted.alcohol_content, form_data.alcohol_content),
        check_net_contents(extracted.net_contents, form_data.net_contents),
        check_fuzzy_field("producer_name_address", extracted.producer_name_address, form_data.producer_name_address, threshold=70, partial=True),
        check_exact_field("country_of_origin", extracted.country_of_origin, form_data.country_of_origin),
        check_government_warning(extracted.government_warning, form_data.government_warning),
        check_sulfites(extracted.contains_sulfites, form_data.contains_sulfites),
    ]
    overall_pass = all(r.status != FieldStatus.FAIL for r in results)
    return VerificationResult(overall_pass=overall_pass, fields=results)
