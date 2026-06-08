"""TTB compliance checks on extracted label data.

Independent of any submitted form data — these are the federal-regulatory rules
a label must satisfy on its own. Mirrors the Phase B logic in
tests/test_integration.py so the UI's "verify" surface and the test suite agree
on what counts as a compliant label.

References:
  27 CFR 4   — wine labelling (sulfite declaration in §4.32a)
  27 CFR 5   — distilled-spirits labelling
  27 CFR 7   — malt-beverage labelling
"""
from typing import Optional


def _is_blank(v) -> bool:
    return v is None or v == ""


# Universal — required on every alcohol label
_REQUIRED_FIELDS = (
    ("brand_name", "Brand name"),
    ("class_type", "Class / Type"),
    ("alcohol_content", "Alcohol content (ABV)"),
    ("net_contents", "Net contents"),
    ("producer_name_address", "Producer name & address"),
    ("government_warning", "Government warning"),
)


def check_compliance(extracted: dict) -> dict:
    """Return a compliance report for the extracted fields.

    Shape:
        {
          "compliant": bool,
          "violations": [
            {"field": str, "label": str, "message": str},
            ...
          ]
        }
    """
    violations = []

    for field, label in _REQUIRED_FIELDS:
        if _is_blank(extracted.get(field)):
            violations.append({
                "field": field,
                "label": label,
                "message": f"{label} is required on every alcohol label",
            })

    gw = extracted.get("government_warning")
    if gw is not None and gw != "GOVERNMENT WARNING":
        violations.append({
            "field": "government_warning",
            "label": "Government warning",
            "message": (
                "Government warning must appear in uppercase exactly as 'GOVERNMENT WARNING'"
            ),
        })

    # Wine-specific: §4.32a requires a sulfite declaration (positive or negative)
    class_type = (extracted.get("class_type") or "").strip().lower()
    if class_type == "wine" and _is_blank(extracted.get("contains_sulfites")):
        violations.append({
            "field": "contains_sulfites",
            "label": "Sulfite declaration",
            "message": (
                "Wines must carry a sulfite declaration (e.g. 'CONTAINS SULFITES' or "
                "'CONTAINS NO DETECTABLE SULFITES') per 27 CFR 4.32a"
            ),
        })

    return {"compliant": not violations, "violations": violations}
