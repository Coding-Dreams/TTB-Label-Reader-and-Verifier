"""
Integration tests — require the API and Ollama containers to be running.

Run all tests:       pytest tests/test_integration.py -v
Run unit tests only: pytest tests/ -v -m "not integration"
"""
import pytest
import httpx
from pathlib import Path
from thefuzz import fuzz

pytestmark = pytest.mark.integration

API_BASE = "http://localhost:8000"
LABELS_DIR = Path(__file__).parent.parent / "testLabels"

# Ground truth for each test label — used to assert extract quality and verify pass.
# Values are (expected_substring_or_value, fuzzy_threshold).
GROUND_TRUTH = {
    "test1.jpg": {
        "brand_name":           ("ABC", 80),
        "class_type":           ("Straight Rye Whisky", 80),
        "alcohol_content":      ("45%", 80),
        "net_contents":         ("750", 80),
        "producer_name_address":("ABC Distillery Frederick, MD", 70),
        "country_of_origin":    ("", 90),       # not required for domestic spirits
        "contains_sulfites":    ("", 70),        # whisky — no sulfite declaration
        "government_warning":   ("GOVERNMENT WARNING:", 70),
    },
    "test2.png": {
        "brand_name":           ("ABC WINERY", 80),
        "class_type":           ("Red Wine", 70),
        "alcohol_content":      ("13%", 80),
        "net_contents":         ("750", 80),
        "producer_name_address":("XYZ Cellars", 70),
        "country_of_origin":    ("", 90),       # not required for domestic wine
        "contains_sulfites":    ("CONTAINS SULFITES", 70),
        "government_warning":   ("GOVERNMENT WARNING:", 70),
    },
    "test3.jpg": {
        "brand_name":           ("12345 Imports", 80),
        "class_type":           ("Rum", 70),
        "alcohol_content":      ("18%", 80),
        "net_contents":         ("200", 80),
        "producer_name_address":("12345 IMPORTS", 70),
        "country_of_origin":    ("Canada", 90),
        "contains_sulfites":    ("", 70),        # rum — no sulfite declaration
        "government_warning":   ("GOVERNMENT WARNING:", 70),
    },
    "test4.png": {
        "brand_name":           ("MALT & HOP", 70),
        "class_type":           ("Ale", 70),
        "alcohol_content":      ("5%", 80),
        "net_contents":         ("PINT", 70),
        "producer_name_address":("MALT & HOP", 70),
        "country_of_origin":    ("", 90),       # not required for domestic beer
        "contains_sulfites":    ("", 70),        # ale — no sulfite declaration
        "government_warning":   ("GOVERNMENT WARNING:", 70),
    },
}

# Submitted form data for each label — should produce overall_pass=True.
_GOV_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

PASSING_SUBMISSIONS = {
    "test1.jpg": {
        "brand_name": "ABC",
        "class_type": "Straight Rye Whisky",
        "alcohol_content": "45% ALC/VOL",
        "net_contents": "750 ML",
        "producer_name_address": "ABC Distillery, Frederick, MD",
        "country_of_origin": "United States",
        "government_warning": _GOV_WARNING,
        "contains_sulfites": "",
    },
    "test2.png": {
        "brand_name": "ABC WINERY",
        "class_type": "American Red Wine",
        "alcohol_content": "13% BY VOL",
        "net_contents": "750 ML",
        "producer_name_address": "XYZ Cellars, City, State",
        "country_of_origin": "",
        "government_warning": _GOV_WARNING,
        "contains_sulfites": "CONTAINS SULFITES",
    },
    "test3.jpg": {
        "brand_name": "12345 IMPORTS",
        "class_type": "Rum with Coconut Liqueur",
        "alcohol_content": "18% ALC/VOL",
        "net_contents": "200 ML",
        "producer_name_address": "12345 IMPORTS MIAMI, FL",
        "country_of_origin": "Canada",
        "government_warning": _GOV_WARNING,
        "contains_sulfites": "",
    },
    "test4.png": {
        "brand_name": "MALT & HOP BREWERY",
        "class_type": "Ale with Honey and Huckleberry Flavor",
        "alcohol_content": "5% ALC./VOL.",
        "net_contents": "1 PINT, 0.9 FL. OZ.",
        "producer_name_address": "MALT & HOP BREWERY HYATTSVILLE, MD",
        "country_of_origin": "United States",
        "government_warning": _GOV_WARNING,
        "contains_sulfites": "",
    },
}


@pytest.fixture(scope="module")
def api():
    """Verify API is reachable before running any integration test."""
    try:
        r = httpx.get(f"{API_BASE}/", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.skip("API not reachable — start Docker containers before running integration tests")


def _field_matches(extracted: str | None, expected: str, threshold: int) -> bool:
    if extracted is None:
        return False
    return fuzz.partial_ratio(expected.lower(), extracted.lower()) >= threshold


# ---------------------------------------------------------------------------
# Extract endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,truth", GROUND_TRUTH.items())
def test_extract_fields(api, filename, truth):
    image_path = LABELS_DIR / filename
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/extract",
            files={"image": (filename, f)},
            timeout=60,
        )

    assert resp.status_code == 200, f"Extract failed: {resp.text}"
    data = resp.json()

    for field, (expected, threshold) in truth.items():
        if not expected:  # empty string means field is not required / not checked for this label
            continue
        extracted = data.get(field)
        assert _field_matches(extracted, expected, threshold), (
            f"{filename} — {field}: expected ~'{expected}' (threshold {threshold}%), "
            f"got '{extracted}'"
        )


# ---------------------------------------------------------------------------
# Verify endpoint tests — passing submissions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,submission", PASSING_SUBMISSIONS.items())
def test_verify_passes_with_correct_data(api, filename, submission):
    image_path = LABELS_DIR / filename
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/verify",
            files={"image": (filename, f)},
            data=submission,
            timeout=60,
        )

    assert resp.status_code == 200, f"Verify failed: {resp.text}"
    result = resp.json()

    failing = [
        f["field"] for f in result["fields"] if f["status"] == "fail"
    ]
    assert result["overall_pass"], (
        f"{filename} — expected overall pass but got failures: {failing}"
    )


# ---------------------------------------------------------------------------
# Verify endpoint test — deliberate mismatch should fail
# ---------------------------------------------------------------------------

def test_verify_fails_with_wrong_brand(api):
    image_path = LABELS_DIR / "test1.jpg"
    if not image_path.exists():
        pytest.skip("test1.jpg not found")

    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/verify",
            files={"image": ("test1.jpg", f)},
            data={
                **PASSING_SUBMISSIONS["test1.jpg"],
                "brand_name": "COMPLETELY WRONG DISTILLERY XYZ",
            },
            timeout=60,
        )

    assert resp.status_code == 200
    result = resp.json()
    brand_result = next(r for r in result["fields"] if r["field"] == "brand_name")
    assert brand_result["status"] == "fail", (
        f"Expected brand_name to fail but got: {brand_result['status']}"
    )


def test_verify_fails_with_wrong_abv(api):
    image_path = LABELS_DIR / "test1.jpg"
    if not image_path.exists():
        pytest.skip("test1.jpg not found")

    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/verify",
            files={"image": ("test1.jpg", f)},
            data={
                **PASSING_SUBMISSIONS["test1.jpg"],
                "alcohol_content": "12% ALC/VOL",
            },
            timeout=60,
        )

    assert resp.status_code == 200
    result = resp.json()
    abv_result = next(r for r in result["fields"] if r["field"] == "alcohol_content")
    assert abv_result["status"] == "fail", (
        f"Expected alcohol_content to fail but got: {abv_result['status']}"
    )


# ---------------------------------------------------------------------------
# History endpoint test
# ---------------------------------------------------------------------------

def test_history_records_verification(api):
    resp = httpx.get(f"{API_BASE}/api/history", timeout=10)
    assert resp.status_code == 200
    records = resp.json()
    assert isinstance(records, list)
    # After running the verify tests above, there should be records
    assert len(records) > 0, "Expected at least one history record after running verify tests"
