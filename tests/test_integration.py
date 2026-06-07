"""
Integration tests — require the API and Ollama containers to be running.

Run all tests:       pytest tests/test_integration.py -v
Run unit tests only: pytest tests/ -v -m "not integration"

Test label folders live in testLabels/. Each folder contains:
  - One image (single label) OR two images named *Front* and *Back*
  - A JSON truth file with expected field values (null = field not present/required)
"""
import contextlib
import json
import unicodedata
import pytest
import httpx
from pathlib import Path
from thefuzz import fuzz
from typing import Optional

pytestmark = pytest.mark.integration

API_BASE = "http://localhost:8000"
LABELS_DIR = Path(__file__).parent.parent / "testLabels"

# Per-field thresholds for extraction quality tests
_FIELD_THRESHOLDS = {
    "brand_name": 80,
    "class_type": 85,
    "alcohol_content": 80,
    "net_contents": 80,
    "producer_name_address": 70,
    "country_of_origin": 90,
    "contains_sulfites": 70,
    "government_warning": 70,
}


@pytest.fixture(scope="module")
def api():
    """Verify API is reachable before running any integration test."""
    try:
        r = httpx.get(f"{API_BASE}/", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.skip("API not reachable — start Docker containers before running integration tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_label_images(folder: Path):
    """Return (front_path, back_path_or_None) for a label folder.

    Detects front/back by looking for 'front'/'back' in filename (case-insensitive).
    Falls back to the single image in the folder if no front/back naming is found.
    """
    images = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    front = next((f for f in images if "front" in f.name.lower()), None)
    back = next((f for f in images if "back" in f.name.lower()), None)
    if front:
        return front, back
    return (images[0] if images else None), None


def _load_truth(folder: Path) -> dict:
    json_files = list(folder.glob("*.json"))
    if not json_files:
        return {}
    with open(json_files[0]) as f:
        return json.load(f)


def _extract(front: Path, back: Optional[Path] = None, timeout: int = 60) -> httpx.Response:
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(open(front, "rb"))
        files = [("image", (front.name, fh))]
        if back:
            bh = stack.enter_context(open(back, "rb"))
            files.append(("back_image", (back.name, bh)))
        return httpx.post(f"{API_BASE}/extract", files=files, timeout=timeout)


def _verify(
    front: Path,
    back: Optional[Path] = None,
    form_data: Optional[dict] = None,
    timeout: int = 60,
) -> httpx.Response:
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(open(front, "rb"))
        files = [("image", (front.name, fh))]
        if back:
            bh = stack.enter_context(open(back, "rb"))
            files.append(("back_image", (back.name, bh)))
        return httpx.post(
            f"{API_BASE}/verify", files=files, data=form_data or {}, timeout=timeout
        )


def _normalize_text(s: str) -> str:
    """Lowercase and strip diacritics so accented chars compare equal to their base letters."""
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", errors="ignore").decode("ascii")


def _field_matches(extracted, expected, threshold: int) -> bool:
    if extracted is None:
        return expected is None or expected == ""
    if expected is None or expected == "":
        return True  # optional — not checked for this label
    return fuzz.partial_ratio(_normalize_text(str(expected)), _normalize_text(str(extracted))) >= threshold


_METADATA_KEYS = {"expected_overall"}

def _truth_as_form(truth: dict) -> dict:
    """Return only fields with real values — None and '' mean 'not provided', so omit them."""
    return {
        k: v
        for k, v in truth.items()
        if k not in _METADATA_KEYS and v is not None and v != ""
    }


# ---------------------------------------------------------------------------
# Discover label folders once for parametrize
# ---------------------------------------------------------------------------

_LABEL_FOLDERS = (
    sorted(d for d in LABELS_DIR.iterdir() if d.is_dir())
    if LABELS_DIR.exists()
    else []
)


# ---------------------------------------------------------------------------
# Extract endpoint — verify extraction quality against ground truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder", _LABEL_FOLDERS, ids=[f.name for f in _LABEL_FOLDERS])
def test_extract_fields(api, folder):
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip(f"No image found in {folder.name}")

    truth = _load_truth(folder)
    if not truth:
        pytest.skip(f"No truth JSON in {folder.name}")

    resp = _extract(front, back)
    assert resp.status_code == 200, f"Extract failed: {resp.text}"
    data = resp.json()

    failures = []
    for field, threshold in _FIELD_THRESHOLDS.items():
        expected = truth.get(field)
        if expected is None or expected == "":
            continue  # not required for this label
        extracted = data.get(field)
        if not _field_matches(extracted, expected, threshold):
            failures.append(
                f"  {field}: expected ~'{expected}' (threshold {threshold}%), got '{extracted}'"
            )

    if failures:
        pytest.fail(f"{folder.name} extraction failures:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# Verify endpoint — passing submissions (truth data should produce overall_pass=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder", _LABEL_FOLDERS, ids=[f.name for f in _LABEL_FOLDERS])
def test_verify_passes_with_truth_data(api, folder):
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip(f"No image found in {folder.name}")

    truth = _load_truth(folder)
    if not truth:
        pytest.skip(f"No truth JSON in {folder.name}")

    expected_overall = truth.get("expected_overall", True)

    resp = _verify(front, back, form_data=_truth_as_form(truth))
    assert resp.status_code == 200, f"Verify failed: {resp.text}"
    result = resp.json()

    failing = [f["field"] for f in result["fields"] if f["status"] == "fail"]
    if expected_overall:
        if not result["overall_pass"]:
            pytest.fail(f"{folder.name} — expected overall pass but got failures: {failing}")
    else:
        if result["overall_pass"]:
            pytest.fail(f"{folder.name} — expected overall FAIL (non-compliant label) but got pass")


# ---------------------------------------------------------------------------
# Verify endpoint — deliberate mismatches should fail
# ---------------------------------------------------------------------------

def test_verify_fails_with_wrong_brand(api):
    folder = LABELS_DIR / "TEST1"
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip("TEST1 not found")

    truth = _load_truth(folder)
    form_data = {**_truth_as_form(truth), "brand_name": "COMPLETELY WRONG DISTILLERY XYZ"}

    resp = _verify(front, back, form_data=form_data)
    assert resp.status_code == 200
    result = resp.json()
    brand_result = next(r for r in result["fields"] if r["field"] == "brand_name")
    assert brand_result["status"] == "fail", (
        f"Expected brand_name to fail but got: {brand_result['status']}"
    )


def test_verify_fails_with_wrong_abv(api):
    folder = LABELS_DIR / "TEST1"
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip("TEST1 not found")

    truth = _load_truth(folder)
    form_data = {**_truth_as_form(truth), "alcohol_content": "12% ALC/VOL"}

    resp = _verify(front, back, form_data=form_data)
    assert resp.status_code == 200
    result = resp.json()
    abv_result = next(r for r in result["fields"] if r["field"] == "alcohol_content")
    assert abv_result["status"] == "fail", (
        f"Expected alcohol_content to fail but got: {abv_result['status']}"
    )


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

def test_history_records_verification(api):
    resp = httpx.get(f"{API_BASE}/api/history", timeout=10)
    assert resp.status_code == 200
    records = resp.json()
    assert isinstance(records, list)
    assert len(records) > 0, "Expected at least one history record after running verify tests"
