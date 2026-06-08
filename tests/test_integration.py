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
import re
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


def _extract(front: Path, back: Optional[Path] = None, timeout: int = 60, verbose: bool = False) -> httpx.Response:
    params = {"verbose": "true"} if verbose else {}
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(open(front, "rb"))
        files = [("image", (front.name, fh))]
        if back:
            bh = stack.enter_context(open(back, "rb"))
            files.append(("back_image", (back.name, bh)))
        return httpx.post(f"{API_BASE}/extract", files=files, params=params, timeout=timeout)


_TRACE_FIELDS = [
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "us_importer", "country_of_origin", "contains_sulfites",
]


def _fmt(v) -> str:
    s = repr(v)
    return (s[:78] + "…'") if len(s) > 80 else s


def _print_pipeline_trace(folder_name: str, data: dict) -> None:
    debug = data.get("_debug", {})

    try:
        main_raw = json.loads(debug.get("main_raw_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        main_raw = {}

    after_parse = debug.get("after_parse", {})
    after_postprocess = debug.get("after_postprocess", {})
    secondary = debug.get("secondary", {})

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  {folder_name}  ·  PIPELINE TRACE")
    print(sep)

    print("\n── [1] Main model output ──")
    for f in _TRACE_FIELDS:
        print(f"  {f:<28} {_fmt(main_raw.get(f))}")

    changed = [
        f for f in _TRACE_FIELDS
        if after_postprocess.get(f) != main_raw.get(f)
    ]
    if changed:
        print("\n── [2] After postprocess (changed fields) ──")
        for f in changed:
            print(f"  {f:<28} {_fmt(after_postprocess.get(f))}  ←  was {_fmt(main_raw.get(f))}")
    else:
        print("\n── [2] After postprocess — no changes ──")

    if secondary:
        print("\n── [3] Secondary passes ──")
        for name, info in secondary.items():
            print(f"  {name:<28} raw={_fmt(info.get('raw'))}  →  accepted={_fmt(info.get('accepted'))}")

    print("\n── [4] Final extracted fields ──")
    for f in _TRACE_FIELDS:
        if f != "us_importer":
            print(f"  {f:<28} {_fmt(data.get(f))}")
    print()


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


def _is_blank(v) -> bool:
    return v is None or v == ""


def _field_matches(extracted, expected, threshold: int) -> bool:
    """Strict OCR-reader match against truth.

    Both blank   -> True  (truth correctly says field is absent and reader agrees)
    One blank    -> False (truth says absent but reader extracted something, or vice versa)
    Both present -> True if fuzzy match >= threshold (or whitespace-only diff)
    """
    if _is_blank(expected) and _is_blank(extracted):
        return True
    if _is_blank(expected) or _is_blank(extracted):
        return False
    norm_exp = _normalize_text(str(expected))
    norm_ext = _normalize_text(str(extracted))
    if re.sub(r"\s+", "", norm_exp) == re.sub(r"\s+", "", norm_ext):
        return True
    return fuzz.partial_ratio(norm_exp, norm_ext) >= threshold


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
# Phase A — OCR reader test: extracted fields must match truth exactly.
# Null in truth means "field is not on this label" and the reader is expected
# to also return null — false positives (reader hallucinated a value) fail too.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder", _LABEL_FOLDERS, ids=[f.name for f in _LABEL_FOLDERS])
def test_extract_fields(api, folder, show_ocr):
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip(f"No image found in {folder.name}")

    truth = _load_truth(folder)
    if not truth:
        pytest.skip(f"No truth JSON in {folder.name}")

    resp = _extract(front, back, verbose=show_ocr)
    assert resp.status_code == 200, f"Extract failed: {resp.text}"
    data = resp.json()

    if show_ocr:
        _print_pipeline_trace(folder.name, data)

    failures = []
    for field, threshold in _FIELD_THRESHOLDS.items():
        expected = truth.get(field)
        extracted = data.get(field)
        if not _field_matches(extracted, expected, threshold):
            failures.append(
                f"{field}: expected ~'{expected}' got '{extracted}'"
            )

    if failures:
        pytest.fail(f"{folder.name} extraction failures: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# Phase B — Compliance check: the extracted fields must satisfy TTB rules
# regardless of what the application form looks like. A label that does not
# carry the required information is non-compliant by definition.
#
# Required on every alcohol label:
#   - brand_name
#   - class_type
#   - alcohol_content (ABV)
#   - net_contents
#   - producer_name_address (US producer OR US importer for imported)
#   - government_warning == "GOVERNMENT WARNING"
#
# Truth files can set 'expected_overall: false' to flag intentionally
# non-compliant labels (this test then expects compliance to FAIL).
# ---------------------------------------------------------------------------

_TTB_REQUIRED_FIELDS = (
    "brand_name", "class_type", "alcohol_content",
    "net_contents", "producer_name_address", "government_warning",
)


def _compliance_violations(extracted: dict) -> list[str]:
    """Return human-readable list of missing/invalid required fields, [] if compliant."""
    violations = []
    for field in _TTB_REQUIRED_FIELDS:
        if _is_blank(extracted.get(field)):
            violations.append(f"{field} missing")
    gw = extracted.get("government_warning")
    if gw is not None and gw != "GOVERNMENT WARNING":
        violations.append(f"government_warning malformed: {gw!r}")
    return violations


@pytest.mark.parametrize("folder", _LABEL_FOLDERS, ids=[f.name for f in _LABEL_FOLDERS])
def test_label_compliance(api, folder):
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip(f"No image found in {folder.name}")
    truth = _load_truth(folder)
    if not truth:
        pytest.skip(f"No truth JSON in {folder.name}")

    expected_compliant = truth.get("expected_overall", True)

    resp = _extract(front, back)
    assert resp.status_code == 200, f"Extract failed: {resp.text}"
    data = resp.json()
    violations = _compliance_violations(data)
    is_compliant = not violations

    if expected_compliant and not is_compliant:
        pytest.fail(f"{folder.name} — expected compliant but found violations: " + "; ".join(violations))
    if not expected_compliant and is_compliant:
        pytest.fail(f"{folder.name} — expected non-compliant (TTB violation) but extracted data passed all required-field checks")


# ---------------------------------------------------------------------------
# Verify endpoint — end-to-end check that submitting truth data as form input
# returns overall_pass=True. Compliance failures (expected_overall: false)
# are owned by test_label_compliance, so we skip them here.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder", _LABEL_FOLDERS, ids=[f.name for f in _LABEL_FOLDERS])
def test_verify_passes_with_truth_data(api, folder):
    front, back = _get_label_images(folder)
    if not front:
        pytest.skip(f"No image found in {folder.name}")

    truth = _load_truth(folder)
    if not truth:
        pytest.skip(f"No truth JSON in {folder.name}")
    if not truth.get("expected_overall", True):
        pytest.skip(f"{folder.name} is a non-compliant label — covered by test_label_compliance")

    resp = _verify(front, back, form_data=_truth_as_form(truth))
    assert resp.status_code == 200, f"Verify failed: {resp.text}"
    result = resp.json()

    if not result["overall_pass"]:
        failing_details = [
            f"{f['field']}: got '{f.get('extracted_value')}' expected '{f.get('submitted_value')}'"
            for f in result["fields"] if f["status"] == "fail"
        ]
        pytest.fail(f"{folder.name} — expected overall pass but got failures: " + "; ".join(failing_details))


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
