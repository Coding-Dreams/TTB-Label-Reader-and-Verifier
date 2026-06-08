# TTB Label Verification App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local, Dockerized AI-powered alcohol label verification prototype that extracts fields from label images using llava-phi3 via Ollama and compares them against applicant-submitted form data using deterministic field-appropriate matching.

**Architecture:** FastAPI backend serves plain HTML + Tailwind frontend and a REST API. Ollama (llava-phi3) handles image-to-JSON field extraction in a single VLM call. Python comparator logic runs deterministic checks per field (exact, fuzzy, numeric). SQLite persists verification history.

**Tech Stack:** Python 3.11+, FastAPI, Ollama (llava-phi3), thefuzz, httpx, SQLite (sqlite3), Tailwind CSS CDN, Docker Compose, pytest

---

## File Map

```
/
├── docker-compose.yml             # Defines ollama + api services
├── Dockerfile                     # Builds the FastAPI app container
├── requirements.txt               # Python dependencies
├── .env.example                   # OLLAMA_BASE_URL=http://ollama:11434
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, startup, routers, static mount
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── verify.py              # POST /extract, POST /verify
│   │   └── batch.py               # POST /batch, GET /batch/{job_id}/status, GET /batch/{job_id}/download
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ollama.py              # extract_label_fields(image_path) -> LabelFields
│   │   ├── comparator.py          # compare_label(extracted, form_data) -> VerificationResult
│   │   └── db.py                  # init_db, save_verification, get_verifications, get_verification
│   ├── models/
│   │   ├── __init__.py
│   │   ├── label.py               # LabelFields (Pydantic)
│   │   └── result.py              # FieldStatus, FieldResult, VerificationResult (Pydantic)
│   └── static/
│       ├── index.html             # Single label verification UI
│       ├── batch.html             # Batch upload UI
│       └── history.html           # Verification history UI
├── tests/
│   ├── conftest.py                # tmp_db fixture
│   └── test_comparator.py        # Unit tests for all comparator rules
└── data/                          # Created at runtime (uploads + verifications.db)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`, `app/routers/__init__.py`, `app/services/__init__.py`, `app/models/__init__.py`
- Create: `app/main.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p app/routers app/services app/models app/static tests data
touch app/__init__.py app/routers/__init__.py app/services/__init__.py app/models/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
httpx==0.27.0
thefuzz==0.22.1
python-Levenshtein==0.25.1
pydantic==2.8.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

- [ ] **Step 3: Write .env.example**

```
OLLAMA_BASE_URL=http://ollama:11434
```

- [ ] **Step 4: Write app/main.py**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import verify, batch
from app.services.db import init_db

app = FastAPI(title="TTB Label Verification")

@app.on_event("startup")
async def startup():
    init_db()

app.include_router(verify.router)
app.include_router(batch.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

@app.get("/batch-upload")
def batch_page():
    return FileResponse("app/static/batch.html")

@app.get("/history")
def history_page():
    return FileResponse("app/static/history.html")

@app.get("/api/history")
def get_history():
    from app.services.db import get_verifications
    return get_verifications()

@app.get("/api/history/{verification_id}")
def get_single(verification_id: int):
    from app.services.db import get_verification
    from fastapi import HTTPException
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return record
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Verify app starts**

```bash
uvicorn app.main:app --reload
```

Expected: Server starts on `http://127.0.0.1:8000` with no import errors. (Static files will 404 until we add HTML — that's fine.)

- [ ] **Step 7: Commit**

```bash
git init
git add requirements.txt .env.example app/main.py app/__init__.py app/routers/__init__.py app/services/__init__.py app/models/__init__.py
git commit -m "feat: project scaffolding and FastAPI entry point"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `app/models/label.py`
- Create: `app/models/result.py`

- [ ] **Step 1: Write app/models/label.py**

```python
from pydantic import BaseModel
from typing import Optional


class LabelFields(BaseModel):
    brand_name: Optional[str] = None
    class_type: Optional[str] = None
    alcohol_content: Optional[str] = None
    net_contents: Optional[str] = None
    producer_name_address: Optional[str] = None
    country_of_origin: Optional[str] = None
    government_warning: Optional[str] = None
```

- [ ] **Step 2: Write app/models/result.py**

```python
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
```

- [ ] **Step 3: Verify models import cleanly**

```bash
python -c "from app.models.label import LabelFields; from app.models.result import VerificationResult, FieldResult, FieldStatus; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/models/label.py app/models/result.py
git commit -m "feat: add LabelFields and VerificationResult Pydantic models"
```

---

## Task 3: SQLite Database Layer

**Files:**
- Create: `app/services/db.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write tests/conftest.py**

```python
import pytest
from pathlib import Path
import app.services.db as db_module


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()
    return db_file
```

- [ ] **Step 2: Write failing tests for db.py**

Create `tests/test_db.py`:

```python
import json
import pytest
from app.services.db import save_verification, get_verifications, get_verification


def test_save_and_retrieve_verification(tmp_db):
    form_data = {"brand_name": "OLD TOM", "government_warning": "GOVERNMENT WARNING: ..."}
    extracted = {"brand_name": "OLD TOM DISTILLERY", "government_warning": "GOVERNMENT WARNING: ..."}
    results = {"overall_pass": True, "fields": []}

    record_id = save_verification(
        image_filename="label.jpg",
        form_data=form_data,
        extracted=extracted,
        results=results,
        overall_pass=True,
    )

    assert record_id == 1
    record = get_verification(record_id)
    assert record["image_filename"] == "label.jpg"
    assert record["overall_pass"] == 1
    assert json.loads(record["form_data"])["brand_name"] == "OLD TOM"


def test_get_verifications_returns_list(tmp_db):
    save_verification("a.jpg", {}, {}, {}, True)
    save_verification("b.jpg", {}, {}, {}, False)

    records = get_verifications()
    assert len(records) == 2
    assert records[0]["image_filename"] == "b.jpg"  # Most recent first


def test_get_verification_returns_none_for_missing(tmp_db):
    assert get_verification(999) is None


def test_batch_id_stored(tmp_db):
    save_verification("label.jpg", {}, {}, {}, True, batch_id="batch-abc")
    records = get_verifications()
    assert records[0]["batch_id"] == "batch-abc"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.db'`

- [ ] **Step 4: Write app/services/db.py**

```python
import sqlite3
import json
from pathlib import Path
from typing import Optional, List

DB_PATH = Path("data/verifications.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_filename TEXT NOT NULL,
                form_data      TEXT NOT NULL,
                extracted      TEXT NOT NULL,
                results        TEXT NOT NULL,
                overall_pass   BOOLEAN NOT NULL,
                batch_id       TEXT
            )
        """)
        conn.commit()


def save_verification(
    image_filename: str,
    form_data: dict,
    extracted: dict,
    results: dict,
    overall_pass: bool,
    batch_id: Optional[str] = None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """INSERT INTO verifications
               (image_filename, form_data, extracted, results, overall_pass, batch_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                image_filename,
                json.dumps(form_data),
                json.dumps(extracted),
                json.dumps(results),
                overall_pass,
                batch_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_verifications(limit: int = 100) -> List[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM verifications ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_verification(verification_id: int) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM verifications WHERE id = ?", (verification_id,)
        ).fetchone()
        return dict(row) if row else None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/db.py tests/conftest.py tests/test_db.py
git commit -m "feat: SQLite db layer with save/retrieve verifications"
```

---

## Task 4: Comparator — Government Warning Check

**Files:**
- Create: `app/services/comparator.py`
- Create: `tests/test_comparator.py`

- [ ] **Step 1: Write failing tests for government warning check**

Create `tests/test_comparator.py`:

```python
import pytest
from app.models.result import FieldStatus
from app.services.comparator import check_government_warning


def test_government_warning_passes_with_all_caps():
    result = check_government_warning(
        extracted="GOVERNMENT WARNING: According to the Surgeon General...",
        submitted="GOVERNMENT WARNING: According to the Surgeon General...",
    )
    assert result.status == FieldStatus.PASS


def test_government_warning_fails_with_title_case():
    result = check_government_warning(
        extracted="Government Warning: According to the Surgeon General...",
        submitted="GOVERNMENT WARNING: According to the Surgeon General...",
    )
    assert result.status == FieldStatus.FAIL


def test_government_warning_fails_when_missing_from_label():
    result = check_government_warning(extracted=None, submitted="GOVERNMENT WARNING: ...")
    assert result.status == FieldStatus.NOT_DETECTED


def test_government_warning_field_name_is_correct():
    result = check_government_warning(extracted="GOVERNMENT WARNING: ...", submitted=None)
    assert result.field == "government_warning"


def test_government_warning_fails_when_substring_absent():
    result = check_government_warning(
        extracted="Warning: Drinking is bad for you.",
        submitted="GOVERNMENT WARNING: ...",
    )
    assert result.status == FieldStatus.FAIL
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_comparator.py -v
```

Expected: FAIL — `ImportError: cannot import name 'check_government_warning'`

- [ ] **Step 3: Write check_government_warning in app/services/comparator.py**

```python
from typing import Optional
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_comparator.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/comparator.py tests/test_comparator.py
git commit -m "feat: government warning exact-match comparator with tests"
```

---

## Task 5: Comparator — Fuzzy Field Check

**Files:**
- Modify: `app/services/comparator.py`
- Modify: `tests/test_comparator.py`

- [ ] **Step 1: Add failing tests for fuzzy field check**

Append to `tests/test_comparator.py`:

```python
from app.services.comparator import check_fuzzy_field


def test_fuzzy_brand_name_passes_case_variation():
    result = check_fuzzy_field("brand_name", "STONE'S THROW", "Stone's Throw", threshold=90)
    assert result.status == FieldStatus.PASS


def test_fuzzy_brand_name_passes_exact():
    result = check_fuzzy_field("brand_name", "OLD TOM DISTILLERY", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.PASS


def test_fuzzy_brand_name_warns_on_near_miss():
    result = check_fuzzy_field("brand_name", "OLD TOM DISTILERY", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.WARN
    assert result.score is not None
    assert 70 <= result.score < 90


def test_fuzzy_brand_name_fails_on_mismatch():
    result = check_fuzzy_field("brand_name", "BLUE RIDGE", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.FAIL


def test_fuzzy_returns_not_detected_when_extracted_is_none():
    result = check_fuzzy_field("brand_name", None, "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.NOT_DETECTED
    assert result.extracted_value is None


def test_fuzzy_field_name_preserved():
    result = check_fuzzy_field("class_type", "Kentucky Bourbon", "Kentucky Bourbon", threshold=85)
    assert result.field == "class_type"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_comparator.py::test_fuzzy_brand_name_passes_case_variation -v
```

Expected: FAIL — `ImportError: cannot import name 'check_fuzzy_field'`

- [ ] **Step 3: Add check_fuzzy_field to app/services/comparator.py**

Add after `check_government_warning`:

```python
from thefuzz import fuzz


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_comparator.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/comparator.py tests/test_comparator.py
git commit -m "feat: fuzzy field comparator with warn/fail thresholds"
```

---

## Task 6: Comparator — ABV Normalization

**Files:**
- Modify: `app/services/comparator.py`
- Modify: `tests/test_comparator.py`

- [ ] **Step 1: Add failing tests for ABV check**

Append to `tests/test_comparator.py`:

```python
from app.services.comparator import check_abv


def test_abv_passes_with_full_format():
    result = check_abv("45% Alc./Vol. (90 Proof)", "45%")
    assert result.status == FieldStatus.PASS


def test_abv_passes_with_decimal():
    result = check_abv("40.0% Alc./Vol.", "40%")
    assert result.status == FieldStatus.PASS


def test_abv_fails_on_mismatch():
    result = check_abv("40%", "45%")
    assert result.status == FieldStatus.FAIL


def test_abv_passes_within_tolerance():
    result = check_abv("45.05%", "45%")
    assert result.status == FieldStatus.PASS


def test_abv_not_detected_when_none():
    result = check_abv(None, "45%")
    assert result.status == FieldStatus.NOT_DETECTED


def test_abv_field_name_is_correct():
    result = check_abv("45%", "45%")
    assert result.field == "alcohol_content"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_comparator.py::test_abv_passes_with_full_format -v
```

Expected: FAIL — `ImportError: cannot import name 'check_abv'`

- [ ] **Step 3: Add check_abv to app/services/comparator.py**

Add after `check_fuzzy_field` (also add `import re` at the top of the file):

```python
import re


def _parse_abv(value: str) -> Optional[float]:
    match = re.search(r"(\d+\.?\d*)\s*%", value)
    return float(match.group(1)) if match else None


def check_abv(extracted: Optional[str], submitted: Optional[str]) -> FieldResult:
    if extracted is None:
        return FieldResult(
            field="alcohol_content",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=None,
            submitted_value=submitted,
        )
    ext_val = _parse_abv(extracted)
    sub_val = _parse_abv(submitted) if submitted else None
    if ext_val is None or sub_val is None:
        return check_fuzzy_field("alcohol_content", extracted, submitted, threshold=90)
    passes = abs(ext_val - sub_val) <= 0.1
    return FieldResult(
        field="alcohol_content",
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_comparator.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/comparator.py tests/test_comparator.py
git commit -m "feat: ABV numeric normalization comparator"
```

---

## Task 7: Comparator — Net Contents + Exact Field + Full compare_label

**Files:**
- Modify: `app/services/comparator.py`
- Modify: `tests/test_comparator.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_comparator.py`:

```python
from app.services.comparator import check_net_contents, check_exact_field, compare_label
from app.models.label import LabelFields


def test_net_contents_passes_case_insensitive_ml():
    result = check_net_contents("750 mL", "750ml")
    assert result.status == FieldStatus.PASS


def test_net_contents_passes_litre_to_ml():
    result = check_net_contents("0.75 L", "750 mL")
    assert result.status == FieldStatus.PASS


def test_net_contents_fails_different_volume():
    result = check_net_contents("750 mL", "1000 mL")
    assert result.status == FieldStatus.FAIL


def test_net_contents_not_detected_when_none():
    result = check_net_contents(None, "750 mL")
    assert result.status == FieldStatus.NOT_DETECTED


def test_exact_field_passes_case_insensitive():
    result = check_exact_field("country_of_origin", "united states", "United States")
    assert result.status == FieldStatus.PASS


def test_exact_field_fails_mismatch():
    result = check_exact_field("country_of_origin", "France", "United States")
    assert result.status == FieldStatus.FAIL


def test_exact_field_not_detected_when_none():
    result = check_exact_field("country_of_origin", None, "United States")
    assert result.status == FieldStatus.NOT_DETECTED


def test_compare_label_all_pass():
    warning = "GOVERNMENT WARNING: According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects."
    extracted = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
        producer_name_address="Old Tom Distillery, Louisville, KY",
        country_of_origin="United States",
        government_warning=warning,
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45%",
        net_contents="750ml",
        producer_name_address="Old Tom Distillery, Louisville, KY",
        country_of_origin="United States",
        government_warning=warning,
    )
    result = compare_label(extracted, form_data)
    assert result.overall_pass is True
    assert all(f.status == FieldStatus.PASS for f in result.fields)


def test_compare_label_fails_on_bad_warning():
    extracted = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning="Government Warning: (lowercase)",
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning="GOVERNMENT WARNING: ...",
    )
    result = compare_label(extracted, form_data)
    assert result.overall_pass is False
    warning_result = next(f for f in result.fields if f.field == "government_warning")
    assert warning_result.status == FieldStatus.FAIL


def test_compare_label_warn_does_not_fail_overall():
    warning = "GOVERNMENT WARNING: ..."
    extracted = LabelFields(
        brand_name="OLD TOM DISTILERY",  # typo — should be WARN not FAIL
        government_warning=warning,
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning=warning,
    )
    result = compare_label(extracted, form_data)
    brand_result = next(f for f in result.fields if f.field == "brand_name")
    assert brand_result.status == FieldStatus.WARN
    assert result.overall_pass is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_comparator.py::test_net_contents_passes_case_insensitive_ml -v
```

Expected: FAIL — `ImportError: cannot import name 'check_net_contents'`

- [ ] **Step 3: Add check_net_contents, check_exact_field, and compare_label to app/services/comparator.py**

Add after `check_abv`:

```python
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
    if extracted is None:
        return FieldResult(
            field="net_contents",
            status=FieldStatus.NOT_DETECTED,
            extracted_value=None,
            submitted_value=submitted,
        )
    ext_ml = _parse_volume_ml(extracted)
    sub_ml = _parse_volume_ml(submitted) if submitted else None
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
    passes = extracted.strip().lower() == submitted.strip().lower()
    return FieldResult(
        field=field,
        status=FieldStatus.PASS if passes else FieldStatus.FAIL,
        extracted_value=extracted,
        submitted_value=submitted,
    )


def compare_label(extracted: LabelFields, form_data: LabelFields) -> VerificationResult:
    results = [
        check_fuzzy_field("brand_name", extracted.brand_name, form_data.brand_name, threshold=90),
        check_fuzzy_field("class_type", extracted.class_type, form_data.class_type, threshold=85),
        check_abv(extracted.alcohol_content, form_data.alcohol_content),
        check_net_contents(extracted.net_contents, form_data.net_contents),
        check_fuzzy_field("producer_name_address", extracted.producer_name_address, form_data.producer_name_address, threshold=80),
        check_exact_field("country_of_origin", extracted.country_of_origin, form_data.country_of_origin),
        check_government_warning(extracted.government_warning, form_data.government_warning),
    ]
    overall_pass = all(r.status != FieldStatus.FAIL for r in results)
    return VerificationResult(overall_pass=overall_pass, fields=results)
```

- [ ] **Step 4: Run all comparator tests**

```bash
pytest tests/test_comparator.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/comparator.py tests/test_comparator.py
git commit -m "feat: net contents, exact field comparators and full compare_label function"
```

---

## Task 8: Ollama Service

**Files:**
- Create: `app/services/ollama.py`

- [ ] **Step 1: Write app/services/ollama.py**

```python
import os
import re
import base64
import json
import httpx
from pathlib import Path
from typing import Optional

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llava-phi3"
TIMEOUT = 30.0

_EXTRACTION_PROMPT = """You are an OCR assistant specialized in reading alcohol beverage labels.
Extract the following fields from this label image and return ONLY valid JSON with no additional text.

Required JSON format:
{
  "brand_name": "string or null",
  "class_type": "string or null",
  "alcohol_content": "string or null",
  "net_contents": "string or null",
  "producer_name_address": "string or null",
  "country_of_origin": "string or null",
  "government_warning": "string or null"
}

Rules:
- Return the exact text as it appears on the label
- Set any field to null if not visible or not present on the label
- For government_warning, include the complete warning text
- Return ONLY the JSON object, no explanation or markdown"""

_STRICT_PROMPT = _EXTRACTION_PROMPT + "\n\nCRITICAL: Your response must begin with { and end with }. No other characters outside the JSON."


async def extract_label_fields(image_path: Path) -> LabelFields:
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        raw = await _call_ollama(client, image_data, _EXTRACTION_PROMPT)
        try:
            return LabelFields(**_parse_json(raw))
        except (json.JSONDecodeError, ValueError):
            raw = await _call_ollama(client, image_data, _STRICT_PROMPT)
            return LabelFields(**_parse_json(raw))


async def _call_ollama(client: httpx.AsyncClient, image_data: str, prompt: str) -> str:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "format": "json",
        },
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # Extract JSON object if wrapped in markdown fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from app.services.ollama import extract_label_fields; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/services/ollama.py
git commit -m "feat: Ollama VLM extraction service for label fields"
```

---

## Task 9: Verify Router

**Files:**
- Create: `app/routers/verify.py`

- [ ] **Step 1: Write app/routers/verify.py**

```python
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
```

- [ ] **Step 2: Verify server starts with the router**

```bash
uvicorn app.main:app --reload
```

Expected: Starts cleanly. `GET http://localhost:8000/docs` shows `/extract` and `/verify` endpoints.

- [ ] **Step 3: Commit**

```bash
git add app/routers/verify.py
git commit -m "feat: /extract and /verify endpoints"
```

---

## Task 10: Batch Router

**Files:**
- Create: `app/routers/batch.py`

- [ ] **Step 1: Write app/routers/batch.py**

```python
import csv
import io
import shutil
import uuid
from pathlib import Path
from typing import Dict, List

import httpx
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.label import LabelFields
from app.services.comparator import compare_label
from app.services.db import save_verification
from app.services.ollama import extract_label_fields

router = APIRouter()

_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_jobs: Dict[str, dict] = {}

_CSV_FIELDS = [
    "image_filename", "brand_name", "class_type", "alcohol_content",
    "net_contents", "producer_name_address", "country_of_origin", "government_warning",
]


@router.post("/batch")
async def start_batch(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    csv_file: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())

    content = await csv_file.read()
    rows = list(csv.DictReader(io.StringIO(content.decode())))
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or malformed")

    saved: Dict[str, Path] = {}
    for img in images:
        dest = _UPLOAD_DIR / f"batch_{job_id}_{img.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(img.file, f)
        saved[img.filename] = dest

    _jobs[job_id] = {"total": len(rows), "completed": 0, "results": [], "status": "running"}
    background_tasks.add_task(_process_batch, job_id, rows, saved)
    return {"job_id": job_id, "total": len(rows)}


@router.get("/batch/{job_id}/status")
def batch_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/batch/{job_id}/download")
def download_csv(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not ready or not found")

    output = io.StringIO()
    result_fields = ["image_filename", "overall_pass", "error"] + [
        f for f in _CSV_FIELDS if f != "image_filename"
    ]
    writer = csv.DictWriter(output, fieldnames=result_fields, extrasaction="ignore")
    writer.writeheader()

    for r in job["results"]:
        row: dict = {
            "image_filename": r.get("image_filename", ""),
            "overall_pass": r.get("overall_pass", False),
            "error": r.get("error", ""),
        }
        for fr in r.get("fields", []):
            row[fr["field"]] = fr["status"]
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{job_id[:8]}.csv"},
    )


async def _process_batch(job_id: str, rows: list, saved_images: Dict[str, Path]):
    batch_id = str(uuid.uuid4())
    for row in rows:
        filename = row.get("image_filename", "").strip()
        image_path = saved_images.get(filename)

        if not image_path or not image_path.exists():
            _jobs[job_id]["results"].append(
                {"image_filename": filename, "error": "Image not found", "overall_pass": False, "fields": []}
            )
            _jobs[job_id]["completed"] += 1
            continue

        try:
            form_data = LabelFields(
                brand_name=row.get("brand_name") or None,
                class_type=row.get("class_type") or None,
                alcohol_content=row.get("alcohol_content") or None,
                net_contents=row.get("net_contents") or None,
                producer_name_address=row.get("producer_name_address") or None,
                country_of_origin=row.get("country_of_origin") or None,
                government_warning=row.get("government_warning") or None,
            )
            extracted = await extract_label_fields(image_path)
            result = compare_label(extracted, form_data)
            save_verification(
                image_filename=filename,
                form_data=form_data.model_dump(),
                extracted=extracted.model_dump(),
                results=result.model_dump(),
                overall_pass=result.overall_pass,
                batch_id=batch_id,
            )
            _jobs[job_id]["results"].append(
                {
                    "image_filename": filename,
                    "overall_pass": result.overall_pass,
                    "fields": [f.model_dump() for f in result.fields],
                }
            )
        except Exception as e:
            _jobs[job_id]["results"].append(
                {"image_filename": filename, "error": str(e), "overall_pass": False, "fields": []}
            )
        finally:
            _jobs[job_id]["completed"] += 1

    _jobs[job_id]["status"] = "done"
    for path in saved_images.values():
        path.unlink(missing_ok=True)
```

- [ ] **Step 2: Verify server starts with both routers**

```bash
uvicorn app.main:app --reload
```

Expected: `GET http://localhost:8000/docs` shows `/extract`, `/verify`, `/batch`, `/batch/{job_id}/status`, `/batch/{job_id}/download`.

- [ ] **Step 3: Commit**

```bash
git add app/routers/batch.py
git commit -m "feat: batch verification endpoint with background processing and CSV download"
```

---

## Task 11: UI — Single Label Verification (index.html)

**Files:**
- Create: `app/static/index.html`

- [ ] **Step 1: Write app/static/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TTB Label Verification</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen">

  <!-- Nav -->
  <nav class="bg-white border-b border-gray-200 px-6 py-3 flex gap-6">
    <a href="/" class="font-semibold text-blue-600">Verify</a>
    <a href="/batch-upload" class="text-gray-600 hover:text-blue-600">Batch</a>
    <a href="/history" class="text-gray-600 hover:text-blue-600">History</a>
  </nav>

  <main class="max-w-5xl mx-auto p-6">
    <h1 class="text-2xl font-bold text-gray-800 mb-6">Label Verification</h1>

    <div class="grid grid-cols-2 gap-6">

      <!-- Left: Image Upload -->
      <div class="bg-white rounded-lg shadow p-5">
        <h2 class="font-semibold text-gray-700 mb-3">Label Image</h2>
        <label id="drop-zone"
          class="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg h-48 cursor-pointer hover:border-blue-400 transition-colors">
          <span class="text-gray-400 text-sm" id="drop-label">Click or drag image here</span>
          <input type="file" id="image-input" accept="image/*" class="hidden" />
        </label>
        <img id="image-preview" class="mt-3 rounded hidden max-h-48 w-full object-contain" />
        <button id="extract-btn"
          class="mt-3 w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded disabled:opacity-50"
          disabled>
          Extract from Image
        </button>
      </div>

      <!-- Right: Form Fields -->
      <div class="bg-white rounded-lg shadow p-5">
        <h2 class="font-semibold text-gray-700 mb-3">Application Data</h2>
        <div class="space-y-3">
          <div>
            <label class="block text-xs text-gray-500 mb-1">Brand Name</label>
            <input id="brand_name" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Class / Type</label>
            <input id="class_type" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Alcohol Content</label>
            <input id="alcohol_content" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Net Contents</label>
            <input id="net_contents" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Producer Name &amp; Address</label>
            <input id="producer_name_address" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Country of Origin</label>
            <input id="country_of_origin" type="text" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1">Government Warning</label>
            <textarea id="government_warning" rows="3" class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"></textarea>
          </div>
        </div>
        <button id="verify-btn"
          class="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded disabled:opacity-50"
          disabled>
          Verify Label
        </button>
      </div>
    </div>

    <!-- Results -->
    <div id="results" class="mt-6 hidden bg-white rounded-lg shadow p-5">
      <div id="overall-banner" class="text-lg font-bold mb-4 p-3 rounded"></div>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-500 border-b">
            <th class="pb-2">Field</th>
            <th class="pb-2">Status</th>
            <th class="pb-2">Extracted from Label</th>
            <th class="pb-2">Submitted Value</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>

    <div id="error-msg" class="mt-4 hidden text-red-600 bg-red-50 border border-red-200 rounded p-3 text-sm"></div>
  </main>

  <script>
    const FIELDS = ['brand_name','class_type','alcohol_content','net_contents','producer_name_address','country_of_origin','government_warning'];
    const FIELD_LABELS = {
      brand_name: 'Brand Name', class_type: 'Class / Type', alcohol_content: 'Alcohol Content',
      net_contents: 'Net Contents', producer_name_address: 'Producer Name & Address',
      country_of_origin: 'Country of Origin', government_warning: 'Government Warning'
    };
    const STATUS_STYLE = {
      pass: 'text-green-600 font-semibold', fail: 'text-red-600 font-semibold',
      warn: 'text-yellow-600 font-semibold', not_detected: 'text-gray-400'
    };
    const STATUS_ICON = { pass: '✓', fail: '✗', warn: '⚠', not_detected: '—' };

    let selectedFile = null;

    document.getElementById('drop-zone').addEventListener('click', () => {
      document.getElementById('image-input').click();
    });

    document.getElementById('image-input').addEventListener('change', (e) => {
      selectedFile = e.target.files[0];
      if (!selectedFile) return;
      document.getElementById('drop-label').textContent = selectedFile.name;
      const preview = document.getElementById('image-preview');
      preview.src = URL.createObjectURL(selectedFile);
      preview.classList.remove('hidden');
      document.getElementById('extract-btn').disabled = false;
      document.getElementById('verify-btn').disabled = false;
    });

    document.getElementById('extract-btn').addEventListener('click', async () => {
      if (!selectedFile) return;
      const btn = document.getElementById('extract-btn');
      btn.textContent = 'Extracting...';
      btn.disabled = true;
      const fd = new FormData();
      fd.append('image', selectedFile);
      try {
        const res = await fetch('/extract', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail);
        const data = await res.json();
        FIELDS.forEach(f => {
          if (data[f]) document.getElementById(f).value = data[f];
        });
      } catch (err) {
        showError(err.message);
      } finally {
        btn.textContent = 'Extract from Image';
        btn.disabled = false;
      }
    });

    document.getElementById('verify-btn').addEventListener('click', async () => {
      if (!selectedFile) return;
      const btn = document.getElementById('verify-btn');
      btn.textContent = 'Verifying...';
      btn.disabled = true;
      document.getElementById('error-msg').classList.add('hidden');
      document.getElementById('results').classList.add('hidden');

      const fd = new FormData();
      fd.append('image', selectedFile);
      FIELDS.forEach(f => fd.append(f, document.getElementById(f).value));

      try {
        const res = await fetch('/verify', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail);
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        showError(err.message);
      } finally {
        btn.textContent = 'Verify Label';
        btn.disabled = false;
      }
    });

    function renderResults(data) {
      const banner = document.getElementById('overall-banner');
      if (data.overall_pass) {
        banner.textContent = '✓ APPROVED — All fields match';
        banner.className = 'text-lg font-bold mb-4 p-3 rounded bg-green-50 text-green-700';
      } else {
        banner.textContent = '✗ REJECTED — One or more fields failed';
        banner.className = 'text-lg font-bold mb-4 p-3 rounded bg-red-50 text-red-700';
      }
      const tbody = document.getElementById('results-body');
      tbody.innerHTML = data.fields.map(f => `
        <tr class="border-b last:border-0">
          <td class="py-2 text-gray-700">${FIELD_LABELS[f.field] || f.field}</td>
          <td class="py-2 ${STATUS_STYLE[f.status]}">${STATUS_ICON[f.status]} ${f.status.replace('_', ' ')}</td>
          <td class="py-2 text-gray-600 text-xs max-w-xs truncate">${f.extracted_value ?? '—'}</td>
          <td class="py-2 text-gray-600 text-xs max-w-xs truncate">${f.submitted_value ?? '—'}</td>
        </tr>`).join('');
      document.getElementById('results').classList.remove('hidden');
    }

    function showError(msg) {
      const el = document.getElementById('error-msg');
      el.textContent = msg;
      el.classList.remove('hidden');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify page loads**

Start server with `uvicorn app.main:app --reload` and open `http://localhost:8000`. Expected: Two-panel layout renders, image upload zone visible, form fields visible.

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "feat: single label verification UI"
```

---

## Task 12: UI — Batch Upload (batch.html)

**Files:**
- Create: `app/static/batch.html`

- [ ] **Step 1: Write app/static/batch.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TTB Label Verification — Batch</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen">

  <nav class="bg-white border-b border-gray-200 px-6 py-3 flex gap-6">
    <a href="/" class="text-gray-600 hover:text-blue-600">Verify</a>
    <a href="/batch-upload" class="font-semibold text-blue-600">Batch</a>
    <a href="/history" class="text-gray-600 hover:text-blue-600">History</a>
  </nav>

  <main class="max-w-3xl mx-auto p-6">
    <h1 class="text-2xl font-bold text-gray-800 mb-2">Batch Verification</h1>
    <p class="text-sm text-gray-500 mb-6">Upload multiple label images and a CSV file with application data.</p>

    <div class="bg-white rounded-lg shadow p-5 space-y-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Label Images (select multiple)</label>
        <input type="file" id="images-input" accept="image/*" multiple
          class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
        <p id="images-count" class="text-xs text-gray-400 mt-1"></p>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Application Data (CSV)</label>
        <input type="file" id="csv-input" accept=".csv"
          class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-gray-50 file:text-gray-700 hover:file:bg-gray-100" />
        <p class="text-xs text-gray-400 mt-1">
          Columns: <code class="bg-gray-100 px-1 rounded">image_filename, brand_name, class_type, alcohol_content, net_contents, producer_name_address, country_of_origin, government_warning</code>
        </p>
      </div>

      <button id="run-btn"
        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded disabled:opacity-50"
        disabled>
        Run Batch
      </button>
    </div>

    <!-- Progress -->
    <div id="progress-section" class="hidden mt-6 bg-white rounded-lg shadow p-5">
      <div class="flex justify-between text-sm text-gray-600 mb-2">
        <span id="progress-label">Processing...</span>
        <span id="progress-count"></span>
      </div>
      <div class="w-full bg-gray-200 rounded-full h-3">
        <div id="progress-bar" class="bg-blue-600 h-3 rounded-full transition-all duration-300" style="width:0%"></div>
      </div>
    </div>

    <!-- Results -->
    <div id="results-section" class="hidden mt-6 bg-white rounded-lg shadow p-5">
      <div class="flex justify-between items-center mb-3">
        <h2 class="font-semibold text-gray-700">Results</h2>
        <a id="download-btn" href="#"
          class="bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium py-1.5 px-3 rounded">
          Download CSV
        </a>
      </div>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-500 border-b">
            <th class="pb-2">File</th>
            <th class="pb-2">Result</th>
            <th class="pb-2">Details</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>

    <div id="error-msg" class="hidden mt-4 text-red-600 bg-red-50 border border-red-200 rounded p-3 text-sm"></div>
  </main>

  <script>
    let jobId = null;
    let pollInterval = null;

    document.getElementById('images-input').addEventListener('change', (e) => {
      const count = e.target.files.length;
      document.getElementById('images-count').textContent = `${count} image${count !== 1 ? 's' : ''} selected`;
      checkReady();
    });

    document.getElementById('csv-input').addEventListener('change', checkReady);

    function checkReady() {
      const hasImages = document.getElementById('images-input').files.length > 0;
      const hasCsv = document.getElementById('csv-input').files.length > 0;
      document.getElementById('run-btn').disabled = !(hasImages && hasCsv);
    }

    document.getElementById('run-btn').addEventListener('click', async () => {
      const btn = document.getElementById('run-btn');
      btn.disabled = true;
      document.getElementById('error-msg').classList.add('hidden');
      document.getElementById('results-section').classList.add('hidden');

      const fd = new FormData();
      Array.from(document.getElementById('images-input').files).forEach(f => fd.append('images', f));
      fd.append('csv_file', document.getElementById('csv-input').files[0]);

      try {
        const res = await fetch('/batch', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail);
        const data = await res.json();
        jobId = data.job_id;
        startPolling(data.total);
      } catch (err) {
        showError(err.message);
        btn.disabled = false;
      }
    });

    function startPolling(total) {
      document.getElementById('progress-section').classList.remove('hidden');
      pollInterval = setInterval(async () => {
        const res = await fetch(`/batch/${jobId}/status`);
        const job = await res.json();
        const pct = total > 0 ? Math.round((job.completed / total) * 100) : 0;
        document.getElementById('progress-bar').style.width = pct + '%';
        document.getElementById('progress-count').textContent = `${job.completed} / ${total}`;
        if (job.status === 'done') {
          clearInterval(pollInterval);
          document.getElementById('progress-label').textContent = 'Complete';
          renderResults(job.results, jobId);
        }
      }, 2000);
    }

    function renderResults(results, jid) {
      document.getElementById('download-btn').href = `/batch/${jid}/download`;
      const tbody = document.getElementById('results-body');
      tbody.innerHTML = results.map(r => {
        const status = r.error
          ? `<span class="text-red-500 font-semibold">Error: ${r.error}</span>`
          : r.overall_pass
            ? '<span class="text-green-600 font-semibold">✓ Pass</span>'
            : '<span class="text-red-600 font-semibold">✗ Fail</span>';
        const fails = (r.fields || []).filter(f => f.status === 'fail').map(f => f.field).join(', ');
        return `<tr class="border-b last:border-0">
          <td class="py-2 text-gray-700">${r.image_filename}</td>
          <td class="py-2">${status}</td>
          <td class="py-2 text-xs text-gray-500">${fails ? 'Failed: ' + fails : (r.error ? '' : 'All fields matched')}</td>
        </tr>`;
      }).join('');
      document.getElementById('results-section').classList.remove('hidden');
    }

    function showError(msg) {
      const el = document.getElementById('error-msg');
      el.textContent = msg;
      el.classList.remove('hidden');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify page loads**

Open `http://localhost:8000/batch-upload`. Expected: File upload inputs visible, Run Batch button disabled until files selected.

- [ ] **Step 3: Commit**

```bash
git add app/static/batch.html
git commit -m "feat: batch upload UI with progress polling and CSV download"
```

---

## Task 13: UI — Verification History (history.html)

**Files:**
- Create: `app/static/history.html`

- [ ] **Step 1: Write app/static/history.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TTB Label Verification — History</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen">

  <nav class="bg-white border-b border-gray-200 px-6 py-3 flex gap-6">
    <a href="/" class="text-gray-600 hover:text-blue-600">Verify</a>
    <a href="/batch-upload" class="text-gray-600 hover:text-blue-600">Batch</a>
    <a href="/history" class="font-semibold text-blue-600">History</a>
  </nav>

  <main class="max-w-5xl mx-auto p-6">
    <h1 class="text-2xl font-bold text-gray-800 mb-6">Verification History</h1>

    <div class="bg-white rounded-lg shadow overflow-hidden">
      <table class="w-full text-sm">
        <thead class="bg-gray-50 border-b">
          <tr class="text-left text-gray-500">
            <th class="px-4 py-3">Date / Time</th>
            <th class="px-4 py-3">File</th>
            <th class="px-4 py-3">Type</th>
            <th class="px-4 py-3">Result</th>
            <th class="px-4 py-3"></th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="5" class="px-4 py-6 text-center text-gray-400">Loading...</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Detail Panel -->
    <div id="detail-panel" class="hidden mt-6 bg-white rounded-lg shadow p-5">
      <div class="flex justify-between items-center mb-3">
        <h2 id="detail-title" class="font-semibold text-gray-700"></h2>
        <button onclick="document.getElementById('detail-panel').classList.add('hidden')"
          class="text-gray-400 hover:text-gray-600 text-lg">✕</button>
      </div>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-500 border-b">
            <th class="pb-2">Field</th>
            <th class="pb-2">Status</th>
            <th class="pb-2">Extracted</th>
            <th class="pb-2">Submitted</th>
          </tr>
        </thead>
        <tbody id="detail-body"></tbody>
      </table>
    </div>
  </main>

  <script>
    const FIELD_LABELS = {
      brand_name: 'Brand Name', class_type: 'Class / Type', alcohol_content: 'Alcohol Content',
      net_contents: 'Net Contents', producer_name_address: 'Producer Name & Address',
      country_of_origin: 'Country of Origin', government_warning: 'Government Warning'
    };
    const STATUS_STYLE = {
      pass: 'text-green-600 font-semibold', fail: 'text-red-600 font-semibold',
      warn: 'text-yellow-600 font-semibold', not_detected: 'text-gray-400'
    };
    const STATUS_ICON = { pass: '✓', fail: '✗', warn: '⚠', not_detected: '—' };

    async function loadHistory() {
      const res = await fetch('/api/history');
      const records = await res.json();
      const tbody = document.getElementById('history-body');
      if (!records.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="px-4 py-6 text-center text-gray-400">No verifications yet.</td></tr>';
        return;
      }
      tbody.innerHTML = records.map(r => {
        const dt = new Date(r.created_at + 'Z').toLocaleString();
        const type = r.batch_id ? 'Batch' : 'Single';
        const result = r.overall_pass
          ? '<span class="text-green-600 font-semibold">✓ Pass</span>'
          : '<span class="text-red-600 font-semibold">✗ Fail</span>';
        return `<tr class="border-b last:border-0 hover:bg-gray-50 cursor-pointer" onclick="showDetail(${r.id}, '${r.image_filename}')">
          <td class="px-4 py-3 text-gray-600">${dt}</td>
          <td class="px-4 py-3 text-gray-800">${r.image_filename}</td>
          <td class="px-4 py-3 text-gray-500">${type}</td>
          <td class="px-4 py-3">${result}</td>
          <td class="px-4 py-3 text-blue-500 text-xs">View</td>
        </tr>`;
      }).join('');
    }

    async function showDetail(id, filename) {
      const res = await fetch(`/api/history/${id}`);
      const record = await res.json();
      const results = JSON.parse(record.results);
      document.getElementById('detail-title').textContent = filename;
      document.getElementById('detail-body').innerHTML = results.fields.map(f => `
        <tr class="border-b last:border-0">
          <td class="py-2 text-gray-700">${FIELD_LABELS[f.field] || f.field}</td>
          <td class="py-2 ${STATUS_STYLE[f.status]}">${STATUS_ICON[f.status]} ${f.status.replace('_', ' ')}</td>
          <td class="py-2 text-gray-600 text-xs max-w-xs truncate">${f.extracted_value ?? '—'}</td>
          <td class="py-2 text-gray-600 text-xs max-w-xs truncate">${f.submitted_value ?? '—'}</td>
        </tr>`).join('');
      document.getElementById('detail-panel').classList.remove('hidden');
    }

    loadHistory();
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify page loads**

Open `http://localhost:8000/history`. Expected: "No verifications yet" shown if DB is empty. After running a verification, the row appears.

- [ ] **Step 3: Commit**

```bash
git add app/static/history.html
git commit -m "feat: verification history UI with expandable detail panel"
```

---

## Task 14: Docker Setup + README

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
version: "3.9"

services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:11434/api/tags || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s

  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy

volumes:
  ollama_data:
```

- [ ] **Step 3: Write README.md**

```markdown
# TTB Label Verification

AI-powered alcohol label verification prototype for the Alcohol and Tobacco Tax and Trade Bureau (TTB). Extracts fields from label images using a local vision model (llava-phi3 via Ollama) and compares them against applicant-submitted application data.

## Requirements

- Docker and Docker Compose
- GPU strongly recommended (llava-phi3 runs in ~2-4s on GPU; ~15-30s on CPU)

## Quick Start

\`\`\`bash
git clone <repo-url>
cd <repo>

# Start services (pulls llava-phi3 on first run — ~2.2GB download)
docker compose up --build

# Pull the vision model (run once)
docker compose exec ollama ollama pull llava-phi3
\`\`\`

Open **http://localhost:8000**

## Usage

### Single Label Verification

1. Upload a label image
2. Click **Extract from Image** to auto-fill fields (optional — review and correct)
3. Fill in the application data fields
4. Click **Verify Label**
5. Review the per-field pass / fail / warn results

### Batch Verification

1. Navigate to **Batch**
2. Upload multiple label images
3. Upload a CSV file with columns: `image_filename, brand_name, class_type, alcohol_content, net_contents, producer_name_address, country_of_origin, government_warning`
4. Click **Run Batch** — progress updates every 2 seconds
5. Download the CSV summary when complete

### History

All verifications are saved and viewable under **History**.

## CSV Format Example

\`\`\`csv
image_filename,brand_name,class_type,alcohol_content,net_contents,producer_name_address,country_of_origin,government_warning
label1.jpg,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45%,750 mL,"Old Tom Distillery, Louisville KY",United States,GOVERNMENT WARNING: ...
\`\`\`

## Running Tests

\`\`\`bash
pip install -r requirements.txt
pytest tests/ -v
\`\`\`

## Field Comparison Rules

| Field | Strategy |
|---|---|
| Brand name | Fuzzy match ≥ 90% |
| Class / type | Fuzzy match ≥ 85% |
| Alcohol content | Numeric normalize (strips %, Alc./Vol., proof) |
| Net contents | Numeric normalize (converts mL, L, fl oz) |
| Producer name / address | Fuzzy match ≥ 80% |
| Country of origin | Exact match (case-insensitive) |
| Government warning | Must contain `GOVERNMENT WARNING:` in all-caps |

**Status meanings:**
- ✓ Pass — match within threshold
- ✗ Fail — match below threshold (or exact check failed)
- ⚠ Warn — near-miss (below pass threshold but ≥ 70%) — flagged for human review, does not auto-reject
- — Not detected — field not visible on label, not auto-rejected

## Known Limitations

- **Latency:** GPU strongly recommended to meet the 5-second target. CPU-only machines will see 15–30s per label with llava-phi3.
- **Decorative fonts:** Small VLMs can misread highly stylized label typography. The Extract & Review flow lets agents correct bad extractions.
- **No authentication:** Prototype scope only.
- **Batch job state:** In-memory only — jobs are lost on server restart.
```

- [ ] **Step 4: Build and start with Docker Compose**

```bash
docker compose up --build
```

Expected: Both services start. `ollama` becomes healthy, then `api` starts on port 8000.

- [ ] **Step 5: Pull the model**

```bash
docker compose exec ollama ollama pull llava-phi3
```

Expected: Model downloads (~2.2GB). Completes with `success`.

- [ ] **Step 6: Smoke test the full flow**

Open `http://localhost:8000`, upload a test label image, fill in form fields, click Verify. Expected: Results panel renders with per-field statuses.

- [ ] **Step 7: Run tests**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 8: Final commit**

```bash
git add Dockerfile docker-compose.yml README.md
git commit -m "feat: Docker Compose setup and README"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] VLM extraction via Ollama (Task 8)
- [x] Deterministic comparator — all 7 fields (Tasks 4–7)
- [x] <5s target documented with hardware note (README)
- [x] Government warning exact match (Task 4)
- [x] Fuzzy matching with warn threshold (Task 5)
- [x] ABV + net contents numeric normalization (Tasks 6–7)
- [x] Single label verification endpoint + UI (Tasks 9, 11)
- [x] Batch upload + polling + CSV download (Tasks 10, 12)
- [x] SQLite persistence (Task 3)
- [x] Verification history UI (Task 13)
- [x] Error handling (Tasks 9, 10 — httpx errors mapped to HTTP codes)
- [x] Docker Compose (Task 14)
- [x] Tests with in-memory DB fixture (Tasks 3–7)
