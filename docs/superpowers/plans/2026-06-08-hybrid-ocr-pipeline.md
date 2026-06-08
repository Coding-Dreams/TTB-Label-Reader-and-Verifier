# Hybrid OCR + VLM Label Extraction Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-VLM extraction pipeline with a three-stage hybrid: pytesseract OCR → rule-based extraction for structured fields → Qwen2.5VL for semantic fields only (brand name, class type, US producer/importer).

**Architecture:** Stage 1 runs pytesseract on both panels at full resolution. Stage 2 applies regex to OCR text for deterministic fields (ABV, volume, government warning, sulfites, class type keywords). Stage 3 sends the front-panel image + OCR text as context to Qwen for semantic fields only (brand_name, class_type confirmed, producer_name_address as US entity, country_of_origin). Results are merged with Stage 2 taking precedence for structured fields.

**Tech Stack:** Python 3.11, pytesseract, Pillow, httpx, Qwen2.5VL:7b via Ollama

**Branch:** `feature/hybrid-ocr-pipeline`

---

## File Map

```
app/services/ollama.py   — full rewrite; add _ocr_image(), _rule_extract(),
                           new _SEMANTIC_PROMPT_TEMPLATE; remove _stitch_images(),
                           _extract_importer(), _extract_sulfites(), _extract_class_type(),
                           _extract_brand_name(), _extract_net_contents(), _is_us_address()
requirements.txt         — add pytesseract==0.3.13
Dockerfile               — add apt-get install tesseract-ocr before pip install
tests/test_rule_extract.py — new; unit tests for _rule_extract() and _ocr_image() integration
```

---

## Task 1: Branch + Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 1: Add pytesseract to requirements.txt**

Replace the file content with:

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
Pillow==10.4.0
pytesseract==0.3.13
```

- [ ] **Step 2: Update Dockerfile to install tesseract-ocr system package**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Install pytesseract locally for test development**

```bash
pip install pytesseract==0.3.13
# Also install system tesseract if not present (Ubuntu/Debian):
# sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

- [ ] **Step 4: Verify pytesseract import works**

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Expected: prints a version like `5.x.x`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt Dockerfile
git commit -m "feat: add pytesseract dependency and tesseract-ocr to Dockerfile"
```

---

## Task 2: Add `_ocr_image()` to ollama.py

**Files:**
- Modify: `app/services/ollama.py`

`_ocr_image()` runs pytesseract on a single image panel and returns the extracted text as a string. It upsizes small images to at least 1200px longest side before OCR (Tesseract accuracy degrades on small images) and applies contrast enhancement.

- [ ] **Step 1: Add the import and function to ollama.py**

Add `import pytesseract` at the top of `app/services/ollama.py` with the other imports.

Then add this function after `_encode_image()`:

```python
def _ocr_image(image_path: Path) -> str:
    """Run Tesseract OCR on a label image. Upscales and enhances contrast before passing to Tesseract."""
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) < 1200:
            scale = 1200 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        return pytesseract.image_to_string(img, config="--psm 6")
```

`--psm 6` tells Tesseract to treat the image as a single uniform block of text, which works well for label panels.

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from app.services.ollama import _ocr_image; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke test on a real label**

```bash
python -c "
from pathlib import Path
from app.services.ollama import _ocr_image
text = _ocr_image(Path('testLabels/COLA1/Label1.jpg'))
print(repr(text[:300]))
"
```

Expected: a string containing readable text from the label (brand name, warning text, etc.). Not empty.

- [ ] **Step 4: Commit**

```bash
git add app/services/ollama.py
git commit -m "feat: add _ocr_image() using pytesseract for Stage 1 OCR"
```

---

## Task 3: Add `_rule_extract()` with unit tests

**Files:**
- Modify: `app/services/ollama.py`
- Create: `tests/test_rule_extract.py`

`_rule_extract()` is a pure function: takes OCR text, returns a dict of field values for the fields it can extract deterministically. It handles alcohol_content, net_contents, government_warning, contains_sulfites, and class_type (keyword match). It never returns wrong values — if the regex doesn't match confidently, it leaves the field absent from the dict (so Stage 3 VLM fills it in).

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_rule_extract.py`:

```python
"""Unit tests for _rule_extract() — the rule-based Stage 2 extraction."""
import pytest
from app.services.ollama import _rule_extract


# --- alcohol_content ---

def test_rule_extract_abv_simple():
    text = "ALCOHOL CONTENT: 35%"
    result = _rule_extract(text)
    assert result.get("alcohol_content") is not None
    assert "35" in result["alcohol_content"]


def test_rule_extract_abv_with_qualifier():
    text = "61% ALC./VOL (122 PROOF)"
    result = _rule_extract(text)
    assert result.get("alcohol_content") is not None
    assert "61" in result["alcohol_content"]


def test_rule_extract_abv_by_vol():
    text = "ALC. 21% BY VOL. / 42 PROOF"
    result = _rule_extract(text)
    assert result.get("alcohol_content") is not None
    assert "21" in result["alcohol_content"]


def test_rule_extract_abv_decimal():
    text = "13.0% ALC by VOL"
    result = _rule_extract(text)
    assert result.get("alcohol_content") is not None
    assert "13" in result["alcohol_content"]


# --- net_contents ---

def test_rule_extract_volume_ml():
    text = "NET CONTENTS: 750ML"
    result = _rule_extract(text)
    assert result.get("net_contents") is not None
    assert "750" in result["net_contents"]


def test_rule_extract_volume_litre():
    text = "1.5L"
    result = _rule_extract(text)
    assert result.get("net_contents") is not None
    assert "1.5" in result["net_contents"]


def test_rule_extract_volume_lowercase():
    text = "100ml net contents"
    result = _rule_extract(text)
    assert result.get("net_contents") is not None
    assert "100" in result["net_contents"]


# --- government_warning ---

def test_rule_extract_government_warning_found():
    text = (
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
        "alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems."
    )
    result = _rule_extract(text)
    assert result.get("government_warning") is not None
    assert result["government_warning"].startswith("GOVERNMENT WARNING:")


def test_rule_extract_government_warning_absent():
    text = "Brand Name: Cascade Val. Alcohol: 11.5%"
    result = _rule_extract(text)
    assert result.get("government_warning") is None


# --- contains_sulfites ---

def test_rule_extract_sulfites_present():
    text = "CONTAINS SULFITES"
    result = _rule_extract(text)
    assert result.get("contains_sulfites") is not None
    assert "sulfite" in result["contains_sulfites"].lower()


def test_rule_extract_sulfites_free():
    text = "SULFITE FREE"
    result = _rule_extract(text)
    assert result.get("contains_sulfites") is not None
    assert "sulfite" in result["contains_sulfites"].lower()


def test_rule_extract_sulfites_absent():
    text = "Distilled Spirits, 750ml, 40% ALC/VOL"
    result = _rule_extract(text)
    assert result.get("contains_sulfites") is None


def test_rule_extract_sulfites_no_detectable():
    text = "Contains No Detectable Sulfites"
    result = _rule_extract(text)
    assert result.get("contains_sulfites") is not None


# --- class_type ---

def test_rule_extract_class_wine():
    text = "California Cabernet Sauvignon Wine"
    result = _rule_extract(text)
    assert result.get("class_type") == "Wine"


def test_rule_extract_class_malt():
    text = "India Pale Ale brewed and bottled by"
    result = _rule_extract(text)
    assert result.get("class_type") == "Malt Beverage"


def test_rule_extract_class_spirits():
    text = "Kentucky Straight Bourbon Whiskey"
    result = _rule_extract(text)
    assert result.get("class_type") == "Distilled Spirits"


def test_rule_extract_class_absent():
    text = "Product of France 750ML"
    result = _rule_extract(text)
    # No beverage type keyword — class_type should be absent so VLM can handle it
    assert result.get("class_type") is None


# --- empty / garbage input ---

def test_rule_extract_empty_string():
    result = _rule_extract("")
    assert result == {}


def test_rule_extract_no_matches():
    result = _rule_extract("Lorem ipsum dolor sit amet")
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/home/nico/Documents/Personal/Treasury/Take Home"
python -m pytest tests/test_rule_extract.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_rule_extract' from 'app.services.ollama'`

- [ ] **Step 3: Add `_rule_extract()` to ollama.py**

Add after `_ocr_image()` in `app/services/ollama.py`. The `_CLASS_TYPE_KEYWORDS` list already exists in the file — this function reuses it.

```python
# Matches the main government warning block
_GOV_WARNING_RE = re.compile(
    r'(GOVERNMENT\s+WARNING\s*:.+)',
    re.IGNORECASE | re.DOTALL,
)

# Matches ABV statements: "35%", "13.0% ALC by VOL", "ALC. 21% BY VOL. / 42 PROOF"
_ABV_RE = re.compile(
    r'(\d+\.?\d*\s*%(?:\s*(?:alc\.?[/\s]?vol\.?|by\s+vol\.?|proof))?(?:\s*/\s*\d+\s*proof)?)',
    re.IGNORECASE,
)

# Matches volume quantities: "750ml", "1.5L", "100mL", "1 pint"
_VOLUME_RE = re.compile(
    r'(\d+\.?\d*\s*(?:ml\b|l\b|fl\.?\s*oz\b|fluid\s*oz\b|pint\b))',
    re.IGNORECASE,
)

# Matches any sulfite mention (positive or negative)
_SULFITE_RE = re.compile(
    r'((?:contains?\s+)?(?:no\s+detectable\s+)?sulfites?'
    r'|sulfiting\s+agents?'
    r'|sulfite\s+free'
    r'|no\s+sulfites?\s+added)',
    re.IGNORECASE,
)


def _rule_extract(text: str) -> dict:
    """Stage 2: extract structured fields from OCR text using deterministic rules.

    Returns only the fields that were matched confidently. Absent keys mean
    Stage 3 (VLM) will handle that field.
    """
    result: dict = {}

    # alcohol_content — take the first % match
    abv_m = _ABV_RE.search(text)
    if abv_m:
        result["alcohol_content"] = abv_m.group(1).strip()

    # net_contents — take the first volume match, but exclude pure-digit-% matches
    # (ABV numbers like "35%" should not be picked up as volumes)
    vol_m = _VOLUME_RE.search(text)
    if vol_m:
        result["net_contents"] = vol_m.group(1).strip()

    # government_warning — grab from the keyword to end of paragraph
    gw_m = _GOV_WARNING_RE.search(text)
    if gw_m:
        # Normalise the heading to required ALL-CAPS form
        warning = re.sub(
            r'^government\s+warning\s*:',
            'GOVERNMENT WARNING:',
            gw_m.group(1).strip(),
            flags=re.IGNORECASE,
        )
        # Trim trailing whitespace / newlines
        result["government_warning"] = warning.strip()

    # contains_sulfites — return the matched phrase verbatim
    sf_m = _SULFITE_RE.search(text)
    if sf_m:
        result["contains_sulfites"] = sf_m.group(0).strip()

    # class_type — keyword scan using the existing lookup table
    text_lower = text.lower()
    for keyword, category in _CLASS_TYPE_KEYWORDS:
        if keyword in text_lower:
            result["class_type"] = category
            break

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_rule_extract.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/ollama.py tests/test_rule_extract.py
git commit -m "feat: add _rule_extract() Stage 2 deterministic field extraction with tests"
```

---

## Task 4: New VLM Prompt (Stage 3)

**Files:**
- Modify: `app/services/ollama.py`

Replace `_FULL_PROMPT` (which asked for all 9 fields) with `_SEMANTIC_PROMPT_TEMPLATE` (asks for 4 semantic fields only, with OCR text injected as context). The `us_importer` field is gone — `producer_name_address` is now always the US entity.

- [ ] **Step 1: Replace `_FULL_PROMPT` in ollama.py**

Find the existing `_FULL_PROMPT = """..."""` constant and replace it entirely with:

```python
_SEMANTIC_PROMPT_TEMPLATE = """\
You are an alcohol label field extractor.

OCR text extracted from this label:
{ocr_text}

Using the OCR text above AND the label image, return ONLY valid JSON with these 4 fields:

{{
  "brand_name": "string or null",
  "class_type": "Wine" or "Malt Beverage" or "Distilled Spirits" or null,
  "producer_name_address": "string or null",
  "country_of_origin": "string or null"
}}

Rules:
- brand_name: the product/label name printed on the front (e.g. "Cascade Val", "Fete Rose", "Barenjager") — NOT the producer company name, NOT the brewery/winery/distillery name
- class_type: EXACTLY "Wine", "Malt Beverage", or "Distilled Spirits". Wine = grape/fruit wine, champagne, cider. Malt Beverage = beer, ale, lager, IPA, hard seltzer. Distilled Spirits = whiskey, bourbon, vodka, gin, rum, tequila, brandy, cognac, liqueur. If VODKA/GIN/RUM/WHISKEY appears on the label, always use "Distilled Spirits" even for canned cocktails.
- producer_name_address: the US BOTTLER, IMPORTER, or DOMESTIC PRODUCER — a company with a United States address. For imported products look for "IMPORTED BY:", "SOLE IMPORTER:", "BOTTLED BY:" followed by a US company and city/state. For domestic products return the US producer address. Return null only if truly no US entity is present.
- country_of_origin: only if the label explicitly states the product's origin country (e.g. "Product of France", "Made in Germany"). Do NOT infer from beverage style. Return null otherwise.
- Set any field to null if not found.

Return ONLY the JSON object, no markdown, no explanation.

Example for an imported Austrian wine with an NJ importer:
{{
  "brand_name": "Fete Rose",
  "class_type": "Wine",
  "producer_name_address": "Niche W. & S., CEDAR KNOLLS, NJ",
  "country_of_origin": "AUSTRIA"
}}"""
```

- [ ] **Step 2: Verify the file still imports cleanly**

```bash
python -c "from app.services.ollama import _SEMANTIC_PROMPT_TEMPLATE; print(_SEMANTIC_PROMPT_TEMPLATE[:80])"
```

Expected: prints the first 80 chars of the prompt string without errors.

- [ ] **Step 3: Commit**

```bash
git add app/services/ollama.py
git commit -m "feat: replace _FULL_PROMPT with _SEMANTIC_PROMPT_TEMPLATE for 4-field semantic extraction"
```

---

## Task 5: Rewrite `extract_label_fields()` to orchestrate the three-stage pipeline

**Files:**
- Modify: `app/services/ollama.py`

This is the main orchestration change. `extract_label_fields()` now runs OCR → rules → VLM and merges the results. `_call_ollama()` is updated to accept an optional `prompt` parameter so Stage 3 can inject OCR text.

- [ ] **Step 1: Replace the `extract_label_fields()` function body**

The existing function signature stays identical (`image_path`, `back_image_path`) so no callers change. Replace the full function body:

```python
async def extract_label_fields(
    image_path: Path, back_image_path: Optional[Path] = None
) -> LabelFields:
    loop = asyncio.get_running_loop()

    # Stage 1 — OCR both panels at full resolution
    try:
        front_text = await loop.run_in_executor(None, _ocr_image, image_path)
        back_text = ""
        if back_image_path and back_image_path.exists():
            back_text = await loop.run_in_executor(None, _ocr_image, back_image_path)
        combined_text = (front_text + "\n\n" + back_text).strip()
    except Exception as e:
        logger.warning("OCR failed, falling back to VLM-only: %s", e)
        combined_text = ""

    # Stage 2 — Rule-based extraction from OCR text
    rule_data = _rule_extract(combined_text) if combined_text else {}

    # Stage 3 — VLM for semantic fields (brand, class, producer, country)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        front_b64 = await loop.run_in_executor(None, _encode_image, image_path, 1024)
        ocr_context = combined_text[:3000] if combined_text else "(OCR unavailable — read from image)"
        prompt = _SEMANTIC_PROMPT_TEMPLATE.format(ocr_text=ocr_context)
        raw = await _call_ollama(client, front_b64, prompt)
        vlm_data = _postprocess(_parse_json(raw))

    # Merge: Stage 2 (rules) takes precedence for structured fields;
    # Stage 3 (VLM) fills semantic fields that rules cannot handle.
    merged: dict = {
        # Semantic fields — VLM only
        "brand_name": vlm_data.get("brand_name"),
        "producer_name_address": vlm_data.get("producer_name_address"),
        "country_of_origin": vlm_data.get("country_of_origin"),
        # class_type — VLM preferred; rule-based keyword match as fallback
        "class_type": vlm_data.get("class_type") or rule_data.get("class_type"),
        # Structured fields — rules take precedence; VLM as fallback
        "alcohol_content": rule_data.get("alcohol_content") or vlm_data.get("alcohol_content"),
        "net_contents": rule_data.get("net_contents") or vlm_data.get("net_contents"),
        "government_warning": rule_data.get("government_warning") or vlm_data.get("government_warning"),
        "contains_sulfites": rule_data.get("contains_sulfites") or vlm_data.get("contains_sulfites"),
    }

    return LabelFields(**merged)
```

Note: `_call_ollama` already accepts an arbitrary `prompt` string — no change needed there.

- [ ] **Step 2: Verify the module still imports**

```bash
python -c "from app.services.ollama import extract_label_fields; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/services/ollama.py
git commit -m "feat: rewrite extract_label_fields() as 3-stage OCR+rules+VLM pipeline"
```

---

## Task 6: Simplify `_postprocess()` — remove dead logic

**Files:**
- Modify: `app/services/ollama.py`

`_postprocess()` was patching over VLM confusion. With the new architecture, three blocks are now dead code and should be removed:

1. The `us_importer` merge into `producer_name_address` — this field no longer exists in the VLM output.
2. The brand_name/class_type swap (where class_type held a product name) — the new focused prompt makes this impossible.
3. The brand_name fallback that inferred brand from producer — the VLM now extracts it directly.

- [ ] **Step 1: Remove the `us_importer` merge block from `_postprocess()`**

Find and delete this block (lines ~542–544 in the current file):

```python
    # If the model extracted a dedicated US importer, it takes precedence over
    # the foreign producer — merge into the single producer_name_address field.
    us_importer = data.pop("us_importer", None)
    if us_importer:
        data["producer_name_address"] = us_importer
```

- [ ] **Step 2: Remove the brand_name/class_type swap block**

Find and delete this block (lines ~528–532):

```python
    # If brand_name is null and class_type doesn't contain any standard beverage-type
    # word, the model likely put the product name in the wrong field — swap them.
    if not data.get("brand_name") and data.get("class_type"):
        ct_lower = data["class_type"].lower()
        if not any(word in ct_lower for word in _BEVERAGE_TYPE_WORDS):
            data["brand_name"] = data["class_type"]
            data["class_type"] = None
```

- [ ] **Step 3: Remove the brand_name-from-producer fallback block**

Find and delete this block (lines ~547–558):

```python
    # Last-resort brand_name fallback for import labels
    if not data.get("brand_name") and data.get("producer_name_address"):
        producer = data["producer_name_address"]
        stripped = re.sub(
            r'^\s*(?:BOTTLED|IMPORTED|PRODUCED|DISTRIBUTED|BREWED|PACKED)\s+BY:?\s*',
            '', producer, flags=re.IGNORECASE,
        ).strip()
        name_part = re.sub(r',?\s+[\w\s]{2,},\s+[A-Z]{2}\s*$', '', stripped).strip()
        if (name_part
                and name_part.lower() != producer.strip().lower()
                and len(name_part) > 2
                and not _COMPANY_TYPE_RE.search(name_part)):
            data["brand_name"] = name_part
```

- [ ] **Step 4: Remove dead regex constants and helper function**

Find and delete these now-unused items from the module:

- The `_US_ADDRESS_TAIL_RE` compiled regex (the big multi-line one with all state names)
- The `_TRAILING_USA_RE` compiled regex
- The `_US_COMPANY_RE` compiled regex
- The `_DOTTED_ABBREV_RE` compiled regex
- The `_is_us_address()` function
- The `_BEVERAGE_TYPE_WORDS` frozenset (only used by the swap block we removed)

- [ ] **Step 5: Verify the module imports and _postprocess still works**

```bash
python -c "
from app.services.ollama import _postprocess
data = {'brand_name': 'Cascade Val', 'class_type': 'Wine', 'alcohol_content': '11.5%',
        'net_contents': '750ML', 'producer_name_address': 'Cascade Winery, Grand Rapids, MI',
        'country_of_origin': None, 'government_warning': 'GOVERNMENT WARNING: test',
        'contains_sulfites': 'CONTAINS SULFITES'}
result = _postprocess(data)
print(result)
"
```

Expected: dict printed with values intact, no errors.

- [ ] **Step 6: Run existing unit tests to confirm no regressions**

```bash
python -m pytest tests/test_comparator.py tests/test_db.py tests/test_rule_extract.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add app/services/ollama.py
git commit -m "refactor: remove dead postprocessing logic (importer merge, brand/class swap, brand fallback)"
```

---

## Task 7: Remove dead extraction functions

**Files:**
- Modify: `app/services/ollama.py`

These five functions were stand-alone second-pass VLM calls. They are now replaced by Stage 2 (rules) or Stage 3 (single VLM call). Delete them:

- `_stitch_images()` — front/back stitching was only needed for the single-image main pass; Stage 1 OCRs each panel separately
- `_extract_brand_name()` — handled by Stage 3 VLM
- `_extract_class_type()` — handled by Stage 3 VLM + Stage 2 keyword fallback
- `_extract_sulfites()` — handled by Stage 2 `_rule_extract()`
- `_extract_importer()` — removed entirely; producer schema fixed in Stage 3 prompt
- `_extract_net_contents()` — handled by Stage 2 `_rule_extract()`

Also remove the `_SINGLE_LINE_FIELDS` set entry for `"us_importer"` since that field no longer exists.

- [ ] **Step 1: Delete the six functions from ollama.py**

Remove the complete function bodies for: `_stitch_images`, `_extract_brand_name`, `_extract_class_type`, `_extract_sulfites`, `_extract_importer`, `_extract_net_contents`.

- [ ] **Step 2: Remove `"us_importer"` from `_SINGLE_LINE_FIELDS`**

The current set is:
```python
_SINGLE_LINE_FIELDS = {"brand_name", "class_type", "alcohol_content", "net_contents", "country_of_origin", "contains_sulfites", "us_importer"}
```

Change to:
```python
_SINGLE_LINE_FIELDS = {"brand_name", "class_type", "alcohol_content", "net_contents", "country_of_origin", "contains_sulfites"}
```

- [ ] **Step 3: Remove `"us_importer"` from `_FIELD_NAMES`**

Current:
```python
_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "us_importer", "country_of_origin", "government_warning", "contains_sulfites",
}
```

Change to:
```python
_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "country_of_origin", "government_warning", "contains_sulfites",
}
```

- [ ] **Step 4: Verify the module imports cleanly**

```bash
python -c "from app.services.ollama import extract_label_fields, _rule_extract, _ocr_image; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run all unit tests**

```bash
python -m pytest tests/test_comparator.py tests/test_db.py tests/test_rule_extract.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/ollama.py
git commit -m "refactor: remove dead extraction functions and us_importer field references"
```

---

## Task 8: End-to-End Smoke Test + Integration Test Run

**Files:**
- No code changes — verification only

- [ ] **Step 1: Confirm the API starts cleanly**

```bash
OLLAMA_BASE_URL=http://localhost:11434 uvicorn app.main:app --port 8000
```

Expected: starts with no import errors.

- [ ] **Step 2: Run all unit tests**

```bash
python -m pytest tests/test_comparator.py tests/test_db.py tests/test_rule_extract.py -v
```

Expected: all PASS.

- [ ] **Step 3: Run integration tests (requires Docker stack running)**

```bash
python -m pytest tests/test_integration.py -v -m integration 2>&1 | tail -40
```

Expected: significantly more tests pass than on master. At minimum:
- All labels where government_warning is in the truth JSON should pass that field (rule extraction)
- All labels where ABV and net contents are in the truth JSON should pass those fields (rule extraction)
- Producer/importer field passes on domestic labels (COLA1, COLA10, COLA13, COLA15, COLA16, COLA19) and imported labels with visible importer text

- [ ] **Step 4: Compare pass rate against master**

```bash
# On feature branch
python -m pytest tests/test_integration.py -v -m integration 2>&1 | grep -E "passed|failed|error"
```

Record the counts. Then checkout master and compare:

```bash
git stash
git checkout master
python -m pytest tests/test_integration.py -v -m integration 2>&1 | grep -E "passed|failed|error"
git checkout feature/hybrid-ocr-pipeline
git stash pop
```

- [ ] **Step 5: Final commit if any last fixes were made**

```bash
git add -p  # review any remaining changes
git commit -m "fix: <specific issue found during smoke test>"
```

---

## Self-Review

**Spec coverage:**
- [x] Stage 1 OCR: `_ocr_image()` — Task 2
- [x] Stage 2 rules: `_rule_extract()` — Task 3
- [x] Stage 3 VLM semantic: `_SEMANTIC_PROMPT_TEMPLATE` — Task 4
- [x] Three-stage orchestration in `extract_label_fields()` — Task 5
- [x] Producer schema fix (US entity only, no `us_importer` split) — Tasks 4 + 6 + 7
- [x] OCR fallback on failure — Task 5 (try/except around Stage 1)
- [x] pytesseract dependency — Task 1
- [x] tesseract-ocr in Dockerfile — Task 1
- [x] Merge strategy (rules take precedence for structured fields) — Task 5
- [x] Dead code removal — Tasks 6 + 7
- [x] No interface changes (callers unchanged) — confirmed in Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `_rule_extract()` defined in Task 3, called in Task 5 ✓
- `_ocr_image()` defined in Task 2, called in Task 5 ✓
- `_SEMANTIC_PROMPT_TEMPLATE` defined in Task 4, used in Task 5 ✓
- `_postprocess()` modified in Task 6, called in Task 5 ✓
- `LabelFields` fields match `merged` dict keys in Task 5 ✓
