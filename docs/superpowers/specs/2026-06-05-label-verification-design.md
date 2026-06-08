# AI-Powered Alcohol Label Verification App — Design Spec
**Date:** 2026-06-05
**Project:** TTB Take-Home Assessment

---

## 1. Problem Statement

The TTB (Alcohol and Tobacco Tax and Trade Bureau) processes ~150,000 label applications per year with 47 agents. The majority of review time is spent on rote data-matching — verifying that what's printed on a label matches what was submitted in the application. This prototype automates that matching using a local vision model, requiring no cloud API dependencies (firewall-safe), and targets a <5 second response time per label.

---

## 2. Goals & Non-Goals

**Goals:**
- Extract label fields from an uploaded image using a local VLM (llava-phi3 via Ollama)
- Compare extracted fields against applicant-submitted form data using deterministic, field-appropriate matching strategies
- Return a per-field pass / fail / warn result with the extracted vs. submitted values shown side-by-side
- Support batch upload (multiple images + CSV of application data)
- Persist verification history in SQLite
- Simple, accessible UI usable by non-technical staff

**Non-Goals:**
- COLA system integration
- Role-based authentication (specialist vs. applicant)
- Real email delivery
- Mobile optimization
- FedRAMP compliance (prototype only)
- Real-time WebSocket progress updates (polling is sufficient)

---

## 3. Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+ + FastAPI |
| Frontend | Plain HTML + Tailwind CSS (served by FastAPI) |
| Database | SQLite (via Python `sqlite3`) |
| AI / VLM | Ollama with `llava-phi3` (~3.8B) |
| Containerization | Docker Compose |
| Testing | pytest |
| Fuzzy matching | `thefuzz` (Levenshtein distance) |

---

## 4. Architecture

Two Docker Compose services:

```
docker-compose
├── ollama    — Ollama service, pulls llava-phi3 on first run
└── api       — FastAPI: serves HTML static files, REST API, SQLite
```

FastAPI serves the static HTML pages directly. It communicates with Ollama via its local REST API at `http://ollama:11434`. SQLite is stored in a Docker volume mounted into the `api` container for persistence across restarts.

### Project Structure

```
/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py                  # FastAPI app, mounts static files, includes routers
│   ├── routers/
│   │   ├── verify.py            # POST /verify — single label verification
│   │   └── batch.py             # POST /batch, GET /batch/{job_id}/status
│   ├── services/
│   │   ├── ollama.py            # VLM call: image → extracted fields JSON
│   │   ├── comparator.py        # Deterministic field comparison logic
│   │   └── db.py                # SQLite init, insert, query helpers
│   ├── models/
│   │   ├── label.py             # Pydantic model: LabelFields
│   │   └── result.py            # Pydantic model: FieldResult, VerificationResult
│   └── static/
│       ├── index.html           # Single label verification UI
│       ├── batch.html           # Batch upload UI
│       └── history.html         # Verification history UI
└── tests/
    ├── test_comparator.py       # Unit tests for all comparison rules
    └── fixtures/                # Sample label images + expected extractions
```

---

## 5. Data Flow

### Single Label Verification

```
User uploads image + fills form fields
        ↓
POST /verify  (multipart: image file + form fields)
        ↓
ollama.py → llava-phi3 (one call, JSON-structured prompt)
        ↓
LabelFields (extracted from image)
        ↓
comparator.py → per-field deterministic checks
        ↓
VerificationResult saved to SQLite
        ↓
Response JSON → UI renders field-by-field results
```

**"Extract from Image" UX flow:** Before the user clicks Verify, they can click "Extract from Image" to pre-fill the form fields with what the VLM reads from the label. The user reviews and corrects the extracted values, then submits for comparison. This reduces manual transcription burden.

### Batch Verification

```
User uploads N images + CSV (one row per label, columns = form fields)
        ↓
POST /batch → FastAPI BackgroundTask, returns job_id
        ↓
Labels processed sequentially (one Ollama call at a time)
        ↓
GET /batch/{job_id}/status → { completed: N, total: M, results: [...] }
        ↓
UI polls every 2s, updates progress bar
        ↓
Results table rendered + "Download CSV" available on completion
```

Sequential processing keeps Ollama from being overwhelmed and keeps per-label latency predictable.

---

## 6. AI Extraction (ollama.py)

One call to `POST http://ollama:11434/api/generate` with the label image (base64-encoded) and a structured prompt instructing llava-phi3 to return only valid JSON containing these fields:

- `brand_name`
- `class_type`
- `alcohol_content`
- `net_contents`
- `producer_name_address`
- `country_of_origin`
- `government_warning`

Fields not visible on the label are returned as `null` (not an automatic fail — shown as "Not detected" in the UI).

On invalid JSON response: retry once with a stricter prompt before returning an extraction error.

---

## 7. Comparison Logic (comparator.py)

Each field has a strategy appropriate to its nature:

| Field | Strategy | Detail |
|---|---|---|
| Brand name | Fuzzy match ≥ 90% | `thefuzz.ratio()` — handles case, punctuation, spacing variations ("STONE'S THROW" vs "Stone's Throw" → pass) |
| Class / type | Fuzzy match ≥ 85% | Slightly looser — handles abbreviations and formatting |
| Alcohol content (ABV) | Numeric normalize | Strip `%`, `Alc./Vol.`, `proof`; compare numeric values with ±0.1% tolerance |
| Net contents | Numeric normalize | Strip `mL`, `L`, `fl oz`; normalize units; compare numeric values |
| Producer name / address | Fuzzy match ≥ 80% | Most flexible — addresses vary in formatting |
| Country of origin | Exact, case-insensitive | |
| Government warning | Exact substring match | Must contain the string `GOVERNMENT WARNING:` in all-caps — checked by code, not the model |

**Result statuses per field:**
- `pass` — match at or above the field's threshold
- `fail` — match below 70%, or exact check failed
- `warn` — fuzzy match ≥70% but below the field's pass threshold (near-miss, flagged for human review)
- `not_detected` — VLM returned `null` for this field

**Overall result:** `pass` if no fields are `fail`. `warn` fields do not block overall pass — they are surfaced in the UI for human review but do not auto-reject the label.

---

## 8. Database Schema

```sql
CREATE TABLE IF NOT EXISTS verifications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_filename TEXT NOT NULL,
    form_data      TEXT NOT NULL,    -- JSON: applicant-submitted fields
    extracted      TEXT NOT NULL,    -- JSON: VLM-extracted fields
    results        TEXT NOT NULL,    -- JSON: per-field FieldResult objects
    overall_pass   BOOLEAN NOT NULL,
    batch_id       TEXT              -- NULL for single; shared UUID for batch runs
);
```

JSON columns keep the schema stable without migrations. `batch_id` groups batch runs for history display.

---

## 9. UI Design

**Shared nav bar:** `Verify` | `Batch` | `History`

### index.html — Single Label Verification
- Left panel: image upload area (drag & drop or click), image preview once uploaded
- Right panel: form fields for all seven application data fields
- "Extract from Image" button: pre-fills fields from VLM, user can edit before submitting
- "Verify" button with loading spinner
- Results section: per-field rows showing extracted value vs. submitted value, status icon (✓ / ✗ / ⚠)

### batch.html — Batch Upload
- Multi-image upload zone
- CSV upload (columns: `image_filename, brand_name, class_type, alcohol_content, net_contents, producer_name_address, country_of_origin, government_warning`)
- "Run Batch" button → progress bar (`X of N complete`)
- Results table: one row per label, overall pass/fail, expandable detail
- "Download CSV" summary report button

### history.html — Verification History
- Table: timestamp, image filename, overall pass/fail, batch or single
- Click row to expand full per-field result detail

---

## 10. Error Handling

| Scenario | Behavior |
|---|---|
| Ollama timeout (> 10s) | Return 504, UI shows "Model took too long — try again" |
| VLM returns invalid JSON | Retry once with stricter prompt; if still invalid, return 422 with details |
| Image unreadable / too small | Return 400, UI shows "Please upload a clearer image" |
| Field not found on label | `null` from VLM → `not_detected` status, not auto-fail |
| Batch CSV malformed | Per-row errors returned; bad rows skipped, processing continues |
| Ollama service unavailable | Return 503, UI shows "Verification service unavailable — is Ollama running?" |

---

## 11. Testing

- `tests/test_comparator.py` — unit tests for every comparison rule:
  - Brand name fuzzy: "STONE'S THROW" vs "Stone's Throw" → `pass`
  - Brand name fuzzy: "STONE'S THROW" vs "STONEY THROW" → `warn` or `fail`
  - ABV normalize: "45% Alc./Vol. (90 Proof)" vs "45%" → `pass`
  - ABV normalize: "40%" vs "45%" → `fail`
  - Government warning exact: "Government Warning:" → `fail`
  - Government warning exact: "GOVERNMENT WARNING:" present → `pass`
  - Net contents: "750 mL" vs "750ml" → `pass`
- Test fixtures: sample extracted JSON + form data pairs with known expected outcomes
- Tests use in-memory SQLite (`:memory:`) — no mocking the DB layer

---

## 12. Known Limitations & Trade-offs

- **llava-phi3 accuracy on decorative fonts:** Smaller models may misread stylized label typography. The "Extract & Review" flow lets users correct bad extractions before verifying. A future upgrade path is `qwen2-vl:7b` for better OCR accuracy (requires GPU).
- **5-second SLA:** Achievable on GPU. On CPU-only machines, llava-phi3 may take 10–20s. Documented in README; hardware requirements stated clearly.
- **Batch is sequential:** Simpler and safer for a prototype. Concurrency can be added later with `asyncio.gather` + a semaphore.
- **No authentication:** Prototype scope only. Production would require role-based auth.
- **SQLite concurrency:** Fine for a prototype with one user. Would need PostgreSQL for multi-user production deployment.
