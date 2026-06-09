# Batch Folder Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CSV-driven batch page with a folder-upload workflow that auto-groups label images into front/back pairs, extracts fields from each group, and lets the user review and submit results to the database.

**Architecture:** All grouping logic runs in the browser after folder selection with zero server round-trips. On "Process", the browser calls the existing `POST /extract` endpoint (extended to include compliance) sequentially per group, renders result cards as they arrive, and submits selected results to a new lightweight `POST /batch/save-group` endpoint. CSV export is generated client-side from in-memory extraction results.

**Tech Stack:** FastAPI (backend), Pydantic v2, SQLite via `app.services.db`, vanilla JS + Tailwind CSS (frontend)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tests/test_batch_save.py` | Create | Unit tests for the new save-group endpoint |
| `app/routers/batch.py` | Rewrite | Remove CSV-based endpoints; add `POST /batch/save-group` |
| `app/routers/verify.py` | Modify | Add `compliance` key to `POST /extract` response |
| `app/static/batch.html` | Rewrite | Full 4-phase UI: folder picker → preview → extraction → results |

---

## Task 1: Backend — POST /batch/save-group (TDD)

**Files:**
- Create: `tests/test_batch_save.py`
- Rewrite: `app/routers/batch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_batch_save.py`:

```python
import pytest
import app.services.db as db_module
from app.routers.batch import save_group, SaveGroupRequest


def test_save_group_returns_ok(tmp_db):
    result = save_group(SaveGroupRequest(
        image_filename="COLA12Front.jpg",
        back_image_filename="COLA12Back.jpg",
        extracted={"brand_name": "TEST BRAND"},
        overall_pass=True,
    ))
    assert result == {"ok": True}


def test_save_group_persists_to_db(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={"brand_name": "MY BRAND"},
        overall_pass=False,
    ))
    records = db_module.get_verifications()
    assert len(records) == 1
    assert records[0]["image_filename"] == "label.jpg"
    assert records[0]["overall_pass"] == 0


def test_save_group_with_back_image_stores_combined_filename(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="front.jpg",
        back_image_filename="back.jpg",
        extracted={},
        overall_pass=True,
    ))
    records = db_module.get_verifications()
    assert records[0]["image_filename"] == "front.jpg + back.jpg"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/home/nico/Documents/Personal/Treasury/Take Home"
pytest tests/test_batch_save.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `SaveGroupRequest` and `save_group` don't exist yet.

- [ ] **Step 3: Rewrite app/routers/batch.py**

Replace the entire file with:

```python
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.db import save_verification

router = APIRouter()


class SaveGroupRequest(BaseModel):
    image_filename: str
    back_image_filename: Optional[str] = None
    extracted: dict
    overall_pass: bool


@router.post("/batch/save-group")
def save_group(body: SaveGroupRequest):
    filename = body.image_filename
    if body.back_image_filename:
        filename = f"{body.image_filename} + {body.back_image_filename}"
    save_verification(
        image_filename=filename,
        form_data=body.extracted,
        extracted=body.extracted,
        results={"overall_pass": body.overall_pass},
        overall_pass=body.overall_pass,
        batch_id=str(uuid.uuid4()),
    )
    return {"ok": True}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_batch_save.py -v
```

Expected output:
```
tests/test_batch_save.py::test_save_group_returns_ok PASSED
tests/test_batch_save.py::test_save_group_persists_to_db PASSED
tests/test_batch_save.py::test_save_group_with_back_image_stores_combined_filename PASSED
3 passed
```

- [ ] **Step 5: Run full unit test suite to confirm nothing regressed**

```bash
pytest tests/ -v -m "not integration"
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_batch_save.py app/routers/batch.py
git commit -m "feat: replace batch CSV endpoints with save-group endpoint"
```

---

## Task 2: Add compliance to POST /extract response

**Files:**
- Modify: `app/routers/verify.py` (lines 28–50)

The `POST /extract` endpoint currently returns only `LabelFields`. The batch frontend needs compliance info from the same call to avoid a second round-trip per label. Adding `compliance` to the response is backward-compatible — existing callers ignore unknown keys.

- [ ] **Step 1: Modify the extract endpoint in app/routers/verify.py**

Find this block (around line 40–47):

```python
        fields = await extract_label_fields(tmp, tmp_back, debug_info=debug_info)
        result = fields.model_dump()
        if verbose and debug_info is not None:
            result["_debug"] = debug_info
        return result
```

Replace with:

```python
        fields = await extract_label_fields(tmp, tmp_back, debug_info=debug_info)
        result = fields.model_dump()
        result["compliance"] = check_compliance(result)
        if verbose and debug_info is not None:
            result["_debug"] = debug_info
        return result
```

- [ ] **Step 2: Add the compliance import at the top of verify.py**

The file already imports from `app.services`. Add `check_compliance` to the imports (around line 13):

```python
from app.services.compliance import check_compliance
```

- [ ] **Step 3: Run unit tests to confirm nothing broke**

```bash
pytest tests/ -v -m "not integration"
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/routers/verify.py
git commit -m "feat: include compliance result in /extract response"
```

---

## Task 3: Full batch.html rewrite — all 4 phases

**Files:**
- Rewrite: `app/static/batch.html`

This is the main UI change. The file is replaced entirely. The four phases (folder picker → grouping preview → extraction progress → results review) are all present in the DOM from page load; JavaScript controls which sections are visible via `hidden` class toggling.

- [ ] **Step 1: Rewrite app/static/batch.html with the complete implementation**

Replace the entire file with:

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

  <main class="max-w-4xl mx-auto p-6">
    <h1 class="text-2xl font-bold text-gray-800 mb-2">Batch Verification</h1>
    <p class="text-sm text-gray-500 mb-6">
      Select a folder of label images. Front/back pairs are detected automatically from filenames.
    </p>

    <!-- Phase 1: Folder picker -->
    <div id="phase-upload" class="bg-white rounded-lg shadow p-5">
      <label class="block text-sm font-medium text-gray-700 mb-2">Label Images Folder</label>
      <input type="file" id="folder-input" webkitdirectory multiple
        class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
      <p id="file-summary" class="text-xs text-gray-400 mt-2"></p>
    </div>

    <!-- Phase 2: Grouping preview -->
    <div id="phase-preview" class="hidden mt-6">
      <div class="bg-white rounded-lg shadow p-5 mb-4">
        <h2 class="font-semibold text-gray-700 mb-1">
          Detected Pairs <span id="pairs-count" class="text-gray-400 font-normal text-sm"></span>
        </h2>
        <p class="text-xs text-gray-400 mb-3">
          Use ⇄ to swap front/back if a grouping looks wrong. ⚠️ means the filename contains both "front" and "back" — verify it.
        </p>
        <div id="pairs-list"></div>
      </div>
      <div class="bg-white rounded-lg shadow p-5 mb-4">
        <h2 class="font-semibold text-gray-700 mb-3">
          Standalones <span id="standalones-count" class="text-gray-400 font-normal text-sm"></span>
        </h2>
        <div id="standalones-list"></div>
      </div>
      <button id="process-btn"
        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded">
        Process All Labels
      </button>
    </div>

    <!-- Phase 3+4: Progress + Results -->
    <div id="phase-results" class="hidden mt-6">
      <!-- Progress bar -->
      <div class="bg-white rounded-lg shadow p-4 mb-4">
        <div class="flex justify-between text-sm text-gray-600 mb-2">
          <span id="progress-label">Extracting...</span>
          <span id="progress-count"></span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
          <div id="progress-bar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width:0%"></div>
        </div>
      </div>

      <!-- Action bar -->
      <div class="flex gap-3 mb-4">
        <button id="submit-all-btn" disabled onclick="submitAll()"
          class="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded text-sm">
          Submit All
        </button>
        <button id="download-csv-btn" disabled onclick="downloadCSV()"
          class="bg-gray-100 hover:bg-gray-200 disabled:opacity-50 text-gray-700 font-semibold py-2 px-4 rounded text-sm">
          Download CSV
        </button>
      </div>

      <!-- Result cards appear here -->
      <div id="results-list"></div>
    </div>
  </main>

  <script>
    // ── State ──────────────────────────────────────────────────────────────
    let currentGroups = { pairs: [], standalones: [], imageCount: 0 };
    let extractionResults = [];

    // ── Constants ──────────────────────────────────────────────────────────
    const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff']);
    const LABEL_FIELDS = [
      'brand_name', 'class_type', 'alcohol_content', 'net_contents',
      'producer_name_address', 'country_of_origin', 'government_warning', 'contains_sulfites',
    ];

    // ── Phase visibility ───────────────────────────────────────────────────
    function showPhase(name) {
      document.getElementById('phase-preview').classList.add('hidden');
      document.getElementById('phase-results').classList.add('hidden');
      if (name === 'preview') document.getElementById('phase-preview').classList.remove('hidden');
      if (name === 'results') document.getElementById('phase-results').classList.remove('hidden');
    }

    // ── Phase 1: Folder picker ─────────────────────────────────────────────
    document.getElementById('folder-input').addEventListener('change', (e) => {
      currentGroups = groupFiles(e.target.files);
      const { pairs, standalones, imageCount } = currentGroups;
      document.getElementById('file-summary').textContent =
        `${imageCount} image${imageCount !== 1 ? 's' : ''} found — ` +
        `${pairs.length} pair${pairs.length !== 1 ? 's' : ''}, ` +
        `${standalones.length} standalone${standalones.length !== 1 ? 's' : ''}`;
      renderPreview(currentGroups);
      showPhase('preview');
    });

    // ── Grouping algorithm ─────────────────────────────────────────────────
    function groupFiles(fileList) {
      const FRONT_RE = /front/i;
      const BACK_RE  = /back/i;

      const images = Array.from(fileList).filter(f => {
        const ext = f.name.split('.').pop().toLowerCase();
        return IMAGE_EXTS.has(ext);
      });

      function parseFile(file) {
        const stem     = file.name.replace(/\.[^.]+$/, '');
        const hasFront = FRONT_RE.test(stem);
        const hasBack  = BACK_RE.test(stem);
        const ambiguous = hasFront && hasBack;

        const frontIdx = stem.search(FRONT_RE);
        const backIdx  = stem.search(BACK_RE);

        let role = 'standalone';
        let base = stem.toLowerCase();

        if (frontIdx !== -1 && (backIdx === -1 || frontIdx <= backIdx)) {
          role = 'front';
          base = (stem.slice(0, frontIdx) + stem.slice(frontIdx + 5)).toLowerCase().trim();
        } else if (backIdx !== -1) {
          role = 'back';
          base = (stem.slice(0, backIdx) + stem.slice(backIdx + 4)).toLowerCase().trim();
        }

        return { file, stem, base, role, ambiguous };
      }

      const parsed  = images.map(parseFile);
      const buckets = {};

      for (const p of parsed) {
        if (!buckets[p.base]) buckets[p.base] = { fronts: [], backs: [], standalones: [] };
        if      (p.role === 'front') buckets[p.base].fronts.push(p);
        else if (p.role === 'back')  buckets[p.base].backs.push(p);
        else                          buckets[p.base].standalones.push(p);
      }

      const pairs      = [];
      const standalones = [];

      for (const bucket of Object.values(buckets)) {
        if (bucket.fronts.length === 1 && bucket.backs.length === 1 && bucket.standalones.length === 0) {
          pairs.push({ front: bucket.fronts[0], back: bucket.backs[0] });
        } else {
          for (const p of [...bucket.fronts, ...bucket.backs, ...bucket.standalones]) {
            standalones.push(p);
          }
        }
      }

      return { pairs, standalones, imageCount: images.length };
    }

    // ── Phase 2: Preview rendering ─────────────────────────────────────────
    function thumb(file) {
      const url = URL.createObjectURL(file);
      return `<img src="${url}" class="h-20 w-20 object-contain rounded border border-gray-200 bg-gray-50">`;
    }

    function warnBadge(parsed) {
      return parsed.ambiguous
        ? `<span class="text-yellow-500 mr-0.5" title="Filename contains both 'front' and 'back' — verify this grouping">⚠️</span>`
        : '';
    }

    function renderPreview(groups) {
      renderPairs(groups.pairs);
      renderStandalones(groups.standalones);
    }

    function renderPairs(pairs) {
      document.getElementById('pairs-count').textContent = `(${pairs.length})`;
      const list = document.getElementById('pairs-list');
      if (pairs.length === 0) {
        list.innerHTML = '<p class="text-sm text-gray-400 italic">No pairs detected.</p>';
        return;
      }
      list.innerHTML = pairs.map((pair, i) => `
        <div class="flex items-center gap-4 p-3 border rounded-lg bg-gray-50 mb-2">
          <div class="text-center min-w-[90px]">
            <div class="text-xs text-blue-500 font-medium mb-1">Front</div>
            ${thumb(pair.front.file)}
            <div class="text-xs text-gray-500 mt-1 max-w-[90px] truncate" title="${pair.front.file.name}">
              ${warnBadge(pair.front)}${pair.front.file.name}
            </div>
          </div>
          <button onclick="swapPair(${i})"
            class="text-gray-300 hover:text-blue-600 text-2xl leading-none flex-shrink-0"
            title="Swap front and back">⇄</button>
          <div class="text-center min-w-[90px]">
            <div class="text-xs text-gray-400 font-medium mb-1">Back</div>
            ${thumb(pair.back.file)}
            <div class="text-xs text-gray-500 mt-1 max-w-[90px] truncate" title="${pair.back.file.name}">
              ${warnBadge(pair.back)}${pair.back.file.name}
            </div>
          </div>
        </div>
      `).join('');
    }

    function renderStandalones(standalones) {
      document.getElementById('standalones-count').textContent = `(${standalones.length})`;
      const list = document.getElementById('standalones-list');
      if (standalones.length === 0) {
        list.innerHTML = '<p class="text-sm text-gray-400 italic">No standalones.</p>';
        return;
      }
      list.innerHTML = '<div class="flex flex-wrap gap-4">' +
        standalones.map(s => `
          <div class="text-center">
            ${thumb(s.file)}
            <div class="text-xs text-gray-500 mt-1 max-w-[80px] truncate" title="${s.file.name}">
              ${warnBadge(s)}${s.file.name}
            </div>
          </div>
        `).join('') + '</div>';
    }

    function swapPair(i) {
      const pair = currentGroups.pairs[i];
      [pair.front, pair.back] = [pair.back, pair.front];
      renderPairs(currentGroups.pairs);
    }

    // ── Phase 3: Extraction ────────────────────────────────────────────────
    document.getElementById('process-btn').addEventListener('click', () => {
      extractionResults = [];
      document.getElementById('results-list').innerHTML = '';
      document.getElementById('submit-all-btn').disabled = true;
      document.getElementById('download-csv-btn').disabled = true;
      showPhase('results');
      processGroups();
    });

    function updateProgress(done, total) {
      const pct = total > 0 ? Math.round((done / total) * 100) : 0;
      document.getElementById('progress-bar').style.width = pct + '%';
      document.getElementById('progress-count').textContent = `${done} / ${total}`;
      if (done === total) {
        document.getElementById('progress-label').textContent = 'Complete';
      }
    }

    async function processGroups() {
      const allGroups = [
        ...currentGroups.pairs.map(p => ({ isPair: true,  front: p.front, back: p.back })),
        ...currentGroups.standalones.map(s => ({ isPair: false, front: s })),
      ];
      const total = allGroups.length;
      updateProgress(0, total);

      for (let i = 0; i < allGroups.length; i++) {
        const result = await extractGroup(allGroups[i]);
        extractionResults.push(result);
        appendResultCard(result, extractionResults.length - 1);
        updateProgress(i + 1, total);
      }

      document.getElementById('submit-all-btn').disabled = false;
      document.getElementById('download-csv-btn').disabled = false;
    }

    async function extractGroup(group) {
      const fd = new FormData();
      fd.append('image', group.front.file);
      if (group.isPair) fd.append('back_image', group.back.file);

      try {
        const res = await fetch('/extract', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || 'Extraction failed');
        const data = await res.json();
        return {
          frontFilename: group.front.file.name,
          backFilename:  group.isPair ? group.back.file.name : null,
          frontFile:     group.front.file,
          backFile:      group.isPair ? group.back.file : null,
          extracted:     data,
          compliance:    data.compliance,
          submitted:     false,
          error:         null,
        };
      } catch (err) {
        return {
          frontFilename: group.front.file.name,
          backFilename:  group.isPair ? group.back.file.name : null,
          frontFile:     group.front.file,
          backFile:      group.isPair ? group.back.file : null,
          extracted:     null,
          compliance:    null,
          submitted:     false,
          error:         err.message,
        };
      }
    }

    // ── Phase 4: Result cards ──────────────────────────────────────────────
    function appendResultCard(result, idx) {
      const container = document.getElementById('results-list');
      const card = document.createElement('div');
      card.id = `card-${idx}`;
      card.className = 'bg-white rounded-lg shadow p-4 mb-3';

      if (result.error) {
        card.innerHTML = `
          <div class="flex items-center gap-2 mb-1">
            <span class="font-medium text-gray-700 text-sm">${result.frontFilename}</span>
            ${result.backFilename ? `<span class="text-gray-400 text-xs">+ ${result.backFilename}</span>` : ''}
          </div>
          <p class="text-red-500 text-sm">Error: ${result.error}</p>
        `;
        container.appendChild(card);
        return;
      }

      const compliant = result.compliance?.compliant;
      const complianceLabel = compliant
        ? '<span class="text-green-600 font-semibold text-sm">✓ Compliant</span>'
        : '<span class="text-red-600 font-semibold text-sm">✗ Non-compliant</span>';

      const violations = (result.compliance?.violations || [])
        .map(v => `<li class="text-xs text-red-400 mt-0.5">${v.label}: ${v.message}</li>`)
        .join('');

      const frontUrl  = URL.createObjectURL(result.frontFile);
      const backThumb = result.backFile
        ? `<img src="${URL.createObjectURL(result.backFile)}" class="h-20 w-20 object-contain rounded border border-gray-200 bg-gray-50 shrink-0">`
        : '';

      const fieldRows = LABEL_FIELDS.map(f => {
        const val = result.extracted[f];
        const display = (val !== null && val !== undefined && val !== '')
          ? val
          : '<span class="text-gray-300 italic text-xs">not detected</span>';
        return `<tr class="border-b last:border-0">
          <td class="py-1 pr-4 text-gray-500 text-xs w-1/3">${f}</td>
          <td class="py-1 text-gray-800 text-sm">${display}</td>
        </tr>`;
      }).join('');

      card.innerHTML = `
        <div class="flex gap-4 mb-3">
          <img src="${frontUrl}" class="h-20 w-20 object-contain rounded border border-gray-200 bg-gray-50 shrink-0">
          ${backThumb}
          <div class="flex-1 min-w-0">
            <div class="font-medium text-gray-700 text-sm truncate">
              ${result.frontFilename}${result.backFilename ? ' <span class="text-gray-400">+</span> ' + result.backFilename : ''}
            </div>
            <div class="mt-1">${complianceLabel}</div>
            ${violations ? `<ul class="mt-1 list-none">${violations}</ul>` : ''}
          </div>
        </div>
        <details class="mb-3">
          <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 select-none">
            Show extracted fields
          </summary>
          <table class="w-full mt-2"><tbody>${fieldRows}</tbody></table>
        </details>
        <div class="flex items-center gap-3">
          <button onclick="submitSingle(${idx})" id="submit-btn-${idx}"
            class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-1.5 px-4 rounded">
            Submit
          </button>
          <span id="submit-status-${idx}" class="text-xs text-gray-400"></span>
        </div>
      `;
      container.appendChild(card);
    }

    // ── Phase 4: Submit ────────────────────────────────────────────────────
    async function submitSingle(idx) {
      const result = extractionResults[idx];
      if (result.submitted || result.error) return;

      const btn = document.getElementById(`submit-btn-${idx}`);
      btn.disabled = true;
      btn.textContent = 'Saving...';

      try {
        const resp = await fetch('/batch/save-group', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_filename:      result.frontFilename,
            back_image_filename: result.backFilename || null,
            extracted:           result.extracted,
            overall_pass:        result.compliance?.compliant ?? false,
          }),
        });
        if (!resp.ok) throw new Error('Save failed');
        result.submitted = true;
        btn.textContent  = '✓ Saved';
        btn.className    = 'bg-green-500 text-white text-sm font-semibold py-1.5 px-4 rounded cursor-default';
        document.getElementById(`submit-status-${idx}`).textContent = 'Saved to database';
      } catch (err) {
        btn.disabled    = false;
        btn.textContent = 'Submit';
        document.getElementById(`submit-status-${idx}`).textContent = 'Error: ' + err.message;
      }
    }

    async function submitAll() {
      const btn = document.getElementById('submit-all-btn');
      btn.disabled = true;
      const pending = extractionResults
        .map((r, i) => i)
        .filter(i => !extractionResults[i].submitted && !extractionResults[i].error);
      for (const i of pending) {
        await submitSingle(i);
      }
      btn.textContent = 'All Submitted';
    }

    // ── Phase 4: CSV download ──────────────────────────────────────────────
    function downloadCSV() {
      const escape = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
      const headers = ['front_filename', 'back_filename', 'compliance_pass', ...LABEL_FIELDS];

      const rows = extractionResults.map(r => {
        if (r.error) {
          return [escape(r.frontFilename), escape(r.backFilename || ''), 'error',
                  ...LABEL_FIELDS.map(() => '""')].join(',');
        }
        return [
          escape(r.frontFilename),
          escape(r.backFilename || ''),
          r.compliance?.compliant ? 'true' : 'false',
          ...LABEL_FIELDS.map(f => escape(r.extracted[f])),
        ].join(',');
      });

      const csv  = [headers.join(','), ...rows].join('\n');
      const link = Object.assign(document.createElement('a'), {
        href:     URL.createObjectURL(new Blob([csv], { type: 'text/csv' })),
        download: `batch_${new Date().toISOString().slice(0, 10)}.csv`,
      });
      link.click();
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: Restart the Docker container to pick up the static file change**

```bash
cd "/home/nico/Documents/Personal/Treasury/Take Home"
docker compose restart
```

Wait ~10 seconds for the server to be ready, then navigate to `http://localhost:8000/batch-upload`.

- [ ] **Step 3: Manual test — folder selection and grouping preview**

1. Open `http://localhost:8000/batch-upload`
2. Click the folder picker and select the `batchUploadTest/` folder
3. Verify the summary line shows the correct image count
4. Verify pairs section lists labeled front/back pairs with thumbnails
5. Verify standalones section shows unpaired images
6. Verify ⚠️ badge appears on any filename containing both "front" and "back"
7. Click a ⇄ button — verify the front and back thumbnails and labels swap
8. Click ⇄ again — verify they swap back

- [ ] **Step 4: Manual test — extraction and results**

1. Click "Process All Labels"
2. Verify progress bar advances as each label is extracted
3. Verify result cards appear one by one as extractions complete
4. Verify each card shows thumbnails, compliance status, and violations list
5. Click "Show extracted fields" on a card — verify the fields table expands
6. Verify "Submit All" and "Download CSV" buttons become enabled when processing finishes

- [ ] **Step 5: Manual test — submit and CSV**

1. Click "Submit" on a single card — verify the button turns green and says "✓ Saved"
2. Verify the submitted result appears in `http://localhost:8000/history`
3. Click "Submit All" — verify remaining un-submitted cards all turn green
4. Click "Download CSV" — verify a CSV file downloads containing all labels with their extracted fields

- [ ] **Step 6: Commit**

```bash
git add app/static/batch.html
git commit -m "feat: implement 4-phase batch folder upload UI"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Folder upload via `webkitdirectory` (Phase 1)
- ✅ Grouping algorithm: first-keyword-wins, normalized base, ambiguity flag (Phase 1/2)
- ✅ Grouping preview with pairs and standalones sections (Phase 2)
- ✅ Swap button to correct front/back assignments (Phase 2)
- ✅ ⚠️ badge on ambiguous filenames (Phase 2)
- ✅ Sequential extraction via existing `/extract` endpoint (Phase 3)
- ✅ Progress bar with N/M counter (Phase 3)
- ✅ Result cards appear incrementally as extractions complete (Phase 3)
- ✅ Extraction errors shown inline without stopping the rest of the batch (Phase 3)
- ✅ Thumbnails, extracted fields table, TTB compliance status per card (Phase 4)
- ✅ Per-label Submit button (Phase 4)
- ✅ Submit All button (Phase 4)
- ✅ Download CSV with all extracted values (Phase 4)
- ✅ No immediate DB save — user controls when to submit (Phase 4)
- ✅ `POST /batch/save-group` endpoint (Task 1)
- ✅ Old CSV-based endpoints removed (Task 1 — batch.py rewrite)

**Type consistency:**
- `SaveGroupRequest` defined in Task 1, imported and called correctly in tests
- `groupFiles()` returns `{ pairs, standalones, imageCount }` — all three keys consumed by the folder change handler
- `swapPair(i)` mutates `currentGroups.pairs[i]` in place — `renderPairs` re-reads from `currentGroups.pairs`
- `extractionResults[idx]` indexed by `appendResultCard(result, extractionResults.length - 1)` — the index passed to `submitSingle(idx)` matches this
- `LABEL_FIELDS` array is defined once and reused in both the field table render and CSV export
