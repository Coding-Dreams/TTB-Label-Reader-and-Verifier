---
title: Batch Folder Upload — Design Spec
date: 2026-06-09
status: approved
---

# Batch Folder Upload

## Overview

Replace the existing batch page (multi-file picker + CSV) with a folder-based upload that auto-groups label images into front/back pairs and standalones, lets the user review and correct groupings, extracts fields from each group, then lets the user review results and selectively submit to the database.

No CSV input is required. A CSV of extracted values is available for download after processing.

---

## Grouping Algorithm

Runs entirely in JavaScript immediately after the user selects a folder — no server round-trip.

**Steps per file:**

1. Strip the file extension to get the stem (`Label14Back (copy)`)
2. Find the **first** case-insensitive occurrence of `"front"` or `"back"` in the stem
3. Remove that occurrence to produce a **base name**, normalize to lowercase (`label14 (copy)`)
4. Record the **role**: `front` if the first keyword was `"front"`, `back` if `"back"`
5. If neither keyword is found: role = `standalone`, base = full normalized stem

**Grouping:**

- Files are bucketed by normalized base name
- A bucket with exactly one `front` + one `back` → **pair**
- All other cases (two fronts, two backs, one file with no keyword, three or more) → each file becomes a **standalone**

**Ambiguity flag:**

Any file whose stem contains **both** `"front"` and `"back"` (case-insensitive) receives a ⚠️ warning badge in the preview regardless of how it was grouped. The algorithm still assigns it deterministically (first-keyword-wins), but the user is alerted to review it.

**Examples:**

| File | First keyword | Base | Role |
|---|---|---|---|
| `COLA12Front.jpg` | Front | `cola12` | front |
| `COLA12Back.jpg` | Back | `cola12` | back |
| `horribleback.jpg` | back | `horrible` | back |
| `horribleFront.jpg` | Front | `horrible` | front |
| `BackBack.jpg` | Back | `back` | back → standalone (no front match) |
| `FrontFront.jpg` | Front | `front` | front → standalone (no back match) |
| `FrontBack.jpg` | Front | `back` | front ⚠️ (ambiguous name) |
| `BackFront.jpg` | Back | `front` | back ⚠️ (ambiguous name) |
| `Label1.jpg` | — | `label1` | standalone |

Non-image files (PDFs, CSVs, etc.) are silently filtered out before grouping.

---

## UI — Four Phases

### Phase 1: Folder Selection

- The current batch page inputs (multi-file picker + CSV uploader) are replaced with a single `<input type="file" webkitdirectory>` folder picker
- On selection, grouping runs immediately in JS; the user sees the preview with no server call
- A file count summary is shown (`N images found, M pairs detected, K standalones`)

### Phase 2: Grouping Preview

Two collapsible sections:

**Pairs**
- Each row: front thumbnail | back thumbnail | front filename | back filename | Swap button
- Swap button flips front ↔ back assignment for that pair
- ⚠️ badge on any filename containing both keywords

**Standalones**
- Each row: thumbnail | filename | ⚠️ badge if ambiguous

A **Process** button activates at the bottom once the folder is loaded. No files are uploaded to the server until this button is clicked.

### Phase 3: Extraction

- On "Process", the browser calls `POST /extract` sequentially for each group
  - Pairs: `image` = front file, `back_image` = back file
  - Standalones: `image` = the single file, no `back_image`
- A progress bar shows `N / M labels processed`
- Result cards appear as each extraction completes (incremental, not all-at-once)
- Errors (timeout, model unavailable) are shown inline on the card without stopping the rest of the batch

### Phase 4: Results Review

Each label gets a result card showing:
- Front thumbnail (and back thumbnail if a pair)
- Extracted fields table (field name | extracted value)
- TTB compliance status (pass / fail with failing fields listed)
- Individual **Submit** button — saves this label's result to the DB

Top action bar:
- **Submit All** — saves all successfully extracted labels to the DB in one action
- **Download CSV** — generates a CSV client-side from all extraction results

**CSV columns:** `group_name`, `front_filename`, `back_filename`, `brand_name`, `class_type`, `alcohol_content`, `net_contents`, `producer_name_address`, `country_of_origin`, `government_warning`, `contains_sulfites`, `compliance_pass`

---

## Backend Changes

### New endpoint: `POST /batch/save-group`

Saves a single extracted label result to the database. Mirrors what `POST /verify` does after extraction, without re-running extraction.

**Request body (JSON):**
```json
{
  "image_filename": "COLA12Front.jpg",
  "back_image_filename": "COLA12Back.jpg",
  "extracted": { ...LabelFields... },
  "overall_pass": true
}
```

**Behavior:** Calls `save_verification()` with the provided data. Returns `{"ok": true}`.

### No changes to `/extract`

The existing `POST /extract` endpoint already accepts an optional `back_image`. The batch page calls it as-is.

---

## Data Flow

```
[Folder picker] → JS grouping algorithm → [Grouping preview]
      ↓ (user clicks Process)
[Sequential fetch to /extract per group] → [Result cards appear]
      ↓ (user clicks Submit / Submit All)
[POST /batch/save-group per label] → [DB record saved]
      ↓ (Download CSV)
[Client-side CSV generation from in-memory results]
```

---

## Error Handling

- Extraction timeout/error: card shows error state, Submit button disabled for that label; rest of batch continues
- Ambiguous filenames: ⚠️ badge shown, no blocking — user decides
- Empty folder / no images found: friendly message before showing Process button
- `save-group` failure: per-card error message, no silent data loss

---

## Files Changed

| File | Change |
|---|---|
| `app/static/batch.html` | Full rewrite — 4-phase UI |
| `app/routers/batch.py` | Add `POST /batch/save-group`; remove old `POST /batch`, `GET /batch/{job_id}/status`, `GET /batch/{job_id}/download` endpoints (CSV-driven job system is fully replaced) |
| `app/main.py` | No changes expected |
