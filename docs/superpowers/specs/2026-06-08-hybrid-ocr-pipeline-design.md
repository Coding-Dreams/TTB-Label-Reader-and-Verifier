# Hybrid OCR + VLM Label Extraction Pipeline — Design Spec
**Date:** 2026-06-08
**Project:** TTB Take-Home — ML Pipeline Robustification

---

## Problem

The current pipeline uses a single Qwen2.5VL:7b call for all field extraction. This causes two systematic failure categories:

1. **Producer/importer confusion** — The prompt asks for the *foreign* producer in `producer_name_address` and a separate `us_importer`, then postprocessing merges them. On imported wines (~60% of test cases), the model fills `producer_name_address` with the foreign winery and frequently fails to also find the US importer in small-print back-panel text. Result: foreign producer in output where the US importer should be.

2. **Fine-print extraction failures** — Sulfite declarations and government warnings live in 6pt back-panel text. Stitching front+back compresses each panel to ~384px effective width. The VLM must simultaneously read and classify this compressed text, which it does unreliably.

---

## Solution: Three-Stage Hybrid Pipeline

```
Stage 1 — OCR (pytesseract)
  Input:  front image + back image at original resolution (no resize)
  Output: combined_text — all text from both panels

Stage 2 — Rule-Based Extraction
  Input:  combined_text
  Output: alcohol_content, net_contents, government_warning,
          contains_sulfites, class_type (keyword-matched)
  Method: deterministic regex + keyword scan

Stage 3 — VLM Semantic Classification (Qwen2.5VL)
  Input:  front image (encoded) + combined_text in prompt
  Output: brand_name, class_type (refined), producer_name_address (US entity)
  Note:   VLM classifies known text — does not read the image cold
```

Stage 2 fills all structured fields it can handle deterministically. Stage 3 handles only semantically ambiguous fields. Fields already extracted in Stage 2 are passed into the Stage 3 prompt as known values, so the VLM focuses only on what remains.

---

## Schema Fix

`producer_name_address` is redefined as **the US entity** throughout — domestic producer or US importer, never a foreign address. This aligns with the ground truth JSON for all 30 test labels.

The `us_importer` split field is removed from the prompt. The postprocessing merge (`us_importer → producer_name_address`) is removed. The `_extract_importer()` heuristic chain is removed.

The VLM prompt for `producer_name_address` will read:
> "The US BOTTLER, IMPORTER, or DOMESTIC PRODUCER — a company with a United States address. For imported products this is the US importer (e.g. 'IMPORTED BY: ACME SPIRITS, MIAMI, FL'). For domestic products this is the US producer/bottler (e.g. 'BOTTLED BY: TommyRotter Distillery, Buffalo, NY'). Return null if no US entity is present."

---

## Rule-Based Extraction (Stage 2)

| Field | Rule |
|---|---|
| `alcohol_content` | Regex: `\d+\.?\d*\s*%\s*(?:alc\.?/?vol\.?|by\s+vol\.?|proof)?` |
| `net_contents` | Regex: `\d+\.?\d*\s*(?:ml\|l\b\|fl\.?\s*oz\|pint)` with unit normalization |
| `government_warning` | Substring search: `GOVERNMENT WARNING:` in OCR text; return full sentence block |
| `contains_sulfites` | Keyword scan: `sulfite`, `sulfating`, `sulfit` — classify present/absent from context |
| `class_type` | Keyword matching against existing `_CLASS_TYPE_KEYWORDS` list |

These fields are **fully deterministic once OCR text is clean**. Tesseract on full-resolution label scans reliably extracts this text.

---

## VLM Prompt (Stage 3)

The prompt is restructured to:
1. Inject OCR text as context: `"OCR text extracted from this label:\n{combined_text}"`
2. Ask only for semantic fields: `brand_name`, `class_type` (confirm/refine keyword match), `producer_name_address` (US entity), `country_of_origin`
3. Remove all fields that Stage 2 already handled

This makes the VLM call shorter, faster, and more reliable — the model is classifying text it already has, not reading an image cold.

---

## Files Changed

| File | Change |
|---|---|
| `app/services/ollama.py` | Rewrite `extract_label_fields()` internals; add `_ocr_image()`, `_rule_extract()`; fix producer schema |
| `requirements.txt` | Add `pytesseract` |
| `Dockerfile` | Add `tesseract-ocr` system package |

No changes to: comparator, routers, models, UI, test files.

---

## Error Handling

- If Tesseract fails (corrupted/unreadable image): fall back to Stage 3 without OCR context injection (VLM reads the image directly, no combined_text in prompt)
- If VLM returns invalid JSON: existing retry/empty-LabelFields behavior unchanged
- No new failure modes introduced at the API surface

---

## Success Criteria

- Producer/importer field: correct US entity extracted on >90% of test labels
- Sulfites: correctly detected/absent on all labels where text is present on back panel
- Government warning: correctly detected on all labels with the standard warning text
- Overall: integration test suite passes on >85% of labels (up from current ~60%)
