# TTB Label Verification

AI-powered alcohol label verification prototype for the Alcohol and Tobacco Tax and Trade Bureau (TTB). Extracts fields from label images using a local vision model (glm-ocr via Ollama) and compares them against applicant-submitted application data.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU strongly recommended (qwen2.5vl:3b runs in ~3-5s on RTX 4090; ~15s on RTX 3070 Mobile)
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed and configured

## Quick Start

```bash
git clone <repo-url>
cd <repo>

# Start services in background
docker compose up --build -d

# Pull the vision model (one-time, ~2GB — takes a few minutes)
docker compose exec ollama ollama pull qwen2.5vl:3b
```

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

```csv
image_filename,brand_name,class_type,alcohol_content,net_contents,producer_name_address,country_of_origin,government_warning
label1.jpg,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45%,750 mL,"Old Tom Distillery, Louisville KY",United States,GOVERNMENT WARNING: ...
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

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

- **Latency:** GPU strongly recommended to meet the 5-second target. CPU-only machines will see 30–60s per label with glm-ocr.
- **Decorative fonts:** glm-ocr may miss fields in circular badges, oval formats, or highly stylized typography. The Extract & Review flow lets agents correct bad extractions before verifying.
- **No authentication:** Prototype scope only.
- **Batch job state:** In-memory only — jobs are lost on server restart.
