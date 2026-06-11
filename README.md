# TTB Label Verification

AI-powered alcohol label verification prototype for the Alcohol and Tobacco Tax and Trade Bureau (TTB). Extracts fields from label images using a local vision model (qwen2.5vl:7b via Ollama) combined with a Tesseract OCR cascade, then compares them against applicant-submitted application data and checks TTB regulatory compliance.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU strongly recommended (qwen2.5vl:7b)
- For GPU support: [NVIDIA Container Toolkit must be installed and configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Hardware Tested On

- Linux kernel 7.0.5-arch
- CPU: i9-14900kf
- GPU: NVIDIA RTX4090
- RAM: 64GB

## Quick Start

```bash
git clone <repo-url>
cd <repo>

# Copy and edit environment file
cp .env.example .env
# Edit .env — set APP_PASSWORD and SESSION_SECRET (see Authentication below)

# Build and start all services — pulls the vision model automatically
docker compose up --build -d
```

Wait for the model-puller service to finish (check with `docker compose logs model-puller`). The api service starts automatically on **http://localhost:8000**.

## Authentication

The web UI is password-protected when `APP_PASSWORD` is set. Leave it empty to disable auth (convenient for local development).

| Variable | Purpose |
|---|---|
| `APP_PASSWORD` | Password for the web UI. Empty = auth disabled. |
| `SESSION_SECRET` | Long random string used to sign session cookies. Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"` |

Sessions last **30 days**. After **5 failed login attempts** from the same IP, that IP is locked out for **10 minutes**.

## Usage

### Single Label Verification

1. Upload a label image (front + back, or single image).
2. Click **Extract from Image** to auto-fill fields.
3. Review and fill in incomplete/incorrect application data fields.
4. Click **Verify Label**.
5. Review the per-field pass / fail / warn results and the TTB compliance check.

### Batch Verification

1. Navigate to **Batch**.
2. Upload a folder with multiple labels. Pairs should be labeled `{labelName}Front` and `{labelName}Back`. Individual images can be named anything.
3. Click **Run Batch**.
4. Review or correct label outputs and submit if desired.
5. Download the CSV summary when complete.

### History *(WIP)*

All verifications are saved and viewable under **History**. Each record links to a pre-filled COLA PDF export.

> **Note:** The History page is currently marked work-in-progress.

## TTB Compliance Check

Every verification runs a compliance check against the extracted label data. Required on every alcohol label:

- Brand name
- Class / type
- Alcohol content (ABV)
- Net contents
- Producer name and address (US producer, or US importer for imported products)
- Government warning statement (all-caps "GOVERNMENT WARNING" prefix and body text; bold formatting shown as informational badge only)
- Sulfite declaration (positive or negative) if type is a wine

## Deployment Behind a Reverse Proxy (nginx + Cloudflare)

The included `docker-compose.yml` can be extended with an nginx service for HTTPS termination. Set `APP_PASSWORD` and `SESSION_SECRET` in `.env`, configure your router to forward ports 80/443 to the host, and point your DNS (e.g., Cloudflare-proxied) at your public IP. The application correctly extracts real client IPs from Cloudflare's `CF-Connecting-IP` header.

## Architecture Overview

```
Browser
   │
   ├── GET  /                     → index.html        (single label verify)
   ├── GET  /batch-upload         → batch.html
   ├── GET  /history              → history.html
   │
   ├── POST /extract              → ollama.py          (vision model + Tesseract cascade)
   ├── POST /verify               → comparator.py + compliance.py
   ├── POST /batch/save-group     → batch.py           (save batch results)
   │
   ├── GET  /api/logs/stream      → log_bus.py         (Server-Sent Events)
   └── GET  /api/history/{id}/pdf → pdf_export.py      (COLA form PDF)

Storage: SQLite (data/verifications.db)
Model:   qwen2.5vl:7b via Ollama (GPU-accelerated)
OCR:     Tesseract cascade (orientation correction, upscaling, binarization)
```
