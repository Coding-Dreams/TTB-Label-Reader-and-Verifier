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
    "contains_sulfites",
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
                contains_sulfites=row.get("contains_sulfites") or None,
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
