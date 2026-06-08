import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from app.routers import verify, batch
from app.services.db import init_db, get_verifications, get_verification
from app.services.ollama import _warmup_model
from app.services.pdf_export import generate_filled_cola

app = FastAPI(title="TTB Label Verification")

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(_warmup_model())

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
    return get_verifications()

@app.get("/api/history/{verification_id}")
def get_single(verification_id: int):
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return record


@app.get("/api/history/{verification_id}/pdf")
def get_cola_pdf(verification_id: int):
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    extracted = json.loads(record["extracted"])
    try:
        pdf_bytes = generate_filled_cola(extracted)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    filename = f"COLA_{verification_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
