from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import verify, batch
from app.services.db import init_db, get_verifications, get_verification

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
    return get_verifications()

@app.get("/api/history/{verification_id}")
def get_single(verification_id: int):
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return record
