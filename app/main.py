from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import verify, batch
from app.services.db import init_db

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
    from app.services.db import get_verifications
    return get_verifications()

@app.get("/api/history/{verification_id}")
def get_single(verification_id: int):
    from app.services.db import get_verification
    from fastapi import HTTPException
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return record
