import asyncio
import json
import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app.routers import verify, batch
from app.services.db import init_db, get_verifications, get_verification
from app.services.ollama import _warmup_model
from app.services.pdf_export import generate_filled_cola

app = FastAPI(title="TTB Label Verification")

# ---------------------------------------------------------------------------
# Security headers — applied to every response
# ---------------------------------------------------------------------------
class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

# ---------------------------------------------------------------------------
# Rate limiting — 20 requests per minute per IP on write endpoints
# ---------------------------------------------------------------------------
_RATE_LIMIT_PATHS = {"/extract", "/verify", "/verify-fields", "/batch/save-group"}
_RATE_WINDOW = 60   # seconds
_RATE_MAX = 20      # requests per window

class _RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    async def dispatch(self, request: StarletteRequest, call_next):
        if request.url.path in _RATE_LIMIT_PATHS:
            ip = (request.client.host if request.client else "unknown")
            now = time.monotonic()
            with self._lock:
                stamps = self._counts[ip]
                self._counts[ip] = [t for t in stamps if now - t < _RATE_WINDOW]
                if len(self._counts[ip]) >= _RATE_MAX:
                    return Response(
                        content='{"detail":"Rate limit exceeded — please wait before trying again"}',
                        status_code=429,
                        media_type="application/json",
                    )
                self._counts[ip].append(now)
        return await call_next(request)

app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_RateLimitMiddleware)

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
