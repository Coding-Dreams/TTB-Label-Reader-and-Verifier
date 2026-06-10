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
# CSP notes:
#   - 'unsafe-inline' for script-src is required because all JS lives in
#     inline <script> blocks. Tailwind CDN is the only external script source.
#   - 'unsafe-inline' for style-src is required because Tailwind injects
#     utility classes as inline <style> at runtime.
#   - blob: in img-src covers URL.createObjectURL() used for image previews.
#   - connect-src 'self' ensures fetch/XHR cannot reach arbitrary hosts even
#     if an XSS payload were injected.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
    "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
    "img-src 'self' blob: data:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

# ---------------------------------------------------------------------------
# Body size pre-check — reject oversized requests before python-multipart
# buffers the entire upload (Content-Length must be present; honest clients
# always send it for multipart uploads). Two images at 20 MB each plus
# multipart overhead → 50 MB ceiling.
# ---------------------------------------------------------------------------
_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB

class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > _MAX_BODY_BYTES:
                    return Response(
                        content='{"detail":"Request body too large — maximum 50 MB"}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                pass  # malformed header — let normal handling deal with it
        return await call_next(request)

# ---------------------------------------------------------------------------
# Rate limiting — 200 requests per minute per IP on write endpoints
# ---------------------------------------------------------------------------
_RATE_LIMIT_PATHS = {"/extract", "/verify", "/verify-fields", "/batch/save-group"}
_RATE_WINDOW = 60   # seconds
_RATE_MAX = 6000     # requests per window (99-image batch = ~200 requests; 600 gives headroom)

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

# Middleware is applied innermost-first: security headers wrap everything,
# rate limiter is next, body size check is outermost (runs first on requests).
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_BodySizeLimitMiddleware)

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
