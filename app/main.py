import asyncio
import json
import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app.routers import verify, batch
from app.services import auth, log_bus
from app.services.db import init_db, get_verifications, get_verification
from app.services.ollama import _warmup_model
from app.services.pdf_export import generate_filled_cola

app = FastAPI(title="TTB Label Verification")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper — real client IP
# ---------------------------------------------------------------------------
def _real_ip(request: StarletteRequest) -> str:
    """Return the real client IP, preferring proxy-forwarded headers over the
    socket address (which is always the reverse proxy when one is in front)."""
    return (
        request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )


# ---------------------------------------------------------------------------
# Security headers — applied to every response
# ---------------------------------------------------------------------------
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
# Body size pre-check
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
                pass
        return await call_next(request)

# ---------------------------------------------------------------------------
# Rate limiting — per real client IP on write endpoints
# ---------------------------------------------------------------------------
_RATE_LIMIT_PATHS = {"/extract", "/verify", "/verify-fields", "/batch/save-group"}
_RATE_WINDOW = 60   # seconds
_RATE_MAX = 600     # requests per window

class _RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    async def dispatch(self, request: StarletteRequest, call_next):
        if request.url.path in _RATE_LIMIT_PATHS:
            ip = _real_ip(request)
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

# ---------------------------------------------------------------------------
# Authentication — session cookie gate
# ---------------------------------------------------------------------------
class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if not auth.auth_enabled():
            return await call_next(request)
        path = request.url.path
        if path == "/login" or path.startswith("/static/"):
            return await call_next(request)
        token = request.cookies.get(auth.COOKIE_NAME)
        if not auth.verify_session_token(token):
            if path.startswith("/api/") or request.method not in ("GET", "HEAD"):
                return Response(
                    content='{"detail":"Unauthorized"}',
                    status_code=401,
                    media_type="application/json",
                )
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)


# ---------------------------------------------------------------------------
# CSRF protection — all non-GET/HEAD requests from authenticated users must
# carry a valid CSRF token (except /login which has no session yet).
# Token is checked from the X-CSRF-Token header (fetch requests) or the
# _csrf_token form field (HTML form submissions).
# ---------------------------------------------------------------------------
_CSRF_EXEMPT = {"/login"}

class _CsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if not auth.auth_enabled():
            return await call_next(request)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        if request.url.path in _CSRF_EXEMPT:
            return await call_next(request)

        session_token = request.cookies.get(auth.COOKIE_NAME)
        if not session_token:
            return await call_next(request)  # auth middleware will reject

        # Accept CSRF token from header (JS fetch) or form field (HTML forms)
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            # For form submissions we need to peek at the body; Starlette
            # caches it after first read so downstream handlers still work.
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
                form = await request.form()
                csrf_token = form.get("_csrf_token")

        if not auth.verify_csrf_token(session_token, csrf_token):
            if request.headers.get("accept", "").startswith("text/html"):
                return RedirectResponse("/login", status_code=303)
            return Response(
                content='{"detail":"CSRF validation failed"}',
                status_code=403,
                media_type="application/json",
            )
        return await call_next(request)


# Middleware is applied innermost-first: security headers wrap everything,
# rate limiter is next, body size check is next, CSRF is next, auth gate is
# outermost (runs first on requests).
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_BodySizeLimitMiddleware)
app.add_middleware(_CsrfMiddleware)
app.add_middleware(_AuthMiddleware)


@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(_warmup_model())


app.include_router(verify.router)
app.include_router(batch.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ---------------------------------------------------------------------------
# CSRF token endpoint — JS calls this to get a token for fetch requests
# ---------------------------------------------------------------------------
@app.get("/api/csrf-token")
def get_csrf_token(request: StarletteRequest):
    session_token = request.cookies.get(auth.COOKIE_NAME)
    if not session_token or not auth.verify_session_token(session_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"csrf_token": auth.make_csrf_token(session_token)}


@app.get("/login")
def login_page():
    return FileResponse("app/static/login.html")


@app.post("/login")
async def login(request: StarletteRequest, password: str = Form(...)):
    ip = _real_ip(request)
    if auth.is_locked_out(ip):
        return RedirectResponse("/login?error=locked", status_code=303)
    if not auth.verify_password(password):
        auth.record_failure(ip)
        remaining = auth._MAX_FAILURES - len(auth._failures.get(ip, []))
        await log_bus.emit(f"Failed login attempt from {ip} ({remaining} attempt(s) remaining)")
        return RedirectResponse("/login?error=1", status_code=303)
    auth.reset_failures(ip)
    await log_bus.emit(f"User signed in from {ip}")
    session_token = auth.make_session_token()
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        session_token,
        httponly=True,
        samesite="lax",
        secure=True,
        max_age=86400 * 30,  # 30 days
    )
    return response


@app.post("/logout")
async def logout(request: StarletteRequest):
    token = request.cookies.get(auth.COOKIE_NAME)
    auth.revoke_session_token(token)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME)
    return response


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


@app.get("/api/logs/stream")
async def stream_logs(request: StarletteRequest):
    q = log_bus.subscribe()
    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            log_bus.unsubscribe(q)
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history/{verification_id}/pdf")
def get_cola_pdf(verification_id: int):
    record = get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    extracted = json.loads(record["extracted"])
    try:
        pdf_bytes = generate_filled_cola(extracted)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="COLA template not found")
    filename = f"COLA_{verification_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
