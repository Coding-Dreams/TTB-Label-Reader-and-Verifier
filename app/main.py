import asyncio
import json
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from starlette.requests import Request as StarletteRequest
from starlette.types import ASGIApp, Receive, Scope, Send

from app.routers import verify, batch
from app.services import auth, log_bus
from app.services.db import init_db, get_verifications, get_verification
from app.services.ollama import _warmup_model
from app.services.pdf_export import generate_filled_cola

app = FastAPI(title="TTB Label Verification")

logger = logging.getLogger(__name__)
_dbg = logging.getLogger("debug.trace")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _real_ip(request: StarletteRequest) -> str:
    """Real client IP for route handlers (uses Request object)."""
    return (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP")
        or (request.client.host if request.client else "unknown")
    )


def _real_ip_from_scope(scope: dict) -> str:
    """Real client IP for pure-ASGI middleware (uses scope headers directly)."""
    headers = dict(scope.get("headers", []))
    return (
        headers.get(b"cf-connecting-ip", b"").decode()
        or headers.get(b"x-forwarded-for", b"").decode().split(",")[0].strip()
        or headers.get(b"x-real-ip", b"").decode()
        or (scope.get("client") or ("unknown", 0))[0]
    )


def _parse_cookies(cookie_header: str) -> dict:
    cookies: dict = {}
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name.strip():
            cookies[name.strip()] = value.strip()
    return cookies


# ---------------------------------------------------------------------------
# Security headers — pure ASGI, safe for SSE streaming
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

_SECURITY_HEADERS = [
    (b"content-security-policy",       _CSP.encode()),
    (b"x-content-type-options",        b"nosniff"),
    (b"x-frame-options",               b"DENY"),
    (b"x-xss-protection",              b"1; mode=block"),
    (b"referrer-policy",               b"strict-origin-when-cross-origin"),
]


class _SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(_SECURITY_HEADERS)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Body size pre-check — pure ASGI
# ---------------------------------------------------------------------------
_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB


class _BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        cl = headers.get(b"content-length")
        if cl:
            try:
                if int(cl) > _MAX_BODY_BYTES:
                    resp = Response(
                        content='{"detail":"Request body too large — maximum 50 MB"}',
                        status_code=413,
                        media_type="application/json",
                    )
                    await resp(scope, receive, send)
                    return
            except ValueError:
                pass
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Rate limiting — pure ASGI
# ---------------------------------------------------------------------------
_RATE_LIMIT_PATHS = {"/extract", "/verify", "/verify-fields", "/batch/save-group"}
_RATE_WINDOW = 60    # seconds
_RATE_MAX    = 600   # requests per window


class _RateLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] in _RATE_LIMIT_PATHS:
            ip  = _real_ip_from_scope(scope)
            now = time.monotonic()
            with self._lock:
                stamps = self._counts[ip]
                self._counts[ip] = [t for t in stamps if now - t < _RATE_WINDOW]
                if len(self._counts[ip]) >= _RATE_MAX:
                    resp = Response(
                        content='{"detail":"Rate limit exceeded — please wait before trying again"}',
                        status_code=429,
                        media_type="application/json",
                    )
                    await resp(scope, receive, send)
                    return
                self._counts[ip].append(now)
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Authentication — pure ASGI
# ---------------------------------------------------------------------------
class _AuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not auth.auth_enabled():
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path == "/login" or path.startswith("/static/"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        cookies = _parse_cookies(headers.get(b"cookie", b"").decode())
        token   = cookies.get(auth.COOKIE_NAME)
        if not auth.verify_session_token(token):
            method = scope["method"]
            if path.startswith("/api/") or method not in ("GET", "HEAD"):
                resp = Response(
                    content='{"detail":"Unauthorized"}',
                    status_code=401,
                    media_type="application/json",
                )
            else:
                resp = RedirectResponse("/login", status_code=303)
            await resp(scope, receive, send)
            return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# CSRF protection — pure ASGI
# For url-encoded forms (e.g. /logout) we buffer the body, extract the token,
# then replay the body so the downstream handler can still read it.
# ---------------------------------------------------------------------------
_CSRF_EXEMPT = {"/login"}


class _CsrfMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not auth.auth_enabled():
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        if method in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        if scope["path"] in _CSRF_EXEMPT:
            await self.app(scope, receive, send)
            return

        headers      = dict(scope.get("headers", []))
        cookies      = _parse_cookies(headers.get(b"cookie", b"").decode())
        session_token = cookies.get(auth.COOKIE_NAME)
        if not session_token:
            # No session — auth middleware will reject; pass through
            await self.app(scope, receive, send)
            return

        csrf_token: str = headers.get(b"x-csrf-token", b"").decode().strip()
        buffered_body: Optional[bytes] = None

        if not csrf_token:
            content_type = headers.get(b"content-type", b"").decode()
            if "application/x-www-form-urlencoded" in content_type:
                chunks: list[bytes] = []
                more = True
                while more:
                    msg  = await receive()
                    chunks.append(msg.get("body", b""))
                    more = msg.get("more_body", False)
                buffered_body = b"".join(chunks)
                from urllib.parse import parse_qs
                form_data  = parse_qs(buffered_body.decode(errors="replace"))
                csrf_token = (form_data.get("_csrf_token") or [""])[0]

        if not auth.verify_csrf_token(session_token, csrf_token):
            accept = headers.get(b"accept", b"").decode()
            if accept.startswith("text/html"):
                resp: Response = RedirectResponse("/login", status_code=303)
            else:
                resp = Response(
                    content='{"detail":"CSRF validation failed"}',
                    status_code=403,
                    media_type="application/json",
                )
            await resp(scope, receive, send)
            return

        if buffered_body is not None:
            # Replay the buffered body so the downstream handler can read it
            replayed = False

            async def replay_receive() -> dict:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": buffered_body, "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, send)
        else:
            await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Request tracing — outermost middleware, logs when the response is actually
# sent back to the client so we can detect if the freeze is in the middleware
# stack vs in the route handler.
# ---------------------------------------------------------------------------
class _RequestTraceMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "?")
        if path not in ("/extract", "/verify", "/verify-fields", "/batch/save-group"):
            await self.app(scope, receive, send)
            return
        t0 = time.monotonic()
        _dbg.debug("[ASGI] %s %s START", scope.get("method", "?"), path)
        response_started = False

        async def traced_send(message: dict) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                status = message.get("status", "?")
                _dbg.debug("[ASGI] %s %s response.start status=%s (%.1fs)",
                            scope.get("method", "?"), path, status, time.monotonic() - t0)
                response_started = True
            elif message["type"] == "http.response.body":
                body_len = len(message.get("body", b""))
                _dbg.debug("[ASGI] %s %s response.body len=%d (%.1fs)",
                            scope.get("method", "?"), path, body_len, time.monotonic() - t0)
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except Exception as exc:
            _dbg.debug("[ASGI] %s %s EXCEPTION: %s (%.1fs, response_started=%s)",
                        scope.get("method", "?"), path, exc, time.monotonic() - t0, response_started)
            raise
        finally:
            _dbg.debug("[ASGI] %s %s END (%.1fs)", scope.get("method", "?"), path, time.monotonic() - t0)


# Middleware is applied innermost-first: security headers wrap everything,
# rate limiter is next, body size check is next, CSRF is next, auth is outermost.
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_BodySizeLimitMiddleware)
app.add_middleware(_CsrfMiddleware)
app.add_middleware(_AuthMiddleware)
app.add_middleware(_RequestTraceMiddleware)


@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(_warmup_model())


app.include_router(verify.router)
app.include_router(batch.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ---------------------------------------------------------------------------
# CSRF token endpoint
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
        max_age=86400 * 30,
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
