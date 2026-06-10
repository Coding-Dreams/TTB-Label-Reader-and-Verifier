import hmac
import os
import secrets
import time
from collections import defaultdict
from threading import Lock

_PASSWORD = os.getenv("APP_PASSWORD", "")
_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
COOKIE_NAME = "session"

# Session lifetime: 30 days (matches cookie max_age)
_SESSION_MAX_AGE = 86400 * 30

# Active sessions: token_hash -> expiry timestamp
_sessions: dict[str, float] = {}
_sessions_lock = Lock()

# Brute-force protection: lock out an IP for _LOCKOUT_WINDOW seconds after
# _MAX_FAILURES failed login attempts within that same window.
_MAX_FAILURES = 5
_LOCKOUT_WINDOW = 600  # 10 minutes

_failures: dict[str, list[float]] = defaultdict(list)
_failures_lock = Lock()

# CSRF secret — used to generate per-session CSRF tokens
_CSRF_SECRET = os.getenv("SESSION_SECRET") or _SECRET


def auth_enabled() -> bool:
    return bool(_PASSWORD)


def verify_password(candidate: str) -> bool:
    return hmac.compare_digest(candidate.encode(), _PASSWORD.encode())


def _hash_token(token: str) -> str:
    """Hash a session token for storage — we never store the raw token."""
    return hmac.new(_SECRET.encode(), token.encode(), digestmod="sha256").hexdigest()


def make_session_token() -> str:
    """Generate a unique random session token and register it server-side."""
    token = secrets.token_urlsafe(48)
    token_hash = _hash_token(token)
    expiry = time.time() + _SESSION_MAX_AGE
    with _sessions_lock:
        _sessions[token_hash] = expiry
    return token


def verify_session_token(token: str | None) -> bool:
    if not token:
        return False
    token_hash = _hash_token(token)
    with _sessions_lock:
        expiry = _sessions.get(token_hash)
        if expiry is None:
            return False
        if time.time() > expiry:
            del _sessions[token_hash]
            return False
        return True


def revoke_session_token(token: str | None) -> None:
    """Remove a session token so it can no longer be used."""
    if not token:
        return
    token_hash = _hash_token(token)
    with _sessions_lock:
        _sessions.pop(token_hash, None)


def cleanup_expired_sessions() -> None:
    """Remove all expired sessions. Called periodically."""
    now = time.time()
    with _sessions_lock:
        expired = [h for h, exp in _sessions.items() if now > exp]
        for h in expired:
            del _sessions[h]


# ── CSRF ──────────────────────────────────────────────────────────────────

def make_csrf_token(session_token: str) -> str:
    """Derive a CSRF token from the session token. Tied to the session so it
    cannot be reused across sessions."""
    return hmac.new(
        _CSRF_SECRET.encode(), session_token.encode(), digestmod="sha256"
    ).hexdigest()


def verify_csrf_token(session_token: str | None, csrf_token: str | None) -> bool:
    if not session_token or not csrf_token:
        return False
    expected = make_csrf_token(session_token)
    return hmac.compare_digest(csrf_token, expected)


# ── Brute-force lockout ───────────────────────────────────────────────────

def is_locked_out(ip: str) -> bool:
    now = time.monotonic()
    with _failures_lock:
        _failures[ip] = [t for t in _failures[ip] if now - t < _LOCKOUT_WINDOW]
        return len(_failures[ip]) >= _MAX_FAILURES


def record_failure(ip: str) -> None:
    now = time.monotonic()
    with _failures_lock:
        _failures[ip] = [t for t in _failures[ip] if now - t < _LOCKOUT_WINDOW]
        _failures[ip].append(now)


def reset_failures(ip: str) -> None:
    with _failures_lock:
        _failures.pop(ip, None)
