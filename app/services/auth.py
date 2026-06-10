import hmac
import os
import secrets
import time
from collections import defaultdict
from threading import Lock

_PASSWORD = os.getenv("APP_PASSWORD", "")
_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
_TOKEN_PAYLOAD = b"authenticated"
COOKIE_NAME = "session"

# Brute-force protection: lock out an IP for _LOCKOUT_WINDOW seconds after
# _MAX_FAILURES failed login attempts within that same window.
_MAX_FAILURES = 5
_LOCKOUT_WINDOW = 600  # 10 minutes

_failures: dict[str, list[float]] = defaultdict(list)
_failures_lock = Lock()


def auth_enabled() -> bool:
    return bool(_PASSWORD)


def verify_password(candidate: str) -> bool:
    return hmac.compare_digest(candidate.encode(), _PASSWORD.encode())


def make_session_token() -> str:
    return hmac.new(_SECRET.encode(), _TOKEN_PAYLOAD, digestmod="sha256").hexdigest()


def verify_session_token(token: str | None) -> bool:
    if not token:
        return False
    return hmac.compare_digest(token, make_session_token())


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
