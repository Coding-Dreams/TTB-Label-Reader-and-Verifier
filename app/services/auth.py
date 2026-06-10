import hmac
import os
import secrets

_PASSWORD = os.getenv("APP_PASSWORD", "")
_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
_TOKEN_PAYLOAD = b"authenticated"
COOKIE_NAME = "session"


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
