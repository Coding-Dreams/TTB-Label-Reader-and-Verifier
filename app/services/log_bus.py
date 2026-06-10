import asyncio
from datetime import datetime, timezone

_subscribers: list[asyncio.Queue] = []


async def emit(message: str) -> None:
    if not _subscribers:
        return
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    entry = f"[{ts}] {message}"
    for q in list(_subscribers):
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            pass


def subscribe() -> "asyncio.Queue[str]":
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def unsubscribe(q: "asyncio.Queue[str]") -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass
