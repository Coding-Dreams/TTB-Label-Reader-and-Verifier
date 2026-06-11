import asyncio
import logging
from datetime import datetime, timezone

_subscribers: list[asyncio.Queue] = []
_dbg = logging.getLogger("debug.trace")


async def emit(message: str) -> None:
    if not _subscribers:
        return
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    entry = f"[{ts}] {message}"
    full_count = 0
    for q in list(_subscribers):
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            full_count += 1
    if full_count:
        _dbg.debug("[log_bus] %d/%d subscriber queues FULL for: %s",
                    full_count, len(_subscribers), message[:80])


def subscribe() -> "asyncio.Queue[str]":
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def unsubscribe(q: "asyncio.Queue[str]") -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass
