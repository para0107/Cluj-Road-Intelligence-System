"""
backend/live_manager.py

WebSocket fan-out for Live mode.

A single in-process ConnectionManager broadcasts every event mutation to all
connected clients, so a hazard reported by one vehicle appears on everyone
else's map within milliseconds — no polling.

Threading model
---------------
REST handlers in routes/live.py are *sync* (`def`), so FastAPI runs them in a
worker thread; the WebSocket loop runs on the main asyncio event loop. To
bridge the two safely, `capture_loop()` stores the running loop at startup and
`broadcast_from_thread()` schedules the coroutine with
`asyncio.run_coroutine_threadsafe`. Never call `asyncio.run()` from handlers.

Scaling beyond one process
--------------------------
This manager is per-process by design. To scale horizontally, replace the
in-memory `_connections` fan-out with a Redis (or NATS) pub/sub channel:
publish mutations in `broadcast()`, and have every instance's WS loop
subscribe and relay. The REST API is already stateless (all state in
PostGIS), so nothing else changes. Clients that cannot hold a WebSocket
fall back to polling GET /api/live/events every few seconds.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Set

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def capture_loop(self) -> None:
        """Call once from an async startup hook so threads can reach the loop."""
        self._loop = asyncio.get_running_loop()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("Live WS connected ({} total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("Live WS disconnected ({} total)", len(self._connections))

    @property
    def client_count(self) -> int:
        return len(self._connections)

    # ── Broadcast ────────────────────────────────────────────────────────────

    async def broadcast(self, message: dict) -> None:
        """
        Send a JSON message to every connected client; drop dead sockets.
        Sends run concurrently so one slow phone on a bad connection cannot
        delay the other thousand clients.
        """
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        async with self._lock:
            targets = list(self._connections)
        results = await asyncio.gather(
            *(ws.send_text(payload) for ws in targets),
            return_exceptions=True,
        )
        dead = [ws for ws, res in zip(targets, results) if isinstance(res, Exception)]
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    def broadcast_from_thread(self, message: dict) -> None:
        """
        Thread-safe fire-and-forget broadcast for sync REST handlers.
        Silently no-ops if the loop was never captured (e.g. unit tests).
        """
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


# Module-level singleton — imported by routes/live.py and main.py
manager = ConnectionManager()
