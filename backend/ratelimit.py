"""
backend/ratelimit.py

In-process rate limiting shared by every route module. Zero dependencies,
guarded by one lock per limiter, keyed by caller-supplied strings such as
"register|1.2.3.4" or "live_report|<user_id>".

Two usage styles:

  * Budget style (most endpoints): ``limiter.hit(key)`` counts one request and
    raises 429 (with a Retry-After header) once the fixed window is exhausted.
  * Failure style (login): ``limiter.check(key)`` raises while locked,
    ``limiter.fail(key)`` counts a failure and locks at the threshold,
    ``limiter.ok(key)`` clears state on success.

All state is per-process and resets on restart — the same accepted trade-off
as the original login limiter (see docs/SECURITY.md). The API runs a single
uvicorn worker by design (WS fan-out in live_manager.py is per-process);
move this to Redis if the API is ever replicated.
"""

import os
import threading
import time

from fastapi import Depends, HTTPException, Request, status

_PRUNE_THRESHOLD = 10_000


def client_ip(request: Request) -> str:
    # Behind the bundled Nginx, X-Real-IP carries the true client address.
    return request.headers.get("X-Real-IP") or (request.client.host if request.client else "?")


class Limiter:
    """Fixed-window counter with optional lockout.

    ``max_events`` and ``window_s`` can be overridden per limiter through the
    environment: RL_<NAME>_MAX and RL_<NAME>_WINDOW_S (name upper-cased).
    """

    def __init__(self, name: str, max_events: int, window_s: float,
                 lockout_s: float = 0.0, detail: str | None = None):
        env = name.upper().replace("-", "_")
        self.name = name
        self.max_events = int(os.getenv(f"RL_{env}_MAX", str(max_events)))
        self.window_s = float(os.getenv(f"RL_{env}_WINDOW_S", str(window_s)))
        self.lockout_s = float(os.getenv(f"RL_{env}_LOCKOUT_S", str(lockout_s)))
        self.detail = detail or "Too many requests. Slow down and try again."
        self._lock = threading.Lock()
        self._counts: dict[str, tuple[int, float]] = {}
        self._locked_until: dict[str, float] = {}

    # ── internals ────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        # Opportunistic cleanup so the dicts cannot grow without bound.
        if len(self._counts) > _PRUNE_THRESHOLD:
            self._counts = {k: v for k, v in self._counts.items()
                            if now - v[1] <= self.window_s}
        if len(self._locked_until) > _PRUNE_THRESHOLD:
            self._locked_until = {k: t for k, t in self._locked_until.items() if t > now}

    def _raise(self, retry_after_s: float) -> None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            self.detail,
            headers={"Retry-After": str(max(1, int(retry_after_s) + 1))},
        )

    # ── budget style ─────────────────────────────────────────────────────────

    def hit(self, key: str) -> None:
        """Count one event for ``key``; raise 429 when the window budget is spent."""
        now = time.time()
        with self._lock:
            self._prune(now)
            until = self._locked_until.get(key, 0.0)
            if until > now:
                self._raise(until - now)
            if until:
                self._locked_until.pop(key, None)
            count, start = self._counts.get(key, (0, now))
            if now - start > self.window_s:
                count, start = 0, now
            count += 1
            if count > self.max_events:
                if self.lockout_s:
                    self._locked_until[key] = now + self.lockout_s
                    self._counts.pop(key, None)
                    self._raise(self.lockout_s)
                self._counts[key] = (count, start)
                self._raise(start + self.window_s - now)
            self._counts[key] = (count, start)

    # ── failure style (login) ────────────────────────────────────────────────

    def check(self, key: str) -> None:
        """Raise 429 while ``key`` is locked out; otherwise a no-op."""
        now = time.time()
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            if until > now:
                self._raise(until - now)
            if until:
                self._locked_until.pop(key, None)

    def fail(self, key: str) -> None:
        """Count a failed attempt; lock the key once max_events is reached."""
        now = time.time()
        with self._lock:
            self._prune(now)
            count, start = self._counts.get(key, (0, now))
            if now - start > self.window_s:
                count, start = 0, now
            count += 1
            if count >= self.max_events:
                self._locked_until[key] = now + (self.lockout_s or self.window_s)
                self._counts.pop(key, None)
            else:
                self._counts[key] = (count, start)

    def ok(self, key: str) -> None:
        """Clear all state for ``key`` (e.g. after a successful sign-in)."""
        with self._lock:
            self._counts.pop(key, None)
            self._locked_until.pop(key, None)


def rate_limited(limiter: Limiter, by: str = "ip"):
    """FastAPI dependency factory: ``Depends(rate_limited(limiter, by="user"))``.

    ``by="ip"``   — keys on the client address (public endpoints).
    ``by="user"`` — keys on the authenticated user id (adds the auth
                    dependency; FastAPI caches it per request, so routes that
                    also declare get_current_user resolve it only once).
    """
    if by == "ip":
        def _dep_ip(request: Request) -> None:
            limiter.hit(f"{limiter.name}|{client_ip(request)}")
        return _dep_ip

    if by == "user":
        from backend.auth import get_current_user

        def _dep_user(request: Request, user=Depends(get_current_user)) -> None:
            limiter.hit(f"{limiter.name}|{user.id}")
        return _dep_user

    raise ValueError(f"rate_limited: unknown key mode {by!r}")
