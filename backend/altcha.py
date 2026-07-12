"""
backend/altcha.py

Self-hosted ALTCHA proof-of-work captcha — stdlib only, no third-party service,
works on networks that block everything except HTTPS to our own host.

Protocol (matches the open-source ALTCHA widget, https://altcha.org):
  1. GET /auth/captcha/challenge returns
         {algorithm: "SHA-256", challenge, maxnumber, salt, signature}
     where challenge = sha256(salt + secret_number) and
     signature = HMAC-SHA256(key, challenge).
  2. The widget brute-forces secret_number in [0, maxnumber] on the visitor's
     device (a few hundred ms — invisible to humans, a real cost to bot farms)
     and submits base64({algorithm, challenge, number, salt, signature, took}).
  3. verify() recomputes the hash, checks the HMAC (constant-time), the expiry
     and minimum solve time embedded in the salt, and rejects replays.

The HMAC key derives from JWT_SECRET, so no extra secret needs configuring.
Replay state is in-process (single worker by design — same trade-off as the
login limiter; see docs/SECURITY.md).
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.auth import JWT_SECRET

CAPTCHA_ENABLED = os.getenv("CAPTCHA_ENABLED", "false").strip().lower() in ("1", "true", "yes")

_MAX_NUMBER = int(os.getenv("CAPTCHA_MAX_NUMBER", "100000"))
_TTL_S = int(os.getenv("CAPTCHA_TTL_S", "600"))
_MIN_SOLVE_S = float(os.getenv("CAPTCHA_MIN_SOLVE_S", "2.0"))

_SIG_KEY = hashlib.sha256(b"altcha:" + JWT_SECRET.encode("utf-8")).digest()

_seen_lock = threading.Lock()
_seen: dict[str, float] = {}   # challenge hex → expiry unix ts


def _signature(challenge_hex: str) -> str:
    return hmac.new(_SIG_KEY, challenge_hex.encode("utf-8"), hashlib.sha256).hexdigest()


def create_challenge() -> dict:
    """A fresh widget challenge; the salt carries expiry + issue time."""
    now = int(time.time())
    secret_number = secrets.randbelow(_MAX_NUMBER)
    salt = f"{secrets.token_hex(12)}?expires={now + _TTL_S}&t={now}"
    challenge = hashlib.sha256(f"{salt}{secret_number}".encode("utf-8")).hexdigest()
    return {
        "algorithm": "SHA-256",
        "challenge": challenge,
        "maxnumber": _MAX_NUMBER,
        "salt": salt,
        "signature": _signature(challenge),
    }


def _salt_params(salt: str) -> dict:
    if "?" not in salt:
        return {}
    query = salt.split("?", 1)[1]
    out = {}
    for pair in query.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def verify(payload_b64: str) -> bool:
    """Validate a widget solution. Any malformed input is simply False."""
    try:
        data = json.loads(base64.b64decode(payload_b64, validate=True))
        algorithm = data["algorithm"]
        challenge = str(data["challenge"])
        number = int(data["number"])
        salt = str(data["salt"])
        signature = str(data["signature"])
    except Exception:
        return False

    if algorithm != "SHA-256" or not (0 <= number <= _MAX_NUMBER):
        return False

    # 1. The solution actually hashes to the challenge.
    expected = hashlib.sha256(f"{salt}{number}".encode("utf-8")).hexdigest()
    if not hmac.compare_digest(expected, challenge):
        return False

    # 2. The challenge is one WE issued (constant-time HMAC check).
    if not hmac.compare_digest(_signature(challenge), signature):
        return False

    # 3. Not expired, and not solved suspiciously fast (headless bots solve
    #    instantly; the widget takes human-page-load time).
    params = _salt_params(salt)
    now = time.time()
    try:
        if float(params.get("expires", 0)) < now:
            return False
        if now - float(params.get("t", 0)) < _MIN_SOLVE_S:
            return False
    except ValueError:
        return False

    # 4. Single use: reject replays until the challenge expires on its own.
    with _seen_lock:
        for key in [k for k, exp in _seen.items() if exp < now]:
            _seen.pop(key, None)
        if challenge in _seen:
            return False
        _seen[challenge] = float(params.get("expires", now + _TTL_S))
    return True
