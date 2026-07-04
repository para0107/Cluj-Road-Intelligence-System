"""
backend/notify.py

Zero-cost e-mail notifications via stdlib smtplib — no SendGrid/Mailgun/SES,
no new dependency, nothing to pay. Works with any free SMTP relay; the two
practical zero-cost options:

  Gmail (recommended)    SMTP_HOST=smtp.gmail.com  SMTP_PORT=587
                         SMTP_USERNAME=you@gmail.com
                         SMTP_PASSWORD=<16-char app password>   (free — create
                         one at myaccount.google.com/apppasswords; requires 2FA)
  Outlook / Yahoo        same pattern with their smtp hosts + app passwords.

Configuration (root .env — all optional):
    SMTP_HOST       e.g. smtp.gmail.com          (unset → e-mail silently off)
    SMTP_PORT       default 587
    SMTP_USERNAME   SMTP login
    SMTP_PASSWORD   SMTP password / app password
    SMTP_FROM       From: address (default: SMTP_USERNAME)
    SMTP_STARTTLS   default true (set false only for port-465 implicit-TLS
                    hosts, which then use SMTP_SSL)

Design rules:
  * Never block or fail a request because of e-mail — every send runs in a
    daemon thread and only logs its outcome.
  * When unconfigured this module is a no-op, so the feature is genuinely
    optional (the project's zero-cost rule stays intact).
"""

from __future__ import annotations

import os
import smtplib
import threading
from email.message import EmailMessage

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "").strip() or SMTP_USERNAME
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").lower() == "true"
_SEND_TIMEOUT_S = 15


def email_enabled() -> bool:
    """True when an SMTP relay is configured (host + a From address)."""
    return bool(SMTP_HOST and SMTP_FROM)


def _deliver(msg: EmailMessage) -> None:
    """Synchronous SMTP delivery. Runs inside the sender thread only."""
    if SMTP_STARTTLS:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=_SEND_TIMEOUT_S) as smtp:
            smtp.starttls()
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=_SEND_TIMEOUT_S) as smtp:
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(msg)


def send_email_async(to_addr: str, subject: str, body: str) -> bool:
    """
    Fire-and-forget e-mail. Returns True if a send was *queued* (SMTP is
    configured), False when e-mail is off. Never raises to the caller.
    """
    if not email_enabled():
        logger.debug("E-mail disabled (SMTP_HOST unset) — skipping '{}'", subject)
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    def _worker() -> None:
        try:
            _deliver(msg)
            logger.info("E-mail sent to {} — '{}'", to_addr, subject)
        except Exception as exc:  # noqa: BLE001 — e-mail must never break a request
            logger.warning("E-mail to {} failed ({}): {}", to_addr, subject, exc)

    threading.Thread(target=_worker, name="rids-email", daemon=True).start()
    return True


def send_welcome_email(to_addr: str, username: str, role: str, city: str | None = None) -> bool:
    """Notification for a freshly registered (local, e-mail based) account."""
    role_line = (
        f"Account type: Municipality operator{f' — {city}' if city else ''}\n"
        "You can resolve live hazards, mark detections repaired, and manage\n"
        "your city's repair queue from the Priority and Live pages."
        if role == "municipality"
        else "Account type: Citizen\n"
             "You can upload survey videos, report live hazards, and confirm\n"
             "or dispute what other drivers see."
    )
    body = (
        f"Hi {username},\n"
        "\n"
        "Your RIDS (Road Infrastructure Detection System) account was created\n"
        "successfully with this e-mail address.\n"
        "\n"
        f"{role_line}\n"
        "\n"
        "If you did not create this account, reply to this e-mail so the\n"
        "administrator can remove it.\n"
        "\n"
        "— RIDS · Babeș-Bolyai University\n"
    )
    return send_email_async(to_addr, "Welcome to RIDS — account created", body)
