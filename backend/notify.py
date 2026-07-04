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

NETWORKS THAT BLOCK SMTP (university/corporate egress filters kill ports
25/465/587 entirely — HTTPS is often the only thing allowed out): set
BREVO_API_KEY instead. Brevo (brevo.com) has a free tier (300 e-mails/day,
no card) and its REST API runs over plain HTTPS 443, so it works anywhere a
browser works. SMTP_FROM must then be a sender address you verified in
Brevo. When both are configured, the HTTPS API wins (it is the more
portable transport); SMTP_HOST is ignored.

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

# HTTPS transport for SMTP-blocked networks (free tier, no card): the key
# from brevo.com → Settings → SMTP & API → API keys. Wins over SMTP when set.
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
_BREVO_URL = "https://api.brevo.com/v3/smtp/email"

_SEND_TIMEOUT_S = 15


def _http_email_enabled() -> bool:
    return bool(BREVO_API_KEY and SMTP_FROM)


def email_enabled() -> bool:
    """True when any transport is configured: HTTPS API or SMTP relay."""
    return _http_email_enabled() or bool(SMTP_HOST and SMTP_FROM)


def _deliver_smtp(to_addr: str, subject: str, body: str) -> None:
    """Synchronous SMTP delivery. Runs inside the sender thread only."""
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
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


def _deliver_http(to_addr: str, subject: str, body: str) -> None:
    """
    Synchronous delivery over HTTPS 443 via the Brevo transactional API —
    the transport for networks whose firewalls swallow all SMTP ports.
    """
    import httpx

    resp = httpx.post(
        _BREVO_URL,
        headers={"api-key": BREVO_API_KEY, "content-type": "application/json"},
        json={
            "sender": {"email": SMTP_FROM, "name": "RIDS"},
            "to": [{"email": to_addr}],
            "subject": subject,
            "textContent": body,
        },
        timeout=_SEND_TIMEOUT_S,
    )
    resp.raise_for_status()


def send_email_async(to_addr: str, subject: str, body: str) -> bool:
    """
    Fire-and-forget e-mail. Returns True if a send was *queued* (a transport
    is configured), False when e-mail is off. Never raises to the caller.
    """
    if not email_enabled():
        logger.debug("E-mail disabled (no BREVO_API_KEY / SMTP_HOST) — skipping '{}'", subject)
        return False

    use_http = _http_email_enabled()

    def _worker() -> None:
        try:
            if use_http:
                _deliver_http(to_addr, subject, body)
            else:
                _deliver_smtp(to_addr, subject, body)
            logger.info("E-mail sent to {} via {} — '{}'",
                        to_addr, "brevo-https" if use_http else "smtp", subject)
        except Exception as exc:  # noqa: BLE001 — e-mail must never break a request
            logger.warning("E-mail to {} failed ({}): {}", to_addr, subject, exc)

    threading.Thread(target=_worker, name="rids-email", daemon=True).start()
    return True


def send_verification_email(to_addr: str, username: str, code: str) -> bool:
    """E-mail confirmation code — the account is only created after this."""
    body = (
        f"Hi {username},\n"
        "\n"
        "Use this code to confirm your e-mail address and finish creating your\n"
        "RIDS account:\n"
        "\n"
        f"        {code}\n"
        "\n"
        "The code expires in 30 minutes. If you did not request an account,\n"
        "simply ignore this message — nothing was created.\n"
        "\n"
        "— RIDS · Babeș-Bolyai University\n"
    )
    return send_email_async(to_addr, f"RIDS e-mail verification code: {code}", body)


def send_admin_approval_request(admin_addrs: list[str], username: str,
                                email: str, city: str | None) -> None:
    """Tell every admin a municipality registration awaits their approval."""
    body = (
        "A municipality account registration is awaiting approval:\n"
        "\n"
        f"    Username : {username}\n"
        f"    E-mail   : {email}\n"
        f"    City     : {city or '—'}\n"
        "\n"
        "Review it on the Admin page (Manage accounts → Pending approvals).\n"
        "The account is NOT active until an admin approves it.\n"
        "\n"
        "— RIDS · Babeș-Bolyai University\n"
    )
    for addr in admin_addrs:
        send_email_async(addr, f"RIDS: municipality approval needed — {username}", body)


def send_approval_result_email(to_addr: str, username: str, approved: bool) -> bool:
    """Tell the applicant their municipality registration was decided."""
    if approved:
        body = (
            f"Hi {username},\n"
            "\n"
            "Good news — an administrator approved your municipality account.\n"
            "You can now sign in and manage your city's repairs on the RIDS\n"
            "platform.\n"
            "\n"
            "— RIDS · Babeș-Bolyai University\n"
        )
        subject = "RIDS: your municipality account was approved"
    else:
        body = (
            f"Hi {username},\n"
            "\n"
            "An administrator reviewed and declined your municipality account\n"
            "registration. If you believe this is a mistake, contact the\n"
            "platform administrator.\n"
            "\n"
            "— RIDS · Babeș-Bolyai University\n"
        )
        subject = "RIDS: your municipality registration was declined"
    return send_email_async(to_addr, subject, body)


def send_thankyou_email(to_addr: str, username: str, damage_type: str) -> bool:
    """Thank a user for reporting road damage (throttled by the caller)."""
    pretty = damage_type.replace("_", " ")
    body = (
        f"Hi {username},\n"
        "\n"
        f"Thank you for reporting a {pretty} on the RIDS live map!\n"
        "\n"
        "Your report is now visible to every driver in the network and to the\n"
        "city's repair team. As other devices pass the same spot it will be\n"
        "cross-validated (UNVERIFIED → CONFIRMED → VERIFIED) and prioritised\n"
        "for repair.\n"
        "\n"
        "Every report makes the roads a little safer — keep them coming.\n"
        "\n"
        "— RIDS · Babeș-Bolyai University\n"
    )
    return send_email_async(to_addr, "Thank you for your road-damage report", body)


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
