"""
backend/routes/contact.py

POST /contact/sales — the pricing page's contact form. Public, but defended:
ALTCHA proof-of-work (when enabled), a honeypot field, and a per-IP budget.
The inquiry is forwarded to every active admin through the existing
fire-and-forget e-mail transport; the response never reveals whether e-mail
is configured (nothing to probe).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend import altcha as captcha
from backend.database import get_db
from backend.models_auth import ROLE_ADMIN, User
from backend.notify import send_contact_sales_email
from backend.ratelimit import Limiter, client_ip

router = APIRouter()

_contact_limiter = Limiter(
    "contact_sales", max_events=3, window_s=3600.0,
    detail="Too many messages from this address. Try again later.",
)


class ContactSalesRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    organization: str = Field("", max_length=160)
    message: str = Field(..., min_length=10, max_length=4000)
    altcha: str | None = Field(None, max_length=2000)
    website: str | None = Field(None, max_length=200)   # honeypot


@router.post("/contact/sales")
def contact_sales(
    payload: ContactSalesRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    _contact_limiter.hit(f"contact|{client_ip(request)}")

    if captcha.CAPTCHA_ENABLED:
        if payload.website or not payload.altcha or not captcha.verify(payload.altcha):
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Verification failed. Refresh the page and try again.")
    elif payload.website:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Verification failed. Refresh the page and try again.")

    admin_addrs = [
        row[0] for row in (
            db.query(User.email)
            .filter(User.role == ROLE_ADMIN, User.is_active.is_(True))
            .all()
        ) if row[0]
    ]
    send_contact_sales_email(
        admin_addrs,
        name=payload.name.strip(),
        email=str(payload.email),
        organization=payload.organization.strip(),
        message=payload.message.strip(),
    )
    return {"received": True}
