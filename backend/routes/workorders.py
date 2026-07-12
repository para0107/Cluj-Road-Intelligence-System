"""
backend/routes/workorders.py

Municipality repair workflow. A work order groups detections into one repair
job for a crew; the status flow is:

    open → scheduled → in_progress → repaired → verified   (or cancelled)

Side effects enforced here:
  * → repaired: completed_at is stamped and every item's detection becomes
    is_fixed=TRUE with fixed_at=now() (feeds the repair-verification loop).
  * → verified: refused with 409 while any item detection is "reopened"
    (the survey pipeline saw it again after fixed_at); otherwise verified_at
    is stamped.

Scoping: municipality operators only see and touch their own city's orders;
admins may pass ?city= to inspect any city (or omit it for all).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import require_operator
from backend.database import get_db
from backend.models import Detection
from backend.models_auth import ROLE_ADMIN, User
from backend.models_work import WO_STATUSES, WorkOrder, WorkOrderItem
from backend.schemas_work import (
    WorkOrderCreate, WorkOrderDetail, WorkOrderItemRead, WorkOrderItemsEdit,
    WorkOrderListResponse, WorkOrderRead, WorkOrderUpdate,
)

router = APIRouter()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _operator_city(user: User, city_param: str | None) -> str | None:
    """The city filter this operator is allowed to act in.

    Municipality accounts are hard-scoped to their own city. Admins may pass
    ?city= or omit it (None → all cities in list endpoints).
    """
    if user.role == ROLE_ADMIN:
        return city_param
    if not user.city:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Set your city on your profile first.")
    return user.city


def _get_scoped_order(db: Session, order_id: UUID, user: User) -> WorkOrder:
    order = db.get(WorkOrder, order_id)
    if not order:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Work order not found.")
    if user.role != ROLE_ADMIN and order.city != user.city:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "This work order belongs to another city.")
    return order


def _is_reopened(det: Detection) -> bool:
    return bool(
        det.is_fixed and det.fixed_at and det.last_detected
        and det.last_detected > det.fixed_at.date()
    )


def _item_read(item: WorkOrderItem, det: Detection) -> WorkOrderItemRead:
    return WorkOrderItemRead(
        detection_id=det.id,
        sort_order=item.sort_order or 0,
        damage_type=det.damage_type,
        severity=det.severity,
        latitude=det.latitude,
        longitude=det.longitude,
        priority_score=det.priority_score,
        detection_count=det.detection_count,
        last_detected=det.last_detected,
        is_fixed=bool(det.is_fixed),
        fixed_at=det.fixed_at,
        reopened=_is_reopened(det),
        has_evidence=bool(det.crop_path),
    )


def _with_count(db: Session, order: WorkOrder) -> WorkOrderRead:
    count = (
        db.query(func.count(WorkOrderItem.id))
        .filter(WorkOrderItem.work_order_id == order.id)
        .scalar() or 0
    )
    read = WorkOrderRead.model_validate(order)
    read.item_count = count
    return read


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/work-orders", response_model=WorkOrderListResponse)
def list_work_orders(
    status_eq: str | None = Query(None, alias="status"),
    city: str | None = Query(None, max_length=80),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    scope_city = _operator_city(user, city)
    q = db.query(WorkOrder)
    if scope_city:
        q = q.filter(WorkOrder.city == scope_city)
    if status_eq:
        if status_eq not in WO_STATUSES:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown status.")
        q = q.filter(WorkOrder.status == status_eq)
    total = q.count()
    orders = (
        q.order_by(WorkOrder.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # One grouped count query instead of one per order.
    counts = dict(
        db.query(WorkOrderItem.work_order_id, func.count(WorkOrderItem.id))
        .filter(WorkOrderItem.work_order_id.in_([o.id for o in orders]))
        .group_by(WorkOrderItem.work_order_id)
        .all()
    ) if orders else {}

    items = []
    for o in orders:
        read = WorkOrderRead.model_validate(o)
        read.item_count = counts.get(o.id, 0)
        items.append(read)
    return WorkOrderListResponse(total=total, page=page, page_size=page_size, items=items)


@router.post("/work-orders", response_model=WorkOrderDetail, status_code=201)
def create_work_order(
    payload: WorkOrderCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    if not user.city:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Set your city on your profile first.")

    detections = (
        db.query(Detection).filter(Detection.id.in_(payload.detection_ids)).all()
    )
    found = {d.id for d in detections}
    missing = [str(i) for i in payload.detection_ids if i not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"Some detections do not exist: {', '.join(missing[:5])}")

    order = WorkOrder(
        city=user.city,
        title=payload.title,
        crew_name=payload.crew_name,
        scheduled_for=payload.scheduled_for,
        due_date=payload.due_date,
        cost_estimate_ron=payload.cost_estimate_ron,
        notes=payload.notes,
        created_by=user.id,
    )
    db.add(order)
    db.flush()

    by_id = {d.id: d for d in detections}
    for idx, det_id in enumerate(payload.detection_ids):
        db.add(WorkOrderItem(work_order_id=order.id, detection_id=det_id, sort_order=idx))
    db.commit()
    db.refresh(order)

    detail = WorkOrderDetail.model_validate(order)
    detail.item_count = len(payload.detection_ids)
    detail.items = [
        _item_read(WorkOrderItem(sort_order=i, detection_id=det_id), by_id[det_id])
        for i, det_id in enumerate(payload.detection_ids)
    ]
    return detail


@router.get("/work-orders/{order_id}", response_model=WorkOrderDetail)
def get_work_order(
    order_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    order = _get_scoped_order(db, order_id, user)
    rows = (
        db.query(WorkOrderItem, Detection)
        .join(Detection, Detection.id == WorkOrderItem.detection_id)
        .filter(WorkOrderItem.work_order_id == order.id)
        .order_by(WorkOrderItem.sort_order.asc())
        .all()
    )
    detail = WorkOrderDetail.model_validate(order)
    detail.item_count = len(rows)
    detail.items = [_item_read(item, det) for item, det in rows]
    return detail


@router.patch("/work-orders/{order_id}", response_model=WorkOrderDetail)
def update_work_order(
    order_id: UUID,
    payload: WorkOrderUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    order = _get_scoped_order(db, order_id, user)

    for field in ("title", "crew_name", "scheduled_for", "due_date",
                  "cost_estimate_ron", "cost_actual_ron", "notes"):
        value = getattr(payload, field)
        if value is not None:
            setattr(order, field, value)

    if payload.item_order is not None:
        positions = {det_id: idx for idx, det_id in enumerate(payload.item_order)}
        items = (
            db.query(WorkOrderItem)
            .filter(WorkOrderItem.work_order_id == order.id)
            .all()
        )
        for item in items:
            if item.detection_id in positions:
                item.sort_order = positions[item.detection_id]

    if payload.status and payload.status != order.status:
        _apply_status_transition(db, order, payload.status)

    db.commit()
    db.refresh(order)
    return get_work_order(order.id, db=db, user=user)


def _apply_status_transition(db: Session, order: WorkOrder, new_status: str) -> None:
    rows = (
        db.query(WorkOrderItem, Detection)
        .join(Detection, Detection.id == WorkOrderItem.detection_id)
        .filter(WorkOrderItem.work_order_id == order.id)
        .all()
    )

    if new_status == "repaired":
        now = _now()
        order.completed_at = now
        for _item, det in rows:
            det.is_fixed = True
            det.fixed_at = now

    elif new_status == "verified":
        reopened = [str(det.id) for _item, det in rows if _is_reopened(det)]
        if reopened:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {
                    "message": "Some repairs were detected again after fixing. "
                               "Inspect them before verifying.",
                    "reopened_detection_ids": reopened,
                },
            )
        order.verified_at = _now()

    order.status = new_status


@router.post("/work-orders/{order_id}/items", response_model=WorkOrderDetail)
def edit_work_order_items(
    order_id: UUID,
    payload: WorkOrderItemsEdit,
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    order = _get_scoped_order(db, order_id, user)

    if payload.remove_ids:
        db.query(WorkOrderItem).filter(
            WorkOrderItem.work_order_id == order.id,
            WorkOrderItem.detection_id.in_(payload.remove_ids),
        ).delete(synchronize_session=False)

    if payload.add_ids:
        existing = {
            row[0] for row in db.query(WorkOrderItem.detection_id)
            .filter(WorkOrderItem.work_order_id == order.id).all()
        }
        valid = {
            row[0] for row in db.query(Detection.id)
            .filter(Detection.id.in_(payload.add_ids)).all()
        }
        next_order = (
            db.query(func.coalesce(func.max(WorkOrderItem.sort_order), -1))
            .filter(WorkOrderItem.work_order_id == order.id)
            .scalar() or 0
        ) + 1
        for det_id in payload.add_ids:
            if det_id in valid and det_id not in existing:
                db.add(WorkOrderItem(
                    work_order_id=order.id, detection_id=det_id, sort_order=next_order,
                ))
                next_order += 1

    db.commit()
    return get_work_order(order.id, db=db, user=user)


@router.delete("/work-orders/{order_id}")
def delete_work_order(
    order_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_operator),
):
    order = _get_scoped_order(db, order_id, user)
    db.delete(order)   # items cascade via FK
    db.commit()
    return {"deleted": True, "id": str(order_id)}
