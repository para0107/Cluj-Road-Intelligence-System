"""
backend/models_work.py

Municipality repair workflow: work orders group detections into a repair job
that moves through a fixed status flow:

    open → scheduled → in_progress → repaired → verified
                                   ↘ cancelled

Transition side effects live in backend/routes/workorders.py:
  * → repaired sets completed_at and marks every item's detection
    is_fixed=TRUE with fixed_at=now().
  * → verified is refused (409) while any item detection is "reopened"
    (last_detected after fixed_at — the survey pipeline saw it again).

Created by Base.metadata.create_all at startup; fresh Docker volumes get the
tables from db/init/07_work_orders.sql — keep both in sync.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import (
    Column, String, Integer, Float, Date, DateTime, Text, ForeignKey, Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.database import Base

WO_STATUSES = ("open", "scheduled", "in_progress", "repaired", "verified", "cancelled")


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    city = Column(String(80), nullable=False)
    title = Column(String(120), nullable=False)
    status = Column(String(16), nullable=False, default="open")

    crew_name = Column(String(80))
    scheduled_for = Column(Date)
    due_date = Column(Date)
    cost_estimate_ron = Column(Float)
    cost_actual_ron = Column(Float)
    notes = Column(Text)

    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    completed_at = Column(DateTime(timezone=True))
    verified_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_work_orders_city_status", "city", "status"),
    )

    def __repr__(self) -> str:
        return f"<WorkOrder {self.title!r} city={self.city} status={self.status}>"


class WorkOrderItem(Base):
    __tablename__ = "work_order_items"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())

    work_order_id = Column(
        UUID(as_uuid=True), ForeignKey("work_orders.id", ondelete="CASCADE"), nullable=False
    )
    detection_id = Column(
        UUID(as_uuid=True), ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    # Persisted crew route order (set by the route planner's "Optimize order").
    sort_order = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("work_order_id", "detection_id", name="uq_woi_pair"),
        Index("idx_woi_order", "work_order_id", "sort_order"),
        Index("idx_woi_detection", "detection_id"),
    )

    def __repr__(self) -> str:
        return f"<WorkOrderItem order={self.work_order_id} detection={self.detection_id}>"
