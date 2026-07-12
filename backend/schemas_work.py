"""
backend/schemas_work.py

Pydantic schemas for the municipality work-order workflow.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.models_work import WO_STATUSES

STATUS_PATTERN = "^(" + "|".join(WO_STATUSES) + ")$"


class WorkOrderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    detection_ids: List[UUID] = Field(..., min_length=1, max_length=200)
    crew_name: Optional[str] = Field(None, max_length=80)
    scheduled_for: Optional[date] = None
    due_date: Optional[date] = None
    cost_estimate_ron: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=2000)


class WorkOrderUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=120)
    status: Optional[str] = Field(None, pattern=STATUS_PATTERN)
    crew_name: Optional[str] = Field(None, max_length=80)
    scheduled_for: Optional[date] = None
    due_date: Optional[date] = None
    cost_estimate_ron: Optional[float] = Field(None, ge=0)
    cost_actual_ron: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=2000)
    # Full route order as detection ids; persists the planner's result.
    item_order: Optional[List[UUID]] = None


class WorkOrderItemsEdit(BaseModel):
    add_ids: List[UUID] = Field(default_factory=list, max_length=200)
    remove_ids: List[UUID] = Field(default_factory=list, max_length=200)


class WorkOrderItemRead(BaseModel):
    detection_id: UUID
    sort_order: int
    damage_type: str
    severity: Optional[int] = None
    latitude: float
    longitude: float
    priority_score: Optional[float] = None
    detection_count: Optional[int] = None
    last_detected: Optional[date] = None
    is_fixed: bool = False
    fixed_at: Optional[datetime] = None
    reopened: bool = False
    has_evidence: bool = False


class WorkOrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    city: str
    title: str
    status: str
    crew_name: Optional[str] = None
    scheduled_for: Optional[date] = None
    due_date: Optional[date] = None
    cost_estimate_ron: Optional[float] = None
    cost_actual_ron: Optional[float] = None
    notes: Optional[str] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    item_count: int = 0


class WorkOrderDetail(WorkOrderRead):
    items: List[WorkOrderItemRead] = Field(default_factory=list)


class WorkOrderListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[WorkOrderRead]
