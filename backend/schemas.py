"""
backend/schemas.py

Pydantic v2 schemas for API request/response validation.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────

class DetectionBase(BaseModel):
    latitude:               float
    longitude:              float
    damage_type:            str
    confidence:             float
    severity:               Optional[int]   = None
    severity_confidence:    Optional[float] = None
    surface_area_cm2:       Optional[float] = None
    depth_estimate_cm:      Optional[float] = None
    depth_confidence:       Optional[float] = None
    edge_sharpness:         Optional[float] = None
    interior_contrast:      Optional[float] = None
    mask_compactness:       Optional[float] = None
    shadow_geometry_score:  Optional[float] = None
    lighting_condition:     Optional[str]   = None
    street_name:            Optional[str]   = None
    road_importance:        Optional[int]   = None
    infra_proximity_m:      Optional[float] = None
    weather:                Optional[Dict[str, Any]] = None
    surrounding_density:    Optional[int]   = None
    first_detected:         Optional[date]  = None
    last_detected:          Optional[date]  = None
    detection_count:        int             = 1
    deterioration_rate:     float           = 0.0
    priority_score:         float           = 0.0
    survey_date:            Optional[date]  = None
    survey_video_file:      Optional[str]   = None
    frame_path:             Optional[str]   = None
    crop_path:              Optional[str]   = None


# ─────────────────────────────────────────────
# Read schema (returned by API)
# ─────────────────────────────────────────────

class DetectionRead(DetectionBase):
    id:         UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────
# List response with pagination
# ─────────────────────────────────────────────

class DetectionListResponse(BaseModel):
    total:      int
    page:       int
    page_size:  int
    items:      List[DetectionRead]


# ─────────────────────────────────────────────
# Stats response
# ─────────────────────────────────────────────

class DamageTypeCount(BaseModel):
    damage_type: str
    count:       int

class SeverityCount(BaseModel):
    severity: int
    count:    int

class StatsResponse(BaseModel):
    total_detections:       int
    last_survey_date:       Optional[date]
    detections_today:       int
    damage_type_breakdown:  List[DamageTypeCount]
    severity_breakdown:     List[SeverityCount]
    avg_severity:           Optional[float]
    most_damaged_street:    Optional[str]
    critical_count:         int    # severity >= 4


# ─────────────────────────────────────────────
# Heatmap response
# ─────────────────────────────────────────────

class HeatmapPoint(BaseModel):
    latitude:   float
    longitude:  float
    weight:     float   # severity score used as heat weight

class HeatmapResponse(BaseModel):
    points: List[HeatmapPoint]


# ─────────────────────────────────────────────
# Priority list response
# ─────────────────────────────────────────────

class PriorityItem(BaseModel):
    id:             UUID
    rank:           int
    priority_score: float
    severity:       int
    damage_type:    str
    street_name:    Optional[str]
    latitude:       float
    longitude:      float
    detection_count: int
    last_detected:  Optional[date]
    crop_path:      Optional[str]

class PriorityListResponse(BaseModel):
    items: List[PriorityItem]


# ─────────────────────────────────────────────
# Survey log response
# ─────────────────────────────────────────────

class SurveyLogRead(BaseModel):
    id:                 int
    survey_date:        date
    started_at:         Optional[datetime]
    finished_at:        Optional[datetime]
    status:             str
    frames_processed:   int
    detections_found:   int
    new_detections:     int
    updated_detections: int
    error_message:      Optional[str]
    video_files:        Optional[List[str]]

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────
# Nearby query params (used by /detections/nearby)
# ─────────────────────────────────────────────

class NearbyQuery(BaseModel):
    latitude:   float   = Field(..., ge=-90,  le=90)
    longitude:  float   = Field(..., ge=-180, le=180)
    radius_m:   float   = Field(100, ge=1,   le=5000)
    limit:      int     = Field(20,  ge=1,   le=100)