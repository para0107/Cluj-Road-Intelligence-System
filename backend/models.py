"""
backend/models.py

SQLAlchemy ORM models — mirror the DB schema.
Note: some enrichment/weather/solar columns were intentionally dropped
from the detections table (street_name, road_importance, infra_proximity_m,
weather, shadow_geometry_score, etc.).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import Column, String, Float, Integer, SmallInteger, Date, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from backend.database import Base


class Detection(Base):
    __tablename__ = "detections"

    # ── Identity ──────────────────────────────────────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ── Spatial ───────────────────────────────────────────────────────────────
    geom = Column(Geometry("POINT", srid=4326), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # ── RT-DETR outputs ───────────────────────────────────────────────────────
    damage_type = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    frame_path = Column(Text)
    # crop_path = Column(Text)

    # ── SAM segmentation geometry ─────────────────────────────────────────────
    surface_area_cm2 = Column(Float)
    edge_sharpness = Column(Float)
    interior_contrast = Column(Float)
    mask_compactness = Column(Float)

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth_estimate_cm = Column(Float)
    depth_confidence = Column(Float)
    lighting_condition = Column(String(15))

    # ── Severity ──────────────────────────────────────────────────────────────
    severity = Column(SmallInteger)
    severity_confidence = Column(Float)

    # ── Spatial density ───────────────────────────────────────────────────────
    surrounding_density = Column(Integer, default=0)

    # ── Temporal ──────────────────────────────────────────────────────────────
    first_detected = Column(Date, nullable=False)
    last_detected = Column(Date, nullable=False)
    detection_count = Column(Integer, default=1)
    deterioration_rate = Column(Float, default=0.0)

    # ── Priority ──────────────────────────────────────────────────────────────
    priority_score = Column(Float, default=0.0)

    # ── Survey metadata ───────────────────────────────────────────────────────
    survey_date = Column(Date, nullable=False)
    survey_video_file = Column(String(255))

    def __repr__(self) -> str:
        return (
            f"<Detection id={self.id} type={self.damage_type} "
            f"severity={self.severity} lat={self.latitude:.4f} lon={self.longitude:.4f}>"
        )

    def compute_priority_score(self) -> float:
        """
        Priority score without enrichment inputs (road_importance/infra_proximity_m were dropped).

        A simple, stable heuristic:
            priority = severity(1..5) * log(detection_count + 1)

        If severity is missing, treat as 1.
        """
        import math

        severity_score = self.severity or 1
        count_factor = math.log((self.detection_count or 1) + 1)

        self.priority_score = round(severity_score * count_factor, 4)
        return self.priority_score


class SurveyLog(Base):
    __tablename__ = "survey_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    survey_date = Column(Date, nullable=False, unique=True)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    status = Column(String(20), default="pending")
    frames_processed = Column(Integer, default=0)
    detections_found = Column(Integer, default=0)
    new_detections = Column(Integer, default=0)
    updated_detections = Column(Integer, default=0)
    error_message = Column(Text)
    video_files = Column(JSONB)

    def __repr__(self) -> str:
        return (
            f"<SurveyLog date={self.survey_date} status={self.status} "
            f"detections={self.detections_found}>"
        )