"""
backend/routes/cities.py

Per-city landmark lookup for the map's fly-to menu.

GET /cities/landmarks?city=Cluj-Napoca[&refresh=true]

Free by construction: landmarks come from OpenStreetMap's Nominatim search
API (no key, no billing) and are cached in the `city_landmarks` table, so a
city is resolved ONCE ever (unless refresh=true). Requests are spaced ≥ 1 s
apart with a descriptive User-Agent per the Nominatim usage policy.

This is UI sugar for navigation only. It is NOT the removed pipeline
"enrichment" stage — it never reads or writes detections, and if the lookup
fails the frontend quietly falls back to its built-in Cluj-Napoca list.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import threading
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_auth import User, CityLandmark
from backend.auth import get_current_user
from backend.schemas_auth import LandmarksResponse, LandmarkRead, CityCenterResponse

router = APIRouter()

_NOMINATIM_URL = os.getenv("NOMINATIM_SEARCH_URL", "https://nominatim.openstreetmap.org/search")
_USER_AGENT = "RDDS-road-intelligence/1.0 (open-source road monitoring)"
_REQUEST_SPACING_S = 1.1        # Nominatim policy: max 1 request/second

# The 1 req/s limit is per service, not per caller — serialize ALL outbound
# Nominatim traffic from this process so two concurrent users can't get the
# free tier banned.
_nominatim_lock = threading.Lock()

# The geocoded city centre is cached in the same table under this kind and is
# excluded from the landmark (fly-to) listings.
_CITY_KIND = "city"

# What counts as an "important zone" — one query per entry, first hit wins.
_LANDMARK_QUERIES = [
    ("central square", "square"),
    ("city hall", "civic"),
    ("train station", "station"),
    ("university", "university"),
    ("stadium", "stadium"),
    ("shopping mall", "mall"),
    ("hospital", "hospital"),
    ("airport", "airport"),
]

# Built-in fallback so the demo city always works, even fully offline.
_FALLBACK_CLUJ = [
    ("Piața Unirii", "square", 46.7694, 23.5899),
    ("Gara CFR", "station", 46.7847, 23.5867),
    ("Cluj Arena", "stadium", 46.7686, 23.5725),
    ("FSEGA", "university", 46.7734, 23.6193),
    ("Iulius Mall", "mall", 46.7735, 23.6320),
    ("Aeroport Intl. Cluj", "airport", 46.7852, 23.6862),
    ("Mănăștur", "district", 46.7568, 23.5567),
    ("Mărăști", "district", 46.7830, 23.6180),
]


def _geocode_city(city: str) -> dict | None:
    """One Nominatim query for the city itself. Returns None on failure."""
    with _nominatim_lock:
        try:
            with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=10) as client:
                resp = client.get(_NOMINATIM_URL, params={
                    "q": city,
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 0,
                })
                resp.raise_for_status()
                rows = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Nominatim city geocode failed for {}: {}", city, exc)
            return None
        finally:
            time.sleep(_REQUEST_SPACING_S)
    if not rows:
        return None
    try:
        return {"latitude": float(rows[0]["lat"]), "longitude": float(rows[0]["lon"])}
    except (KeyError, ValueError):
        return None


def _fetch_from_nominatim(city: str) -> list[dict]:
    """One rate-limited pass over the landmark queries. Returns [] on failure."""
    found: list[dict] = []
    with _nominatim_lock, httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=10) as client:
        for query, kind in _LANDMARK_QUERIES:
            try:
                resp = client.get(_NOMINATIM_URL, params={
                    "q": f"{query}, {city}",
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 0,
                })
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    row = rows[0]
                    name = (row.get("name") or row.get("display_name", "").split(",")[0]).strip()
                    if name:
                        found.append({
                            "name": name[:160],
                            "kind": kind,
                            "latitude": float(row["lat"]),
                            "longitude": float(row["lon"]),
                        })
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning("Nominatim lookup failed for {} in {}: {}", query, city, exc)
            time.sleep(_REQUEST_SPACING_S)

    # De-duplicate identical spots returned by different queries
    unique: dict[tuple, dict] = {}
    for lm in found:
        key = (round(lm["latitude"], 4), round(lm["longitude"], 4))
        unique.setdefault(key, lm)
    return list(unique.values())


@router.get("/cities/landmarks", response_model=LandmarksResponse)
def city_landmarks(
    city: str = Query(..., min_length=2, max_length=80),
    refresh: bool = Query(False, description="Force a fresh Nominatim lookup"),
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    city_norm = city.strip()

    # 1 — cache (one lookup per city, ever)
    if not refresh:
        cached = (
            db.query(CityLandmark)
            .filter(CityLandmark.city.ilike(city_norm), CityLandmark.kind != _CITY_KIND)
            .order_by(CityLandmark.name)
            .all()
        )
        if cached:
            return LandmarksResponse(
                city=city_norm, source="cache",
                items=[LandmarkRead.model_validate(c) for c in cached],
            )

    # 2 — free OSM/Nominatim lookup (rate-limited; first call for a city
    #     takes ~10 s by design, then it is cached forever)
    items = _fetch_from_nominatim(city_norm)
    if items:
        db.query(CityLandmark).filter(
            CityLandmark.city.ilike(city_norm), CityLandmark.kind != _CITY_KIND
        ).delete(synchronize_session=False)
        for lm in items:
            db.add(CityLandmark(city=city_norm, **lm))
        db.commit()
        return LandmarksResponse(
            city=city_norm, source="nominatim",
            items=[LandmarkRead(**lm) for lm in items],
        )

    # 3 — offline fallback (demo city only)
    if city_norm.lower().startswith("cluj"):
        return LandmarksResponse(
            city=city_norm, source="fallback",
            items=[
                LandmarkRead(name=n, kind=k, latitude=la, longitude=lo)
                for n, k, la, lo in _FALLBACK_CLUJ
            ],
        )
    return LandmarksResponse(city=city_norm, source="fallback", items=[])


@router.get("/cities/center", response_model=CityCenterResponse)
def city_center(
    city: str = Query(..., min_length=2, max_length=80),
    refresh: bool = Query(False, description="Force a fresh Nominatim geocode"),
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Geocoded centre of a city, so every map opens on the caller's own city.
    One free Nominatim query per city EVER — the result is cached in
    `city_landmarks` under kind='city' (excluded from the fly-to menu).
    """
    city_norm = city.strip()

    if not refresh:
        cached = (
            db.query(CityLandmark)
            .filter(CityLandmark.city.ilike(city_norm), CityLandmark.kind == _CITY_KIND)
            .first()
        )
        if cached:
            return CityCenterResponse(
                city=city_norm, source="cache",
                latitude=cached.latitude, longitude=cached.longitude,
            )

    hit = _geocode_city(city_norm)
    if hit is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not locate '{city_norm}'. Please check the city name spelling.",
        )

    db.query(CityLandmark).filter(
        CityLandmark.city.ilike(city_norm), CityLandmark.kind == _CITY_KIND
    ).delete(synchronize_session=False)
    db.add(CityLandmark(city=city_norm, name=city_norm, kind=_CITY_KIND, **hit))
    db.commit()

    return CityCenterResponse(city=city_norm, source="nominatim", **hit)
