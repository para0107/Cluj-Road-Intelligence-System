"""
pipeline/enricher.py
--------------------
Stage 6 of the road damage detection inference pipeline.

Responsibilities:
  - Accept frames from Stage 5 (severity_results.json["frames"])
  - For every frame with valid GPS coordinates (latitude, longitude):
      * Nominatim reverse geocoding → street_name, road_class, road_importance
      * OSM Overpass API            → confirmed road_importance, infra_proximity (m)
      * Open-Meteo historical API   → weather at detection timestamp
  - Frames without GPS are forwarded with all enrichment fields set to None.
  - All API credentials and URLs are read from the .env file via python-dotenv.
    No secrets are hardcoded in this file.

Environment variables read (from .env):
    NOMINATIM_USER_AGENT   — required by Nominatim usage policy (max 1 req/s)
    OSM_OVERPASS_URL       — Overpass endpoint
    OPEN_METEO_BASE_URL    — Open-Meteo endpoint

API compliance:
  Nominatim — nominatim.openstreetmap.org
    - Max 1 request per second enforced via delay_nominatim_s (default 1.1 s).
    - User-Agent is read from NOMINATIM_USER_AGENT in .env.
    - Reference: https://operations.osmfoundation.org/policies/nominatim/
  OSM Overpass — overpass-api.de (or OVERPASS_URL from .env)
    - Rate-limited by delay_overpass_s (default 0.5 s).
  Open-Meteo — api.open-meteo.com (or OPEN_METEO_BASE_URL from .env)
    - Free for non-commercial use. No API key required.
    - Reference: https://open-meteo.com/en/terms

Road importance mapping (OSM highway tag → int):
  3 → primary, trunk, motorway and their links
  2 → secondary, tertiary, unclassified
  1 → residential, living_street, service, track, path, footway, cycleway

Infrastructure proximity:
  Haversine distance (metres) from detection GPS to nearest OSM node tagged
  amenity=hospital|fire_station|school within 500 m.
  None if not found.

Usage (module):
    from pipeline.enricher import Enricher, EnricherConfig
    results = Enricher(EnricherConfig()).run(frames)

Usage (CLI):
    python pipeline/enricher.py
        --input   data/validation_nrdd_2024/severity/severity_results.json
        --output  data/validation_nrdd_2024/enriched/
        [--skip_weather] [--skip_overpass] [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env — must happen before reading os.environ
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API configuration — all from .env, never hardcoded
# ---------------------------------------------------------------------------
_NOMINATIM_URL       = "https://nominatim.openstreetmap.org/reverse"
_OVERPASS_URL        = os.environ.get("OSM_OVERPASS_URL",      "https://overpass-api.de/api/interpreter")
_OPENMETEO_URL       = os.environ.get("OPEN_METEO_BASE_URL",   "https://api.open-meteo.com/v1/forecast")
_NOMINATIM_UA        = os.environ.get(
    "NOMINATIM_USER_AGENT",
    "RIDS/1.0 (Babes-Bolyai University thesis 2026)"
)

# ---------------------------------------------------------------------------
# OSM highway tag → road importance
# ---------------------------------------------------------------------------
_HIGHWAY_IMPORTANCE: Dict[str, int] = {
    "motorway": 3, "motorway_link": 3,
    "trunk":    3, "trunk_link":    3,
    "primary":  3, "primary_link":  3,
    "secondary":     2, "secondary_link":  2,
    "tertiary":      2, "tertiary_link":   2,
    "unclassified":  2,
    "residential":   1, "living_street":   1,
    "service":       1, "track":           1,
    "path":          1, "footway":         1,
    "cycleway":      1,
}

_INFRA_AMENITY_TAGS   = ["hospital", "fire_station", "school"]
_INFRA_SEARCH_RADIUS_M = 500


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnricherConfig:
    """
    All tunable parameters for Stage 6.
    Credentials and URLs are NOT stored here — they are read from .env.

    delay_nominatim_s:
        Seconds between Nominatim requests. Must be >= 1.0 per usage policy.
        Default 1.1 adds a safety margin.

    delay_overpass_s:
        Seconds between Overpass requests. Default 0.5 is polite.

    timeout_s:
        HTTP timeout in seconds for all API calls.

    skip_weather:
        If True, skip Open-Meteo calls (faster, useful offline).

    skip_overpass:
        If True, skip OSM Overpass calls.
    """
    delay_nominatim_s: float = 1.1
    delay_overpass_s:  float = 0.5
    timeout_s:         float = 10.0
    skip_weather:      bool  = False
    skip_overpass:     bool  = False


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WeatherInfo:
    temperature_c:    Optional[float]
    precipitation_mm: Optional[float]
    wind_speed_kmh:   Optional[float]
    condition:        Optional[str]

    def to_dict(self) -> dict:
        return {
            "temperature_c":    self.temperature_c,
            "precipitation_mm": self.precipitation_mm,
            "wind_speed_kmh":   self.wind_speed_kmh,
            "condition":        self.condition,
        }


@dataclass
class EnrichedBox:
    """One detection from Stage 5, enriched with location and weather data."""
    raw:             dict
    street_name:     Optional[str]
    road_class:      Optional[str]
    road_importance: Optional[int]
    infra_proximity: Optional[float]
    weather:         Optional[WeatherInfo]

    def to_dict(self) -> dict:
        d = dict(self.raw)
        d["enrichment"] = {
            "street_name":     self.street_name,
            "road_class":      self.road_class,
            "road_importance": self.road_importance,
            "infra_proximity": (
                round(self.infra_proximity, 2)
                if self.infra_proximity is not None else None
            ),
            "weather": self.weather.to_dict() if self.weather is not None else None,
        }
        return d


@dataclass
class EnrichmentResult:
    """Stage 6 output for one frame."""
    frame_path:    str
    frame_index:   int
    timestamp_s:   float
    latitude:      Optional[float]
    longitude:     Optional[float]
    lighting:      str
    sun_elevation: Optional[float]
    image_width:   int
    image_height:  int
    gps_available: bool
    boxes:         List[EnrichedBox] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> dict:
        return {
            "frame_path":    self.frame_path,
            "frame_index":   self.frame_index,
            "timestamp_s":   self.timestamp_s,
            "latitude":      self.latitude,
            "longitude":     self.longitude,
            "lighting":      self.lighting,
            "sun_elevation": self.sun_elevation,
            "image_width":   self.image_width,
            "image_height":  self.image_height,
            "gps_available": self.gps_available,
            "boxes":         [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _wmo_to_condition(code: Optional[int]) -> str:
    """Map WMO weather code to a human-readable condition string."""
    if code is None:
        return "unknown"
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "cloudy"
    if 51 <= code <= 67:
        return "rain"
    if 71 <= code <= 77:
        return "snow"
    if 80 <= code <= 99:
        return "rain"
    return "unknown"


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

class Enricher:
    """
    Stage 6 — spatial and weather enrichment.

    All API credentials are loaded from .env at module import time.
    No secrets are passed as constructor arguments or stored in config.
    """

    def __init__(self, cfg: EnricherConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        # User-Agent from .env — required by Nominatim usage policy
        self._session.headers.update({"User-Agent": _NOMINATIM_UA})
        self._last_nominatim: float = 0.0
        self._last_overpass:  float = 0.0
        # Overpass cache: avoid re-querying when consecutive frames are on the same road
        self._overpass_cache_result: Optional[dict] = None
        self._overpass_cache_lat:    Optional[float] = None
        self._overpass_cache_lon:    Optional[float] = None
        self._overpass_cache_radius_m: float         = 50.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        frames: List[dict],
        output_dir: Optional[str] = None,
    ) -> List[EnrichmentResult]:
        """
        Enrich all frames from severity_results.json["frames"].

        Per-frame enrichment: one Nominatim + one Overpass + one Open-Meteo
        call per frame (not per box). Result is shared by all boxes in the frame.
        """
        t_start = time.perf_counter()
        results: List[EnrichmentResult] = []

        n_gps_ok = n_gps_miss = 0
        n_nom_ok = n_nom_fail = 0
        n_ov_ok  = n_ov_fail  = 0
        n_wx_ok  = n_wx_fail  = 0
        n_boxes  = 0

        for frame in frames:
            lat    = frame.get("latitude")
            lon    = frame.get("longitude")
            gps_ok = lat is not None and lon is not None

            if gps_ok:
                n_gps_ok += 1
            else:
                n_gps_miss += 1
                logger.debug(
                    "Frame %s — no GPS, enrichment skipped",
                    frame.get("frame_path", "?"),
                )

            street_name = road_class = road_importance = infra_proximity = weather = None

            if gps_ok:
                nom = self._nominatim_reverse(lat, lon)
                if nom:
                    street_name     = nom["street_name"]
                    road_class      = nom["road_class"]
                    road_importance = nom["road_importance"]
                    n_nom_ok += 1
                else:
                    n_nom_fail += 1

                if not self.cfg.skip_overpass:
                    ov = self._overpass_query(lat, lon)
                    if ov:
                        if ov["road_importance"] is not None:
                            road_importance = ov["road_importance"]
                        infra_proximity = ov["infra_proximity"]
                        n_ov_ok += 1
                    else:
                        n_ov_fail += 1

                if not self.cfg.skip_weather:
                    wx = self._fetch_weather(
                        lat, lon,
                        frame.get("wall_time"),       # ISO string from FrameResult
                        frame.get("timestamp_s", 0.0), # relative fallback
                    )
                    if wx:
                        weather = wx
                        n_wx_ok += 1
                    else:
                        n_wx_fail += 1

            boxes: List[EnrichedBox] = []
            for box in frame.get("boxes", []):
                boxes.append(EnrichedBox(
                    raw             = box,
                    street_name     = street_name,
                    road_class      = road_class,
                    road_importance = road_importance,
                    infra_proximity = infra_proximity,
                    weather         = weather,
                ))
                n_boxes += 1

            results.append(EnrichmentResult(
                frame_path    = frame.get("frame_path", ""),
                frame_index   = frame.get("frame_index", -1),
                timestamp_s   = frame.get("timestamp_s", 0.0),
                latitude      = lat,
                longitude     = lon,
                lighting      = frame.get("lighting", "unknown"),
                sun_elevation = frame.get("sun_elevation"),
                image_width   = frame.get("image_width", 640),
                image_height  = frame.get("image_height", 360),
                gps_available = gps_ok,
                boxes         = boxes,
            ))

        elapsed = time.perf_counter() - t_start

        logger.info("=== Enrichment complete ===")
        logger.info("  Frames processed       : %d", len(frames))
        logger.info("  Frames with GPS        : %d", n_gps_ok)
        logger.info("  Frames without GPS     : %d", n_gps_miss)
        logger.info("  Nominatim OK / fail    : %d / %d", n_nom_ok, n_nom_fail)
        logger.info("  Overpass   OK / fail   : %d / %d", n_ov_ok, n_ov_fail)
        logger.info("  Weather    OK / fail   : %d / %d", n_wx_ok, n_wx_fail)
        logger.info("  Boxes total            : %d", n_boxes)
        logger.info("  Elapsed                : %.1f s", elapsed)

        if output_dir:
            self.save_enriched(results, output_dir)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_enriched(results: List[EnrichmentResult], output_dir: str) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "enriched.json"
        payload = {
            "n_frames": len(results),
            "n_boxes":  sum(r.n_detections for r in results),
            "n_gps":    sum(1 for r in results if r.gps_available),
            "frames":   [r.to_dict() for r in results],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Enriched results saved: %s", out_path)
        return str(out_path)

    @staticmethod
    def load_enriched(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        logger.info("Enriched data loaded: %s (%d frames)", path, len(payload["frames"]))
        return payload["frames"]

    # ------------------------------------------------------------------
    # Nominatim — 1 req/s rate limit enforced here
    # ------------------------------------------------------------------

    def _nominatim_reverse(self, lat: float, lon: float) -> Optional[dict]:
        elapsed = time.perf_counter() - self._last_nominatim
        if elapsed < self.cfg.delay_nominatim_s:
            time.sleep(self.cfg.delay_nominatim_s - elapsed)

        try:
            resp = self._session.get(
                _NOMINATIM_URL,
                params={
                    "lat": lat, "lon": lon,
                    "format": "jsonv2",
                    "zoom": 18,
                    "addressdetails": 1,
                    "extratags": 1,
                },
                timeout=self.cfg.timeout_s,
            )
            self._last_nominatim = time.perf_counter()
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Nominatim failed (%.5f, %.5f): %s", lat, lon, exc)
            self._last_nominatim = time.perf_counter()
            return None

        address    = data.get("address", {}) or {}
        street_name = (
            address.get("road") or address.get("pedestrian")
            or address.get("footway") or address.get("cycleway")
            or address.get("residential") or "unknown"
        )
        extratags   = data.get("extratags", {}) or {}
        road_class  = extratags.get("highway") or data.get("type") or "unknown"
        road_importance = _HIGHWAY_IMPORTANCE.get(road_class, 1)

        logger.debug(
            "Nominatim → street=%s  class=%s  importance=%d",
            street_name, road_class, road_importance,
        )
        return {
            "street_name":     street_name,
            "road_class":      road_class,
            "road_importance": road_importance,
        }

    # ------------------------------------------------------------------
    # Overpass — with GPS-proximity cache to avoid redundant calls
    # ------------------------------------------------------------------

    def _overpass_query(self, lat: float, lon: float) -> Optional[dict]:
        # Reuse the cached result if the new GPS point is within 50 m of the
        # last queried point. Consecutive KITTI/dashcam frames are typically
        # <1 m apart — without caching, every frame triggers a new Overpass
        # call, causing 429 rate-limit and timeout errors.
        if (
            self._overpass_cache_result is not None
            and self._overpass_cache_lat is not None
            and self._overpass_cache_lon is not None
        ):
            dist = _haversine_m(
                lat, lon,
                self._overpass_cache_lat,
                self._overpass_cache_lon,
            )
            if dist <= self._overpass_cache_radius_m:
                logger.debug(
                    "Overpass cache hit (%.1f m from last query)", dist
                )
                return self._overpass_cache_result

        elapsed = time.perf_counter() - self._last_overpass
        if elapsed < self.cfg.delay_overpass_s:
            time.sleep(self.cfg.delay_overpass_s - elapsed)

        amenity_filter = "|".join(_INFRA_AMENITY_TAGS)
        query = (
            f"[out:json][timeout:10];\n"
            f"(\n"
            f"  way(around:50,{lat},{lon})[highway];\n"
            f"  node(around:{_INFRA_SEARCH_RADIUS_M},{lat},{lon})"
            f'[amenity~"{amenity_filter}"];\n'
            f");\nout body;\n"
        )

        try:
            resp = self._session.post(
                _OVERPASS_URL,
                data={"data": query},
                timeout=self.cfg.timeout_s,
            )
            self._last_overpass = time.perf_counter()
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
        except requests.RequestException as exc:
            logger.warning("Overpass failed (%.5f, %.5f): %s", lat, lon, exc)
            self._last_overpass = time.perf_counter()
            return None

        road_importance = None
        min_dist        = float("inf")

        for el in elements:
            tags = el.get("tags", {})
            if el.get("type") == "way" and road_importance is None:
                road_importance = _HIGHWAY_IMPORTANCE.get(tags.get("highway", ""))
            if el.get("type") == "node" and tags.get("amenity") in _INFRA_AMENITY_TAGS:
                el_lat, el_lon = el.get("lat"), el.get("lon")
                if el_lat and el_lon:
                    dist = _haversine_m(lat, lon, el_lat, el_lon)
                    if dist < min_dist:
                        min_dist = dist

        infra_proximity = round(min_dist, 1) if min_dist < float("inf") else None
        logger.debug(
            "Overpass → road_importance=%s  infra_proximity=%s m",
            road_importance, infra_proximity,
        )
        result = {"road_importance": road_importance, "infra_proximity": infra_proximity}
        # Cache this result for nearby frames
        self._overpass_cache_result = result
        self._overpass_cache_lat    = lat
        self._overpass_cache_lon    = lon
        return result

    # ------------------------------------------------------------------
    # Open-Meteo — no API key required
    # ------------------------------------------------------------------

    def _fetch_weather(
        self,
        lat: float,
        lon: float,
        wall_time: Optional[str]   = None,
        timestamp_s: float         = 0.0,
    ) -> Optional[WeatherInfo]:
        """
        Fetch weather from Open-Meteo.

        wall_time: ISO 8601 string from FrameResult (e.g. "2011-09-26T13:02:25+00:00").
                   Used when available — gives the correct absolute date.
        timestamp_s: relative seconds since drive start — NOT a Unix epoch timestamp.
                   Only used as a last resort if wall_time is absent.

        Open-Meteo free tier covers approximately the last 90 days.
        Historical dates outside this range are detected and skipped gracefully.
        """
        from datetime import timedelta

        # --- Resolve the wall-clock datetime ---
        dt: Optional[datetime] = None

        if wall_time:
            try:
                # FrameResult.wall_time is stored as an ISO string
                dt = datetime.fromisoformat(wall_time)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                dt = None

        if dt is None:
            # timestamp_s is relative (seconds since drive start), not Unix epoch.
            # Do not call datetime.fromtimestamp() on it — that gives wrong results.
            # Use today as the best available approximation.
            logger.debug(
                "weather: wall_time unavailable, using today as date approximation"
            )
            dt = datetime.now(tz=timezone.utc)

        # --- Check if date is within Open-Meteo free tier range (last ~90 days) ---
        now = datetime.now(tz=timezone.utc)
        age_days = (now - dt.replace(tzinfo=timezone.utc if dt.tzinfo is None else dt.tzinfo)).days
        if age_days > 90:
            logger.debug(
                "weather: date %s is %d days old (>90) — outside Open-Meteo free tier range, skipping",
                dt.strftime("%Y-%m-%d"), age_days,
            )
            return None

        date_str = dt.strftime("%Y-%m-%d")

        try:
            resp = self._session.get(
                _OPENMETEO_URL,
                params={
                    "latitude":       lat,
                    "longitude":      lon,
                    "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
                    "start_date":     date_str,
                    "end_date":       date_str,
                    "timezone":       "UTC",
                    "windspeed_unit": "kmh",
                },
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
        except requests.RequestException as exc:
            logger.warning("Open-Meteo failed (%.5f, %.5f): %s", lat, lon, exc)
            return None

        times = hourly.get("time", [])
        if not times:
            return None

        best_idx = min(range(len(times)), key=lambda i: abs(i - dt.hour))

        def _val(key: str) -> Optional[float]:
            vals = hourly.get(key, [])
            v = vals[best_idx] if best_idx < len(vals) else None
            return float(v) if v is not None else None

        wmo_list  = hourly.get("weathercode", [])
        wmo_code  = (
            int(wmo_list[best_idx])
            if best_idx < len(wmo_list) and wmo_list[best_idx] is not None
            else None
        )

        return WeatherInfo(
            temperature_c    = _val("temperature_2m"),
            precipitation_mm = _val("precipitation"),
            wind_speed_kmh   = _val("windspeed_10m"),
            condition        = _wmo_to_condition(wmo_code),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6 — spatial and weather enrichment."
    )
    parser.add_argument("--input",  required=True,
                        help="severity_results.json from Stage 5")
    parser.add_argument("--output", required=True,
                        help="Output directory for enriched.json")
    parser.add_argument("--delay_nominatim", type=float, default=1.1,
                        help="Seconds between Nominatim requests (min 1.0)")
    parser.add_argument("--delay_overpass",  type=float, default=0.5)
    parser.add_argument("--skip_weather",  action="store_true")
    parser.add_argument("--skip_overpass", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input not found: %s", in_path)
        raise SystemExit(1)

    with in_path.open("r", encoding="utf-8") as f:
        severity_data = json.load(f)

    frames = severity_data.get("frames", [])
    logger.info("Loaded %d frames from %s", len(frames), in_path)

    cfg = EnricherConfig(
        delay_nominatim_s = args.delay_nominatim,
        delay_overpass_s  = args.delay_overpass,
        skip_weather      = args.skip_weather,
        skip_overpass     = args.skip_overpass,
    )
    Enricher(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()