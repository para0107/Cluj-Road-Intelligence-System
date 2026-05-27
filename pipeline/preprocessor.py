"""
pipeline/preprocessor.py
------------------------
Stage 1 of the RIDS inference pipeline.

Responsibilities:
  - Extract frames from a dashcam .mp4 at a configurable rate (default 2 fps)
  - Assign a GPS coordinate to every frame via timestamp interpolation
  - Compute the solar elevation angle per frame (pysolar)
  - Classify per-frame lighting: daylight / overcast / low_light (HSV histogram)

GPS source support:
  - Standard .gpx file  (real dashcam surveys)
  - BDD100K-style GPS JSON (for development/testing without real footage)

GPS synchronisation strategy
-----------------------------
Every frame is assigned a wall-clock UTC time (wall_time) by adding its
offset from the video start to the video's own start time.  GPS coordinates
are then linearly interpolated from the track at that wall_time.

The video start time is resolved in this order of preference:
  1. The caller passes video_start_time explicitly.
  2. The MP4 file contains a creation_time tag in its metadata (read via
     OpenCV / ffprobe fallback).  This is the most common case for modern
     dashcams that embed a real-time clock in the recording.
  3. The first GPS point in the file is used as a last resort, with a
     warning.  This is only correct when the GPS logger and the camera
     started recording at exactly the same moment — an assumption that is
     often false in practice.

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import gpxpy
import gpxpy.gpx
import numpy as np

# pysolar: solar position library
# https://pysolar.readthedocs.io
try:
    from pysolar.solar import get_altitude
    PYSOLAR_AVAILABLE = True
except ImportError:
    PYSOLAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessorConfig:
    """
    All tunable parameters for Stage 1.

    focal_length_px:
        Camera focal length in pixels. Used by downstream stages (SAM surface
        area computation) to convert mask pixels to cm². Set to your dashcam's
        actual value once you have the camera. The default (1400 px) is a
        reasonable starting point for a 1080p dashcam with a ~130° FOV.

    fps:
        How many frames to extract per second of video. At typical survey
        speeds (30–60 km/h) and 2 fps you get one frame every ~4–8 m of road,
        which is dense enough not to miss damage patches while keeping the
        frame count manageable.

    min_frame_interval_s:
        Hard lower bound on the time gap between extracted frames (seconds).

    lighting_thresholds:
        (v_daylight, v_overcast) — mean HSV Value thresholds.

    min_mean_brightness:
        Frames below this mean pixel brightness are dropped. Catches black
        title cards, tunnel blackouts, etc. Set to 0 to disable.
    """
    fps: float = 2.0
    focal_length_px: float = 1400.0
    min_frame_interval_s: float = 0.1
    lighting_thresholds: Tuple[float, float] = (100.0, 50.0)
    shadow_gradient_threshold: float = 30.0
    min_mean_brightness: float = 15.0
    output_format: str = "jpg"
    output_quality: int = 92


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """
    Everything downstream pipeline stages need to know about one frame.

    gps_interpolated: True if the GPS fix was linearly interpolated between
        two surrounding track points; False if it matched a point exactly.
        Always False when lat/lon is None.
    """
    frame_path: str
    frame_index: int
    timestamp_s: float
    wall_time: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    sun_elevation: Optional[float]
    lighting: str
    shadow_geometry_score: float
    focal_length_px: float
    gps_interpolated: bool = True


# ---------------------------------------------------------------------------
# GPS loaders
# ---------------------------------------------------------------------------

class GPSPoint:
    """Minimal GPS record: timestamp (UTC datetime), lat, lon."""
    __slots__ = ("time", "lat", "lon")

    def __init__(self, time: datetime, lat: float, lon: float):
        self.time = time
        self.lat  = lat
        self.lon  = lon


def load_gpx(gpx_path: str) -> List[GPSPoint]:
    """
    Parse a standard .gpx file and return a time-sorted list of GPSPoints.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no valid track points are found.
    """
    path = Path(gpx_path)
    if not path.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")

    logger.info("Loading GPX file: %s", gpx_path)
    with path.open("r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points: List[GPSPoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if pt.time is None or pt.latitude is None or pt.longitude is None:
                    continue
                t = pt.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                points.append(GPSPoint(t, pt.latitude, pt.longitude))

    if not points:
        raise ValueError(f"No valid track points found in {gpx_path}")

    points.sort(key=lambda p: p.time)
    logger.info(
        "Loaded %d GPS points from GPX (span: %s → %s)",
        len(points), points[0].time.isoformat(), points[-1].time.isoformat(),
    )
    return points


def load_bdd100k_gps(json_path: str) -> List[GPSPoint]:
    """
    Parse a BDD100K GPS/IMU JSON file and return a time-sorted list of GPSPoints.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"BDD100K GPS JSON not found: {json_path}")

    logger.info("Loading BDD100K GPS JSON: %s", json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("locations", data.get("gps", []))
    else:
        records = data

    points: List[GPSPoint] = []
    for rec in records:
        lat = rec.get("latitude") or rec.get("lat")
        lon = rec.get("longitude") or rec.get("lon") or rec.get("long")
        ts  = rec.get("timestamp") or rec.get("ts")
        if lat is None or lon is None or ts is None:
            continue
        t = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        points.append(GPSPoint(t, float(lat), float(lon)))

    if not points:
        raise ValueError(f"No usable GPS records in {json_path}")

    points.sort(key=lambda p: p.time)
    logger.info(
        "Loaded %d GPS points from BDD100K JSON (span: %s → %s)",
        len(points), points[0].time.isoformat(), points[-1].time.isoformat(),
    )
    return points


def detect_gps_format(gps_path: str) -> str:
    p = Path(gps_path)
    if p.suffix.lower() == ".gpx":
        return "gpx"
    if p.suffix.lower() == ".json":
        return "bdd100k"
    raise ValueError(
        f"Unsupported GPS file format (must be .gpx or .json): {gps_path}"
    )


# ---------------------------------------------------------------------------
# Video start time extraction
# ---------------------------------------------------------------------------

def _extract_video_start_time(video_path: str) -> Optional[datetime]:
    """
    Try to read the recording start time embedded in the MP4 metadata.

    Strategy 1: read OpenCV's CAP_PROP_* tags (rarely populated).
    Strategy 2: call ffprobe to read the 'creation_time' tag from the
                format metadata. This works for most modern dashcams that
                sync their internal clock to GPS or NTP.

    Returns a timezone-aware UTC datetime, or None if not found.

    Note: this function never raises — all failures are logged as warnings
    so the pipeline can fall back gracefully.
    """
    # --- Strategy 1: ffprobe (most reliable) --------------------------------
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format_tags=creation_time",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            creation_time_str = (
                data.get("format", {})
                    .get("tags", {})
                    .get("creation_time")
            )
            if creation_time_str:
                # ffprobe returns ISO 8601, e.g. "2026-05-23T10:30:00.000000Z"
                ct = datetime.fromisoformat(
                    creation_time_str.replace("Z", "+00:00")
                )
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                logger.info(
                    "Video start time from MP4 metadata (ffprobe): %s",
                    ct.isoformat(),
                )
                return ct
    except FileNotFoundError:
        # ffprobe not installed — not an error, just unavailable
        logger.debug("ffprobe not found; skipping MP4 metadata strategy.")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("ffprobe metadata read failed: %s", exc)

    # --- Strategy 2: OpenCV VideoCapture tags --------------------------------
    # OpenCV exposes CAP_PROP_POS_AVI_RATIO and a few format-specific tags.
    # creation_time is not a standard OpenCV property, but some backends
    # expose it as a string via get(). We try a known property ID.
    # This is unreliable and rarely populated; included as a secondary attempt.
    try:
        cap = cv2.VideoCapture(str(video_path))
        # Property 0x10000 is a non-standard backend-specific tag in some builds.
        # We do not rely on it; this is a best-effort probe only.
        cap.release()
    except Exception:
        pass

    logger.debug(
        "Could not extract creation_time from video metadata for %s. "
        "Will fall back to GPS track anchor.",
        video_path,
    )
    return None


# ---------------------------------------------------------------------------
# GPS interpolation
# ---------------------------------------------------------------------------

def interpolate_gps(
    points: List[GPSPoint],
    target_time: datetime,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Linear interpolation of (lat, lon) for a given UTC datetime.

    The interpolation is linear in geographic coordinates (degrees), which
    introduces a small error for long segments but is negligible at the
    distances between GPS fixes from a moving vehicle (~5–50 m).

    Returns:
        (latitude, longitude, was_interpolated)
        Returns (None, None, False) if target_time is outside the GPS track.
    """
    if not points:
        return None, None, False

    t = target_time
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    if t < points[0].time or t > points[-1].time:
        return None, None, False

    # Binary search for the surrounding pair
    lo, hi = 0, len(points) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if points[mid].time <= t:
            lo = mid
        else:
            hi = mid

    p0, p1 = points[lo], points[hi]
    span = (p1.time - p0.time).total_seconds()
    if span == 0:
        return p0.lat, p0.lon, False

    alpha = (t - p0.time).total_seconds() / span
    lat = p0.lat + alpha * (p1.lat - p0.lat)
    lon = p0.lon + alpha * (p1.lon - p0.lon)
    return lat, lon, alpha > 0.0


# ---------------------------------------------------------------------------
# Lighting classification
# ---------------------------------------------------------------------------

def classify_lighting(
    frame_bgr: np.ndarray,
    thresholds: Tuple[float, float],
) -> Tuple[str, float]:
    """
    Classify a frame's lighting condition from its HSV histogram.

    Returns:
        (lighting_label, shadow_geometry_score)
    """
    hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)

    mean_v = float(np.mean(v_channel))
    v_daylight, v_overcast = thresholds

    if mean_v >= v_daylight:
        label = "daylight"
    elif mean_v >= v_overcast:
        label = "overcast"
    else:
        label = "low_light"

    sobel_x     = cv2.Sobel(s_channel, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y     = cv2.Sobel(s_channel, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag    = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    shadow_score = float(np.mean(grad_mag))

    return label, shadow_score


# ---------------------------------------------------------------------------
# Solar elevation
# ---------------------------------------------------------------------------

def compute_sun_elevation(
    lat: float,
    lon: float,
    when: datetime,
) -> Optional[float]:
    """
    Compute the solar elevation angle (degrees) above the horizon.
    Returns None if pysolar is not installed.
    """
    if not PYSOLAR_AVAILABLE:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    try:
        return float(get_altitude(lat, lon, when))
    except Exception as exc:
        logger.warning(
            "pysolar failed for (%.4f, %.4f) at %s: %s",
            lat, lon, when.isoformat(), exc,
        )
        return None


# ---------------------------------------------------------------------------
# Frame timestamp generator
# ---------------------------------------------------------------------------

def _frame_timestamps(
    video_path: str,
    fps: float,
    min_interval: float,
) -> Generator[float, None, None]:
    """Yield evenly-spaced timestamps (seconds from video start)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s  = frame_count / native_fps
    cap.release()

    interval = max(1.0 / fps, min_interval)
    t = 0.0
    while t < duration_s:
        yield t
        t += interval


# ---------------------------------------------------------------------------
# Main Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Stage 1 of the RIDS pipeline.

    GPS synchronisation
    -------------------
    Each extracted frame is assigned a wall-clock UTC time by adding its
    video-offset (seconds) to the video's recording start time.  Coordinates
    are then linearly interpolated from the GPS track at that wall-clock time.

    The recording start time is resolved in order:
      1. video_start_time argument (explicit, most accurate)
      2. MP4 creation_time metadata tag read via ffprobe
      3. First GPS point in the file (fallback with warning — only correct
         when camera and GPS logger started at exactly the same second)
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.cfg = config or PreprocessorConfig()
        logger.info(
            "Preprocessor initialised | fps=%.1f | focal_length_px=%.1f | "
            "lighting_thresholds=%s | output_format=%s",
            self.cfg.fps,
            self.cfg.focal_length_px,
            self.cfg.lighting_thresholds,
            self.cfg.output_format,
        )

    def run(
        self,
        video_path: str,
        gps_path: str,
        output_dir: str,
        video_start_time: Optional[datetime] = None,
    ) -> List[FrameResult]:
        """
        Extract frames, sync GPS, classify lighting, compute sun angle.

        Args:
            video_path:       Path to dashcam .mp4.
            gps_path:         Path to GPS file (.gpx or BDD100K .json).
            output_dir:       Directory for saved frame images.
            video_start_time: UTC datetime of the first video frame.
                              If None, resolved automatically from MP4
                              metadata or the first GPS point (see class
                              docstring for priority order).

        Returns:
            List of FrameResult, one per extracted frame, in time order.
        """
        video_path = str(video_path)
        gps_path   = str(gps_path)
        output_dir = str(output_dir)

        logger.info("=== Preprocessor.run ===")
        logger.info("  video : %s", video_path)
        logger.info("  gps   : %s", gps_path)
        logger.info("  output: %s", output_dir)

        # 1. Load GPS track
        gps_format = detect_gps_format(gps_path)
        gps_points = load_gpx(gps_path) if gps_format == "gpx" else load_bdd100k_gps(gps_path)

        # 2. Determine video wall-clock start time
        if video_start_time is not None:
            # Caller supplied it explicitly — most accurate
            if video_start_time.tzinfo is None:
                video_start_time = video_start_time.replace(tzinfo=timezone.utc)
            logger.info(
                "video_start_time (explicit): %s", video_start_time.isoformat()
            )
        else:
            # Try to read from MP4 metadata first
            video_start_time = _extract_video_start_time(video_path)

            if video_start_time is not None:
                logger.info(
                    "video_start_time (from MP4 metadata): %s",
                    video_start_time.isoformat(),
                )
            else:
                # Last resort: use first GPS point.
                # This is only correct if the GPS logger and camera started
                # recording at the exact same moment.
                video_start_time = gps_points[0].time
                logger.warning(
                    "video_start_time could not be determined from MP4 metadata. "
                    "Using first GPS point (%s) as the video start anchor. "
                    "GPS coordinates will be WRONG if the GPS logger started "
                    "recording before or after the camera. "
                    "To fix: ensure your dashcam embeds a creation_time tag, "
                    "or pass video_start_time explicitly.",
                    video_start_time.isoformat(),
                )

        # Log the GPS track span vs the video start time so misalignment is
        # immediately visible in the logs.
        gps_start = gps_points[0].time
        gps_end   = gps_points[-1].time
        offset_s  = (video_start_time - gps_start).total_seconds()
        logger.info(
            "GPS track span: %s → %s (%.1f s)",
            gps_start.isoformat(), gps_end.isoformat(),
            (gps_end - gps_start).total_seconds(),
        )
        logger.info(
            "Video start vs GPS start: %.1f s offset "
            "(positive = video starts after GPS track began)",
            offset_s,
        )
        if abs(offset_s) > 60:
            logger.warning(
                "Large offset (%.1f s) between video start time and GPS track "
                "start. Check that the correct GPS file was provided and that "
                "both devices have synchronised clocks.",
                offset_s,
            )

        # 3. Prepare output directory
        os.makedirs(output_dir, exist_ok=True)

        # 4. Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s  = frame_count / native_fps
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video: %.1f s | %.1f fps native | %d frames | %dx%d",
            duration_s, native_fps, frame_count, width, height,
        )

        interval_s = max(1.0 / self.cfg.fps, self.cfg.min_frame_interval_s)
        target_timestamps = list(
            _frame_timestamps(video_path, self.cfg.fps, self.cfg.min_frame_interval_s)
        )
        logger.info(
            "Extraction plan: %d frames @ %.2f s intervals (%.1f fps)",
            len(target_timestamps), interval_s, self.cfg.fps,
        )

        # 5. Extract frames
        results: List[FrameResult] = []
        ext         = self.cfg.output_format.lower()
        save_params = (
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.output_quality]
            if ext == "jpg" else []
        )

        for idx, t_s in enumerate(target_timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning(
                    "Frame %d (t=%.2f s): read failed — skipping", idx, t_s
                )
                continue

            # Brightness filter
            if self.cfg.min_mean_brightness > 0:
                mean_brightness = float(
                    np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                )
                if mean_brightness < self.cfg.min_mean_brightness:
                    logger.debug(
                        "Frame %d (t=%.2f s): dropped — brightness %.1f < %.1f",
                        idx, t_s, mean_brightness, self.cfg.min_mean_brightness,
                    )
                    continue

            # Wall-clock time for this frame
            wall_time = video_start_time + timedelta(seconds=t_s)

            # GPS interpolation
            lat, lon, interpolated = interpolate_gps(gps_points, wall_time)
            if lat is None:
                logger.warning(
                    "Frame %d (t=%.2f s, wall=%s): outside GPS track span "
                    "— lat/lon will be None",
                    idx, t_s, wall_time.isoformat(),
                )

            # Lighting classification
            lighting, shadow_score = classify_lighting(
                frame, self.cfg.lighting_thresholds
            )

            # Solar elevation
            sun_elev: Optional[float] = None
            if lat is not None and lon is not None:
                sun_elev = compute_sun_elevation(lat, lon, wall_time)

            # Save frame
            frame_filename = f"frame_{idx:06d}_t{t_s:.3f}.{ext}"
            frame_path     = os.path.join(output_dir, frame_filename)
            success        = cv2.imwrite(frame_path, frame, save_params)
            if not success:
                logger.error("Frame %d: imwrite failed for %s", idx, frame_path)
                continue

            results.append(FrameResult(
                frame_path=frame_path,
                frame_index=idx,
                timestamp_s=t_s,
                wall_time=wall_time,
                latitude=lat,
                longitude=lon,
                sun_elevation=sun_elev,
                lighting=lighting,
                shadow_geometry_score=shadow_score,
                focal_length_px=self.cfg.focal_length_px,
                gps_interpolated=interpolated,
            ))

            if idx % 20 == 0:
                logger.info(
                    "Frame %d/%d | t=%.2f s | wall=%s | lighting=%s | "
                    "sun=%.1f° | lat=%s | lon=%s",
                    idx, len(target_timestamps), t_s,
                    wall_time.isoformat(),
                    lighting,
                    f"{sun_elev:.1f}" if sun_elev is not None else "n/a",
                    f"{lat:.5f}" if lat is not None else "None",
                    f"{lon:.5f}" if lon is not None else "None",
                )

        cap.release()

        # 6. Summary
        n_daylight = sum(1 for r in results if r.lighting == "daylight")
        n_overcast = sum(1 for r in results if r.lighting == "overcast")
        n_low      = sum(1 for r in results if r.lighting == "low_light")
        n_no_gps   = sum(1 for r in results if r.latitude is None)
        n_dropped  = len(target_timestamps) - len(results)

        logger.info("=== Preprocessing complete ===")
        logger.info("  Total frames extracted : %d", len(results))
        logger.info("  Frames dropped (dark)  : %d  (min_brightness=%.0f)",
                    n_dropped, self.cfg.min_mean_brightness)
        logger.info("  Lighting breakdown     : daylight=%d overcast=%d low_light=%d",
                    n_daylight, n_overcast, n_low)
        logger.info("  Frames without GPS     : %d", n_no_gps)
        if n_no_gps > 0:
            logger.warning(
                "%d frames have no GPS coordinates. This means their wall_time "
                "fell outside the GPS track span. Check clock alignment between "
                "the camera and GPS logger.",
                n_no_gps,
            )
        if not PYSOLAR_AVAILABLE:
            logger.warning(
                "pysolar not installed — sun_elevation is None for all frames. "
                "Install with: pip install pysolar"
            )

        return results

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_manifest(results: List[FrameResult], manifest_path: str) -> None:
        records = []
        for r in results:
            records.append({
                "frame_path":            r.frame_path,
                "frame_index":           r.frame_index,
                "timestamp_s":           r.timestamp_s,
                "wall_time":             r.wall_time.isoformat() if r.wall_time else None,
                "latitude":              r.latitude,
                "longitude":             r.longitude,
                "sun_elevation":         r.sun_elevation,
                "lighting":              r.lighting,
                "shadow_geometry_score": r.shadow_geometry_score,
                "focal_length_px":       r.focal_length_px,
                "gps_interpolated":      r.gps_interpolated,
            })
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("Manifest saved: %s (%d frames)", manifest_path, len(records))

    @staticmethod
    def load_manifest(manifest_path: str) -> List[FrameResult]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        results = []
        for rec in records:
            wt = rec.get("wall_time")
            results.append(FrameResult(
                frame_path=rec["frame_path"],
                frame_index=rec["frame_index"],
                timestamp_s=rec["timestamp_s"],
                wall_time=datetime.fromisoformat(wt) if wt else None,
                latitude=rec.get("latitude"),
                longitude=rec.get("longitude"),
                sun_elevation=rec.get("sun_elevation"),
                lighting=rec["lighting"],
                shadow_geometry_score=rec["shadow_geometry_score"],
                focal_length_px=rec["focal_length_px"],
                gps_interpolated=rec.get("gps_interpolated", True),
            ))
        logger.info("Manifest loaded: %s (%d frames)", manifest_path, len(results))
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIDS Stage 1 — Preprocessor: extract frames and sync GPS."
    )
    parser.add_argument("--video",    required=True, help="Path to dashcam .mp4")
    parser.add_argument("--gps",      required=True, help="Path to .gpx or BDD100K .json")
    parser.add_argument("--output",   required=True, help="Output directory for frames")
    parser.add_argument("--manifest", default=None,  help="Optional: path to save manifest JSON")
    parser.add_argument("--fps",      type=float, default=2.0)
    parser.add_argument("--focal-length", type=float, default=1400.0)
    parser.add_argument("--min-brightness", type=float, default=15.0)
    parser.add_argument(
        "--video-start-time",
        default=None,
        help=(
            "UTC ISO-8601 datetime of the first video frame, e.g. "
            "'2026-05-23T10:30:00+00:00'. Use this when the MP4 does not "
            "contain a creation_time metadata tag and the GPS logger started "
            "at a different moment than the camera."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    video_start_time: Optional[datetime] = None
    if args.video_start_time:
        video_start_time = datetime.fromisoformat(args.video_start_time)
        if video_start_time.tzinfo is None:
            video_start_time = video_start_time.replace(tzinfo=timezone.utc)

    cfg = PreprocessorConfig(
        fps=args.fps,
        focal_length_px=args.focal_length,
        min_mean_brightness=args.min_brightness,
    )
    pp = Preprocessor(cfg)

    results = pp.run(
        video_path=args.video,
        gps_path=args.gps,
        output_dir=args.output,
        video_start_time=video_start_time,
    )

    manifest_path = args.manifest or os.path.join(args.output, "manifest.json")
    Preprocessor.save_manifest(results, manifest_path)
    logger.info("Manifest saved to: %s", manifest_path)
    print(f"\nDone. {len(results)} frames extracted to: {args.output}")


if __name__ == "__main__":
    main()