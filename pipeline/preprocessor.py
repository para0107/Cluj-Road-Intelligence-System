"""
pipeline/preprocessor.py
------------------------
Stage 1 of the CRIS inference pipeline.

Responsibilities:
  - Extract frames from a dashcam .mp4 at a configurable rate (default 2 fps)
  - Assign a GPS coordinate to every frame via timestamp interpolation
  - Compute the solar elevation angle per frame (pysolar)
  - Classify per-frame lighting: daylight / overcast / low_light (HSV histogram)

GPS source support:
  - Standard .gpx file  (real dashcam surveys)
  - BDD100K-style GPS JSON (for development/testing without real footage)

All public functions log every significant action so the full processing
history can be replayed, plotted, or audited later.

Usage (module):
    from pipeline.preprocessor import Preprocessor, PreprocessorConfig
    cfg = PreprocessorConfig(fps=2, focal_length_px=1400.0)
    pp  = Preprocessor(cfg)
    frames = pp.run(video_path="data/raw/footage/clip.mp4",
                    gps_path="data/raw/gps_logs/clip.gpx")

Usage (CLI):
    python -m pipeline.preprocessor \
        --video  data/raw/footage/clip.mp4 \
        --gps    data/raw/gps_logs/clip.gpx \
        --output data/processed/frames/clip/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    from pysolar.util import extraterrestrial_irrad
    PYSOLAR_AVAILABLE = True
except ImportError:
    PYSOLAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging setup — module-level logger; callers can attach their own handlers
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# All camera-specific parameters live here so the preprocessor itself never
# has any hardcoded values.
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
        Guards against accidentally requesting more frames than the video
        actually contains at high fps values.

    lighting_thresholds:
        (v_daylight, v_overcast) — mean HSV Value thresholds.
        Frames with mean V >= v_daylight → daylight
        Frames with mean V in [v_overcast, v_daylight) → overcast
        Frames with mean V < v_overcast → low_light

    shadow_gradient_threshold:
        Sobel gradient magnitude threshold for shadow detection score.
        Passed through to the FrameResult; not used by the preprocessor itself.
    """
    fps: float = 2.0
    focal_length_px: float = 1400.0          # CONFIGURE when camera is known
    min_frame_interval_s: float = 0.1
    lighting_thresholds: Tuple[float, float] = (100.0, 50.0)  # (daylight, overcast)
    shadow_gradient_threshold: float = 30.0
    output_format: str = "jpg"               # "jpg" or "png"
    output_quality: int = 92                 # JPEG quality (0-100), ignored for PNG


# ---------------------------------------------------------------------------
# Output dataclass for a single extracted frame
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """
    Everything the downstream pipeline stages need to know about one frame.

    frame_path:       Absolute path to the saved frame image.
    frame_index:      Sequential index within this video (0-based).
    timestamp_s:      Time offset from video start in seconds.
    wall_time:        UTC datetime corresponding to this frame (interpolated
                      from GPS track start + timestamp_s).
    latitude:         WGS84 latitude  (None if GPS sync failed for this frame).
    longitude:        WGS84 longitude (None if GPS sync failed for this frame).
    sun_elevation:    Solar elevation angle in degrees above the horizon.
                      Negative = below horizon (night). None if pysolar not
                      available or GPS coords are missing.
    lighting:         "daylight" | "overcast" | "low_light"
    shadow_geometry_score: Mean Sobel gradient magnitude in HSV-S channel.
                      Higher values indicate stronger shadow edges.
    focal_length_px:  Forwarded from config; lets downstream stages be
                      self-contained without importing the config object.
    gps_interpolated: True if the GPS fix was interpolated between two
                      surrounding track points; False if it was an exact match.
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
        self.lat = lat
        self.lon = lon


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
                # Ensure timezone-aware UTC
                t = pt.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                points.append(GPSPoint(t, pt.latitude, pt.longitude))

    if not points:
        raise ValueError(f"No valid track points found in {gpx_path}")

    points.sort(key=lambda p: p.time)
    logger.info("Loaded %d GPS points from GPX (span: %s → %s)",
                len(points), points[0].time.isoformat(), points[-1].time.isoformat())
    return points


def load_bdd100k_gps(json_path: str) -> List[GPSPoint]:
    """
    Parse a BDD100K GPS/IMU JSON file and return a time-sorted list of GPSPoints.

    BDD100K stores GPS as a JSON array. Each element looks like:
        {
          "timestamp": 1512337014123,   <- epoch ms
          "latitude":  37.776413,
          "longitude": -122.416785
        }

    The IMU fields (speed, course, accuracy) are ignored here — only lat/lon/time
    are used for GPS sync.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no GPS records with latitude+longitude are found.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"BDD100K GPS JSON not found: {json_path}")

    logger.info("Loading BDD100K GPS JSON: %s", json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # BDD100K wraps GPS in a top-level "locations" key in some versions
    if isinstance(data, dict):
        records = data.get("locations", data.get("gps", []))
    else:
        records = data  # raw list

    points: List[GPSPoint] = []
    for rec in records:
        lat = rec.get("latitude") or rec.get("lat")
        lon = rec.get("longitude") or rec.get("lon") or rec.get("long")
        ts  = rec.get("timestamp") or rec.get("ts")

        if lat is None or lon is None or ts is None:
            continue

        # BDD100K timestamps are epoch milliseconds
        t = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        points.append(GPSPoint(t, float(lat), float(lon)))

    if not points:
        raise ValueError(f"No usable GPS records in {json_path}")

    points.sort(key=lambda p: p.time)
    logger.info("Loaded %d GPS points from BDD100K JSON (span: %s → %s)",
                len(points), points[0].time.isoformat(), points[-1].time.isoformat())
    return points


def detect_gps_format(gps_path: str) -> str:
    """
    Detect whether a GPS file is GPX or BDD100K JSON based on extension and
    a quick content sniff. Returns "gpx" or "bdd100k".
    """
    p = Path(gps_path)
    if p.suffix.lower() == ".gpx":
        return "gpx"
    if p.suffix.lower() == ".json":
        # Quick content check: GPX-converted JSON would have a "type": "gpx"
        # key; BDD100K JSON is typically an array or has "locations"/"gps" keys
        with p.open("r", encoding="utf-8") as f:
            head = f.read(512)
        return "bdd100k"  # default for .json
    raise ValueError(f"Unsupported GPS file format (must be .gpx or .json): {gps_path}")


# ---------------------------------------------------------------------------
# GPS interpolation
# ---------------------------------------------------------------------------

def interpolate_gps(
    points: List[GPSPoint],
    target_time: datetime,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Linear interpolation of (lat, lon) for a given UTC datetime.

    Returns:
        (latitude, longitude, was_interpolated)
        was_interpolated is False only if target_time matches a point exactly.
        Returns (None, None, False) if target_time is outside the GPS track span.
    """
    if not points:
        return None, None, False

    t = target_time
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    t_start = points[0].time
    t_end   = points[-1].time

    if t < t_start or t > t_end:
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
    interpolated = alpha > 0.0
    return lat, lon, interpolated


# ---------------------------------------------------------------------------
# Lighting classification
# ---------------------------------------------------------------------------

def classify_lighting(
    frame_bgr: np.ndarray,
    thresholds: Tuple[float, float],
) -> Tuple[str, float]:
    """
    Classify a frame's lighting condition from its HSV histogram.

    Args:
        frame_bgr:   BGR image as returned by cv2.
        thresholds:  (v_daylight, v_overcast) — mean V-channel cutoffs.

    Returns:
        (lighting_label, shadow_geometry_score)
        lighting_label: "daylight" | "overcast" | "low_light"
        shadow_geometry_score: mean Sobel magnitude on the S channel,
            proxy for shadow edge presence.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
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

    # Shadow geometry score: Sobel on saturation channel
    sobel_x = cv2.Sobel(s_channel, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(s_channel, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
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

    Negative values mean the sun is below the horizon (night/twilight).
    Returns None if pysolar is not installed.
    """
    if not PYSOLAR_AVAILABLE:
        return None

    # pysolar requires a timezone-aware datetime
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    try:
        elevation = get_altitude(lat, lon, when)
        return float(elevation)
    except Exception as exc:  # pysolar can raise on edge-case inputs
        logger.warning("pysolar failed for (%.4f, %.4f) at %s: %s",
                       lat, lon, when.isoformat(), exc)
        return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _frame_timestamps(
    video_path: str,
    fps: float,
    min_interval: float,
) -> Generator[float, None, None]:
    """
    Yield evenly-spaced timestamps (seconds from video start) at which frames
    should be extracted.

    Args:
        video_path:    Path to the video file.
        fps:           Target extraction rate.
        min_interval:  Minimum gap between frames in seconds.

    Yields:
        Float timestamps in [0, video_duration).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s    = frame_count / native_fps
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
    Stage 1 of the CRIS pipeline.

    Example:
        cfg    = PreprocessorConfig(fps=2, focal_length_px=1400.0)
        pp     = Preprocessor(cfg)
        frames = pp.run(
            video_path="data/raw/footage/clip.mp4",
            gps_path="data/raw/gps_logs/clip.gpx",
            output_dir="data/processed/frames/clip/",
        )
        for f in frames:
            print(f.frame_path, f.lighting, f.sun_elevation)
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        gps_path: str,
        output_dir: str,
        video_start_time: Optional[datetime] = None,
    ) -> List[FrameResult]:
        """
        Extract frames from a video, sync GPS, classify lighting, compute sun angle.

        Args:
            video_path:         Path to the dashcam .mp4 (or any OpenCV-readable format).
            gps_path:           Path to the GPS file (.gpx or BDD100K .json).
            output_dir:         Directory where frame images will be saved.
                                Created if it does not exist.
            video_start_time:   UTC datetime of the first frame in the video.
                                If None, it is inferred from the first GPS point
                                (works well for GPX files that contain absolute
                                timestamps; less accurate for BDD100K).

        Returns:
            List of FrameResult objects, one per extracted frame, in time order.
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
        if gps_format == "gpx":
            gps_points = load_gpx(gps_path)
        else:
            gps_points = load_bdd100k_gps(gps_path)

        # 2. Determine video wall-clock start time
        if video_start_time is None:
            video_start_time = gps_points[0].time
            logger.info(
                "video_start_time not supplied; using first GPS point: %s",
                video_start_time.isoformat(),
            )
        else:
            if video_start_time.tzinfo is None:
                video_start_time = video_start_time.replace(tzinfo=timezone.utc)
            logger.info("video_start_time (supplied): %s", video_start_time.isoformat())

        # 3. Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Output directory ready: %s", output_dir)

        # 4. Open video and validate
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = frame_count / native_fps
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video: %.1f s | %.1f fps native | %d frames | %dx%d",
            duration_s, native_fps, frame_count, width, height,
        )

        interval_s = max(1.0 / self.cfg.fps, self.cfg.min_frame_interval_s)
        target_timestamps = list(_frame_timestamps(
            video_path, self.cfg.fps, self.cfg.min_frame_interval_s
        ))
        logger.info(
            "Extraction plan: %d frames @ %.2f s intervals (%.1f fps)",
            len(target_timestamps), interval_s, self.cfg.fps,
        )

        # 5. Extract frames
        results: List[FrameResult] = []
        ext = self.cfg.output_format.lower()
        save_params = (
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.output_quality]
            if ext == "jpg"
            else []
        )

        for idx, t_s in enumerate(target_timestamps):
            # Seek to target timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.warning("Frame %d (t=%.2f s): read failed — skipping", idx, t_s)
                continue

            # 5a. GPS sync
            wall_time = video_start_time.replace(
                tzinfo=video_start_time.tzinfo
            )
            from datetime import timedelta
            wall_time = video_start_time + timedelta(seconds=t_s)

            lat, lon, interpolated = interpolate_gps(gps_points, wall_time)

            if lat is None:
                logger.warning(
                    "Frame %d (t=%.2f s): GPS sync failed (outside track span) "
                    "— lat/lon will be None",
                    idx, t_s,
                )

            # 5b. Lighting classification
            lighting, shadow_score = classify_lighting(
                frame, self.cfg.lighting_thresholds
            )

            # 5c. Solar elevation
            sun_elev: Optional[float] = None
            if lat is not None and lon is not None:
                sun_elev = compute_sun_elevation(lat, lon, wall_time)

            # 5d. Save frame to disk
            frame_filename = f"frame_{idx:06d}_t{t_s:.3f}.{ext}"
            frame_path = os.path.join(output_dir, frame_filename)
            success = cv2.imwrite(frame_path, frame, save_params)

            if not success:
                logger.error("Frame %d: imwrite failed for %s", idx, frame_path)
                continue

            result = FrameResult(
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
            )
            results.append(result)

            if idx % 20 == 0:
                logger.info(
                    "Frame %d/%d | t=%.2f s | lighting=%s | "
                    "sun=%.1f° | lat=%.5f | lon=%.5f",
                    idx, len(target_timestamps), t_s,
                    lighting,
                    sun_elev if sun_elev is not None else float("nan"),
                    lat if lat is not None else float("nan"),
                    lon if lon is not None else float("nan"),
                )

        cap.release()

        # 6. Summary
        n_daylight  = sum(1 for r in results if r.lighting == "daylight")
        n_overcast  = sum(1 for r in results if r.lighting == "overcast")
        n_low       = sum(1 for r in results if r.lighting == "low_light")
        n_no_gps    = sum(1 for r in results if r.latitude is None)

        logger.info("=== Preprocessing complete ===")
        logger.info("  Total frames extracted : %d", len(results))
        logger.info("  Lighting breakdown     : daylight=%d overcast=%d low_light=%d",
                    n_daylight, n_overcast, n_low)
        logger.info("  Frames without GPS     : %d", n_no_gps)
        if not PYSOLAR_AVAILABLE:
            logger.warning(
                "pysolar not installed — sun_elevation is None for all frames. "
                "Install with: pip install pysolar"
            )

        return results

    # ------------------------------------------------------------------
    # Convenience: save results manifest as JSON
    # ------------------------------------------------------------------

    @staticmethod
    def save_manifest(results: List[FrameResult], manifest_path: str) -> None:
        """
        Persist the list of FrameResult objects to a JSON manifest.
        Useful for passing preprocessor output to the next pipeline stage
        without keeping everything in memory.
        """
        records = []
        for r in results:
            records.append({
                "frame_path":          r.frame_path,
                "frame_index":         r.frame_index,
                "timestamp_s":         r.timestamp_s,
                "wall_time":           r.wall_time.isoformat() if r.wall_time else None,
                "latitude":            r.latitude,
                "longitude":           r.longitude,
                "sun_elevation":       r.sun_elevation,
                "lighting":            r.lighting,
                "shadow_geometry_score": r.shadow_geometry_score,
                "focal_length_px":     r.focal_length_px,
                "gps_interpolated":    r.gps_interpolated,
            })

        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        logger.info("Manifest saved: %s (%d frames)", manifest_path, len(records))

    @staticmethod
    def load_manifest(manifest_path: str) -> List[FrameResult]:
        """
        Load a previously saved manifest back into FrameResult objects.
        Used by downstream pipeline stages to pick up where preprocessing left off.
        """
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
        description="CRIS Stage 1 — Preprocessor: extract frames and sync GPS."
    )
    parser.add_argument("--video",  required=True,  help="Path to dashcam .mp4")
    parser.add_argument("--gps",    required=True,  help="Path to .gpx or BDD100K .json")
    parser.add_argument("--output", required=True,  help="Output directory for frames")
    parser.add_argument("--manifest", default=None, help="Optional: path to save manifest JSON")
    parser.add_argument("--fps",    type=float, default=2.0,  help="Extraction rate (default 2)")
    parser.add_argument(
        "--focal-length", type=float, default=1400.0,
        help="Camera focal length in pixels (default 1400; update when camera is known)"
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    cfg = PreprocessorConfig(
        fps=args.fps,
        focal_length_px=args.focal_length,
    )
    pp = Preprocessor(cfg)

    results = pp.run(
        video_path=args.video,
        gps_path=args.gps,
        output_dir=args.output,
    )

    if args.manifest:
        Preprocessor.save_manifest(results, args.manifest)
    else:
        # Default: save manifest alongside frames
        manifest_path = os.path.join(args.output, "manifest.json")
        Preprocessor.save_manifest(results, manifest_path)
        logger.info("Manifest auto-saved to: %s", manifest_path)

    print(f"\nDone. {len(results)} frames extracted to: {args.output}")


if __name__ == "__main__":
    main()