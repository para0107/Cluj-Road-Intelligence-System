"""
pipeline/live_camera.py

Live-mode edge agent — turns a vehicle into a roaming road-damage sensor.

Runs the SAME baseline detector as the survey pipeline (RT-DETR-L fine-tuned
on N-RDD2024, ml/weights/best.pt) over a dashcam stream (video file or
webcam), synchronises frames to GPS (optional .gpx log), and pushes every
qualifying detection to the Live API:

    POST {api}/api/live/reports
        { device_id, latitude, longitude, damage_type, confidence, severity }

The server clusters sightings from multiple vehicles into validated events
(unverified → confirmed → verified), so nothing else is needed on the edge —
no DB access, no session state. Run one instance per camera/vehicle.

Examples
--------
# Replay a recorded drive with its GPS track (GPU host):
python pipeline/live_camera.py --video data/raw/footage/drive.mp4 \
    --gps data/raw/gps_logs/drive.gpx --device-id car-01 --device cuda

# Webcam, fixed location (bench test):
python pipeline/live_camera.py --camera 0 --lat 46.7712 --lon 23.6236 \
    --device-id bench-cam --device cpu

Local duplicate suppression: the agent does not re-report the same class
within --min-gap-m metres of its previous report of that class inside
--cooldown-s seconds — the same physical pothole seen on 12 consecutive
frames produces one report, not twelve.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ── Defaults (env-overridable) ──────────────────────────────────────────────
_WEIGHTS = Path(os.getenv("WEIGHTS_DIR", "ml/weights")) / os.getenv("RTDETR_WEIGHTS", "best.pt")
_DEFAULT_API = os.getenv("LIVE_API_URL", "http://localhost:8000")
_DEFAULT_CONF = float(os.getenv("LIVE_MIN_CONF", "0.40"))


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ── GPS track (.gpx → time-interpolated positions) ──────────────────────────

class GpsTrack:
    """Linear time interpolation over the points of a .gpx log."""

    def __init__(self, gpx_path: str):
        import gpxpy
        with open(gpx_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        pts = [
            (p.time.timestamp(), p.latitude, p.longitude)
            for trk in gpx.tracks for seg in trk.segments for p in seg.points
            if p.time is not None
        ]
        if len(pts) < 2:
            raise ValueError("GPX log needs at least 2 timestamped points")
        pts.sort(key=lambda x: x[0])
        self.t0 = pts[0][0]
        self.pts = pts

    def at(self, video_elapsed_s: float) -> tuple[float, float]:
        """Position at t0 + elapsed seconds (clamped to the track ends)."""
        t = self.t0 + video_elapsed_s
        pts = self.pts
        if t <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if t >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]
        # Binary search would be nicer; tracks are small enough for a scan.
        for (t1, la1, lo1), (t2, la2, lo2) in zip(pts, pts[1:]):
            if t1 <= t <= t2:
                f = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)
                return la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
        return pts[-1][1], pts[-1][2]


# ── Severity heuristic on the edge ──────────────────────────────────────────
# The full pipeline derives severity from SAM geometry + Monodepth2; the edge
# agent has neither, so estimate coarsely from confidence and bbox area. The
# server keeps the max across devices, and survey runs refine it later.

def edge_severity(conf: float, box_area_frac: float) -> int:
    score = 0.6 * conf + 0.4 * min(box_area_frac * 25.0, 1.0)
    if score < 0.25: return 1
    if score < 0.45: return 2
    if score < 0.62: return 3
    if score < 0.8:  return 4
    return 5


def post_report(api: str, payload: dict, timeout: float = 5.0) -> dict | None:
    try:
        r = requests.post(f"{api}/api/live/reports", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        print(f"  [!] report failed: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="RIDS live-mode edge camera agent")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Dashcam video file to replay")
    src.add_argument("--camera", type=int, help="Webcam index (e.g. 0)")
    ap.add_argument("--gps", help=".gpx log matching the video (optional)")
    ap.add_argument("--lat", type=float, help="Fixed latitude when no GPX")
    ap.add_argument("--lon", type=float, help="Fixed longitude when no GPX")
    ap.add_argument("--api", default=_DEFAULT_API, help=f"Backend base URL (default {_DEFAULT_API})")
    ap.add_argument("--device-id", default=f"cam-{os.getpid()}", help="Stable identity of this camera/vehicle")
    ap.add_argument("--weights", default=str(_WEIGHTS), help="Detector checkpoint (default ml/weights/best.pt)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    ap.add_argument("--conf", type=float, default=_DEFAULT_CONF, help="Min detector confidence to report")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--every", type=int, default=15, help="Run inference every Nth frame (latency/compute knob)")
    ap.add_argument("--min-gap-m", type=float, default=20.0, help="Local dedup: min metres between reports of the same class")
    ap.add_argument("--cooldown-s", type=float, default=30.0, help="Local dedup: min seconds between reports of the same class nearby")
    ap.add_argument("--realtime", action="store_true", help="Replay video at native speed instead of as fast as possible")
    args = ap.parse_args()

    if not args.gps and (args.lat is None or args.lon is None):
        ap.error("Provide --gps, or --lat and --lon for a fixed position.")

    # Heavy import here so --help stays fast
    from ultralytics import RTDETR
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[x] Detector weights not found: {weights}")
        return 1
    model = RTDETR(str(weights))
    names = model.names  # class-id → damage_type (N-RDD2024 schema)

    track = GpsTrack(args.gps) if args.gps else None
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("[x] Could not open video source")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[live-camera] device_id={args.device_id} api={args.api}")
    print(f"[live-camera] weights={weights} device={args.device} conf>={args.conf} every={args.every}f")

    last_report: dict[str, tuple[float, float, float]] = {}  # class → (t, lat, lon)
    frame_idx, sent = 0, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[live-camera] end of stream")
                break
            frame_idx += 1
            if frame_idx % args.every:
                if args.realtime:
                    time.sleep(1.0 / fps)
                continue

            elapsed = frame_idx / fps
            lat, lon = track.at(elapsed) if track else (args.lat, args.lon)

            h, w = frame.shape[:2]
            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                    device=args.device, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)

                    # Local duplicate suppression per class
                    now = time.time()
                    prev = last_report.get(cls)
                    if prev:
                        pt, plat, plon = prev
                        if (now - pt) < args.cooldown_s and haversine_m(lat, lon, plat, plon) < args.min_gap_m:
                            continue

                    payload = {
                        "device_id": args.device_id,
                        "latitude": lat,
                        "longitude": lon,
                        "damage_type": cls,
                        "confidence": round(conf, 3),
                        "severity": edge_severity(conf, area_frac),
                    }
                    resp = post_report(args.api, payload)
                    if resp:
                        last_report[cls] = (now, lat, lon)
                        sent += 1
                        ev = resp.get("event") or {}
                        print(
                            f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                            f"{resp.get('action', '?'):7s} {cls:24s} conf={conf:.2f} "
                            f"→ event {str(ev.get('id'))[:8]} status={ev.get('status')} "
                            f"devices={ev.get('reporter_devices')}"
                        )

            if args.realtime:
                time.sleep(args.every / fps)
    except KeyboardInterrupt:
        print("\n[live-camera] stopped by user")
    finally:
        cap.release()

    print(f"[live-camera] done — {sent} reports sent from {frame_idx} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
