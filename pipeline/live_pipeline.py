"""
pipeline/live_pipeline.py

The LITE pipeline — one instance per user/vehicle. Zero cloud cost by design:
all inference runs on the user's own hardware; the shared backend only stores
and fans out the results.

    dashcam stream ──► motion gate ──► RT-DETR (fp16 / ONNX) ──► per-class
    confidence filter ──► lite severity (stage-5 formula on CV proxies)
    ──► local dedup ──► POST /api/live/reports (JWT) ──► live map

How it stays cheap while keeping the survey pipeline's behaviour
----------------------------------------------------------------
* SAME detector checkpoint (ml/weights/best.pt), SAME per-class confidence
  thresholds imported from pipeline/detector.py — detection behaviour is
  identical to survey stage 2, just on fewer, well-chosen frames.
* SAME stage-5 severity formula (severity_classifier.classify_box) — only the
  four signals come from ~1 ms classical-CV proxies instead of SAM+Monodepth2
  (see lite_severity.py); scores stay on the calibrated S1–S5 scale and carry
  an honest ~0.5 severity_confidence.
* Motion gating: frames that look like the previous processed frame (car
  stopped at a light, parked) skip inference entirely.
* Frame stride (--every) bounds worst-case compute; fp16 halves GPU cost;
  --imgsz 480 roughly halves RT-DETR FLOPs vs 640 with minimal recall loss on
  near-field damage (the only damage a moving car should report anyway).
* CPU-only users: one-time free ONNX export (--export-onnx [--quantize]) then
  run with --weights ml/weights/best.onnx --device cpu.
* Local dedup: the same pothole seen on 12 consecutive frames posts once.

Scaling model
-------------
Each driver runs THEIR OWN instance of this file. The backend stays a thin,
stateless FastAPI+PostGIS aggregator, so "scaling the pipeline" means users
joining with their own compute — the server's per-report cost is one indexed
ST_DWithin query. Nothing here calls a paid service.

Auth
----
Live reporting requires an account (see backend/routes/auth.py). Options:
  --pair <CODE>                              # code from Live page → Devices panel;
                                             # saves a token to ~/.rids_live_token
  --token <jwt>                              # or env RIDS_TOKEN
  --email x@y.z --password ...               # or env RIDS_EMAIL / RIDS_PASSWORD

Examples
--------
# GPU vehicle, replaying a recorded drive with its GPS track:
python pipeline/live_pipeline.py --video data/raw/footage/drive.mp4 \
    --gps data/raw/gps_logs/drive.gpx --email me@example.com --password ...

# One-time export for CPU-only users (free, local):
python pipeline/live_pipeline.py --export-onnx            # writes best.onnx
python pipeline/live_pipeline.py --video drive.mp4 --lat 46.77 --lon 23.62 \
    --weights ml/weights/best.onnx --device cpu --email ... --password ...
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipeline.detector import CLASS_CONF_THRESHOLDS  # noqa: E402 — single source of truth
from pipeline.lite_severity import assess_box  # noqa: E402

load_dotenv()

_WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "ml/weights"))
_DEFAULT_PT = _WEIGHTS_DIR / os.getenv("RTDETR_WEIGHTS", "best.pt")
_DEFAULT_API = os.getenv("LIVE_API_URL", "http://localhost:8000")
_DEVICE_ID_FILE = Path.home() / ".rids_device_id"
_TOKEN_FILE = Path.home() / ".rids_live_token"   # written by --pair, reused on later runs


# ── Small helpers ───────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def stable_device_id() -> str:
    """Persistent per-machine identity so validation counts stay honest."""
    try:
        if _DEVICE_ID_FILE.exists():
            return _DEVICE_ID_FILE.read_text(encoding="utf-8").strip()
        did = f"edge-{uuid.uuid4().hex[:10]}"
        _DEVICE_ID_FILE.write_text(did, encoding="utf-8")
        return did
    except OSError:
        return f"edge-{uuid.uuid4().hex[:10]}"


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
        t = self.t0 + video_elapsed_s
        pts = self.pts
        if t <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if t >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]
        for (t1, la1, lo1), (t2, la2, lo2) in zip(pts, pts[1:]):
            if t1 <= t <= t2:
                f = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)
                return la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
        return pts[-1][1], pts[-1][2]


def _saved_token() -> str | None:
    """Token persisted by a previous `--pair <CODE>` run (see LivePage → Devices)."""
    try:
        if _TOKEN_FILE.exists():
            return _TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    return None


def pair_with_code(base: str, code: str, device_id: str) -> str:
    """
    Exchange a Live-page pairing code for a JWT — the vehicle machine never
    needs the account password. The token is persisted for later runs.
    """
    r = requests.post(
        f"{base.rstrip('/')}/api/live/devices/claim",
        json={"code": code.strip().upper(), "device_id": device_id},
        timeout=8,
    )
    if r.status_code != 200:
        print(f"[x] Pairing failed ({r.status_code}): {r.text[:200]}")
        raise SystemExit(1)
    data = r.json()
    token = data["access_token"]
    try:
        _TOKEN_FILE.write_text(token, encoding="utf-8")
        saved = f" (token saved to {_TOKEN_FILE})"
    except OSError:
        saved = " (could not persist token — pass --token on future runs)"
    dev = data.get("device") or {}
    print(f"[pair] connected as device '{dev.get('name', device_id)}' [{device_id}]{saved}")
    return token


class LiveApi:
    """Tiny authenticated client for the live endpoints (auto re-login once)."""

    def __init__(self, base: str, token: str | None, email: str | None, password: str | None):
        self.base = base.rstrip("/")
        self.email = email
        self.password = password
        self.token = token or _saved_token() or self._login()

    def _login(self) -> str:
        if not (self.email and self.password):
            print(
                "[x] Live reporting requires an account.\n"
                "    Easiest: open the Live page → Devices → 'Pair a dashcam / PC'\n"
                "    and run:  python pipeline/live_pipeline.py --pair <CODE>\n"
                "    Or pass --token, or --email/--password (or set RIDS_TOKEN / "
                "RIDS_EMAIL / RIDS_PASSWORD in the environment)."
            )
            raise SystemExit(1)
        r = requests.post(
            f"{self.base}/api/auth/login",
            json={"identifier": self.email, "password": self.password},
            timeout=8,
        )
        if r.status_code != 200:
            print(f"[x] Login failed ({r.status_code}): {r.text[:200]}")
            raise SystemExit(1)
        print(f"[auth] logged in as {self.email}")
        return r.json()["access_token"]

    def post_report(self, payload: dict) -> dict | None:
        for attempt in (1, 2):
            try:
                r = requests.post(
                    f"{self.base}/api/live/reports",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=5,
                )
                if r.status_code == 401 and attempt == 1:
                    if self.email:
                        self.token = self._login()   # expired token → one retry
                        continue
                    print(
                        "  [!] token expired/revoked — re-pair from the Live page "
                        "(Devices → Pair a dashcam / PC → --pair <CODE>)"
                    )
                    return None
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                print(f"  [!] report failed: {exc}")
                return None
        return None


# ── Optimisations ───────────────────────────────────────────────────────────

class MotionGate:
    """
    Skip inference on frames that look like the last PROCESSED frame.
    Cost: one 64×36 grayscale diff (~50 µs). A car waiting at a red light
    stops paying for inference entirely.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._last = None

    def should_process(self, frame_bgr) -> bool:
        small = cv2.cvtColor(
            cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype("float32")
        if self._last is None:
            self._last = small
            return True
        diff = float(abs(small - self._last).mean())
        if diff >= self.threshold:
            self._last = small
            return True
        return False


def export_onnx(weights: Path, imgsz: int, quantize: bool) -> int:
    """One-time, local, free: best.pt → best.onnx [→ best.int8.onnx]."""
    from ultralytics import RTDETR
    print(f"[export] {weights} → ONNX (imgsz={imgsz}) …")
    model = RTDETR(str(weights))
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"[export] wrote {onnx_path}")
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q_path = str(Path(onnx_path).with_suffix("")) + ".int8.onnx"
            quantize_dynamic(str(onnx_path), q_path, weight_type=QuantType.QUInt8)
            print(f"[export] wrote {q_path} (dynamic INT8 — ~4× smaller, CPU-friendly;")
            print("          validate accuracy before trusting it: DETR heads can be")
            print("          sensitive to quantisation)")
        except ImportError:
            print("[export] onnxruntime not installed — skip INT8 (pip install onnxruntime)")
        except Exception as exc:  # quantisation is best-effort
            print(f"[export] INT8 quantisation failed: {exc}")
    return 0


# ── Main loop ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="RIDS lite pipeline — per-user live edge agent")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", help="Dashcam video file to replay")
    src.add_argument("--camera", type=int, help="Webcam index (e.g. 0)")
    ap.add_argument("--gps", help=".gpx log matching the video (optional)")
    ap.add_argument("--lat", type=float, help="Fixed latitude when no GPX")
    ap.add_argument("--lon", type=float, help="Fixed longitude when no GPX")

    ap.add_argument("--api", default=_DEFAULT_API, help=f"Backend base URL (default {_DEFAULT_API})")
    ap.add_argument("--device-id", default=None, help="Stable device identity (default: ~/.rids_device_id)")
    ap.add_argument("--token", default=os.getenv("RIDS_TOKEN"), help="JWT (or env RIDS_TOKEN)")
    ap.add_argument("--email", default=os.getenv("RIDS_EMAIL"), help="Account e-mail (or env RIDS_EMAIL)")
    ap.add_argument("--password", default=os.getenv("RIDS_PASSWORD"), help="Account password (or env RIDS_PASSWORD)")

    ap.add_argument("--weights", default=str(_DEFAULT_PT), help="best.pt or an exported .onnx")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    ap.add_argument("--imgsz", type=int, default=480, help="Inference size (480 ≈ half the FLOPs of 640)")
    ap.add_argument("--every", type=int, default=10, help="Consider every Nth frame")
    ap.add_argument("--motion-thresh", type=float, default=2.0,
                    help="Mean gray diff (0-255) below which a frame is 'static' and skipped")
    ap.add_argument("--min-gap-m", type=float, default=20.0, help="Local dedup: metres between same-class reports")
    ap.add_argument("--cooldown-s", type=float, default=30.0, help="Local dedup: seconds between same-class reports nearby")
    ap.add_argument("--realtime", action="store_true", help="Replay at native speed")

    ap.add_argument("--pair", metavar="CODE",
                    help="Pairing code from the Live page (Devices panel); claims a token, "
                         "saves it to ~/.rids_live_token, and exits unless a source is given")
    ap.add_argument("--export-onnx", action="store_true", help="Export best.pt to ONNX and exit")
    ap.add_argument("--quantize", action="store_true", help="With --export-onnx: also write dynamic-INT8 model")
    args = ap.parse_args()

    if args.export_onnx:
        return export_onnx(Path(args.weights), args.imgsz, args.quantize)

    if args.pair:
        pair_with_code(args.api, args.pair, args.device_id or stable_device_id())
        if args.video is None and args.camera is None:
            print("[pair] done — run again with --video/--camera to start reporting.")
            return 0

    if args.video is None and args.camera is None:
        ap.error("Provide --video or --camera (or --export-onnx / --pair).")
    if not args.gps and (args.lat is None or args.lon is None):
        ap.error("Provide --gps, or --lat and --lon for a fixed position.")

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[x] Detector weights not found: {weights}")
        return 1

    api = LiveApi(args.api, args.token, args.email, args.password)
    device_id = args.device_id or stable_device_id()

    # Heavy import late so --help / --export-onnx stay fast
    from ultralytics import RTDETR
    model = RTDETR(str(weights))
    names = model.names
    is_onnx = weights.suffix.lower() == ".onnx"
    use_half = args.device == "cuda" and not is_onnx   # fp16: free 1.5-2× on GPU

    track = GpsTrack(args.gps) if args.gps else None
    cap = cv2.VideoCapture(args.video if args.video is not None else args.camera)
    if not cap.isOpened():
        print("[x] Could not open video source")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gate = MotionGate(args.motion_thresh)
    last_report: dict[str, tuple[float, float, float]] = {}

    print(f"[lite] device_id={device_id} api={args.api}")
    print(f"[lite] weights={weights.name} backend={'onnx' if is_onnx else 'torch'} "
          f"device={args.device} half={use_half} imgsz={args.imgsz} every={args.every}f "
          f"motion>={args.motion_thresh}")

    n_read = n_considered = n_gated = n_inferred = n_sent = 0
    infer_ms = sev_ms = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[lite] end of stream")
                break
            n_read += 1
            frame_idx += 1
            if frame_idx % args.every:
                if args.realtime:
                    time.sleep(1.0 / fps)
                continue
            n_considered += 1

            # ── Optimisation 1: motion gate ────────────────────────────────
            if not gate.should_process(frame):
                n_gated += 1
                continue

            elapsed = frame_idx / fps
            lat, lon = track.at(elapsed) if track else (args.lat, args.lon)

            # ── Optimisation 2: fp16 + reduced imgsz detector pass ─────────
            t0 = time.perf_counter()
            results = model.predict(
                frame, conf=0.001, imgsz=args.imgsz, device=args.device,
                half=use_half, verbose=False,
            )
            infer_ms += (time.perf_counter() - t0) * 1000
            n_inferred += 1

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    # Survey stage-2 behaviour: per-class operational thresholds
                    if conf < CLASS_CONF_THRESHOLDS.get(cls, 0.35):
                        continue

                    # ── Optimisation 3: local per-class dedup ──────────────
                    now = time.time()
                    prev = last_report.get(cls)
                    if prev:
                        pt, plat, plon = prev
                        if (now - pt) < args.cooldown_s and haversine_m(lat, lon, plat, plon) < args.min_gap_m:
                            continue

                    # ── Lite severity: stage-5 formula on ~1 ms proxies ────
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    t1 = time.perf_counter()
                    sev = assess_box(frame, x1, y1, x2, y2, cls)
                    sev_ms += (time.perf_counter() - t1) * 1000

                    resp = api.post_report({
                        "device_id": device_id,
                        "latitude": lat,
                        "longitude": lon,
                        "damage_type": cls,
                        "confidence": round(conf, 3),
                        "severity": sev.severity,
                        "note": f"lite:score={sev.severity_score:.3f},conf={sev.severity_confidence:.2f}",
                    })
                    if resp:
                        last_report[cls] = (now, lat, lon)
                        n_sent += 1
                        ev = resp.get("event") or {}
                        print(
                            f"  [{time.strftime('%H:%M:%S')}] {resp.get('action', '?'):7s} "
                            f"{cls:24s} conf={conf:.2f} {sev.severity_level} "
                            f"→ event {str(ev.get('id'))[:8]} status={ev.get('status')} "
                            f"devices={ev.get('reporter_devices')}"
                        )

            if args.realtime:
                time.sleep(args.every / fps)
    except KeyboardInterrupt:
        print("\n[lite] stopped by user")
    finally:
        cap.release()

    print("\n[lite] ── session summary ─────────────────────────────")
    print(f"  frames read        : {n_read}")
    print(f"  considered (1/{args.every:<3d}) : {n_considered}")
    print(f"  motion-gated       : {n_gated} "
          f"({(100 * n_gated / max(n_considered, 1)):.0f}% of considered — free)")
    print(f"  inference passes   : {n_inferred}"
          + (f"  (avg {infer_ms / max(n_inferred, 1):.1f} ms)" if n_inferred else ""))
    print(f"  severity scoring   : avg {sev_ms / max(n_sent, 1):.2f} ms/box (vs ~130 ms SAM+depth)")
    print(f"  reports sent       : {n_sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
