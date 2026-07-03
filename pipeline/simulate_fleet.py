"""
pipeline/simulate_fleet.py

Fleet simulator for Live (Waze-like) mode — no GPU, no weights, no camera.

Spawns N virtual vehicles driving predefined routes through Cluj-Napoca.
The routes share a set of "ground-truth" damage hotspots; when a vehicle
passes near one it reports it (with GPS noise and imperfect recall) to
POST /api/live/reports — exactly what pipeline/live_camera.py does with a
real camera. Because several routes cross the same hotspots, you can watch
events escalate live: unverified → confirmed → verified, as independent
devices re-sight them.

Usage
-----
python pipeline/simulate_fleet.py                          # 4 cars, forever
python pipeline/simulate_fleet.py --vehicles 6 --duration 300
python pipeline/simulate_fleet.py --api http://localhost:8000 --tick 2.0

Open the Live page (http://localhost:3000/live) and watch.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time

import requests

# ── Routes: waypoints along real Cluj-Napoca corridors (approximate) ────────
ROUTES = {
    "east-west": [   # Mănăștur → centre → Mărăști
        (46.7568, 23.5567), (46.7620, 23.5720), (46.7660, 23.5830),
        (46.7694, 23.5899), (46.7700, 23.5990), (46.7712, 23.6100),
        (46.7712, 23.6236), (46.7760, 23.6330), (46.7830, 23.6180),
    ],
    "north-south": [  # Gara CFR → centre → Calea Turzii
        (46.7847, 23.5867), (46.7800, 23.5900), (46.7750, 23.5920),
        (46.7694, 23.5899), (46.7650, 23.5950), (46.7600, 23.6010),
        (46.7550, 23.6070), (46.7500, 23.6120),
    ],
    "ring": [        # Cluj Arena → centre → Iulius Mall → Mărăști
        (46.7686, 23.5725), (46.7694, 23.5899), (46.7712, 23.6236),
        (46.7735, 23.6320), (46.7790, 23.6280), (46.7830, 23.6180),
        (46.7800, 23.6050), (46.7712, 23.6100),
    ],
    "airport": [     # centre → Aeroport
        (46.7712, 23.6236), (46.7760, 23.6330), (46.7800, 23.6500),
        (46.7830, 23.6680), (46.7852, 23.6862),
    ],
}

# ── Ground-truth damage sites — several sit on multiple routes on purpose ───
HOTSPOTS = [
    {"lat": 46.7694, "lon": 23.5899, "type": "pothole",            "severity": 4},
    {"lat": 46.7712, "lon": 23.6236, "type": "alligator_crack",    "severity": 3},
    {"lat": 46.7712, "lon": 23.6100, "type": "pothole",            "severity": 5},
    {"lat": 46.7830, "lon": 23.6180, "type": "rutting",            "severity": 3},
    {"lat": 46.7620, "lon": 23.5720, "type": "longitudinal_crack", "severity": 2},
    {"lat": 46.7760, "lon": 23.6330, "type": "patchy_road",        "severity": 3},
    {"lat": 46.7650, "lon": 23.5950, "type": "transverse_crack",   "severity": 2},
    {"lat": 46.7800, "lon": 23.5900, "type": "pothole",            "severity": 4},
    {"lat": 46.7550, "lon": 23.6070, "type": "manhole_cover",      "severity": 2},
    {"lat": 46.7686, "lon": 23.5725, "type": "lane_line_blur",     "severity": 1},
    {"lat": 46.7830, "lon": 23.6680, "type": "pothole",            "severity": 3},
    {"lat": 46.7735, "lon": 23.6320, "type": "alligator_crack",    "severity": 4},
]

DETECT_RADIUS_M = 35.0    # a camera "sees" a hotspot inside this radius
DETECT_PROB = 0.85        # ...with imperfect recall
GPS_NOISE_M = 7.0         # typical dashcam GPS jitter
REPORT_COOLDOWN_S = 60.0  # per vehicle+hotspot: no machine-gun re-reports


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def jitter(lat: float, lon: float, metres: float) -> tuple[float, float]:
    dn = random.gauss(0, metres)
    de = random.gauss(0, metres)
    return lat + dn / 111_320.0, lon + de / (111_320.0 * math.cos(math.radians(lat)))


class Vehicle(threading.Thread):
    """Drives a route back and forth, reporting hotspots it passes."""

    def __init__(self, vid: str, route: list, api: str, speed_kmh: float, tick_s: float, stop: threading.Event):
        super().__init__(daemon=True, name=vid)
        self.vid = vid
        self.route = route
        self.api = api
        self.speed_ms = speed_kmh / 3.6
        self.tick_s = tick_s
        self.stop_evt = stop
        self.progress = random.random() * (len(route) - 1)   # random start point
        self.direction = 1
        self.last_report: dict[int, float] = {}               # hotspot idx → time

    # Position interpolated along the polyline (progress in segment units)
    def position(self) -> tuple[float, float]:
        i = int(self.progress)
        f = self.progress - i
        la1, lo1 = self.route[i]
        la2, lo2 = self.route[min(i + 1, len(self.route) - 1)]
        return la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)

    def advance(self) -> None:
        i = int(self.progress)
        la1, lo1 = self.route[i]
        la2, lo2 = self.route[min(i + 1, len(self.route) - 1)]
        seg_len = max(haversine_m(la1, lo1, la2, lo2), 1.0)
        self.progress += self.direction * (self.speed_ms * self.tick_s) / seg_len
        if self.progress >= len(self.route) - 1:
            self.progress = len(self.route) - 1
            self.direction = -1
        elif self.progress <= 0:
            self.progress = 0
            self.direction = 1

    def maybe_report(self) -> None:
        lat, lon = self.position()
        now = time.time()
        for idx, spot in enumerate(HOTSPOTS):
            if haversine_m(lat, lon, spot["lat"], spot["lon"]) > DETECT_RADIUS_M:
                continue
            if now - self.last_report.get(idx, 0) < REPORT_COOLDOWN_S:
                continue
            if random.random() > DETECT_PROB:
                continue
            jlat, jlon = jitter(spot["lat"], spot["lon"], GPS_NOISE_M)
            payload = {
                "device_id": self.vid,
                "latitude": jlat,
                "longitude": jlon,
                "damage_type": spot["type"],
                "confidence": round(random.uniform(0.55, 0.97), 3),
                "severity": max(1, min(5, spot["severity"] + random.choice((-1, 0, 0, 0, 1)))),
            }
            try:
                r = requests.post(f"{self.api}/api/live/reports", json=payload, timeout=5)
                r.raise_for_status()
                data = r.json()
                ev = data.get("event") or {}
                print(
                    f"[{self.vid}] {data.get('action', '?'):7s} {spot['type']:22s} "
                    f"→ status={ev.get('status'):10s} devices={ev.get('reporter_devices')}"
                )
                self.last_report[idx] = now
            except requests.RequestException as exc:
                print(f"[{self.vid}] report failed: {exc}")

    def run(self) -> None:
        while not self.stop_evt.is_set():
            self.advance()
            self.maybe_report()
            time.sleep(self.tick_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="RIDS live-mode fleet simulator")
    ap.add_argument("--api", default="http://localhost:8000", help="Backend base URL")
    ap.add_argument("--vehicles", type=int, default=4, help="Number of simulated cars")
    ap.add_argument("--speed", type=float, default=40.0, help="Vehicle speed (km/h)")
    ap.add_argument("--tick", type=float, default=1.5, help="Simulation tick (s)")
    ap.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = run forever)")
    args = ap.parse_args()

    # Fail fast if the API is down
    try:
        requests.get(f"{args.api}/api/live/stats", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        print(f"[x] Live API unreachable at {args.api}: {exc}")
        return 1

    stop = threading.Event()
    route_names = list(ROUTES.keys())
    fleet = [
        Vehicle(
            vid=f"sim-car-{i + 1:02d}",
            route=ROUTES[route_names[i % len(route_names)]],
            api=args.api,
            speed_kmh=args.speed * random.uniform(0.85, 1.15),
            tick_s=args.tick,
            stop=stop,
        )
        for i in range(args.vehicles)
    ]
    for v in fleet:
        v.start()
    print(f"[fleet] {len(fleet)} vehicles driving — open /live and watch. Ctrl+C to stop.")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for v in fleet:
            v.join(timeout=3)
    print("[fleet] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
