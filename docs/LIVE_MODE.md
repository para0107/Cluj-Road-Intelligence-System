# RDDS Live Mode — Waze for road damage

Live mode turns RDDS from a batch survey tool into a real-time, crowd-validated
hazard network. It coexists with Survey mode in the same app — the landing page
lets you pick either path.

|                     | **Survey mode** (classic)             | **Live mode** (Waze-like)                     |
|---------------------|---------------------------------------|-----------------------------------------------|
| Input               | .mp4 + .gpx upload                    | Streams from many cameras + user taps         |
| Detection           | RT-DETR-L on the GPU host (7 stages)  | Same `ml/weights/best.pt` on each edge device |
| Validation          | SAM geometry + Monodepth2 + rules     | Multi-device cross-confirmation               |
| Latency             | Minutes–hours per survey              | Milliseconds (WebSocket push)                 |
| Storage             | `detections` (permanent audit)        | `live_events` / `live_reports` (TTL-expiring) |

## How validation works

1. A device (camera agent, simulator vehicle, or a user tapping the map) POSTs a
   sighting to `POST /api/live/reports`.
2. The server clusters it: an active event of the **same damage type within
   `LIVE_CLUSTER_RADIUS_M` (default 25 m)** absorbs the report; otherwise a new
   event is created.
3. Status escalates by **distinct devices** (a device can't self-boost):
   - 1 device → `unverified`
   - ≥ `LIVE_CONFIRM_DEVICES` (2) → `confirmed`
   - ≥ `LIVE_VERIFY_DEVICES` (3) → `verified`
4. Anyone can vote on an event:
   - `POST /api/live/events/{id}/confirm` — "still there" (extends TTL)
   - `POST /api/live/events/{id}/dispute` — "not there"; when distinct
     disputers ≥ max(`LIVE_DISPUTE_MIN`, supporters) the event is removed
   - `POST /api/live/events/{id}/resolve` — operator marks it repaired
5. Events expire after `LIVE_EVENT_TTL_H` (default 72 h) without fresh support —
   stale hazards fall off the map with no cron job (reads sweep lazily).

## Real-time transport

- **WebSocket** `ws://…/api/live/ws` — on connect the server sends
  `{type:"hello", events:[…]}`, then pushes `event_upsert` / `event_removed`
  for every mutation. Nginx is configured for the Upgrade handshake; the Vite
  dev proxy has `ws: true`.
- **Polling fallback** — clients that lose the socket poll
  `GET /api/live/events` every 5 s (the UI switches automatically and shows the
  transport state).

## Scaling notes

- REST handlers are **stateless** — all state is in PostGIS (GIST index on
  event geometry; the cluster lookup is one indexed `ST_DWithin` query).
  Run as many backend replicas as you like behind a load balancer.
- The WS fan-out is per-process by design. To scale horizontally, publish
  mutations to Redis/NATS pub-sub in `live_manager.broadcast()` and have every
  instance's socket loop subscribe and relay (single documented touch point).
- Writes are tiny (one report row + one event upsert per sighting); Postgres
  handles thousands of vehicles without ceremony. Edge agents locally
  de-duplicate (same class within 20 m / 30 s) so a pothole seen on 12
  consecutive frames produces one report, not twelve.

## The lite pipeline — one instance per user, zero cloud cost

`pipeline/live_pipeline.py` is a *softened* version of the 7-stage survey
pipeline that runs on each user's own machine (this is how the system scales:
users bring their own compute; the server is just a stateless aggregator):

| Survey stage | Lite equivalent | Cost |
|---|---|---|
| 2 · RT-DETR detection | same `best.pt`, same per-class thresholds, fp16 @ imgsz 480 (or free ONNX export for CPU) | ~½ the FLOPs of the survey pass |
| 3 · SAM geometry | Otsu-mask proxies with the *same* feature definitions (area / boundary Sobel sharpness / interior-vs-ring contrast / compactness) | ~0.4 ms per box, no VRAM |
| 4 · Monodepth2 | the pipeline's own documented geometry-proxy fallback, verbatim | free |
| 5 · Severity | **unmodified** `severity_classifier.classify_box` — same weights, same S1–S5 bands, marking classes still capped ≤ S2; `severity_confidence` honestly reports ~0.5 (proxy) | free |
| 6 · Dedup | local per-class gap/cooldown + server-side `ST_DWithin` clustering | free |
| 7 · DB write | `POST /api/live/reports` | one indexed query |

Extra inference savings: **motion gating** (a 64×36 gray-diff skips frames
where the car is stationary), frame stride (`--every`), and fp16. A one-time
`--export-onnx [--quantize]` produces a CPU-friendly model for users without
GPUs — all local, all free.

## Accounts & roles

All live actions require a signed-in account (JWT): `user` reports and votes,
`municipality` (bound to a city) and `admin` can additionally resolve events
and manage repairs; admins manage accounts at `/admin`. The starting admin is
seeded at backend startup (`ADMIN_USERNAME`/`ADMIN_EMAIL`/`ADMIN_PASSWORD`).
Edge agents authenticate with `--email/--password`, `--token`, or the
`RDDS_EMAIL`/`RDDS_PASSWORD`/`RDDS_TOKEN` environment variables.

## Running the demo

```bash
# 1. Stack up (backend creates live + auth tables on startup)
docker compose up -d --build

# 2. Sign in at http://localhost:3000 (seed admin: tudypara) and open /live

# 3a. Simulated fleet — no GPU or weights needed
python pipeline/simulate_fleet.py --vehicles 6 --email <you> --password <pw>

# 3b. Real per-user lite pipeline — same baseline model as the survey pipeline
python pipeline/live_pipeline.py --video data/raw/footage/drive.mp4 \
    --gps data/raw/gps_logs/drive.gpx --email <you> --password <pw>

# 3c. CPU-only user (one-time free export, then run on ONNX)
python pipeline/live_pipeline.py --export-onnx
python pipeline/live_pipeline.py --video drive.mp4 --lat 46.77 --lon 23.62 \
    --weights ml/weights/best.onnx --device cpu --email <you> --password <pw>
```

Watch events appear as `unverified`, then flip to `confirmed`/`verified` as
other vehicles drive through the same hotspots. Click an event to vote.

## Configuration (env, all optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LIVE_CLUSTER_RADIUS_M` | 25 | sightings within this radius merge into one event |
| `LIVE_EVENT_TTL_H` | 72 | hours an event survives without fresh support |
| `LIVE_CONFIRM_DEVICES` | 2 | distinct devices → `confirmed` |
| `LIVE_VERIFY_DEVICES` | 3 | distinct devices → `verified` |
| `LIVE_DISPUTE_MIN` | 2 | min distinct disputers to remove an event |
| `JWT_SECRET` | dev value | JWT signing key — change before exposing the API |
| `ADMIN_USERNAME/EMAIL/PASSWORD` | seed admin | starting admin account |
| `GOOGLE_CLIENT_ID` | empty | enables free Google sign-in when set |

See `docs/FREE_DEPLOYMENT.md` for zero-cost hosting options.
