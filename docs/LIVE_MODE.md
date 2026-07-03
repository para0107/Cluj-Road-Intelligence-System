# RIDS Live Mode — Waze for road damage

Live mode turns RIDS from a batch survey tool into a real-time, crowd-validated
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

## Running the demo

```bash
# 1. Stack up (backend now creates live tables on startup)
docker compose up -d --build

# 2. Open the Live map
#    http://localhost:3000/live

# 3a. Simulated fleet — no GPU or weights needed
python pipeline/simulate_fleet.py --vehicles 6

# 3b. Real edge camera — same baseline model as the survey pipeline
python pipeline/live_camera.py --video data/raw/footage/drive.mp4 \
    --gps data/raw/gps_logs/drive.gpx --device-id car-01 --device cuda
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
