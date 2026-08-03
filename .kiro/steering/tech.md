---
inclusion: always
---

# Stack and the invariants that break it

**Backend** FastAPI + SQLAlchemy + GeoAlchemy2 on PostGIS · **Frontend** React 18 +
Vite 5 + react-leaflet + Recharts · **ML** PyTorch + Ultralytics (RT-DETR) ·
**Deploy** Docker Compose (db + backend + nginx-served static frontend)

---

## The two-context execution model — understand this before touching the pipeline

The Dockerised backend **cannot run ML**. It has no GPU and no torch;
`backend/requirements.backend.txt` is runtime-only and must stay that way. The pipeline
runs on the **Windows host** in `.venv` with CUDA.

The two sides communicate **only through files** under the shared bind mount
`./data` ⇄ `/app/data`. No sockets, no queues, no shared memory.

1. `POST /api/ingest/upload` writes `data/jobs/<job_id>.json` with container-side paths
2. `pipeline/job_watcher.py` (host daemon) polls, rewrites `/app/data` → the host path,
   and spawns the orchestrator
3. The orchestrator writes `session.json` after every stage
4. `GET /api/ingest/status/{job_id}` reads that file

Adding torch to the backend image, or calling the pipeline over HTTP, breaks the model.
Do not.

`job_id == session_id`, format `YYYYMMDD_HHMMSS`. One job at a time; the upload route
409s while a job is pending or running.

---

## Things that silently break

**A pipeline stage key must stay in sync in three places.** `pipeline/orchestrator.py`
(the stage list), `backend/routes/ingest.py` (the `stages` array in session.json), and
`frontend/src/pages/IngestionPage.jsx` (the `STAGES` / label maps). Miss one and the UI
shows a blank stage with no error.

**Adding a column or index to an existing table means editing three places.**
`create_all` will not alter an existing table. You need (1) the model, (2) an idempotent
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in the `_upgrade_ddl` list in
`main.py::startup_event`, and (3) the matching `db/init/*.sql` for fresh volumes.

**Never add an explicit `Index(..., "geom")` to a GeoAlchemy2 model.** The column
auto-creates `idx_<table>_<col>`; the name collision silently breaks `create_all`.

**The frontend is a static build baked into the nginx image.** Changes under
`frontend/src/**` are invisible until `docker compose up -d --build frontend`.

**Port nuance.** Inside Docker the backend uses `db:5432`; host-run code (pipeline,
db_writer) uses `localhost:${POSTGRES_PORT}`, and `.env` sets `POSTGRES_PORT=5433`.

**Keep ONE uvicorn worker.** Every rate limiter, cache and the WebSocket fan-out is
per-process. Redis pub/sub is the documented multi-worker upgrade path in
`live_manager.py`.

---

## Frontend conventions

- **One global context only**: `AuthContext`. Everything else is page-local hooks. The
  cross-page pipeline signal is `localStorage['rids_active_job']`.
- All pages are `React.lazy` chunks; vendors are split in `vite.config.js`.
- Styling is inline style objects plus CSS design tokens (`--bg`, `--accent`, …) in
  `index.css`. **No Tailwind, no CSS modules.**
- All API calls go through `frontend/src/utils/api.js` (axios, JWT interceptor).
- **Maps never hardcode a city.** They open on the signed-in user's city via
  `hooks/useCityCenter.js`.
- Gate every WebGL/GSAP effect on `useMotionOk()`. `AnimatedContent` sets
  `visibility:hidden` until GSAP reveals it, so rendering it with motion off hides the
  content entirely.
- ReactBits components are **vendored** under `src/reactbits/`, not installed.

---

## Verification

There is **no pytest suite and no linter** for the application code. Verification is
stand-alone scripts run individually (`scripts/validate_severity.py`,
`validate_deduplication.py`, `validate_depth.py`, `validate_db_write.py`).

The assistant is the exception and does have a regression suite: after any edit to
`frontend/src/assistant/knowledge.js`, re-run `npm run embed:kb` then
`npm run test:assistant` (27 offline checks). The KB dense vectors are precomputed and
will silently fall back if stale.
