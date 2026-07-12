# Session TODO — RDDS upgrade

Working checklist for the current upgrade session. Plan: visual redesign
(ReactBits), citizen gamification, municipality suite, Road Quality Index,
public developer API, pricing page, in-browser assistant, platform hardening
and scaling. Everything is zero-cost: no paid APIs, no new services.

## Done — backend + pipeline

- [x] **WS0 Platform base**: shared rate limiter (`backend/ratelimit.py`),
      security headers middleware (`backend/middleware.py`), CORS credentials
      fix, nginx gzip + immutable asset caching + CSP, loopback-only DB/backend
      ports, WebSocket origin check + per-IP/total connection caps + message
      size/rate caps, `/stats` micro-cache, `/heatmap` SQL aggregation (was
      unbounded), streamed CSV export (was full-table into RAM), paginated
      admin lists, React.lazy routes + vendor chunks, Leaflet canvas renderer.
- [x] **WS1 Schema**: `user_points`, `user_stats`, `user_badges`,
      `notifications`, `work_orders`, `work_order_items`, `api_keys`; new
      columns `detections.crop_path` / `detections.fixed_at`,
      `live_reports.user_id`, `live_events.promoted_detection_id` /
      `dismissed_at`. Models + idempotent startup DDL + `db/init/06–08*.sql`
      + `scripts/setup_db.py` all in sync.
- [x] **WS2 Gamification**: `backend/gamification.py` (points ledger with
      idempotent awards, streaks, badges, notifications), hooks in
      `routes/live.py`, `routes/engagement.py` (`/engagement/me`,
      `/engagement/leaderboard`, `/notifications`, `/notifications/read`).
      Points only on community-validated outcomes; per-user report cooldown
      and daily cap.
- [x] **WS3 Municipality suite**: triage (`/live/triage`, promote, dismiss),
      work orders CRUD + status transitions + reopened-guard
      (`routes/workorders.py`), ops analytics (`routes/analytics.py`),
      evidence crops in `pipeline/db_writer.py` + guarded media route
      (`routes/media.py`).
- [x] **WS4 Road Quality Index**: `routes/quality.py` grid + CSV/GeoJSON export.
- [x] **WS5 Public developer API**: `routes/public_api.py` — API keys (hash
      only, shown once) + `/v1/public/detections|road-quality|stats`.
- [x] **WS6 Anti-bot**: self-hosted ALTCHA proof-of-work (`backend/altcha.py`,
      stdlib only) + honeypot, wired into register/login/contact.
- [x] **WS8 Contact sales**: `routes/contact.py` + `notify.send_contact_sales_email`.
- [x] **Frontend API layer**: all new endpoints wired in `frontend/src/utils/api.js`.
- [x] **Verification**: backend imports clean (61 routes), both CV pipelines
      compile and import under CUDA, `db_writer` dry-run writes an evidence
      crop with a correct data-relative path.

## Remaining — frontend

- [ ] **Foundations**: npm deps, vendored ReactBits (JS-CSS), `useMotionOk`
      hook, `utils/routePlan.js` (haversine + nearest-neighbour + 2-opt),
      `constants.js` additions (work-order statuses, badges, RQI bands).
- [ ] **New pages**: Impact (points/badges/leaderboard), Triage, Work orders
      (+ route planner panel), Quality map, Developers, Pricing, Assistant.
- [ ] **New components**: notifications bell, onboarding tour, budget planner,
      work-order route panel.
- [ ] **Integrations**: routes in `App.jsx`, grouped `Navbar`, MapPage zone →
      work order + evidence photo, PriorityPage create-order + budget,
      StatsPage operations section + monthly report.
- [ ] **Assistant**: knowledge base, hybrid retrieval, intent handlers,
      optional in-browser model (WebGPU), grounding guard.
- [ ] **Final verification**: `npm run build`, `docker compose up -d --build`,
      schema parity, product walk.
