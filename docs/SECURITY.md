# Security hardening — what was fixed and how

This document records the security review of the RDDS codebase and the fixes
applied. Numbers match the original review findings.

## Fixed

### 1. Hardcoded admin credentials (CRITICAL)
**Was:** `backend/main.py` and `docker-compose.yml` shipped real default admin
credentials (username, personal e-mail, and password) as in-repo fallbacks —
anyone reading the public repository could sign in as admin on any deployment
that had not overridden them.
**Now:** there are **no credential defaults anywhere in the repo**. The seed
admin is created at startup **only** when `ADMIN_USERNAME`, `ADMIN_EMAIL` and
`ADMIN_PASSWORD` are all set in `.env` (git-ignored); otherwise seeding is
skipped with a warning. docker-compose passes the variables through empty.
> The old password existed in public git history — treat it as burnt and never
> reuse it anywhere.

### 2. Default JWT signing secret (CRITICAL)
**Was:** `JWT_SECRET` fell back to the string `rids-dev-secret-change-me`,
letting anyone forge admin tokens against an unconfigured deployment.
**Now:** if `JWT_SECRET` is unset **or equals the old known value**, the
backend generates a random per-process secret and logs a loud warning. The
app keeps working, but sessions reset on restart until a real secret is set —
a safe failure mode instead of a forgeable one.

### 3. Uploads buffered fully in RAM (HIGH)
**Was:** `POST /api/ingest/upload` read the whole video into memory
(`await video.read()`) while Nginx allowed bodies up to 10 GB — a single large
upload could OOM the backend container.
**Now:** uploads stream to disk in 8 MB chunks and are capped
(`MAX_UPLOAD_MB`, default 4096; `MAX_GPS_MB`, default 50). Oversized uploads
get **413** and the partial file is deleted.

### 4. Unvalidated `job_id` in filesystem paths (HIGH)
**Was:** `GET /api/ingest/status/{job_id}` interpolated the raw path parameter
into filesystem paths (path-traversal risk, especially for host-run dev
backends on Windows) and required no authentication.
**Now:** `job_id` must match the backend-generated `YYYYMMDD_HHMMSS` pattern
(`400` otherwise) and the endpoint requires a signed-in user.

### 5. Municipality self-registration granted operator power instantly (HIGH)
**Was:** anyone could register a "municipality" account and immediately
resolve hazards and delete detections.
**Now:** municipality registrations (a) must confirm their e-mail address and
(b) are **held for admin approval** — at least one admin must approve them on
the Admin page before the account is created. Admins are notified by e-mail
when a request arrives; the applicant is notified of the decision.

### 7. No login rate limiting (HIGH)
**Was:** unlimited online password guessing against `/api/auth/login`.
**Now:** per **identifier + client IP** sliding window — `LOGIN_MAX_ATTEMPTS`
(5) failures within `LOGIN_WINDOW_S` (900 s) lock the pair out for
`LOGIN_LOCKOUT_S` (900 s) with **429**. Additionally PBKDF2 iterations were
raised from 200 000 to **600 000** (OWASP guidance); existing hashes keep
verifying because the iteration count is stored per hash.

### 11. Registration race → 500 (MEDIUM)
**Was:** two concurrent registrations with the same username/e-mail could both
pass the pre-check; the second `INSERT` died with an unhandled
`IntegrityError` (HTTP 500).
**Now:** every account/pending-registration insert catches `IntegrityError`,
rolls back, and returns a clean **409**.

## Also part of this hardening round

- **E-mail verification before account creation** — accounts live in
  `pending_registrations` until the 6-digit code (30 min TTL, single active
  code, constant-time comparison) is confirmed; only then is the `users` row
  created. When SMTP is not configured (dev), the step is skipped for citizen
  accounts; municipality approval still applies. Google sign-ins skip the code
  (Google already verified the e-mail).
- **Role-scoped API** — survey/analytics endpoints (`/detections*`,
  `/heatmap`, `/priority-list`, `/export/csv`, `/ingest/*`) now require the
  municipality/admin role; `/stats` requires any signed-in user; a public
  `/api/health` probe replaced the anonymous use of `/stats`.
- **Account lifecycle** — self-service deletion (`DELETE /auth/me`, password
  re-typed for local accounts), admin delete/disable/enable, and "last active
  admin" protections on both self-deletion and self-demotion.
- **Secrets hygiene** — `.idea/` removed from git tracking; `.gitignore` now
  covers `.env`, `.env.*`, `*.env`, and `.idea/`.

## Second hardening pass (upgrade release)

Added alongside the gamification, municipality and public-API features. All of
it is zero-cost: no third-party service, no new Python dependency.

- **Rate limiting generalized** (`backend/ratelimit.py`). The login limiter was
  extracted into a shared `Limiter` (fixed window, optional lockout, one lock,
  opportunistic pruning) plus a `rate_limited(...)` FastAPI dependency, and is
  now applied to every abusable endpoint, not just login:

  | Endpoint | Budget | Keyed on |
  |---|---|---|
  | `POST /auth/login` | 5 failures / 15 min, then a 15 min lockout | identifier + IP |
  | `POST /auth/register` | 5 / hour | IP |
  | `POST /auth/oauth/google` | 20 / hour | IP |
  | `GET /auth/captcha/challenge` | 30 / min | IP |
  | `POST /live/reports` | 1 / 15 s **and** 40 / day | user |
  | `POST /live/events/{id}/confirm` and `/dispute` | 30 / hour | user |
  | `POST /live/devices/pair` | 10 / hour | user |
  | `POST /live/devices/claim` | 10 / hour | IP |
  | `POST /contact/sales` | 3 / hour | IP |
  | `GET /v1/public/*` | per-key `rate_limit_per_min` (default 60) | API key |

  The report cooldown and daily cap matter more than they look: points make
  spamming attractive, so the economy is defended at the write path.

- **Proof-of-work captcha, self-hosted** (`backend/altcha.py`). Implements the
  open ALTCHA protocol with nothing but the standard library: the browser
  brute-forces a number whose SHA-256 matches a signed challenge. Verification
  checks the hash, the HMAC (constant time, keyed off `JWT_SECRET`), the expiry
  and a minimum solve time, and rejects replays from an in-process seen-set.
  There is no captcha vendor, no outbound call, and it works on a network that
  blocks everything except HTTPS to our own host. Off by default; enable with
  `CAPTCHA_ENABLED=true`. A honeypot field (`website`) backs it up on register
  and on the contact form.

- **Points cannot be farmed.** A raw report earns nothing. Points are granted
  only when an event is validated by *other* devices (confirmed, verified) or
  acted on by the city (promoted, fixed). The ledger has a unique
  `(user_id, reason, ref_id)` constraint, so a replayed or double-fired hook
  inserts nothing.

- **WebSocket hardening** (`/api/live/ws`). Was unauthenticated, uncapped and
  unbounded. Now: an Origin check against `CORS_ORIGINS` (skipped while it is
  `*`), a per-IP connection cap (`LIVE_WS_MAX_PER_IP`, 4) and a total cap
  (`LIVE_WS_MAX_TOTAL`, 2000), and the keep-alive loop drops clients that send
  oversized (>512 char) or too-frequent (>30/min) frames.

- **CORS credentials fixed.** `allow_origins=["*"]` was paired with
  `allow_credentials=True`. Credentials are now enabled only when real origins
  are configured. The app authenticates with a Bearer header, not cookies, so
  nothing depended on it.

- **Security headers.** `backend/middleware.py` sets `X-Content-Type-Options`,
  `X-Frame-Options: DENY`, `Referrer-Policy` and a `Permissions-Policy` on every
  API response, and `frontend/nginx.conf` adds a Content-Security-Policy to the
  SPA document. The CSP allow-list is deliberate and minimal: Google Identity
  (optional login), Google Fonts, unpkg (Leaflet CSS), CARTO/Esri (map tiles),
  Hugging Face (the assistant's model download), and `'wasm-unsafe-eval'` +
  `worker-src blob:` for in-browser inference.

- **Ports closed.** `docker-compose.yml` published Postgres and the API on all
  host interfaces. Both are now bound to `127.0.0.1`, so the only thing reachable
  from the network is Nginx. This also makes the `X-Real-IP` the rate limiter
  trusts unforgeable, since it can only arrive through our own proxy.

- **Unbounded reads bounded.** `/heatmap` returned every row (it now aggregates
  into ~11 m bins in SQL, capped at 20k points); `/export/csv` built the entire
  table in memory (it now streams in batches); the admin user and registration
  lists silently truncated at 500 (they now paginate).

- **Evidence photos are path-safe.** `GET /media/evidence/{id}` resolves the
  stored `crop_path` under `PROJECT_DATA_DIR` and refuses anything absolute,
  containing `..`, escaping the data root, or not ending in `.jpg`. A tampered
  DB value cannot read outside the data tree.

- **API keys are never stored.** Only a SHA-256 hash and a display prefix are
  kept; the plaintext is shown once, at creation. Public endpoints are GET-only
  and expose no user data, no device data and no photos.

## Accepted / known residual risks

- Live-mode device identity is self-declared for anonymous devices —
  distinct-device validation is honest-majority, not Sybil-proof (paired
  devices with revocation mitigate). The captcha raises the cost of minting
  accounts, but a determined Sybil attacker with many accounts and devices can
  still confirm their own report.
- Every limiter, cache and captcha replay-set is **in-process** (reset on
  restart; per-replica when scaled out). This is why the API must stay on ONE
  uvicorn worker, which is also what the WebSocket fan-out requires. Redis is
  the documented upgrade path for both (`backend/live_manager.py`).
- JWTs cannot be revoked before expiry (`JWT_TTL_H`, default 7 days); role
  checks and `is_active` are enforced from the DB on every request, so
  disabling an account takes effect immediately even with a live token.
- CORS remains `allow_origins=["*"]` by default for local development — set
  `CORS_ORIGINS` to the real frontend origin for any public deployment. Note
  the WebSocket Origin check only engages once you do.
- The CSP needs `style-src 'unsafe-inline'` because the app's styling model is
  inline style objects. Removing it would mean rewriting every page.
- A volumetric denial-of-service is out of scope for the application layer. The
  free Cloudflare proxy in front of the host is the documented answer
  (`docs/FREE_DEPLOYMENT.md`); nothing in the app can absorb that traffic.
- `.env` holds all secrets in one file on the deployment host — acceptable for
  a small deployment; use a secret manager beyond that.

## New settings (all optional, safe defaults)

| Variable | Default | Meaning |
|---|---|---|
| `CAPTCHA_ENABLED` | `false` | Require the ALTCHA proof-of-work on register, login and contact |
| `CAPTCHA_TTL_S` | `600` | How long a challenge stays valid |
| `CAPTCHA_MIN_SOLVE_S` | `2.0` | Reject solutions returned faster than a human page load |
| `LIVE_WS_MAX_PER_IP` | `4` | WebSocket connections allowed per address |
| `LIVE_WS_MAX_TOTAL` | `2000` | WebSocket connections allowed in total |
| `RL_<NAME>_MAX` / `RL_<NAME>_WINDOW_S` | per limiter | Override any budget in the table above |
| `EVIDENCE_CROPS` | `true` | Let the pipeline save a photo crop per detection |
| `API_KEYS_PER_USER` | `3` | Active developer keys allowed per account |
| `STATS_CACHE_S` | `10` | Micro-cache on `/stats` |
| `RQI_MAX_CELLS` | `5000` | Refuse a quality-grid request larger than this |
| `APP_TIMEZONE` | `Europe/Bucharest` | Local day boundary for reporting streaks |
