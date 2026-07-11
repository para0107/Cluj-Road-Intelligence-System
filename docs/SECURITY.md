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

## Accepted / known residual risks

- Live-mode device identity is self-declared for anonymous devices —
  distinct-device validation is honest-majority, not Sybil-proof (paired
  devices with revocation mitigate).
- The login rate limiter and thank-you throttle are in-process (reset on
  restart; per-replica when scaled out). Move to Redis if the API is ever
  replicated.
- JWTs cannot be revoked before expiry (`JWT_TTL_H`, default 7 days); role
  checks and `is_active` are enforced from the DB on every request, so
  disabling an account takes effect immediately even with a live token.
- CORS remains `allow_origins=["*"]` for local development — tighten to the
  real frontend origin for any public deployment.
- `.env` holds all secrets in one file on the deployment host — acceptable for
  a small deployment; use a secret manager beyond that.
