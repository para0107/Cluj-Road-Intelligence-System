# End-to-end free deployment — Kaggle + Oracle + Vercel

The exact topology you asked for, verified against how this repo actually
works. Total cost: **0 RON** (Oracle asks for a card for identity only; the
Always Free tier never charges).

```
 Kaggle Models                 Oracle Always Free Ampere VM            Vercel (Hobby)
 ─────────────                 ────────────────────────────            ──────────────
 best.pt (RT-DETR-L)   ─┐      db        PostGIS                       React frontend
 sam2.1_hiera_tiny.pt  ─┤      backend   FastAPI  ◄────────────────►   VITE_API_URL →
                        │      [worker]  survey pipeline (CPU)         the VM's HTTPS
                        ▼                    ▲ HTTPS (DuckDNS+Caddy)
 users download weights ──► pipeline/live_pipeline.py on THEIR machines
                            (per-user lite pipeline → POST /api/live/reports)
```

**Role of each piece — one important clarification:** Kaggle hosts model
*files*, it cannot serve inference. That's exactly what this architecture
needs: Kaggle = free weights CDN; inference runs on each user's machine
(lite pipeline) or on the VM's CPU (survey mode).

---

## Step 1 — Oracle VM (backend + DB, and optionally the survey worker)

1. Create an Oracle Cloud account → **Always Free** resources only.
2. Launch an instance: shape **VM.Standard.A1.Flex**, 4 OCPU / 24 GB
   (the full free allowance), **Ubuntu 22.04**. If you get "out of capacity",
   retry later or switch availability domain — it's a known free-tier quirk.
3. In the VCN's security list, allow ingress TCP **80** and **443**.
4. On the VM:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   sudo usermod -aG docker $USER && newgrp docker
   git clone https://github.com/para0107/pothole-detection.git && cd pothole-detection
   git checkout feature/waze-live-mode
   cp .env.example .env 2>/dev/null || nano .env   # create it (see README) with STRONG values:
   #   POSTGRES_PASSWORD=<random>  JWT_SECRET=<random 64 chars>  ADMIN_PASSWORD=<new>
   docker compose up -d --build
   ```
5. **ARM gotcha:** Ampere is arm64. `python:3.11-slim`, `node:20-alpine` and
   `nginx:alpine` are multi-arch, but if `postgis/postgis:15-3.3` fails with
   *"no matching manifest for linux/arm64"*, switch the db image in
   `docker-compose.yml` to the community arm64 build:
   ```yaml
   image: imresamu/postgis:15-3.3   # arm64 PostGIS, same major versions
   ```
6. Smoke test from your laptop: `curl http://<vm-ip>:8000/health`.

## Step 2 — Free HTTPS (required: browser geolocation only works on HTTPS)

Cheapest **stable** path (no domain purchase):

1. Grab a free subdomain at **duckdns.org** (e.g. `rids.duckdns.org`) pointed
   at the VM's public IP.
2. Run **Caddy** on the VM as the TLS front door (automatic Let's Encrypt):
   ```bash
   sudo apt install -y caddy
   # /etc/caddy/Caddyfile
   rids.duckdns.org {
       reverse_proxy localhost:3000        # nginx frontend (serves SPA + /api proxy + WS)
   }
   sudo systemctl reload caddy
   ```
   The repo's nginx already proxies `/api/` and upgrades `/api/live/ws`, so a
   single hostname gives you the app, the API, and WebSockets.

Alternatives:
- **Cloudflare quick tunnel** (`cloudflared tunnel --url http://localhost:3000`)
  — free, no account, instant HTTPS URL; but the URL is random and rotates on
  restart. Great for a one-off demo, wrong for real users. A *named* tunnel is
  also free but requires a domain you own added to Cloudflare (domains cost
  money — so DuckDNS+Caddy is the truly-free stable option).
- Skip Vercel entirely: with Step 2 done, `https://rids.duckdns.org` **is**
  the full app (the VM's nginx serves the same frontend). Vercel below is
  optional polish (global CDN, preview deploys).

## Step 3 — Vercel frontend (optional, free Hobby tier)

1. vercel.com → Add New Project → import the GitHub repo.
   - Root Directory: `frontend` · Framework: Vite ·
     Build: `npm run build` · Output: `dist` · Production branch:
     `feature/waze-live-mode`.
2. Project → Settings → Environment Variables:
   ```
   VITE_API_URL = https://rids.duckdns.org
   ```
   The frontend reads this at build time (`utils/api.js`); REST **and the
   Live WebSocket** then talk straight to your VM. (Vercel rewrites cannot
   proxy WebSockets, which is why the direct-origin env var exists.)
3. Add `frontend/vercel.json` so React Router deep links work:
   ```json
   { "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
   ```
4. Production hardening on the VM: in `backend/main.py` CORS, replace
   `allow_origins=["*"]` with your Vercel URL + your DuckDNS URL.

## Step 4 — Weights from Kaggle (per machine that runs a pipeline)

- Lite pipeline users need **only** `best.pt`:
  https://www.kaggle.com/models/paraschiv/rt-detr-l-fine-tuned-on-nrdd2024
  → place at `ml/weights/best.pt`. CPU-only users then run the free one-time
  export: `python pipeline/live_pipeline.py --export-onnx` (optionally
  publish the resulting `best.onnx` as a GitHub Release so users skip the
  export).
- The survey pipeline additionally needs
  https://www.kaggle.com/models/paraschiv/sam2-1-hiera-tiny-pt
  (→ `ml/weights/sam2.1_hiera_tiny.pt`) plus the Monodepth2 weights/code —
  exact layout in the README's *Model weights* section.
- Scriptable download: `pip install kaggle` + a (free) API token, then
  `kaggle models instances versions download paraschiv/...`.

## Step 5 — Survey mode processing (pick one)

| Option | How | Trade-off |
|---|---|---|
| **A. On the VM (CPU)** | install host deps in a venv on the VM, put the weights there, set `PIPELINE_DEVICE=cpu` and a low `PIPELINE_FPS` (e.g. 1.0), run `python pipeline/job_watcher.py` under systemd | fully cloud, but slow — fine for overnight batches of short surveys |
| **B. On your GPU PC (hybrid)** | run **Syncthing** (free) to mirror the VM's `data/` folder to your Windows PC; keep `job_watcher.py` running at home as today | uploads land on the VM, sync down, process on your RTX at full speed, results sync back — the file-based job handoff was designed for exactly this kind of decoupling |

Live mode needs neither — it's pure REST/WS from the users' machines.

## Step 6 — What a real user does (the end-to-end flow)

1. Opens `https://rids.vercel.app` (or the DuckDNS URL) → **registers**
   (citizen or municipality) → the browser shares location (HTTPS ✓).
2. Sees the live map; taps **Report damage** for manual reports.
3. A driver with a dashcam: clones the repo (or just downloads
   `pipeline/live_pipeline.py` + `lite_severity.py` + `best.pt`), then
   ```bash
   python pipeline/live_pipeline.py --camera 0 --gps track.gpx \
       --api https://rids.duckdns.org --email me@x.ro --password ...
   ```
   Detections are auto-scored (lite severity) and appear on everyone's map in
   real time; other vehicles passing the same spot escalate them to
   confirmed/verified.
4. The municipality account triages `/priority`, resolves fixed hazards; the
   admin (`tudypara` — change the password!) manages roles at `/admin`.

## Pre-launch checklist

- [ ] `.env`: new `POSTGRES_PASSWORD`, `JWT_SECRET`, `ADMIN_PASSWORD`
- [ ] CORS restricted to your two frontend origins
- [ ] HTTPS working (geolocation + clipboard APIs need it)
- [ ] `docker compose down` does NOT use `-v` in your muscle memory (that
      wipes the DB); set up a nightly `pg_dump` cron to the VM disk
- [ ] Nominatim landmark lookups stay rate-limited (already built in)
