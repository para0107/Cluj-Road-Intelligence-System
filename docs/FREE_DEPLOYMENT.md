# Deploying RDDS for free — options and recommendations

These are **suggestions only** — nothing here is wired into the code, and none
of it requires a credit card. Ordered by how well each fits this project.

## 0. The built-in answer: don't host the model at all

RDDS is already architected so the **server never runs ML**. The lite pipeline
(`pipeline/live_pipeline.py`) executes RT-DETR on each user's own machine and
posts only tiny JSON reports. That means:

- model "deployment" = distributing a weights file, not running a GPU service;
- your only hosted component is FastAPI + PostGIS + static frontend — the
  cheapest possible shape.

**Distribute the weights free:** GitHub Releases (up to 2 GB per file),
Kaggle Models (already used for `best.pt`), or Hugging Face Hub model repos.
CPU-only users run the free one-time ONNX export
(`live_pipeline.py --export-onnx`).

## 1. Host the web stack (API + DB + frontend) at zero cost

| Option | What you get | Catch |
|---|---|---|
| **Your own PC + Cloudflare Tunnel** | The existing `docker compose` stack becomes publicly reachable over HTTPS via a free `cloudflared` tunnel. Zero changes to the repo. | PC must stay on; free tunnel has no SLA. **Best for a live demo.** |
| **Oracle Cloud Always Free** | Ampere A1 VM (up to 4 OCPU / 24 GB RAM) — comfortably runs Docker + PostGIS + even CPU ONNX inference. Genuinely always-free, not a trial. | Sign-up requires a card for identity (not charged); capacity in EU regions is sometimes scarce — retry. |
| **Hugging Face Spaces (Docker)** | Free CPU Space runs a container — good for a public detector demo (upload image → boxes). | 16 GB disk / 2 vCPU; sleeps when idle; not meant for a persistent PostGIS. |
| **Render free web service** | Managed deploy of the FastAPI image from the repo. | Spins down after ~15 min idle (cold starts); free Postgres expires after 90 days. |
| **Supabase free tier** | Managed Postgres **with PostGIS extension** — can replace the db container if you want a hosted DB. | 500 MB storage; pauses after a week of inactivity. |

Avoid for this project: Fly.io (free allowances now minimal), Railway (trial
credit only), anything GPU-hosted (all paid).

## 2. If you ever must serve inference centrally (still free)

- **Hugging Face Spaces + Gradio**: wrap `RTDETR('best.pt')` in a 30-line
  Gradio app; free CPU inference, public URL, perfect for reviewers.
- **ONNX Runtime on the Oracle free VM**: export once with
  `--export-onnx --quantize`; INT8 CPU inference of RT-DETR-L runs at a few
  hundred ms/image on 4 Ampere cores — fine for an async demo endpoint.
- **Client-side in the browser** (advanced): `onnxruntime-web` (WASM/WebGPU)
  can run the exported ONNX directly in the user's browser — the ultimate
  zero-cost deployment, at the price of a ~60 MB model download.

## 3. Rules of thumb before publishing anything

- Change `JWT_SECRET`, `ADMIN_PASSWORD`, and `POSTGRES_PASSWORD` in `.env`;
  tighten the backend CORS `allow_origins` list.
- Free tiers sleep: keep the polling fallback (already in the Live page) so a
  cold-started API degrades gracefully.
- Respect the Nominatim policy (the landmarks route already rate-limits to
  1 req/s and caches per city forever).
