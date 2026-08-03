---
inclusion: always
---

# RDDS — Road Degradation Detection System

A municipal road-monitoring platform, not a research demo. It is deployed and has real
users, which is what distinguishes it from a benchmark project.

Detect → triage → plan → repair → verify, end to end:

- **Survey mode**: a 7-stage CV pipeline turns dashcam video into georeferenced,
  severity-scored damage records (RT-DETR-L → SAM 2.1 → Monodepth2 → rule-based
  severity → DBSCAN dedup → PostGIS).
- **Live mode**: a Waze-like crowd layer where phones and dashcams report hazards that
  escalate through distinct-device validation.
- **Municipality suite**: triage, work orders, repair verification.
- **Road Quality Index**: a city-level A–E heatmap for planning.
- **Public API**, **citizen gamification**, and an **in-browser assistant**.

## Naming

The brand is "RDDS — Road Degradation Detection System". Internal keys
(`rids_*` in localStorage, `cluj_monitor_*` containers) are deliberately NOT renamed —
changing them breaks existing sessions and deployments for no benefit.

## Zero-cost is a hard constraint, not a preference

The project runs on free tiers by design. Do not introduce anything that bills:

- No paid model API anywhere, and **especially never in the assistant** — it runs
  entirely in the user's browser, which is the whole point.
- No routing API for work-order planning. Route optimisation is client-side
  (haversine + nearest-neighbour + 2-opt) both because it is free and because sending
  a city's repair schedule to a third party leaks it.
- No paid captcha vendor. `backend/altcha.py` is a stdlib proof-of-work implementation.
- Apple Sign-In is deliberately absent (it requires a paid Apple program). Google OAuth
  is free and is supported.

## User-facing copy

Plain sentences. No em-dashes, no academic references, no marketing voice. This applies
to UI text, API error `detail` strings, and e-mails (`backend/notify.py`, signed
"The RDDS team").

## Design decisions that look like bugs and are not

Each of these has been "fixed" before and had to be reverted.

- **A raw citizen report earns zero points.** Points come only from outcomes a lone
  spammer cannot manufacture: community confirmation, operator promotion, or a verified
  repair. Awarding points on report re-opens the farm.
- **A work order cannot be verified while any item is reopened.** A detection reopens
  when the pipeline sees the damage again after it was signed off. This is the feature
  that makes the product defensible. Do not soften it.
- **A GPS-less pipeline run that finishes `complete` is correct.** With no usable GPS,
  stages 6–7 are skipped and nothing is written to the DB. That is the intended guard,
  not a failure.
- **No credential defaults anywhere.** An unset `JWT_SECRET` generates a random
  per-process secret; an incomplete admin config seeds no admin. Adding fallbacks
  would ship a known-secret deployment.
- **Pipeline enrichment is permanently removed.** No Nominatim, Overpass or Open-Meteo
  calls in the pipeline; no `street_name` / `road_importance` / `weather` columns. GPS
  lat/lon is the only location metadata. (City landmarks in
  `backend/routes/cities.py` are UI-only and are not covered by this ban.)
