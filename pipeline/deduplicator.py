"""
pipeline/deduplicator.py
------------------------
Stage 7 of the road damage detection inference pipeline.

Responsibilities:
  - Accept frames from Stage 6 (enriched.json["frames"])
  - Cluster spatially close detections using DBSCAN on lat/lon with a
    Haversine radius read from DEDUP_CLUSTER_RADIUS_M in .env (default 2 m)
  - Within each cluster, keep the detection with the highest severity_score
  - Produce deduplicated.json and an HTML report (before/after map + chart)
  - Detections without GPS are forwarded unchanged (no DBSCAN applied)
  - If no GPS-equipped detections exist, all detections pass through with a
    warning logged — no exception is raised

Environment variables read (from .env):
    DEDUP_CLUSTER_RADIUS_M       — DBSCAN epsilon in metres (default 2)
    SURROUNDING_DENSITY_RADIUS_M — density search radius in metres (default 50)

DBSCAN reference:
  Ester, Kriegel, Sander, Xu — KDD 1996.
  "A Density-Based Algorithm for Discovering Clusters in Large Spatial
  Databases with Noise." dl.acm.org/doi/10.5555/3001460.3001507

Usage (module):
    from pipeline.deduplicator import Deduplicator, DeduplicatorConfig
    results = Deduplicator(DeduplicatorConfig()).run(enriched_frames)

Usage (CLI):
    python pipeline/deduplicator.py
        --input   data/validation_nrdd_2024/enriched/enriched.json
        --output  data/validation_nrdd_2024/deduplicated/
        [--verbose]

Author: Paraschiv Tudor, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from .env
# ---------------------------------------------------------------------------
_EARTH_RADIUS_M = 6_371_000.0

_EPS_M_DEFAULT      = float(os.environ.get("DEDUP_CLUSTER_RADIUS_M",       "2.0"))
_DENSITY_RADIUS_M   = float(os.environ.get("SURROUNDING_DENSITY_RADIUS_M", "50.0"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DeduplicatorConfig:
    """
    Parameters for Stage 7 DBSCAN deduplication.

    eps_m:
        DBSCAN epsilon in metres. Read from DEDUP_CLUSTER_RADIUS_M in .env.
        Default 2.0 m — approximately the width of one lane marking, ensuring
        the same physical damage seen from slightly different positions is
        collapsed into one record.

    min_samples:
        DBSCAN min_samples. Default 1 means every detection is a core point
        (no noise removal — DBSCAN is used for grouping only).

    selection_metric:
        Which value determines the representative within a cluster.
        "severity_score" (default) — highest severity wins.
        "confidence"               — highest RT-DETR confidence wins.
    """
    eps_m:            float = _EPS_M_DEFAULT
    min_samples:      int   = 1
    selection_metric: str   = "severity_score"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DedupBox:
    raw:          dict
    cluster_id:   int
    is_duplicate: bool
    cluster_size: int

    def to_dict(self) -> dict:
        d = dict(self.raw)
        d["dedup"] = {
            "cluster_id":   self.cluster_id,
            "is_duplicate": self.is_duplicate,
            "cluster_size": self.cluster_size,
        }
        return d


@dataclass
class DeduplicationResult:
    frame_path:    str
    frame_index:   int
    timestamp_s:   float
    latitude:      Optional[float]
    longitude:     Optional[float]
    lighting:      str
    image_width:   int
    image_height:  int
    boxes:         List[DedupBox] = field(default_factory=list)

    @property
    def retained(self) -> List[DedupBox]:
        return [b for b in self.boxes if not b.is_duplicate]

    @property
    def removed(self) -> List[DedupBox]:
        return [b for b in self.boxes if b.is_duplicate]

    def to_dict(self) -> dict:
        return {
            "frame_path":   self.frame_path,
            "frame_index":  self.frame_index,
            "timestamp_s":  self.timestamp_s,
            "latitude":     self.latitude,
            "longitude":    self.longitude,
            "lighting":     self.lighting,
            "image_width":  self.image_width,
            "image_height": self.image_height,
            "boxes":        [b.to_dict() for b in self.boxes],
        }


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    Stage 7 — DBSCAN spatial deduplication.

    Runs globally across all frames in a session: the same physical damage
    appears across consecutive frames as the vehicle passes, and across
    multiple survey passes. DBSCAN collapses these into one representative
    record per unique damage location.
    """

    def __init__(self, cfg: DeduplicatorConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        frames: List[dict],
        output_dir: Optional[str] = None,
    ) -> List[DeduplicationResult]:
        try:
            from sklearn.cluster import DBSCAN as _DBSCAN
        except ImportError:
            logger.error(
                "scikit-learn is required for Stage 7. "
                "Install with: pip install scikit-learn"
            )
            raise

        # ------------------------------------------------------------------
        # Flatten detections into GPS and no-GPS pools
        # ------------------------------------------------------------------
        pool_gps:   List[Tuple[int, int, float, float, dict]] = []
        pool_nogps: List[Tuple[int, int, dict]]               = []
        shells:     List[DeduplicationResult]                  = []

        for fi, frame in enumerate(frames):
            lat    = frame.get("latitude")
            lon    = frame.get("longitude")
            gps_ok = lat is not None and lon is not None

            shells.append(DeduplicationResult(
                frame_path   = frame.get("frame_path", ""),
                frame_index  = frame.get("frame_index", fi),
                timestamp_s  = frame.get("timestamp_s", 0.0),
                latitude     = lat,
                longitude    = lon,
                lighting     = frame.get("lighting", "unknown"),
                image_width  = frame.get("image_width", 640),
                image_height = frame.get("image_height", 360),
                boxes        = [],
            ))

            for bi, box in enumerate(frame.get("boxes", [])):
                if gps_ok:
                    pool_gps.append((fi, bi, lat, lon, box))
                else:
                    pool_nogps.append((fi, bi, box))

        n_total  = len(pool_gps) + len(pool_nogps)
        logger.info(
            "Deduplication pool: %d total  (%d with GPS, %d without GPS)",
            n_total, len(pool_gps), len(pool_nogps),
        )

        # Attach no-GPS detections immediately as pass-through
        for fi, bi, box in pool_nogps:
            shells[fi].boxes.append(DedupBox(
                raw=box, cluster_id=-1, is_duplicate=False, cluster_size=1,
            ))

        # ------------------------------------------------------------------
        # Early exit: no GPS at all
        # ------------------------------------------------------------------
        if not pool_gps:
            logger.warning(
                "No GPS-equipped detections found — DBSCAN skipped. "
                "All detections forwarded unchanged. "
                "Use a GPS-synchronised survey run for spatial deduplication."
            )
            if output_dir:
                self._save(shells, output_dir, skipped=True)
            return shells

        # ------------------------------------------------------------------
        # DBSCAN on GPS pool
        # ------------------------------------------------------------------
        coords_rad = np.array([
            [math.radians(lat), math.radians(lon)]
            for _, _, lat, lon, _ in pool_gps
        ])
        eps_rad = self.cfg.eps_m / _EARTH_RADIUS_M

        labels = _DBSCAN(
            eps         = eps_rad,
            min_samples = self.cfg.min_samples,
            algorithm   = "ball_tree",
            metric      = "haversine",
        ).fit_predict(coords_rad)

        n_unique = len(set(labels))
        logger.info(
            "DBSCAN: %d GPS detections → %d clusters "
            "(eps=%.1f m, min_samples=%d)",
            len(pool_gps), n_unique, self.cfg.eps_m, self.cfg.min_samples,
        )

        # ------------------------------------------------------------------
        # Per-cluster: pick representative, mark the rest as duplicates
        # ------------------------------------------------------------------
        clusters: Dict[int, List[int]] = defaultdict(list)
        for pool_idx, label in enumerate(labels):
            clusters[label].append(pool_idx)

        is_dup:        Dict[int, bool] = {}
        cluster_sizes: Dict[int, int]  = {}

        for cluster_id, indices in clusters.items():
            cluster_sizes[cluster_id] = len(indices)
            if len(indices) == 1:
                is_dup[indices[0]] = False
                continue
            best_idx = max(indices, key=lambda i: self._score(pool_gps[i][4]))
            for pi in indices:
                is_dup[pi] = (pi != best_idx)

        # Re-attach GPS detections to their frame shells
        for pool_idx, (fi, bi, lat, lon, box) in enumerate(pool_gps):
            shells[fi].boxes.append(DedupBox(
                raw          = box,
                cluster_id   = int(labels[pool_idx]),
                is_duplicate = is_dup.get(pool_idx, False),
                cluster_size = cluster_sizes.get(int(labels[pool_idx]), 1),
            ))

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------
        n_retained = sum(len(r.retained) for r in shells)
        n_removed  = sum(len(r.removed)  for r in shells)
        n_multi    = sum(1 for s in cluster_sizes.values() if s > 1)

        class_removed:  Dict[str, int] = defaultdict(int)
        class_retained: Dict[str, int] = defaultdict(int)
        for r in shells:
            for b in r.boxes:
                cls = b.raw.get("class_name", "unknown")
                (class_removed if b.is_duplicate else class_retained)[cls] += 1

        logger.info("=== Deduplication complete ===")
        logger.info("  Total        : %d", n_total)
        logger.info("  Retained     : %d", n_retained)
        logger.info("  Removed      : %d", n_removed)
        logger.info("  Multi-clusters: %d", n_multi)
        logger.info("  Per-class removals:")
        for cls in sorted(class_removed):
            logger.info(
                "    %-35s  removed=%d  retained=%d",
                cls, class_removed[cls], class_retained.get(cls, 0),
            )

        if output_dir:
            self._save(
                shells, output_dir,
                skipped=False,
                class_removed=dict(class_removed),
                class_retained=dict(class_retained),
                n_clusters=n_multi,
            )

        return shells

    # ------------------------------------------------------------------
    # Persistence + HTML report
    # ------------------------------------------------------------------

    def _save(
        self,
        results: List[DeduplicationResult],
        output_dir: str,
        skipped: bool = False,
        class_removed: Optional[Dict[str, int]] = None,
        class_retained: Optional[Dict[str, int]] = None,
        n_clusters: int = 0,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_total    = sum(len(r.boxes) for r in results)
        n_retained = sum(len(r.retained) for r in results)
        n_removed  = sum(len(r.removed) for r in results)

        payload = {
            "n_frames":          len(results),
            "n_detections":      n_total,
            "n_retained":        n_retained,
            "n_removed":         n_removed,
            "n_clusters_multi":  n_clusters,
            "dbscan_skipped":    skipped,
            "config": {
                "eps_m":            self.cfg.eps_m,
                "min_samples":      self.cfg.min_samples,
                "selection_metric": self.cfg.selection_metric,
            },
            "class_removed":     class_removed or {},
            "class_retained":    class_retained or {},
            "frames":            [r.to_dict() for r in results],
        }

        json_path = out / "deduplicated.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Deduplicated results saved: %s", json_path)

        if not skipped and class_removed:
            self._write_html_report(
                out / "dedup_report.html", results,
                class_removed, class_retained or {},
                n_total, n_retained, n_removed, n_clusters,
            )

    def _write_html_report(
        self,
        path: Path,
        results: List[DeduplicationResult],
        class_removed: Dict[str, int],
        class_retained: Dict[str, int],
        n_total: int,
        n_retained: int,
        n_removed: int,
        n_clusters: int,
    ) -> None:
        """
        Self-contained HTML with:
          - Summary statistics table
          - Chart.js stacked bar: retained vs removed per class
          - Leaflet.js map: green = retained, red = removed
        """
        retained_pts, removed_pts = [], []
        for r in results:
            for b in r.boxes:
                lat, lon = r.latitude, r.longitude
                if lat is None or lon is None:
                    continue
                cls = b.raw.get("class_name", "unknown")
                sev_raw = b.raw.get("severity", {})
                sev = sev_raw.get("severity_level", "?") if isinstance(sev_raw, dict) else "?"
                label = f"{cls} | {sev} | cluster {b.cluster_id}"
                (removed_pts if b.is_duplicate else retained_pts).append(
                    (lat, lon, label)
                )

        all_classes   = sorted(set(list(class_removed) + list(class_retained)))
        before_vals   = [class_retained.get(c, 0) + class_removed.get(c, 0) for c in all_classes]
        after_vals    = [class_retained.get(c, 0) for c in all_classes]
        removed_vals  = [class_removed.get(c, 0)  for c in all_classes]

        all_lats = [p[0] for p in retained_pts + removed_pts]
        all_lons = [p[1] for p in retained_pts + removed_pts]
        clat = sum(all_lats) / len(all_lats) if all_lats else float(os.environ.get("CITY_LAT", "46.7712"))
        clon = sum(all_lons) / len(all_lons) if all_lons else float(os.environ.get("CITY_LON", "23.6236"))

        def _js_markers(pts: list, colour: str) -> str:
            lines = []
            for lat, lon, label in pts:
                safe = label.replace("'", "\\'")
                lines.append(
                    f"L.circleMarker([{lat},{lon}],{{radius:5,color:'{colour}',"
                    f"fillOpacity:0.8}}).bindPopup('{safe}').addTo(map);"
                )
            return "\n".join(lines)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RDDS — Stage 7 Deduplication Report</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5}}
  h1,h2{{color:#2c3e50}}
  table{{border-collapse:collapse;margin-bottom:20px}}
  th,td{{border:1px solid #bbb;padding:6px 14px}}
  th{{background:#2c3e50;color:#fff}}
  #map{{width:100%;height:500px;border:1px solid #ccc;margin-bottom:20px}}
  #chart-box{{max-width:900px;margin-bottom:20px}}
</style>
</head>
<body>
<h1>RDDS — Stage 7: Deduplication Report</h1>
<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total detections (before)</td><td>{n_total}</td></tr>
  <tr><td>Retained (after dedup)</td><td>{n_retained}</td></tr>
  <tr><td>Removed (duplicates)</td><td>{n_removed}</td></tr>
  <tr><td>Reduction</td><td>{n_removed/max(n_total,1)*100:.1f}%</td></tr>
  <tr><td>Multi-detection clusters</td><td>{n_clusters}</td></tr>
  <tr><td>DBSCAN eps</td><td>{self.cfg.eps_m} m</td></tr>
  <tr><td>Selection metric</td><td>{self.cfg.selection_metric}</td></tr>
</table>
<h2>Per-Class Breakdown</h2>
<div id="chart-box"><canvas id="cls-chart"></canvas></div>
<h2>Spatial Map (green = retained, red = removed)</h2>
<div id="map"></div>
<script>
const ctx = document.getElementById('cls-chart').getContext('2d');
new Chart(ctx, {{
  type:'bar',
  data:{{
    labels:{json.dumps(all_classes)},
    datasets:[
      {{label:'Retained',data:{json.dumps(after_vals)},backgroundColor:'rgba(39,174,96,0.75)'}},
      {{label:'Removed',data:{json.dumps(removed_vals)},backgroundColor:'rgba(231,76,60,0.75)'}}
    ]
  }},
  options:{{responsive:true,
    plugins:{{title:{{display:true,text:'Detections per class — before vs after deduplication'}}}},
    scales:{{x:{{stacked:true}},y:{{stacked:true,beginAtZero:true}}}}
  }}
}});
const map = L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OpenStreetMap contributors'}}).addTo(map);
{_js_markers(retained_pts,'#27ae60')}
{_js_markers(removed_pts, '#e74c3c')}
</script>
</body>
</html>"""

        with path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Deduplication HTML report saved: %s", path)

    def _score(self, box: dict) -> float:
        if self.cfg.selection_metric == "confidence":
            return float(box.get("confidence", 0.0))
        sev = box.get("severity", {})
        if isinstance(sev, dict):
            return float(sev.get("severity_score", 0.0))
        return 0.0

    @staticmethod
    def load_deduplicated(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        logger.info(
            "Deduplicated data loaded: %s (%d frames)", path, len(payload["frames"])
        )
        return payload["frames"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7 — DBSCAN spatial deduplication."
    )
    parser.add_argument("--input",  required=True,
                        help="enriched.json from Stage 6")
    parser.add_argument("--output", required=True,
                        help="Output directory for deduplicated.json and report")
    parser.add_argument(
        "--eps_m", type=float,
        default=float(os.environ.get("DEDUP_CLUSTER_RADIUS_M", "2.0")),
        help="DBSCAN epsilon in metres (default from .env: DEDUP_CLUSTER_RADIUS_M)",
    )
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument(
        "--selection_metric", default="severity_score",
        choices=["severity_score", "confidence"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input not found: %s", in_path)
        raise SystemExit(1)

    with in_path.open("r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    frames = enriched_data.get("frames", [])
    logger.info("Loaded %d frames from %s", len(frames), in_path)

    cfg = DeduplicatorConfig(
        eps_m            = args.eps_m,
        min_samples      = args.min_samples,
        selection_metric = args.selection_metric,
    )
    Deduplicator(cfg).run(frames, output_dir=args.output)


if __name__ == "__main__":
    main()