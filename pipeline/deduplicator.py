"""
pipeline/deduplicator.py
------------------------
Stage 7 of the road damage detection inference pipeline.

Responsibilities:
  - Accept frames from Stage 6 (enriched.json["frames"])
  - Collect all detections that have valid GPS coordinates
  - Cluster spatially close detections (same physical road damage seen
    across multiple consecutive frames, or revisited in multiple survey
    passes) using DBSCAN on lat/lon with a 2-metre Haversine radius
  - Within each cluster, keep the detection with the highest severity_score
    as the representative instance; all others are marked as duplicates
  - Produce a deduplication report: per-class removal counts, before/after
    totals, and an HTML map showing removed vs retained detections
  - Return a list of DeduplicationResult objects
  - Detections without GPS are forwarded unchanged (no DBSCAN applied)
  - If no GPS-equipped detections are present in the input, the stage
    passes all detections through and logs a warning

DBSCAN parameters:
  eps    : 2 metres (≈ 2e-5 degrees at Romanian latitudes)
           Converts metres to radians for BallTree: 2.0 / 6_371_000.0
  min_samples : 1  — every point is at least a core point; this means
                     even isolated detections are kept. We use DBSCAN
                     here purely for clustering, not for noise removal.

Reference:
  Ester, Kriegel, Sander, Xu — KDD 1996.
  "A Density-Based Algorithm for Discovering Clusters in Large Spatial
  Databases with Noise." ACM KDD 1996. dl.acm.org/doi/10.5555/3001460.3001507

Usage (module):
    from pipeline.deduplicator import Deduplicator, DeduplicatorConfig
    results = Deduplicator(DeduplicatorConfig()).run(enriched_frames)

Usage (CLI):
    python pipeline/deduplicator.py
        --input   data/validation_nrdd_2024/enriched/enriched.json
        --output  data/validation_nrdd_2024/deduplicated/
        [--eps_m 2.0]
        [--verbose]

Author: Paraschiv Tudor -- Babes-Bolyai University, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Earth radius for Haversine / radian conversion
_EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DeduplicatorConfig:
    """
    Parameters for Stage 7 DBSCAN deduplication.

    eps_m:
        DBSCAN epsilon in metres. Two detections within this distance
        are considered neighbours and may be clustered together.
        Default 2.0 m — approximately the width of one lane marking,
        ensuring that the same physical damage seen from slightly different
        positions is collapsed into one record.

    min_samples:
        DBSCAN min_samples. Set to 1 so every GPS-equipped detection is
        assigned to a cluster (no noise points). We use DBSCAN for
        spatial grouping, not outlier removal.

    selection_metric:
        Which field to use for selecting the representative detection
        within each cluster.
        "severity_score" (default) — highest severity wins.
        "confidence"               — highest RT-DETR confidence wins.
    """
    eps_m:            float = 2.0
    min_samples:      int   = 1
    selection_metric: str   = "severity_score"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DedupBox:
    """
    One detection after deduplication.
    raw         : the original enriched box dict (all fields preserved)
    cluster_id  : DBSCAN cluster label (-1 = no GPS / not clustered)
    is_duplicate: True if this detection was suppressed by a better cluster member
    cluster_size: total number of detections in this cluster (1 = no duplicate)
    """
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
    """Stage 7 output: all boxes with dedup metadata, per-frame."""
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
# Deduplicator class
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    Stage 7 — DBSCAN spatial deduplication.

    The DBSCAN clustering operates globally across all frames in a survey
    session (not per-frame), because temporal duplicates of the same
    physical damage appear across consecutive frames as the vehicle passes.

    Steps:
      1. Flatten all GPS-equipped boxes from all frames into one pool.
      2. Run DBSCAN on (lat, lon) with Haversine metric.
      3. Within each cluster, pick the representative with the highest
         severity_score (or confidence if selection_metric="confidence").
         All others are marked is_duplicate=True.
      4. Re-attach results to their source frames.
      5. GPS-missing detections bypass DBSCAN and are forwarded with
         cluster_id=-1, is_duplicate=False, cluster_size=1.
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
        """
        Run deduplication over all enriched frames.

        Parameters
        ----------
        frames     : list of frame dicts from enriched.json["frames"]
        output_dir : if given, saves deduplicated.json and report/ there

        Returns
        -------
        List[DeduplicationResult]
        """
        try:
            from sklearn.cluster import DBSCAN as _DBSCAN
        except ImportError:
            logger.error(
                "scikit-learn is required for Stage 7. "
                "Install with: pip install scikit-learn"
            )
            raise

        # ------------------------------------------------------------------
        # Flatten all detections into a pool
        # ------------------------------------------------------------------
        # pool_entry: (frame_idx, box_idx_in_frame, lat, lon, box_dict)
        pool_gps:    List[Tuple[int, int, float, float, dict]] = []
        pool_nogps:  List[Tuple[int, int, dict]]               = []

        results_shell: List[DeduplicationResult] = []

        for fi, frame in enumerate(frames):
            lat = frame.get("latitude")
            lon = frame.get("longitude")
            gps_ok = lat is not None and lon is not None

            shell = DeduplicationResult(
                frame_path   = frame.get("frame_path", ""),
                frame_index  = frame.get("frame_index", fi),
                timestamp_s  = frame.get("timestamp_s", 0.0),
                latitude     = lat,
                longitude    = lon,
                lighting     = frame.get("lighting", "unknown"),
                image_width  = frame.get("image_width", 640),
                image_height = frame.get("image_height", 360),
                boxes        = [],
            )
            results_shell.append(shell)

            for bi, box in enumerate(frame.get("boxes", [])):
                if gps_ok:
                    pool_gps.append((fi, bi, lat, lon, box))
                else:
                    pool_nogps.append((fi, bi, box))

        n_total  = len(pool_gps) + len(pool_nogps)
        n_gps    = len(pool_gps)
        n_no_gps = len(pool_nogps)

        logger.info(
            "Deduplication pool: %d total detections  "
            "(%d with GPS, %d without GPS)",
            n_total, n_gps, n_no_gps,
        )

        # ------------------------------------------------------------------
        # Attach no-GPS detections immediately (pass-through)
        # ------------------------------------------------------------------
        for fi, bi, box in pool_nogps:
            results_shell[fi].boxes.append(DedupBox(
                raw          = box,
                cluster_id   = -1,
                is_duplicate = False,
                cluster_size = 1,
            ))

        # ------------------------------------------------------------------
        # Early exit: no GPS data at all
        # ------------------------------------------------------------------
        if not pool_gps:
            logger.warning(
                "No GPS-equipped detections found — DBSCAN skipped. "
                "All detections forwarded unchanged. "
                "Use a GPS-synchronised survey run for spatial deduplication."
            )
            # Build placeholder results
            results = results_shell
            if output_dir:
                self._save(results, output_dir, skipped=True)
            return results

        # ------------------------------------------------------------------
        # Run DBSCAN on GPS pool
        # ------------------------------------------------------------------
        coords_rad = np.array(
            [[math.radians(lat), math.radians(lon)] for _, _, lat, lon, _ in pool_gps]
        )
        eps_rad = self.cfg.eps_m / _EARTH_RADIUS_M

        db = _DBSCAN(
            eps            = eps_rad,
            min_samples    = self.cfg.min_samples,
            algorithm      = "ball_tree",
            metric         = "haversine",
        )
        labels = db.fit_predict(coords_rad)

        logger.info(
            "DBSCAN complete: %d unique clusters from %d GPS detections "
            "(eps=%.1f m, min_samples=%d)",
            len(set(labels)), n_gps, self.cfg.eps_m, self.cfg.min_samples,
        )

        # ------------------------------------------------------------------
        # Per-cluster: select best representative
        # ------------------------------------------------------------------
        # cluster_id → list of pool indices
        clusters: Dict[int, List[int]] = defaultdict(list)
        for pool_idx, label in enumerate(labels):
            clusters[label].append(pool_idx)

        # pool_idx → is_duplicate
        is_dup: Dict[int, bool]         = {}
        cluster_sizes: Dict[int, int]    = {}

        for cluster_id, pool_indices in clusters.items():
            cluster_sizes[cluster_id] = len(pool_indices)

            if len(pool_indices) == 1:
                is_dup[pool_indices[0]] = False
                continue

            # Select representative
            best_idx  = pool_indices[0]
            best_val  = self._selection_value(pool_gps[pool_indices[0]][4])

            for pi in pool_indices[1:]:
                val = self._selection_value(pool_gps[pi][4])
                if val > best_val:
                    best_val = val
                    best_idx = pi

            for pi in pool_indices:
                is_dup[pi] = (pi != best_idx)

        # ------------------------------------------------------------------
        # Re-attach GPS detections to their frame shells
        # ------------------------------------------------------------------
        for pool_idx, (fi, bi, lat, lon, box) in enumerate(pool_gps):
            cluster_id = int(labels[pool_idx])
            results_shell[fi].boxes.append(DedupBox(
                raw          = box,
                cluster_id   = cluster_id,
                is_duplicate = is_dup.get(pool_idx, False),
                cluster_size = cluster_sizes.get(cluster_id, 1),
            ))

        results = results_shell

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------
        n_retained  = sum(1 for r in results for b in r.boxes if not b.is_duplicate)
        n_removed   = sum(1 for r in results for b in r.boxes if b.is_duplicate)
        n_clusters  = len([c for c in clusters if cluster_sizes[c] > 1])

        # Per-class removal counts
        class_removed: Dict[str, int]   = defaultdict(int)
        class_retained: Dict[str, int]  = defaultdict(int)

        for r in results:
            for b in r.boxes:
                cls = b.raw.get("class_name", "unknown")
                if b.is_duplicate:
                    class_removed[cls]  += 1
                else:
                    class_retained[cls] += 1

        logger.info("=== Deduplication complete ===")
        logger.info("  Total detections  : %d", n_total)
        logger.info("  Retained          : %d", n_retained)
        logger.info("  Removed (dups)    : %d", n_removed)
        logger.info("  Multi-det clusters: %d", n_clusters)
        logger.info("  Per-class removals:")
        for cls in sorted(class_removed.keys()):
            logger.info(
                "    %-35s  removed=%d  retained=%d",
                cls, class_removed[cls], class_retained.get(cls, 0),
            )

        if output_dir:
            self._save(results, output_dir, skipped=False,
                       class_removed=dict(class_removed),
                       class_retained=dict(class_retained),
                       n_clusters=n_clusters)

        return results

    # ------------------------------------------------------------------
    # Persistence + report
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
        n_removed  = sum(len(r.removed)  for r in results)

        payload = {
            "n_frames":        len(results),
            "n_detections":    n_total,
            "n_retained":      n_retained,
            "n_removed":       n_removed,
            "n_clusters_multi": n_clusters,
            "dbscan_skipped":  skipped,
            "config": {
                "eps_m":            self.cfg.eps_m,
                "min_samples":      self.cfg.min_samples,
                "selection_metric": self.cfg.selection_metric,
            },
            "class_removed":   class_removed or {},
            "class_retained":  class_retained or {},
            "frames":          [r.to_dict() for r in results],
        }

        json_path = out / "deduplicated.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Deduplicated results saved: %s", json_path)

        # Build HTML report
        if not skipped and class_removed:
            report_path = out / "dedup_report.html"
            self._write_html_report(
                report_path, results,
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
        Write a self-contained HTML report with:
          - Summary statistics table
          - Per-class bar chart (before vs after deduplication)
          - Leaflet.js map showing retained (green) and removed (red) detections
        """
        # Collect map points
        retained_pts = []
        removed_pts  = []
        for r in results:
            for b in r.boxes:
                lat = r.latitude
                lon = r.longitude
                if lat is None or lon is None:
                    continue
                cls   = b.raw.get("class_name", "unknown")
                sev   = b.raw.get("severity", {}).get("severity_level", "?") if isinstance(b.raw.get("severity"), dict) else "?"
                label = f"{cls} | {sev} | cluster {b.cluster_id}"
                if b.is_duplicate:
                    removed_pts.append((lat, lon, label))
                else:
                    retained_pts.append((lat, lon, label))

        # Compute chart data
        all_classes = sorted(set(list(class_removed.keys()) + list(class_retained.keys())))
        before_vals  = [class_retained.get(c, 0) + class_removed.get(c, 0) for c in all_classes]
        after_vals   = [class_retained.get(c, 0) for c in all_classes]
        removed_vals = [class_removed.get(c, 0) for c in all_classes]

        # Map centre
        all_lats = [p[0] for p in retained_pts + removed_pts]
        all_lons = [p[1] for p in retained_pts + removed_pts]
        centre_lat = sum(all_lats) / len(all_lats) if all_lats else 46.77
        centre_lon = sum(all_lons) / len(all_lons) if all_lons else 23.59

        def js_points(pts: list, colour: str) -> str:
            lines = []
            for lat, lon, label in pts:
                safe_label = label.replace("'", "\\'")
                lines.append(
                    f"L.circleMarker([{lat},{lon}], "
                    f"{{radius:5, color:'{colour}', fillOpacity:0.8}})"
                    f".bindPopup('{safe_label}').addTo(map);"
                )
            return "\n".join(lines)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RIDS Stage 7 — Deduplication Report</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1   {{ color: #2c3e50; }}
  .stats-table {{ border-collapse: collapse; margin-bottom: 24px; }}
  .stats-table th, .stats-table td {{
    border: 1px solid #bbb; padding: 6px 14px; text-align: left;
  }}
  .stats-table th {{ background: #2c3e50; color: white; }}
  #map  {{ width: 100%; height: 500px; margin-bottom: 24px; border: 1px solid #ccc; }}
  #chart-container {{ max-width: 900px; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>RIDS — Stage 7: Deduplication Report</h1>

<h2>Summary</h2>
<table class="stats-table">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total detections (before)</td><td>{n_total}</td></tr>
  <tr><td>Retained (after dedup)</td><td>{n_retained}</td></tr>
  <tr><td>Removed (duplicates)</td><td>{n_removed}</td></tr>
  <tr><td>Reduction</td><td>{(n_removed/max(n_total,1)*100):.1f}%</td></tr>
  <tr><td>Multi-detection clusters</td><td>{n_clusters}</td></tr>
  <tr><td>DBSCAN eps</td><td>{self.cfg.eps_m} m</td></tr>
  <tr><td>Selection metric</td><td>{self.cfg.selection_metric}</td></tr>
</table>

<h2>Per-Class Breakdown</h2>
<div id="chart-container">
<canvas id="classChart"></canvas>
</div>

<h2>Spatial Map (green = retained, red = removed)</h2>
<div id="map"></div>

<script>
// --- Chart ---
const ctx = document.getElementById('classChart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(all_classes)},
    datasets: [
      {{
        label: 'Retained',
        data: {json.dumps(after_vals)},
        backgroundColor: 'rgba(39,174,96,0.75)'
      }},
      {{
        label: 'Removed (duplicates)',
        data: {json.dumps(removed_vals)},
        backgroundColor: 'rgba(231,76,60,0.75)'
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Detections per class: before vs after deduplication' }} }},
    scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, beginAtZero: true }} }}
  }}
}});

// --- Map ---
const map = L.map('map').setView([{centre_lat}, {centre_lon}], 14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors'
}}).addTo(map);

{js_points(retained_pts, '#27ae60')}
{js_points(removed_pts,  '#e74c3c')}
</script>
</body>
</html>"""

        with path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Deduplication HTML report saved: %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selection_value(self, box: dict) -> float:
        """
        Return the scalar value used to select the representative detection
        within a cluster. Higher is better.
        """
        if self.cfg.selection_metric == "confidence":
            return float(box.get("confidence", 0.0))
        # Default: severity_score
        severity = box.get("severity", {})
        if isinstance(severity, dict):
            return float(severity.get("severity_score", 0.0))
        return 0.0

    @staticmethod
    def load_deduplicated(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        logger.info("Deduplicated data loaded: %s (%d frames)", path, len(payload["frames"]))
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
    parser.add_argument("--eps_m", type=float, default=2.0,
                        help="DBSCAN epsilon in metres (default: 2.0)")
    parser.add_argument("--min_samples", type=int, default=1,
                        help="DBSCAN min_samples (default: 1)")
    parser.add_argument("--selection_metric", default="severity_score",
                        choices=["severity_score", "confidence"],
                        help="How to pick representative in a cluster")
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