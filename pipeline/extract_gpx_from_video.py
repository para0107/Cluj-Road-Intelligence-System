"""
scripts/extract_gpx_from_video.py
──────────────────────────────────────────────────────────────────────────────
Extract a .gpx file from a GPS-embedded dashcam MP4.

Supports three extraction strategies, tried in priority order:

  1. GoPro GPMF  — Hero 5/6/7/8/9/10/11/12, Max, Session 5 Black, Fusion
                    GPS embedded as GPMF telemetry track in the MP4 moov atom.
                    Requires: gpmf-python (pip install gpmf) OR the bundled
                    struct-based parser (zero extra deps, always available).

  2. Novatek / VIOFO / Blackvue atom
                    GPS stored as a proprietary atom ('GPS ', 'free', 'gps0')
                    inside the MP4 container.
                    Requires: zero extra deps (struct + stdlib only).

  3. exiftool CAMM / generic
                    Falls back to exiftool -ee -p gpx.fmt if the above two
                    strategies yield no points.
                    Requires: exiftool installed and on PATH.

Output
──────
A standard GPX 1.1 file:

    <gpx version="1.1">
      <trk>
        <trkseg>
          <trkpt lat="46.7712" lon="23.6236">
            <ele>350.0</ele>
            <time>2025-06-01T09:12:34.000000Z</time>
            <extensions>
              <speed>8.3</speed>   <!-- m/s -->
            </extensions>
          </trkpt>
          ...
        </trkseg>
      </trk>
    </gpx>

Usage
─────
    # Basic — writes <video_stem>.gpx next to the input file
    python scripts/extract_gpx_from_video.py path/to/survey.mp4

    # Specify output path explicitly
    python scripts/extract_gpx_from_video.py path/to/survey.mp4 --out path/to/output.gpx

    # Force a specific strategy
    python scripts/extract_gpx_from_video.py path/to/survey.mp4 --strategy gpmf
    python scripts/extract_gpx_from_video.py path/to/survey.mp4 --strategy novatek
    python scripts/extract_gpx_from_video.py path/to/survey.mp4 --strategy exiftool

    # Dry-run: print detected points, do not write file
    python scripts/extract_gpx_from_video.py path/to/survey.mp4 --dry_run

RIDS pipeline handoff
─────────────────────
After extraction, drop the video + gpx into the frontend upload form, or call
the orchestrator directly:

    python pipeline/orchestrator.py \\
        --video data/raw/footage/survey.mp4 \\
        --gps   data/raw/gps_logs/survey.gpx \\
        --device cuda

Requirements
────────────
    gpxpy       — pip install gpxpy  (already in requirements.txt)
    gpmf        — pip install gpmf   (optional; strategy 1 has a built-in fallback)
    exiftool    — https://exiftool.org  (optional; only needed for strategy 3)

Chain of imports is always guarded; no strategy silently crashes — each either
succeeds with ≥1 GPS point or raises an explicit ExtractorError with context.
"""

from __future__ import annotations

import argparse
import logging
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from xml.etree.ElementTree import indent  # Python 3.9+

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_gpx")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPSPoint:
    """One timestamped GPS fix."""
    lat:       float
    lon:       float
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None
    speed_mps: Optional[float] = None    # metres per second

    def is_valid(self) -> bool:
        """Reject sentinel zeros and obviously wrong fixes."""
        if self.lat == 0.0 and self.lon == 0.0:
            return False
        if not (-90.0 <= self.lat <= 90.0):
            return False
        if not (-180.0 <= self.lon <= 180.0):
            return False
        return True


class ExtractorError(RuntimeError):
    """Raised when a strategy cannot produce any GPS points."""


# ---------------------------------------------------------------------------
# Strategy 1: GoPro GPMF
# ---------------------------------------------------------------------------

def _extract_gpmf(mp4_path: Path) -> List[GPSPoint]:
    """
    Extract GPS from GoPro GPMF telemetry track.

    Tries the `gpmf` PyPI package first (high-level, well-tested).
    Falls back to a minimal struct-based GPMF stream walker when the package
    is not installed. The struct walker handles the most common GPS5 / GPS9
    keys; it does NOT handle every GPMF key type.

    References
    ──────────
    GPMF spec: https://github.com/gopro/gpmf-parser
    gpmf PyPI: https://pypi.org/project/gpmf/
    """
    log.info("[GPMF] Attempting GoPro GPMF extraction from %s", mp4_path.name)

    # ── Try high-level gpmf package ───────────────────────────────────────
    try:
        import gpmf                                                 # type: ignore
        stream    = gpmf.io.extract_gpmf_stream(str(mp4_path))
        gps_blocks = gpmf.gps.extract_gps_blocks(stream)
        gps_data  = list(map(gpmf.gps.parse_gps_block, gps_blocks))

        points: List[GPSPoint] = []
        for block in gps_data:
            # Each block is a list of GPS readings
            readings = block if isinstance(block, list) else [block]
            for r in readings:
                try:
                    ts = None
                    if hasattr(r, "timestamp") and r.timestamp is not None:
                        ts = r.timestamp if isinstance(r.timestamp, datetime) \
                             else datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
                    p = GPSPoint(
                        lat=float(r.latitude),
                        lon=float(r.longitude),
                        elevation=float(r.altitude) if hasattr(r, "altitude") and r.altitude is not None else None,
                        timestamp=ts,
                        speed_mps=float(r.speed) if hasattr(r, "speed") and r.speed is not None else None,
                    )
                    if p.is_valid():
                        points.append(p)
                except (AttributeError, TypeError, ValueError):
                    continue

        if points:
            log.info("[GPMF] gpmf package: extracted %d valid points", len(points))
            return points
        log.warning("[GPMF] gpmf package returned 0 valid points; trying struct fallback")

    except ImportError:
        log.debug("[GPMF] gpmf package not installed; using struct fallback")
    except Exception as exc:                                        # noqa: BLE001
        log.warning("[GPMF] gpmf package raised %s; using struct fallback", exc)

    # ── Struct-based GPMF stream walker (zero-dep fallback) ───────────────
    return _extract_gpmf_struct(mp4_path)


def _locate_gpmf_atom(data: bytes) -> Optional[bytes]:
    """
    Walk MP4 atoms looking for the GPMF data track's mdat payload.
    Returns the raw GPMF bytes or None.

    This is a simplified atom walker — it finds 'GoPro MET' or 'tmcd' handler
    atoms but primarily looks for a 'GoPro TCD' or a track whose handler name
    contains 'GoPro MET'.  We extract the raw trak/mdat bytes for that track.
    """
    # The most reliable approach without a full MP4 demuxer is to use ffprobe
    # to identify the GPMF stream index, then ffmpeg to extract it.
    return None   # Handled separately in _extract_gpmf_struct


def _extract_gpmf_struct(mp4_path: Path) -> List[GPSPoint]:
    """
    Use ffmpeg to dump the GoPro metadata track, then parse GPS5/GPS9 KLV.

    Requires ffmpeg on PATH. If ffmpeg is absent, raises ExtractorError.
    """
    # Step 1: find the data stream index for the GPMF track
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-of", "json",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        raise ExtractorError(
            "ffprobe not found on PATH. Install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise ExtractorError(f"ffprobe timed out on {mp4_path}")

    import json as _json
    try:
        probe_data = _json.loads(result.stdout)
    except _json.JSONDecodeError:
        raise ExtractorError(f"ffprobe returned invalid JSON for {mp4_path}")

    # Find GPMF stream by codec_tag_string == 'GoPro MET' or handler_name
    gpmf_index: Optional[int] = None
    for s in probe_data.get("streams", []):
        tag = s.get("codec_tag_string", "")
        handler = s.get("tags", {}).get("handler_name", "")
        if "GoPro MET" in tag or "GoPro Met" in handler or "GoPro MET" in handler:
            gpmf_index = s["index"]
            break

    if gpmf_index is None:
        raise ExtractorError(
            f"[GPMF] No GoPro MET data track found in {mp4_path.name}. "
            "This file may not be a GoPro recording."
        )

    # Step 2: extract raw GPMF bytes via ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    extract_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mp4_path),
        "-map", f"0:{gpmf_index}",
        "-c", "copy",
        "-f", "rawvideo",
        str(tmp_path),
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, timeout=120, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ExtractorError(
            f"[GPMF] ffmpeg extraction failed: {exc.stderr.decode(errors='replace')[:300]}"
        )

    raw = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    if not raw:
        raise ExtractorError("[GPMF] ffmpeg returned empty GPMF stream")

    points = _parse_gpmf_bytes(raw)
    if not points:
        raise ExtractorError(
            "[GPMF] Parsed 0 valid GPS points from GPMF stream. "
            "GPS may have had no fix during recording."
        )
    log.info("[GPMF] struct parser: extracted %d valid points", len(points))
    return points


def _parse_gpmf_bytes(data: bytes) -> List[GPSPoint]:
    """
    Minimal KLV parser for the GPMF format.
    Handles GPS5 (lat, lon, alt, speed_2d, speed_3d) and GPS9 keys.

    GPMF KLV layout (each element):
        FourCC  : 4 bytes  — key name (e.g. b'GPS5')
        type    : 1 byte   — 'l' = int32, 'L' = uint32, 'f' = float, etc.
        size    : 1 byte   — bytes per individual value
        repeat  : 2 bytes  — number of values
        data    : size * repeat bytes

    GPS5 stores: [lat * 1e7, lon * 1e7, alt * 100, speed2D * 100, speed3D * 100]
    as int32 * 5, repeat = N fixes in the block.

    Reference: https://github.com/gopro/gpmf-parser/blob/main/README.md
    """
    points: List[GPSPoint] = []
    offset = 0
    n = len(data)

    current_timestamp: Optional[datetime] = None

    while offset + 8 <= n:
        fourcc = data[offset:offset + 4]
        type_char = chr(data[offset + 4])
        size      = data[offset + 5]
        repeat    = struct.unpack_from(">H", data, offset + 6)[0]
        payload_len = size * repeat
        # Align to 4-byte boundary
        total_len   = 8 + payload_len + (4 - payload_len % 4) % 4
        payload     = data[offset + 8: offset + 8 + payload_len]
        offset += total_len

        if fourcc == b"TSMP":
            # Timestamp in microseconds from stream start — not wall-clock,
            # skip for simplicity (we rely on GPSU for wall-clock)
            pass

        elif fourcc == b"GPSU":
            # UTC timestamp string: "YYMMDDHHmmss.sss"
            try:
                ts_str = payload.decode("ascii").strip("\x00")
                current_timestamp = datetime.strptime(ts_str[:15], "%y%m%d%H%M%S.%f").replace(
                    tzinfo=timezone.utc
                )
            except (ValueError, UnicodeDecodeError):
                pass

        elif fourcc == b"GPS5":
            # type 'l' = signed int32, 5 values per fix
            if type_char != "l" or size != 4 * 5:
                continue
            for i in range(repeat):
                base = i * 20
                if base + 20 > len(payload):
                    break
                lat_raw, lon_raw, alt_raw, s2d_raw, _ = struct.unpack_from(
                    ">5l", payload, base
                )
                p = GPSPoint(
                    lat=lat_raw / 1e7,
                    lon=lon_raw / 1e7,
                    elevation=alt_raw / 100.0,
                    timestamp=current_timestamp,
                    speed_mps=s2d_raw / 100.0 if s2d_raw != 0 else None,
                )
                if p.is_valid():
                    points.append(p)

        elif fourcc == b"GPS9":
            # Extended GPS: lat, lon, alt, speed2D, speed3D, hdop, dvop,
            # satellite_count, fix_type  (9 × float or mixed)
            # type 'f' = float32
            if type_char != "f" or size < 9 * 4:
                continue
            for i in range(repeat):
                base = i * size
                if base + 36 > len(payload):
                    break
                vals = struct.unpack_from(">9f", payload, base)
                p = GPSPoint(
                    lat=vals[0],
                    lon=vals[1],
                    elevation=vals[2],
                    timestamp=current_timestamp,
                    speed_mps=vals[3],
                )
                if p.is_valid():
                    points.append(p)

    return points


# ---------------------------------------------------------------------------
# Strategy 2: Novatek / VIOFO atom parser
# ---------------------------------------------------------------------------

# Novatek GPS atom magic bytes and struct layout.
# Reference: https://sergei.nz/extracting-gps-data-from-viofo-a119-and-other-novatek-powered-cameras/
# The GPS atom contains fixed-size 48-byte records:
#   hour(1) min(1) sec(1) year(1) month(1) day(1) active(1) lat_hemi(1)
#   lat(4f) lon_hemi(1) lon(4f) speed(4f) bearing(4f) ??? (4 padding)
#   ... actual layout varies by firmware; we handle the two most common.

_NOVATEK_ATOM_NAMES = (b"GPS ", b"gps0", b"free", b"moov")
_NOVATEK_RECORD_SIZE = 48


def _find_atoms(data: bytes, target: bytes, max_depth: int = 6) -> list[tuple[int, int]]:
    """
    Walk MP4 box hierarchy looking for atoms with a given 4-byte name.
    Returns list of (offset_of_payload, payload_length).
    """
    results = []
    pos = 0
    n   = len(data)

    while pos + 8 <= n and max_depth > 0:
        size_raw = struct.unpack_from(">I", data, pos)[0]
        name     = data[pos + 4: pos + 8]

        if size_raw == 0:
            # Atom extends to end of file
            atom_size = n - pos
        elif size_raw == 1:
            if pos + 16 > n:
                break
            atom_size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_start = pos + 16
        else:
            atom_size     = size_raw
            payload_start = pos + 8

        if atom_size < 8 or pos + atom_size > n:
            pos += max(8, atom_size)
            continue

        if name == target:
            results.append((payload_start, pos + atom_size - payload_start))

        # Recurse into container atoms
        if name in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta"):
            inner = _find_atoms(
                data[payload_start: pos + atom_size],
                target,
                max_depth - 1,
            )
            results.extend(
                (payload_start + off, ln) for off, ln in inner
            )

        pos += atom_size

    return results


def _parse_novatek_record(record: bytes) -> Optional[GPSPoint]:
    """
    Parse one 48-byte Novatek GPS record.
    Returns None if the record is invalid or GPS fix was not active.
    """
    if len(record) < 48:
        return None

    try:
        hour, minute, second, year, month, day, active, lat_hemi = struct.unpack_from(
            "8B", record, 0
        )
    except struct.error:
        return None

    # active == ord('A') means valid fix; ord('V') means void
    if active != ord("A"):
        return None

    try:
        lat_raw = struct.unpack_from(">f", record,  8)[0]
        lon_hemi = record[12]
        lon_raw  = struct.unpack_from(">f", record, 13)[0]
        speed_kn = struct.unpack_from(">f", record, 17)[0]
    except struct.error:
        return None

    # Novatek stores NMEA-style DDmm.mmmm
    lat_deg = int(lat_raw / 100)
    lat_min = lat_raw - lat_deg * 100
    lat     = lat_deg + lat_min / 60.0
    if chr(lat_hemi).upper() == "S":
        lat = -lat

    lon_deg = int(lon_raw / 100)
    lon_min = lon_raw - lon_deg * 100
    lon     = lon_deg + lon_min / 60.0
    if chr(lon_hemi).upper() == "W":
        lon = -lon

    try:
        ts = datetime(
            2000 + year, month, day, hour, minute, second, tzinfo=timezone.utc
        )
    except ValueError:
        ts = None

    p = GPSPoint(
        lat=lat,
        lon=lon,
        elevation=None,
        timestamp=ts,
        speed_mps=speed_kn * 0.514444,   # knots → m/s
    )
    return p if p.is_valid() else None


def _extract_novatek(mp4_path: Path) -> List[GPSPoint]:
    """
    Extract GPS from Novatek/VIOFO GPS atom.
    Reads the full file into memory — suitable for dashcam clips up to ~2 GB.
    For larger files, use strategy 3 (exiftool) instead.
    """
    log.info("[Novatek] Attempting Novatek/VIOFO atom extraction from %s", mp4_path.name)

    file_size = mp4_path.stat().st_size
    if file_size > 2 * 1024 ** 3:
        log.warning(
            "[Novatek] File is %.1f GB; reading entirely into memory. "
            "Consider using --strategy exiftool for large files.",
            file_size / 1024 ** 3,
        )

    try:
        data = mp4_path.read_bytes()
    except OSError as exc:
        raise ExtractorError(f"[Novatek] Cannot read {mp4_path}: {exc}")

    points: List[GPSPoint] = []

    for atom_name in (b"GPS ", b"gps0"):
        hits = _find_atoms(data, atom_name)
        for payload_off, payload_len in hits:
            payload = data[payload_off: payload_off + payload_len]
            n_records = payload_len // _NOVATEK_RECORD_SIZE
            for i in range(n_records):
                record = payload[i * _NOVATEK_RECORD_SIZE: (i + 1) * _NOVATEK_RECORD_SIZE]
                p = _parse_novatek_record(record)
                if p is not None:
                    points.append(p)

    if not points:
        raise ExtractorError(
            f"[Novatek] No valid GPS records found in {mp4_path.name}. "
            "The file may not use a Novatek chipset, or GPS had no fix."
        )

    log.info("[Novatek] Extracted %d valid GPS points", len(points))
    return points


# ---------------------------------------------------------------------------
# Strategy 3: exiftool (generic fallback)
# ---------------------------------------------------------------------------
_EXIFTOOL_GPX_FMT = """\
#------------------------------------------------------------------------------
# File:         gpx.fmt
# Description:  Print GPX (GPS eXchange) data from geotagged file
#------------------------------------------------------------------------------
#[HEAD]<?xml version="1.0" encoding="utf-8"?>
#[HEAD]<gpx version="1.1" creator="ExifTool" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.topografix.com/GPX/1/1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
#[HEAD]<trk>
#[HEAD]<trkseg>
#[BODY]<trkpt lat="${GPSLatitude#}" lon="${GPSLongitude#}"><time>${GPSDateTime;s/^(\d+):(\d+):(\d+) /$1-$2-$3T/}</time></trkpt>
#[TAIL]</trkseg>
#[TAIL]</trk>
#[TAIL]</gpx>
"""

def _extract_exiftool(mp4_path: Path) -> List[GPSPoint]:
    """
    Use exiftool -ee (ExtractEmbedded) to pull GPS data from any supported
    format (CAMM, GPMF, Sony RTMD, DJI, Google CAMM, etc.).
    Parses the emitted GPX and returns a list of GPSPoints.
    Requires exiftool ≥ 12.0 on PATH.
    """
    log.info("[exiftool] Attempting exiftool extraction from %s", mp4_path.name)

    # Write the fmt file to a temp location
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fmt", delete=False, encoding="utf-8"
    ) as fmt_file:
        fmt_file.write(_EXIFTOOL_GPX_FMT)
        fmt_path = Path(fmt_file.name)

    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as out_file:
        gpx_path = Path(out_file.name)

    try:
        cmd = [
            "exiftool", "-ee", "-p", str(fmt_path),
            str(mp4_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        fmt_path.unlink(missing_ok=True)
        gpx_path.unlink(missing_ok=True)
        raise ExtractorError(
            "[exiftool] exiftool not found on PATH. "
            "Install from https://exiftool.org or: sudo apt install exiftool"
        )
    except subprocess.TimeoutExpired:
        fmt_path.unlink(missing_ok=True)
        gpx_path.unlink(missing_ok=True)
        raise ExtractorError(
            f"[exiftool] Timed out after 300s processing {mp4_path.name}"
        )
    finally:
        fmt_path.unlink(missing_ok=True)

    gpx_xml = result.stdout.strip()
    gpx_path.unlink(missing_ok=True)

    if not gpx_xml or "<trkpt" not in gpx_xml:
        raise ExtractorError(
            f"[exiftool] No GPS track points found in {mp4_path.name}. "
            f"exiftool stderr: {result.stderr[:300]}"
        )

    # Parse the GPX XML
    points = _parse_gpx_xml(gpx_xml)
    if not points:
        raise ExtractorError(
            "[exiftool] Parsed 0 valid GPS points from exiftool output."
        )
    log.info("[exiftool] Extracted %d valid GPS points", len(points))
    return points


def _parse_gpx_xml(gpx_xml: str) -> List[GPSPoint]:
    """Parse a GPX 1.1 XML string into a list of GPSPoints."""
    import xml.etree.ElementTree as ET

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    try:
        root = ET.fromstring(gpx_xml)
    except ET.ParseError as exc:
        log.warning("[exiftool] GPX XML parse error: %s", exc)
        return []

    points: List[GPSPoint] = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        try:
            lat = float(trkpt.get("lat", "0"))
            lon = float(trkpt.get("lon", "0"))
        except ValueError:
            continue

        ele_el = trkpt.find("gpx:ele", ns)
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else None

        time_el = trkpt.find("gpx:time", ns)
        ts: Optional[datetime] = None
        if time_el is not None and time_el.text:
            try:
                ts = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        p = GPSPoint(lat=lat, lon=lon, elevation=ele, timestamp=ts)
        if p.is_valid():
            points.append(p)

    return points


# ---------------------------------------------------------------------------
# Strategy 4: OCR overlay (Vidometer / burned-in GPS text)
# ---------------------------------------------------------------------------
#
# Some dashcam apps (Vidometer, DashCam, AutoBoy, etc.) do NOT embed GPS as
# a metadata track.  Instead they burn the coordinates directly onto every
# video frame as a text overlay, typically at the bottom of the frame in the
# format:
#
#     LAT: 47.048          (or  LAT 47.048 / Lat: 47.048 / N 47.048 …)
#     LONG: 21.955         (or  LON / LNG / E 21.955 …)
#
# This strategy:
#   1. Samples N evenly-spaced frames from the video with OpenCV.
#   2. Crops the bottom strip (configurable; default 20 % of frame height)
#      where overlays live, and optionally the top strip too.
#   3. Runs EasyOCR (preferred) or Tesseract (fallback) on each crop.
#   4. Parses the resulting text with a tolerant regex that handles common
#      label variants and decimal / DMS coordinate formats.
#   5. Recovers the wall-clock timestamp from the overlay when present
#      (Vidometer format: "YYYY-MM-DD HH:MM:SS") or falls back to computing
#      it from the frame index and video FPS.
#
# Requirements
# ────────────
#   easyocr   — pip install easyocr          (preferred; GPU-accelerated)
#   OR
#   pytesseract + Tesseract binary            (fallback; usually slower)
#   opencv-python — pip install opencv-python (always required for this strategy)
#
# Tuning knobs (passed via CLI or programmatically)
# ────────────────────────────────────────────────
#   --ocr_samples     N frames to sample  (default 120)
#   --ocr_strip_frac  fraction of height to crop as overlay strip (default 0.20)
#   --ocr_backend     "easyocr" | "tesseract" | "auto"  (default "auto")

import re as _re

# ── Regex patterns for coordinate parsing ────────────────────────────────────

# Matches a bare coordinate value (decimal or DMS), NO trailing hemisphere.
# Hemisphere handling is done in the caller to avoid ambiguous greedy matches
# (e.g. a standalone compass-bearing "W" between LAT and LONG values).
_DEG_DEC  = r"\d{1,3}(?:[.,]\d+)?"           # decimal degrees, e.g. 47.048
_DEG_DMS  = r"\d{1,3}[°d]\s*\d{1,2}[\'′]\s*\d{1,2}(?:[.,]\d+)?[\"″]?"

_COORD_RE = _re.compile(rf"[+-]?(?:{_DEG_DMS}|{_DEG_DEC})")

# Label prefixes — deliberately loose to handle OCR noise.
# Capture group 1 = hemisphere letter embedded in the label (e.g. "LAT N:").
_LAT_PREFIX = _re.compile(
    r"(?:lat(?:itude)?|lt)\s*[:\-=]?\s*([NnSs]?)\s*",
    _re.IGNORECASE,
)
_LON_PREFIX = _re.compile(
    r"(?:lon(?:g(?:itude)?)?|lng|lo)\s*[:\-=]?\s*([EeWw]?)\s*",
    _re.IGNORECASE,
)

# Vidometer timestamp: "2026-06-10 16:41:05" or "2026-06-10T16:41:05"
_TS_RE = _re.compile(
    r"(\d{4})[/-](\d{2})[/-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2})"
)


def _dms_to_decimal(text: str) -> Optional[float]:
    """Convert a DMS string like '47°02′53.1″' to decimal degrees."""
    m = _re.match(
        r"(\d{1,3})[°d]\s*(\d{1,2})[\'′]\s*(\d{1,2}(?:[.,]\d+)?)[\"″]?",
        text.strip(),
    )
    if not m:
        return None
    d, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3).replace(",", "."))
    return d + mn / 60.0 + s / 3600.0


def _parse_coord_value(raw: str, hemisphere: str) -> Optional[float]:
    """
    Parse a raw coordinate string (decimal or DMS) and apply hemisphere sign.
    Returns None if parsing fails.
    """
    raw = raw.strip().replace(",", ".")
    # Try DMS first
    val = _dms_to_decimal(raw)
    if val is None:
        try:
            val = float(raw)
        except ValueError:
            return None
    if hemisphere.upper() in ("S", "W"):
        val = -val
    return val


def _parse_ocr_text(text: str) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
    """
    Parse a block of OCR text to extract (lat, lon, timestamp).
    All three fields are optional — returns None for any that are absent.

    Handles:
      - "LAT: 47.048  LONG: 21.955"
      - "Lat 47°02'53\" N  Lon 021°57'18\" E"
      - "N47.048 E21.955"  (prefix-only format)
    """
    lat: Optional[float] = None
    lon: Optional[float] = None
    ts:  Optional[datetime] = None

    # ── Timestamp ────────────────────────────────────────────────────────────
    ts_m = _TS_RE.search(text)
    if ts_m:
        try:
            ts = datetime(
                int(ts_m.group(1)), int(ts_m.group(2)), int(ts_m.group(3)),
                int(ts_m.group(4)), int(ts_m.group(5)), int(ts_m.group(6)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass

    # ── Latitude ─────────────────────────────────────────────────────────────
    lat_m = _LAT_PREFIX.search(text)
    if lat_m:
        remainder = text[lat_m.end():]
        coord_m = _COORD_RE.match(remainder)
        if coord_m:
            hemi = lat_m.group(1)
            # Hemisphere may trail the number, but only accept it when it is
            # immediately adjacent AND not followed by ':' or '=' (which would
            # indicate the start of a new label such as "LONG:").
            tail = remainder[coord_m.end():].lstrip()
            if not hemi and tail and tail[0].upper() in ("N", "S"):
                next_ch = tail[1:2]
                if next_ch not in (":", "=") and not next_ch.isalpha():
                    hemi = tail[0]
            lat = _parse_coord_value(coord_m.group(), hemi)

    # ── Longitude ────────────────────────────────────────────────────────────
    lon_m = _LON_PREFIX.search(text)
    if lon_m:
        remainder = text[lon_m.end():]
        coord_m = _COORD_RE.match(remainder)
        if coord_m:
            hemi = lon_m.group(1)
            # Same guard as latitude: only accept trailing hemisphere when the
            # character immediately after is not alphabetic (avoids consuming
            # unrelated words like "VIDOMETER").
            tail = remainder[coord_m.end():].lstrip()
            if not hemi and tail and tail[0].upper() in ("E", "W"):
                next_ch = tail[1:2]
                if next_ch not in (":", "=") and not next_ch.isalpha():
                    hemi = tail[0]
            lon = _parse_coord_value(coord_m.group(), hemi)

    # ── Prefix-only format: N47.048 E21.955 ─────────────────────────────────
    if lat is None or lon is None:
        prefix_re = _re.compile(
            r"([NnSs])\s*(\d{1,3}(?:[.,]\d+)?)"
            r".*?"
            r"([EeWw])\s*(\d{1,3}(?:[.,]\d+)?)",
            _re.DOTALL,
        )
        pm = prefix_re.search(text)
        if pm:
            if lat is None:
                lat = _parse_coord_value(pm.group(2), pm.group(1))
            if lon is None:
                lon = _parse_coord_value(pm.group(4), pm.group(3))

    return lat, lon, ts


def _ocr_frame_easyocr(
    reader,   # easyocr.Reader instance
    crop: "np.ndarray",
) -> str:
    """Run EasyOCR on a single crop, return concatenated text."""
    results = reader.readtext(crop, detail=0, paragraph=False)
    return "  ".join(results)


def _ocr_frame_tesseract(crop: "np.ndarray") -> str:
    """Run pytesseract on a single crop, return text."""
    import pytesseract  # type: ignore
    import PIL.Image    # type: ignore
    import numpy as _np
    rgb = crop[..., ::-1] if crop.ndim == 3 else crop   # BGR → RGB
    pil_img = PIL.Image.fromarray(rgb.astype(_np.uint8))
    return pytesseract.image_to_string(pil_img, config="--psm 6")


def _preprocess_crop(crop: "np.ndarray") -> "np.ndarray":
    """
    Light preprocessing to improve OCR accuracy on dashcam overlays.
    Converts to grayscale, applies adaptive threshold to handle varying
    background brightness (road, sky, etc.).
    """
    import cv2 as _cv2
    import numpy as _np
    gray = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale small crops — OCR engines struggle below ~30px font height
    h, w = gray.shape[:2]
    if h < 80:
        scale = max(2, 80 // h)
        gray = _cv2.resize(gray, (w * scale, h * scale), interpolation=_cv2.INTER_CUBIC)
    # Adaptive threshold: white text on dark background and vice-versa
    binary = _cv2.adaptiveThreshold(
        gray, 255,
        _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        _cv2.THRESH_BINARY, 31, 10,
    )
    # Return both original and inverted — caller concatenates horizontally
    # so the OCR engine sees both polarities in one pass
    combined = _np.hstack([gray, _cv2.bitwise_not(gray)])
    return combined


def _extract_ocr(
    mp4_path: Path,
    n_samples: int = 120,
    strip_frac: float = 0.20,
    backend: str = "auto",
) -> List[GPSPoint]:
    """
    Extract GPS by running OCR on the burned-in overlay of dashcam frames.

    Parameters
    ──────────
    mp4_path   : path to the MP4 file
    n_samples  : number of evenly-spaced frames to sample (default 120)
    strip_frac : fraction of frame height to use as the overlay crop zone,
                 measured from the bottom (default 0.20 = bottom 20 %)
    backend    : "easyocr" | "tesseract" | "auto"
                 "auto" tries easyocr first, falls back to tesseract.

    Returns
    ───────
    List[GPSPoint] with at least one entry, or raises ExtractorError.
    """
    log.info(
        "[OCR] Starting overlay OCR extraction | samples=%d strip=%.0f%% backend=%s",
        n_samples, strip_frac * 100, backend,
    )

    # ── Import OpenCV ────────────────────────────────────────────────────────
    try:
        import cv2  # type: ignore
    except ImportError:
        raise ExtractorError(
            "[OCR] opencv-python is required for the OCR strategy. "
            "Install with: pip install opencv-python"
        )

    # ── Resolve OCR backend ──────────────────────────────────────────────────
    use_easyocr    = False
    use_tesseract  = False
    easyocr_reader = None

    if backend in ("easyocr", "auto"):
        try:
            import easyocr  # type: ignore
            log.info("[OCR] Loading EasyOCR model (first run may download weights)…")
            easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            use_easyocr = True
            log.info("[OCR] EasyOCR ready")
        except ImportError:
            if backend == "easyocr":
                raise ExtractorError(
                    "[OCR] easyocr not installed. Run: pip install easyocr"
                )
            log.debug("[OCR] easyocr not installed; trying tesseract fallback")

    if not use_easyocr and backend in ("tesseract", "auto"):
        try:
            import pytesseract  # type: ignore  # noqa: F401
            pytesseract.get_tesseract_version()   # raises if binary missing
            use_tesseract = True
            log.info("[OCR] Tesseract backend ready")
        except Exception as exc:  # noqa: BLE001
            if backend == "tesseract":
                raise ExtractorError(
                    f"[OCR] Tesseract not available: {exc}. "
                    "Install from https://github.com/tesseract-ocr/tesseract"
                )
            log.debug("[OCR] Tesseract not available: %s", exc)

    if not use_easyocr and not use_tesseract:
        raise ExtractorError(
            "[OCR] No OCR backend available. Install at least one of:\n"
            "  pip install easyocr          # preferred\n"
            "  pip install pytesseract      # + Tesseract binary from https://github.com/tesseract-ocr/tesseract"
        )

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise ExtractorError(f"[OCR] Cannot open video file: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ExtractorError(
            f"[OCR] Could not determine frame count for {mp4_path.name}. "
            "The file may be corrupt or use an unsupported container."
        )

    log.info(
        "[OCR] Video: %d frames, %.2f fps, height=%d px",
        total_frames, fps, frame_h,
    )

    # Sample indices: evenly spaced across the full duration, avoiding the
    # very first and last frames which may be black/transition frames.
    margin   = max(1, total_frames // 50)
    indices  = [
        int(i)
        for i in _np.linspace(margin, total_frames - margin - 1, n_samples)
    ]

    # Strip height in pixels
    strip_px = max(40, int(frame_h * strip_frac))

    # ── Per-frame OCR loop ───────────────────────────────────────────────────
    points:    List[GPSPoint] = []
    n_parsed   = 0
    n_failed   = 0

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            n_failed += 1
            continue

        # Crop bottom strip (primary overlay location for Vidometer et al.)
        crop = frame[frame_h - strip_px:, :]
        processed = _preprocess_crop(crop)

        # OCR
        try:
            if use_easyocr:
                text = _ocr_frame_easyocr(easyocr_reader, processed)
            else:
                text = _ocr_frame_tesseract(processed)
        except Exception as exc:  # noqa: BLE001
            log.debug("[OCR] frame %d OCR error: %s", frame_idx, exc)
            n_failed += 1
            continue

        lat, lon, ts = _parse_ocr_text(text)

        if lat is None or lon is None:
            # Try the top strip too — some apps put overlay at top
            top_crop = frame[:strip_px, :]
            top_processed = _preprocess_crop(top_crop)
            try:
                if use_easyocr:
                    top_text = _ocr_frame_easyocr(easyocr_reader, top_processed)
                else:
                    top_text = _ocr_frame_tesseract(top_processed)
                lat, lon, ts = _parse_ocr_text(top_text)
            except Exception:  # noqa: BLE001
                pass

        if lat is None or lon is None:
            n_failed += 1
            continue

        # Derive timestamp from frame index + fps if overlay had none
        if ts is None:
            ts = datetime.fromtimestamp(frame_idx / fps, tz=timezone.utc)

        p = GPSPoint(lat=lat, lon=lon, timestamp=ts)
        if p.is_valid():
            points.append(p)
            n_parsed += 1

    cap.release()

    log.info(
        "[OCR] Processed %d frames → %d valid points, %d failed/unparsed",
        len(indices), n_parsed, n_failed,
    )

    if not points:
        raise ExtractorError(
            f"[OCR] Could not extract any GPS coordinates from {mp4_path.name} "
            "via OCR. Possible causes:\n"
            "  • The video has no burned-in GPS overlay.\n"
            "  • The overlay uses an unrecognised format (check --verbose output).\n"
            "  • Poor OCR quality — try adjusting --ocr_strip_frac or --ocr_backend."
        )

    # Deduplicate: consecutive identical fixes (static vehicle) collapse to one
    deduped: List[GPSPoint] = [points[0]]
    for p in points[1:]:
        prev = deduped[-1]
        if p.lat != prev.lat or p.lon != prev.lon:
            deduped.append(p)

    log.info("[OCR] After deduplication: %d unique GPS points", len(deduped))
    return deduped


# numpy is needed by _extract_ocr; import lazily to avoid hard dependency
# for users who only use the non-OCR strategies.
try:
    import numpy as _np  # type: ignore
except ImportError:
    _np = None  # type: ignore  # _extract_ocr will fail with a clear error if called


# ---------------------------------------------------------------------------
# GPX writer
# ---------------------------------------------------------------------------

def _build_gpx_xml(points: List[GPSPoint], source_name: str) -> str:
    """
    Render GPSPoint list to a GPX 1.1 XML string.
    Uses only stdlib xml.etree — no gpxpy required.
    """
    import xml.etree.ElementTree as ET

    GPX_NS  = "http://www.topografix.com/GPX/1/1"
    XSI_NS  = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA  = (
        "http://www.topografix.com/GPX/1/1 "
        "http://www.topografix.com/GPX/1/1/gpx.xsd"
    )

    ET.register_namespace("", GPX_NS)
    ET.register_namespace("xsi", XSI_NS)

    gpx = ET.Element(f"{{{GPX_NS}}}gpx")
    gpx.set("version", "1.1")
    gpx.set("creator", f"RIDS extract_gpx_from_video.py — source: {source_name}")
    gpx.set(f"{{{XSI_NS}}}schemaLocation", SCHEMA)

    trk     = ET.SubElement(gpx, f"{{{GPX_NS}}}trk")
    name_el = ET.SubElement(trk, f"{{{GPX_NS}}}name")
    name_el.text = source_name
    trkseg  = ET.SubElement(trk, f"{{{GPX_NS}}}trkseg")

    for p in points:
        trkpt = ET.SubElement(trkseg, f"{{{GPX_NS}}}trkpt")
        trkpt.set("lat", f"{p.lat:.7f}")
        trkpt.set("lon", f"{p.lon:.7f}")

        if p.elevation is not None:
            ele = ET.SubElement(trkpt, f"{{{GPX_NS}}}ele")
            ele.text = f"{p.elevation:.2f}"

        if p.timestamp is not None:
            time_el = ET.SubElement(trkpt, f"{{{GPX_NS}}}time")
            time_el.text = p.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        if p.speed_mps is not None:
            ext = ET.SubElement(trkpt, f"{{{GPX_NS}}}extensions")
            spd = ET.SubElement(ext, f"{{{GPX_NS}}}speed")
            spd.text = f"{p.speed_mps:.3f}"

    # Pretty-print (Python ≥ 3.9)
    try:
        indent(gpx, space="  ")
    except TypeError:
        pass   # Python 3.8 — emit without indentation

    tree = ET.ElementTree(gpx)
    from io import StringIO
    buf = StringIO()
    tree.write(buf, encoding="unicode", xml_declaration=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Orchestrator: try strategies in order
# ---------------------------------------------------------------------------

STRATEGIES = ("gpmf", "novatek", "exiftool", "ocr")


def extract(
    mp4_path: Path,
    strategy: Optional[str] = None,
    ocr_samples: int = 120,
    ocr_strip_frac: float = 0.20,
    ocr_backend: str = "auto",
) -> tuple[List[GPSPoint], str]:
    """
    Try extraction strategies in order until one succeeds.
    Returns (points, strategy_name_that_succeeded).
    Raises ExtractorError with all error messages if all strategies fail.
    """
    if strategy is not None:
        strategies = [strategy]
    else:
        strategies = list(STRATEGIES)

    errors: list[str] = []

    def _ocr_bound(p: Path) -> List[GPSPoint]:
        return _extract_ocr(p, n_samples=ocr_samples, strip_frac=ocr_strip_frac, backend=ocr_backend)

    dispatch = {
        "gpmf":      _extract_gpmf,
        "novatek":   _extract_novatek,
        "exiftool":  _extract_exiftool,
        "ocr":       _ocr_bound,
    }

    for s in strategies:
        if s not in dispatch:
            raise ValueError(
                f"Unknown strategy {s!r}. Valid options: {STRATEGIES}"
            )
        try:
            points = dispatch[s](mp4_path)
            return points, s
        except ExtractorError as exc:
            log.warning("Strategy %r failed: %s", s, exc)
            errors.append(f"[{s}] {exc}")

    raise ExtractorError(
        f"All strategies failed for {mp4_path.name}:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GPS track from a dashcam MP4 and write a .gpx file. "
            "Supports GoPro GPMF, Novatek/VIOFO, and exiftool CAMM / generic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mp4",
        type=Path,
        help="Path to the dashcam .mp4 file.",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help=(
            "Output .gpx path. Default: <mp4_stem>.gpx next to the input file."
        ),
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=STRATEGIES,
        default=None,
        help=(
            "Force a specific extraction strategy. "
            "Default: try gpmf → novatek → exiftool in order."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print detected GPS points; do not write any file.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Set log level to DEBUG.",
    )
    # ── OCR-specific options ──────────────────────────────────────────────────
    ocr_group = parser.add_argument_group(
        "OCR strategy options",
        "Only relevant when --strategy ocr is used (or when all other strategies "
        "fail and ocr is reached in the fallback chain).",
    )
    ocr_group.add_argument(
        "--ocr_samples",
        type=int,
        default=120,
        metavar="N",
        help=(
            "Number of evenly-spaced frames to sample for OCR. "
            "Higher values give denser GPS tracks at the cost of runtime. "
            "Default: 120."
        ),
    )
    ocr_group.add_argument(
        "--ocr_strip_frac",
        type=float,
        default=0.20,
        metavar="F",
        help=(
            "Fraction of frame height to crop as the overlay strip, "
            "measured from the bottom (and top) of the frame. "
            "E.g. 0.20 = bottom 20%%. Default: 0.20."
        ),
    )
    ocr_group.add_argument(
        "--ocr_backend",
        choices=("auto", "easyocr", "tesseract"),
        default="auto",
        help=(
            "OCR engine to use. 'auto' tries easyocr first, then tesseract. "
            "Default: auto."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    mp4_path = args.mp4.resolve()
    if not mp4_path.exists():
        log.error("Input file not found: %s", mp4_path)
        sys.exit(1)
    if not mp4_path.suffix.lower() == ".mp4":
        log.warning(
            "Input file does not have .mp4 extension: %s. Proceeding anyway.",
            mp4_path.name,
        )

    # Run extraction
    try:
        points, used_strategy = extract(
            mp4_path,
            strategy=args.strategy,
            ocr_samples=args.ocr_samples,
            ocr_strip_frac=args.ocr_strip_frac,
            ocr_backend=args.ocr_backend,
        )
    except ExtractorError as exc:
        log.error("Extraction failed:\n%s", exc)
        sys.exit(1)

    log.info(
        "Extraction complete | strategy=%s | points=%d | "
        "time_range=[%s → %s]",
        used_strategy,
        len(points),
        points[0].timestamp.isoformat() if points[0].timestamp else "N/A",
        points[-1].timestamp.isoformat() if points[-1].timestamp else "N/A",
    )

    if args.dry_run:
        print(f"\n{'─'*60}")
        print(f"  DRY RUN — {len(points)} GPS points extracted via [{used_strategy}]")
        print(f"{'─'*60}")
        for i, p in enumerate(points[:10]):
            ts_str = p.timestamp.isoformat() if p.timestamp else "no timestamp"
            spd_str = f"  speed={p.speed_mps:.1f} m/s" if p.speed_mps is not None else ""
            print(f"  [{i:04d}] lat={p.lat:.6f}  lon={p.lon:.6f}  {ts_str}{spd_str}")
        if len(points) > 10:
            print(f"  ... and {len(points) - 10} more points.")
        print(f"{'─'*60}\n")
        return

    # Determine output path
    out_path = args.out
    if out_path is None:
        out_path = mp4_path.with_suffix(".gpx")

    # Write GPX
    gpx_xml = _build_gpx_xml(points, source_name=mp4_path.stem)
    try:
        out_path.write_text(gpx_xml, encoding="utf-8")
    except OSError as exc:
        log.error("Cannot write GPX file %s: %s", out_path, exc)
        sys.exit(1)

    log.info("GPX written → %s  (%d points)", out_path, len(points))

    # Print RIDS handoff hint
    print(
        f"\n✓  Extracted {len(points)} GPS points  [{used_strategy}]"
        f"\n   Output → {out_path}"
        f"\n\nRIDS pipeline handoff:"
        f"\n   python pipeline/orchestrator.py \\"
        f"\n       --video {mp4_path} \\"
        f"\n       --gps   {out_path} \\"
        f"\n       --device cuda\n"
    )


if __name__ == "__main__":
    main()