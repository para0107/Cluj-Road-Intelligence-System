"""
scripts/extract_gpx_from_video.py
──────────────────────────────────────────────────────────────────────────────
Extract a .gpx file from a GPS-embedded dashcam MP4.

Supports seven extraction strategies, tried in priority order
(``STRATEGIES`` below). The first that yields ≥1 valid GPS point wins; each
strategy fails soft (raises ExtractorError, caught by ``extract``) so a missing
tool or wrong camera never aborts the chain.

  1. sidecar   — a .gpx / .nmea / .csv / .json / .srt file sharing the video's
                  stem. Pure stdlib. (Mostly for manual/CLI runs — the frontend
                  uploads only the .mp4.)

  2. gpmf      — GoPro (Hero 5+/Max/Fusion). GPMF telemetry track. Uses the
                  `gpmf` pip package if installed, else a built-in recursive KLV
                  parser (ffmpeg to dump the 'gpmd' stream, then SCAL-scaled
                  GPS5/GPS9). Requires ffmpeg for the built-in path.

  3. novatek   — Novatek/VIOFO/Blackvue 'freeGPS' blocks scanned from the MP4.
                  Pure stdlib. Range-validated, fail-soft.

  4. subtitle  — GPS carried in an embedded subtitle stream (DJI, many dashcams)
                  or sidecar already handled above. DJI brackets / DJI GPS() /
                  NMEA $GPRMC|$GPGGA / generic. Requires ffmpeg for embedded.

  5. exiftool  — exiftool -ee over CAMM / Sony RTMD / Garmin / DJI / generic
                  embedded metadata. Requires exiftool on PATH.

  6. iso6709    — single QuickTime ©xyz / ISO 6709 location atom (phone videos).
                  One point for the whole clip. Pure stdlib.

  7. ocr       — burned-in overlay text (Vidometer etc.) read with EasyOCR or
                  Tesseract, parsed (labelled / hemisphere / DMS / bare pair),
                  then geo-filtered. Requires opencv + an OCR backend.

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
    Use ffmpeg to dump the GoPro 'gpmd' metadata track, then parse the GPMF KLV.

    Requires ffmpeg/ffprobe on PATH. If absent, raises ExtractorError.
    """
    probe_cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mp4_path)]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
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

    # GoPro telemetry is a 'data' stream tagged 'gpmd' (handler 'GoPro MET').
    gpmf_index: Optional[int] = None
    for s in probe_data.get("streams", []):
        tag = s.get("codec_tag_string", "")
        handler = s.get("tags", {}).get("handler_name", "")
        if tag == "gpmd" or "GoPro MET" in handler or "GoPro MET" in tag:
            gpmf_index = s["index"]
            break

    if gpmf_index is None:
        raise ExtractorError(
            f"[GPMF] No GoPro 'gpmd' data track found in {mp4_path.name}. "
            "This file may not be a GoPro recording."
        )

    # Dump the raw stream. Use the 'data' muxer (NOT rawvideo, which rejects a
    # non-video stream and silently produces garbage).
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    extract_cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(mp4_path),
        "-map", f"0:{gpmf_index}", "-c", "copy", "-copy_unknown",
        "-f", "data", str(tmp_path),
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, timeout=180, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ExtractorError(
            f"[GPMF] ffmpeg extraction failed: "
            f"{exc.stderr.decode(errors='replace')[:300]}"
        )

    raw = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    if not raw:
        raise ExtractorError("[GPMF] ffmpeg returned an empty GPMF stream")

    points = _parse_gpmf_bytes(raw)
    if not points:
        raise ExtractorError(
            "[GPMF] Parsed 0 valid GPS points from GPMF stream "
            "(GPS may have had no fix during recording)."
        )
    log.info("[GPMF] struct parser: extracted %d valid points", len(points))
    return points


# GPMF scalar type-char → struct format. (Reference: gopro/gpmf-parser.)
_GPMF_NUM_FMT = {
    ord("b"): "b", ord("B"): "B", ord("s"): "h", ord("S"): "H",
    ord("l"): "i", ord("L"): "I", ord("f"): "f", ord("d"): "d",
}


def _gpmf_numbers(type_char: int, payload: bytes) -> Optional[list]:
    """Read a GPMF scalar payload (e.g. a SCAL block) into a list of floats."""
    fmt = _GPMF_NUM_FMT.get(type_char)
    if not fmt:
        return None
    width = struct.calcsize(">" + fmt)
    if width == 0:
        return None
    return [
        float(struct.unpack_from(">" + fmt, payload, i * width)[0])
        for i in range(len(payload) // width)
    ]


def _parse_gpmf_bytes(data: bytes) -> List[GPSPoint]:
    """
    Recursive KLV parser for the GPMF telemetry format.

    Each KLV item:  FourCC(4) · type(1) · sample_size(1) · repeat(2 BE) · payload
    (payload padded to a 4-byte boundary). A ``type`` byte of 0x00 means the
    payload is itself nested KLV — GoPro nests GPS data as DEVC ▸ STRM ▸ {SCAL,
    GPSU, GPS5/GPS9}. SCAL carries the per-stream divisors, GPSU the UTC time.
    The previous flat walker skipped nested containers and never found GPS5;
    this one descends and applies SCAL correctly.

    Handles GPS5 (lat, lon, alt, 2D, 3D — int32 scaled by SCAL) and GPS9
    (best-effort; lat/lon/alt scaled by SCAL).
    """
    points: List[GPSPoint] = []
    _walk_gpmf(data, points, {"scal": None, "ts": None})
    return points


def _walk_gpmf(data: bytes, points: List[GPSPoint], ctx: dict) -> None:
    offset, n = 0, len(data)
    while offset + 8 <= n:
        fourcc = data[offset:offset + 4]
        type_char = data[offset + 4]
        sample_size = data[offset + 5]
        repeat = struct.unpack_from(">H", data, offset + 6)[0]
        payload_len = sample_size * repeat
        payload = data[offset + 8: offset + 8 + payload_len]
        total = 8 + payload_len + ((4 - payload_len % 4) % 4)
        offset += total if total > 8 else 8

        if type_char == 0:                         # nested container
            child = {"scal": None, "ts": ctx["ts"]} if fourcc == b"STRM" else ctx
            _walk_gpmf(payload, points, child)
            ctx["ts"] = child["ts"]
            continue

        if fourcc == b"SCAL":
            ctx["scal"] = _gpmf_numbers(type_char, payload)

        elif fourcc == b"GPSU":
            s = payload.decode("ascii", "ignore").strip("\x00")
            for fmt, ln in (("%y%m%d%H%M%S.%f", 16), ("%y%m%d%H%M%S", 12)):
                try:
                    ctx["ts"] = datetime.strptime(s[:ln], fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue

        elif fourcc in (b"GPS5", b"GPS9"):
            cnt = sample_size // 4
            if cnt < 2:
                continue
            scal = ctx["scal"]

            def _div(idx: int, raw: float) -> float:
                if not scal:
                    return raw / (1e7 if idx < 2 else 1000.0)
                s = scal[idx] if idx < len(scal) else scal[0]
                return raw / s if s else raw

            for i in range(repeat):
                try:
                    vals = struct.unpack_from(f">{cnt}i", payload, i * sample_size)
                except struct.error:
                    break
                lat, lon = _div(0, vals[0]), _div(1, vals[1])
                ele = _div(2, vals[2]) if cnt > 2 else None
                spd = (_div(3, vals[3]) if (fourcc == b"GPS5" and cnt > 3) else None)
                p = GPSPoint(lat=lat, lon=lon, elevation=ele,
                             timestamp=ctx["ts"], speed_mps=spd)
                if p.is_valid():
                    points.append(p)


# ---------------------------------------------------------------------------
# Strategy 2: Novatek / VIOFO atom parser
# ---------------------------------------------------------------------------

# Novatek/VIOFO cameras embed GPS in 'freeGPS' blocks scattered through the MP4.
# Firmware layouts differ, so rather than trust a fixed offset we scan for the
# magic and locate each fix by its signature — the active flag 'A' immediately
# followed by valid N/S and E/W hemisphere bytes — then read the little-endian
# lat/lon/speed floats (NMEA DDmm.mmmm) that follow. Everything is range-checked.
# Reference: https://sergei.nz/extracting-gps-data-from-viofo-a119-and-other-novatek-powered-cameras/


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


def _novatek_coord(raw_deg_min: float, hemi: int) -> Optional[float]:
    """Convert a Novatek NMEA-style DDmm.mmmm float to signed decimal degrees."""
    if raw_deg_min != raw_deg_min or abs(raw_deg_min) > 18000.0:   # NaN / absurd
        return None
    a = abs(raw_deg_min)
    deg = int(a // 100)
    minutes = a - deg * 100
    dec = deg + minutes / 60.0
    if chr(hemi).upper() in ("S", "W"):
        dec = -dec
    return dec


def _parse_freegps_block(block: bytes) -> Optional[GPSPoint]:
    """
    Parse one Novatek 'freeGPS' record. Firmware layouts vary, so we locate the
    fix by its signature — the active flag 'A' immediately followed by valid
    N/S and E/W hemisphere bytes — then read the three little-endian floats
    (lat, lon, speed) that follow. Range-validated, so a wrong guess is rejected.
    """
    for i in range(len(block) - 16):
        if block[i] != 0x41:                                 # 'A' (active fix)
            continue
        lat_h, lon_h = block[i + 1], block[i + 2]
        if lat_h not in (0x4E, 0x53) or lon_h not in (0x45, 0x57):  # N/S, E/W
            continue
        try:
            lat_raw, lon_raw, spd_raw = struct.unpack_from("<3f", block, i + 3)
        except struct.error:
            continue
        lat = _novatek_coord(lat_raw, lat_h)
        lon = _novatek_coord(lon_raw, lon_h)
        if lat is None or lon is None:
            continue
        # Optional UTC: 6 little-endian uint32 (h,m,s,Y,M,D) just before 'A'.
        ts: Optional[datetime] = None
        if i >= 24:
            try:
                hh, mm, ss, yy, mo, dd = struct.unpack_from("<6I", block, i - 24)
                if hh < 24 and mm < 60 and ss < 60 and 1 <= mo <= 12 and 1 <= dd <= 31:
                    ts = datetime(2000 + (yy % 100), mo, dd, hh, mm, ss, tzinfo=timezone.utc)
            except (struct.error, ValueError):
                ts = None
        spd = spd_raw * 0.514444 if 0.0 <= spd_raw < 200.0 else None   # knots → m/s
        p = GPSPoint(lat=lat, lon=lon, timestamp=ts, speed_mps=spd)
        if p.is_valid():
            return p
    return None


def _extract_novatek(mp4_path: Path) -> List[GPSPoint]:
    """
    Extract GPS from a Novatek/VIOFO recording by scanning the file for
    'freeGPS' blocks and parsing each. Reads the file into memory.
    """
    log.info("[Novatek] Attempting Novatek/VIOFO 'freeGPS' extraction from %s", mp4_path.name)
    try:
        data = mp4_path.read_bytes()
    except OSError as exc:
        raise ExtractorError(f"[Novatek] Cannot read {mp4_path}: {exc}")

    points: List[GPSPoint] = []
    magic = b"freeGPS "
    start = 0
    while True:
        idx = data.find(magic, start)
        if idx == -1:
            break
        p = _parse_freegps_block(data[idx: idx + 256])
        if p is not None:
            points.append(p)
        start = idx + len(magic)

    if not points:
        raise ExtractorError(
            f"[Novatek] No 'freeGPS' fixes found in {mp4_path.name}. Not a "
            "Novatek/VIOFO file, GPS had no fix, or the firmware encrypts the block."
        )
    points = _geofilter(points)
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

# A decimal coordinate MUST carry a fractional part of at least this many
# digits. This is the single most important guard in the OCR parser: without
# it the regex happily matches bare integers that share the overlay strip —
# speed ("65"), altitude ("250"), heading ("180"), the year ("2026") — and
# emits them as "coordinates", producing fixes scattered all over the map.
# (That is the root cause of "points spread across Europe instead of Oradea".)
_MIN_FRAC_DIGITS = 3

# A dashcam survey is geographically contiguous. After extraction we drop any
# fix farther than this from the median position — a cheap, format-independent
# outlier filter that removes the residual OCR garbage the parser still lets
# through (a single misread digit yields a wrong-but-in-range coordinate).
_GEOFILTER_RADIUS_KM = 50.0

# Decimal degrees WITH a mandatory fractional part, e.g. 47.0480 or 21,9550.
_DEG_DEC = rf"\d{{1,3}}[.,]\d{{{_MIN_FRAC_DIGITS},}}"
# Degrees-minutes-seconds, e.g. 47°02'53.1"  (spaces tolerated between parts).
_DEG_DMS = r"\d{1,3}\s*[°d]\s*\d{1,2}\s*['′]\s*\d{1,2}(?:[.,]\d+)?\s*[\"″]?"

# A bare coordinate token (DMS preferred over decimal). No hemisphere here.
_COORD_RE = _re.compile(rf"(?:{_DEG_DMS}|{_DEG_DEC})")

# A coordinate token with an OPTIONAL adjacent hemisphere letter on either side.
# Captured so we can tell latitude (N/S) from longitude (E/W) regardless of the
# order they appear in — handles "N47.04 E21.95" and "47.04N 21.95E" alike.
_HEMI_COORD_RE = _re.compile(
    rf"(?P<pre>[NSEWnsew])?\s*(?P<val>{_DEG_DMS}|{_DEG_DEC})"
    rf"\s*(?P<post>[NSEWnsew](?![.,\d]))?"   # trailing hemi must NOT precede another number
)

# Label prefixes (longest alternative first so "longitude" wins over "lon").
# Capture group 1 = hemisphere letter embedded in the label (e.g. "LAT N:").
_LAT_PREFIX = _re.compile(
    r"(?:latitude|lat)\s*[:\-=]?\s*([NnSs]?)\s*",
    _re.IGNORECASE,
)
_LON_PREFIX = _re.compile(
    r"(?:longitude|long|lng|lon)\s*[:\-=]?\s*([EeWw]?)\s*",
    _re.IGNORECASE,
)

# Boundary used to stop a labelled-coordinate search before the NEXT axis label.
# Without it, a garbled latitude value (e.g. EasyOCR reading "47.050" as
# "47.05O") fails to parse and the search reaches forward into the longitude
# field, returning the longitude for BOTH axes — collapsing lat==lon and
# throwing the whole survey to the wrong country.
_NEXT_LABEL_RE = _re.compile(
    r"(?:latitude|longitude|lat|long|lng|lon)", _re.IGNORECASE
)

# Vidometer timestamp: "2026-06-10 16:41:05" or "2026-06-10T16:41:05"
_TS_RE = _re.compile(
    r"(\d{4})[/-](\d{2})[/-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2})"
)


def _dms_to_decimal(text: str) -> Optional[float]:
    """Convert a DMS string like '47°02′53.1″' to decimal degrees."""
    m = _re.match(
        r"\s*(\d{1,3})\s*[°d]\s*(\d{1,2})\s*['′]\s*(\d{1,2}(?:[.,]\d+)?)\s*[\"″]?",
        text,
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


def _labeled_coord(text: str, prefix_re: "_re.Pattern", axis_hemis: tuple) -> Optional[float]:
    """
    Find a coordinate that follows a 'LAT'/'LON' label.

    Searches (not anchors) for the coordinate token in a short window after the
    label, so a stray OCR character between label and number does not break the
    match. The hemisphere sign is taken from the label itself or from a letter
    adjacent to the number, but only when that letter belongs to THIS axis
    (N/S for latitude, E/W for longitude) — never the next field's letter.
    """
    m = prefix_re.search(text)
    if not m:
        return None
    window = text[m.end(): m.end() + 28]
    # Truncate before the next axis label so this field cannot swallow the
    # adjacent one's value when its own (e.g. a mis-OCR'd latitude) won't parse.
    nxt = _NEXT_LABEL_RE.search(window)
    if nxt:
        window = window[:nxt.start()]
    cm = _HEMI_COORD_RE.search(window)
    if not cm:
        return None
    hemi = (m.group(1) or "").upper()
    if not hemi:
        for cand in (cm.group("pre"), cm.group("post")):
            if cand and cand.upper() in axis_hemis:
                hemi = cand.upper()
                break
    return _parse_coord_value(cm.group("val"), hemi if hemi in axis_hemis else "")


def _hemi_scan(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Scan for hemisphere-tagged coordinates anywhere in the text, in any order.
    A token tagged N/S becomes latitude; one tagged E/W becomes longitude.
    Handles 'N47.0480 E21.9550', '47.0480N 21.9550E', etc.
    """
    lat: Optional[float] = None
    lon: Optional[float] = None
    for cm in _HEMI_COORD_RE.finditer(text):
        hemi = (cm.group("pre") or cm.group("post") or "").upper()
        if not hemi:
            continue
        val = _parse_coord_value(cm.group("val"), hemi)
        if val is None:
            continue
        if hemi in ("N", "S") and lat is None:
            lat = val
        elif hemi in ("E", "W") and lon is None:
            lon = val
    return lat, lon


def _bare_pair(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Last-resort parse for an unlabeled decimal pair, e.g. '47.0480 21.9550' or
    '47.0480, 21.9550'. Assumes latitude first, longitude second, but corrects
    an obvious swap (a value whose magnitude exceeds 90 cannot be a latitude).
    Both values must fall in valid coordinate ranges or nothing is returned.
    """
    vals = [
        v for v in (_parse_coord_value(cm.group(), "") for cm in _COORD_RE.finditer(text))
        if v is not None
    ]
    if len(vals) < 2:
        return None, None
    a, b = vals[0], vals[1]
    if abs(a) > 90.0 and abs(b) <= 90.0:   # first looks like a longitude — swap
        a, b = b, a
    if abs(a) <= 90.0 and abs(b) <= 180.0:
        return a, b
    return None, None


def _parse_ocr_text(text: str) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
    """
    Parse a block of OCR text to extract (lat, lon, timestamp).
    All three fields are optional — returns None for any that are absent.

    Resolution order (most explicit → least):
      1. Labeled        — "LAT: 47.0480  LONG: 21.9550"
      2. Hemisphere-tag — "N47.0480 E21.9550", "47.0480N 21.9550E" (any order)
      3. Bare pair      — "47.0480 21.9550" / "47.0480, 21.9550"

    A returned coordinate is always range-checked (lat∈[-90,90], lon∈[-180,180]).
    """
    text = text.replace("\n", " ")
    ts: Optional[datetime] = None

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

    # 1) Labeled latitude / longitude.
    lat = _labeled_coord(text, _LAT_PREFIX, ("N", "S"))
    lon = _labeled_coord(text, _LON_PREFIX, ("E", "W"))

    # 2) Hemisphere-tagged tokens for whatever the labels did not yield.
    if lat is None or lon is None:
        hlat, hlon = _hemi_scan(text)
        lat = lat if lat is not None else hlat
        lon = lon if lon is not None else hlon

    # 3) Unlabeled decimal pair.
    if lat is None and lon is None:
        lat, lon = _bare_pair(text)

    # ── Range validation — never emit an out-of-range coordinate ─────────────
    if lat is not None and not (-90.0 <= lat <= 90.0):
        lat = None
    if lon is not None and not (-180.0 <= lon <= 180.0):
        lon = None

    # lat exactly equal to lon is the signature of one field being back-filled
    # from the other (garbled label value) — reject rather than emit a fake fix.
    if lat is not None and lon is not None and lat == lon:
        return None, None, ts

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

    Converts to grayscale and upscales small crops so the text is large enough
    for the recognizer. We deliberately do NOT binarize here: EasyOCR is a deep
    recognizer that performs best on the natural grayscale image, and the
    previous version computed an adaptive threshold only to discard it and
    return the grayscale concatenated with its own photographic negative — that
    duplicated every glyph side-by-side, which corrupts downstream parsing.
    Polarity is handled by the caller, which retries on an inverted copy when
    the first pass yields no coordinate.
    """
    import cv2 as _cv2
    gray = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale small crops — OCR engines struggle below ~30 px font height.
    h, w = gray.shape[:2]
    if 0 < h < 80:
        scale = max(2, 80 // h)
        gray = _cv2.resize(
            gray, (w * scale, h * scale), interpolation=_cv2.INTER_CUBIC
        )
    return gray


def _invert(img: "np.ndarray") -> "np.ndarray":
    """Photometric negative — used to retry OCR on the opposite polarity."""
    import cv2 as _cv2
    return _cv2.bitwise_not(img)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    import math
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _geofilter(
    points: List[GPSPoint],
    radius_km: float = _GEOFILTER_RADIUS_KM,
) -> List[GPSPoint]:
    """
    Drop geographic outliers. A real dashcam survey is contiguous, so any fix
    far from the median position is almost certainly a misparse (one wrong OCR
    digit → a wrong-but-in-range coordinate). We anchor on the component-wise
    median (robust to up to ~50 % bad points) and keep everything within
    ``radius_km`` of it. If filtering would remove everything, the input is
    returned unchanged so a genuinely long route is never silently emptied.
    """
    if len(points) < 4:
        return points
    lats = sorted(p.lat for p in points)
    lons = sorted(p.lon for p in points)
    med_lat = lats[len(lats) // 2]
    med_lon = lons[len(lons) // 2]
    kept = [
        p for p in points
        if _haversine_km(p.lat, p.lon, med_lat, med_lon) <= radius_km
    ]
    dropped = len(points) - len(kept)
    if dropped:
        log.warning(
            "[geofilter] Dropped %d/%d fix(es) >%.0f km from median "
            "(%.5f, %.5f) — likely OCR misparses.",
            dropped, len(points), radius_km, med_lat, med_lon,
        )
    return kept or points


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

    # ── OCR helper bound to the resolved backend ─────────────────────────────
    def _ocr(img: "np.ndarray") -> str:
        return (
            _ocr_frame_easyocr(easyocr_reader, img)
            if use_easyocr else _ocr_frame_tesseract(img)
        )

    def _read_overlay(frame: "np.ndarray") -> tuple:
        """
        Try to read a coordinate from a frame: bottom strip first, then top,
        each at normal and inverted polarity. Uses the REAL decoded height so a
        rotated/anamorphic frame is cropped correctly. Returns (lat, lon, ts).
        """
        height = frame.shape[0]
        strip = max(40, int(height * strip_frac))
        for region in (frame[height - strip:, :], frame[:strip, :]):
            base = _preprocess_crop(region)
            for img in (base, _invert(base)):
                try:
                    txt = _ocr(img)
                except Exception as exc:  # noqa: BLE001
                    log.debug("[OCR] recognizer error: %s", exc)
                    continue
                la, lo, t = _parse_ocr_text(txt)
                if la is not None and lo is not None:
                    return la, lo, t
        return None, None, None

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise ExtractorError(f"[OCR] Cannot open video file: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Build the sample plan. When the frame count is known we seek by timestamp
    # (CAP_PROP_POS_MSEC) — far more reliable across H.264/H.265 long-GOP files
    # than exact-frame seeking, which lands on the wrong frame or fails. When it
    # is unknown (some containers report 0), we fall back to decoding the stream
    # sequentially and sampling at a fixed stride.
    sample_ts_ms: Optional[list] = None
    stride = 1
    if total_frames > 0:
        duration_s = total_frames / fps
        log.info(
            "[OCR] Video: %d frames, %.2f fps, %.1f s — timestamp sampling",
            total_frames, fps, duration_s,
        )
        sample_ts_ms = [
            1000.0 * duration_s * (i + 0.5) / n_samples for i in range(n_samples)
        ]
    else:
        stride = max(1, int(round(fps / 2.0)))   # ~2 fps
        log.info(
            "[OCR] Frame count unknown — sequential sampling every %d frame(s) "
            "(%.2f fps source)", stride, fps,
        )

    # ── Sampling loop ────────────────────────────────────────────────────────
    points:  List[GPSPoint] = []
    n_parsed = 0
    n_failed = 0
    n_seen   = 0

    if sample_ts_ms is not None:
        for t_ms in sample_ts_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ret, frame = cap.read()
            if not ret or frame is None:
                n_failed += 1
                continue
            n_seen += 1
            lat, lon, ts = _read_overlay(frame)
            if lat is None or lon is None:
                n_failed += 1
                continue
            if ts is None:
                ts = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            p = GPSPoint(lat=lat, lon=lon, timestamp=ts)
            if p.is_valid():
                points.append(p)
                n_parsed += 1
    else:
        idx = -1
        while n_seen < n_samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            idx += 1
            if idx % stride != 0:
                continue
            n_seen += 1
            lat, lon, ts = _read_overlay(frame)
            if lat is None or lon is None:
                n_failed += 1
                continue
            if ts is None:
                ts = datetime.fromtimestamp(idx / fps, tz=timezone.utc)
            p = GPSPoint(lat=lat, lon=lon, timestamp=ts)
            if p.is_valid():
                points.append(p)
                n_parsed += 1

    cap.release()

    log.info(
        "[OCR] Sampled %d frame(s) → %d valid points, %d failed/unparsed",
        n_seen, n_parsed, n_failed,
    )

    if not points:
        raise ExtractorError(
            f"[OCR] Could not extract any GPS coordinates from {mp4_path.name} "
            "via OCR. Possible causes:\n"
            "  • The video has no burned-in GPS overlay.\n"
            "  • The overlay uses an unrecognised format (check --verbose output).\n"
            "  • The overlay is not in the bottom/top strip — adjust --ocr_strip_frac.\n"
            "  • Poor OCR quality — try --ocr_backend tesseract."
        )

    # Reject geographic outliers (the 'spread across Europe' misparses) BEFORE
    # collapsing duplicates, so a single bad fix cannot anchor the track.
    points = _geofilter(points)

    # Deduplicate consecutive identical fixes (static vehicle) into one.
    deduped: List[GPSPoint] = [points[0]]
    for p in points[1:]:
        prev = deduped[-1]
        if p.lat != prev.lat or p.lon != prev.lon:
            deduped.append(p)

    log.info(
        "[OCR] %d point(s) after geo-filter + dedup (from %d parsed)",
        len(deduped), n_parsed,
    )
    return deduped


# numpy is needed by _extract_ocr; import lazily to avoid hard dependency
# for users who only use the non-OCR strategies.
try:
    import numpy as _np  # type: ignore
except ImportError:
    _np = None  # type: ignore  # _extract_ocr will fail with a clear error if called


# ---------------------------------------------------------------------------
# Strategy: subtitle / SRT GPS track  (DJI, many dashcams, NMEA loggers)
# ---------------------------------------------------------------------------
#
# A large class of cameras store GPS not as a binary telemetry atom but as a
# text SUBTITLE stream — either embedded in the MP4 as a mov_text/SubRip track,
# or written next to the video as a sidecar `.srt` / `.nmea` file.
# Common payload formats handled here:
#   • DJI      "[latitude: 47.0480] [longitude: 21.9550]"  and
#              "GPS (21.9550,47.0480, ...)"   (DJI order is lon,lat)
#   • NMEA     "$GPRMC,...", "$GPGGA,...", "$GNRMC,..."
#   • generic  any line that _parse_ocr_text can read (labelled / hemisphere)
#
# This is fast and exact when present, so it runs before the OCR fallback.

_DJI_LAT_RE = _re.compile(r"latitude\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", _re.IGNORECASE)
_DJI_LON_RE = _re.compile(r"longitude\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", _re.IGNORECASE)
_DJI_GPS_RE = _re.compile(
    r"GPS\s*[\(:]\s*([+-]?\d+\.\d+)\s*[,;]\s*([+-]?\d+\.\d+)", _re.IGNORECASE
)


def _nmea_to_deg(value: str, hemi: str) -> Optional[float]:
    """Convert an NMEA ddmm.mmmm / dddmm.mmmm field to signed decimal degrees."""
    if not value:
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    deg = int(f // 100)
    minutes = f - deg * 100
    dec = deg + minutes / 60.0
    if hemi.upper() in ("S", "W"):
        dec = -dec
    return dec


def _nmea_datetime(tod: str, dat: str) -> Optional[datetime]:
    """Build a UTC datetime from NMEA time (hhmmss[.sss]) + date (ddmmyy)."""
    try:
        hh, mm, ss = int(tod[0:2]), int(tod[2:4]), int(tod[4:6])
        dd, mo, yy = int(dat[0:2]), int(dat[2:4]), 2000 + int(dat[4:6])
        return datetime(yy, mo, dd, hh, mm, ss, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def _parse_nmea_line(line: str) -> Optional[GPSPoint]:
    """Parse a single NMEA RMC or GGA sentence into a GPSPoint (or None)."""
    parts = line.split(",")
    tag = parts[0]
    try:
        if tag.endswith("RMC") and len(parts) >= 7 and parts[2] == "A":
            lat = _nmea_to_deg(parts[3], parts[4])
            lon = _nmea_to_deg(parts[5], parts[6])
            spd = None
            if len(parts) >= 8 and parts[7]:
                try:
                    spd = float(parts[7]) * 0.514444   # knots → m/s
                except ValueError:
                    spd = None
            ts = _nmea_datetime(parts[1], parts[9]) if len(parts) >= 10 else None
            if lat is not None and lon is not None:
                p = GPSPoint(lat=lat, lon=lon, timestamp=ts, speed_mps=spd)
                return p if p.is_valid() else None
        elif tag.endswith("GGA") and len(parts) >= 6:
            lat = _nmea_to_deg(parts[2], parts[3])
            lon = _nmea_to_deg(parts[4], parts[5])
            ele = None
            if len(parts) >= 10 and parts[9]:
                try:
                    ele = float(parts[9])
                except ValueError:
                    ele = None
            if lat is not None and lon is not None:
                p = GPSPoint(lat=lat, lon=lon, elevation=ele)
                return p if p.is_valid() else None
    except (ValueError, IndexError):
        return None
    return None


def _parse_subtitle_blob(blob: str) -> List[GPSPoint]:
    """Parse a whole SRT / NMEA text blob into GPSPoints (DJI / NMEA / generic)."""
    pts: List[GPSPoint] = []
    for raw in blob.splitlines():
        line = raw.strip()
        if not line:
            continue

        # NMEA sentences.
        if line.startswith("$") and ("RMC" in line[:8] or "GGA" in line[:8]):
            p = _parse_nmea_line(line)
            if p is not None:
                pts.append(p)
            continue

        lat = lon = None
        ts: Optional[datetime] = None

        m_la, m_lo = _DJI_LAT_RE.search(line), _DJI_LON_RE.search(line)
        if m_la and m_lo:
            lat, lon = float(m_la.group(1)), float(m_lo.group(1))
        else:
            m_gps = _DJI_GPS_RE.search(line)
            if m_gps:
                a, b = float(m_gps.group(1)), float(m_gps.group(2))
                # DJI writes GPS(longitude, latitude); disambiguate by range.
                if abs(a) <= 90.0 and abs(b) > 90.0:
                    lat, lon = a, b
                elif abs(b) <= 90.0 and abs(a) > 90.0:
                    lat, lon = b, a
                else:
                    lon, lat = a, b            # default to DJI lon,lat order

        if lat is None or lon is None:
            la2, lo2, ts = _parse_ocr_text(line)
            lat = lat if lat is not None else la2
            lon = lon if lon is not None else lo2

        if lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            p = GPSPoint(lat=lat, lon=lon, timestamp=ts)
            if p.is_valid():
                pts.append(p)
    return pts


def _ffmpeg_dump_subtitles(mp4_path: Path) -> Optional[str]:
    """Return the first embedded subtitle stream as SRT text, or None."""
    import json as _json
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "s",
             "-show_streams", "-of", "json", str(mp4_path)],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        log.debug("[Subtitle] ffprobe not on PATH — skipping embedded check")
        return None
    except subprocess.TimeoutExpired:
        log.debug("[Subtitle] ffprobe timed out")
        return None
    try:
        streams = _json.loads(probe.stdout).get("streams", [])
    except _json.JSONDecodeError:
        return None
    if not streams:
        return None
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(mp4_path),
             "-map", "0:s:0", "-f", "srt", "pipe:1"],
            capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout or None


def _extract_subtitle(mp4_path: Path) -> List[GPSPoint]:
    """
    Extract GPS from an embedded subtitle track or a sidecar .srt / .nmea file.
    """
    log.info("[Subtitle] Attempting subtitle/SRT GPS extraction from %s", mp4_path.name)

    blobs: List[str] = []

    # Sidecar files next to the video (DJI exports VIDEO.SRT; loggers write .nmea).
    for ext in (".srt", ".SRT", ".nmea", ".NMEA"):
        side = mp4_path.with_suffix(ext)
        if side.exists():
            try:
                blobs.append(side.read_text(encoding="utf-8", errors="replace"))
                log.info("[Subtitle] Read sidecar %s", side.name)
            except OSError as exc:
                log.debug("[Subtitle] Could not read %s: %s", side.name, exc)

    # Embedded subtitle stream.
    embedded = _ffmpeg_dump_subtitles(mp4_path)
    if embedded:
        blobs.append(embedded)
        log.info("[Subtitle] Pulled embedded subtitle stream via ffmpeg")

    if not blobs:
        raise ExtractorError(
            "[Subtitle] No embedded subtitle track or sidecar .srt/.nmea found "
            f"for {mp4_path.name}."
        )

    points: List[GPSPoint] = []
    for blob in blobs:
        points.extend(_parse_subtitle_blob(blob))

    if not points:
        raise ExtractorError(
            "[Subtitle] Subtitle data was present but contained no recognisable "
            "GPS coordinates (DJI / NMEA / labelled formats)."
        )

    points = _geofilter(points)
    log.info("[Subtitle] Extracted %d GPS points", len(points))
    return points


# ---------------------------------------------------------------------------
# Strategy: sidecar files next to the video (.gpx / .nmea / .csv / .json / .srt)
# ---------------------------------------------------------------------------

def _first_key(d: dict, names: tuple):
    for n in names:
        if n in d:
            return d[n]
    return None


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _parse_any_time(v) -> Optional[datetime]:
    """Best-effort timestamp parse: ISO 8601, common formats, or epoch seconds."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        f = float(s)
        if f > 1e8:
            return datetime.fromtimestamp(f, tz=timezone.utc)
    except ValueError:
        pass
    return None


def _parse_csv_points(text: str) -> List[GPSPoint]:
    """Parse a CSV telemetry export with lat/lon (and optional ele/time) columns."""
    import csv
    import io
    pts: List[GPSPoint] = []
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return pts
    cols = {c.lower().strip(): c for c in reader.fieldnames if c}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    lat_c = pick("lat", "latitude", "gpslatitude")
    lon_c = pick("lon", "lng", "long", "longitude", "gpslongitude")
    ele_c = pick("ele", "elevation", "alt", "altitude")
    t_c = pick("time", "timestamp", "datetime", "date", "utc")
    if not lat_c or not lon_c:
        return pts
    for row in reader:
        try:
            lat = float(str(row[lat_c]).replace(",", "."))
            lon = float(str(row[lon_c]).replace(",", "."))
        except (ValueError, TypeError, KeyError):
            continue
        ele = None
        if ele_c and row.get(ele_c) and _is_num(str(row[ele_c]).replace(",", ".")):
            ele = float(str(row[ele_c]).replace(",", "."))
        ts = _parse_any_time(row.get(t_c)) if t_c else None
        p = GPSPoint(lat=lat, lon=lon, elevation=ele, timestamp=ts)
        if p.is_valid():
            pts.append(p)
    return pts


def _json_walk_points(node, pts: List[GPSPoint]) -> None:
    if isinstance(node, dict):
        lat = _first_key(node, ("lat", "latitude", "Lat", "Latitude"))
        lon = _first_key(node, ("lon", "lng", "long", "longitude", "Lon", "Longitude"))
        if _is_num(lat) and _is_num(lon):
            ele = _first_key(node, ("ele", "alt", "altitude", "elevation"))
            tval = _first_key(node, ("time", "timestamp", "datetime", "utc", "date"))
            p = GPSPoint(
                lat=float(lat), lon=float(lon),
                elevation=float(ele) if _is_num(ele) else None,
                timestamp=_parse_any_time(tval),
            )
            if p.is_valid():
                pts.append(p)
        for v in node.values():
            _json_walk_points(v, pts)
    elif isinstance(node, list):
        for v in node:
            _json_walk_points(v, pts)


def _parse_json_points(text: str) -> List[GPSPoint]:
    """Parse a JSON telemetry export, finding any nested {lat, lon, ...} objects."""
    import json as _json
    try:
        obj = _json.loads(text)
    except _json.JSONDecodeError:
        return []
    pts: List[GPSPoint] = []
    _json_walk_points(obj, pts)
    return pts


def _extract_sidecar(mp4_path: Path) -> List[GPSPoint]:
    """
    Extract GPS from a sidecar file sharing the video's stem:
    .gpx, .nmea, .csv, .json, or .srt. (When a survey is uploaded through the
    frontend only the .mp4 is present, so this mostly helps manual/CLI runs.)
    """
    log.info("[Sidecar] Looking for a GPS sidecar next to %s", mp4_path.name)
    points: List[GPSPoint] = []
    found: List[str] = []
    for ext in (".gpx", ".GPX", ".nmea", ".NMEA", ".csv", ".CSV",
                ".json", ".JSON", ".srt", ".SRT"):
        side = mp4_path.with_suffix(ext)
        if not side.exists():
            continue
        try:
            text = side.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found.append(side.name)
        e = ext.lower()
        if e == ".gpx":
            points += _parse_gpx_xml(text)
        elif e in (".nmea", ".srt"):
            points += _parse_subtitle_blob(text)
        elif e == ".csv":
            points += _parse_csv_points(text)
        elif e == ".json":
            points += _parse_json_points(text)

    if not found:
        raise ExtractorError(
            f"[Sidecar] No .gpx/.nmea/.csv/.json/.srt sidecar found for {mp4_path.name}."
        )
    if not points:
        raise ExtractorError(
            f"[Sidecar] Sidecar(s) {found} contained no usable GPS coordinates."
        )
    log.info("[Sidecar] Parsed %d points from %s", len(points), ", ".join(found))
    return points


# ---------------------------------------------------------------------------
# Strategy: ISO 6709 location atom (QuickTime ©xyz — phones, single location)
# ---------------------------------------------------------------------------

_ISO6709_RE = _re.compile(
    r"([+-]\d{1,3}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)(?:([+-]\d+(?:\.\d+)?))?"
)


def _extract_iso6709(mp4_path: Path) -> List[GPSPoint]:
    """
    Single-location fallback: the ISO 6709 string in the MP4 'udta' atom
    (QuickTime ©xyz / Apple 'com.apple.quicktime.location.ISO6709'). Many phone
    videos carry one location for the whole clip — enough to place a survey.
    """
    log.info("[ISO6709] Looking for a QuickTime location atom in %s", mp4_path.name)
    try:
        data = mp4_path.read_bytes()
    except OSError as exc:
        raise ExtractorError(f"[ISO6709] Cannot read {mp4_path}: {exc}")

    iso: Optional[str] = None
    pos = data.find(b"\xa9xyz")                       # ©xyz atom payload: [len:2][lang:2][str]
    if pos != -1:
        try:
            slen = struct.unpack_from(">H", data, pos + 4)[0]
            iso = data[pos + 8: pos + 8 + slen].decode("ascii", "ignore")
        except (struct.error, UnicodeDecodeError):
            iso = None
    if not iso:                                       # fall back to a raw scan of udta
        m = _re.search(rb"[+-]\d{1,2}\.\d+[+-]\d{1,3}\.\d+", data)
        iso = m.group(0).decode("ascii", "ignore") if m else None
    if not iso:
        raise ExtractorError(f"[ISO6709] No location atom found in {mp4_path.name}.")

    m = _ISO6709_RE.search(iso)
    if not m:
        raise ExtractorError(f"[ISO6709] Could not parse location string {iso!r}.")
    p = GPSPoint(
        lat=float(m.group(1)), lon=float(m.group(2)),
        elevation=float(m.group(3)) if m.group(3) else None,
    )
    if not p.is_valid():
        raise ExtractorError(f"[ISO6709] Parsed out-of-range location: {iso!r}.")
    log.info("[ISO6709] Single location: lat=%.6f lon=%.6f", p.lat, p.lon)
    return [p]


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

STRATEGIES = ("sidecar", "gpmf", "novatek", "subtitle", "exiftool", "iso6709", "ocr")


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
        "sidecar":   _extract_sidecar,
        "gpmf":      _extract_gpmf,
        "novatek":   _extract_novatek,
        "subtitle":  _extract_subtitle,
        "exiftool":  _extract_exiftool,
        "iso6709":   _extract_iso6709,
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
    # The console banners below use Unicode glyphs (─ → ✓); the default Windows
    # console codepage (cp1252) cannot encode them and would crash on print().
    # Force UTF-8 on stdout/stderr where the runtime supports reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")        # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

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