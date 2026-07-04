/**
 * frontend/src/pages/MapPage.jsx
 *
 * The operational map of Cluj-Napoca.
 *  - severity + class + repaired-status filtering
 *  - three basemaps (dark / streets / satellite)
 *  - detection detail drawer (mark repaired · delete · zoom)
 *  - box-select zone analysis with confidence histogram
 *  - heatmap mode, landmark fly-to, printable report
 *  - silent live refresh while a pipeline job runs (localStorage['rids_active_job'])
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { useLocation } from 'react-router-dom'
import { BarChart, Bar, ResponsiveContainer, XAxis, Tooltip } from 'recharts'
import { MapContainer, TileLayer, CircleMarker, useMap, useMapEvents, Rectangle } from 'react-leaflet'
import {
  FileText, RefreshCw, Eye, EyeOff, AlertTriangle, Flame, PenTool, XCircle,
  Radio, X, CheckCircle2, Trash2, Crosshair, MapPin, ChevronDown, Wrench,
} from 'lucide-react'
import {
  CLASS_COLORS, CLASS_LABELS, CLASS_ICONS,
  SEVERITY_COLORS, SEVERITY_LABELS, SEVERITY_ACTIONS,
  CITY_ZOOM, CLUJ_LANDMARKS, BASEMAPS,
} from '../utils/constants'
import { fmtCoord, fmtDate, fmtPct } from '../utils/format'
import {
  fetchDetections, fetchStats, fetchJobStatus,
  updateDetectionStatus, deleteDetectionsBulk, fetchCityLandmarks,
} from '../utils/api'
import { SevBadge, ClassChip, ClassDot, KvRow, Spinner, Toggle } from '../components/ui'
import { useIsDark } from '../hooks/useTheme'
import { useAuth } from '../context/AuthContext'
import useCityCenter from '../hooks/useCityCenter'
import useIsMobile from '../hooks/useIsMobile'

// ─── Live-update polling interval (ms) — matches IngestionPage ────────────
const LIVE_POLL_MS = 10_000

// ── Auto-fit map to data bounds (first load only) ─────────────────────────
function FitBounds({ detections }) {
  const map = useMap()
  const done = useRef(false)
  useEffect(() => {
    if (done.current || !detections || detections.length === 0) return
    const lats = detections.map(d => d.latitude)
    const lons = detections.map(d => d.longitude)
    map.fitBounds([
      [Math.min(...lats) - 0.002, Math.min(...lons) - 0.002],
      [Math.max(...lats) + 0.002, Math.max(...lons) + 0.002],
    ], { padding: [50, 50] })
    done.current = true
  }, [detections, map])
  return null
}

// ── Imperative fly-to helper ───────────────────────────────────────────────
function FlyTo({ target }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo([target.lat, target.lon], target.zoom ?? 16, { duration: 0.9 })
  }, [target, map])
  return null
}

// ── Point in Rectangle ─────────────────────────────────────────────────────
function isPointInRect(point, rect) {
  const [start, end] = rect
  const latMin = Math.min(start[0], end[0])
  const latMax = Math.max(start[0], end[0])
  const lngMin = Math.min(start[1], end[1])
  const lngMax = Math.max(start[1], end[1])
  return point[0] >= latMin && point[0] <= latMax && point[1] >= lngMin && point[1] <= lngMax
}

// ── Zone drawing handler ───────────────────────────────────────────────────
function MapClickHandler({ drawingMode, setStart, setEnd, finishDrawing }) {
  const [isDrawing, setIsDrawing] = useState(false)
  const map = useMap()

  useMapEvents({
    mousedown(e) {
      if (drawingMode) {
        setIsDrawing(true)
        map.dragging.disable()
        setStart([e.latlng.lat, e.latlng.lng])
        setEnd([e.latlng.lat, e.latlng.lng])
      }
    },
    mousemove(e) {
      if (drawingMode && isDrawing) setEnd([e.latlng.lat, e.latlng.lng])
    },
    mouseup() {
      if (drawingMode && isDrawing) {
        setIsDrawing(false)
        map.dragging.enable()
        finishDrawing()
      }
    },
  })

  useEffect(() => {
    if (!drawingMode) {
      map.dragging.enable()
      setIsDrawing(false)
    }
  }, [drawingMode, map])

  return null
}

// ── Printable report (opens a print window) ────────────────────────────────
function generateReport(detections, stats, city) {
  const cityTitle = city ? `${city} Road Condition Report` : 'Road Condition Report'
  const byClass = {}
  const bySev = {}
  detections.forEach(d => {
    byClass[d.damage_type] = (byClass[d.damage_type] || 0) + 1
    bySev[d.severity] = (bySev[d.severity] || 0) + 1
  })
  const sevColors = { 1: '#3ddc84', 2: '#d4a900', 3: '#e08a30', 4: '#e04848', 5: '#a21caf' }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RIDS — ${cityTitle}</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #f6f7f9; color: #10141c; margin:0; }
  .cover { background: #05070b; color: #eaff3d; padding: 56px 48px 36px; }
  .cover h1 { font-size: 34px; font-weight: 800; margin: 0 0 6px; letter-spacing: -1px; }
  .cover p { color: #a8b0c2; font-size: 13px; margin: 0; }
  .dash { height: 4px; width: 160px; margin-top: 18px;
          background-image: linear-gradient(90deg,#eaff3d 0 26px, transparent 26px 42px);
          background-size: 42px 4px; }
  .body { padding: 36px 48px; }
  .section { margin-bottom: 34px; }
  h2 { font-size: 19px; font-weight: 700; border-bottom: 2px solid #eaff3d; padding-bottom: 8px; margin: 0 0 18px; }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 30px; }
  .stat-card { background: white; border-radius: 10px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }
  .stat-val { font-size: 30px; font-weight: 800; }
  .stat-lbl { font-size: 11px; color: #626b80; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  th { background: #05070b; color: #eaff3d; font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em; padding: 11px 15px; text-align: left; }
  td { padding: 10px 15px; border-bottom: 1px solid #eef0f4; font-size: 12.5px; }
  tr:last-child td { border: none; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
  .footer { background: #05070b; color: #626b80; padding: 22px 48px; font-size: 11.5px; text-align: center; margin-top: 36px; }
</style>
</head>
<body>
<div class="cover">
  <h1>${cityTitle}</h1>
  <p>RIDS — Road Infrastructure Detection System · Babeș-Bolyai University · Generated ${new Date().toLocaleString()}</p>
  <div class="dash"></div>
</div>
<div class="body">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val">${stats?.total_detections ?? detections.length}</div><div class="stat-lbl">Total detections</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#e04848">${stats?.critical_count ?? 0}</div><div class="stat-lbl">Critical (S4–S5)</div></div>
    <div class="stat-card"><div class="stat-val">${stats?.avg_severity?.toFixed(1) ?? '—'}</div><div class="stat-lbl">Avg severity</div></div>
    <div class="stat-card"><div class="stat-val">${stats?.last_survey_date ?? '—'}</div><div class="stat-lbl">Last survey</div></div>
  </div>

  <div class="section">
    <h2>Detections by class</h2>
    <table>
      <thead><tr><th>Class</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
        ${Object.entries(byClass).sort((a, b) => b[1] - a[1]).map(([cls, cnt]) =>
          `<tr><td>${CLASS_LABELS[cls] || cls}</td><td><strong>${cnt}</strong></td><td>${((cnt / detections.length) * 100).toFixed(1)}%</td></tr>`
        ).join('')}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Severity distribution</h2>
    <table>
      <thead><tr><th>Level</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
        ${[1, 2, 3, 4, 5].map(s => {
          const cnt = bySev[s] || 0
          return `<tr><td><span class="badge" style="background:${sevColors[s]}22;color:${sevColors[s]}">${SEVERITY_LABELS[s]}</span></td><td><strong>${cnt}</strong></td><td>${detections.length ? ((cnt / detections.length) * 100).toFixed(1) : 0}%</td></tr>`
        }).join('')}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Top 30 priority detections</h2>
    <table>
      <thead><tr><th>#</th><th>Type</th><th>Severity</th><th>Priority</th><th>GPS</th><th>Date</th></tr></thead>
      <tbody>
        ${[...detections].sort((a, b) => (b.priority_score || 0) - (a.priority_score || 0)).slice(0, 30).map((d, i) => {
          const sc = sevColors[d.severity] || '#888'
          return `<tr>
            <td style="color:#626b80;font-size:11px">${i + 1}</td>
            <td>${CLASS_LABELS[d.damage_type] || d.damage_type}</td>
            <td><span class="badge" style="background:${sc}22;color:${sc}">S${d.severity}</span></td>
            <td style="font-family:monospace">${(d.priority_score || 0).toFixed(4)}</td>
            <td style="font-family:monospace;font-size:11px">${d.latitude?.toFixed(5)}, ${d.longitude?.toFixed(5)}</td>
            <td style="font-size:11px;color:#626b80">${d.last_detected || '—'}</td>
          </tr>`
        }).join('')}
      </tbody>
    </table>
  </div>
</div>
<div class="footer">
  RIDS · Road Infrastructure Detection System · Babeș-Bolyai University, Faculty of Mathematics and Computer Science, 2026
</div>
<script>window.onload = () => window.print()</script>
</body></html>`

  const w = window.open('', '_blank')
  w.document.write(html)
  w.document.close()
}

// ── Main MapPage ───────────────────────────────────────────────────────────
export default function MapPage() {
  const [detections, setDetections] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Filters
  const [activeClasses, setActiveClasses] = useState(new Set())
  const [activeSeverities, setActiveSeverities] = useState(new Set([1, 2, 3, 4, 5]))
  const [showFixed, setShowFixed] = useState(true)

  // View
  const [selected, setSelected] = useState(null)
  // Phones start with the layer panel folded — the map is the point.
  const [showLegend, setShowLegend] = useState(() => !window.matchMedia('(max-width: 768px)').matches)
  const [heatmapMode, setHeatmapMode] = useState(false)
  // Basemap follows the app theme (dark → Dark tiles, light → Streets tiles)
  // until the user picks one explicitly with the switcher.
  const isDark = useIsDark()
  const [basemapChoice, setBasemapChoice] = useState(null)
  const basemap = basemapChoice ?? (isDark ? 'dark' : 'voyager')
  const [flyTarget, setFlyTarget] = useState(null)
  const [landmarksOpen, setLandmarksOpen] = useState(false)

  // Map opens on the operator's own city (geocoded once, cached — never a
  // hardcoded default).
  const isMobile = useIsMobile()
  const { center, zoom, cityCenter } = useCityCenter()
  const cityFlown = useRef(false)
  useEffect(() => {
    // Marked only after load, so the one-shot glide below has had its
    // render; afterwards the camera belongs to the user / FitBounds.
    if (cityCenter && !loading) cityFlown.current = true
  }, [cityCenter, loading])

  // Landmarks: per-city from the backend (free OSM lookup, cached). The
  // built-in list is an offline fallback for the demo city only.
  const { user } = useAuth()
  const [landmarks, setLandmarks] = useState(() => (
    (user?.city || '').toLowerCase().startsWith('cluj')
      ? CLUJ_LANDMARKS.map(lm => ({ name: lm.name, latitude: lm.lat, longitude: lm.lon }))
      : []
  ))
  useEffect(() => {
    if (!user?.city) return undefined
    let alive = true
    fetchCityLandmarks(user.city)
      .then(res => { if (alive && res.items?.length) setLandmarks(res.items) })
      .catch(() => { /* keep fallback list */ })
    return () => { alive = false }
  }, [user?.city])

  // Focus request coming from ExplorerPage ("Show on map")
  const routerLocation = useLocation()
  const focusDone = useRef(false)

  // Zone drawing
  const [drawingMode, setDrawingMode] = useState(false)
  const [rectStart, setRectStart] = useState(null)
  const [rectEnd, setRectEnd] = useState(null)
  const [finishedRect, setFinishedRect] = useState(null)

  // ── Live update state — see original design notes ───────────────────────
  const [liveJobId, setLiveJobId] = useState(() => localStorage.getItem('rids_active_job') || null)
  const [liveActive, setLiveActive] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(null)
  const pollRef = useRef(null)

  const refreshData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const [det, st] = await Promise.all([
        fetchDetections({ page: 1, page_size: 5000 }),
        fetchStats(),
      ])
      setDetections(det.items || [])
      setStats(st)
      if (!silent) {
        setActiveClasses(new Set((det.items || []).map(d => d.damage_type)))
      }
      if (silent) setLastRefresh(new Date().toISOString())
    } catch (e) {
      if (!silent) setError(e.message)
    } finally {
      if (!silent) setLoading(false)
    }
  }, [])

  useEffect(() => { refreshData(false) }, [refreshData])

  // Fly to + open the detection requested by ExplorerPage's "Show on map".
  useEffect(() => {
    const focus = routerLocation.state?.focus
    if (!focus || focusDone.current || detections.length === 0) return
    focusDone.current = true
    const target = detections.find(d => d.id === focus.id)
    setFlyTarget({ lat: focus.lat, lon: focus.lon, zoom: 18 })
    if (target) setSelected(target)
  }, [routerLocation.state, detections])

  // Live polling — refresh silently while a pipeline job runs
  useEffect(() => {
    if (!liveJobId) {
      if (pollRef.current) clearInterval(pollRef.current)
      setLiveActive(false)
      return
    }

    setLiveActive(true)
    localStorage.setItem('rids_active_job', liveJobId)

    const stop = () => {
      setLiveActive(false)
      setLiveJobId(null)
      localStorage.removeItem('rids_active_job')
      if (pollRef.current) clearInterval(pollRef.current)
    }

    const tick = async () => {
      try {
        const jobData = await fetchJobStatus(liveJobId)
        const s = jobData.status
        if (s === 'running' || s === 'initialising' || s === 'pending') {
          refreshData(true)
        } else if (s === 'complete') {
          await refreshData(true)
          stop()
        } else {
          stop()
        }
      } catch (err) {
        if (err?.response?.status === 404) stop()
        // network errors: keep trying
      }
    }

    tick()
    pollRef.current = setInterval(tick, LIVE_POLL_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [liveJobId, refreshData])

  const toggleClass = useCallback((cls) => {
    setActiveClasses(prev => {
      const next = new Set(prev)
      next.has(cls) ? next.delete(cls) : next.add(cls)
      return next
    })
  }, [])

  const toggleSeverity = useCallback((s) => {
    setActiveSeverities(prev => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })
  }, [])

  // ── Detail-drawer actions ────────────────────────────────────────────────
  const [actionBusy, setActionBusy] = useState(false)

  const markFixed = async (d, fixed) => {
    setActionBusy(true)
    try {
      const updated = await updateDetectionStatus(d.id, fixed)
      setDetections(prev => prev.map(x => (x.id === d.id ? { ...x, is_fixed: updated.is_fixed } : x)))
      setSelected(prev => (prev && prev.id === d.id ? { ...prev, is_fixed: updated.is_fixed } : prev))
    } catch (e) {
      alert(`Could not update: ${e?.response?.data?.detail || e.message}`)
    } finally {
      setActionBusy(false)
    }
  }

  const deleteOne = async (d) => {
    if (!window.confirm(`Delete this ${CLASS_LABELS[d.damage_type] || d.damage_type} record? This cannot be undone.`)) return
    setActionBusy(true)
    try {
      await deleteDetectionsBulk([d.id])
      setDetections(prev => prev.filter(x => x.id !== d.id))
      setSelected(null)
    } catch (e) {
      alert(`Could not delete: ${e?.response?.data?.detail || e.message}`)
    } finally {
      setActionBusy(false)
    }
  }

  // ── Derived data ─────────────────────────────────────────────────────────
  const classCounts = useMemo(() => {
    const counts = {}
    detections.forEach(d => { counts[d.damage_type] = (counts[d.damage_type] || 0) + 1 })
    return counts
  }, [detections])

  const currentRect = finishedRect || (drawingMode && rectStart && rectEnd ? [rectStart, rectEnd] : null)

  const visible = useMemo(() => detections.filter(d => {
    if (!activeClasses.has(d.damage_type)) return false
    if (d.severity && !activeSeverities.has(d.severity)) return false
    if (!showFixed && d.is_fixed) return false
    if (currentRect) return isPointInRect([d.latitude, d.longitude], currentRect)
    return true
  }), [detections, activeClasses, activeSeverities, showFixed, currentRect])

  const displayStats = currentRect ? {
    total_detections: visible.length,
    critical_count: visible.filter(d => d.severity >= 4).length,
    avg_severity: visible.length ? (visible.reduce((acc, d) => acc + (d.severity || 0), 0) / visible.length) : 0,
    avg_confidence: visible.length ? (visible.reduce((acc, d) => acc + d.confidence, 0) / visible.length) : 0,
    last_survey_date: stats?.last_survey_date,
  } : stats

  const confidenceData = useMemo(() => {
    if (!finishedRect) return []
    const bins = { '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0 }
    visible.forEach(d => {
      const c = d.confidence
      if (c < 0.4) bins['20-40%']++
      else if (c < 0.6) bins['40-60%']++
      else if (c < 0.8) bins['60-80%']++
      else bins['80-100%']++
    })
    return Object.entries(bins).map(([name, count]) => ({ name, count }))
  }, [finishedRect, visible])

  const tiles = BASEMAPS[basemap]

  return (
    <div style={styles.page}>

      {/* ── Map ─────────────────────────────────────────────────────────── */}
      <MapContainer
        center={center}
        zoom={zoom}
        maxZoom={20}
        minZoom={3}
        style={{ width: '100%', height: '100%', cursor: drawingMode ? 'crosshair' : 'grab' }}
        zoomControl={false}
      >
        <TileLayer key={basemap} url={tiles.url} attribution={tiles.attr} maxZoom={20} maxNativeZoom={19} />
        <FitBounds detections={detections} />
        <FlyTo target={flyTarget} />
        {/* First visit on this browser: glide to the user's city once it
            geocodes — unless there is data to fit or an explicit fly. */}
        {!loading && detections.length === 0 && cityCenter && !cityFlown.current && !flyTarget && (
          <FlyTo target={{ lat: cityCenter[0], lon: cityCenter[1], zoom: CITY_ZOOM }} />
        )}
        <MapClickHandler
          drawingMode={drawingMode}
          setStart={setRectStart}
          setEnd={setRectEnd}
          finishDrawing={() => {
            if (rectStart && rectEnd && rectStart[0] !== rectEnd[0]) {
              setFinishedRect([rectStart, rectEnd])
            }
            setDrawingMode(false)
          }}
        />

        {finishedRect && (
          <Rectangle bounds={finishedRect} pathOptions={{ color: '#eaff3d', fillColor: '#eaff3d', fillOpacity: 0.08, weight: 2 }} />
        )}
        {!finishedRect && rectStart && rectEnd && (
          <Rectangle bounds={[rectStart, rectEnd]} pathOptions={{ color: '#eaff3d', fillColor: '#eaff3d', fillOpacity: 0.08, weight: 2, dashArray: '4' }} />
        )}

        {visible.map(d => {
          const color = CLASS_COLORS[d.damage_type] || '#888'
          const sevColor = SEVERITY_COLORS[d.severity] || color
          const isSel = selected && selected.id === d.id
          return (
            <CircleMarker
              key={d.id}
              center={[d.latitude, d.longitude]}
              radius={heatmapMode ? (d.severity * 8) : (isSel ? 11 : d.severity >= 4 ? 9 : d.severity === 3 ? 7 : 5)}
              pathOptions={{
                color: heatmapMode ? 'transparent' : (isSel ? '#eaff3d' : sevColor),
                fillColor: heatmapMode ? sevColor : (d.is_fixed ? '#3ddc84' : color),
                fillOpacity: heatmapMode ? 0.3 : (d.is_fixed ? 0.45 : 0.85),
                weight: heatmapMode ? 0 : (isSel ? 3 : d.severity >= 4 ? 2 : 1),
                className: heatmapMode ? 'heatmap-blob' : (d.severity === 5 ? 'marker-critical' : ''),
              }}
              eventHandlers={{ click: () => !heatmapMode && setSelected(d) }}
            />
          )
        })}
      </MapContainer>

      {/* ── Top-left: live badge + landmarks ────────────────────────────── */}
      <div style={styles.topLeft}>
        {liveActive && (
          <div style={styles.liveBadge}>
            <Radio size={11} style={{ animation: 'pulse 1.5s ease-in-out infinite' }} />
            LIVE · every {LIVE_POLL_MS / 1000}s
            {lastRefresh && (
              <span style={{ marginLeft: 6, opacity: 0.6 }}>
                · {new Date(lastRefresh).toLocaleTimeString()}
              </span>
            )}
          </div>
        )}

        <div style={{ position: 'relative' }}>
          <button className="btn btn-sm glass" style={{ borderRadius: 999 }} onClick={() => setLandmarksOpen(v => !v)}>
            <MapPin size={12} /> Landmarks <ChevronDown size={11} />
          </button>
          {landmarksOpen && (
            <div className="glass anim-fade-in" style={styles.landmarkMenu}>
              {landmarks.map(lm => (
                <button
                  key={lm.name}
                  className="table-row-hover"
                  style={styles.landmarkItem}
                  onClick={() => { setFlyTarget({ lat: lm.latitude, lon: lm.longitude, zoom: 15 }); setLandmarksOpen(false) }}
                >
                  <Crosshair size={11} style={{ color: 'var(--accent)' }} />
                  {lm.name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Top-right action bar ─────────────────────────────────────────── */}
      <div style={styles.actions}>
        {/* Basemap switcher */}
        <div className="glass" style={styles.basemapGroup}>
          {Object.entries(BASEMAPS).map(([key, bm]) => (
            <button
              key={key}
              className="btn btn-sm btn-ghost"
              style={{
                border: 'none', borderRadius: 6,
                color: basemap === key ? 'var(--accent)' : 'var(--text-muted)',
                background: basemap === key ? 'var(--accent-dim)' : 'transparent',
              }}
              onClick={() => setBasemapChoice(key)}
            >
              {bm.label}
            </button>
          ))}
        </div>

        {!drawingMode && !finishedRect && (
          <button className="btn btn-sm glass" onClick={() => setDrawingMode(true)}>
            <PenTool size={13} /> Zone
          </button>
        )}
        {(drawingMode || finishedRect) && (
          <button
            className="btn btn-sm glass btn-danger"
            onClick={() => { setDrawingMode(false); setFinishedRect(null); setRectStart(null); setRectEnd(null) }}
          >
            <XCircle size={13} /> {drawingMode ? 'Cancel' : 'Clear zone'}
          </button>
        )}

        <button
          className={`btn btn-sm glass ${heatmapMode ? 'btn-active' : ''}`}
          onClick={() => setHeatmapMode(v => !v)}
        >
          <Flame size={13} /> Heat
        </button>
        <button className="btn btn-sm glass" onClick={() => refreshData(true)} title="Refresh map data">
          <RefreshCw size={13} />
        </button>
        <button
          className="btn btn-sm btn-accent"
          onClick={() => generateReport(detections, stats, user?.city)}
          disabled={detections.length === 0}
        >
          <FileText size={13} /> Report
        </button>
      </div>

      {/* ── Bottom stat strip ────────────────────────────────────────────── */}
      {displayStats && !selected && (
        <div
          style={{
            ...styles.statStrip,
            ...(isMobile ? { maxWidth: 'calc(100vw - 16px)', overflowX: 'auto', gap: 12, padding: '8px 14px' } : null),
          }}
          className="glass anim-fade-up"
        >
          <StatChip label={currentRect ? 'Zone total' : 'Total'} value={displayStats.total_detections} color="var(--accent)" />
          <div style={styles.stripDivider} />
          <StatChip label="Critical" value={displayStats.critical_count} color="var(--red)" />
          <div style={styles.stripDivider} />
          <StatChip label="Avg severity" value={typeof displayStats.avg_severity === 'number' ? displayStats.avg_severity.toFixed(1) : (displayStats.avg_severity ?? '—')} color="var(--orange)" />
          <div style={styles.stripDivider} />
          <StatChip label="Avg conf" value={typeof displayStats.avg_confidence === 'number' ? `${(displayStats.avg_confidence * 100).toFixed(0)}%` : '—'} color="var(--cyan)" />
          <div style={styles.stripDivider} />
          <StatChip label="Visible" value={visible.length} color="var(--text)" />
          {displayStats.last_survey_date && (
            <>
              <div style={styles.stripDivider} />
              <StatChip label="Last survey" value={fmtDate(displayStats.last_survey_date)} color="var(--text-muted)" />
            </>
          )}
        </div>
      )}

      {/* ── Filter panel (bottom-left) ──────────────────────────────────── */}
      <div
        style={{
          ...styles.filterPanel,
          ...(isMobile ? { left: 8, right: 8, maxWidth: 'none', bottom: 72 } : null),
        }}
        className="glass"
      >
        <div style={styles.filterHeader}>
          <span className="overline">Layers</span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button className="btn btn-sm btn-ghost" style={styles.tinyBtn}
              onClick={() => setActiveClasses(new Set(Object.keys(classCounts)))}>
              <Eye size={11} /> ALL
            </button>
            <button className="btn btn-sm btn-ghost" style={styles.tinyBtn}
              onClick={() => setActiveClasses(new Set())}>
              <EyeOff size={11} /> NONE
            </button>
            <button className="btn btn-sm btn-ghost" style={styles.tinyBtn}
              onClick={() => setShowLegend(v => !v)}>
              {showLegend ? '▾' : '▴'}
            </button>
          </div>
        </div>

        {showLegend && (
          <>
            {/* Severity pills */}
            <div style={styles.sevRow}>
              {[1, 2, 3, 4, 5].map(s => {
                const active = activeSeverities.has(s)
                const color = SEVERITY_COLORS[s]
                return (
                  <button
                    key={s}
                    onClick={() => toggleSeverity(s)}
                    className="mono"
                    style={{
                      flex: 1, padding: '4px 0', borderRadius: 6, cursor: 'pointer',
                      fontSize: 10.5, fontWeight: 700, transition: 'var(--transition)',
                      border: `1px solid ${active ? `${color}88` : 'var(--border)'}`,
                      background: active ? `${color}1c` : 'transparent',
                      color: active ? color : 'var(--text-muted)',
                    }}
                  >
                    S{s}
                  </button>
                )
              })}
            </div>

            {/* Class chips */}
            <div style={styles.filterList}>
              {Object.entries(classCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, cnt]) => (
                  <ClassChip
                    key={cls}
                    cls={cls}
                    count={cnt}
                    active={activeClasses.has(cls)}
                    onClick={() => toggleClass(cls)}
                  />
                ))}
            </div>

            <div style={{ padding: '8px 14px 12px', borderTop: '1px solid var(--border)' }}>
              <Toggle checked={showFixed} onChange={setShowFixed} label="Show repaired" />
            </div>
          </>
        )}

        {finishedRect && showLegend && (
          <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)' }}>
            <span className="overline" style={{ display: 'block', marginBottom: 8 }}>Zone confidence</span>
            <div style={{ height: 100, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceData}>
                  <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} interval={0} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', fontSize: 11, borderRadius: 8 }}
                    itemStyle={{ color: 'var(--accent)' }}
                    cursor={{ fill: 'var(--accent-dim)' }}
                  />
                  <Bar dataKey="count" fill="var(--accent)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* ── Detail drawer (right) ────────────────────────────────────────── */}
      {selected && (
        <div
          style={{
            ...styles.drawer,
            ...(isMobile ? { left: 8, right: 8, width: 'auto', top: 'auto', bottom: 8, maxHeight: '62vh' } : null),
          }}
          className="glass anim-slide-right"
        >
          <div style={styles.drawerHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <ClassDot cls={selected.damage_type} size={34} />
              <div>
                <div className="display" style={{ fontSize: 15, fontWeight: 700 }}>
                  {CLASS_LABELS[selected.damage_type] || selected.damage_type}
                </div>
                <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                  {fmtCoord(selected.latitude, selected.longitude)}
                </div>
              </div>
            </div>
            <button className="btn btn-sm btn-ghost" style={{ width: 28, height: 28, padding: 0 }} onClick={() => setSelected(null)}>
              <X size={14} />
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 18px 12px' }}>
            <SevBadge s={selected.severity} />
            {selected.is_fixed && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                background: 'rgba(61,220,132,0.14)', color: 'var(--green)',
                border: '1px solid rgba(61,220,132,0.4)', borderRadius: 5,
                padding: '2px 8px', fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)',
              }}>
                <CheckCircle2 size={11} /> REPAIRED
              </span>
            )}
            <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
              conf {fmtPct(selected.confidence)}
            </span>
          </div>

          <div style={{ padding: '0 18px 8px', fontSize: 11.5, color: 'var(--text-dim)', lineHeight: 1.6 }}>
            {SEVERITY_ACTIONS[selected.severity]}
          </div>

          <div style={{ padding: '4px 18px 8px', overflowY: 'auto', flex: 1 }}>
            <KvRow k="Priority score" v={(selected.priority_score || 0).toFixed(4)} mono />
            <KvRow k="Times observed" v={`${selected.detection_count}×`} mono />
            <KvRow k="First detected" v={fmtDate(selected.first_detected)} />
            <KvRow k="Last detected" v={fmtDate(selected.last_detected)} />
            <KvRow k="Survey" v={selected.survey_video_file || '—'} mono />
            <KvRow k="Lighting" v={selected.lighting_condition || '—'} />
            <KvRow k="Surface area (mask px)" v={selected.surface_area_cm2 != null ? Math.round(selected.surface_area_cm2).toLocaleString() : '—'} mono />
            <KvRow k="Depth estimate" v={selected.depth_estimate_cm != null ? `${selected.depth_estimate_cm.toFixed(1)} (rel)` : '—'} mono />
            <KvRow k="Depth confidence" v={selected.depth_confidence != null ? fmtPct(selected.depth_confidence) : '—'} mono />
            <KvRow k="Edge sharpness" v={selected.edge_sharpness != null ? selected.edge_sharpness.toFixed(2) : '—'} mono />
            <KvRow k="Interior contrast" v={selected.interior_contrast != null ? selected.interior_contrast.toFixed(2) : '—'} mono />
            <KvRow k="Mask compactness" v={selected.mask_compactness != null ? selected.mask_compactness.toFixed(3) : '—'} mono />
            <KvRow k="Severity confidence" v={selected.severity_confidence != null ? fmtPct(selected.severity_confidence) : '—'} mono />
            <KvRow k="ID" v={String(selected.id).slice(0, 8) + '…'} mono />
          </div>

          <div style={styles.drawerActions}>
            <button
              className="btn btn-sm"
              style={{ flex: 1 }}
              onClick={() => setFlyTarget({ lat: selected.latitude, lon: selected.longitude, zoom: 18 })}
            >
              <Crosshair size={13} /> Zoom
            </button>
            <button
              className={`btn btn-sm ${selected.is_fixed ? '' : 'btn-accent'}`}
              style={{ flex: 1.4 }}
              disabled={actionBusy}
              onClick={() => markFixed(selected, !selected.is_fixed)}
            >
              <Wrench size={13} /> {selected.is_fixed ? 'Reopen' : 'Mark repaired'}
            </button>
            <button className="btn btn-sm btn-danger" disabled={actionBusy} onClick={() => deleteOne(selected)}>
              <Trash2 size={13} />
            </button>
          </div>
        </div>
      )}

      {/* ── Loading overlay ─────────────────────────────────────────────── */}
      {loading && (
        <div style={styles.overlay}>
          <Spinner label="Loading detections…" />
        </div>
      )}

      {/* ── Error banner ────────────────────────────────────────────────── */}
      {error && !loading && (
        <div style={styles.errorBanner}>
          <AlertTriangle size={14} />
          <span>Could not reach API: {error}</span>
          <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 8 }}>
            Make sure the backend is running on port 8000
          </span>
        </div>
      )}
    </div>
  )
}

function StatChip({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div className="mono" style={{ fontSize: 15, fontWeight: 700, color }}>
        {value}
      </div>
      <div style={{ fontSize: 9.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.07em' }}>
        {label}
      </div>
    </div>
  )
}

const styles = {
  page: {
    position: 'fixed',
    inset: 'var(--nav-h) 0 0 0',
    overflow: 'hidden',
  },

  topLeft: {
    position: 'absolute',
    top: 14, left: 14,
    zIndex: 800,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 8,
  },
  liveBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    background: 'rgba(234,255,61,0.08)',
    border: '1px solid rgba(234,255,61,0.35)',
    borderRadius: 20,
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    color: 'var(--accent)',
    letterSpacing: '.06em',
    backdropFilter: 'blur(8px)',
  },
  landmarkMenu: {
    position: 'absolute',
    top: 'calc(100% + 6px)',
    left: 0,
    minWidth: 200,
    padding: 6,
    zIndex: 900,
    display: 'flex',
    flexDirection: 'column',
  },
  landmarkItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 10px',
    background: 'transparent',
    border: 'none',
    borderRadius: 7,
    color: 'var(--text-dim)',
    fontSize: 12,
    cursor: 'pointer',
    textAlign: 'left',
  },

  actions: {
    position: 'absolute',
    top: 14, right: 14,
    zIndex: 800,
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
  },
  basemapGroup: {
    display: 'flex',
    gap: 2,
    padding: 3,
    borderRadius: 9,
  },

  statStrip: {
    position: 'absolute',
    bottom: 16,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 800,
    display: 'flex',
    alignItems: 'center',
    gap: 18,
    padding: '10px 24px',
    borderRadius: 40,
  },
  stripDivider: {
    width: 1,
    height: 24,
    background: 'var(--border)',
  },

  filterPanel: {
    position: 'absolute',
    bottom: 80,
    left: 14,
    zIndex: 800,
    maxWidth: 340,
  },
  filterHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    borderBottom: '1px solid var(--border)',
  },
  tinyBtn: {
    padding: '3px 7px',
    fontSize: 10,
    border: '1px solid var(--border)',
  },
  sevRow: {
    display: 'flex',
    gap: 5,
    padding: '10px 14px 4px',
  },
  filterList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    padding: '10px 14px',
    maxHeight: 170,
    overflowY: 'auto',
  },

  drawer: {
    position: 'absolute',
    top: 14,
    right: 14,
    bottom: 16,
    width: 330,
    zIndex: 850,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  drawerHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    padding: '16px 18px 12px',
  },
  drawerActions: {
    display: 'flex',
    gap: 8,
    padding: '12px 18px 16px',
    borderTop: '1px solid var(--border)',
  },

  overlay: {
    position: 'absolute',
    inset: 0,
    background: 'rgba(5,7,11,0.72)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 900,
    backdropFilter: 'blur(4px)',
  },
  errorBanner: {
    position: 'absolute',
    top: 16,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 900,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 18px',
    background: 'rgba(255,93,93,0.12)',
    border: '1px solid rgba(255,93,93,0.4)',
    borderRadius: 'var(--radius)',
    color: 'var(--red)',
    fontSize: 12,
    fontWeight: 600,
    backdropFilter: 'blur(8px)',
  },
}
