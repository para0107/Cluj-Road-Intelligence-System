/**
 * frontend/src/pages/MapPage.jsx
 *
 * The operational map of Cluj-Napoca.
 *  - severity + class + repaired-status filtering
 *  - three basemaps (dark / streets / satellite)
 *  - detection detail drawer (evidence photo · mark repaired · delete · zoom)
 *  - box-select zone analysis with confidence histogram + "create work order"
 *  - heatmap mode, landmark fly-to, printable report
 *  - silent live refresh while a pipeline job runs (localStorage['rids_active_job'])
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
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
  fetchEvidenceUrl,
} from '../utils/api'
import { SevBadge, ClassChip, ClassDot, KvRow, Spinner, Toggle, EmptyState } from '../components/ui'
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
function ZoomTracker({ onZoom }) {
  const map = useMapEvents({ zoomend: () => onZoom(map.getZoom()) })
  useEffect(() => { onZoom(map.getZoom()) }, [map, onZoom])
  return null
}

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
  const sevColors = SEVERITY_COLORS

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RDDS — ${cityTitle}</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #f6f7f9; color: #10141c; margin:0; }
  .cover { background: #1a1a18; color: #cf5a30; padding: 56px 48px 36px; }
  .cover h1 { font-size: 34px; font-weight: 800; margin: 0 0 6px; letter-spacing: -1px; }
  .cover p { color: #a8b0c2; font-size: 13px; margin: 0; }
  .dash { height: 4px; width: 160px; margin-top: 18px;
          background-image: linear-gradient(90deg,#cf5a30 0 26px, transparent 26px 42px);
          background-size: 42px 4px; }
  .body { padding: 36px 48px; }
  .section { margin-bottom: 34px; }
  h2 { font-size: 19px; font-weight: 700; border-bottom: 2px solid #cf5a30; padding-bottom: 8px; margin: 0 0 18px; }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 30px; }
  .stat-card { background: white; border-radius: 10px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }
  .stat-val { font-size: 30px; font-weight: 800; }
  .stat-lbl { font-size: 11px; color: #817c6e; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  th { background: #1a1a18; color: #cf5a30; font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em; padding: 11px 15px; text-align: left; }
  td { padding: 10px 15px; border-bottom: 1px solid #eef0f4; font-size: 12.5px; }
  tr:last-child td { border: none; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
  .footer { background: #1a1a18; color: #817c6e; padding: 22px 48px; font-size: 11.5px; text-align: center; margin-top: 36px; }
</style>
</head>
<body>
<div class="cover">
  <h1>${cityTitle}</h1>
  <p>RDDS · Road Degradation Detection System · Generated ${new Date().toLocaleString()}</p>
  <div class="dash"></div>
</div>
<div class="body">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val">${stats?.total_detections ?? detections.length}</div><div class="stat-lbl">Total detections</div></div>
    <div class="stat-card"><div class="stat-val" style="color:${SEVERITY_COLORS[4]}">${stats?.critical_count ?? 0}</div><div class="stat-lbl">Critical (S4–S5)</div></div>
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
            <td style="color:#817c6e;font-size:11px">${i + 1}</td>
            <td>${CLASS_LABELS[d.damage_type] || d.damage_type}</td>
            <td><span class="badge" style="background:${sc}22;color:${sc}">S${d.severity}</span></td>
            <td style="font-family:monospace">${(d.priority_score || 0).toFixed(4)}</td>
            <td style="font-family:monospace;font-size:11px">${d.latitude?.toFixed(5)}, ${d.longitude?.toFixed(5)}</td>
            <td style="font-size:11px;color:#817c6e">${d.last_detected || '—'}</td>
          </tr>`
        }).join('')}
      </tbody>
    </table>
  </div>
</div>
<div class="footer">
  RDDS · Road Degradation Detection System · 2026
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
  const [showTech, setShowTech] = useState(false)
  const [showAllClasses, setShowAllClasses] = useState(false)
  // Basemap follows the app theme (dark → Dark tiles, light → Streets tiles)
  // until the user picks one explicitly with the switcher.
  const isDark = useIsDark()
  const [basemapChoice, setBasemapChoice] = useState(null)
  const basemap = basemapChoice ?? (isDark ? 'dark' : 'voyager')
  const [flyTarget, setFlyTarget] = useState(null)
  const [landmarksOpen, setLandmarksOpen] = useState(false)
  const [mapZoom, setMapZoom] = useState(13)

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
  const navigate = useNavigate()

  // Zone drawing
  const [drawingMode, setDrawingMode] = useState(false)
  const [rectStart, setRectStart] = useState(null)
  const [rectEnd, setRectEnd] = useState(null)
  const [finishedRect, setFinishedRect] = useState(null)

  // ── Live update state — see original design notes ───────────────────────
  const [liveJobId, setLiveJobId] = useState(() => localStorage.getItem('rids_active_job') || null)
  const [liveActive, setLiveActive] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
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

  // Manual refresh gets visible feedback; the silent live-poll path stays quiet.
  const manualRefresh = useCallback(async () => {
    setRefreshing(true)
    await refreshData(true)
    setLastRefresh(new Date().toISOString())
    setRefreshing(false)
  }, [refreshData])

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

  // ── Evidence photo for the selected detection ────────────────────────────
  // The media route needs the Bearer header, so an <img src> cannot load it:
  // fetch the JPG as a blob and hand the <img> an object URL. Every URL is
  // revoked when the selection changes, otherwise clicking through a few
  // hundred sites would hold every photo in memory for the life of the tab.
  const [evidenceUrl, setEvidenceUrl] = useState(null)
  const [evidenceLoading, setEvidenceLoading] = useState(false)

  useEffect(() => {
    if (!selected?.id || !selected?.has_evidence) {
      setEvidenceUrl(null)
      setEvidenceLoading(false)
      return undefined
    }
    let alive = true
    let objectUrl = null
    setEvidenceUrl(null)
    setEvidenceLoading(true)
    fetchEvidenceUrl(selected.id)
      .then(url => {
        if (!alive) {
          if (url) URL.revokeObjectURL(url)
          return
        }
        objectUrl = url
        setEvidenceUrl(url)
      })
      .catch(() => { if (alive) setEvidenceUrl(null) })
      .finally(() => { if (alive) setEvidenceLoading(false) })
    return () => {
      alive = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [selected?.id, selected?.has_evidence])

  // ── Detail-drawer actions ────────────────────────────────────────────────
  const [actionBusy, setActionBusy] = useState(false)
  const [toasts, setToasts] = useState([])
  const pushToast = useCallback((toast) => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, ...toast }])
    if (toast.ttl !== 0) setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), toast.ttl || 3500)
    return id
  }, [])
  const dismissToast = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), [])

  const markFixed = async (d, fixed) => {
    setActionBusy(true)
    try {
      const updated = await updateDetectionStatus(d.id, fixed)
      setDetections(prev => prev.map(x => (x.id === d.id ? { ...x, is_fixed: updated.is_fixed } : x)))
      setSelected(prev => (prev && prev.id === d.id ? { ...prev, is_fixed: updated.is_fixed } : prev))
      pushToast({ tone: 'success', text: updated.is_fixed ? 'Marked repaired.' : 'Reopened.' })
    } catch (e) {
      pushToast({ tone: 'error', text: `Could not update: ${e?.response?.data?.detail || e.message}` })
    } finally {
      setActionBusy(false)
    }
  }

  // Optimistic soft-delete: pull it from the map now, commit to the API after a
  // 6s undo window. Undo restores it and cancels the call; an API failure puts
  // the record back with an error toast. No irreversible browser confirm.
  const deleteOne = (d) => {
    setDetections(prev => prev.filter(x => x.id !== d.id))
    setSelected(null)
    let undone = false
    const timer = setTimeout(async () => {
      if (undone) return
      try {
        await deleteDetectionsBulk([d.id])
      } catch (e) {
        setDetections(prev => (prev.some(x => x.id === d.id) ? prev : [...prev, d]))
        pushToast({ tone: 'error', text: `Could not delete: ${e?.response?.data?.detail || e.message}` })
      }
    }, 6000)
    pushToast({
      tone: 'default', ttl: 6000,
      text: `Deleted ${CLASS_LABELS[d.damage_type] || d.damage_type}.`,
      actionLabel: 'Undo',
      onAction: () => { undone = true; clearTimeout(timer); setDetections(prev => (prev.some(x => x.id === d.id) ? prev : [...prev, d])) },
    })
  }

  // ── Keyboard: Escape closes the drawer, cancels a zone draw, folds menus ──
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      if (selected) setSelected(null)
      else if (drawingMode || finishedRect) {
        setDrawingMode(false); setFinishedRect(null); setRectStart(null); setRectEnd(null)
      } else if (landmarksOpen) setLandmarksOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, drawingMode, finishedRect, landmarksOpen])

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

  // City-wide zoom with thousands of marks: draw only severity 3+ until the
  // user zooms in. Zone analytics and counters still use the full list.
  const declutterActive = !heatmapMode && visible.length > 3000 && mapZoom < 14
  const rendered = useMemo(
    () => (declutterActive ? visible.filter(d => (d.severity || 0) >= 3) : visible),
    [visible, declutterActive],
  )

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

  // ── Keyboard: step markers (j / k), act on the selected one (r / Delete),
  // toggle heat (h). Gives the map a real keyboard path without a mouse. ──
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'j' || e.key === 'k') {
        if (rendered.length === 0) return
        e.preventDefault()
        const idx = selected ? rendered.findIndex(d => d.id === selected.id) : -1
        const next = e.key === 'j'
          ? rendered[(idx + 1 + rendered.length) % rendered.length]
          : rendered[(idx - 1 + rendered.length) % rendered.length]
        if (next) { setSelected(next); setFlyTarget({ lat: next.latitude, lon: next.longitude, zoom: Math.max(mapZoom, 16) }) }
      } else if (e.key === 'h' || e.key === 'H') {
        setHeatmapMode(v => !v)
      } else if (selected && (e.key === 'r' || e.key === 'R')) {
        markFixed(selected, !selected.is_fixed)
      } else if (selected && (e.key === 'Delete' || e.key === 'Backspace')) {
        e.preventDefault(); deleteOne(selected)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [rendered, selected, mapZoom]) // eslint-disable-line react-hooks/exhaustive-deps

  const tiles = BASEMAPS[basemap]

  return (
    <div style={styles.page}>

      {/* ── Map ─────────────────────────────────────────────────────────── */}
      <MapContainer
        center={center}
        zoom={zoom}
        maxZoom={20}
        minZoom={3}
        preferCanvas={true}
        style={{ width: '100%', height: '100%', cursor: drawingMode ? 'crosshair' : 'grab' }}
        zoomControl={false}
      >
        <TileLayer key={basemap} url={tiles.url} attribution={tiles.attr} maxZoom={20} maxNativeZoom={19} />
        <ZoomTracker onZoom={setMapZoom} />
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
          <Rectangle bounds={finishedRect} pathOptions={{ color: '#cf5a30', fillColor: '#cf5a30', fillOpacity: 0.08, weight: 2 }} />
        )}
        {!finishedRect && rectStart && rectEnd && (
          <Rectangle bounds={[rectStart, rectEnd]} pathOptions={{ color: '#cf5a30', fillColor: '#cf5a30', fillOpacity: 0.08, weight: 2, dashArray: '4' }} />
        )}

        {rendered.map(d => {
          const sevColor = SEVERITY_COLORS[d.severity] || '#888'
          const isSel = selected && selected.id === d.id
          return (
            <CircleMarker
              key={d.id}
              center={[d.latitude, d.longitude]}
              radius={heatmapMode ? (d.severity * 8) : (isSel ? 11 : d.severity >= 4 ? 9 : d.severity === 3 ? 7 : 5)}
              pathOptions={{
                // Severity leads the fill (it is what dispatches a crew); repaired
                // reads as the earthed green. A thin ink ring gives every dot a
                // monograph hairline; selection turns that ring brick.
                color: heatmapMode ? 'transparent' : (isSel ? '#cf5a30' : 'rgba(26,26,24,0.55)'),
                fillColor: heatmapMode ? sevColor : (d.is_fixed ? '#7ba05b' : sevColor),
                fillOpacity: heatmapMode ? 0.3 : (d.is_fixed ? 0.5 : 0.9),
                weight: heatmapMode ? 0 : (isSel ? 3 : 1.5),
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

        {declutterActive && (
          <div className="glass" style={{
            padding: '4px 10px', borderRadius: 999, fontSize: 11,
            color: 'var(--text-dim)', border: '1px solid var(--border)',
          }}>
            Showing severity 3+ only. Zoom in to see everything.
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
        <button className="btn btn-sm glass" onClick={manualRefresh} disabled={refreshing} title="Refresh map data">
          <RefreshCw size={13} style={refreshing ? { animation: 'spin 0.8s linear infinite' } : undefined} />
          {lastRefresh && !isMobile && (
            <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {new Date(lastRefresh).toLocaleTimeString('en-GB')}
            </span>
          )}
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

            {/* Class chips — top 5 by count, the rest behind "more" */}
            <div style={styles.filterList}>
              {Object.entries(classCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, showAllClasses ? undefined : 5)
                .map(([cls, cnt]) => (
                  <ClassChip
                    key={cls}
                    cls={cls}
                    count={cnt}
                    active={activeClasses.has(cls)}
                    onClick={() => toggleClass(cls)}
                  />
                ))}
              {Object.keys(classCounts).length > 5 && (
                <button className="chip" onClick={() => setShowAllClasses(v => !v)}
                  style={{ color: 'var(--accent)', borderColor: 'var(--border-accent)' }}>
                  {showAllClasses ? 'Show less' : `+${Object.keys(classCounts).length - 5} more`}
                </button>
              )}
            </div>

            <div style={{ padding: '8px 14px 12px', borderTop: '1px solid var(--border)' }}>
              <Toggle checked={showFixed} onChange={setShowFixed} label="Show repaired" />
            </div>

            {/* Legend — decode the marker without recall */}
            <div style={{ padding: '2px 14px 12px', borderTop: '1px solid var(--border)' }}>
              <span className="overline" style={{ display: 'block', marginBottom: 8 }}>Legend</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}>
                {[1, 2, 3, 4, 5].map(s => (
                  <span key={s} title={SEVERITY_LABELS[s]} style={{ width: 12, height: 12, borderRadius: '50%', background: SEVERITY_COLORS[s], border: '1px solid rgba(26,26,24,0.35)' }} />
                ))}
                <span>dot colour = severity</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11, color: 'var(--text-dim)' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', background: '#7ba05b', border: '1px solid rgba(26,26,24,0.35)' }} />
                <span>green = repaired · larger = worse</span>
              </div>
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

        {/* Hand the zone straight to the repair board. WorkOrdersPage reads
            location.state.detectionIds and opens its create modal with them. */}
        {finishedRect && (
          <div style={{ padding: '10px 14px 12px', borderTop: '1px solid var(--border)' }}>
            <button
              className="btn btn-sm btn-accent"
              style={{ width: '100%', justifyContent: 'center' }}
              disabled={visible.length === 0}
              onClick={() => navigate('/workorders', {
                state: { detectionIds: visible.slice(0, 200).map(d => d.id) },
              })}
            >
              <Wrench size={13} /> Create work order from this zone
            </button>
            {visible.length > 200 && (
              <div style={{ marginTop: 6, fontSize: 10.5, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Only the first 200 sites in this zone will be added.
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Detail drawer (right) ────────────────────────────────────────── */}
      {selected && (
        <div
          role="dialog"
          aria-label={`${CLASS_LABELS[selected.damage_type] || selected.damage_type} detection detail`}
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
            {selected.reopened && (
              <div style={styles.reopenedWarn}>
                <div style={styles.reopenedTitle}>
                  <AlertTriangle size={12} /> Seen again after repair
                </div>
                <div style={styles.reopenedText}>
                  This site was marked repaired on {fmtDate(selected.fixed_at)} but damage was
                  detected here again.
                </div>
              </div>
            )}

            {selected.has_evidence && (evidenceLoading || evidenceUrl) && (
              <div style={{ marginBottom: 12 }}>
                {evidenceLoading ? (
                  // .skeleton has no intrinsic height — give the placeholder the
                  // photo's box so the drawer does not jump when it lands.
                  <div className="skeleton" style={{ ...styles.evidenceBox, height: 200 }} />
                ) : (
                  <img src={evidenceUrl} alt="" style={styles.evidenceBox} />
                )}
                <div style={styles.evidenceCaption}>
                  Photo captured by the survey pipeline.
                </div>
              </div>
            )}

            <KvRow k="Priority score" v={(selected.priority_score || 0).toFixed(4)} mono />
            <KvRow k="Times observed" v={`${selected.detection_count}×`} mono />
            <KvRow k="First detected" v={fmtDate(selected.first_detected)} />
            <KvRow k="Last detected" v={fmtDate(selected.last_detected)} />
            <KvRow k="Survey" v={selected.survey_video_file || '—'} mono />
            <KvRow k="Lighting" v={selected.lighting_condition || '—'} />
            <KvRow k="ID" v={String(selected.id).slice(0, 8) + '…'} mono />

            {/* The pipeline's raw signals sit below a fold — they are not an
                operator's decision inputs, so they no longer crowd the action. */}
            <button
              className="btn btn-sm btn-ghost"
              style={{ width: '100%', justifyContent: 'space-between', marginTop: 8, fontSize: 11 }}
              onClick={() => setShowTech(v => !v)}
              aria-expanded={showTech}
            >
              Technical measurements
              <ChevronDown size={12} style={{ transform: showTech ? 'rotate(180deg)' : 'none', transition: 'transform .18s' }} />
            </button>
            {showTech && (
              <div style={{ marginTop: 2 }}>
                <KvRow k="Surface area (mask px)" v={selected.surface_area_cm2 != null ? Math.round(selected.surface_area_cm2).toLocaleString() : '—'} mono />
                <KvRow k="Depth estimate" v={selected.depth_estimate_cm != null ? `${selected.depth_estimate_cm.toFixed(1)} (rel)` : '—'} mono />
                <KvRow k="Depth confidence" v={selected.depth_confidence != null ? fmtPct(selected.depth_confidence) : '—'} mono />
                <KvRow k="Edge sharpness" v={selected.edge_sharpness != null ? selected.edge_sharpness.toFixed(2) : '—'} mono />
                <KvRow k="Interior contrast" v={selected.interior_contrast != null ? selected.interior_contrast.toFixed(2) : '—'} mono />
                <KvRow k="Mask compactness" v={selected.mask_compactness != null ? selected.mask_compactness.toFixed(3) : '—'} mono />
                <KvRow k="Severity confidence" v={selected.severity_confidence != null ? fmtPct(selected.severity_confidence) : '—'} mono />
              </div>
            )}
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

      {/* ── Toasts (undo / success / error) ─────────────────────────────── */}
      {toasts.length > 0 && (
        <div style={styles.toastWrap} aria-live="polite">
          {toasts.map(t => (
            <div key={t.id} className="glass" style={{
              ...styles.toast,
              borderColor: t.tone === 'error' ? 'var(--red)' : t.tone === 'success' ? 'var(--green)' : 'var(--border-bright)',
            }}>
              <span style={{ color: t.tone === 'error' ? 'var(--red)' : t.tone === 'success' ? 'var(--green)' : 'var(--text)' }}>{t.text}</span>
              {t.actionLabel && (
                <button className="btn btn-sm btn-ghost" style={{ padding: '3px 10px' }}
                  onClick={() => { if (t.onAction) t.onAction(); dismissToast(t.id) }}>
                  {t.actionLabel}
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Empty state (fresh city, no data) ───────────────────────────── */}
      {!loading && !error && detections.length === 0 && (
        <div style={styles.emptyWrap}>
          <div className="glass" style={styles.emptyCard}>
            <EmptyState
              icon={MapPin}
              title={`No detections in ${user?.city || 'your city'} yet`}
              sub="Process a dashcam survey and the map fills with scored, ranked road damage."
              action={<button className="btn btn-accent btn-sm" onClick={() => navigate('/ingest')}>Upload a survey</button>}
            />
          </div>
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

  evidenceBox: {
    display: 'block',
    width: '100%',
    maxHeight: 200,
    objectFit: 'cover',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
  },
  evidenceCaption: {
    marginTop: 5,
    fontSize: 10.5,
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },

  reopenedWarn: {
    marginBottom: 12,
    padding: '8px 10px',
    background: 'rgba(255,93,93,0.1)',
    border: '1px solid var(--red)',
    borderRadius: 'var(--radius)',
    color: 'var(--red)',
  },
  reopenedTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    fontSize: 11,
    fontWeight: 700,
    fontFamily: 'var(--font-mono)',
    letterSpacing: '.05em',
    textTransform: 'uppercase',
  },
  reopenedText: {
    marginTop: 5,
    fontSize: 11,
    lineHeight: 1.55,
    opacity: 0.9,
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
  emptyWrap: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 850,
    pointerEvents: 'none',
  },
  emptyCard: {
    pointerEvents: 'auto',
    padding: '26px 30px',
    maxWidth: 400,
  },
  toastWrap: {
    position: 'absolute',
    bottom: 78,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 950,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    alignItems: 'center',
    pointerEvents: 'none',
  },
  toast: {
    pointerEvents: 'auto',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '9px 14px',
    fontSize: 12.5,
    minWidth: 220,
    justifyContent: 'space-between',
  },
}
