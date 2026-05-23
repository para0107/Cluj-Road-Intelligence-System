import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { BarChart, Bar, ResponsiveContainer, XAxis, Tooltip } from 'recharts'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents, Rectangle } from 'react-leaflet'
import { useNavigate } from 'react-router-dom'
import { BarChart2, FileText, RefreshCw, Eye, EyeOff, AlertTriangle, Map, PenTool, XCircle, Check, Radio } from 'lucide-react'
import {
  CLASS_COLORS, CLASS_LABELS, CLASS_ICONS,
  SEVERITY_COLORS, SEVERITY_LABELS,
  CLUJ_CENTER, CLUJ_ZOOM, TILE_URL, TILE_ATTR,
} from '../utils/constants'
import { fetchDetections, fetchStats, fetchJobStatus } from '../utils/api'

// ─── Live-update polling interval (ms) ────────────────────────────────────
// Matches the poll interval used on IngestionPage.
// The map silently refreshes detections from the DB each tick.
const LIVE_POLL_MS = 10_000

// ── Auto-fit map to data bounds ───────────────────────────────────────────
function FitBounds({ detections }) {
  const map = useMap()
  useEffect(() => {
    if (!detections || detections.length === 0) return
    const lats = detections.map(d => d.latitude)
    const lons = detections.map(d => d.longitude)
    const bounds = [
      [Math.min(...lats) - 0.002, Math.min(...lons) - 0.002],
      [Math.max(...lats) + 0.002, Math.max(...lons) + 0.002],
    ]
    map.fitBounds(bounds, { padding: [40, 40] })
  }, [detections, map])
  return null
}

// ── Point in Rectangle ──────────────────────────────────────────────────────
function isPointInRect(point, rect) {
  const [start, end] = rect;
  const latMin = Math.min(start[0], end[0]);
  const latMax = Math.max(start[0], end[0]);
  const lngMin = Math.min(start[1], end[1]);
  const lngMax = Math.max(start[1], end[1]);
  return point[0] >= latMin && point[0] <= latMax && point[1] >= lngMin && point[1] <= lngMax;
}

// ── Map Click Handler ─────────────────────────────────────────────────────
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
      if (drawingMode && isDrawing) {
        setEnd([e.latlng.lat, e.latlng.lng])
      }
    },
    mouseup(e) {
      if (drawingMode && isDrawing) {
        setIsDrawing(false)
        map.dragging.enable()
        finishDrawing()
      }
    }
  })

  useEffect(() => {
    if (!drawingMode) {
      map.dragging.enable()
      setIsDrawing(false)
    }
  }, [drawingMode, map])

  return null
}

// ── Class filter toggle pill ──────────────────────────────────────────────
function ClassPill({ cls, active, count, onToggle }) {
  const color = CLASS_COLORS[cls] || '#888'
  return (
    <button
      onClick={() => onToggle(cls)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        borderRadius: 20,
        border: `1px solid ${active ? color : 'var(--border)'}`,
        background: active ? `${color}18` : 'transparent',
        color: active ? color : 'var(--text-muted)',
        fontSize: 11,
        fontFamily: 'var(--font-sans)',
        cursor: 'pointer',
        transition: 'var(--transition)',
        whiteSpace: 'nowrap',
      }}
    >
      <span style={{
        width: 7, height: 7, borderRadius: '50%',
        background: active ? color : 'var(--text-muted)',
        flexShrink: 0,
      }} />
      {CLASS_LABELS[cls] || cls}
      <span style={{
        background: active ? `${color}30` : 'var(--border)',
        borderRadius: 8, padding: '0 5px',
        fontSize: 10, color: active ? color : 'var(--text-muted)',
      }}>
        {count}
      </span>
    </button>
  )
}

// ── Severity badge ────────────────────────────────────────────────────────
function SevBadge({ s }) {
  const color = SEVERITY_COLORS[s] || '#888'
  return (
    <span style={{
      background: `${color}25`, color, border: `1px solid ${color}50`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700,
    }}>
      {SEVERITY_LABELS[s] || `S${s}`}
    </span>
  )
}

// ── Report generator ──────────────────────────────────────────────────────
function generateReport(detections, stats) {
  const byClass = {}
  const bySev   = {}
  detections.forEach(d => {
    byClass[d.damage_type] = (byClass[d.damage_type] || 0) + 1
    bySev[d.severity]      = (bySev[d.severity]      || 0) + 1
  })

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RIDS — Detection Report</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; color: #1a1a2e; margin:0; padding:0; }
  .cover { background: #0a0c10; color: #e8ff47; padding: 60px 48px 40px; }
  .cover h1 { font-size: 36px; font-weight: 800; margin-bottom: 8px; letter-spacing:-1px; }
  .cover p { color: #9ca3af; font-size: 14px; margin: 0; }
  .body { padding: 40px 48px; }
  .section { margin-bottom: 36px; }
  h2 { font-size: 20px; font-weight: 700; border-bottom: 2px solid #e8ff47; padding-bottom: 8px; margin-bottom: 20px; }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
  .stat-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align:center; }
  .stat-val { font-size: 32px; font-weight: 800; color: #0a0c10; }
  .stat-lbl { font-size: 12px; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  th { background: #0a0c10; color: #e8ff47; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; padding: 12px 16px; text-align: left; }
  td { padding: 11px 16px; border-bottom: 1px solid #f1f3f5; font-size: 13px; }
  tr:last-child td { border: none; }
  .badge { display:inline-block; padding: 2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .footer { background:#0a0c10; color:#6b7280; padding: 24px 48px; font-size: 12px; text-align: center; margin-top: 40px; }
  @media print { .no-print { display: none; } }
</style>
</head>
<body>
<div class="cover">
  <h1>RIDS Detection Report</h1>
  <p>Road Infrastructure Detection System · Babeș-Bolyai University · Generated ${new Date().toLocaleString()}</p>
</div>
<div class="body">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val">${stats?.total_detections ?? detections.length}</div><div class="stat-lbl">Total Detections</div></div>
    <div class="stat-card"><div class="stat-val">${stats?.critical_count ?? 0}</div><div class="stat-lbl">Critical (S4–S5)</div></div>
    <div class="stat-card"><div class="stat-val">${stats?.avg_severity?.toFixed(1) ?? '—'}</div><div class="stat-lbl">Avg Severity</div></div>
    <div class="stat-card"><div class="stat-val">${stats?.last_survey_date ?? '—'}</div><div class="stat-lbl">Last Survey</div></div>
  </div>

  <div class="section">
    <h2>Detections by Class</h2>
    <table>
      <thead><tr><th>Class</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
        ${Object.entries(byClass).sort((a,b)=>b[1]-a[1]).map(([cls,cnt]) =>
          `<tr><td>${CLASS_LABELS[cls]||cls}</td><td><strong>${cnt}</strong></td><td>${((cnt/detections.length)*100).toFixed(1)}%</td></tr>`
        ).join('')}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Severity Distribution</h2>
    <table>
      <thead><tr><th>Level</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
        ${[1,2,3,4,5].map(s => {
          const cnt = bySev[s] || 0
          const colors = {1:'#4ade80',2:'#fbbf24',3:'#fb923c',4:'#f87171',5:'#a21caf'}
          return `<tr><td><span class="badge" style="background:${colors[s]}22;color:${colors[s]}">${SEVERITY_LABELS[s]}</span></td><td><strong>${cnt}</strong></td><td>${detections.length ? ((cnt/detections.length)*100).toFixed(1) : 0}%</td></tr>`
        }).join('')}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Top 30 Priority Detections</h2>
    <table>
      <thead><tr><th>#</th><th>Type</th><th>Severity</th><th>Priority</th><th>GPS</th><th>Date</th></tr></thead>
      <tbody>
        ${[...detections].sort((a,b)=>(b.priority_score||0)-(a.priority_score||0)).slice(0,30).map((d,i) => {
          const colors={1:'#4ade80',2:'#fbbf24',3:'#fb923c',4:'#f87171',5:'#a21caf'}
          const sc = colors[d.severity]||'#888'
          return `<tr>
            <td style="color:#6b7280;font-size:11px">${i+1}</td>
            <td>${CLASS_LABELS[d.damage_type]||d.damage_type}</td>
            <td><span class="badge" style="background:${sc}22;color:${sc}">S${d.severity}</span></td>
            <td style="font-family:monospace">${(d.priority_score||0).toFixed(4)}</td>
            <td style="font-family:monospace;font-size:11px">${d.latitude?.toFixed(5)}, ${d.longitude?.toFixed(5)}</td>
            <td style="font-size:11px;color:#6b7280">${d.last_detected||'—'}</td>
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

// ── Main MapPage ──────────────────────────────────────────────────────────
export default function MapPage() {
  const navigate = useNavigate()

  const [detections, setDetections] = useState([])
  const [stats,      setStats]      = useState(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [activeClasses, setActiveClasses] = useState(new Set())
  const [selected,   setSelected]   = useState(null)
  const [showLegend, setShowLegend] = useState(true)
  const [heatmapMode, setHeatmapMode] = useState(false)

  // Drawing state
  const [drawingMode, setDrawingMode] = useState(false)
  const [rectStart, setRectStart] = useState(null)
  const [rectEnd, setRectEnd] = useState(null)
  const [finishedRect, setFinishedRect] = useState(null)

  // ── Live update state ──────────────────────────────────────────────────
  // liveJobId: the currently running pipeline session_id, read from
  // localStorage. IngestionPage writes this key when a job starts so the
  // map can begin polling even if the user navigates away from IngestionPage.
  const [liveJobId,   setLiveJobId]   = useState(() => localStorage.getItem('rids_active_job') || null)
  const [liveActive,  setLiveActive]  = useState(false)  // true while job is running/initialising
  const [lastRefresh, setLastRefresh] = useState(null)   // ISO timestamp of last silent refresh
  const pollRef = useRef(null)

  // ── Data loading helpers ────────────────────────────────────────────────

  /**
   * Refresh detections + stats from the DB.
   * Called on initial mount AND silently during live polling.
   * Silent = does NOT set loading=true so the map doesn't flicker.
   */
  const refreshData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const [det, st] = await Promise.all([
        fetchDetections({ page: 1, page_size: 5000 }),
        fetchStats(),
      ])
      setDetections(det.items || [])
      setStats(st)
      // On first load, activate all known classes; on silent refresh, preserve user selection
      if (!silent) {
        const classes = new Set((det.items || []).map(d => d.damage_type))
        setActiveClasses(classes)
      }
      if (silent) setLastRefresh(new Date().toISOString())
    } catch (e) {
      if (!silent) setError(e.message)
      // On silent refresh failures, swallow — the user is not staring at a blank map
    } finally {
      if (!silent) setLoading(false)
    }
  }, [])

  // Initial load on mount
  useEffect(() => { refreshData(false) }, [refreshData])

  // ── Live polling effect ────────────────────────────────────────────────
  // Runs whenever liveJobId changes (set to null = stop polling).
  // Logic:
  //   1. Read the active job status.
  //   2. If status is running/initialising → stay active, refresh map data.
  //   3. If status is complete → do one final map refresh, clear job, stop.
  //   4. If status is failed/unknown → stop polling, leave data as-is.
  useEffect(() => {
    if (!liveJobId) {
      if (pollRef.current) clearInterval(pollRef.current)
      setLiveActive(false)
      return
    }

    setLiveActive(true)
    localStorage.setItem('rids_active_job', liveJobId)

    const tick = async () => {
      try {
        const jobData = await fetchJobStatus(liveJobId)
        const s = jobData.status

        if (s === 'running' || s === 'initialising') {
          // Pipeline still going — refresh map data silently
          refreshData(true)
        } else if (s === 'complete') {
          // Pipeline done — do one final refresh to get all new detections
          await refreshData(true)
          setLiveActive(false)
          setLiveJobId(null)
          localStorage.removeItem('rids_active_job')
          if (pollRef.current) clearInterval(pollRef.current)
        } else {
          // failed or unknown — stop polling
          setLiveActive(false)
          setLiveJobId(null)
          localStorage.removeItem('rids_active_job')
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch (err) {
        // 404 means job_id is no longer valid — stop quietly
        if (err?.response?.status === 404) {
          setLiveActive(false)
          setLiveJobId(null)
          localStorage.removeItem('rids_active_job')
          if (pollRef.current) clearInterval(pollRef.current)
        }
        // Network errors: keep trying
      }
    }

    tick()  // fire immediately
    pollRef.current = setInterval(tick, LIVE_POLL_MS)

    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [liveJobId, refreshData])

  const toggleClass = useCallback((cls) => {
    setActiveClasses(prev => {
      const next = new Set(prev)
      next.has(cls) ? next.delete(cls) : next.add(cls)
      return next
    })
  }, [])

  // Count per class across ALL detections
  const classCounts = {}
  detections.forEach(d => {
    classCounts[d.damage_type] = (classCounts[d.damage_type] || 0) + 1
  })

  const currentRect = finishedRect || (drawingMode && rectStart && rectEnd ? [rectStart, rectEnd] : null)

  const visible = detections.filter(d => {
    if (!activeClasses.has(d.damage_type)) return false
    if (currentRect) {
      return isPointInRect([d.latitude, d.longitude], currentRect)
    }
    return true
  })

  const displayStats = currentRect ? {
    total_detections: visible.length,
    critical_count: visible.filter(d => d.severity >= 4).length,
    avg_severity: visible.length ? (visible.reduce((acc, d) => acc + d.severity, 0) / visible.length) : 0,
    avg_confidence: visible.length ? (visible.reduce((acc, d) => acc + d.confidence, 0) / visible.length) : 0,
    last_survey_date: stats?.last_survey_date
  } : stats;

  const confidenceData = useMemo(() => {
    if (!finishedRect) return []
    const bins = { '20-40%':0, '40-60%':0, '60-80%':0, '80-100%':0 }
    visible.forEach(d => {
      const c = d.confidence
      if (c < 0.4) bins['20-40%']++
      else if (c < 0.6) bins['40-60%']++
      else if (c < 0.8) bins['60-80%']++
      else bins['80-100%']++
    })
    return Object.entries(bins).map(([name, count]) => ({ name, count }))
  }, [finishedRect, visible])

  return (
    <div style={styles.page}>

      {/* ── Map ─────────────────────────────────────────────────────── */}
      <MapContainer
        center={CLUJ_CENTER}
        zoom={CLUJ_ZOOM}
        maxZoom={20}
        minZoom={3}
        style={{ ...styles.map, cursor: drawingMode ? 'crosshair' : 'grab' }}
        zoomControl={false}
      >
        <TileLayer url={TILE_URL} attribution={TILE_ATTR} maxZoom={30} maxNativeZoom={19}/>
        <FitBounds detections={detections} />
        <MapClickHandler 
          drawingMode={drawingMode} 
          setStart={setRectStart}
          setEnd={setRectEnd}
          finishDrawing={() => {
            if (rectStart && rectEnd && rectStart[0] !== rectEnd[0]) {
              setFinishedRect([rectStart, rectEnd])
              setDrawingMode(false)
            } else {
              setDrawingMode(false)
            }
          }}
        />

        {finishedRect && (
          <Rectangle bounds={finishedRect} pathOptions={{ color: 'var(--accent)', fillColor: 'var(--accent)', fillOpacity: 0.1, weight: 2 }} />
        )}
        {!finishedRect && rectStart && rectEnd && (
          <Rectangle bounds={[rectStart, rectEnd]} pathOptions={{ color: 'var(--accent)', fillColor: 'var(--accent)', fillOpacity: 0.1, weight: 2, dashArray: '4' }} />
        )}

        {visible.map(d => {
          const color = CLASS_COLORS[d.damage_type] || '#888'
          const sevColor = SEVERITY_COLORS[d.severity] || color
          return (
            <CircleMarker
              key={d.id}
              center={[d.latitude, d.longitude]}
              radius={heatmapMode ? (d.severity * 8) : (d.severity >= 4 ? 9 : d.severity === 3 ? 7 : 5)}
              pathOptions={{
                color: heatmapMode ? 'transparent' : sevColor,
                fillColor: heatmapMode ? sevColor : color,
                fillOpacity: heatmapMode ? 0.3 : 0.85,
                weight: heatmapMode ? 0 : (d.severity >= 4 ? 2 : 1),
                className: heatmapMode ? 'heatmap-blob' : ''
              }}
              eventHandlers={{ click: () => !heatmapMode && setSelected(d) }}
            >
              {!heatmapMode && (
                <Popup>
                <div style={styles.popup}>
                  <div style={styles.popupHeader}>
                    <span style={{ color: CLASS_COLORS[d.damage_type] || '#888', fontSize: 18 }}>
                      {CLASS_ICONS[d.damage_type] || '●'}
                    </span>
                    <div>
                      <div style={styles.popupTitle}>
                        {CLASS_LABELS[d.damage_type] || d.damage_type}
                      </div>
                      <div style={styles.popupSub}>
                        {d.latitude?.toFixed(5)}, {d.longitude?.toFixed(5)}
                      </div>
                    </div>
                  </div>
                  <div style={styles.popupRow}>
                    <SevBadge s={d.severity} />
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      conf {(d.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {d.street_name && (
                    <div style={styles.popupStreet}>{d.street_name}</div>
                  )}
                  <div style={styles.popupMeta}>
                    <span>Seen {d.detection_count}×</span>
                    <span>Priority {(d.priority_score || 0).toFixed(3)}</span>
                  </div>
                </div>
              </Popup>
              )}
            </CircleMarker>
          )
        })}
      </MapContainer>

      {/* ── Top-right action buttons ─────────────────────────────────── */}
      <div style={styles.actions}>
        {!drawingMode && !finishedRect && (
          <button style={styles.actionBtn} onClick={() => setDrawingMode(true)}>
            <PenTool size={14} /> Draw Zone
          </button>
        )}
        {drawingMode && (
          <>
            <button 
              style={{ ...styles.actionBtn, color: 'var(--red)' }}
              onClick={() => {
                setDrawingMode(false)
                setRectStart(null)
                setRectEnd(null)
              }}
            >
              <XCircle size={14} /> Cancel
            </button>
          </>
        )}
        {finishedRect && (
          <button style={{ ...styles.actionBtn, color: 'var(--red)' }} onClick={() => {
            setFinishedRect(null)
            setRectStart(null)
            setRectEnd(null)
          }}>
            <XCircle size={14} /> Clear Zone
          </button>
        )}

        <button 
          style={{ ...styles.actionBtn, ...(heatmapMode ? styles.actionBtnActive : {}) }} 
          onClick={() => setHeatmapMode(!heatmapMode)}
        >
          <Map size={14} />
          {heatmapMode ? 'Points' : 'Heatmap'}
        </button>
        <button style={styles.actionBtn} onClick={() => navigate('/stats')}>
          <BarChart2 size={14} />
          Stats
        </button>
        {/* Manual refresh — always available */}
        <button
          style={styles.actionBtn}
          onClick={() => refreshData(true)}
          title="Refresh map data"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
        <button
          style={{ ...styles.actionBtn, ...styles.actionBtnAccent }}
          onClick={() => generateReport(detections, stats)}
          disabled={detections.length === 0}
        >
          <FileText size={14} />
          Report
        </button>
      </div>

      {/* ── Live update indicator ─────────────────────────────────────── */}
      {/* Shown only while a pipeline job is actively running.
          Sits top-left, unobtrusive, tells the operator the map is live. */}
      {liveActive && (
        <div style={styles.liveBadge}>
          <Radio size={11} style={{ animation: 'pulse 1.5s ease-in-out infinite' }} />
          LIVE · updating every {LIVE_POLL_MS / 1000}s
          {lastRefresh && (
            <span style={{ marginLeft: 6, opacity: 0.6 }}>
              · {new Date(lastRefresh).toLocaleTimeString()}
            </span>
          )}
        </div>
      )}

      {/* ── Bottom stat strip ────────────────────────────────────────── */}
      {displayStats && (
        <div style={styles.statStrip}>
          <StatChip label={currentRect ? "Zone Total" : "Total"} value={displayStats.total_detections} color="var(--accent)" />
          <div style={styles.stripDivider} />
          <StatChip label="Critical" value={displayStats.critical_count} color="var(--red)" />
          <div style={styles.stripDivider} />
          <StatChip label="Avg Severity" value={typeof displayStats.avg_severity === 'number' ? displayStats.avg_severity.toFixed(1) : (displayStats.avg_severity ?? '—')} color="var(--orange)" />
          <div style={styles.stripDivider} />
          <StatChip label="Avg Conf" value={typeof displayStats.avg_confidence === 'number' ? `${(displayStats.avg_confidence * 100).toFixed(0)}%` : '—'} color="var(--blue)" />
          <div style={styles.stripDivider} />
          <StatChip label="Visible" value={visible.length} color="#ffffff" />
          {displayStats.last_survey_date && (
            <>
              <div style={styles.stripDivider} />
              <StatChip label="Last Survey" value={displayStats.last_survey_date} color="var(--text-muted)" />
            </>
          )}
        </div>
      )}

      {/* ── Class filter panel ───────────────────────────────────────── */}
      <div style={styles.filterPanel}>
        <div style={styles.filterHeader}>
          <span style={styles.filterTitle}>LAYERS</span>
          <div style={styles.filterHeaderBtns}>
            <button style={styles.tinyBtn}
              onClick={() => setActiveClasses(new Set(Object.keys(classCounts)))}>
              <Eye size={11} /> ALL
            </button>
            <button style={styles.tinyBtn}
              onClick={() => setActiveClasses(new Set())}>
              <EyeOff size={11} /> NONE
            </button>
            <button style={styles.tinyBtn}
              onClick={() => setShowLegend(v => !v)}>
              {showLegend ? '▲' : '▼'}
            </button>
          </div>
        </div>
        {showLegend && (
          <div style={styles.filterList}>
            {Object.entries(classCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, cnt]) => (
                <ClassPill
                  key={cls}
                  cls={cls}
                  count={cnt}
                  active={activeClasses.has(cls)}
                  onToggle={toggleClass}
                />
              ))}
          </div>
        )}
        {finishedRect && showLegend && (
          <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)' }}>
            <span style={{ ...styles.filterTitle, display: 'block', marginBottom: 8 }}>ZONE CONFIDENCE</span>
            <div style={{ height: 100, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceData}>
                  <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 9 }} interval={0} />
                  <Tooltip 
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', fontSize: 11 }}
                    itemStyle={{ color: 'var(--accent)' }}
                  />
                  <Bar dataKey="count" fill="var(--accent)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* ── Loading overlay ──────────────────────────────────────────── */}
      {loading && (
        <div style={styles.overlay}>
          <div style={styles.overlayContent}>
            <div style={styles.spinner} />
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 13 }}>
              Loading detections…
            </span>
          </div>
        </div>
      )}

      {/* ── Error banner ─────────────────────────────────────────────── */}
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
      <div style={{ fontSize: 16, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.06em' }}>
        {label}
      </div>
    </div>
  )
}

const styles = {
  page: {
    position: 'fixed',
    inset: '48px 0 0 0',
    overflow: 'hidden',
  },
  map: {
    width: '100%',
    height: '100%',
  },

  // Action buttons
  actions: {
    position: 'absolute',
    top: 16,
    right: 16,
    zIndex: 800,
    display: 'flex',
    gap: 8,
  },
  actionBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 14px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)',
    color: 'var(--text)',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'var(--transition)',
    backdropFilter: 'blur(8px)',
  },
  actionBtnActive: {
    background: 'var(--accent-dim)',
    border: '1px solid var(--accent)',
    color: 'var(--accent)',
  },
  actionBtnAccent: {
    background: 'var(--accent)',
    border: '1px solid var(--accent)',
    color: '#0a0c10',
  },

  // Live badge
  liveBadge: {
    position: 'absolute',
    top: 16,
    left: 16,
    zIndex: 800,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    background: 'rgba(232,255,71,0.08)',
    border: '1px solid rgba(232,255,71,0.35)',
    borderRadius: 20,
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    color: 'var(--accent)',
    letterSpacing: '.06em',
    backdropFilter: 'blur(8px)',
  },

  // Bottom stat strip
  statStrip: {
    position: 'absolute',
    bottom: 16,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 800,
    display: 'flex',
    alignItems: 'center',
    gap: 20,
    padding: '10px 24px',
    background: 'rgba(17,19,24,0.92)',
    border: '1px solid var(--border-bright)',
    borderRadius: 40,
    backdropFilter: 'blur(12px)',
    boxShadow: 'var(--shadow-lg)',
  },
  stripDivider: {
    width: 1,
    height: 24,
    background: 'var(--border)',
  },

  // Filter panel
  filterPanel: {
    position: 'absolute',
    bottom: 80,
    left: 16,
    zIndex: 800,
    background: 'rgba(17,19,24,0.94)',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius-lg)',
    backdropFilter: 'blur(12px)',
    maxWidth: 340,
    boxShadow: 'var(--shadow)',
  },
  filterHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    borderBottom: '1px solid var(--border)',
  },
  filterTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    color: 'var(--text-muted)',
    letterSpacing: '.12em',
  },
  filterHeaderBtns: { display: 'flex', gap: 4 },
  tinyBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 3,
    padding: '3px 7px',
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: 4,
    color: 'var(--text-muted)',
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    cursor: 'pointer',
    transition: 'var(--transition)',
  },
  filterList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    padding: '10px 14px',
    maxHeight: 200,
    overflowY: 'auto',
  },

  // Popup
  popup: {
    minWidth: 200,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  popupHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  popupTitle: {
    fontWeight: 600,
    fontSize: 13,
    color: 'var(--text)',
  },
  popupSub: {
    fontSize: 10,
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },
  popupRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  popupStreet: {
    fontSize: 11,
    color: 'var(--text-dim)',
    borderTop: '1px solid var(--border)',
    paddingTop: 6,
  },
  popupMeta: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 10,
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },

  // Overlays
  overlay: {
    position: 'absolute',
    inset: 0,
    background: 'rgba(10,12,16,0.75)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 900,
    backdropFilter: 'blur(4px)',
  },
  overlayContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 16,
  },
  spinner: {
    width: 32,
    height: 32,
    border: '3px solid var(--border)',
    borderTop: '3px solid var(--accent)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
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
    background: 'rgba(255,68,68,0.12)',
    border: '1px solid rgba(255,68,68,0.4)',
    borderRadius: 'var(--radius)',
    color: 'var(--red)',
    fontSize: 12,
    fontWeight: 600,
    backdropFilter: 'blur(8px)',
  },
}
