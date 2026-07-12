/**
 * frontend/src/components/WorkOrderRoutePanel.jsx
 *
 * Crew route for one work order: the stops on a map, in order, with the
 * distance and the time a crew should expect to spend.
 *
 * Ordering is done client-side (planRoute — nearest neighbour + 2-opt on
 * straight-line distance), so no third party ever sees the city's repair
 * schedule and the whole thing stays free. Distances are straight-line, so
 * treat them as a planning aid, not a driving estimate.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, TileLayer, Polyline, Marker, useMap } from 'react-leaflet'
import L from 'leaflet'
import { Route, Printer, Wand2, MapPin, Clock, Milestone } from 'lucide-react'
import { planRoute, legKms, totalKm as routeTotalKm } from '../utils/routePlan'
import { useIsDark } from '../hooks/useTheme'
import { BASEMAPS, SEVERITY_COLORS, CLASS_LABELS, CITY_ZOOM } from '../utils/constants'
import { EmptyState } from './ui'

// Same assumptions the planner uses, kept here so the numbers shown next to
// the map and the numbers planRoute optimises for never drift apart.
const MINUTES_PER_STOP = 20
const AVG_SPEED_KMH = 25

const ACCENT_FALLBACK = '#eaff3d'

/** Leaflet needs a real colour, not a CSS variable. Read the token at render. */
function readAccent() {
  if (typeof window === 'undefined') return ACCENT_FALLBACK
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue('--accent')
    return (v || '').trim() || ACCENT_FALLBACK
  } catch (e) {
    return ACCENT_FALLBACK
  }
}

const hasCoords = (p) =>
  Number.isFinite(p?.latitude) && Number.isFinite(p?.longitude)

/** "about 2 h 30 m" — a plain-language estimate, not a promise. */
function fmtMinutes(mins) {
  const m = Math.max(0, Math.round(mins || 0))
  if (m < 60) return `about ${m} m`
  const h = Math.floor(m / 60)
  const rest = m % 60
  return rest === 0 ? `about ${h} h` : `about ${h} h ${rest} m`
}

const fmtKm = (km) => `${(km || 0).toFixed(1)} km`

/** Numbered stop pin, coloured by severity. */
function stopIcon(n, color) {
  return L.divIcon({
    className: 'wo-route-stop',
    html:
      `<div style="width:24px;height:24px;border-radius:50%;background:${color};` +
      `color:#0a0d05;border:2px solid rgba(0,0,0,0.45);box-shadow:0 2px 6px rgba(0,0,0,0.5);` +
      `font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;` +
      `line-height:20px;text-align:center">${n}</div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })
}

/** Fit the view to the stops whenever the order (or the set of stops) changes. */
function FitStops({ points }) {
  const map = useMap()
  const signature = points.map((p) => `${p.latitude},${p.longitude}`).join('|')

  useEffect(() => {
    // The panel often mounts inside a drawer that is still animating.
    const t = setTimeout(() => {
      map.invalidateSize()
      if (points.length === 0) return
      if (points.length === 1) {
        map.setView([points[0].latitude, points[0].longitude], 16)
        return
      }
      const lats = points.map((p) => p.latitude)
      const lons = points.map((p) => p.longitude)
      map.fitBounds(
        [
          [Math.min(...lats), Math.min(...lons)],
          [Math.max(...lats), Math.max(...lons)],
        ],
        { padding: [40, 40], maxZoom: 17 },
      )
    }, 180)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, map])

  return null
}

export default function WorkOrderRoutePanel({ items, onSaveOrder, orderTitle, crewName }) {
  const isDark = useIsDark()
  const tiles = isDark ? BASEMAPS.dark : BASEMAPS.voyager

  const [order, setOrder] = useState([])
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  // Start from whatever order the work order was saved with.
  const savedSignature = (items || [])
    .map((it) => `${it.detection_id}:${it.sort_order}`)
    .join('|')

  useEffect(() => {
    const next = [...(items || [])].sort(
      (a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0),
    )
    setOrder(next)
    setSaveError('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedSignature])

  const stops = useMemo(() => order.filter(hasCoords), [order])

  const metrics = useMemo(() => {
    const legs = legKms(stops)
    const km = routeTotalKm(stops)
    const driveMinutes = AVG_SPEED_KMH > 0 ? (km / AVG_SPEED_KMH) * 60 : 0
    const cumulative = []
    let running = 0
    legs.forEach((leg) => {
      running += leg
      cumulative.push(running)
    })
    return {
      legs,
      cumulative,
      km,
      minutes: Math.round(driveMinutes + stops.length * MINUTES_PER_STOP),
    }
  }, [stops])

  const accent = readAccent()

  const optimize = useCallback(async () => {
    const plan = planRoute(order, {
      minutesPerStop: MINUTES_PER_STOP,
      avgSpeedKmh: AVG_SPEED_KMH,
    })
    if (plan.stops.length === 0) return

    // Sites without GPS keep their place at the end instead of vanishing.
    const placed = new Set(plan.stops.map((s) => s.detection_id))
    const rest = (order || []).filter((it) => !placed.has(it.detection_id))
    const next = [...plan.stops, ...rest]
    setOrder(next)

    if (!onSaveOrder) return
    setSaveError('')
    setSaving(true)
    try {
      await onSaveOrder(next.map((it) => it.detection_id))
    } catch (e) {
      if (mounted.current) setSaveError('The new order could not be saved. Try again.')
    } finally {
      if (mounted.current) setSaving(false)
    }
  }, [order, onSaveOrder])

  const printSheet = () => {
    const rows = stops
    const sevColors = { 1: '#3ddc84', 2: '#d4a900', 3: '#e08a30', 4: '#e04848', 5: '#a21caf' }
    const legs = metrics.legs
    const cum = metrics.cumulative

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>RDDS route sheet</title>
<style>
  body { font-family:'Segoe UI',sans-serif; color:#10141c; background:#ffffff; margin:0; }
  .head { padding:34px 40px 18px; border-bottom:3px solid #10141c; }
  .head h1 { margin:0 0 6px; font-size:24px; }
  .head p { margin:0; color:#55607a; font-size:13px; }
  .body { padding:26px 40px 40px; }
  table { width:100%; border-collapse:collapse; }
  th { background:#10141c; color:#ffffff; text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; padding:9px 12px; }
  td { padding:9px 12px; border-bottom:1px solid #e6e9ef; font-size:12.5px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .totals { margin-top:20px; display:flex; gap:28px; font-size:14px; font-weight:700; }
  .note { font-size:10.5px; color:#7a8296; margin-top:24px; }
</style></head><body>
<div class="head">
  <h1>Route sheet: ${escapeHtml(orderTitle || 'Work order')}</h1>
  <p>Crew: ${escapeHtml(crewName || 'unassigned')} · ${rows.length} stops · printed ${new Date().toLocaleString()}</p>
</div>
<div class="body">
<table>
<thead><tr><th>#</th><th>Damage</th><th>Severity</th><th>GPS</th><th>Leg</th><th>Total so far</th></tr></thead>
<tbody>
${rows.map((it, i) => {
  const sc = sevColors[it.severity] || '#888'
  return `<tr>
    <td style="font-family:monospace">${i + 1}</td>
    <td>${escapeHtml(CLASS_LABELS[it.damage_type] || it.damage_type || '—')}</td>
    <td><span class="badge" style="background:${sc}22;color:${sc}">S${it.severity ?? '—'}</span></td>
    <td style="font-family:monospace;font-size:11px">${it.latitude.toFixed(5)}, ${it.longitude.toFixed(5)}</td>
    <td style="font-family:monospace">${(legs[i] || 0).toFixed(2)} km</td>
    <td style="font-family:monospace">${(cum[i] || 0).toFixed(2)} km</td>
  </tr>`
}).join('')}
</tbody></table>
<div class="totals">
  <span>Total distance: ${metrics.km.toFixed(1)} km</span>
  <span>Estimated time: ${fmtMinutes(metrics.minutes)}</span>
  <span>Stops: ${rows.length}</span>
</div>
<p class="note">
  Distances are straight-line between stops, and the time assumes ${AVG_SPEED_KMH} km/h of city driving
  plus ${MINUTES_PER_STOP} minutes on each site. Use it to sequence the day, not to promise an arrival time.
</p>
</div>
<script>window.onload = () => window.print()</script>
</body></html>`

    const w = window.open('', '_blank')
    if (!w) return
    w.document.write(html)
    w.document.close()
  }

  if (!items || items.length === 0) {
    return (
      <EmptyState
        icon={Route}
        title="No stops to route yet"
        sub="Add damage sites to this work order and the crew route shows up here."
      />
    )
  }

  if (stops.length === 0) {
    return (
      <EmptyState
        icon={MapPin}
        title="No GPS on these sites"
        sub="A route needs coordinates. These sites were recorded without a GPS fix."
      />
    )
  }

  const center = [stops[0].latitude, stops[0].longitude]

  return (
    <div style={styles.wrap}>
      {/* ── Summary + actions ─────────────────────────────────────────── */}
      <div style={styles.bar}>
        <div style={styles.metrics}>
          <span style={styles.metric}>
            <Milestone size={13} style={{ color: 'var(--accent)' }} />
            <span className="mono" style={styles.metricValue}>{fmtKm(metrics.km)}</span>
          </span>
          <span style={styles.metric}>
            <Clock size={13} style={{ color: 'var(--cyan)' }} />
            <span className="mono" style={styles.metricValue}>{fmtMinutes(metrics.minutes)}</span>
          </span>
          <span style={styles.metric}>
            <MapPin size={13} style={{ color: 'var(--text-muted)' }} />
            <span className="mono" style={styles.metricValue}>
              {stops.length} {stops.length === 1 ? 'stop' : 'stops'}
            </span>
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-sm" onClick={optimize} disabled={saving || stops.length < 2}>
            <Wand2 size={13} /> {saving ? 'Saving…' : 'Optimize order'}
          </button>
          <button className="btn btn-sm" onClick={printSheet}>
            <Printer size={13} /> Print route sheet
          </button>
        </div>
      </div>

      {saveError && <div style={styles.error}>{saveError}</div>}

      {/* ── Map ───────────────────────────────────────────────────────── */}
      <div style={styles.mapBox}>
        <MapContainer
          center={center}
          zoom={CITY_ZOOM}
          preferCanvas
          scrollWheelZoom={false}
          style={{ width: '100%', height: '100%' }}
        >
          <TileLayer key={tiles.url} url={tiles.url} attribution={tiles.attr} maxZoom={20} maxNativeZoom={19} />
          <FitStops points={stops} />

          <Polyline
            positions={stops.map((p) => [p.latitude, p.longitude])}
            pathOptions={{ color: accent, weight: 3, opacity: 0.9 }}
          />

          {stops.map((p, i) => (
            <Marker
              key={p.detection_id ?? `${p.latitude},${p.longitude},${i}`}
              position={[p.latitude, p.longitude]}
              icon={stopIcon(i + 1, SEVERITY_COLORS[p.severity] || '#94a3b8')}
            />
          ))}
        </MapContainer>
      </div>

      <div style={styles.foot}>
        Straight-line distances between stops. The time assumes {AVG_SPEED_KMH} km/h in the city
        and {MINUTES_PER_STOP} minutes on each site.
      </div>
    </div>
  )
}

/** Small guard so a damage label or title can never break the printed page. */
function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

const styles = {
  wrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  bar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  metrics: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    flexWrap: 'wrap',
  },
  metric: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
  },
  metricValue: {
    fontSize: 12.5,
    fontWeight: 700,
    color: 'var(--text)',
  },
  mapBox: {
    height: 360,
    width: '100%',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border)',
    overflow: 'hidden',
  },
  foot: {
    fontSize: 10.5,
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },
  error: {
    background: 'rgba(255, 93, 93, 0.10)',
    border: '1px solid rgba(255, 93, 93, 0.4)',
    color: 'var(--red)',
    borderRadius: 'var(--radius)',
    padding: '8px 12px',
    fontSize: 12,
  },
}
