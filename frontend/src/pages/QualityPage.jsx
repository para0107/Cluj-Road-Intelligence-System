/**
 * frontend/src/pages/QualityPage.jsx
 *
 * Road Quality Index — the sellable view of the network's condition.
 *
 * The city is cut into a grid (about 120 m per square). Every square gets a
 * 0-100 score and an A-E band from the severity, the recency and the repair
 * state of the damage inside it. Lower score = worse road.
 *
 *  - cells are fetched for the CURRENT viewport only (debounced on moveend)
 *  - the backend answers 413 when the requested area holds too many cells,
 *    which we surface as a "zoom in or pick a bigger square" panel
 *  - CSV / GeoJSON export of exactly what is on screen
 *
 * Deliberately lean: no WebGL backgrounds, no entry animations. The map needs
 * the frames.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, Rectangle, Popup, useMap, useMapEvents } from 'react-leaflet'
import {
  Gauge, RefreshCw, Download, AlertTriangle, Layers, ArrowRight, MapPin,
} from 'lucide-react'
import { BASEMAPS, RQI_BANDS, CITY_ZOOM } from '../utils/constants'
import { fetchQualityGrid, downloadQualityExport } from '../utils/api'
import { SevBadge, KvRow } from '../components/ui'
import { useIsDark } from '../hooks/useTheme'
import useCityCenter from '../hooks/useCityCenter'
import useIsMobile from '../hooks/useIsMobile'

// Refetch is debounced so a drag does not fire one request per frame.
const DEBOUNCE_MS = 600

const CELL_SIZES = [80, 120, 200, 400]

// Band order, worst-last, plus the score window each band covers.
const BAND_ORDER = ['A', 'B', 'C', 'D', 'E']
const BAND_RANGE = {
  A: '85-100',
  B: '70-84',
  C: '50-69',
  D: '30-49',
  E: '0-29',
}

const METERS_PER_DEG_LAT = 111320

/** Square bounds around a cell centroid, in degrees. */
function cellBounds(lat, lon, cellM) {
  const dLat = cellM / METERS_PER_DEG_LAT
  // Longitude degrees shrink towards the poles; never divide by ~0.
  const cos = Math.max(0.01, Math.abs(Math.cos((lat * Math.PI) / 180)))
  const dLon = cellM / (METERS_PER_DEG_LAT * cos)
  return [
    [lat - dLat / 2, lon - dLon / 2],
    [lat + dLat / 2, lon + dLon / 2],
  ]
}

/** "minLon,minLat,maxLon,maxLat" from a Leaflet bounds object. */
function bboxOf(map) {
  const b = map.getBounds()
  const minLon = Math.max(-180, b.getWest())
  const minLat = Math.max(-90, b.getSouth())
  const maxLon = Math.min(180, b.getEast())
  const maxLat = Math.min(90, b.getNorth())
  return [minLon, minLat, maxLon, maxLat].map(v => v.toFixed(6)).join(',')
}

function errText(e) {
  const d = e?.response?.data?.detail
  if (typeof d === 'string') return d
  if (d?.message) return d.message
  return e?.message || 'Something went wrong.'
}

// ── Reports the viewport up on every move / zoom (and once on mount) ───────
function ViewportWatcher({ onBounds }) {
  const map = useMapEvents({
    moveend: () => onBounds(bboxOf(map)),
    zoomend: () => onBounds(bboxOf(map)),
  })
  useEffect(() => { onBounds(bboxOf(map)) }, [map, onBounds])
  return null
}

// ── One-shot glide to the operator's city once it geocodes ────────────────
function FlyToCity({ target }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo(target, CITY_ZOOM, { duration: 0.9 })
  }, [target, map])
  return null
}

export default function QualityPage() {
  const navigate = useNavigate()
  const isDark = useIsDark()
  const isMobile = useIsMobile()
  const { center, zoom, cityCenter } = useCityCenter()

  const [bbox, setBbox] = useState(null)
  const [cellM, setCellM] = useState(120)

  const [cells, setCells] = useState([])
  const [gridCellM, setGridCellM] = useState(120)   // cell size the API actually used
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [tooLarge, setTooLarge] = useState(false)

  const mounted = useRef(true)
  const reqId = useRef(0)
  useEffect(() => () => { mounted.current = false }, [])

  const load = useCallback(async (bb, cm) => {
    if (!bb) return
    const id = ++reqId.current
    setLoading(true)
    setError(null)
    try {
      const data = await fetchQualityGrid(bb, cm)
      if (!mounted.current || id !== reqId.current) return
      setCells(Array.isArray(data?.cells) ? data.cells : [])
      setGridCellM(data?.cell_m || cm)
      setTooLarge(false)
    } catch (e) {
      if (!mounted.current || id !== reqId.current) return
      if (e?.response?.status === 413) {
        setTooLarge(true)
        setCells([])
      } else {
        setError(errText(e))
      }
    } finally {
      if (mounted.current && id === reqId.current) setLoading(false)
    }
  }, [])

  // Debounced viewport fetch.
  useEffect(() => {
    if (!bbox) return undefined
    const t = setTimeout(() => { load(bbox, cellM) }, DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [bbox, cellM, load])

  const onBounds = useCallback((bb) => { setBbox(bb) }, [])

  // ── Derived summary of what is on screen ────────────────────────────────
  const summary = useMemo(() => {
    if (cells.length === 0) return { avg: null, bands: {} }
    let sum = 0
    const bands = {}
    cells.forEach(c => {
      sum += c.score || 0
      bands[c.band] = (bands[c.band] || 0) + 1
    })
    return { avg: sum / cells.length, bands }
  }, [cells])

  const rects = useMemo(() => cells.map((c, i) => ({
    key: `${c.lat.toFixed(5)}_${c.lon.toFixed(5)}_${i}`,
    cell: c,
    bounds: cellBounds(c.lat, c.lon, gridCellM),
    color: RQI_BANDS[c.band]?.color || 'var(--text-muted)',
  })), [cells, gridCellM])

  const tiles = BASEMAPS[isDark ? 'dark' : 'voyager']
  const avgColor = summary.avg == null
    ? 'var(--text)'
    : (RQI_BANDS[BAND_ORDER.find(b => summary.avg >= RQI_BANDS[b].min) || 'E']?.color || 'var(--text)')

  return (
    <div style={styles.page}>

      {/* ── Map ─────────────────────────────────────────────────────────── */}
      <MapContainer
        center={center}
        zoom={zoom}
        maxZoom={20}
        minZoom={3}
        preferCanvas={true}
        zoomControl={false}
        style={{ width: '100%', height: '100%' }}
      >
        <TileLayer key={isDark ? 'dark' : 'voyager'} url={tiles.url} attribution={tiles.attr} maxZoom={20} maxNativeZoom={19} />
        <ViewportWatcher onBounds={onBounds} />
        <FlyToCity target={cityCenter} />

        {rects.map(({ key, cell, bounds, color }) => (
          <Rectangle
            key={key}
            bounds={bounds}
            pathOptions={{
              color,
              fillColor: color,
              fillOpacity: 0.45,
              weight: 0.5,
              opacity: 0.7,
            }}
          >
            <Popup>
              <div style={{ minWidth: 208 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <span style={{
                    width: 38, height: 38, borderRadius: 10, flexShrink: 0,
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    background: `${color}22`, border: `1px solid ${color}66`,
                    color, fontFamily: 'var(--font-display)', fontSize: 17, fontWeight: 700,
                  }}>
                    {cell.band}
                  </span>
                  <div>
                    <div className="display" style={{ fontSize: 18, fontWeight: 700, color: 'var(--text)', lineHeight: 1.1 }}>
                      {Math.round(cell.score)}
                      <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 500 }}> / 100</span>
                    </div>
                    <div style={{ fontSize: 11.5, color: 'var(--text-dim)' }}>
                      {RQI_BANDS[cell.band]?.label || 'Unknown'}
                    </div>
                  </div>
                </div>

                <KvRow k="Detections" v={`${cell.n ?? 0}`} mono />
                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border)', fontSize: 12,
                }}>
                  <span style={{ color: 'var(--text-muted)' }}>Worst severity</span>
                  {cell.worst ? <SevBadge s={cell.worst} compact /> : <span style={{ color: 'var(--text)' }}>None</span>}
                </div>
                <KvRow k="Square" v={`${gridCellM} m`} mono />

                <button
                  className="btn btn-sm btn-accent"
                  style={{ width: '100%', marginTop: 10, justifyContent: 'center' }}
                  onClick={() => navigate('/map', { state: { focus: { id: null, lat: cell.lat, lon: cell.lon } } })}
                >
                  <MapPin size={12} /> Open this area on the map <ArrowRight size={12} />
                </button>
              </div>
            </Popup>
          </Rectangle>
        ))}
      </MapContainer>

      {/* ── Control panel (top-left) ────────────────────────────────────── */}
      <div className="glass" style={{ ...styles.controls, ...(isMobile ? { left: 8, right: 8, top: 8 } : null) }}>
        <div style={styles.controlHeader}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
            <Gauge size={14} style={{ color: 'var(--accent)' }} />
            <span className="overline">Road Quality Index</span>
          </span>
          <button
            className="btn btn-sm btn-ghost"
            style={styles.iconBtn}
            onClick={() => load(bbox, cellM)}
            disabled={!bbox}
            title="Score this view again"
          >
            <RefreshCw size={12} />
          </button>
        </div>

        <div style={{ padding: '10px 14px 12px' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 7 }}>
            Square size
          </div>
          <div style={styles.sizeRow}>
            {CELL_SIZES.map(m => {
              const active = cellM === m
              return (
                <button
                  key={m}
                  onClick={() => setCellM(m)}
                  className="mono"
                  style={{
                    flex: 1, padding: '5px 0', borderRadius: 6, cursor: 'pointer',
                    fontSize: 10.5, fontWeight: 700, transition: 'var(--transition)',
                    border: `1px solid ${active ? 'var(--border-accent)' : 'var(--border)'}`,
                    background: active ? 'var(--accent-dim)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--text-muted)',
                  }}
                >
                  {m}m
                </button>
              )
            })}
          </div>

          <div style={styles.countRow}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)' }}>
              <Layers size={12} />
              {cells.length === 1 ? '1 square scored' : `${cells.length.toLocaleString()} squares scored`}
            </span>
            {loading && (
              <span className="mono" style={styles.scoringChip}>
                <span style={styles.scoringDot} /> Scoring…
              </span>
            )}
          </div>

          <div style={styles.exportRow}>
            <button
              className="btn btn-sm"
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => downloadQualityExport('csv', bbox)}
              disabled={!bbox || cells.length === 0}
            >
              <Download size={12} /> Export CSV
            </button>
            <button
              className="btn btn-sm"
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => downloadQualityExport('geojson', bbox)}
              disabled={!bbox || cells.length === 0}
            >
              <Download size={12} /> GeoJSON
            </button>
          </div>
        </div>

        {tooLarge && (
          <div style={styles.notice}>
            This area is too large to score at this detail. Zoom in or choose a bigger cell size.
          </div>
        )}
      </div>

      {/* ── Summary strip (bottom centre) ───────────────────────────────── */}
      {cells.length > 0 && (
        <div
          className="glass"
          style={{
            ...styles.summary,
            ...(isMobile ? { left: 8, right: 8, transform: 'none', bottom: 8, gap: 10, padding: '8px 12px', overflowX: 'auto' } : null),
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: avgColor }}>
              {summary.avg != null ? summary.avg.toFixed(1) : '—'}
            </div>
            <div style={styles.summaryLabel}>Average score</div>
          </div>
          <div style={styles.divider} />
          <div style={{ display: 'flex', gap: 6 }}>
            {BAND_ORDER.map(b => {
              const n = summary.bands[b] || 0
              const color = RQI_BANDS[b].color
              return (
                <span
                  key={b}
                  className="mono"
                  title={`${RQI_BANDS[b].label} (${BAND_RANGE[b]})`}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    padding: '3px 8px', borderRadius: 999, fontSize: 10.5, fontWeight: 700,
                    border: `1px solid ${n ? `${color}66` : 'var(--border)'}`,
                    background: n ? `${color}18` : 'transparent',
                    color: n ? color : 'var(--text-muted)',
                  }}
                >
                  {b} {n}
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Legend (bottom-right) ───────────────────────────────────────── */}
      {!isMobile && (
        <div className="glass" style={styles.legend}>
          <div style={styles.legendIntro}>
            Each square scores about {gridCellM} m of road. Lower scores mean more damage, worse
            severity, and more recent sightings.
          </div>
          <div style={{ padding: '10px 14px 12px' }}>
            {BAND_ORDER.map(b => (
              <div key={b} style={styles.legendRow}>
                <span style={{
                  width: 14, height: 14, borderRadius: 4, flexShrink: 0,
                  background: RQI_BANDS[b].color, opacity: 0.75,
                  border: `1px solid ${RQI_BANDS[b].color}`,
                }} />
                <span className="mono" style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text)', width: 12 }}>{b}</span>
                <span style={{ fontSize: 11.5, color: 'var(--text-dim)', flex: 1 }}>{RQI_BANDS[b].label}</span>
                <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>{BAND_RANGE[b]}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Error banner ────────────────────────────────────────────────── */}
      {error && (
        <div className="glass" style={styles.errorBanner}>
          <AlertTriangle size={14} />
          <span>Could not score this view: {error}</span>
          <button className="btn btn-sm btn-ghost" style={{ marginLeft: 4 }} onClick={() => load(bbox, cellM)}>
            Try again
          </button>
        </div>
      )}
    </div>
  )
}

const styles = {
  page: {
    position: 'fixed',
    inset: 'var(--nav-h) 0 0 0',
    overflow: 'hidden',
  },

  controls: {
    position: 'absolute',
    top: 14,
    left: 14,
    zIndex: 800,
    width: 268,
    maxWidth: 'calc(100vw - 28px)',
  },
  controlHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    borderBottom: '1px solid var(--border)',
  },
  iconBtn: {
    width: 26,
    height: 26,
    padding: 0,
    justifyContent: 'center',
    border: '1px solid var(--border)',
  },
  sizeRow: {
    display: 'flex',
    gap: 5,
  },
  countRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
    marginTop: 12,
    fontSize: 11.5,
    minHeight: 22,
  },
  scoringChip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '2px 8px',
    borderRadius: 999,
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '.05em',
    color: 'var(--accent)',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
  },
  scoringDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'var(--accent)',
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  exportRow: {
    display: 'flex',
    gap: 6,
    marginTop: 12,
  },
  notice: {
    padding: '10px 14px',
    borderTop: '1px solid var(--border)',
    fontSize: 11.5,
    lineHeight: 1.55,
    color: 'var(--orange)',
    background: 'rgba(255,159,67,0.08)',
    borderBottomLeftRadius: 'var(--radius-lg)',
    borderBottomRightRadius: 'var(--radius-lg)',
  },

  summary: {
    position: 'absolute',
    bottom: 16,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 800,
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    padding: '10px 20px',
    borderRadius: 40,
  },
  summaryLabel: {
    fontSize: 9.5,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '.07em',
    whiteSpace: 'nowrap',
  },
  divider: {
    width: 1,
    height: 24,
    background: 'var(--border)',
  },

  legend: {
    position: 'absolute',
    bottom: 16,
    right: 14,
    zIndex: 800,
    width: 262,
  },
  legendIntro: {
    padding: '11px 14px',
    borderBottom: '1px solid var(--border)',
    fontSize: 11.5,
    lineHeight: 1.55,
    color: 'var(--text-dim)',
  },
  legendRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    padding: '5px 0',
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
    padding: '8px 14px',
    borderRadius: 'var(--radius)',
    border: '1px solid rgba(255,93,93,0.4)',
    background: 'rgba(255,93,93,0.12)',
    color: 'var(--red)',
    fontSize: 12,
    fontWeight: 600,
  },
}
