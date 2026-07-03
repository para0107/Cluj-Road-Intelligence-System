/**
 * frontend/src/pages/LivePage.jsx — Waze-like live hazard map.
 *
 * Every camera / user is a sensor. Reports stream in over a WebSocket
 * (polling fallback), cluster server-side into events, and escalate as
 * independent devices re-sight them: UNVERIFIED → CONFIRMED → VERIFIED.
 * Anyone can vote "Still there" / "Not there"; enough disputes remove an
 * event, and stale events expire on their own.
 */

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { MapContainer, TileLayer, CircleMarker, useMapEvents, useMap } from 'react-leaflet'
import {
  Radio, WifiOff, Plus, X, ThumbsUp, ThumbsDown, CheckCircle2, ShieldCheck,
  Shield, ShieldAlert, Car, Activity, Users, MapPin, Crosshair,
} from 'lucide-react'
import {
  CLASS_COLORS, CLASS_LABELS, CLASS_ICONS, ALL_CLASSES,
  SEVERITY_COLORS, CLUJ_CENTER, CLUJ_ZOOM, BASEMAPS,
} from '../utils/constants'
import {
  getDeviceId, fetchLiveEvents, fetchLiveStats,
  postLiveReport, confirmLiveEvent, disputeLiveEvent, resolveLiveEvent,
  openLiveSocket,
} from '../utils/live'
import { ClassDot } from '../components/ui'

const POLL_FALLBACK_MS = 5_000

// ── Status meta ─────────────────────────────────────────────────────────────
const STATUS_META = {
  unverified: { label: 'UNVERIFIED', color: 'var(--text-muted)', icon: Shield },
  confirmed:  { label: 'CONFIRMED',  color: 'var(--cyan)',       icon: ShieldAlert },
  verified:   { label: 'VERIFIED',   color: 'var(--green)',      icon: ShieldCheck },
}

function fmtAgo(iso, now) {
  if (!iso) return '—'
  const s = Math.max(0, (now - new Date(iso).getTime()) / 1000)
  if (s < 60) return `${Math.floor(s)}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

function StatusChip({ status }) {
  const meta = STATUS_META[status] || STATUS_META.unverified
  const Icon = meta.icon
  return (
    <span className="mono" style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      fontSize: 9.5, fontWeight: 700, letterSpacing: '0.08em',
      color: meta.color, border: `1px solid ${meta.color}55`,
      background: `color-mix(in srgb, ${meta.color} 12%, transparent)`,
      borderRadius: 5, padding: '2px 7px',
    }}>
      <Icon size={10} />
      {meta.label}
    </span>
  )
}

// ── Map interaction helpers ─────────────────────────────────────────────────
function ReportClickCatcher({ reporting, onPick }) {
  useMapEvents({
    click(e) {
      if (reporting) onPick([e.latlng.lat, e.latlng.lng])
    },
  })
  return null
}

function FlyTo({ target }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo([target.lat, target.lon], 17, { duration: 0.8 })
  }, [target, map])
  return null
}

// ── Main page ───────────────────────────────────────────────────────────────
export default function LivePage() {
  const [events, setEvents] = useState({})          // id → event
  const [stats, setStats] = useState(null)
  const [wsState, setWsState] = useState('connecting')  // connecting|open|closed
  const [toasts, setToasts] = useState([])
  const [now, setNow] = useState(Date.now())

  // Reporting flow
  const [reporting, setReporting] = useState(false)      // map-click armed
  const [pickerAt, setPickerAt] = useState(null)         // [lat, lon] awaiting type choice
  const [busyId, setBusyId] = useState(null)

  const [selectedId, setSelectedId] = useState(null)
  const [flyTarget, setFlyTarget] = useState(null)
  const knownIds = useRef(new Set())
  const pollRef = useRef(null)

  const deviceId = useMemo(getDeviceId, [])

  // Clock tick for "x ago" labels
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000)
    return () => clearInterval(t)
  }, [])

  const pushToast = useCallback((text, color = 'var(--accent)') => {
    const id = Math.random().toString(36).slice(2)
    setToasts(prev => [...prev.slice(-3), { id, text, color }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4500)
  }, [])

  const ingestSnapshot = useCallback((list) => {
    const map = {}
    list.forEach(e => { map[e.id] = e; knownIds.current.add(e.id) })
    setEvents(map)
  }, [])

  const upsertEvent = useCallback((e) => {
    const isNew = !knownIds.current.has(e.id)
    knownIds.current.add(e.id)
    setEvents(prev => ({ ...prev, [e.id]: e }))
    if (isNew) {
      pushToast(`New ${CLASS_LABELS[e.damage_type] || e.damage_type} reported`, CLASS_COLORS[e.damage_type])
    } else if (e.status === 'verified') {
      pushToast(`${CLASS_LABELS[e.damage_type] || e.damage_type} VERIFIED by ${e.reporter_devices} devices`, 'var(--green)')
    }
  }, [pushToast])

  const removeEvent = useCallback((id) => {
    setEvents(prev => {
      if (!prev[id]) return prev
      const next = { ...prev }
      delete next[id]
      return next
    })
    setSelectedId(sel => (sel === id ? null : sel))
  }, [])

  // ── WebSocket with polling fallback ──────────────────────────────────────
  useEffect(() => {
    const close = openLiveSocket({
      onHello: (list) => ingestSnapshot(list),
      onUpsert: upsertEvent,
      onRemoved: removeEvent,
      onStatus: (s) => setWsState(s),
    })
    return close
  }, [ingestSnapshot, upsertEvent, removeEvent])

  useEffect(() => {
    // While the socket is down, poll REST so the map stays fresh.
    if (wsState === 'open') {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
      return
    }
    const tick = async () => {
      try { ingestSnapshot((await fetchLiveEvents()).items || []) } catch { /* backend down */ }
    }
    tick()
    pollRef.current = setInterval(tick, POLL_FALLBACK_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [wsState, ingestSnapshot])

  // Header stats, refreshed lazily
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try { const s = await fetchLiveStats(); if (alive) setStats(s) } catch { /* noop */ }
    }
    tick()
    const t = setInterval(tick, 15_000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  // ── Actions ───────────────────────────────────────────────────────────────
  const submitReport = async (type) => {
    if (!pickerAt) return
    const [lat, lon] = pickerAt
    setPickerAt(null)
    setReporting(false)
    try {
      const res = await postLiveReport({ latitude: lat, longitude: lon, damage_type: type })
      if (res.event) upsertEvent(res.event)
      pushToast(res.action === 'merged' ? 'Merged into an existing hazard' : 'Hazard reported — thank you!')
    } catch (e) {
      pushToast(`Report failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
    }
  }

  const vote = async (ev, kind) => {
    setBusyId(ev.id)
    try {
      const fn = kind === 'confirm' ? confirmLiveEvent : disputeLiveEvent
      const res = await fn(ev.id)
      if (res.action === 'removed') {
        removeEvent(ev.id)
        pushToast('Hazard removed after community disputes', 'var(--orange)')
      } else if (res.event) {
        upsertEvent(res.event)
      }
    } catch (e) {
      pushToast(`Vote failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
    } finally {
      setBusyId(null)
    }
  }

  const resolve = async (ev) => {
    if (!window.confirm('Mark this hazard as repaired / cleared?')) return
    setBusyId(ev.id)
    try {
      await resolveLiveEvent(ev.id)
      removeEvent(ev.id)
      pushToast('Hazard resolved — nice work', 'var(--green)')
    } catch (e) {
      pushToast(`Resolve failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
    } finally {
      setBusyId(null)
    }
  }

  const eventList = useMemo(
    () => Object.values(events).sort((a, b) => new Date(b.last_reported) - new Date(a.last_reported)),
    [events],
  )
  const selected = selectedId ? events[selectedId] : null

  return (
    <div style={styles.page}>
      {/* ── Map ──────────────────────────────────────────────────────── */}
      <MapContainer
        center={CLUJ_CENTER}
        zoom={CLUJ_ZOOM}
        maxZoom={20}
        minZoom={3}
        style={{ width: '100%', height: '100%', cursor: reporting ? 'crosshair' : 'grab' }}
        zoomControl={false}
      >
        <TileLayer url={BASEMAPS.dark.url} attribution={BASEMAPS.dark.attr} maxZoom={20} maxNativeZoom={19} />
        <ReportClickCatcher reporting={reporting} onPick={setPickerAt} />
        <FlyTo target={flyTarget} />

        {eventList.map(ev => {
          const color = CLASS_COLORS[ev.damage_type] || '#888'
          const sevColor = SEVERITY_COLORS[ev.severity] || color
          const isSel = selectedId === ev.id
          const r = ev.status === 'verified' ? 11 : ev.status === 'confirmed' ? 9 : 7
          return (
            <CircleMarker
              key={ev.id}
              center={[ev.latitude, ev.longitude]}
              radius={isSel ? r + 3 : r}
              pathOptions={{
                color: isSel ? '#eaff3d' : (STATUS_META[ev.status] ? sevColor : color),
                fillColor: color,
                fillOpacity: ev.status === 'unverified' ? 0.5 : 0.9,
                weight: isSel ? 3 : ev.status === 'verified' ? 2.5 : 1.5,
                dashArray: ev.status === 'unverified' ? '3 3' : null,
                className: ev.status === 'verified' && ev.severity >= 4 ? 'marker-critical' : '',
              }}
              eventHandlers={{ click: () => setSelectedId(ev.id) }}
            />
          )
        })}

        {/* Pending report position */}
        {pickerAt && (
          <CircleMarker
            center={pickerAt}
            radius={10}
            pathOptions={{ color: '#eaff3d', fillColor: '#eaff3d', fillOpacity: 0.25, weight: 2, dashArray: '4 4' }}
          />
        )}
      </MapContainer>

      {/* ── Top-left: connection + stats ─────────────────────────────── */}
      <div style={styles.topLeft}>
        <div style={{
          ...styles.connBadge,
          borderColor: wsState === 'open' ? 'rgba(61,220,132,0.45)' : 'rgba(255,159,67,0.45)',
          color: wsState === 'open' ? 'var(--green)' : 'var(--orange)',
          background: wsState === 'open' ? 'rgba(61,220,132,0.08)' : 'rgba(255,159,67,0.08)',
        }}>
          {wsState === 'open'
            ? <><Radio size={11} style={{ animation: 'pulse 1.5s ease-in-out infinite' }} /> LIVE · WEBSOCKET</>
            : <><WifiOff size={11} /> POLLING · {POLL_FALLBACK_MS / 1000}s</>}
        </div>

        <div className="glass" style={styles.statsCard}>
          <MiniStat icon={Activity} label="Active" value={eventList.length} color="var(--accent)" />
          <MiniStat icon={ShieldCheck} label="Verified" value={stats?.verified_events ?? '—'} color="var(--green)" />
          <MiniStat icon={Car} label="Reports/h" value={stats?.reports_last_hour ?? '—'} color="var(--cyan)" />
          <MiniStat icon={Users} label="Devices/h" value={stats?.devices_last_hour ?? '—'} color="var(--purple)" />
        </div>

        <div className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', paddingLeft: 4 }}>
          you are <span style={{ color: 'var(--accent)' }}>{deviceId}</span>
        </div>
      </div>

      {/* ── Report button ────────────────────────────────────────────── */}
      <div style={styles.reportWrap}>
        {reporting && !pickerAt && (
          <div className="glass anim-fade-in" style={styles.reportHint}>
            <Crosshair size={12} style={{ color: 'var(--accent)' }} />
            Tap the map where the damage is
          </div>
        )}
        <button
          className={`btn ${reporting ? 'btn-danger' : 'btn-accent'}`}
          style={styles.reportBtn}
          onClick={() => { setReporting(v => !v); setPickerAt(null) }}
        >
          {reporting ? <><X size={16} /> Cancel</> : <><Plus size={16} /> Report damage</>}
        </button>
      </div>

      {/* ── Type picker (after map click) ────────────────────────────── */}
      {pickerAt && (
        <div className="glass anim-fade-up" style={styles.picker}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span className="overline">What do you see?</span>
            <button className="btn btn-sm btn-ghost" style={{ width: 24, height: 24, padding: 0 }} onClick={() => setPickerAt(null)}>
              <X size={12} />
            </button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {ALL_CLASSES.map(c => (
              <button
                key={c}
                className="btn btn-sm"
                style={{ justifyContent: 'flex-start', gap: 8, borderColor: `${CLASS_COLORS[c]}55` }}
                onClick={() => submitReport(c)}
              >
                <span style={{ color: CLASS_COLORS[c], fontSize: 13 }}>{CLASS_ICONS[c]}</span>
                <span style={{ fontSize: 11 }}>{CLASS_LABELS[c]}</span>
              </button>
            ))}
          </div>
          <div className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', marginTop: 8 }}>
            <MapPin size={9} style={{ display: 'inline', marginRight: 3 }} />
            {pickerAt[0].toFixed(5)}, {pickerAt[1].toFixed(5)}
          </div>
        </div>
      )}

      {/* ── Right: live feed ─────────────────────────────────────────── */}
      <div style={styles.feed} className="glass">
        <div style={styles.feedHeader}>
          <span className="overline">Live feed</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {eventList.length} active
          </span>
        </div>
        <div style={styles.feedBody}>
          {eventList.length === 0 && (
            <div style={{ padding: '28px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.7 }}>
              No live hazards right now.<br />
              Drive with <span className="mono">live_camera.py</span>, run the
              fleet simulator, or tap <strong>Report damage</strong>.
            </div>
          )}
          {eventList.map(ev => {
            const isSel = selectedId === ev.id
            return (
              <div
                key={ev.id}
                onClick={() => { setSelectedId(ev.id); setFlyTarget({ lat: ev.latitude, lon: ev.longitude }) }}
                style={{
                  ...styles.feedCard,
                  borderColor: isSel ? 'var(--border-accent)' : 'var(--border)',
                  background: isSel ? 'var(--accent-dim)' : 'var(--bg-card2)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <ClassDot cls={ev.damage_type} size={30} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 12.5, fontWeight: 700 }}>
                        {CLASS_LABELS[ev.damage_type] || ev.damage_type}
                      </span>
                      {ev.severity && (
                        <span className="mono" style={{ fontSize: 9.5, fontWeight: 700, color: SEVERITY_COLORS[ev.severity] }}>
                          S{ev.severity}
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 3 }}>
                      <StatusChip status={ev.status} />
                      <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                        <Users size={9} style={{ display: 'inline', marginRight: 2 }} />
                        {ev.reporter_devices}
                      </span>
                      <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                        {fmtAgo(ev.last_reported, now)}
                      </span>
                    </div>
                  </div>
                </div>

                {isSel && (
                  <div style={styles.voteRow} className="anim-fade-in">
                    <button
                      className="btn btn-sm"
                      style={{ flex: 1, color: 'var(--green)', borderColor: 'rgba(61,220,132,0.4)' }}
                      disabled={busyId === ev.id}
                      onClick={(e) => { e.stopPropagation(); vote(ev, 'confirm') }}
                    >
                      <ThumbsUp size={12} /> Still there
                    </button>
                    <button
                      className="btn btn-sm"
                      style={{ flex: 1, color: 'var(--orange)', borderColor: 'rgba(255,159,67,0.4)' }}
                      disabled={busyId === ev.id}
                      onClick={(e) => { e.stopPropagation(); vote(ev, 'dispute') }}
                    >
                      <ThumbsDown size={12} /> Not there
                    </button>
                    <button
                      className="btn btn-sm"
                      title="Mark repaired (operator)"
                      style={{ color: 'var(--cyan)', borderColor: 'rgba(76,201,240,0.4)' }}
                      disabled={busyId === ev.id}
                      onClick={(e) => { e.stopPropagation(); resolve(ev) }}
                    >
                      <CheckCircle2 size={12} />
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Toasts ───────────────────────────────────────────────────── */}
      <div style={styles.toasts}>
        {toasts.map(t => (
          <div key={t.id} className="glass anim-slide-right" style={{ ...styles.toast, borderColor: `${t.color}66` }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
            {t.text}
          </div>
        ))}
      </div>
    </div>
  )
}

function MiniStat({ icon: Icon, label, value, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
      <Icon size={13} style={{ color }} />
      <div>
        <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', lineHeight: 1.1 }}>{value}</div>
        <div style={{ fontSize: 8.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.07em' }}>{label}</div>
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
    gap: 8,
    alignItems: 'flex-start',
  },
  connBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    borderRadius: 20,
    border: '1px solid',
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '.06em',
    backdropFilter: 'blur(8px)',
  },
  statsCard: {
    display: 'flex',
    gap: 16,
    padding: '10px 16px',
  },

  reportWrap: {
    position: 'absolute',
    bottom: 22,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 850,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 10,
  },
  reportHint: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    padding: '7px 14px',
    borderRadius: 20,
    fontSize: 11.5,
    color: 'var(--text-dim)',
  },
  reportBtn: {
    padding: '13px 26px',
    fontSize: 14,
    borderRadius: 999,
    boxShadow: 'var(--shadow-lg)',
  },

  picker: {
    position: 'absolute',
    bottom: 90,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 900,
    width: 330,
    padding: 14,
  },

  feed: {
    position: 'absolute',
    top: 14, right: 14, bottom: 16,
    width: 312,
    zIndex: 800,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  feedHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 15px',
    borderBottom: '1px solid var(--border)',
  },
  feedBody: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 7,
    padding: 10,
  },
  feedCard: {
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '10px 12px',
    cursor: 'pointer',
    transition: 'var(--transition)',
  },
  voteRow: {
    display: 'flex',
    gap: 6,
    marginTop: 10,
  },

  toasts: {
    position: 'absolute',
    bottom: 22, right: 340,
    zIndex: 900,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    alignItems: 'flex-end',
  },
  toast: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    padding: '9px 14px',
    fontSize: 12,
    color: 'var(--text)',
    maxWidth: 320,
  },
}
