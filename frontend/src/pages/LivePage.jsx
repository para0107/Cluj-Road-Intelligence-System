/**
 * frontend/src/pages/LivePage.jsx — Waze-like live hazard map.
 *
 * Every camera / user is a sensor. Reports stream in over a WebSocket
 * (polling fallback), cluster server-side into events, and escalate as
 * independent devices re-sight them: UNVERIFIED → CONFIRMED → VERIFIED.
 * Anyone can vote "Still there" / "Not there"; enough disputes remove an
 * event, and stale events expire on their own.
 *
 * Driver-first UX:
 *  - the map opens on the user's own city (useCityCenter, never hardcoded)
 *  - a Waze-style arrow puck shows your live GPS position + heading, with
 *    follow-mode that re-centres the map as you move (drag to break away)
 *  - "Report" defaults to YOUR position — one tap, pick the type, done;
 *    tapping the map to place a report precisely still works
 *  - drive mode auto-reports impacts, keeps the screen awake, and shows a
 *    HUD with speed / jolt / nearest hazard
 *  - on phones the feed folds into a bottom sheet and controls are
 *    thumb-sized
 */

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { MapContainer, TileLayer, CircleMarker, Circle, Marker, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import {
  Radio, WifiOff, Plus, X, ThumbsUp, ThumbsDown, CheckCircle2, ShieldCheck,
  Shield, ShieldAlert, Car, Activity, Users, MapPin, Crosshair,
  Smartphone, Copy, Trash2, Gauge, Link2, ChevronUp, ChevronDown, AlertTriangle,
} from 'lucide-react'
import {
  CLASS_COLORS, CLASS_LABELS, CLASS_ICONS, ALL_CLASSES,
  SEVERITY_COLORS, CITY_ZOOM, BASEMAPS,
} from '../utils/constants'
import {
  getDeviceId, fetchLiveEvents, fetchLiveStats,
  postLiveReport, confirmLiveEvent, disputeLiveEvent, resolveLiveEvent,
  openLiveSocket,
  pairThisDevice, createPairCode, fetchMyDevices, disconnectDevice,
} from '../utils/live'
import { startDriveMode } from '../utils/driveMode'
import { ClassDot } from '../components/ui'
import { useIsDark } from '../hooks/useTheme'
import useCityCenter from '../hooks/useCityCenter'
import useIsMobile from '../hooks/useIsMobile'
import { useAuth } from '../context/AuthContext'

const POLL_FALLBACK_MS = 5_000
const FIX_FRESH_MS = 15_000          // a GPS fix younger than this anchors "Report here"
const FOLLOW_ZOOM = 16

// Pothole is by far the most reported hazard — it leads the picker.
const PICKER_CLASSES = ['pothole', ...ALL_CLASSES.filter(c => c !== 'pothole')]

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

function fmtDist(m) {
  if (m == null) return '—'
  return m < 950 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`
}

/** Great-circle distance in metres. */
function distM(lat1, lon1, lat2, lon2) {
  const R = 6371000
  const toR = Math.PI / 180
  const dLat = (lat2 - lat1) * toR
  const dLon = (lon2 - lon1) * toR
  const s = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * toR) * Math.cos(lat2 * toR) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(s))
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
    if (target) map.flyTo([target.lat, target.lon], target.zoom ?? 17, { duration: 0.8 })
  }, [target, map])
  return null
}

/** Re-centre on every fix while follow is on; a manual drag breaks follow. */
function FollowUser({ fix, follow, onManualPan }) {
  const map = useMap()
  useMapEvents({ dragstart: () => onManualPan() })
  useEffect(() => {
    if (follow && fix) {
      map.setView([fix.lat, fix.lon], Math.max(map.getZoom(), FOLLOW_ZOOM), { animate: true })
    }
  }, [fix, follow, map])
  return null
}

// Waze-style position puck: an arrow that rotates with the GPS heading.
const ARROW_SVG = `
<svg viewBox="0 0 40 40" width="40" height="40" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="15" fill="#1d9bf0" stroke="#ffffff" stroke-width="3"/>
  <path d="M20 9 L27 25 L20 21.5 L13 25 Z" fill="#ffffff"/>
</svg>`

function UserMarker({ fix, heading }) {
  const icon = useMemo(() => {
    // Round the rotation so the icon isn't rebuilt on every tiny change.
    const rot = Math.round((heading ?? 0) / 5) * 5
    return L.divIcon({
      className: 'user-puck-wrap',
      html: `<div class="user-puck" style="transform: rotate(${rot}deg)">${ARROW_SVG}</div>`,
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    })
  }, [heading])

  if (!fix) return null
  return (
    <>
      {fix.accuracy != null && fix.accuracy < 250 && (
        <Circle
          center={[fix.lat, fix.lon]}
          radius={fix.accuracy}
          pathOptions={{ color: '#1d9bf0', weight: 1, opacity: 0.35, fillColor: '#1d9bf0', fillOpacity: 0.08 }}
        />
      )}
      <Marker position={[fix.lat, fix.lon]} icon={icon} zIndexOffset={1000} interactive={false} />
    </>
  )
}

// ── Main page ───────────────────────────────────────────────────────────────
export default function LivePage() {
  const { isOperator } = useAuth()
  const isMobile = useIsMobile()
  const { center, zoom, cityCenter } = useCityCenter()

  const [events, setEvents] = useState({})          // id → event
  const [stats, setStats] = useState(null)
  const [wsState, setWsState] = useState('connecting')  // connecting|open|closed
  const [toasts, setToasts] = useState([])
  const [now, setNow] = useState(Date.now())

  // Reporting flow
  const [reporting, setReporting] = useState(false)      // map-click armed
  const [pickerAt, setPickerAt] = useState(null)         // [lat, lon] awaiting type choice
  const [pickerMode, setPickerMode] = useState(null)     // 'gps' | 'map'
  const [busyId, setBusyId] = useState(null)

  const [selectedId, setSelectedId] = useState(null)
  const [flyTarget, setFlyTarget] = useState(null)
  const [feedOpen, setFeedOpen] = useState(false)        // mobile bottom sheet
  const knownIds = useRef(new Set())
  const pollRef = useRef(null)
  const cityFlown = useRef(false)                        // glide to the city at most once

  // GPS position (the "you" puck) + follow mode
  const [fix, setFix] = useState(null)                   // {lat, lon, heading, speed, accuracy, ts}
  const [heading, setHeading] = useState(0)
  const [follow, setFollow] = useState(false)
  const [geoDenied, setGeoDenied] = useState(false)

  // Paired devices + phone drive mode
  const [devicesOpen, setDevicesOpen] = useState(false)
  const [devices, setDevices] = useState(null)
  const [pairInfo, setPairInfo] = useState(null)          // pending device w/ code
  const [driveOn, setDriveOn] = useState(false)
  const [driveStats, setDriveStats] = useState({ jolt: 0, speed: null, hasFix: false, sent: 0 })
  const driveStopRef = useRef(null)
  const wakeLockRef = useRef(null)

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

  // ── GPS watch: powers the "you" puck, follow mode, and report-here ───────
  useEffect(() => {
    if (!('geolocation' in navigator)) return undefined
    const id = navigator.geolocation.watchPosition(
      (pos) => {
        setGeoDenied(false)
        setFix({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          heading: pos.coords.heading,
          speed: pos.coords.speed,
          accuracy: pos.coords.accuracy,
          ts: Date.now(),
        })
        // Heading is null when stationary — keep the last one so the arrow
        // doesn't snap north at every red light.
        if (pos.coords.heading != null && !Number.isNaN(pos.coords.heading)
            && (pos.coords.speed == null || pos.coords.speed > 0.5)) {
          setHeading(pos.coords.heading)
        }
      },
      (err) => { if (err.code === err.PERMISSION_DENIED) setGeoDenied(true) },
      { enableHighAccuracy: true, maximumAge: 3000, timeout: 20000 },
    )
    return () => navigator.geolocation.clearWatch(id)
  }, [])

  const fixFresh = fix && Date.now() - fix.ts < FIX_FRESH_MS

  // Once the city centre is known, the one-time glide (rendered below) has
  // had its chance — never fight the user for the camera afterwards.
  useEffect(() => {
    if (cityCenter) cityFlown.current = true
  }, [cityCenter])

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
      return undefined
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
  const closePicker = () => { setPickerAt(null); setPickerMode(null); setReporting(false) }

  const openReport = () => {
    if (reporting || pickerAt) { closePicker(); return }
    if (fixFresh) {
      // Waze model: you report what you just drove past — your position IS
      // the report position. Map-tap stays available from inside the picker.
      setPickerAt([fix.lat, fix.lon])
      setPickerMode('gps')
    } else {
      setReporting(true)
      setPickerMode('map')
    }
  }

  const pickOnMapInstead = () => {
    setPickerAt(null)
    setPickerMode('map')
    setReporting(true)
  }

  const submitReport = async (type) => {
    if (!pickerAt) return
    const [lat, lon] = pickerAt
    closePicker()
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

  const locateMe = () => {
    if (geoDenied) {
      pushToast('Location permission is blocked — allow it in the browser settings.', 'var(--red)')
      return
    }
    if (fix) {
      setFollow(true)
      setFlyTarget({ lat: fix.lat, lon: fix.lon, zoom: FOLLOW_ZOOM })
    } else {
      pushToast('Waiting for a GPS fix…', 'var(--cyan)')
    }
  }

  // ── Paired devices & phone drive mode ────────────────────────────────────
  const loadDevices = useCallback(async () => {
    try { setDevices((await fetchMyDevices()).items || []) } catch { /* backend down */ }
  }, [])

  const toggleDevicesPanel = () => {
    setDevicesOpen(v => !v)
    if (!devicesOpen) loadDevices()
  }

  const makePairCode = async () => {
    try {
      const dev = await createPairCode('Dashcam / PC', 'dashcam')
      setPairInfo(dev)
      loadDevices()
    } catch (e) {
      pushToast(`Pairing failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
    }
  }

  const removeDevice = async (d) => {
    if (d.device_id && !window.confirm(`Disconnect "${d.name}"? Its future uploads will be rejected.`)) return
    try {
      const res = await disconnectDevice(d.id)
      setDevices(res.items || [])
      if (pairInfo?.id === d.id) setPairInfo(null)
    } catch (e) {
      pushToast(`Disconnect failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
    }
  }

  // Screen wake lock while driving (free browser API; silently unsupported
  // browsers just dim as usual).
  const acquireWakeLock = useCallback(async () => {
    try { wakeLockRef.current = await navigator.wakeLock?.request('screen') } catch { /* unsupported */ }
  }, [])
  const releaseWakeLock = useCallback(() => {
    try { wakeLockRef.current?.release() } catch { /* noop */ }
    wakeLockRef.current = null
  }, [])
  useEffect(() => {
    if (!driveOn) return undefined
    const onVisible = () => { if (document.visibilityState === 'visible') acquireWakeLock() }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [driveOn, acquireWakeLock])

  const stopDrive = useCallback(() => {
    driveStopRef.current?.()
    driveStopRef.current = null
    releaseWakeLock()
    setDriveOn(false)
  }, [releaseWakeLock])

  const toggleDrive = async () => {
    if (driveOn) { stopDrive(); return }
    try { await pairThisDevice('This phone / browser', 'phone') } catch { /* keep going — reporting still works */ }
    let failed = false
    const stop = await startDriveMode({
      onHit: async (hit) => {
        try {
          const res = await postLiveReport({
            latitude: hit.latitude,
            longitude: hit.longitude,
            damage_type: 'pothole',
            confidence: hit.confidence,
            severity: hit.severity,
            note: `phone-motion:jolt=${hit.jolt}m/s2`,
          })
          if (res.event) upsertEvent(res.event)
          setDriveStats(s => ({ ...s, sent: s.sent + 1 }))
          pushToast(`Impact detected (${hit.jolt} m/s²) — pothole auto-reported`)
        } catch (e) {
          pushToast(`Auto-report failed: ${e?.response?.data?.detail || e.message}`, 'var(--red)')
        }
      },
      onTick: (t) => setDriveStats(s => ({ ...s, ...t })),
      onError: (msg) => { failed = true; pushToast(msg, 'var(--red)'); stopDrive() },
    })
    if (failed) { stop?.(); return }
    driveStopRef.current = stop
    setDriveOn(true)
    setDevicesOpen(false)
    setFollow(true)               // the map drives with you, Waze-style
    acquireWakeLock()
    pushToast('Drive mode ON — impacts auto-report as potholes')
  }

  // Stop sensors when leaving the page
  useEffect(() => () => { driveStopRef.current?.(); releaseWakeLock() }, [releaseWakeLock])

  const eventList = useMemo(
    () => Object.values(events).sort((a, b) => new Date(b.last_reported) - new Date(a.last_reported)),
    [events],
  )
  const selected = selectedId ? events[selectedId] : null

  // Nearest active hazard to the current fix — the HUD's "watch out" number.
  const nearest = useMemo(() => {
    if (!fix || eventList.length === 0) return null
    let best = null
    for (const ev of eventList) {
      const d = distM(fix.lat, fix.lon, ev.latitude, ev.longitude)
      if (!best || d < best.d) best = { d, ev }
    }
    return best
  }, [fix, eventList])

  // Tiles follow the app theme for consistency with MapPage:
  // dark mode → Dark tiles, light mode → Streets tiles.
  const isDark = useIsDark()
  const tiles = isDark ? BASEMAPS.dark : BASEMAPS.voyager

  const speedKmh = driveStats.speed != null ? Math.round(driveStats.speed * 3.6)
    : fix?.speed != null ? Math.round(fix.speed * 3.6) : null

  return (
    <div style={styles.page}>
      {/* ── Map ──────────────────────────────────────────────────────── */}
      <MapContainer
        center={center}
        zoom={zoom}
        maxZoom={20}
        minZoom={3}
        style={{ width: '100%', height: '100%', cursor: reporting ? 'crosshair' : 'grab' }}
        zoomControl={false}
      >
        <TileLayer key={tiles.url} url={tiles.url} attribution={tiles.attr} maxZoom={20} maxNativeZoom={19} />
        <ReportClickCatcher reporting={reporting} onPick={(ll) => { setPickerAt(ll); setPickerMode('map') }} />
        <FlyTo target={flyTarget} />
        {/* City centre resolved after mount (first login on this browser):
            glide over ONCE, and only if nothing else owns the camera. */}
        {cityCenter && !cityFlown.current && !follow && !flyTarget && (
          <FlyTo target={{ lat: cityCenter[0], lon: cityCenter[1], zoom: CITY_ZOOM }} />
        )}
        <FollowUser fix={fix} follow={follow} onManualPan={() => setFollow(false)} />

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

        {/* You — Waze-style arrow puck */}
        <UserMarker fix={fix} heading={heading} />

        {/* Pending report position */}
        {pickerAt && (
          <CircleMarker
            center={pickerAt}
            radius={10}
            pathOptions={{ color: '#eaff3d', fillColor: '#eaff3d', fillOpacity: 0.25, weight: 2, dashArray: '4 4' }}
          />
        )}
      </MapContainer>

      {/* ── Top-left: connection + stats + devices ───────────────────── */}
      <div className="live-topbar">
        <div className="live-topbar-row">
          <div style={{
            ...styles.connBadge,
            borderColor: wsState === 'open' ? 'rgba(61,220,132,0.45)' : 'rgba(255,159,67,0.45)',
            color: wsState === 'open' ? 'var(--green)' : 'var(--orange)',
            background: wsState === 'open' ? 'rgba(61,220,132,0.08)' : 'rgba(255,159,67,0.08)',
          }}>
            {wsState === 'open'
              ? <><Radio size={11} style={{ animation: 'pulse 1.5s ease-in-out infinite' }} /> {isMobile ? 'LIVE' : 'LIVE · WEBSOCKET'}</>
              : <><WifiOff size={11} /> {isMobile ? 'POLL' : `POLLING · ${POLL_FALLBACK_MS / 1000}s`}</>}
          </div>

          <button className="btn btn-sm glass" onClick={toggleDevicesPanel}
                  style={{ gap: 6, borderColor: devicesOpen ? 'var(--border-accent)' : undefined }}>
            <Smartphone size={12} /> Devices
            {driveOn && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)' }} />}
          </button>
        </div>

        <div className="glass live-stats">
          <MiniStat icon={Activity} label="Active" value={eventList.length} color="var(--accent)" />
          <MiniStat icon={ShieldCheck} label="Verified" value={stats?.verified_events ?? '—'} color="var(--green)" />
          <MiniStat icon={Car} label="Reports/h" value={stats?.reports_last_hour ?? '—'} color="var(--cyan)" />
          <MiniStat icon={Users} label="Devices/h" value={stats?.devices_last_hour ?? '—'} color="var(--purple)" />
        </div>

        {!isMobile && (
          <div className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', paddingLeft: 4 }}>
            you are <span style={{ color: 'var(--accent)' }}>{deviceId}</span>
          </div>
        )}
      </div>

      {/* ── Drive HUD (top-centre while drive mode is on) ────────────── */}
      {driveOn && (
        <div className="glass live-hud anim-fade-in">
          <span className="mono live-hud-cell">
            <Gauge size={13} style={{ color: 'var(--cyan)' }} />
            <b>{speedKmh ?? '—'}</b>&nbsp;km/h
          </span>
          <span className="mono live-hud-cell">
            jolt <b style={{ color: driveStats.jolt > 9 ? 'var(--red)' : 'var(--text)' }}>{driveStats.jolt}</b>
          </span>
          <span className="mono live-hud-cell">
            sent <b style={{ color: 'var(--accent)' }}>{driveStats.sent}</b>
          </span>
          <span className="mono live-hud-cell" style={{
            color: nearest && nearest.d < 150 ? 'var(--red)' : 'var(--text-dim)',
          }}>
            <AlertTriangle size={12} />
            {nearest ? fmtDist(nearest.d) : '—'}
          </span>
        </div>
      )}

      {/* ── Devices panel ─────────────────────────────────────────────── */}
      {devicesOpen && (
        <div className="glass anim-fade-up live-devices">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span className="overline">My sensors</span>
            <button className="btn btn-sm btn-ghost" style={{ width: 28, height: 28, padding: 0 }}
                    onClick={() => setDevicesOpen(false)}>
              <X size={13} />
            </button>
          </div>

          {/* Phone drive mode */}
          <button className={`btn ${driveOn ? 'btn-danger' : 'btn-accent'}`}
                  style={{ width: '100%', gap: 8, padding: '11px 0' }} onClick={toggleDrive}>
            <Gauge size={14} />
            {driveOn ? 'Stop drive mode' : 'Start drive mode (this phone)'}
          </button>
          {driveOn && (
            <div className="mono" style={styles.driveReadout}>
              <span>jolt <b style={{ color: driveStats.jolt > 9 ? 'var(--red)' : 'var(--text)' }}>{driveStats.jolt}</b> m/s²</span>
              <span>gps <b style={{ color: driveStats.hasFix ? 'var(--green)' : 'var(--orange)' }}>{driveStats.hasFix ? 'fix' : '—'}</b></span>
              <span>sent <b style={{ color: 'var(--accent)' }}>{driveStats.sent}</b></span>
            </div>
          )}
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', margin: '8px 2px 12px', lineHeight: 1.5 }}>
            Mount the phone in the car and drive — impacts are detected with the
            motion sensor + GPS and uploaded automatically. The screen stays awake.
          </div>

          {/* Dashcam / PC pairing */}
          <button className="btn btn-sm" style={{ width: '100%', gap: 8 }} onClick={makePairCode}>
            <Link2 size={12} /> Pair a dashcam / PC
          </button>
          {pairInfo?.pair_code && (
            <div style={styles.pairBox} className="anim-fade-in">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="mono" style={{ fontSize: 18, fontWeight: 800, letterSpacing: '0.2em', color: 'var(--accent)' }}>
                  {pairInfo.pair_code}
                </span>
                <button className="btn btn-sm btn-ghost" title="Copy pairing command"
                        onClick={() => {
                          navigator.clipboard?.writeText(`python pipeline/live_pipeline.py --pair ${pairInfo.pair_code}`)
                          pushToast('Pairing command copied')
                        }}>
                  <Copy size={12} />
                </button>
              </div>
              <div className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', marginTop: 4 }}>
                python pipeline/live_pipeline.py --pair {pairInfo.pair_code}
              </div>
              <div style={{ fontSize: 9.5, color: 'var(--text-muted)', marginTop: 4 }}>
                Single-use, expires in ~15 min. The agent then detects damage with the
                RT-DETR lite pipeline and uploads it under your account.
              </div>
            </div>
          )}

          {/* Device list */}
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 180, overflowY: 'auto' }}>
            {devices === null && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading…</span>}
            {devices?.length === 0 && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>No connected devices yet.</span>}
            {devices?.map(d => (
              <div key={d.id} style={styles.deviceRow}>
                <span style={{
                  width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                  background: !d.device_id ? 'var(--orange)' : d.is_active ? 'var(--green)' : 'var(--text-muted)',
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 11.5, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {d.name} <span className="mono" style={{ fontSize: 9, color: 'var(--text-muted)' }}>· {d.kind}</span>
                  </div>
                  <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                    {!d.device_id ? `code ${d.pair_code || '…'} — waiting`
                      : !d.is_active ? 'disconnected'
                      : `${d.reports_sent} reports · ${fmtAgo(d.last_seen_at, now)}`}
                  </div>
                </div>
                <button className="btn btn-sm btn-ghost" title="Disconnect" style={{ width: 28, height: 28, padding: 0 }}
                        onClick={() => removeDevice(d)}>
                  <Trash2 size={11} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Locate-me (bottom-right) ─────────────────────────────────── */}
      <button
        className={`glass live-locate ${follow ? 'live-locate-on' : ''}`}
        title={follow ? 'Following you — drag the map to stop' : 'Centre on my position'}
        onClick={locateMe}
      >
        <Crosshair size={18} />
      </button>

      {/* ── Report button ────────────────────────────────────────────── */}
      <div className="live-fab-wrap" style={{ bottom: isMobile ? 92 : 22 }}>
        {reporting && !pickerAt && (
          <div className="glass anim-fade-in" style={styles.reportHint}>
            <Crosshair size={12} style={{ color: 'var(--accent)' }} />
            Tap the map where the damage is
          </div>
        )}
        {(!isMobile || !feedOpen) && (
          <button
            className={`btn ${reporting || pickerAt ? 'btn-danger' : 'btn-accent'} live-fab`}
            onClick={openReport}
          >
            {reporting || pickerAt ? <><X size={16} /> Cancel</> : <><Plus size={18} /> Report</>}
          </button>
        )}
      </div>

      {/* ── Type picker ──────────────────────────────────────────────── */}
      {pickerAt && (
        <div className="glass anim-fade-up live-picker">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span className="overline">What do you see?</span>
            <button className="btn btn-sm btn-ghost" style={{ width: 28, height: 28, padding: 0 }} onClick={closePicker}>
              <X size={13} />
            </button>
          </div>

          <div className="mono" style={{
            display: 'flex', alignItems: 'center', gap: 5, fontSize: 10,
            color: pickerMode === 'gps' ? 'var(--cyan)' : 'var(--text-muted)', marginBottom: 10,
          }}>
            <MapPin size={10} />
            {pickerMode === 'gps' ? 'At your position' : `${pickerAt[0].toFixed(5)}, ${pickerAt[1].toFixed(5)}`}
            {pickerMode === 'gps' && (
              <button className="link-btn" style={styles.linkBtn} onClick={pickOnMapInstead}>
                pick on map instead
              </button>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {PICKER_CLASSES.map((c, i) => (
              <button
                key={c}
                className="btn live-picker-btn"
                style={{
                  justifyContent: 'flex-start', gap: 8,
                  borderColor: `${CLASS_COLORS[c]}55`,
                  gridColumn: i === 0 ? '1 / -1' : undefined,
                  background: i === 0 ? `color-mix(in srgb, ${CLASS_COLORS[c]} 10%, transparent)` : undefined,
                }}
                onClick={() => submitReport(c)}
              >
                <span style={{ color: CLASS_COLORS[c], fontSize: 14 }}>{CLASS_ICONS[c]}</span>
                <span style={{ fontSize: 12 }}>{CLASS_LABELS[c]}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Selected hazard card (mobile — thumb-sized votes) ─────────── */}
      {isMobile && selected && !pickerAt && (
        <div className="glass anim-fade-up live-eventcard">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <ClassDot cls={selected.damage_type} size={32} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>
                {CLASS_LABELS[selected.damage_type] || selected.damage_type}
                {selected.severity && (
                  <span className="mono" style={{ fontSize: 10, fontWeight: 700, color: SEVERITY_COLORS[selected.severity], marginLeft: 6 }}>
                    S{selected.severity}
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 3 }}>
                <StatusChip status={selected.status} />
                <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                  <Users size={9} style={{ display: 'inline', marginRight: 2 }} />{selected.reporter_devices}
                </span>
                <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                  {fix ? fmtDist(distM(fix.lat, fix.lon, selected.latitude, selected.longitude)) : fmtAgo(selected.last_reported, now)}
                </span>
              </div>
            </div>
            <button className="btn btn-sm btn-ghost" style={{ width: 30, height: 30, padding: 0 }}
                    onClick={() => setSelectedId(null)}>
              <X size={14} />
            </button>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
            <button className="btn" disabled={busyId === selected.id}
                    style={{ flex: 1, padding: '11px 0', color: 'var(--green)', borderColor: 'rgba(61,220,132,0.4)' }}
                    onClick={() => vote(selected, 'confirm')}>
              <ThumbsUp size={14} /> Still there
            </button>
            <button className="btn" disabled={busyId === selected.id}
                    style={{ flex: 1, padding: '11px 0', color: 'var(--orange)', borderColor: 'rgba(255,159,67,0.4)' }}
                    onClick={() => vote(selected, 'dispute')}>
              <ThumbsDown size={14} /> Not there
            </button>
            {isOperator && (
              <button className="btn" title="Mark repaired (operator)" disabled={busyId === selected.id}
                      style={{ padding: '11px 14px', color: 'var(--cyan)', borderColor: 'rgba(76,201,240,0.4)' }}
                      onClick={() => resolve(selected)}>
                <CheckCircle2 size={14} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Feed: right panel (desktop) / bottom sheet (mobile) ───────── */}
      <div className={`glass live-feed ${isMobile ? (feedOpen ? 'live-feed-open' : 'live-feed-peek') : ''}`}>
        <div
          style={{ ...styles.feedHeader, cursor: isMobile ? 'pointer' : 'default' }}
          onClick={() => isMobile && setFeedOpen(v => !v)}
        >
          <span className="overline" style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            {isMobile && (feedOpen ? <ChevronDown size={13} /> : <ChevronUp size={13} />)}
            Live feed
          </span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {eventList.length} active
          </span>
        </div>
        <div style={styles.feedBody}>
          {eventList.length === 0 && (
            <div style={{ padding: '28px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.7 }}>
              No live hazards right now.<br />
              Start <strong>drive mode</strong>, pair a dashcam from
              the <strong>Devices</strong> panel, or tap <strong>Report</strong>.
            </div>
          )}
          {eventList.map(ev => {
            const isSel = selectedId === ev.id
            return (
              <div
                key={ev.id}
                onClick={() => {
                  setSelectedId(ev.id)
                  setFollow(false)
                  setFlyTarget({ lat: ev.latitude, lon: ev.longitude })
                  if (isMobile) setFeedOpen(false)
                }}
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

                {!isMobile && isSel && (
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
                    {isOperator && (
                      <button
                        className="btn btn-sm"
                        title="Mark repaired (operator)"
                        style={{ color: 'var(--cyan)', borderColor: 'rgba(76,201,240,0.4)' }}
                        disabled={busyId === ev.id}
                        onClick={(e) => { e.stopPropagation(); resolve(ev) }}
                      >
                        <CheckCircle2 size={12} />
                      </button>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Toasts ───────────────────────────────────────────────────── */}
      <div className="live-toasts">
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
        <div style={{ fontSize: 8.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.07em', whiteSpace: 'nowrap' }}>{label}</div>
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
    flexShrink: 0,
  },

  driveReadout: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 10,
    fontSize: 10.5,
    color: 'var(--text-dim)',
    padding: '7px 10px',
    marginTop: 6,
    border: '1px solid var(--border)',
    borderRadius: 8,
    background: 'var(--bg-card2)',
  },
  pairBox: {
    marginTop: 8,
    padding: '10px 12px',
    border: '1px dashed var(--border-accent)',
    borderRadius: 10,
    background: 'var(--accent-dim)',
  },
  deviceRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '7px 9px',
    border: '1px solid var(--border)',
    borderRadius: 9,
    background: 'var(--bg-card2)',
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
  linkBtn: {
    background: 'none', border: 'none', padding: 0, marginLeft: 6, cursor: 'pointer',
    color: 'var(--accent)', fontWeight: 600, fontSize: 10, fontFamily: 'var(--font-mono)',
    textDecoration: 'underline',
  },

  feedHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 15px',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
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

  toast: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    padding: '9px 14px',
    fontSize: 12,
    color: 'var(--text)',
    maxWidth: 320,
    pointerEvents: 'none',
  },
}
