/**
 * frontend/src/utils/live.js
 *
 * Live (Waze-like) mode client:
 *  - a stable per-browser device identity (localStorage)
 *  - REST wrappers for reports / confirm / dispute / resolve
 *  - a WebSocket client with auto-reconnect; callers should fall back to
 *    fetchLiveEvents() polling while the socket is down.
 */

import api, { API_ORIGIN } from './api'

// ── Device identity ─────────────────────────────────────────────────────────

export function getDeviceId() {
  let id = localStorage.getItem('rids_device_id')
  if (!id) {
    id = `web-${crypto.randomUUID ? crypto.randomUUID().slice(0, 8) : Math.random().toString(36).slice(2, 10)}`
    localStorage.setItem('rids_device_id', id)
  }
  return id
}

// ── REST ────────────────────────────────────────────────────────────────────

export const fetchLiveEvents = () =>
  api.get('/live/events').then(r => r.data)

export const fetchLiveStats = () =>
  api.get('/live/stats').then(r => r.data)

export const postLiveReport = ({ latitude, longitude, damage_type, confidence = null, severity = null, note = null }) =>
  api.post('/live/reports', {
    device_id: getDeviceId(),
    latitude, longitude, damage_type, confidence, severity, note,
  }).then(r => r.data)

export const confirmLiveEvent = (id, note = null) =>
  api.post(`/live/events/${id}/confirm`, { device_id: getDeviceId(), note }).then(r => r.data)

export const disputeLiveEvent = (id, note = null) =>
  api.post(`/live/events/${id}/dispute`, { device_id: getDeviceId(), note }).then(r => r.data)

export const resolveLiveEvent = (id) =>
  api.post(`/live/events/${id}/resolve`).then(r => r.data)

// ── Paired devices (phone drive mode / dashcam edge agents) ─────────────────

/** Register THIS browser/phone as a device (instant, no code). */
export const pairThisDevice = (name, kind = 'phone') =>
  api.post('/live/devices/pair', { name, kind, device_id: getDeviceId() }).then(r => r.data)

/** Create a pending device + single-use pairing code for an external agent. */
export const createPairCode = (name, kind = 'dashcam') =>
  api.post('/live/devices/pair', { name, kind }).then(r => r.data)

export const fetchMyDevices = () =>
  api.get('/live/devices').then(r => r.data)

export const disconnectDevice = (id) =>
  api.delete(`/live/devices/${id}`).then(r => r.data)

// ── WebSocket ───────────────────────────────────────────────────────────────

/**
 * Open the live push channel.
 *
 * @param {object} handlers
 *   onHello(events, clients)   initial snapshot
 *   onUpsert(event)            event created/updated
 *   onRemoved(eventId)         event expired/disputed/resolved
 *   onStatus('open'|'closed')  connection state (drive the polling fallback)
 * @returns {() => void} close function
 */
export function openLiveSocket({ onHello, onUpsert, onRemoved, onStatus }) {
  let ws = null
  let closed = false
  let retryMs = 1000
  let pingTimer = null

  // Same-origin by default; when the frontend is hosted elsewhere (Vercel),
  // VITE_API_URL points the socket straight at the backend origin. Vercel
  // rewrites cannot proxy WebSockets, so this direct path is what keeps the
  // Live page on push instead of the polling fallback.
  const url = API_ORIGIN
    ? `${API_ORIGIN.replace(/^http/, 'ws')}/api/live/ws`
    : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/live/ws`

  const connect = () => {
    if (closed) return
    try {
      ws = new WebSocket(url)
    } catch {
      scheduleRetry()
      return
    }

    ws.onopen = () => {
      retryMs = 1000
      onStatus?.('open')
      // Keep intermediaries from idling the connection out
      pingTimer = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) ws.send('ping')
      }, 25_000)
    }

    ws.onmessage = (e) => {
      let msg
      try { msg = JSON.parse(e.data) } catch { return }
      if (msg.type === 'hello') onHello?.(msg.events || [], msg.clients)
      else if (msg.type === 'event_upsert' && msg.event) onUpsert?.(msg.event)
      else if (msg.type === 'event_removed') onRemoved?.(msg.event_id)
    }

    ws.onclose = () => {
      if (pingTimer) { clearInterval(pingTimer); pingTimer = null }
      onStatus?.('closed')
      scheduleRetry()
    }
    ws.onerror = () => { try { ws.close() } catch { /* noop */ } }
  }

  const scheduleRetry = () => {
    if (closed) return
    setTimeout(connect, retryMs)
    retryMs = Math.min(retryMs * 2, 15_000)   // capped exponential backoff
  }

  connect()

  return () => {
    closed = true
    if (pingTimer) clearInterval(pingTimer)
    try { ws?.close() } catch { /* noop */ }
  }
}
