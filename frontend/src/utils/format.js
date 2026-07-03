// Small formatting helpers shared across pages.

export const fmtNum = (n, digits = 0) =>
  n === null || n === undefined || Number.isNaN(n)
    ? '—'
    : Number(n).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const fmtPct = (x, digits = 0) =>
  x === null || x === undefined ? '—' : `${(x * 100).toFixed(digits)}%`

export const fmtCoord = (lat, lon) =>
  lat === null || lat === undefined ? '—' : `${lat.toFixed(5)}, ${lon.toFixed(5)}`

export const fmtDate = (d) => {
  if (!d) return '—'
  const date = typeof d === 'string' ? new Date(d) : d
  if (Number.isNaN(date.getTime())) return String(d)
  return date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
}

export const fmtTime = (d) => {
  if (!d) return '—'
  const date = typeof d === 'string' ? new Date(d) : d
  if (Number.isNaN(date.getTime())) return String(d)
  return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const fmtRon = (n) =>
  n === null || n === undefined
    ? '—'
    : `${Math.round(n).toLocaleString('en-US')} RON`

export const fmtDuration = (startIso, endIso) => {
  if (!startIso || !endIso) return '—'
  const ms = new Date(endIso) - new Date(startIso)
  if (Number.isNaN(ms) || ms < 0) return '—'
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `${m}m ${s % 60}s`
}
