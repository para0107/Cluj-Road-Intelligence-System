/**
 * frontend/src/pages/TriagePage.jsx — Operator triage inbox (/triage).
 *
 * Active citizen-reported live events, most community-validated first. The
 * operator either promotes one into an official detection (it then shows up on
 * the survey map, in the priority queue and in work orders) or dismisses it
 * (duplicate, off-road, bad signal).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Inbox, ShieldCheck, Users, Radio, MapPin, Gauge, Clock,
  Check, X, AlertTriangle, ArrowRight,
} from 'lucide-react'
import { fetchTriage, promoteEvent, dismissEvent } from '../utils/api'
import { fmtPct, fmtCoord, fmtDate } from '../utils/format'
import { CLASS_LABELS, LIVE_STATUS_LABELS, LIVE_STATUS_COLORS } from '../utils/constants'
import { SevBadge, ClassDot, Spinner, CenterState, EmptyState, Kpi } from '../components/ui'
import SpotlightCard from '../reactbits/SpotlightCard/SpotlightCard'
import AnimatedContent from '../reactbits/AnimatedContent/AnimatedContent'
import useIsMobile from '../hooks/useIsMobile'

const PAGE_SIZE = 200

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'verified', label: 'Verified' },
  { key: 'confirmed', label: 'Confirmed' },
  { key: 'unverified', label: 'Reported once' },
]

/** API errors carry `detail` as a string, an object, or a validation array. */
function errMsg(err) {
  const d = err?.response?.data?.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map(x => x?.msg || String(x)).join(', ')
  if (d && typeof d === 'object') return d.message || JSON.stringify(d)
  return err?.message || 'Something went wrong. Please try again.'
}

const tally = (list, status) => list.filter(e => e.status === status).length

export default function TriagePage() {
  const isMobile = useIsMobile()

  const [filter, setFilter] = useState('all')
  const [items, setItems] = useState([])        // rows currently shown (server-filtered)
  const [baseItems, setBaseItems] = useState([]) // unfiltered set, drives the KPI strip
  const [totalWaiting, setTotalWaiting] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(() => new Set()) // per-row action in flight
  const [toast, setToast] = useState(null)

  const toastTimer = useRef(null)

  const showToast = useCallback((msg) => {
    setToast(msg)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 7000)
  }, [])

  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current) }, [])

  // ── Load the inbox ──────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true
    setLoading(true)
    setError(null)

    const listParams = filter === 'all'
      ? { page_size: PAGE_SIZE }
      : { status: filter, page_size: PAGE_SIZE }

    Promise.all([
      fetchTriage(listParams),
      // When a chip is active the list is only a slice, so pull the unfiltered
      // set as well to keep the KPI strip honest.
      filter === 'all' ? Promise.resolve(null) : fetchTriage({ page_size: PAGE_SIZE }),
    ])
      .then(([list, all]) => {
        if (!alive) return
        const base = all || list
        setItems(list.items || [])
        setBaseItems(base.items || [])
        setTotalWaiting(base.total ?? (base.items || []).length)
      })
      .catch(err => {
        if (!alive) return
        setError(errMsg(err))
        setItems([])
      })
      .finally(() => { if (alive) setLoading(false) })

    return () => { alive = false }
  }, [filter])

  const counts = useMemo(() => ({
    verified: tally(baseItems, 'verified'),
    confirmed: tally(baseItems, 'confirmed'),
    unverified: tally(baseItems, 'unverified'),
  }), [baseItems])

  // ── Row actions ─────────────────────────────────────────────────────────
  const setRowBusy = (id, on) => setBusy(prev => {
    const next = new Set(prev)
    on ? next.add(id) : next.delete(id)
    return next
  })

  const dropRow = (id) => {
    setItems(prev => prev.filter(e => e.id !== id))
    setBaseItems(prev => prev.filter(e => e.id !== id))
    setTotalWaiting(t => Math.max(0, t - 1))
  }

  const onPromote = async (ev) => {
    setRowBusy(ev.id, true)
    setError(null)
    try {
      await promoteEvent(ev.id)
      dropRow(ev.id)
      showToast('Report is now an official record. It is on the map and in the repair queue.')
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setRowBusy(ev.id, false)
    }
  }

  const onDismiss = async (ev) => {
    const label = CLASS_LABELS[ev.damage_type] || ev.damage_type
    if (!window.confirm(`Dismiss this ${label} report? It leaves the live map and no official record is created.`)) return
    setRowBusy(ev.id, true)
    setError(null)
    try {
      await dismissEvent(ev.id)
      dropRow(ev.id)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setRowBusy(ev.id, false)
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="page-grid-bg" style={styles.page}>
      {/* SpotlightCard ships its own dark, hardcoded surface (#111 / #222 /
          2rem). Re-skin it with the design tokens so it matches the house
          cards and still works in the light theme. Two classes = higher
          specificity, so this wins whatever the stylesheet order is. */}
      <style>{`
        .triage-row.card-spotlight {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow);
          padding: 14px 16px;
          transition: var(--transition);
        }
        .triage-row.card-spotlight:hover { border-color: var(--border-bright); }
      `}</style>

      {/* Header */}
      <div className="anim-fade-up">
        <div className="overline" style={{ color: 'var(--accent)', marginBottom: 6 }}>Citizen reports</div>
        <h1 className="display" style={styles.h1}>Triage</h1>
        <div className="road-divider" style={{ margin: '14px 0 12px' }} />
        <p style={styles.lede}>
          Reports that several drivers confirmed independently are the most reliable, so they are listed first.
        </p>
      </div>

      {/* KPI strip */}
      <div style={styles.kpiGrid}>
        <Kpi delay="delay-1" icon={Inbox} label="Waiting" countTo={totalWaiting}
             sub="citizen reports needing a decision" />
        <Kpi delay="delay-2" icon={ShieldCheck} label="Verified" countTo={counts.verified}
             sub="many drivers, strongest signal" color="var(--green)" />
        <Kpi delay="delay-3" icon={Users} label="Confirmed" countTo={counts.confirmed}
             sub="backed by a second driver" color="var(--yellow)" />
        <Kpi delay="delay-4" icon={Radio} label="Reported once" countTo={counts.unverified}
             sub="a single driver so far" color="var(--text-muted)" />
      </div>

      {/* Filters */}
      <div style={styles.filterRow}>
        {FILTERS.map(f => {
          const active = filter === f.key
          return (
            <button
              key={f.key}
              className="chip"
              onClick={() => setFilter(f.key)}
              style={{
                borderColor: active ? 'var(--accent)' : 'var(--border)',
                background: active ? 'var(--accent-dim)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--text-muted)',
                fontWeight: active ? 700 : 500,
              }}
            >
              {f.label}
            </button>
          )
        })}
      </div>

      {/* Success banner */}
      {toast && (
        <div className="anim-fade-up" style={styles.toast}>
          <Check size={15} style={{ color: 'var(--green)', flexShrink: 0 }} />
          <span style={{ flex: 1 }}>{toast}</span>
          <button className="btn btn-sm btn-ghost" onClick={() => setToast(null)} aria-label="Dismiss message">
            <X size={13} />
          </button>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="anim-fade-up" style={styles.errorBanner}>
          <AlertTriangle size={15} style={{ color: 'var(--red)', flexShrink: 0 }} />
          <span style={{ flex: 1 }}>{error}</span>
          <button className="btn btn-sm btn-ghost" onClick={() => setError(null)} aria-label="Dismiss error">
            <X size={13} />
          </button>
        </div>
      )}

      {/* Inbox */}
      {loading ? (
        <CenterState><Spinner label="Loading citizen reports…" /></CenterState>
      ) : items.length === 0 ? (
        <EmptyState
          icon={Inbox}
          title="No citizen reports are waiting"
          sub={filter === 'all'
            ? 'New ones appear here as drivers report them.'
            : 'Nothing matches this filter right now. Try another one.'}
          action={filter === 'all'
            ? <Link to="/live" className="btn">Open the live map <ArrowRight size={13} /></Link>
            : <button className="btn" onClick={() => setFilter('all')}>Show all reports</button>}
        />
      ) : (
        <AnimatedContent distance={30} duration={0.5} threshold={0.05}>
          <div style={styles.list}>
            {items.map(ev => {
              const rowBusy = busy.has(ev.id)
              const statusColor = LIVE_STATUS_COLORS[ev.status] || 'var(--text-muted)'
              const statusLabel = LIVE_STATUS_LABELS[ev.status] || ev.status
              const reporters = ev.reporter_devices ?? 0
              const disputes = ev.dispute_devices ?? 0

              return (
                <SpotlightCard key={ev.id} className="triage-row" spotlightColor="var(--accent-glow)">
                  <div style={{ ...styles.row, flexDirection: isMobile ? 'column' : 'row' }}>
                    {/* Identity */}
                    <div style={styles.rowMain}>
                      <ClassDot cls={ev.damage_type} size={34} />

                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={styles.titleLine}>
                          <span className="display" style={{ fontWeight: 700, fontSize: 13.5 }}>
                            {CLASS_LABELS[ev.damage_type] || ev.damage_type}
                          </span>

                          <span style={{
                            background: `${statusColor}22`, color: statusColor,
                            border: `1px solid ${statusColor}55`, borderRadius: 5,
                            padding: '2px 8px', fontSize: 10.5, fontWeight: 700,
                            fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap',
                          }}>
                            {statusLabel}
                          </span>

                          {ev.severity != null && <SevBadge s={ev.severity} compact />}

                          {ev.max_confidence != null && (
                            <span className="mono" style={styles.meta}>
                              <Gauge size={10} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
                              {fmtPct(ev.max_confidence)} confidence
                            </span>
                          )}
                        </div>

                        <div style={styles.subLine}>
                          {reporters} {reporters === 1 ? 'driver' : 'drivers'} reported, {disputes} disputed
                        </div>

                        <div style={styles.metaLine}>
                          <span className="mono" style={styles.meta}>
                            <MapPin size={10} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
                            {fmtCoord(ev.latitude, ev.longitude)}
                          </span>
                          <span style={styles.meta}>
                            <Clock size={10} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
                            reported {fmtDate(ev.first_reported)}, last seen {fmtDate(ev.last_reported)}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div style={{ ...styles.actions, width: isMobile ? '100%' : 'auto' }}>
                      <button
                        className="btn btn-sm btn-accent"
                        disabled={rowBusy}
                        onClick={() => onPromote(ev)}
                        style={{ flex: isMobile ? 1 : 'none' }}
                      >
                        <Check size={13} /> {rowBusy ? 'Working…' : 'Make it official'}
                      </button>
                      <button
                        className="btn btn-sm btn-ghost"
                        disabled={rowBusy}
                        onClick={() => onDismiss(ev)}
                        style={{ flex: isMobile ? 1 : 'none' }}
                      >
                        <X size={13} /> Dismiss
                      </button>
                      <Link to="/live" className="btn btn-sm btn-ghost" style={{ color: 'var(--text-muted)' }}>
                        View on live map
                      </Link>
                    </div>
                  </div>
                </SpotlightCard>
              )
            })}
          </div>
        </AnimatedContent>
      )}
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 28px)',
    paddingBottom: 60,
    paddingLeft: 20,
    paddingRight: 20,
    maxWidth: 1160,
    margin: '0 auto',
  },
  h1: {
    fontSize: 30,
    fontWeight: 700,
    letterSpacing: '-0.02em',
    lineHeight: 1.15,
  },
  lede: {
    fontSize: 13,
    color: 'var(--text-dim)',
    maxWidth: 620,
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: 14,
    marginTop: 22,
  },
  filterRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    margin: '22px 0 14px',
  },
  toast: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '11px 14px',
    marginBottom: 12,
    borderRadius: 'var(--radius)',
    border: '1px solid rgba(61, 220, 132, 0.4)',
    background: 'rgba(61, 220, 132, 0.08)',
    color: 'var(--text)',
    fontSize: 12.5,
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '11px 14px',
    marginBottom: 12,
    borderRadius: 'var(--radius)',
    border: '1px solid rgba(255, 93, 93, 0.45)',
    background: 'rgba(255, 93, 93, 0.08)',
    color: 'var(--text)',
    fontSize: 12.5,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
  },
  rowMain: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
    flex: 1,
    minWidth: 0,
    width: '100%',
  },
  titleLine: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
  },
  subLine: {
    fontSize: 11.5,
    color: 'var(--text-dim)',
    marginTop: 4,
  },
  metaLine: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    flexWrap: 'wrap',
    marginTop: 4,
  },
  meta: {
    fontSize: 10.5,
    color: 'var(--text-muted)',
    whiteSpace: 'nowrap',
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexShrink: 0,
    flexWrap: 'wrap',
  },
}
