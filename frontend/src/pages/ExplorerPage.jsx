/**
 * frontend/src/pages/ExplorerPage.jsx — Detection audit table.
 *
 * Server-side filtering / sorting / pagination via GET /api/detections.
 * Row actions: view on map, copy GPS, mark repaired, bulk delete, CSV export.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight, X, Download,
  MapPin, Copy, Check, Trash2, Wrench, RotateCcw, Table as TableIcon,
  AlertTriangle,
} from 'lucide-react'
import { fetchDetections, updateDetectionStatus, deleteDetectionsBulk, downloadCsv } from '../utils/api'
import { CLASS_LABELS, SEVERITY_COLORS, ALL_CLASSES } from '../utils/constants'
import { fmtDate } from '../utils/format'
import { SevBadge, ClassDot, SectionTitle, Spinner, CenterState, EmptyState } from '../components/ui'

const PAGE_SIZE = 25

function SortIcon({ col, sortCol, sortDir }) {
  if (sortCol !== col) return <ChevronUp size={11} style={{ opacity: 0.2 }} />
  return sortDir === 'desc'
    ? <ChevronDown size={11} style={{ color: 'var(--accent)' }} />
    : <ChevronUp size={11} style={{ color: 'var(--accent)' }} />
}

export default function ExplorerPage() {
  const navigate = useNavigate()

  // Pagination + filters + sort (all server-side)
  const [page, setPage] = useState(1)
  const [damageType, setDamageType] = useState('')
  const [severityMin, setSeverityMin] = useState('')
  const [severityMax, setSeverityMax] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [sortCol, setSortCol] = useState('priority_score')
  const [sortDir, setSortDir] = useState('desc')

  // Data + UI state
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedIds, setSelectedIds] = useState([])
  const [deleteSurveyLog, setDeleteSurveyLog] = useState(false)
  const [copiedId, setCopiedId] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = {
        page,
        page_size: PAGE_SIZE,
        ...(damageType && { damage_type: damageType }),
        ...(severityMin && { severity_min: Number(severityMin) }),
        ...(severityMax && { severity_max: Number(severityMax) }),
        ...(dateFrom && { date_from: dateFrom }),
        ...(dateTo && { date_to: dateTo }),
        sort_by: sortCol,
        sort_order: sortDir,
      }
      const result = await fetchDetections(params)
      setData(result)
      setSelectedIds([])
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }, [page, damageType, severityMin, severityMax, dateFrom, dateTo, sortCol, sortDir])

  useEffect(() => { load() }, [load])

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => (d === 'desc' ? 'asc' : 'desc'))
    else { setSortCol(col); setSortDir('desc') }
    setPage(1)
  }

  const resetFilters = () => {
    setDamageType(''); setSeverityMin(''); setSeverityMax('')
    setDateFrom(''); setDateTo(''); setPage(1)
  }

  const hasFilters = damageType || severityMin || severityMax || dateFrom || dateTo

  const items = data?.items || []
  const total = data?.total || 0
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const allChecked = items.length > 0 && selectedIds.length === items.length
  const toggleAll = () =>
    setSelectedIds(allChecked ? [] : items.map(d => d.id))
  const toggleOne = (id) =>
    setSelectedIds(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]))

  const copyGps = (d) => {
    navigator.clipboard?.writeText(`${d.latitude.toFixed(6)}, ${d.longitude.toFixed(6)}`)
    setCopiedId(d.id)
    setTimeout(() => setCopiedId(null), 1200)
  }

  const bulkDelete = async () => {
    if (selectedIds.length === 0) return
    if (!window.confirm(`Delete ${selectedIds.length} detection(s)? This cannot be undone.`)) return
    setBusy(true)
    try {
      await deleteDetectionsBulk(selectedIds, deleteSurveyLog)
      await load()
    } catch (e) {
      alert(`Delete failed: ${e?.response?.data?.detail || e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const toggleFixed = async (d) => {
    setBusy(true)
    try {
      const updated = await updateDetectionStatus(d.id, !d.is_fixed)
      setData(prev => ({
        ...prev,
        items: prev.items.map(x => (x.id === d.id ? { ...x, is_fixed: updated.is_fixed } : x)),
      }))
    } catch (e) {
      alert(`Update failed: ${e?.response?.data?.detail || e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const headers = [
    { key: '_check', label: '', sortable: false, width: 34 },
    { key: 'damage_type', label: 'Type', sortable: true },
    { key: 'severity', label: 'Severity', sortable: true },
    { key: 'confidence', label: 'Conf', sortable: true },
    { key: 'priority_score', label: 'Priority', sortable: true },
    { key: 'detection_count', label: 'Seen', sortable: true },
    { key: 'latitude', label: 'GPS', sortable: true },
    { key: 'last_detected', label: 'Last seen', sortable: true },
    { key: '_status', label: 'Status', sortable: false },
    { key: '_actions', label: '', sortable: false, width: 120 },
  ]

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>
        <SectionTitle
          overline="Audit"
          title="Detection explorer"
          right={
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-sm" onClick={downloadCsv}>
                <Download size={13} /> Export CSV
              </button>
            </div>
          }
        />

        {/* ── Filter bar ─────────────────────────────────────────────── */}
        <div className="card anim-fade-up" style={styles.filterBar}>
          <select className="select" value={damageType} onChange={e => { setDamageType(e.target.value); setPage(1) }}>
            <option value="">All classes</option>
            {ALL_CLASSES.map(c => <option key={c} value={c}>{CLASS_LABELS[c]}</option>)}
          </select>

          <select className="select" value={severityMin} onChange={e => { setSeverityMin(e.target.value); setPage(1) }}>
            <option value="">Min severity</option>
            {[1, 2, 3, 4, 5].map(s => <option key={s} value={s}>S{s}+</option>)}
          </select>

          <select className="select" value={severityMax} onChange={e => { setSeverityMax(e.target.value); setPage(1) }}>
            <option value="">Max severity</option>
            {[1, 2, 3, 4, 5].map(s => <option key={s} value={s}>≤ S{s}</option>)}
          </select>

          <input className="input" type="date" value={dateFrom} onChange={e => { setDateFrom(e.target.value); setPage(1) }} title="From date" />
          <input className="input" type="date" value={dateTo} onChange={e => { setDateTo(e.target.value); setPage(1) }} title="To date" />

          {hasFilters && (
            <button className="btn btn-sm btn-ghost" onClick={resetFilters}>
              <X size={12} /> Clear
            </button>
          )}

          <span style={{ marginLeft: 'auto', fontSize: 11.5, color: 'var(--text-muted)' }}>
            <span className="mono" style={{ color: 'var(--accent)' }}>{total.toLocaleString()}</span> records
          </span>
        </div>

        {/* ── Bulk action bar ───────────────────────────────────────── */}
        {selectedIds.length > 0 && (
          <div className="glass anim-fade-in" style={styles.bulkBar}>
            <span style={{ fontSize: 12.5 }}>
              <span className="mono" style={{ color: 'var(--accent)', fontWeight: 700 }}>{selectedIds.length}</span> selected
            </span>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5, color: 'var(--text-muted)', cursor: 'pointer' }}>
              <input type="checkbox" checked={deleteSurveyLog} onChange={e => setDeleteSurveyLog(e.target.checked)} />
              also delete affected survey-log rows
            </label>
            <button className="btn btn-sm btn-danger" onClick={bulkDelete} disabled={busy}>
              <Trash2 size={13} /> Delete selected
            </button>
          </div>
        )}

        {/* ── Table ─────────────────────────────────────────────────── */}
        <div className="card anim-fade-up delay-1" style={{ overflow: 'hidden' }}>
          {error ? (
            <EmptyState icon={AlertTriangle} title="Could not load detections" sub={error}
              action={<button className="btn" onClick={load}>Retry</button>} />
          ) : loading && !data ? (
            <CenterState><Spinner label="Loading records…" /></CenterState>
          ) : items.length === 0 ? (
            <EmptyState icon={TableIcon} title="No records match"
              sub={hasFilters ? 'Try clearing some filters.' : 'Upload a survey to create detections.'} />
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    {headers.map(h => (
                      <th
                        key={h.key}
                        style={{ ...styles.th, width: h.width, cursor: h.sortable ? 'pointer' : 'default' }}
                        onClick={h.sortable ? () => toggleSort(h.key) : undefined}
                      >
                        {h.key === '_check' ? (
                          <input type="checkbox" checked={allChecked} onChange={toggleAll} />
                        ) : (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                            {h.label}
                            {h.sortable && <SortIcon col={h.key} sortCol={sortCol} sortDir={sortDir} />}
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {items.map(d => (
                    <tr key={d.id} className="table-row-hover" style={{ opacity: d.is_fixed ? 0.55 : 1 }}>
                      <td style={styles.td}>
                        <input type="checkbox" checked={selectedIds.includes(d.id)} onChange={() => toggleOne(d.id)} />
                      </td>
                      <td style={styles.td}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                          <ClassDot cls={d.damage_type} size={24} />
                          <span style={{ fontWeight: 600, fontSize: 12.5 }}>{CLASS_LABELS[d.damage_type] || d.damage_type}</span>
                        </span>
                      </td>
                      <td style={styles.td}><SevBadge s={d.severity} compact /></td>
                      <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                        {(d.confidence * 100).toFixed(0)}%
                      </td>
                      <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--accent)' }}>
                        {(d.priority_score || 0).toFixed(3)}
                      </td>
                      <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                        {d.detection_count}×
                      </td>
                      <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                        {d.latitude.toFixed(4)}, {d.longitude.toFixed(4)}
                      </td>
                      <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-dim)' }}>
                        {fmtDate(d.last_detected)}
                      </td>
                      <td style={styles.td}>
                        {d.is_fixed ? (
                          <span style={{ color: 'var(--green)', fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>REPAIRED</span>
                        ) : (
                          <span style={{ color: SEVERITY_COLORS[d.severity] || 'var(--text-muted)', fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>OPEN</span>
                        )}
                      </td>
                      <td style={{ ...styles.td, whiteSpace: 'nowrap' }}>
                        <span style={{ display: 'inline-flex', gap: 4 }}>
                          <IconBtn title="Show on map" onClick={() => navigate('/map')}>
                            <MapPin size={12} />
                          </IconBtn>
                          <IconBtn title="Copy GPS" onClick={() => copyGps(d)}>
                            {copiedId === d.id ? <Check size={12} style={{ color: 'var(--green)' }} /> : <Copy size={12} />}
                          </IconBtn>
                          <IconBtn title={d.is_fixed ? 'Reopen' : 'Mark repaired'} onClick={() => toggleFixed(d)} disabled={busy}>
                            {d.is_fixed ? <RotateCcw size={12} /> : <Wrench size={12} />}
                          </IconBtn>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── Pagination footer ───────────────────────────────────── */}
          {items.length > 0 && (
            <div style={styles.pager}>
              <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Page <span className="mono" style={{ color: 'var(--text)' }}>{page}</span> of{' '}
                <span className="mono">{pageCount}</span>
              </span>
              <div style={{ display: 'flex', gap: 6 }}>
                <button className="btn btn-sm" disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>
                  <ChevronLeft size={13} /> Prev
                </button>
                <button className="btn btn-sm" disabled={page >= pageCount || loading} onClick={() => setPage(p => p + 1)}>
                  Next <ChevronRight size={13} />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function IconBtn({ children, title, onClick, disabled }) {
  return (
    <button
      className="btn btn-sm btn-ghost"
      style={{ width: 26, height: 26, padding: 0, border: '1px solid var(--border)' }}
      title={title}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 26px)',
    paddingBottom: 40,
  },
  inner: {
    maxWidth: 1160,
    margin: '0 auto',
    padding: '0 26px',
  },
  filterBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
    padding: '12px 14px',
    marginBottom: 12,
  },
  bulkBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    padding: '10px 16px',
    marginBottom: 12,
    borderColor: 'var(--border-accent)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  th: {
    textAlign: 'left',
    padding: '11px 12px',
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-bright)',
    background: 'var(--bg-card2)',
    whiteSpace: 'nowrap',
    userSelect: 'none',
  },
  td: {
    padding: '9px 12px',
    fontSize: 12.5,
    borderBottom: '1px solid var(--border)',
    verticalAlign: 'middle',
  },
  pager: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 16px',
  },
}
