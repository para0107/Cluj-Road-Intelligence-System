import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Filter, X, Download, MapPin, Copy, Check } from 'lucide-react'
import { fetchDetections, updateDetectionStatus, deleteDetectionsBulk } from '../utils/api'
import { CLASS_COLORS, CLASS_LABELS, SEVERITY_COLORS } from '../utils/constants'

const DAMAGE_TYPES = [
  'longitudinal_crack','transverse_crack','alligator_crack','repaired_crack',
  'pothole','pedestrian_crossing_blur','lane_line_blur','manhole_cover',
  'patchy_road','rutting',
]

function SortIcon({ col, sortCol, sortDir }) {
  if (sortCol !== col) return <ChevronUp size={11} style={{ opacity: 0.2 }} />
  return sortDir === 'desc'
    ? <ChevronDown size={11} style={{ color: 'var(--accent)' }} />
    : <ChevronUp   size={11} style={{ color: 'var(--accent)' }} />
}

export default function ExplorerPage() {
  const navigate = useNavigate()

  // Pagination
  const [page,     setPage]     = useState(1)
  const PAGE_SIZE = 25

  // Filters
  const [damageType,  setDamageType]  = useState('')
  const [severityMin, setSeverityMin] = useState('')
  const [severityMax, setSeverityMax] = useState('')
  const [dateFrom,    setDateFrom]    = useState('')
  const [dateTo,      setDateTo]      = useState('')

  // Sort (client-side on the fetched page)
  const [sortCol, setSortCol] = useState('priority_score')
  const [sortDir, setSortDir] = useState('desc')

  // Data
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [selectedIds, setSelectedIds] = useState([])
  const [deleteSurveyLog, setDeleteSurveyLog] = useState(true)
  const [copiedId, setCopiedId] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = {
        page,
        page_size: PAGE_SIZE,
        ...(damageType  && { damage_type:  damageType  }),
        ...(severityMin && { severity_min: Number(severityMin) }),
        ...(severityMax && { severity_max: Number(severityMax) }),
        ...(dateFrom    && { date_from:    dateFrom    }),
        ...(dateTo      && { date_to:      dateTo      }),
        sort_by: sortCol,
        sort_order: sortDir,
      }
      const result = await fetchDetections(params)
      setData(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [page, damageType, severityMin, severityMax, dateFrom, dateTo, sortCol, sortDir])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    setSelectedIds([])
  }, [page, damageType, severityMin, severityMax, dateFrom, dateTo, sortCol, sortDir])

  // Reset to page 1 when filters change
  const applyFilter = () => { setSelectedIds([]); setPage(1); load() }
  const clearFilters = () => {
    setSelectedIds([])
    setDamageType(''); setSeverityMin(''); setSeverityMax('')
    setDateFrom(''); setDateTo(''); setPage(1)
  }

  const items = data?.items || []

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  const toggleSort = (col) => {
    setSelectedIds([])
    setPage(1) // Reset to page 1 on sort change
    if (sortCol === col) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortCol(col); setSortDir('desc')
    }
  }

  const handleToggleFixed = async (id, currentFixed) => {
    try {
      const updated = await updateDetectionStatus(id, !currentFixed)
      setData(prev => ({
        ...prev,
        items: prev.items.map(item => item.id === id ? { ...item, is_fixed: updated.is_fixed } : item)
      }))
    } catch (e) {
      console.error("Failed to update status:", e)
    }
  }

  // Copy "lat, lon" to the clipboard so a repair crew can paste the exact
  // location into a maps app / work order. 6 decimals ≈ 0.1 m precision.
  const handleCopyCoords = async (item) => {
    if (item.latitude == null || item.longitude == null) return
    const text = `${item.latitude.toFixed(6)}, ${item.longitude.toFixed(6)}`
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(item.id)
      setTimeout(() => setCopiedId(c => (c === item.id ? null : c)), 1500)
    } catch (e) {
      console.error('Failed to copy coordinates:', e)
    }
  }

  const toggleSelected = (id) => {
    setSelectedIds(prev => (
      prev.includes(id)
        ? prev.filter(selectedId => selectedId !== id)
        : [...prev, id]
    ))
  }

  const selectAllVisible = () => {
    const visibleIds = items.map(item => item.id)
    if (!visibleIds.length) return

    const allVisibleSelected = visibleIds.every(id => selectedIds.includes(id))
    setSelectedIds(prev => (
      allVisibleSelected
        ? prev.filter(id => !visibleIds.includes(id))
        : Array.from(new Set([...prev, ...visibleIds]))
    ))
  }

  const handleBulkDelete = async () => {
    if (!selectedIds.length) return

    const confirmMessage = deleteSurveyLog
      ? `Delete ${selectedIds.length} selected detections and matching survey_log rows?`
      : `Delete ${selectedIds.length} selected detections?`

    if (!window.confirm(confirmMessage)) return

    try {
      setLoading(true)
      await deleteDetectionsBulk(selectedIds, deleteSurveyLog)
      setSelectedIds([])
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSingleDelete = async (id) => {
    const item = items.find(entry => entry.id === id)
    const confirmMessage = deleteSurveyLog
      ? `Delete this detection and matching survey_log row?`
      : `Delete this detection?`

    if (!window.confirm(confirmMessage)) return

    try {
      setLoading(true)
      await deleteDetectionsBulk([id], deleteSurveyLog)
      setSelectedIds(prev => prev.filter(selectedId => selectedId !== id))
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const hasFilters = damageType || severityMin || severityMax || dateFrom || dateTo
  const visibleIds = items.map(item => item.id)
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every(id => selectedIds.includes(id))
  const selectedCount = selectedIds.length

  return (
    <div style={styles.page}>
      {/* ── Header ────────────────────────────────────────────────── */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.backBtn} onClick={() => navigate('/')}>
            <ArrowLeft size={14} /> MAP
          </button>
          <div>
            <h1 style={styles.title}>Explorer</h1>
            <p style={styles.subtitle}>
              {data ? `${data.total.toLocaleString()} detections` : 'Loading…'}
              {hasFilters && <span style={{ color: 'var(--accent)', marginLeft: 6 }}>· filtered</span>}
            </p>
          </div>
        </div>
        <div style={styles.headerRight}>
          <a href="/api/export/csv" download style={styles.exportBtn}>
            <Download size={13} /> EXPORT CSV
          </a>
        </div>
      </div>

      <div style={styles.body}>
        {/* ── Filter bar ─────────────────────────────────────────── */}
        <div style={styles.filterBar}>
          <div style={styles.filterBarLeft}>
            <Filter size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />

            {/* Damage type */}
            <select
              value={damageType}
              onChange={e => { setDamageType(e.target.value); setPage(1) }}
              style={styles.select}
            >
              <option value="">All classes</option>
              {DAMAGE_TYPES.map(t => (
                <option key={t} value={t}>{CLASS_LABELS[t] || t}</option>
              ))}
            </select>

            {/* Severity range */}
            <select
              value={severityMin}
              onChange={e => { setSeverityMin(e.target.value); setPage(1) }}
              style={styles.select}
            >
              <option value="">Sev ≥</option>
              {[1,2,3,4,5].map(s => <option key={s} value={s}>S{s}+</option>)}
            </select>
            <select
              value={severityMax}
              onChange={e => { setSeverityMax(e.target.value); setPage(1) }}
              style={styles.select}
            >
              <option value="">Sev ≤</option>
              {[1,2,3,4,5].map(s => <option key={s} value={s}>S{s}-</option>)}
            </select>

            {/* Date range */}
            <input
              type="date"
              value={dateFrom}
              onChange={e => { setDateFrom(e.target.value); setPage(1) }}
              style={styles.dateInput}
              placeholder="From"
            />
            <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>→</span>
            <input
              type="date"
              value={dateTo}
              onChange={e => { setDateTo(e.target.value); setPage(1) }}
              style={styles.dateInput}
              placeholder="To"
            />

          </div>

          <div style={styles.filterActions}>
            {selectedCount > 0 && (
              <div style={styles.bulkActions}>
                <span style={styles.bulkCount}>{selectedCount} selected</span>
                <label style={styles.bulkCheckLabel}>
                  <input
                    type="checkbox"
                    checked={deleteSurveyLog}
                    onChange={e => setDeleteSurveyLog(e.target.checked)}
                    style={styles.bulkCheckbox}
                  />
                  Delete survey_log too
                </label>
                <button style={styles.deleteBtn} onClick={handleBulkDelete}>
                  Delete selected
                </button>
              </div>
            )}

            {hasFilters && (
              <button style={styles.clearBtn} onClick={clearFilters}>
                <X size={12} /> Clear
              </button>
            )}
          </div>
        </div>

        {/* ── Table ──────────────────────────────────────────────── */}
        <div style={styles.tableWrap}>
          {/* Head */}
          <div style={styles.tableHead}>
            <div style={{ ...styles.th, justifyContent: 'center' }}>Select</div>
            {[
              { key: 'damage_type',     label: 'Class'          },
              { key: 'severity',        label: 'Severity'       },
              { key: 'confidence',      label: 'Confidence'     },
              { key: 'priority_score',  label: 'Priority'       },
              { key: 'detection_count', label: 'Seen'           },
              { key: 'latitude',        label: 'GPS (Lat, Lon)' },
            ].map(col => (
              <div
                key={col.key}
                style={{ ...styles.th, cursor: 'pointer', userSelect: 'none' }}
                onClick={() => toggleSort(col.key)}
              >
                {col.label}
                <SortIcon col={col.key} sortCol={sortCol} sortDir={sortDir} />
              </div>
            ))}
            <div style={{ ...styles.th, justifyContent: 'center' }}>Fixed</div>
            <div style={{ ...styles.th, justifyContent: 'center' }}>Delete</div>
          </div>

          {/* Body */}
          {loading && (
            <div style={styles.tableLoading}>
              <div style={styles.spinner} />
              <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading…</span>
            </div>
          )}

          {!loading && error && (
            <div style={styles.tableError}>
              Could not load detections: {error}
            </div>
          )}

          {!loading && !error && items.map((item, i) => {
            const clsColor = CLASS_COLORS[item.damage_type] || '#888'
            const sevColor = SEVERITY_COLORS[item.severity] || '#888'
            const hasCoords = item.latitude != null && item.longitude != null
            return (
              <div
                key={item.id}
                style={{
                  ...styles.tableRow,
                  background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.012)',
                  boxShadow: selectedIds.includes(item.id) ? 'inset 0 0 0 1px var(--accent)' : 'none',
                }}
              >
                <div style={{ ...styles.td, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(item.id)}
                    onChange={() => toggleSelected(item.id)}
                    style={styles.rowCheckbox}
                    aria-label={`Select detection ${item.id}`}
                  />
                </div>

                {/* Class */}
                <div style={{ ...styles.td, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: clsColor, flexShrink: 0,
                  }} />
                  <span style={{ fontSize: 12 }}>
                    {CLASS_LABELS[item.damage_type] || item.damage_type}
                  </span>
                </div>

                {/* Severity */}
                <div style={styles.td}>
                  <span style={{
                    background: `${sevColor}20`, color: sevColor,
                    border: `1px solid ${sevColor}45`,
                    borderRadius: 4, padding: '2px 7px',
                    fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700,
                  }}>
                    S{item.severity}
                  </span>
                </div>

                {/* Confidence */}
                <div style={styles.td}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{
                      width: 48, height: 4, borderRadius: 2,
                      background: 'var(--border-bright)', overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${(item.confidence || 0) * 100}%`,
                        height: '100%', background: clsColor,
                      }} />
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>
                      {((item.confidence || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Priority */}
                <div style={{ ...styles.td, fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 12 }}>
                  {(item.priority_score || 0).toFixed(4)}
                </div>

                {/* Seen */}
                <div style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                  {item.detection_count}×
                </div>

                {/* GPS coordinates (click to copy) */}
                <div style={styles.td}>
                  <button
                    type="button"
                    onClick={() => handleCopyCoords(item)}
                    style={styles.gpsCell}
                    title={hasCoords ? 'Click to copy coordinates' : 'No GPS for this detection'}
                    disabled={!hasCoords}
                  >
                    <MapPin size={12} style={{ color: clsColor, flexShrink: 0 }} />
                    <span style={styles.gpsCoords}>
                      {hasCoords
                        ? `${item.latitude.toFixed(6)}, ${item.longitude.toFixed(6)}`
                        : '—'}
                    </span>
                    {hasCoords && (copiedId === item.id
                      ? <Check size={12} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      : <Copy  size={12} style={{ color: 'var(--text-muted)', flexShrink: 0, opacity: 0.6 }} />)}
                  </button>
                </div>

                {/* Fixed Checkbox */}
                <div style={{ ...styles.td, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={item.is_fixed}
                    onChange={() => handleToggleFixed(item.id, item.is_fixed)}
                    style={styles.rowCheckbox}
                  />
                </div>

                {/* Delete action */}
                <div style={{ ...styles.td, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <button
                    type="button"
                    onClick={() => handleSingleDelete(item.id)}
                    style={styles.rowDeleteBtn}
                    aria-label={`Delete detection ${item.id}`}
                    title="Delete detection"
                  >
                    Delete
                  </button>
                </div>
              </div>
            )
          })}

          {!loading && !error && items.length === 0 && (
            <div style={styles.tableEmpty}>No detections match the current filters.</div>
          )}
        </div>

        {/* ── Pagination ─────────────────────────────────────────── */}
        {data && data.total > PAGE_SIZE && (
          <div style={styles.pagination}>
            <button
              style={styles.pageBtn}
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft size={14} />
            </button>

            <div style={styles.pageInfo}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                Page {page} of {totalPages}
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                ({data.total.toLocaleString()} total)
              </span>
            </div>

            <button
              style={styles.pageBtn}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              <ChevronRight size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// Explicit per-column widths give every column its own breathing room and keep
// the wide GPS column from being squeezed by the narrow numeric ones.
// Order: Select · Class · Severity · Confidence · Priority · Seen · GPS · Fixed · Delete
const COL = '56px minmax(150px,1.6fr) 96px 150px 110px 80px minmax(190px,1.5fr) 72px 96px'

const styles = {
  page: { paddingTop: 48, minHeight: '100vh', background: 'var(--bg)' },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '24px 32px 0',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 16 },
  backBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '6px 12px', background: 'transparent',
    border: '1px solid var(--border-bright)', borderRadius: 'var(--radius)',
    color: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)',
    fontWeight: 700, cursor: 'pointer', letterSpacing: '.08em',
  },
  headerRight: { display: 'flex', alignItems: 'center' },
  filterActions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
    flexWrap: 'wrap',
    marginLeft: 'auto',
  },
  exportBtn: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 14px', background: 'var(--accent-dim)',
    border: '1px solid var(--accent)', borderRadius: 'var(--radius)',
    color: 'var(--accent)', fontSize: 11, fontFamily: 'var(--font-mono)',
    fontWeight: 700, cursor: 'pointer', letterSpacing: '.08em', textDecoration: 'none',
    transition: 'var(--transition)',
  },
  title: { fontSize: 26, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.5px' },
  subtitle: { fontSize: 12, color: 'var(--text-muted)', marginTop: 2 },
  body: { padding: '20px 32px 48px' },

  filterBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 8, padding: '12px 16px',
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', marginBottom: 16,
    flexWrap: 'wrap',
  },
  filterBarLeft: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', flex: 1 },
  select: {
    background: 'var(--bg-card2)', border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)', color: 'var(--text)',
    fontSize: 12, padding: '5px 10px', cursor: 'pointer',
    fontFamily: 'var(--font-sans)',
  },
  dateInput: {
    background: 'var(--bg-card2)', border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)', color: 'var(--text)',
    fontSize: 12, padding: '5px 10px',
    fontFamily: 'var(--font-sans)',
  },
  searchWrap: { position: 'relative', display: 'flex', alignItems: 'center' },
  clearBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '4px 9px', background: 'rgba(255,68,68,0.08)',
    border: '1px solid rgba(255,68,68,0.24)', borderRadius: 'var(--radius)',
    color: 'var(--text)', fontSize: 10, cursor: 'pointer',
    fontFamily: 'var(--font-mono)', fontWeight: 700,
  },
  bulkActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '5px 8px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255,68,68,0.05)',
    border: '1px solid rgba(255,68,68,0.16)',
    flexWrap: 'wrap',
    maxWidth: '100%',
  },
  bulkCount: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    color: 'var(--text-muted)',
    whiteSpace: 'nowrap',
  },
  bulkCheckLabel: {
    display: 'flex', alignItems: 'center', gap: 5,
    fontSize: 10, color: 'var(--text-muted)',
    cursor: 'pointer', userSelect: 'none',
    whiteSpace: 'nowrap',
  },
  bulkCheckbox: {
    width: 13, height: 13,
    cursor: 'pointer', accentColor: 'var(--accent)',
  },
  deleteBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '4px 9px', background: 'rgba(255,68,68,0.1)',
    border: '1px solid rgba(255,68,68,0.28)', borderRadius: 'var(--radius)',
    color: 'var(--text)', fontSize: 10, cursor: 'pointer',
    fontFamily: 'var(--font-mono)', fontWeight: 700,
    whiteSpace: 'nowrap',
  },

  tableWrap: {
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', overflowX: 'auto',
  },
  tableHead: {
    display: 'grid', gridTemplateColumns: COL, columnGap: 10,
    background: 'var(--bg-card2)', borderBottom: '1px solid var(--border)',
    minWidth: 'fit-content',
  },
  tableRow: {
    display: 'grid', gridTemplateColumns: COL, columnGap: 10,
    borderBottom: '1px solid var(--border)',
    transition: 'background 0.1s',
    minWidth: 'fit-content',
  },
  th: {
    padding: '12px 16px', fontSize: 10, fontFamily: 'var(--font-mono)',
    fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '.08em',
    textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 4,
  },
  td: { padding: '12px 16px', fontSize: 13, color: 'var(--text)', alignSelf: 'center' },
  rowCheckbox: {
    width: 15,
    height: 15,
    cursor: 'pointer',
    accentColor: 'var(--accent)',
  },
  gpsCell: {
    display: 'inline-flex', alignItems: 'center', gap: 7,
    maxWidth: '100%', padding: '4px 8px',
    background: 'var(--bg-card2)', border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)', cursor: 'pointer',
    fontFamily: 'var(--font-mono)',
  },
  gpsCoords: {
    fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-dim)',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
    letterSpacing: '.02em',
  },
  rowDeleteBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 56,
    height: 28,
    padding: '0 10px',
    borderRadius: 'var(--radius)',
    border: '1px solid rgba(255,68,68,0.18)',
    background: 'rgba(255,68,68,0.05)',
    color: 'var(--red)',
    cursor: 'pointer',
    flexShrink: 0,
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '.06em',
  },

  tableLoading: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    gap: 12, padding: 48,
  },
  tableError: {
    padding: 32, textAlign: 'center', color: 'var(--red)', fontSize: 13,
  },
  tableEmpty: {
    padding: 48, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13,
  },
  spinner: {
    width: 18, height: 18, border: '2px solid var(--border)',
    borderTop: '2px solid var(--accent)', borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },

  pagination: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    gap: 16, padding: '20px 0 0',
  },
  pageBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: 32, height: 32,
    background: 'var(--bg-card)', border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)', color: 'var(--text)', cursor: 'pointer',
    transition: 'var(--transition)',
  },
  pageInfo: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 },
}
