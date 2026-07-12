/**
 * frontend/src/pages/WorkOrdersPage.jsx — the repair workflow board.
 *
 * A work order groups nearby damage into one job for one crew and moves from
 * open to verified. The board is the flow; the drawer is one job: its fields,
 * its sites, and the crew route for the day.
 *
 * Verifying is guarded on the server: if a site that was marked fixed shows up
 * in a later survey, the PATCH comes back 409 and we say so instead of
 * pretending the job is done.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import {
  ClipboardList, Plus, X, Trash2, AlertTriangle, Save, HardHat,
  Calendar, Banknote, MapPin, RefreshCw, Route as RouteIcon,
} from 'lucide-react'
import {
  fetchWorkOrders, fetchWorkOrder, createWorkOrder, updateWorkOrder, deleteWorkOrder,
} from '../utils/api'
import {
  WORK_ORDER_BOARD, WORK_ORDER_STATUSES, WORK_ORDER_LABELS, WORK_ORDER_COLORS,
  CLASS_LABELS,
} from '../utils/constants'
import { fmtDate, fmtRon } from '../utils/format'
import { SevBadge, ClassDot, Spinner, CenterState, EmptyState, KvRow } from '../components/ui'
import SpotlightCard from '../reactbits/SpotlightCard/SpotlightCard'
import WorkOrderRoutePanel from '../components/WorkOrderRoutePanel'
import useIsMobile from '../hooks/useIsMobile'

// ── Error text from an axios failure (detail can be a string or an object) ──
function errText(e, fallback) {
  const d = e?.response?.data?.detail
  if (typeof d === 'string') return d
  if (d && typeof d.message === 'string') return d.message
  return e?.message || fallback
}

const EMPTY_FORM = {
  title: '',
  crew_name: '',
  scheduled_for: '',
  due_date: '',
  cost_estimate_ron: '',
  cost_actual_ron: '',
  notes: '',
}

const num = (v) => {
  if (v === '' || v === null || v === undefined) return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

export default function WorkOrdersPage() {
  const isMobile = useIsMobile()
  const location = useLocation()
  const preselected = useMemo(
    () => location.state?.detectionIds || [],
    [location.state],
  )

  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const reload = useCallback(() => setReloadKey((k) => k + 1), [])

  const [filter, setFilter] = useState('open')          // mobile list only

  // ── New work order modal ──────────────────────────────────────────────
  const [modalOpen, setModalOpen] = useState(false)
  const [newForm, setNewForm] = useState(EMPTY_FORM)
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState('')

  // ── Detail drawer ─────────────────────────────────────────────────────
  const [openId, setOpenId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [form, setForm] = useState(EMPTY_FORM)
  const [savingFields, setSavingFields] = useState(false)
  const [statusBusy, setStatusBusy] = useState(false)
  const [drawerNote, setDrawerNote] = useState('')
  const [verifyBlock, setVerifyBlock] = useState(null)   // {message, ids:[]}
  const [deleting, setDeleting] = useState(false)

  // ── Load the board ────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true
    setLoading(true)
    fetchWorkOrders({ page_size: 200 })
      .then((res) => {
        if (!alive) return
        setOrders(res?.items || [])
        setError('')
      })
      .catch((e) => {
        if (!alive) return
        setError(errText(e, 'Could not load the work orders.'))
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [reloadKey])

  // ── Load one work order into the drawer ───────────────────────────────
  useEffect(() => {
    if (!openId) {
      setDetail(null)
      setDetailError('')
      setVerifyBlock(null)
      setDrawerNote('')
      return
    }
    let alive = true
    setDetailLoading(true)
    setDetailError('')
    fetchWorkOrder(openId)
      .then((wo) => {
        if (!alive) return
        setDetail(wo)
        setForm({
          title: wo.title || '',
          crew_name: wo.crew_name || '',
          scheduled_for: (wo.scheduled_for || '').slice(0, 10),
          due_date: (wo.due_date || '').slice(0, 10),
          cost_estimate_ron: wo.cost_estimate_ron ?? '',
          cost_actual_ron: wo.cost_actual_ron ?? '',
          notes: wo.notes || '',
        })
      })
      .catch((e) => {
        if (!alive) return
        setDetailError(errText(e, 'Could not open this work order.'))
      })
      .finally(() => { if (alive) setDetailLoading(false) })
    return () => { alive = false }
  }, [openId])

  // ── Actions ───────────────────────────────────────────────────────────
  const create = async () => {
    if (preselected.length === 0 || !newForm.title.trim()) return
    setCreating(true)
    setCreateError('')
    try {
      const wo = await createWorkOrder({
        title: newForm.title.trim(),
        detection_ids: preselected,
        crew_name: newForm.crew_name.trim() || null,
        scheduled_for: newForm.scheduled_for || null,
        due_date: newForm.due_date || null,
        cost_estimate_ron: num(newForm.cost_estimate_ron),
        notes: newForm.notes.trim() || null,
      })
      setModalOpen(false)
      setNewForm(EMPTY_FORM)
      reload()
      if (wo?.id) setOpenId(wo.id)
    } catch (e) {
      setCreateError(errText(e, 'The work order could not be created.'))
    } finally {
      setCreating(false)
    }
  }

  const changeStatus = async (next) => {
    if (!detail || next === detail.status) return
    setStatusBusy(true)
    setDrawerNote('')
    setVerifyBlock(null)
    try {
      const wo = await updateWorkOrder(detail.id, { status: next })
      setDetail(wo)
      reload()
    } catch (e) {
      if (e?.response?.status === 409) {
        const d = e.response.data?.detail || {}
        setVerifyBlock({
          message: d.message || 'Some sites were detected again after they were marked fixed.',
          ids: d.reopened_detection_ids || [],
        })
      } else {
        setDrawerNote(errText(e, 'The status could not be changed.'))
      }
    } finally {
      setStatusBusy(false)
    }
  }

  const saveFields = async () => {
    if (!detail) return
    setSavingFields(true)
    setDrawerNote('')
    try {
      const wo = await updateWorkOrder(detail.id, {
        title: form.title.trim() || detail.title,
        crew_name: form.crew_name.trim() || null,
        scheduled_for: form.scheduled_for || null,
        due_date: form.due_date || null,
        cost_estimate_ron: num(form.cost_estimate_ron),
        cost_actual_ron: num(form.cost_actual_ron),
        notes: form.notes.trim() || null,
      })
      setDetail(wo)
      reload()
      setDrawerNote('Saved.')
    } catch (e) {
      setDrawerNote(errText(e, 'The changes could not be saved.'))
    } finally {
      setSavingFields(false)
    }
  }

  const saveOrder = useCallback(async (detectionIds) => {
    if (!openId) return
    await updateWorkOrder(openId, { item_order: detectionIds })
  }, [openId])

  const remove = async () => {
    if (!detail) return
    const ok = window.confirm(`Delete "${detail.title}"? The damage sites stay in the system.`)
    if (!ok) return
    setDeleting(true)
    try {
      await deleteWorkOrder(detail.id)
      setOpenId(null)
      reload()
    } catch (e) {
      setDrawerNote(errText(e, 'The work order could not be deleted.'))
    } finally {
      setDeleting(false)
    }
  }

  // ── Derived ───────────────────────────────────────────────────────────
  const byStatus = useMemo(() => {
    const map = {}
    WORK_ORDER_BOARD.forEach((s) => { map[s] = [] })
    orders.forEach((o) => {
      if (map[o.status]) map[o.status].push(o)
    })
    return map
  }, [orders])

  const listed = useMemo(
    () => (filter === 'all' ? orders : orders.filter((o) => o.status === filter)),
    [orders, filter],
  )

  const reopenedIds = new Set(verifyBlock?.ids || [])

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="page-grid-bg" style={styles.page}>
      {/* Scoped skin for the vendored SpotlightCard (it ships its own dark
          hardcoded look; two classes beat one, so these win in both themes). */}
      <style>{`
        .wo-card.card-spotlight {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 13px 14px;
          cursor: pointer;
          box-shadow: var(--shadow);
          transition: var(--transition);
        }
        .wo-card.card-spotlight:hover {
          border-color: var(--border-accent);
          transform: translateY(-2px);
          box-shadow: var(--shadow-lg);
        }
      `}</style>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="anim-fade-up" style={{ marginBottom: 22 }}>
        <div style={styles.headRow}>
          <div>
            <div className="overline" style={{ color: 'var(--accent)', marginBottom: 6 }}>
              Repair workflow
            </div>
            <h1 className="display" style={styles.h1}>Work orders</h1>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-sm" onClick={reload} disabled={loading}>
              <RefreshCw size={13} /> Refresh
            </button>
            <button className="btn btn-sm btn-accent" onClick={() => { setCreateError(''); setModalOpen(true) }}>
              <Plus size={13} /> New work order
            </button>
          </div>
        </div>
        <div className="road-divider" style={{ margin: '14px 0 12px' }} />
        <p style={styles.lede}>
          A work order groups nearby damage into one job for one crew, and moves from open to verified.
        </p>
      </div>

      {/* ── Board ──────────────────────────────────────────────────────── */}
      {loading ? (
        <CenterState><Spinner label="Loading work orders…" /></CenterState>
      ) : error ? (
        <div style={styles.errorBanner}>
          <AlertTriangle size={15} />
          <span>{error}</span>
          <button className="btn btn-sm btn-ghost" onClick={reload}>Try again</button>
        </div>
      ) : orders.length === 0 ? (
        <EmptyState
          icon={ClipboardList}
          title="No work orders yet"
          sub="Create one from the Repairs page, or by selecting a zone on the Map."
        />
      ) : isMobile ? (
        <>
          <select
            className="input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '100%', marginBottom: 12 }}
          >
            <option value="all">All statuses</option>
            {WORK_ORDER_STATUSES.map((s) => (
              <option key={s} value={s}>{WORK_ORDER_LABELS[s]}</option>
            ))}
          </select>

          {listed.length === 0 ? (
            <EmptyState
              icon={ClipboardList}
              title="Nothing here"
              sub="No work orders have this status right now."
            />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {listed.map((o, i) => (
                <OrderCard key={o.id} order={o} index={i} onOpen={() => setOpenId(o.id)} showStatus />
              ))}
            </div>
          )}
        </>
      ) : (
        <div style={styles.board}>
          {WORK_ORDER_BOARD.map((status) => {
            const color = WORK_ORDER_COLORS[status]
            const column = byStatus[status] || []
            return (
              <div key={status} style={styles.column}>
                <div style={styles.columnHead}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <span className="display" style={{ fontSize: 12.5, fontWeight: 700 }}>
                      {WORK_ORDER_LABELS[status]}
                    </span>
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {column.length}
                    </span>
                  </div>
                  <div style={{ height: 3, borderRadius: 2, background: color, marginTop: 8, opacity: 0.9 }} />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                  {column.length === 0 ? (
                    <div style={styles.columnEmpty}>Nothing here</div>
                  ) : (
                    column.map((o, i) => (
                      <OrderCard key={o.id} order={o} index={i} onOpen={() => setOpenId(o.id)} />
                    ))
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── New work order modal ───────────────────────────────────────── */}
      {modalOpen && (
        <div style={styles.overlay} onClick={() => setModalOpen(false)}>
          <div
            className="card anim-fade-up"
            style={styles.modal}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHead}>
              <span className="display" style={{ fontSize: 15, fontWeight: 700 }}>New work order</span>
              <button className="btn btn-sm btn-ghost" onClick={() => setModalOpen(false)}>
                <X size={14} />
              </button>
            </div>

            {preselected.length > 0 ? (
              <div style={styles.selectedNote}>
                <MapPin size={13} style={{ color: 'var(--accent)' }} />
                <span>{preselected.length} detections selected</span>
              </div>
            ) : (
              <div style={styles.hintNote}>
                This page cannot pick damage sites on its own. Create work orders from the Repairs page,
                or by selecting a zone on the Map.
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
              <Field label="Title">
                <input
                  className="input"
                  style={styles.fill}
                  value={newForm.title}
                  placeholder="Manastur potholes, week 24"
                  onChange={(e) => setNewForm({ ...newForm, title: e.target.value })}
                />
              </Field>
              <Field label="Crew name">
                <input
                  className="input"
                  style={styles.fill}
                  value={newForm.crew_name}
                  placeholder="Crew A"
                  onChange={(e) => setNewForm({ ...newForm, crew_name: e.target.value })}
                />
              </Field>
              <div style={styles.twoCol}>
                <Field label="Scheduled for">
                  <input
                    type="date"
                    className="input"
                    style={styles.fill}
                    value={newForm.scheduled_for}
                    onChange={(e) => setNewForm({ ...newForm, scheduled_for: e.target.value })}
                  />
                </Field>
                <Field label="Due date">
                  <input
                    type="date"
                    className="input"
                    style={styles.fill}
                    value={newForm.due_date}
                    onChange={(e) => setNewForm({ ...newForm, due_date: e.target.value })}
                  />
                </Field>
              </div>
              <Field label="Cost estimate (RON)">
                <input
                  type="number"
                  min="0"
                  className="input"
                  style={styles.fill}
                  value={newForm.cost_estimate_ron}
                  onChange={(e) => setNewForm({ ...newForm, cost_estimate_ron: e.target.value })}
                />
              </Field>
              <Field label="Notes">
                <textarea
                  className="input"
                  style={{ ...styles.fill, minHeight: 70, resize: 'vertical' }}
                  value={newForm.notes}
                  onChange={(e) => setNewForm({ ...newForm, notes: e.target.value })}
                />
              </Field>
            </div>

            {createError && <div style={{ ...styles.errorBanner, marginTop: 12 }}>{createError}</div>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button className="btn btn-sm btn-ghost" onClick={() => setModalOpen(false)}>Cancel</button>
              <button
                className="btn btn-sm btn-accent"
                onClick={create}
                disabled={creating || preselected.length === 0 || !newForm.title.trim()}
              >
                {creating ? 'Creating…' : 'Create work order'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Detail drawer ─────────────────────────────────────────────── */}
      {openId && (
        <>
          <div style={styles.drawerBackdrop} onClick={() => setOpenId(null)} />
          <div className="glass anim-slide-right" style={styles.drawer}>
            <div style={styles.drawerHead}>
              <span className="overline">Work order</span>
              <button className="btn btn-sm btn-ghost" onClick={() => setOpenId(null)}>
                <X size={14} />
              </button>
            </div>

            {detailLoading ? (
              <CenterState><Spinner label="Opening…" /></CenterState>
            ) : detailError ? (
              <div style={styles.errorBanner}>
                <AlertTriangle size={15} />
                <span>{detailError}</span>
              </div>
            ) : detail ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
                <div>
                  <h2 className="display" style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
                    {detail.title}
                  </h2>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <StatusPill status={detail.status} />
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {detail.item_count ?? (detail.items || []).length} sites
                    </span>
                    {detail.city && (
                      <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {detail.city}
                      </span>
                    )}
                  </div>
                </div>

                {/* Verify guard */}
                {verifyBlock && (
                  <div style={styles.warnBanner}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <AlertTriangle size={15} />
                      <strong style={{ fontSize: 12.5 }}>This job cannot be verified yet</strong>
                    </div>
                    <div style={{ fontSize: 12, lineHeight: 1.5 }}>
                      {verifyBlock.ids.length} {verifyBlock.ids.length === 1 ? 'site was' : 'sites were'} detected
                      again after being marked fixed. They are highlighted below. Repair them again, then verify.
                    </div>
                    {verifyBlock.message && (
                      <div style={{ fontSize: 11.5, color: 'var(--text-dim)', marginTop: 6 }}>
                        {verifyBlock.message}
                      </div>
                    )}
                  </div>
                )}

                {drawerNote && <div style={styles.note}>{drawerNote}</div>}

                {/* Status */}
                <div>
                  <div className="overline" style={{ marginBottom: 6 }}>Status</div>
                  <select
                    className="input"
                    style={styles.fill}
                    value={detail.status}
                    disabled={statusBusy}
                    onChange={(e) => changeStatus(e.target.value)}
                  >
                    {WORK_ORDER_STATUSES.map((s) => (
                      <option key={s} value={s}>{WORK_ORDER_LABELS[s]}</option>
                    ))}
                  </select>
                </div>

                {/* Fields */}
                <div>
                  <div className="overline" style={{ marginBottom: 8 }}>Details</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <Field label="Title">
                      <input
                        className="input"
                        style={styles.fill}
                        value={form.title}
                        onChange={(e) => setForm({ ...form, title: e.target.value })}
                      />
                    </Field>
                    <Field label="Crew">
                      <input
                        className="input"
                        style={styles.fill}
                        value={form.crew_name}
                        placeholder="Unassigned"
                        onChange={(e) => setForm({ ...form, crew_name: e.target.value })}
                      />
                    </Field>
                    <div style={styles.twoCol}>
                      <Field label="Scheduled for">
                        <input
                          type="date"
                          className="input"
                          style={styles.fill}
                          value={form.scheduled_for}
                          onChange={(e) => setForm({ ...form, scheduled_for: e.target.value })}
                        />
                      </Field>
                      <Field label="Due date">
                        <input
                          type="date"
                          className="input"
                          style={styles.fill}
                          value={form.due_date}
                          onChange={(e) => setForm({ ...form, due_date: e.target.value })}
                        />
                      </Field>
                    </div>
                    <div style={styles.twoCol}>
                      <Field label="Cost estimate (RON)">
                        <input
                          type="number"
                          min="0"
                          className="input"
                          style={styles.fill}
                          value={form.cost_estimate_ron}
                          onChange={(e) => setForm({ ...form, cost_estimate_ron: e.target.value })}
                        />
                      </Field>
                      <Field label="Cost actual (RON)">
                        <input
                          type="number"
                          min="0"
                          className="input"
                          style={styles.fill}
                          value={form.cost_actual_ron}
                          onChange={(e) => setForm({ ...form, cost_actual_ron: e.target.value })}
                        />
                      </Field>
                    </div>
                    <Field label="Notes">
                      <textarea
                        className="input"
                        style={{ ...styles.fill, minHeight: 66, resize: 'vertical' }}
                        value={form.notes}
                        onChange={(e) => setForm({ ...form, notes: e.target.value })}
                      />
                    </Field>
                    <button
                      className="btn btn-sm btn-accent"
                      onClick={saveFields}
                      disabled={savingFields}
                      style={{ alignSelf: 'flex-start' }}
                    >
                      <Save size={13} /> {savingFields ? 'Saving…' : 'Save changes'}
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div>
                  <div className="overline" style={{ marginBottom: 6 }}>Summary</div>
                  <KvRow k="Created" v={fmtDate(detail.created_at)} />
                  <KvRow k="Completed" v={detail.completed_at ? fmtDate(detail.completed_at) : 'not yet'} />
                  <KvRow k="Verified" v={detail.verified_at ? fmtDate(detail.verified_at) : 'not yet'} />
                  <KvRow k="Estimate" v={detail.cost_estimate_ron != null ? fmtRon(detail.cost_estimate_ron) : '—'} mono />
                  <KvRow k="Spent" v={detail.cost_actual_ron != null ? fmtRon(detail.cost_actual_ron) : '—'} mono />
                </div>

                {/* Sites */}
                <div>
                  <div className="overline" style={{ marginBottom: 8 }}>
                    Sites ({(detail.items || []).length})
                  </div>
                  {(detail.items || []).length === 0 ? (
                    <div style={styles.columnEmpty}>No damage sites on this work order.</div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                      {(detail.items || []).map((it, i) => {
                        const flagged = it.reopened || reopenedIds.has(it.detection_id)
                        return (
                          <div
                            key={it.detection_id}
                            style={{
                              ...styles.siteRow,
                              borderColor: flagged ? 'rgba(255, 93, 93, 0.5)' : 'var(--border)',
                              background: flagged ? 'rgba(255, 93, 93, 0.08)' : 'var(--bg-card)',
                            }}
                          >
                            <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)', width: 16 }}>
                              {i + 1}
                            </span>
                            <ClassDot cls={it.damage_type} size={26} />
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                                <span style={{ fontSize: 12, fontWeight: 600 }}>
                                  {CLASS_LABELS[it.damage_type] || it.damage_type}
                                </span>
                                <SevBadge s={it.severity} compact />
                                {flagged && <span style={styles.reopenChip}>Seen again</span>}
                              </div>
                              <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                                {it.latitude != null ? `${it.latitude.toFixed(5)}, ${it.longitude.toFixed(5)}` : 'no GPS'}
                              </div>
                            </div>
                            {it.is_fixed && !flagged && (
                              <span style={styles.fixedChip}>Fixed</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>

                {/* Route */}
                <div>
                  <div className="overline" style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <RouteIcon size={12} /> Crew route
                  </div>
                  <WorkOrderRoutePanel
                    items={detail.items || []}
                    onSaveOrder={saveOrder}
                    orderTitle={detail.title}
                    crewName={detail.crew_name}
                  />
                </div>

                <button
                  className="btn btn-sm btn-danger"
                  onClick={remove}
                  disabled={deleting}
                  style={{ alignSelf: 'flex-start', marginBottom: 10 }}
                >
                  <Trash2 size={13} /> {deleting ? 'Deleting…' : 'Delete work order'}
                </button>
              </div>
            ) : null}
          </div>
        </>
      )}
    </div>
  )
}

// ── Pieces ──────────────────────────────────────────────────────────────────

function Field({ label, children }) {
  return (
    <label style={{ display: 'block' }}>
      <span className="overline" style={{ display: 'block', marginBottom: 4 }}>{label}</span>
      {children}
    </label>
  )
}

function StatusPill({ status }) {
  const color = WORK_ORDER_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${color}22`, color, border: `1px solid ${color}55`,
      borderRadius: 999, padding: '2px 10px', fontSize: 11, fontWeight: 700,
      whiteSpace: 'nowrap',
    }}>
      {WORK_ORDER_LABELS[status] || status}
    </span>
  )
}

function OrderCard({ order, index, onOpen, showStatus = false }) {
  const color = WORK_ORDER_COLORS[order.status] || '#94a3b8'
  return (
    <div
      onClick={onOpen}
      className={`anim-fade-up delay-${Math.min(Math.floor(index / 3) + 1, 6)}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onOpen() }}
    >
      <SpotlightCard className="wo-card" spotlightColor="rgba(234, 255, 61, 0.14)">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
          <span className="display" style={{ fontSize: 12.5, fontWeight: 700, lineHeight: 1.35 }}>
            {order.title}
          </span>
          {showStatus && <StatusPill status={order.status} />}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
          <span className="mono" style={{ fontSize: 10.5, color }}>
            {order.item_count ?? 0} {order.item_count === 1 ? 'site' : 'sites'}
          </span>
          {order.crew_name && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 10.5, color: 'var(--text-muted)' }}>
              <HardHat size={10} /> {order.crew_name}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6, flexWrap: 'wrap' }}>
          {order.due_date && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 10.5, color: 'var(--text-muted)' }}>
              <Calendar size={10} /> {fmtDate(order.due_date)}
            </span>
          )}
          {order.cost_estimate_ron != null && (
            <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 10.5, color: 'var(--text-muted)' }}>
              <Banknote size={10} /> {fmtRon(order.cost_estimate_ron)}
            </span>
          )}
        </div>
      </SpotlightCard>
    </div>
  )
}

// ── Styles ──────────────────────────────────────────────────────────────────

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
  headRow: {
    display: 'flex',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 16,
    flexWrap: 'wrap',
  },
  h1: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: '-0.02em',
  },
  lede: {
    fontSize: 13,
    color: 'var(--text-dim)',
    maxWidth: 620,
  },
  board: {
    display: 'grid',
    gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
    gap: 12,
    alignItems: 'start',
  },
  column: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    minWidth: 0,
  },
  columnHead: {
    padding: '10px 12px',
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
  },
  columnEmpty: {
    fontSize: 11.5,
    color: 'var(--text-muted)',
    padding: '14px 12px',
    textAlign: 'center',
    border: '1px dashed var(--border)',
    borderRadius: 'var(--radius)',
  },
  overlay: {
    position: 'fixed',
    inset: 0,
    zIndex: 1200,
    background: 'rgba(0, 0, 0, 0.55)',
    backdropFilter: 'blur(3px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  modal: {
    width: 'min(520px, 100%)',
    maxHeight: '85vh',
    overflowY: 'auto',
    padding: 20,
  },
  modalHead: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  selectedNote: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    fontSize: 12,
    color: 'var(--text)',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    borderRadius: 'var(--radius)',
    padding: '8px 12px',
  },
  hintNote: {
    fontSize: 12,
    lineHeight: 1.55,
    color: 'var(--text-muted)',
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '10px 12px',
  },
  twoCol: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 10,
  },
  fill: {
    width: '100%',
  },
  drawerBackdrop: {
    position: 'fixed',
    inset: 0,
    zIndex: 1090,
    background: 'rgba(0, 0, 0, 0.45)',
  },
  drawer: {
    position: 'fixed',
    top: 'var(--nav-h)',
    right: 0,
    bottom: 0,
    width: 'min(460px, 100%)',
    zIndex: 1100,
    overflowY: 'auto',
    padding: 18,
    borderRadius: 0,
  },
  drawerHead: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  siteRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    padding: '8px 10px',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
  },
  reopenChip: {
    background: 'rgba(255, 93, 93, 0.16)',
    color: 'var(--red)',
    border: '1px solid rgba(255, 93, 93, 0.45)',
    borderRadius: 5,
    padding: '1px 7px',
    fontSize: 10,
    fontWeight: 700,
    whiteSpace: 'nowrap',
  },
  fixedChip: {
    background: 'rgba(61, 220, 132, 0.14)',
    color: 'var(--green)',
    border: '1px solid rgba(61, 220, 132, 0.4)',
    borderRadius: 5,
    padding: '1px 7px',
    fontSize: 10,
    fontWeight: 700,
    whiteSpace: 'nowrap',
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    background: 'rgba(255, 93, 93, 0.10)',
    border: '1px solid rgba(255, 93, 93, 0.4)',
    color: 'var(--red)',
    borderRadius: 'var(--radius)',
    padding: '10px 13px',
    fontSize: 12.5,
  },
  warnBanner: {
    background: 'rgba(255, 93, 93, 0.10)',
    border: '1px solid rgba(255, 93, 93, 0.45)',
    color: 'var(--red)',
    borderRadius: 'var(--radius)',
    padding: '11px 13px',
  },
  note: {
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    color: 'var(--text-dim)',
    borderRadius: 'var(--radius)',
    padding: '8px 12px',
    fontSize: 12,
  },
}
