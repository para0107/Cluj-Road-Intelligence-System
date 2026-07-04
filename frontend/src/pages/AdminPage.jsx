/**
 * frontend/src/pages/AdminPage.jsx — user & role management (admin only).
 */

import React, { useEffect, useState } from 'react'
import {
  Shield, AlertTriangle, Users, Landmark, CheckCircle2, XCircle,
  Trash2, Ban, RotateCcw, MailCheck, Clock,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import {
  fetchUsers, setUserRole, adminDeleteUser, adminSetActive,
  fetchPendingRegistrations, approveRegistration, denyRegistration,
} from '../utils/api'
import { fmtDate } from '../utils/format'
import { SectionTitle, Spinner, CenterState, EmptyState } from '../components/ui'

const ROLE_COLORS = {
  admin: 'var(--red)',
  municipality: 'var(--cyan)',
  user: 'var(--green)',
}

export default function AdminPage() {
  const { user: me } = useAuth()
  const [users, setUsers] = useState(null)
  const [pending, setPending] = useState([])
  const [error, setError] = useState(null)
  const [busyId, setBusyId] = useState(null)

  const load = async () => {
    try {
      const data = await fetchUsers()
      setUsers(data.items || [])
      const p = await fetchPendingRegistrations()
      setPending(p.items || [])
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
  }

  useEffect(() => { load() }, [])

  const changeRole = async (u, role) => {
    let city = u.city
    if (role === 'municipality' && !city) {
      city = window.prompt(`City for ${u.username}'s municipality account:`)
      if (!city) return
    }
    setBusyId(u.id)
    try {
      const updated = await setUserRole(u.id, role, city)
      setUsers(prev => prev.map(x => (x.id === u.id ? updated : x)))
    } catch (e) {
      alert(e?.response?.data?.detail || e.message)
    } finally {
      setBusyId(null)
    }
  }

  const toggleActive = async (u) => {
    setBusyId(u.id)
    try {
      const updated = await adminSetActive(u.id, !u.is_active)
      setUsers(prev => prev.map(x => (x.id === u.id ? updated : x)))
    } catch (e) {
      alert(e?.response?.data?.detail || e.message)
    } finally {
      setBusyId(null)
    }
  }

  const removeUser = async (u) => {
    if (!window.confirm(`Permanently delete the account "${u.username}" (${u.email})? This cannot be undone.`)) return
    setBusyId(u.id)
    try {
      await adminDeleteUser(u.id)
      setUsers(prev => prev.filter(x => x.id !== u.id))
    } catch (e) {
      alert(e?.response?.data?.detail || e.message)
    } finally {
      setBusyId(null)
    }
  }

  const decide = async (p, approve) => {
    if (!approve && !window.confirm(`Deny and delete the registration of "${p.username}"?`)) return
    setBusyId(p.id)
    try {
      if (approve) await approveRegistration(p.id)
      else await denyRegistration(p.id)
      await load()
    } catch (e) {
      alert(e?.response?.data?.detail || e.message)
    } finally {
      setBusyId(null)
    }
  }

  if (me?.role !== 'admin') {
    return (
      <div style={styles.page} className="page-grid-bg">
        <EmptyState icon={Shield} title="Admins only" sub="This page requires the admin role." />
      </div>
    )
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>
        <SectionTitle overline="Administration" title="Accounts & roles" />

        {/* ── Pending registrations (municipality approvals + unverified) ── */}
        {pending.length > 0 && (
          <div className="card anim-fade-up" style={{ marginBottom: 22, padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '13px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <Landmark size={14} style={{ color: 'var(--cyan)' }} />
              <span style={{ fontSize: 13, fontWeight: 700 }}>Pending registrations</span>
              <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>({pending.length})</span>
            </div>
            {pending.map(p => (
              <div key={p.id} style={styles.pendingRow}>
                <span style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ fontWeight: 600, fontSize: 12.5 }}>{p.username}</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>{p.email}</span>
                  <span style={{ display: 'block', fontSize: 10.5, color: 'var(--text-dim)', marginTop: 2 }}>
                    {p.role}{p.city ? ` · ${p.city}` : ''} · registered {fmtDate(p.created_at)}
                  </span>
                </span>
                {p.status === 'awaiting_approval' ? (
                  <>
                    <span className="mono" style={{ ...styles.pendingChip, color: 'var(--cyan)', borderColor: 'var(--cyan)' }}>
                      <MailCheck size={10} /> E-MAIL OK — NEEDS APPROVAL
                    </span>
                    <button className="btn btn-sm" disabled={busyId === p.id}
                            style={{ color: 'var(--green)', borderColor: 'rgba(61,220,132,0.4)' }}
                            onClick={() => decide(p, true)}>
                      <CheckCircle2 size={12} /> Approve
                    </button>
                    <button className="btn btn-sm" disabled={busyId === p.id}
                            style={{ color: 'var(--red)', borderColor: 'rgba(255,93,93,0.4)' }}
                            onClick={() => decide(p, false)}>
                      <XCircle size={12} /> Deny
                    </button>
                  </>
                ) : (
                  <span className="mono" style={styles.pendingChip}>
                    <Clock size={10} /> AWAITING E-MAIL CODE
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {error ? (
          <EmptyState icon={AlertTriangle} title="Could not load users" sub={error} />
        ) : users === null ? (
          <CenterState><Spinner label="Loading accounts…" /></CenterState>
        ) : users.length === 0 ? (
          <EmptyState icon={Users} title="No accounts yet" />
        ) : (
          <div className="card anim-fade-up" style={{ overflow: 'hidden' }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    {['User', 'E-mail', 'Role', 'City', 'Provider', 'Last login', ''].map(h => (
                      <th key={h} style={styles.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} className="table-row-hover">
                      <td style={styles.td}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                          <span style={{ ...styles.avatar, borderColor: `${ROLE_COLORS[u.role]}66`, color: ROLE_COLORS[u.role] }}>
                            {(u.username || '?')[0].toUpperCase()}
                          </span>
                          <span>
                            <span style={{ fontWeight: 600, display: 'block', opacity: u.is_active ? 1 : 0.5 }}>
                              {u.username}
                              {!u.is_active && (
                                <span className="mono" style={{ fontSize: 9, color: 'var(--orange)', marginLeft: 6 }}>DISABLED</span>
                              )}
                            </span>
                            <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>{u.full_name || '—'}</span>
                          </span>
                        </span>
                      </td>
                      <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{u.email}</td>
                      <td style={styles.td}>
                        <span className="mono" style={{
                          fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
                          color: ROLE_COLORS[u.role],
                          border: `1px solid ${ROLE_COLORS[u.role]}55`,
                          background: `color-mix(in srgb, ${ROLE_COLORS[u.role]} 10%, transparent)`,
                          borderRadius: 5, padding: '2px 8px', textTransform: 'uppercase',
                        }}>
                          {u.role}
                        </span>
                      </td>
                      <td style={{ ...styles.td, fontSize: 12 }}>{u.city || '—'}</td>
                      <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-muted)' }}>{u.auth_provider}</td>
                      <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-dim)' }}>{fmtDate(u.last_login_at)}</td>
                      <td style={styles.td}>
                        {u.id !== me.id ? (
                          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <select
                              className="select"
                              style={{ fontSize: 11.5, padding: '5px 8px' }}
                              value={u.role}
                              disabled={busyId === u.id}
                              onChange={e => changeRole(u, e.target.value)}
                            >
                              <option value="user">user</option>
                              <option value="municipality">municipality</option>
                              <option value="admin">admin</option>
                            </select>
                            <button className="btn btn-ghost btn-sm" disabled={busyId === u.id}
                                    title={u.is_active ? 'Disable account (keeps data)' : 'Re-enable account'}
                                    style={{ width: 28, height: 28, padding: 0, color: u.is_active ? 'var(--orange)' : 'var(--green)' }}
                                    onClick={() => toggleActive(u)}>
                              {u.is_active ? <Ban size={12} /> : <RotateCcw size={12} />}
                            </button>
                            <button className="btn btn-ghost btn-sm" disabled={busyId === u.id}
                                    title="Delete account permanently"
                                    style={{ width: 28, height: 28, padding: 0, color: 'var(--red)' }}
                                    onClick={() => removeUser(u)}>
                              <Trash2 size={12} />
                            </button>
                          </span>
                        ) : (
                          <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>you</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 26px)',
    paddingBottom: 40,
  },
  inner: { maxWidth: 980, margin: '0 auto', padding: '0 26px' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: {
    textAlign: 'left', padding: '11px 14px', fontSize: 10,
    fontFamily: 'var(--font-mono)', fontWeight: 700, letterSpacing: '0.1em',
    textTransform: 'uppercase', color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-bright)', background: 'var(--bg-card2)',
    whiteSpace: 'nowrap',
  },
  td: { padding: '10px 14px', fontSize: 12.5, borderBottom: '1px solid var(--border)', verticalAlign: 'middle' },
  avatar: {
    width: 28, height: 28, borderRadius: 8, flexShrink: 0,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    border: '1px solid', background: 'var(--bg-card2)',
    fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-display)',
  },
  pendingRow: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '11px 16px', borderBottom: '1px solid var(--border)',
  },
  pendingChip: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    fontSize: 9, fontWeight: 700, letterSpacing: '0.07em',
    color: 'var(--text-muted)', border: '1px solid var(--border)',
    borderRadius: 5, padding: '3px 8px', whiteSpace: 'nowrap',
  },
}
