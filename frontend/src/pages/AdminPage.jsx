/**
 * frontend/src/pages/AdminPage.jsx — user & role management (admin only).
 */

import React, { useEffect, useState } from 'react'
import { Shield, AlertTriangle, Users } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { fetchUsers, setUserRole } from '../utils/api'
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
  const [error, setError] = useState(null)
  const [busyId, setBusyId] = useState(null)

  const load = async () => {
    try {
      const data = await fetchUsers()
      setUsers(data.items || [])
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
  }

  useEffect(() => { load() }, [])

  const changeRole = async (u, role) => {
    let city = u.city
    if (role === 'municipality' && !city) {
      city = window.prompt(`City for ${u.username}'s municipality account:`, 'Cluj-Napoca')
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
                            <span style={{ fontWeight: 600, display: 'block' }}>{u.username}</span>
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
}
