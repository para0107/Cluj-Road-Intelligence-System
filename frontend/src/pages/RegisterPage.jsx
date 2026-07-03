/**
 * frontend/src/pages/RegisterPage.jsx — create an account.
 *
 * Two self-service roles:
 *   Citizen       — report and vote on live hazards, upload surveys
 *   Municipality  — a city administration account; must pick its city and
 *                   additionally gets operator powers (resolve, mark repaired)
 * Full admin is granted only by an existing admin (Admin page).
 */

import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { UserPlus, AlertTriangle, User as UserIcon, Landmark } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [form, setForm] = useState({
    full_name: '', username: '', email: '', password: '',
    role: 'user', city: '',
  })
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    if (form.role === 'municipality' && !form.city.trim()) {
      setError('Municipality accounts must select their city.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      await register({
        ...form,
        full_name: form.full_name.trim() || null,
        city: form.city.trim() || null,
      })
      navigate('/', { replace: true })
    } catch (err) {
      const detail = err?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : (detail?.[0]?.msg || err.message || 'Registration failed'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div className="card anim-fade-up" style={styles.card}>
        <div className="overline" style={{ color: 'var(--accent)', marginBottom: 8 }}>
          JOIN THE NETWORK
        </div>
        <h1 className="display" style={styles.title}>Create an account</h1>
        <div className="road-divider" style={{ width: 120, margin: '14px 0 22px' }} />

        {error && (
          <div style={styles.error}>
            <AlertTriangle size={13} style={{ flexShrink: 0 }} />
            {error}
          </div>
        )}

        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Role choice */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <RoleCard
              icon={UserIcon} label="Citizen" active={form.role === 'user'}
              sub="Report & confirm hazards"
              onClick={() => setForm(f => ({ ...f, role: 'user' }))}
            />
            <RoleCard
              icon={Landmark} label="Municipality" active={form.role === 'municipality'}
              sub="Manage a city's repairs"
              onClick={() => setForm(f => ({ ...f, role: 'municipality' }))}
            />
          </div>

          <label style={styles.label}>
            Full name
            <input className="input" value={form.full_name} onChange={set('full_name')} autoComplete="name" />
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <label style={styles.label}>
              Username
              <input className="input" value={form.username} onChange={set('username')}
                     required minLength={3} maxLength={40} pattern="[a-zA-Z0-9_.\-]+"
                     title="Letters, digits, dots, dashes, underscores" autoComplete="username" />
            </label>
            <label style={styles.label}>
              City {form.role === 'municipality' && <span style={{ color: 'var(--accent)' }}>*</span>}
              <input className="input" value={form.city} onChange={set('city')}
                     placeholder="Cluj-Napoca" required={form.role === 'municipality'} />
            </label>
          </div>
          <label style={styles.label}>
            E-mail
            <input className="input" type="email" value={form.email} onChange={set('email')}
                   required autoComplete="email" />
          </label>
          <label style={styles.label}>
            Password
            <input className="input" type="password" value={form.password} onChange={set('password')}
                   required minLength={8} autoComplete="new-password" />
          </label>

          <button className="btn btn-accent" style={{ padding: '11px 0', marginTop: 6 }} disabled={busy}>
            <UserPlus size={15} /> {busy ? 'Creating…' : 'Create account'}
          </button>
        </form>

        <div style={styles.footer}>
          <span style={{ color: 'var(--text-muted)' }}>Already registered?</span>{' '}
          <Link to="/login" style={{ color: 'var(--accent)', fontWeight: 600 }}>Sign in</Link>
        </div>
      </div>
    </div>
  )
}

function RoleCard({ icon: Icon, label, sub, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="card"
      style={{
        padding: '12px 12px', cursor: 'pointer', textAlign: 'left',
        borderColor: active ? 'var(--border-accent)' : 'var(--border)',
        background: active ? 'var(--accent-dim)' : 'var(--bg-card2)',
      }}
    >
      <Icon size={15} style={{ color: active ? 'var(--accent)' : 'var(--text-muted)', marginBottom: 6 }} />
      <div style={{ fontSize: 12.5, fontWeight: 700, color: active ? 'var(--text)' : 'var(--text-dim)' }}>{label}</div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
    </button>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 'var(--nav-h)',
  },
  card: { width: 440, padding: '30px 32px', margin: '40px 16px' },
  title: { fontSize: 26, fontWeight: 700, letterSpacing: '-0.02em' },
  label: { display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: 'var(--text-dim)' },
  error: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '9px 12px', marginBottom: 14, borderRadius: 8,
    background: 'rgba(255,93,93,0.1)', border: '1px solid rgba(255,93,93,0.35)',
    color: 'var(--red)', fontSize: 12,
  },
  footer: { marginTop: 18, fontSize: 12.5, textAlign: 'center' },
}
