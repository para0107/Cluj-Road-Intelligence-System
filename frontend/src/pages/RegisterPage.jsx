/**
 * frontend/src/pages/RegisterPage.jsx — create an account.
 *
 * Two self-service roles:
 *   Citizen       — report and vote on live hazards, upload surveys
 *   Municipality  — a city administration account; must pick its city and
 *                   additionally gets operator powers (resolve, mark repaired)
 * Full admin is granted only by an existing admin (Admin page).
 */

import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  UserPlus, AlertTriangle, User as UserIcon, Landmark,
  MailCheck, ShieldCheck, RefreshCw,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { authVerifyEmail, authResendCode, fetchAuthConfig } from '../utils/api'
import CaptchaField, { HoneypotField } from '../components/CaptchaField'

export default function RegisterPage() {
  const { register, adoptSession } = useAuth()
  const navigate = useNavigate()

  const [form, setForm] = useState({
    full_name: '', username: '', email: '', password: '',
    role: 'user', city: '',
  })
  // stage: form → verify (e-mail code) → pending (municipality approval)
  const [stage, setStage] = useState('form')
  const [code, setCode] = useState('')
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  // Anti-bot. Both are inert unless the backend has CAPTCHA_ENABLED set.
  const [captchaEnabled, setCaptchaEnabled] = useState(false)
  const [captchaToken, setCaptchaToken] = useState(null)
  const [honeypot, setHoneypot] = useState('')

  useEffect(() => {
    fetchAuthConfig()
      .then(c => setCaptchaEnabled(Boolean(c.captcha_enabled)))
      .catch(() => {})
  }, [])

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  const fail = (err, fallback) => {
    const detail = err?.response?.data?.detail
    setError(typeof detail === 'string' ? detail : (detail?.[0]?.msg || err.message || fallback))
  }

  const submit = async (e) => {
    e.preventDefault()
    if (!form.city.trim()) {
      setError('Please select your city. The map opens on it.')
      return
    }
    if (captchaEnabled && !captchaToken) {
      setError('Finish the verification check first.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const outcome = await register({
        ...form,
        full_name: form.full_name.trim() || null,
        city: form.city.trim(),
        altcha: captchaToken || undefined,
        website: honeypot,   // must stay empty; only a script fills it in
      })
      if (outcome.status === 'verify_email') {
        setInfo(outcome.message)
        setStage('verify')
      } else if (outcome.status === 'awaiting_approval') {
        setInfo(outcome.message)
        setStage('pending')
      } else {
        navigate('/', { replace: true })
      }
    } catch (err) {
      fail(err, 'Registration failed')
    } finally {
      setBusy(false)
    }
  }

  const submitCode = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const outcome = await authVerifyEmail(form.email.trim(), code.trim())
      if (outcome.status === 'awaiting_approval') {
        setInfo(outcome.message)
        setStage('pending')
      } else {
        adoptSession(outcome)
        navigate('/', { replace: true })
      }
    } catch (err) {
      fail(err, 'Verification failed')
    } finally {
      setBusy(false)
    }
  }

  const resend = async () => {
    setBusy(true)
    setError(null)
    try {
      const outcome = await authResendCode(form.email.trim())
      setInfo(outcome.message)
    } catch (err) {
      fail(err, 'Could not resend the code')
    } finally {
      setBusy(false)
    }
  }

  // ── Stage: e-mail code entry ──────────────────────────────────────────────
  if (stage === 'verify') {
    return (
      <div style={styles.page} className="page-grid-bg">
        <div className="card anim-fade-up" style={styles.card}>
          <div className="overline" style={{ color: 'var(--accent)', marginBottom: 8 }}>
            ONE MORE STEP
          </div>
          <h1 className="display" style={styles.title}>Confirm your e-mail</h1>
          <div className="road-divider" style={{ width: 120, margin: '14px 0 22px' }} />

          {error && (
            <div style={styles.error}><AlertTriangle size={13} style={{ flexShrink: 0 }} />{error}</div>
          )}
          <div style={styles.notice}>
            <MailCheck size={13} style={{ flexShrink: 0, color: 'var(--accent)' }} />
            {info || `We sent a 6-digit code to ${form.email}.`}
          </div>

          <form onSubmit={submitCode} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label style={styles.label}>
              Confirmation code
              <input className="input mono" autoFocus required
                     value={code} onChange={e => setCode(e.target.value)}
                     inputMode="numeric" minLength={4} maxLength={12}
                     placeholder="123456"
                     style={{ letterSpacing: '0.35em', fontSize: 18, textAlign: 'center' }} />
            </label>
            <button className="btn btn-accent" style={{ padding: '11px 0' }} disabled={busy || !code.trim()}>
              <ShieldCheck size={15} /> {busy ? 'Checking…' : 'Confirm e-mail'}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={resend} disabled={busy}>
              <RefreshCw size={12} /> Re-send the code
            </button>
          </form>

          <div style={styles.footer}>
            <span style={{ color: 'var(--text-muted)' }}>Wrong e-mail?</span>{' '}
            <button className="link-btn" style={styles.linkBtn} onClick={() => { setStage('form'); setError(null); setInfo(null) }}>
              Start over
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Stage: municipality awaiting admin approval ───────────────────────────
  if (stage === 'pending') {
    return (
      <div style={styles.page} className="page-grid-bg">
        <div className="card anim-fade-up" style={{ ...styles.card, textAlign: 'center' }}>
          <ShieldCheck size={34} style={{ color: 'var(--green)', marginBottom: 12 }} />
          <h1 className="display" style={styles.title}>Awaiting approval</h1>
          <div className="road-divider" style={{ width: 120, margin: '14px auto 18px' }} />
          <p style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.7 }}>
            {info || 'A platform administrator must approve municipality accounts. You will receive an e-mail once yours is reviewed.'}
          </p>
          <Link to="/login" className="btn btn-accent" style={{ marginTop: 18, padding: '10px 22px' }}>
            Back to sign in
          </Link>
        </div>
      </div>
    )
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

        {form.role === 'municipality' && (
          <div style={styles.notice}>
            <Landmark size={13} style={{ flexShrink: 0, color: 'var(--cyan)' }} />
            Municipality accounts confirm their e-mail AND are reviewed by a
            platform admin before activation.
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
              City <span style={{ color: 'var(--accent)' }}>*</span>
              <input className="input" value={form.city} onChange={set('city')}
                     placeholder="e.g. Cluj-Napoca" required minLength={2} maxLength={80}
                     title="The map opens on your city" />
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

          <HoneypotField value={honeypot} onChange={setHoneypot} />
          <CaptchaField enabled={captchaEnabled} onToken={setCaptchaToken} />

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
  card: { width: '100%', maxWidth: 440, padding: '30px 32px', margin: '40px 16px' },
  title: { fontSize: 26, fontWeight: 700, letterSpacing: '-0.02em' },
  label: { display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: 'var(--text-dim)' },
  error: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '9px 12px', marginBottom: 14, borderRadius: 8,
    background: 'rgba(255,93,93,0.1)', border: '1px solid rgba(255,93,93,0.35)',
    color: 'var(--red)', fontSize: 12,
  },
  notice: {
    display: 'flex', alignItems: 'flex-start', gap: 8,
    padding: '9px 12px', marginBottom: 14, borderRadius: 8,
    background: 'var(--accent-dim)', border: '1px solid var(--border-accent)',
    color: 'var(--text-dim)', fontSize: 12, lineHeight: 1.55,
  },
  linkBtn: {
    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
    color: 'var(--accent)', fontWeight: 600, fontSize: 12.5,
  },
  footer: { marginTop: 18, fontSize: 12.5, textAlign: 'center' },
}
