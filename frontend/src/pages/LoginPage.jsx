/**
 * frontend/src/pages/LoginPage.jsx — sign in with username/e-mail + password.
 *
 * Google sign-in appears only when the backend reports GOOGLE_CLIENT_ID is
 * configured (a free OAuth client id — no billing). Apple Sign-In is
 * deliberately absent: it requires the paid Apple Developer Program.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { LogIn, AlertTriangle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { fetchAuthConfig } from '../utils/api'

export default function LoginPage() {
  const { login, loginWithGoogle, isAuthed } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from || '/'

  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [googleClientId, setGoogleClientId] = useState('')
  const googleBtnRef = useRef(null)

  useEffect(() => {
    if (isAuthed) navigate(from, { replace: true })
  }, [isAuthed, navigate, from])

  useEffect(() => {
    fetchAuthConfig()
      .then(c => setGoogleClientId(c.google_enabled ? (c.google_client_id || '') : ''))
      .catch(() => {})
  }, [])

  // ── Real "Sign in with Google" button (Google Identity Services, free) ───
  const onGoogleCredential = useCallback(async (resp) => {
    setError(null)
    try {
      await loginWithGoogle(resp.credential)
      navigate(from, { replace: true })
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Google sign-in failed')
    }
  }, [loginWithGoogle, navigate, from])

  useEffect(() => {
    if (!googleClientId || !googleBtnRef.current) return
    const render = () => {
      if (!window.google?.accounts?.id || !googleBtnRef.current) return
      window.google.accounts.id.initialize({
        client_id: googleClientId,
        callback: onGoogleCredential,
      })
      window.google.accounts.id.renderButton(googleBtnRef.current, {
        theme: 'outline', size: 'large', width: 316, text: 'signin_with',
      })
    }
    if (window.google?.accounts?.id) { render(); return }
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.onload = render
    document.head.appendChild(script)
    // The GIS script is a harmless singleton — no cleanup needed.
  }, [googleClientId, onGoogleCredential])

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(identifier.trim(), password)
      navigate(from, { replace: true })
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Login failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div className="card anim-fade-up" style={styles.card}>
        <div className="overline" style={{ color: 'var(--accent)', marginBottom: 8 }}>
          CLUJ-NAPOCA · ROAD INTELLIGENCE
        </div>
        <h1 className="display" style={styles.title}>Sign in</h1>
        <div className="road-divider" style={{ width: 120, margin: '14px 0 22px' }} />

        {error && (
          <div style={styles.error}>
            <AlertTriangle size={13} style={{ flexShrink: 0 }} />
            {error}
          </div>
        )}

        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label style={styles.label}>
            Username or e-mail
            <input
              className="input" style={styles.input} autoFocus
              value={identifier} onChange={e => setIdentifier(e.target.value)}
              autoComplete="username" required minLength={3}
            />
          </label>
          <label style={styles.label}>
            Password
            <input
              className="input" style={styles.input} type="password"
              value={password} onChange={e => setPassword(e.target.value)}
              autoComplete="current-password" required
            />
          </label>
          <button className="btn btn-accent" style={{ padding: '11px 0', marginTop: 6 }} disabled={busy}>
            <LogIn size={15} /> {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        {googleClientId && (
          <>
            <div style={styles.divider}>
              <span style={styles.dividerLine} />
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>or</span>
              <span style={styles.dividerLine} />
            </div>
            <div ref={googleBtnRef} style={{ display: 'flex', justifyContent: 'center' }} />
          </>
        )}

        <div style={styles.footer}>
          <span style={{ color: 'var(--text-muted)' }}>New here?</span>{' '}
          <Link to="/register" style={{ color: 'var(--accent)', fontWeight: 600 }}>Create an account</Link>
        </div>

        <div style={styles.fineprint}>
          Sign in with Google is free to enable (set <span className="mono">GOOGLE_CLIENT_ID</span>).
          Apple Sign-In is not offered — it requires the paid Apple Developer Program,
          and this project runs at zero cost.
        </div>
      </div>
    </div>
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
  card: { width: 380, padding: '30px 32px', margin: '40px 16px' },
  title: { fontSize: 26, fontWeight: 700, letterSpacing: '-0.02em' },
  label: { display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: 'var(--text-dim)' },
  input: { width: '100%' },
  error: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '9px 12px', marginBottom: 14, borderRadius: 8,
    background: 'rgba(255,93,93,0.1)', border: '1px solid rgba(255,93,93,0.35)',
    color: 'var(--red)', fontSize: 12,
  },
  divider: {
    display: 'flex', alignItems: 'center', gap: 10,
    margin: '16px 0 12px',
  },
  dividerLine: { flex: 1, height: 1, background: 'var(--border)' },
  footer: { marginTop: 18, fontSize: 12.5, textAlign: 'center' },
  fineprint: {
    marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)',
    fontSize: 10.5, color: 'var(--text-muted)', lineHeight: 1.6,
  },
}
