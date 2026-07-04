/**
 * frontend/src/components/CityGate.jsx
 *
 * Every account must have a city — it is what the maps open on. Local
 * registrations collect it in the form; Google sign-ins (and legacy
 * accounts) land here on first login: a small blocking dialog that saves
 * the city to the profile (PATCH /auth/me) and then gets out of the way.
 */

import React, { useState } from 'react'
import { MapPin, ArrowRight, AlertTriangle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { updateMyProfile } from '../utils/api'

export default function CityGate() {
  const { setUser } = useAuth()
  const [city, setCity] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    if (!city.trim()) return
    setBusy(true)
    setError(null)
    try {
      const updated = await updateMyProfile({ city: city.trim() })
      setUser(updated)                       // unmounts the gate
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Could not save your city')
      setBusy(false)
    }
  }

  return (
    <div style={styles.backdrop}>
      <div className="card anim-fade-up" style={styles.card}>
        <MapPin size={26} style={{ color: 'var(--accent)', marginBottom: 10 }} />
        <h2 className="display" style={{ fontSize: 20, fontWeight: 700 }}>Which city are you in?</h2>
        <p style={{ fontSize: 12.5, color: 'var(--text-dim)', lineHeight: 1.6, margin: '10px 0 18px' }}>
          RIDS opens the map on your city and routes your reports to the right
          municipality. You can change it later from your profile menu.
        </p>

        {error && (
          <div style={styles.error}>
            <AlertTriangle size={13} style={{ flexShrink: 0 }} />{error}
          </div>
        )}

        <form onSubmit={submit} style={{ display: 'flex', gap: 8 }}>
          <input
            className="input"
            style={{ flex: 1 }}
            autoFocus
            required
            minLength={2}
            maxLength={80}
            placeholder="e.g. Cluj-Napoca"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
          <button className="btn btn-accent" disabled={busy || !city.trim()}>
            {busy ? 'Saving…' : <>Go <ArrowRight size={14} /></>}
          </button>
        </form>
      </div>
    </div>
  )
}

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    zIndex: 2000,
    background: 'rgba(5,7,11,0.7)',
    backdropFilter: 'blur(6px)',
    WebkitBackdropFilter: 'blur(6px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  card: {
    width: '100%',
    maxWidth: 400,
    padding: '26px 26px 24px',
    textAlign: 'left',
  },
  error: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 11px', marginBottom: 12, borderRadius: 8,
    background: 'rgba(255,93,93,0.1)', border: '1px solid rgba(255,93,93,0.35)',
    color: 'var(--red)', fontSize: 12,
  },
}
