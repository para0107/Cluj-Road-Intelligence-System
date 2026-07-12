/**
 * frontend/src/components/CaptchaField.jsx
 *
 * The proof-of-work check on the sign-in, sign-up and contact forms.
 *
 * The browser solves a small puzzle before the form is accepted. A person
 * never notices it (it finishes while they are still reading the form), but it
 * makes a bot farm pay real processor time for every account it tries to
 * create. It is self-hosted: the challenge comes from our own backend, so no
 * captcha company is involved and nothing about the visitor is sent anywhere.
 *
 * Renders nothing at all unless the backend reports captcha_enabled, so with
 * the feature switched off (the default) the forms behave exactly as before.
 *
 * Usage:
 *   const [token, setToken] = useState(null)
 *   <CaptchaField enabled={config?.captcha_enabled} onToken={setToken} />
 */

import React, { useEffect, useRef, useState } from 'react'

// The widget reads these to theme itself; they map onto our design tokens so
// it does not look like a bolted-on third-party box.
const WIDGET_STYLE = {
  '--altcha-border-width': '1px',
  '--altcha-border-radius': 'var(--radius)',
  '--altcha-color-border': 'var(--border)',
  '--altcha-color-text': 'var(--text-dim)',
  '--altcha-color-base': 'var(--bg-card2)',
  '--altcha-color-border-focus': 'var(--accent)',
  '--altcha-max-width': '100%',
  width: '100%',
}

export default function CaptchaField({ enabled, onToken }) {
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const ref = useRef(null)

  // Load the custom element on demand. It is only ever fetched by visitors who
  // actually see a protected form.
  useEffect(() => {
    if (!enabled) return undefined
    let alive = true
    import('altcha')
      .then(() => { if (alive) setReady(true) })
      .catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [enabled])

  // React does not bind custom-element events, so wire it up by hand.
  useEffect(() => {
    if (!enabled || !ready) return undefined
    const el = ref.current
    if (!el) return undefined

    const onStateChange = (e) => {
      const { state, payload } = e.detail || {}
      onToken(state === 'verified' && payload ? payload : null)
    }

    el.addEventListener('statechange', onStateChange)
    return () => el.removeEventListener('statechange', onStateChange)
  }, [enabled, ready, onToken])

  if (!enabled) return null

  if (failed) {
    return (
      <div style={styles.failed}>
        The verification check could not load. Reload the page and try again.
      </div>
    )
  }

  if (!ready) {
    return <div className="skeleton" style={styles.skeleton} />
  }

  return (
    <altcha-widget
      ref={ref}
      challengeurl="/api/auth/captcha/challenge"
      auto="onload"
      hidelogo=""
      hidefooter=""
      style={WIDGET_STYLE}
    />
  )
}

/**
 * The honeypot. A real person never sees this field and never fills it, so
 * anything that arrives with it filled in is a script. Kept off-screen rather
 * than display:none, because some bots skip hidden inputs.
 */
export function HoneypotField({ value, onChange }) {
  return (
    <input
      type="text"
      name="website"
      tabIndex={-1}
      autoComplete="off"
      aria-hidden="true"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        position: 'absolute',
        left: '-9999px',
        width: 1,
        height: 1,
        opacity: 0,
        pointerEvents: 'none',
      }}
    />
  )
}

const styles = {
  skeleton: { height: 44, borderRadius: 'var(--radius)', width: '100%' },
  failed: {
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    border: '1px solid var(--orange)',
    background: 'color-mix(in srgb, var(--orange) 10%, transparent)',
    color: 'var(--orange)',
    fontSize: 12,
  },
}
