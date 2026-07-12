/**
 * frontend/src/components/OnboardingTour.jsx
 *
 * First-run coach tour. A centered modal walks a new account through the
 * parts of RDDS it can actually use (citizens get five steps, operators get
 * the same five plus their three work tools). It shows once per browser:
 * finishing or skipping writes localStorage['rids_tour_done_v1'].
 *
 * Deliberately a centered modal, not a highlight-the-element tour — there is
 * no positioning maths to break on a phone.
 */

import React, { useState, useEffect, useMemo } from 'react'
import {
  Compass, Radio, ThumbsUp, Award, Bell, Inbox, ClipboardList, Gauge,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const TOUR_KEY = 'rids_tour_done_v1'

const CITIZEN_STEPS = [
  {
    icon: Compass,
    title: 'Welcome to RDDS',
    body: 'RDDS shows road damage around you and lets you report it in one tap.',
  },
  {
    icon: Radio,
    title: 'The live map',
    body: 'Open Live to see hazards near you. Tap the report button when you hit one.',
  },
  {
    icon: ThumbsUp,
    title: 'Confirm what others found',
    body: 'If another driver already reported it, confirm it. Two independent confirmations make it trusted.',
  },
  {
    icon: Award,
    title: 'Your impact',
    body: 'Points, badges and your city rank live on the My impact page.',
  },
  {
    icon: Bell,
    title: 'Notifications',
    body: 'We tell you here when your report is confirmed or fixed.',
  },
]

const OPERATOR_STEPS = [
  {
    icon: Inbox,
    title: 'Triage',
    body: 'Citizen reports land here. Make the good ones official.',
  },
  {
    icon: ClipboardList,
    title: 'Work orders',
    body: 'Group damage into a job, plan the crew route, mark it repaired.',
  },
  {
    icon: Gauge,
    title: 'Road quality',
    body: 'The Road Quality Index scores every 120 m of road.',
  },
]

/** Clear the "seen it" flag and reload, so the tour plays again. */
export function restartTour() {
  localStorage.removeItem(TOUR_KEY)
  window.location.reload()
}

export default function OnboardingTour() {
  const { isAuthed, isOperator } = useAuth()
  const [done, setDone] = useState(() => Boolean(localStorage.getItem(TOUR_KEY)))
  const [step, setStep] = useState(0)

  const steps = useMemo(
    () => (isOperator ? [...CITIZEN_STEPS, ...OPERATOR_STEPS] : CITIZEN_STEPS),
    [isOperator],
  )

  const visible = isAuthed && !done

  // Escape closes the tour the same way Skip does.
  useEffect(() => {
    if (!visible) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') {
        localStorage.setItem(TOUR_KEY, '1')
        setDone(true)
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [visible])

  if (!visible) return null

  const current = steps[Math.min(step, steps.length - 1)]
  const isLast = step >= steps.length - 1
  const Icon = current.icon

  const finish = () => {
    localStorage.setItem(TOUR_KEY, '1')
    setDone(true)
  }

  const next = () => {
    if (isLast) finish()
    else setStep(s => s + 1)
  }

  return (
    <div style={styles.overlay} className="anim-fade-in">
      <div className="card anim-fade-up" style={styles.card}>
        <div className="grad-line" style={{ marginBottom: 18 }} />

        <div style={styles.head}>
          <span style={styles.iconWrap}>
            <Icon size={19} style={{ color: 'var(--accent)' }} />
          </span>
          <span className="overline">{step + 1} of {steps.length}</span>
        </div>

        <h2 className="display" style={styles.title}>{current.title}</h2>
        <p style={styles.body}>{current.body}</p>

        <div style={styles.dots}>
          {steps.map((s, i) => (
            <span
              key={s.title}
              onClick={() => setStep(i)}
              style={{
                ...styles.dot,
                width: i === step ? 18 : 7,
                background: i === step ? 'var(--accent)' : 'var(--border-bright)',
              }}
            />
          ))}
        </div>

        <div style={styles.actions}>
          <button className="btn btn-ghost btn-sm" onClick={finish}>Skip</button>
          <button className="btn btn-accent btn-sm" onClick={next}>
            {isLast ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  )
}

const styles = {
  overlay: {
    position: 'fixed',
    inset: 0,
    zIndex: 2000,
    background: 'rgba(0, 0, 0, 0.55)',
    backdropFilter: 'blur(2px)',
    WebkitBackdropFilter: 'blur(2px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 18,
  },
  card: {
    width: '100%',
    maxWidth: 420,
    padding: '20px 24px 22px',
    boxShadow: 'var(--shadow-lg)',
  },
  head: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    marginBottom: 14,
  },
  iconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    flexShrink: 0,
  },
  title: {
    fontSize: 19,
    fontWeight: 700,
    letterSpacing: '-0.01em',
    marginBottom: 8,
  },
  body: {
    fontSize: 13,
    lineHeight: 1.6,
    color: 'var(--text-dim)',
  },
  dots: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    margin: '20px 0 18px',
  },
  dot: {
    height: 7,
    borderRadius: 999,
    cursor: 'pointer',
    transition: 'var(--transition)',
  },
  actions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },
}
