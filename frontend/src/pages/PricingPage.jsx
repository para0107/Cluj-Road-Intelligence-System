/**
 * frontend/src/pages/PricingPage.jsx  ·  route /pricing (public)
 *
 * The commercial storefront: three tiers, a compact comparison table and a
 * contact form. There is deliberately no payment processor in RDDS, so no
 * tier can be bought here. Paid tiers are "contact us" only and every plan
 * on this page is currently free.
 *
 * Anti-bot on the contact form: a hidden honeypot input ("website") plus the
 * ALTCHA proof-of-work widget, loaded lazily and only when the backend says
 * captcha is enabled (GET /auth/config).
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  Check, Minus, ArrowRight, Send, Building2, User, Code2,
  ShieldCheck, CheckCircle2, AlertTriangle, Mail,
} from 'lucide-react'
import { fetchAuthConfig, contactSales } from '../utils/api'
import { SectionTitle, Spinner } from '../components/ui'
import useMotionOk from '../hooks/useMotionOk'
import useIsMobile from '../hooks/useIsMobile'
import SpotlightCard from '../reactbits/SpotlightCard/SpotlightCard'
import Aurora from '../reactbits/Aurora/Aurora'
import SplitText from '../reactbits/SplitText/SplitText'
import ShinyText from '../reactbits/ShinyText/ShinyText'

/* SpotlightCard ships with a hard-coded dark shell (#111 / #222). Re-skin it
   with the design tokens so both themes work. Scoped to this page. */
const PAGE_CSS = `
.pricing-hero-title {
  font-size: clamp(32px, 5vw, 52px);
  font-weight: 700;
  line-height: 1.06;
  letter-spacing: -0.025em;
}
.pricing-card.card-spotlight {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 26px 24px 24px;
  width: 100%;
  flex: 1;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
  transition: var(--transition);
}
.pricing-card.card-spotlight:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
  border-color: var(--border-bright);
}
.pricing-card.pricing-card-featured.card-spotlight {
  border-color: var(--border-accent);
  box-shadow: var(--shadow), 0 0 40px rgba(234, 255, 61, 0.06);
}
.pricing-card.pricing-card-featured.card-spotlight:hover {
  border-color: var(--accent);
}
.pricing-honeypot {
  position: absolute;
  left: -9999px;
  top: 0;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}
`

const TIERS = [
  {
    key: 'citizen',
    icon: User,
    name: 'Citizen',
    price: 'Free',
    priceSub: 'Free forever. No card, no trial clock.',
    who: 'Drivers and residents',
    accent: 'var(--cyan)',
    features: [
      'Report a hazard in one tap',
      'Live hazard map with voice alerts while you drive',
      'Drive mode that reports bumps automatically',
      'Points, badges and the city leaderboard',
      'The RDDS assistant for questions about your street',
    ],
    cta: { label: 'Create a free account', to: '/register' },
  },
  {
    key: 'municipality',
    icon: Building2,
    name: 'Municipality',
    price: 'Free pilot',
    priceSub: 'Contact us for city-wide rollout.',
    who: 'City road departments',
    accent: 'var(--accent)',
    featured: true,
    features: [
      'Everything in Citizen',
      'Survey pipeline: upload dashcam video, get automatic detection and severity',
      'Triage inbox for citizen reports',
      'Work orders with crew route planning',
      'Repair verification that tells you when a repair failed and the damage came back',
      'Road Quality Index for every part of the city',
      'Operations analytics and printable monthly reports',
    ],
    cta: { label: 'Talk to us', scroll: true },
  },
  {
    key: 'data',
    icon: Code2,
    name: 'Data & API',
    price: 'Free in preview',
    priceSub: 'Keys are issued while the API is in preview.',
    who: 'Fleets, insurers, researchers, planners',
    accent: 'var(--purple)',
    features: [
      'Read-only developer API',
      'Road Quality Index by grid cell',
      'CSV and GeoJSON exports',
      '60 requests per minute per key',
    ],
    cta: { label: 'Read the API docs', to: '/developers' },
  },
]

const COMPARE = [
  { label: 'Report hazards from your phone', citizen: true, muni: true, data: false },
  { label: 'Live hazard map and voice alerts', citizen: true, muni: true, data: false },
  { label: 'Drive mode, automatic bump reports', citizen: true, muni: true, data: false },
  { label: 'Points, badges and leaderboard', citizen: true, muni: true, data: false },
  { label: 'Dashcam survey pipeline', citizen: false, muni: true, data: false },
  { label: 'Triage inbox', citizen: false, muni: true, data: false },
  { label: 'Work orders and crew routes', citizen: false, muni: true, data: false },
  { label: 'Repair verification', citizen: false, muni: true, data: false },
  { label: 'Road Quality Index', citizen: false, muni: true, data: true },
  { label: 'Operations analytics and reports', citizen: false, muni: true, data: false },
  { label: 'Developer API keys', citizen: false, muni: true, data: true },
  { label: 'CSV and GeoJSON export', citizen: false, muni: true, data: true },
]

export default function PricingPage() {
  const motionOk = useMotionOk()
  const isMobile = useIsMobile()
  const contactRef = useRef(null)

  const scrollToContact = useCallback(() => {
    contactRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  return (
    <div className="page-grid-bg" style={styles.page}>
      <style>{PAGE_CSS}</style>

      {/* Aurora glow behind the hero only. Desktop, motion-friendly visitors. */}
      {motionOk && (
        <div style={styles.auroraLayer} aria-hidden="true">
          <Aurora colorStops={['#eaff3d', '#4cc9f0', '#3ddc84']} amplitude={0.8} blend={0.4} />
        </div>
      )}

      <div style={styles.content}>

        {/* ── Hero ──────────────────────────────────────────────────────── */}
        <section style={styles.hero}>
          <div className="overline anim-fade-up" style={{ color: 'var(--accent)', marginBottom: 14 }}>
            Pricing
          </div>

          {motionOk ? (
            <SplitText
              text="Plans that fit a city budget"
              tag="h1"
              className="display pricing-hero-title"
              splitType="words"
              delay={38}
              duration={0.8}
              textAlign="left"
            />
          ) : (
            <h1 className="display anim-fade-up delay-1" style={styles.heroTitle}>
              Plans that fit a city budget
            </h1>
          )}

          <div className="road-divider anim-fade-up delay-2" style={{ width: 180, margin: '22px 0' }} />

          <p className="anim-fade-up delay-2" style={styles.heroSub}>
            RDDS turns dashcam video and citizen reports into a ranked repair plan. The citizen
            app is free forever, and cities pay for the operations tools.
          </p>
        </section>

        {/* ── Tiers ─────────────────────────────────────────────────────── */}
        <section style={styles.tierGrid}>
          {TIERS.map((tier, i) => (
            <TierCard
              key={tier.key}
              tier={tier}
              delay={`delay-${i + 1}`}
              onTalk={scrollToContact}
            />
          ))}
        </section>

        {/* ── Comparison ────────────────────────────────────────────────── */}
        <section style={{ marginTop: 52 }}>
          <SectionTitle overline="Side by side" title="What each plan includes" />
          <div className="card" style={{ padding: isMobile ? 12 : 18, overflowX: 'auto' }}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={{ ...styles.th, textAlign: 'left', minWidth: 220 }}>Capability</th>
                  <th style={styles.th}>Citizen</th>
                  <th style={{ ...styles.th, color: 'var(--accent)' }}>Municipality</th>
                  <th style={styles.th}>Data &amp; API</th>
                </tr>
              </thead>
              <tbody>
                {COMPARE.map(row => (
                  <tr key={row.label} className="table-row-hover">
                    <td style={{ ...styles.td, textAlign: 'left', color: 'var(--text-dim)' }}>{row.label}</td>
                    <td style={styles.td}><Mark on={row.citizen} /></td>
                    <td style={styles.td}><Mark on={row.muni} /></td>
                    <td style={styles.td}><Mark on={row.data} /></td>
                  </tr>
                ))}
                <tr>
                  <td style={{ ...styles.td, textAlign: 'left', color: 'var(--text-muted)' }}>Price today</td>
                  <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-dim)' }}>Free</td>
                  <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--accent)' }}>Free pilot</td>
                  <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-dim)' }}>Free preview</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Contact ───────────────────────────────────────────────────── */}
        <section id="contact" ref={contactRef} style={{ marginTop: 56, scrollMarginTop: 'calc(var(--nav-h) + 20px)' }}>
          <SectionTitle overline="Talk to us" title="Start a pilot in your city" />
          <ContactForm />
        </section>

        {/* ── Footer note ───────────────────────────────────────────────── */}
        <footer style={{ marginTop: 44 }}>
          <div className="road-divider" style={{ width: '100%', marginBottom: 18, opacity: 0.3 }} />
          <p style={{ fontSize: 11.5, color: 'var(--text-muted)', textAlign: 'center' }}>
            No payment is taken on this site. Pilots are agreed by email.
          </p>
        </footer>
      </div>
    </div>
  )
}

/* ── Tier card ─────────────────────────────────────────────────────────── */

function TierCard({ tier, delay, onTalk }) {
  const Icon = tier.icon
  const featured = !!tier.featured

  const body = (
    <SpotlightCard
      className={`pricing-card ${featured ? 'pricing-card-featured' : ''}`}
      spotlightColor={featured ? 'rgba(234, 255, 61, 0.16)' : 'rgba(255, 255, 255, 0.10)'}
    >
      {featured && (
        <span style={styles.popularPill}>Most popular</span>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <span style={{
          ...styles.tierIcon,
          background: `color-mix(in srgb, ${tier.accent} 14%, transparent)`,
          border: `1px solid color-mix(in srgb, ${tier.accent} 40%, transparent)`,
          color: tier.accent,
        }}>
          <Icon size={17} />
        </span>
        {featured ? (
          <span className="display" style={{ fontSize: 17, fontWeight: 700 }}>
            <ShinyText text={tier.name} speed={3} color="var(--text)" shineColor="#eaff3d" />
          </span>
        ) : (
          <span className="display" style={{ fontSize: 17, fontWeight: 700 }}>{tier.name}</span>
        )}
      </div>

      <div className="display" style={{ fontSize: 30, fontWeight: 700, lineHeight: 1.15, letterSpacing: '-0.02em' }}>
        {tier.price}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, minHeight: 34 }}>
        {tier.priceSub}
      </div>

      <div style={styles.tierWho}>
        <span className="overline">For</span>
        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{tier.who}</span>
      </div>

      <ul style={styles.featureList}>
        {tier.features.map(f => (
          <li key={f} style={styles.featureItem}>
            <Check size={14} style={{ color: 'var(--green)', flexShrink: 0, marginTop: 2 }} />
            <span>{f}</span>
          </li>
        ))}
      </ul>

      <div style={{ marginTop: 'auto', paddingTop: 20 }}>
        {tier.cta.scroll ? (
          <button
            type="button"
            onClick={onTalk}
            className="btn btn-accent"
            style={{ width: '100%', padding: '11px 18px', fontSize: 13 }}
          >
            <Send size={14} /> {tier.cta.label}
          </button>
        ) : (
          <Link
            to={tier.cta.to}
            className={featured ? 'btn btn-accent' : 'btn'}
            style={{ width: '100%', padding: '11px 18px', fontSize: 13 }}
          >
            {tier.cta.label} <ArrowRight size={14} />
          </Link>
        )}
      </div>
    </SpotlightCard>
  )

  return <div className={`anim-fade-up ${delay}`} style={{ display: 'flex' }}>{body}</div>
}

function Mark({ on }) {
  return on
    ? <Check size={15} style={{ color: 'var(--green)' }} />
    : <Minus size={15} style={{ color: 'var(--text-muted)', opacity: 0.6 }} />
}

/* ── Contact form ──────────────────────────────────────────────────────── */

function ContactForm() {
  const [form, setForm] = useState({ name: '', email: '', organization: '', message: '', website: '' })
  const [captchaEnabled, setCaptchaEnabled] = useState(false)
  const [widgetReady, setWidgetReady] = useState(false)
  const [altchaPayload, setAltchaPayload] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [sentTo, setSentTo] = useState(null)
  const widgetRef = useRef(null)

  // Ask the backend whether proof-of-work is on, then load the widget lazily.
  useEffect(() => {
    let alive = true
    fetchAuthConfig()
      .then(async (cfg) => {
        if (!alive || !cfg?.captcha_enabled) return
        setCaptchaEnabled(true)
        try {
          await import('altcha')
          if (alive) setWidgetReady(true)
        } catch {
          // Widget could not load. The form still submits and the server
          // decides what to do with a missing proof.
        }
      })
      .catch(() => {})
    return () => { alive = false }
  }, [])

  // The widget is a custom element, so its events need a real listener.
  useEffect(() => {
    if (!widgetReady || sentTo) return undefined
    const el = widgetRef.current
    if (!el) return undefined
    const onStateChange = (e) => {
      const detail = e.detail || {}
      setAltchaPayload(detail.payload || '')
    }
    el.addEventListener('statechange', onStateChange)
    return () => el.removeEventListener('statechange', onStateChange)
  }, [widgetReady, sentTo])

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setError(null)

    const name = form.name.trim()
    const email = form.email.trim()
    const message = form.message.trim()

    if (name.length < 2) return setError('Please tell us your name.')
    if (!email.includes('@')) return setError('Please enter an email we can reply to.')
    if (message.length < 10) return setError('Please add a few words about what you need.')

    setBusy(true)
    try {
      await contactSales({
        name,
        email,
        organization: form.organization.trim(),
        message,
        website: form.website,          // honeypot, empty for humans
        altcha: altchaPayload || null,
      })
      setSentTo(email)
    } catch (err) {
      const status = err?.response?.status
      if (status === 429) {
        setError('You have sent several messages already. Try again later.')
      } else {
        const detail = err?.response?.data?.detail
        setError(typeof detail === 'string' ? detail : 'We could not send that. Please try again.')
      }
    } finally {
      setBusy(false)
    }
  }

  if (sentTo) {
    return (
      <div className="card anim-fade-up" style={{ ...styles.contactCard, textAlign: 'center' }}>
        <div style={styles.sentIcon}><CheckCircle2 size={22} /></div>
        <div className="display" style={{ fontSize: 17, fontWeight: 700, marginBottom: 6 }}>
          Thanks. We will reply to {sentTo} shortly.
        </div>
        <p style={{ fontSize: 12.5, color: 'var(--text-muted)', maxWidth: 420, margin: '0 auto' }}>
          Nothing was charged and no card was asked for. A person reads every message and
          answers by email.
        </p>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="card" style={styles.contactCard} noValidate>
      <div style={styles.formGrid}>
        <label style={styles.field}>
          <span className="overline">Your name</span>
          <input
            className="input" value={form.name} onChange={set('name')}
            placeholder="Ana Popescu" autoComplete="name" required
          />
        </label>

        <label style={styles.field}>
          <span className="overline">Email</span>
          <input
            className="input" type="email" value={form.email} onChange={set('email')}
            placeholder="ana@primaria.ro" autoComplete="email" required
          />
        </label>

        <label style={{ ...styles.field, gridColumn: '1 / -1' }}>
          <span className="overline">Organization</span>
          <input
            className="input" value={form.organization} onChange={set('organization')}
            placeholder="City hall, fleet, university, company" autoComplete="organization"
          />
        </label>

        <label style={{ ...styles.field, gridColumn: '1 / -1' }}>
          <span className="overline">How can we help</span>
          <textarea
            className="input" value={form.message} onChange={set('message')}
            placeholder="Tell us about your city, how many kilometres of road you look after, and what you want to try first."
            rows={5}
            style={{ resize: 'vertical', minHeight: 110, fontFamily: 'var(--font-sans)', lineHeight: 1.6 }}
            required
          />
        </label>
      </div>

      {/* Honeypot. Bots fill it, people never see it. */}
      <input
        className="pricing-honeypot"
        type="text"
        name="website"
        value={form.website}
        onChange={set('website')}
        tabIndex={-1}
        autoComplete="off"
        aria-hidden="true"
      />

      {captchaEnabled && widgetReady && (
        <div style={{ marginTop: 14 }}>
          <altcha-widget
            ref={widgetRef}
            challengeurl="/api/auth/captcha/challenge"
            auto="onload"
            hidelogo="true"
            hidefooter="true"
          />
        </div>
      )}

      {error && (
        <div style={styles.errorBanner}>
          <AlertTriangle size={14} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      <div style={styles.formFooter}>
        <span style={styles.privacyNote}>
          <ShieldCheck size={13} style={{ color: 'var(--green)', flexShrink: 0 }} />
          We only use your address to answer you.
        </span>

        <button type="submit" className="btn btn-accent" disabled={busy} style={{ padding: '11px 20px', fontSize: 13 }}>
          {busy ? <Spinner size={14} /> : <Mail size={14} />}
          {busy ? 'Sending' : 'Send message'}
        </button>
      </div>
    </form>
  )
}

/* ── Styles ────────────────────────────────────────────────────────────── */

const styles = {
  page: {
    position: 'relative',
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 28px)',
    paddingBottom: 60,
    paddingLeft: 20,
    paddingRight: 20,
    maxWidth: 1160,
    margin: '0 auto',
  },
  auroraLayer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 420,
    zIndex: 0,
    opacity: 0.35,
    pointerEvents: 'none',
  },
  content: {
    position: 'relative',
    zIndex: 1,
  },
  hero: {
    padding: '26px 0 40px',
    maxWidth: 680,
  },
  heroTitle: {
    fontSize: 'clamp(32px, 5vw, 52px)',
    fontWeight: 700,
    lineHeight: 1.06,
    letterSpacing: '-0.025em',
  },
  heroSub: {
    fontSize: 14.5,
    color: 'var(--text-dim)',
    lineHeight: 1.75,
    maxWidth: 560,
  },
  tierGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 300px), 1fr))',
    gap: 16,
    alignItems: 'stretch',
  },
  popularPill: {
    position: 'absolute',
    top: 14,
    right: 14,
    padding: '3px 10px',
    borderRadius: 999,
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    color: 'var(--accent)',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  tierIcon: {
    width: 34,
    height: 34,
    borderRadius: 10,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  tierWho: {
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
    padding: '12px 0 4px',
    marginTop: 12,
    borderTop: '1px solid var(--border)',
  },
  featureList: {
    listStyle: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: 9,
    marginTop: 16,
  },
  featureItem: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 9,
    fontSize: 12.5,
    color: 'var(--text-dim)',
    lineHeight: 1.55,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    minWidth: 520,
  },
  th: {
    textAlign: 'center',
    padding: '10px 12px',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-bright)',
    whiteSpace: 'nowrap',
  },
  td: {
    textAlign: 'center',
    padding: '10px 12px',
    fontSize: 12.5,
    borderBottom: '1px solid var(--border)',
    verticalAlign: 'middle',
  },
  contactCard: {
    padding: 24,
    maxWidth: 720,
  },
  formGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))',
    gap: 14,
  },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginTop: 14,
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255, 93, 93, 0.10)',
    border: '1px solid rgba(255, 93, 93, 0.35)',
    color: 'var(--red)',
    fontSize: 12.5,
  },
  formFooter: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 14,
    flexWrap: 'wrap',
    marginTop: 18,
  },
  privacyNote: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    fontSize: 11.5,
    color: 'var(--text-muted)',
  },
  sentIcon: {
    width: 46,
    height: 46,
    borderRadius: 13,
    margin: '0 auto 14px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(61, 220, 132, 0.12)',
    border: '1px solid rgba(61, 220, 132, 0.4)',
    color: 'var(--green)',
  },
}
