/**
 * frontend/src/pages/DevelopersPage.jsx — route /developers.
 *
 * Public developer surface for the read-only RDDS API: quick start, the three
 * public endpoints, the limits, and (only when signed in) API key management.
 * The plaintext key is shown once, right after creation, and never again.
 */

import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Code2, Terminal, KeyRound, Copy, Check, Trash2, Plus, Gauge,
  ShieldCheck, Database, Map as MapIcon, BarChart3, AlertTriangle, LogIn,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { fetchApiKeys, createApiKey, revokeApiKey } from '../utils/api'
import { fmtDate, fmtNum } from '../utils/format'
import { RQI_BANDS } from '../utils/constants'
import { SectionTitle, Spinner, CenterState, EmptyState } from '../components/ui'
import SpotlightCard from '../reactbits/SpotlightCard/SpotlightCard'
import FadeContent from '../reactbits/FadeContent/FadeContent'
import AnimatedContent from '../reactbits/AnimatedContent/AnimatedContent'
import useIsMobile from '../hooks/useIsMobile'

const MAX_ACTIVE_KEYS = 3

const CURL_EXAMPLE = `curl -H "X-API-Key: rdds_your_key_here" \\
  "https://your-rdds-host/api/v1/public/detections?bbox=23.5,46.7,23.7,46.8&page_size=50"`

const ENDPOINTS = [
  {
    key: 'detections',
    icon: Database,
    method: 'GET',
    path: '/api/v1/public/detections',
    desc: 'Road damage records, one row per de-duplicated damage site.',
    params: [
      ['bbox', 'string', 'minLon,minLat,maxLon,maxLat. Limits results to a map area.'],
      ['damage_type', 'string', 'One damage class, for example pothole.'],
      ['severity_min', 'int', 'Only return sites at this severity or worse (1 to 5).'],
      ['page', 'int', 'Page number, starts at 1.'],
      ['page_size', 'int', 'Rows per page, up to 100.'],
    ],
    response: `{
  "total": 1284,
  "page": 1,
  "page_size": 50,
  "items": [
    {
      "id": 913,
      "damage_type": "pothole",
      "severity": 4,
      "latitude": 46.7712,
      "longitude": 23.6009,
      "first_detected": "2026-03-04T08:12:00Z",
      "last_detected": "2026-06-28T17:40:00Z",
      "detection_count": 6,
      "is_fixed": false,
      "priority_score": 0.778
    }
  ]
}`,
  },
  {
    key: 'road-quality',
    icon: MapIcon,
    method: 'GET',
    path: '/api/v1/public/road-quality',
    desc: 'The Road Quality Index on a square grid, ready to paint on a map.',
    params: [
      ['bbox', 'string', 'minLon,minLat,maxLon,maxLat. The area to grade.'],
      ['cell_m', 'int', 'Grid cell size in metres, 40 to 1000. Default 120.'],
    ],
    response: `{
  "cell_m": 120,
  "count": 412,
  "cells": [
    {
      "lat": 46.7701,
      "lon": 23.5912,
      "score": 62.4,
      "band": "C",
      "n": 7,
      "worst": 4
    }
  ]
}`,
    bands: true,
  },
  {
    key: 'stats',
    icon: BarChart3,
    method: 'GET',
    path: '/api/v1/public/stats',
    desc: 'City-wide totals. No parameters, good for a dashboard tile.',
    params: [],
    response: `{
  "total_detections": 1284,
  "open_detections": 1043,
  "fixed_detections": 241,
  "last_survey_date": "2026-06-28",
  "by_damage_type": { "pothole": 512, "alligator_crack": 208 },
  "by_severity": { "1": 190, "2": 355, "3": 402, "4": 271, "5": 66 }
}`,
  },
]

// ── Small local pieces ────────────────────────────────────────────────────

function CopyButton({ text, small = false }) {
  const [done, setDone] = useState(false)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      return
    }
    if (!alive.current) return
    setDone(true)
    setTimeout(() => { if (alive.current) setDone(false) }, 1600)
  }

  return (
    <button className={`btn btn-sm ${small ? 'btn-ghost' : ''}`} onClick={copy} type="button">
      {done ? <Check size={12} /> : <Copy size={12} />}
      {done ? 'Copied' : 'Copy'}
    </button>
  )
}

function CodeBlock({ children }) {
  return <pre className="mono" style={styles.code}>{children}</pre>
}

function ErrorBanner({ children }) {
  if (!children) return null
  return (
    <div style={styles.errorBanner}>
      <AlertTriangle size={14} style={{ flexShrink: 0 }} />
      <span>{children}</span>
    </div>
  )
}

function ParamTable({ params }) {
  if (params.length === 0) {
    return (
      <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 0' }}>
        This endpoint takes no parameters.
      </div>
    )
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={styles.table}>
        <thead>
          <tr>
            <th className="overline" style={styles.th}>Name</th>
            <th className="overline" style={styles.th}>Type</th>
            <th className="overline" style={styles.th}>What it does</th>
          </tr>
        </thead>
        <tbody>
          {params.map(([name, type, desc]) => (
            <tr key={name} className="table-row-hover">
              <td className="mono" style={{ ...styles.td, color: 'var(--accent)', whiteSpace: 'nowrap' }}>{name}</td>
              <td className="mono" style={{ ...styles.td, color: 'var(--text-muted)' }}>{type}</td>
              <td style={{ ...styles.td, color: 'var(--text-dim)' }}>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function BandLegend() {
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
        Every cell gets a score from 0 to 100 and a band from A to E. A means the road is in good
        shape, E means it is badly broken up.
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
        {Object.entries(RQI_BANDS).map(([band, info]) => (
          <span
            key={band}
            className="chip"
            style={{
              cursor: 'default',
              borderColor: `${info.color}55`,
              background: `${info.color}14`,
              color: info.color,
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: info.color }} />
            <span className="mono" style={{ fontWeight: 700 }}>{band}</span>
            {info.label}
            <span className="mono" style={{ fontSize: 10, opacity: 0.75 }}>{info.min}+</span>
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function DevelopersPage() {
  const { isAuthed, booting } = useAuth()
  const isMobile = useIsMobile()

  const [keys, setKeys] = useState([])
  const [loadingKeys, setLoadingKeys] = useState(false)
  const [keysError, setKeysError] = useState('')
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState('')
  const [newKey, setNewKey] = useState(null)
  const alive = useRef(true)

  const load = async () => {
    setLoadingKeys(true)
    try {
      const res = await fetchApiKeys()
      if (!alive.current) return
      setKeys(res?.items || [])
      setKeysError('')
    } catch {
      if (alive.current) setKeysError('Could not load your keys. Try again in a moment.')
    } finally {
      if (alive.current) setLoadingKeys(false)
    }
  }

  useEffect(() => {
    alive.current = true
    if (isAuthed) load()
    else {
      setKeys([])
      setNewKey(null)
    }
    return () => { alive.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthed])

  const activeCount = keys.filter(k => k.is_active).length
  const atLimit = activeCount >= MAX_ACTIVE_KEYS
  const canCreate = !creating && !atLimit && name.trim().length > 0

  const onCreate = async (e) => {
    e.preventDefault()
    if (!canCreate) return
    setCreating(true)
    setCreateError('')
    try {
      const res = await createApiKey(name.trim())
      if (!alive.current) return
      setNewKey(res)
      setName('')
      await load()
    } catch (err) {
      if (!alive.current) return
      const detail = err?.response?.data?.detail
      setCreateError(typeof detail === 'string' ? detail : 'Could not create the key. Try again.')
    } finally {
      if (alive.current) setCreating(false)
    }
  }

  const onRevoke = async (k) => {
    const ok = window.confirm(
      `Revoke the key "${k.name}"? Anything using it stops working straight away.`
    )
    if (!ok) return
    try {
      await revokeApiKey(k.id)
      if (!alive.current) return
      if (newKey && newKey.id === k.id) setNewKey(null)
      await load()
    } catch {
      if (alive.current) setKeysError('Could not revoke that key. Try again in a moment.')
    }
  }

  return (
    <div className="page-grid-bg" style={styles.page}>
      <style>{`
        .dev-spot.card-spotlight {
          background-color: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 20px 22px;
          box-shadow: var(--shadow);
          transition: var(--transition);
        }
        .dev-spot.card-spotlight:hover { border-color: var(--border-bright); }
      `}</style>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <FadeContent duration={600} blur>
        <div style={{ marginBottom: 34 }}>
          <div className="overline" style={{ color: 'var(--accent)', marginBottom: 10 }}>
            Developer API
          </div>
          <h1 className="display" style={styles.h1}>Build on RDDS road data</h1>
          <p style={styles.lede}>
            The RDDS API serves road damage records and the Road Quality Index as JSON. It is read
            only, so you can pull the data into a map, a dashboard, or a report without touching
            anything in the system. Every call needs an API key sent in a header. The API is free
            while it is in preview.
          </p>
          <div className="road-divider" style={{ marginTop: 22, maxWidth: 180 }} />
        </div>
      </FadeContent>

      {/* ── Quick start ───────────────────────────────────────────────────── */}
      <AnimatedContent distance={40} duration={0.6} threshold={0.05}>
        <div className="card" style={styles.section}>
          <SectionTitle
            overline="Quick start"
            title="Three steps to your first response"
            right={<Terminal size={16} style={{ color: 'var(--text-muted)' }} />}
          />

          <div style={{ display: 'grid', gap: 10, marginBottom: 18 }}>
            {[
              'Create a key below.',
              'Send it as the X-API-Key header on every request.',
              'Call an endpoint and read the JSON.',
            ].map((step, i) => (
              <div key={step} style={styles.step}>
                <span className="mono" style={styles.stepNum}>{i + 1}</span>
                <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>{step}</span>
              </div>
            ))}
          </div>

          <div style={styles.codeHeader}>
            <span className="overline">Example request</span>
            <CopyButton text={CURL_EXAMPLE} small />
          </div>
          <CodeBlock>{CURL_EXAMPLE}</CodeBlock>
        </div>
      </AnimatedContent>

      {/* ── Endpoints ─────────────────────────────────────────────────────── */}
      <div style={{ marginTop: 34 }}>
        <SectionTitle
          overline="Reference"
          title="Endpoints"
          right={<Code2 size={16} style={{ color: 'var(--text-muted)' }} />}
        />

        <div style={{ display: 'grid', gap: 16 }}>
          {ENDPOINTS.map((ep, i) => {
            const Icon = ep.icon
            return (
              <AnimatedContent key={ep.key} distance={40} duration={0.6} delay={i * 0.06} threshold={0.05}>
                <SpotlightCard className="dev-spot" spotlightColor="rgba(234, 255, 61, 0.12)">
                  <div style={styles.epHead}>
                    <div style={styles.epIcon}>
                      <Icon size={15} style={{ color: 'var(--accent)' }} />
                    </div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span className="mono" style={styles.method}>{ep.method}</span>
                        <span
                          className="mono"
                          style={{
                            fontSize: isMobile ? 11.5 : 13,
                            fontWeight: 700,
                            color: 'var(--text)',
                            wordBreak: 'break-all',
                          }}
                        >
                          {ep.path}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                        {ep.desc}
                      </div>
                    </div>
                  </div>

                  <div className="overline" style={styles.subLabel}>Parameters</div>
                  <ParamTable params={ep.params} />

                  {ep.bands && <BandLegend />}

                  <div className="overline" style={{ ...styles.subLabel, marginTop: 16 }}>Response</div>
                  <CodeBlock>{ep.response}</CodeBlock>
                </SpotlightCard>
              </AnimatedContent>
            )
          })}
        </div>
      </div>

      {/* ── Limits ────────────────────────────────────────────────────────── */}
      <AnimatedContent distance={40} duration={0.6} threshold={0.05}>
        <div className="card" style={{ ...styles.section, marginTop: 34 }}>
          <SectionTitle
            overline="Fair use"
            title="Limits and what a key can see"
            right={<Gauge size={16} style={{ color: 'var(--text-muted)' }} />}
          />
          <div style={{
            display: 'grid',
            gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)',
            gap: 12,
          }}>
            {[
              {
                icon: Gauge,
                title: '60 requests a minute',
                body: 'That is the default rate for each key. Ask us if you need more.',
              },
              {
                icon: AlertTriangle,
                title: 'Slow down on a 429',
                body: 'Going over the rate returns HTTP 429 with a Retry-After header. Wait that many seconds, then retry.',
              },
              {
                icon: ShieldCheck,
                title: 'Read only, no personal data',
                body: 'Keys can only read. They never expose who reported anything, and they never return photos.',
              },
              {
                icon: KeyRound,
                title: 'Up to 3 active keys',
                body: 'Each account can hold three active keys. Revoke one to make room for a new one.',
              },
            ].map(({ icon: Icon, title, body }) => (
              <div key={title} style={styles.limitCard}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <Icon size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                  <span className="display" style={{ fontSize: 13, fontWeight: 700 }}>{title}</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>{body}</div>
              </div>
            ))}
          </div>
        </div>
      </AnimatedContent>

      {/* ── Your keys ─────────────────────────────────────────────────────── */}
      <div style={{ marginTop: 34 }}>
        <SectionTitle
          overline="Access"
          title="Your keys"
          right={<KeyRound size={16} style={{ color: 'var(--text-muted)' }} />}
        />

        {booting ? (
          <div className="card" style={styles.section}>
            <CenterState><Spinner label="Checking your session…" /></CenterState>
          </div>
        ) : !isAuthed ? (
          <div className="card" style={{ ...styles.section, textAlign: 'center' }}>
            <div style={styles.lockIcon}>
              <KeyRound size={22} style={{ color: 'var(--accent)' }} />
            </div>
            <div className="display" style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>
              Sign in to create an API key
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 16 }}>
              Keys belong to an account. Sign in and you can make one in a few seconds.
            </div>
            <Link to="/login" className="btn btn-accent">
              <LogIn size={13} /> Sign in
            </Link>
          </div>
        ) : (
          <div className="card" style={styles.section}>
            {/* Create form */}
            <form onSubmit={onCreate} style={styles.form}>
              <input
                className="input"
                style={{ flex: 1, minWidth: isMobile ? '100%' : 220 }}
                placeholder="Name this key, for example City dashboard"
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={60}
                disabled={atLimit}
              />
              <button type="submit" className="btn btn-accent" disabled={!canCreate}>
                <Plus size={13} /> {creating ? 'Creating…' : 'Create key'}
              </button>
            </form>

            {atLimit && (
              <div style={styles.hint}>
                You already have {MAX_ACTIVE_KEYS} active keys, which is the limit. Revoke one below
                to create another.
              </div>
            )}

            <ErrorBanner>{createError}</ErrorBanner>

            {/* One-time reveal */}
            {newKey?.key && (
              <div className="anim-fade-up" style={styles.reveal}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <KeyRound size={14} style={{ color: 'var(--accent)' }} />
                  <span className="display" style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)' }}>
                    {newKey.name} is ready
                  </span>
                </div>
                <div style={styles.revealRow}>
                  <code className="mono" style={styles.revealKey}>{newKey.key}</code>
                  <CopyButton text={newKey.key} />
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 10 }}>
                  This is the only time the key is shown. Copy it now and store it somewhere safe.
                  If you lose it, revoke the key and create a new one.
                </div>
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  style={{ marginTop: 12 }}
                  onClick={() => setNewKey(null)}
                >
                  I have copied it
                </button>
              </div>
            )}

            <ErrorBanner>{keysError}</ErrorBanner>

            {/* Key list */}
            <div style={{ marginTop: 18 }}>
              {loadingKeys ? (
                <CenterState><Spinner label="Loading your keys…" /></CenterState>
              ) : keys.length === 0 ? (
                <EmptyState
                  icon={KeyRound}
                  title="No keys yet"
                  sub="Create your first key above and it shows up here."
                />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {keys.map((k) => (
                    <div key={k.id} className="card table-row-hover" style={styles.keyRow}>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                          <span className="display" style={{ fontSize: 13, fontWeight: 700 }}>
                            {k.name}
                          </span>
                          <span
                            className="chip"
                            style={{
                              cursor: 'default',
                              borderColor: k.is_active ? 'rgba(61, 220, 132, 0.45)' : 'var(--border)',
                              background: k.is_active ? 'rgba(61, 220, 132, 0.10)' : 'transparent',
                              color: k.is_active ? 'var(--green)' : 'var(--text-muted)',
                            }}
                          >
                            {k.is_active ? 'Active' : 'Revoked'}
                          </span>
                        </div>
                        <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-dim)', marginTop: 4 }}>
                          {k.prefix}…
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                          Created {fmtDate(k.created_at)} · {fmtNum(k.usage_count)} calls · last used{' '}
                          {k.last_used_at ? fmtDate(k.last_used_at) : 'never'}
                        </div>
                      </div>
                      {k.is_active && (
                        <button
                          type="button"
                          className="btn btn-sm btn-danger"
                          onClick={() => onRevoke(k)}
                          style={{ flexShrink: 0 }}
                        >
                          <Trash2 size={12} /> Revoke
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
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
    paddingTop: 'calc(var(--nav-h) + 28px)',
    paddingBottom: 60,
    paddingLeft: 20,
    paddingRight: 20,
    maxWidth: 1000,
    margin: '0 auto',
  },
  h1: {
    fontSize: 'clamp(26px, 5vw, 40px)',
    fontWeight: 700,
    letterSpacing: '-0.02em',
    lineHeight: 1.12,
    marginBottom: 14,
  },
  lede: {
    fontSize: 14,
    color: 'var(--text-dim)',
    lineHeight: 1.7,
    maxWidth: 700,
  },
  section: {
    padding: '22px 24px',
  },
  step: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  stepNum: {
    width: 24,
    height: 24,
    borderRadius: 7,
    flexShrink: 0,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 11,
    fontWeight: 700,
    color: 'var(--accent)',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
  },
  codeHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    marginBottom: 7,
  },
  code: {
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '13px 15px',
    overflowX: 'auto',
    fontSize: 12,
    lineHeight: 1.65,
    color: 'var(--text-dim)',
    whiteSpace: 'pre',
  },
  epHead: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
    marginBottom: 16,
  },
  epIcon: {
    width: 30,
    height: 30,
    borderRadius: 8,
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
  },
  method: {
    fontSize: 10.5,
    fontWeight: 700,
    letterSpacing: '0.06em',
    padding: '2px 7px',
    borderRadius: 5,
    color: 'var(--cyan)',
    background: 'var(--cyan-dim)',
    border: '1px solid rgba(76, 201, 240, 0.35)',
  },
  subLabel: {
    color: 'var(--text-muted)',
    marginBottom: 8,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    minWidth: 420,
  },
  th: {
    textAlign: 'left',
    padding: '7px 10px 7px 0',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-muted)',
  },
  td: {
    padding: '8px 10px 8px 0',
    borderBottom: '1px solid var(--border)',
    fontSize: 12,
    verticalAlign: 'top',
  },
  limitCard: {
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '13px 15px',
  },
  form: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  hint: {
    fontSize: 12,
    color: 'var(--text-muted)',
    marginTop: 10,
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginTop: 12,
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    border: '1px solid rgba(255, 93, 93, 0.4)',
    background: 'rgba(255, 93, 93, 0.08)',
    color: 'var(--red)',
    fontSize: 12,
  },
  reveal: {
    marginTop: 16,
    padding: '15px 16px',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border-accent)',
    background: 'var(--accent-dim)',
  },
  revealRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
  },
  revealKey: {
    flex: 1,
    minWidth: 200,
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '9px 12px',
    fontSize: 12,
    color: 'var(--text)',
    overflowX: 'auto',
    whiteSpace: 'nowrap',
  },
  lockIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    margin: '0 auto 14px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
  },
  keyRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '13px 15px',
    background: 'var(--bg-card2)',
  },
}
