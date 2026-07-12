/**
 * frontend/src/pages/ImpactPage.jsx — Citizen engagement ("My impact").
 *
 * Route /impact, open to every signed-in role. Shows what the user's reports
 * actually did: points, badges, where they stand against other drivers, and
 * the current state of every hazard they sent in.
 *
 * Points only ever come from reports that other drivers confirm or that the
 * city repairs, so this page rewards accuracy, not volume.
 */

import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Award, MapPin, CheckCircle2, Wrench, Flame, Trophy, AlertTriangle, Inbox,
} from 'lucide-react'
import { fetchMyImpact, fetchLeaderboard } from '../utils/api'
import { fmtDate, fmtCoord, fmtNum } from '../utils/format'
import {
  BADGES, ALL_BADGE_KEYS, CLASS_LABELS,
  LIVE_STATUS_LABELS, LIVE_STATUS_COLORS,
} from '../utils/constants'
import { Kpi, SectionTitle, Spinner, CenterState, EmptyState, ClassDot } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import useMotionOk from '../hooks/useMotionOk'
import SpotlightCard from '../reactbits/SpotlightCard/SpotlightCard'
import AnimatedContent from '../reactbits/AnimatedContent/AnimatedContent'

/* SpotlightCard ships its own dark-only look (#111 background, #222 border,
   2rem padding). We cannot restyle it through props, so this doubled-specificity
   rule re-skins it with design tokens and wins over the vendored CSS no matter
   which stylesheet loads first. */
const SPOTLIGHT_SKIN = `
.impact-badge.card-spotlight {
  background-color: var(--bg-card);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-lg);
  padding: 16px 14px;
  height: 100%;
}
`

const MEDALS = { 1: '🥇', 2: '🥈', 3: '🥉' }

/** Section wrapper. GSAP reveals it on scroll, but only when motion is on —
 *  AnimatedContent starts hidden, so with motion off we render a plain div and
 *  the page stays fully readable. */
function Reveal({ children, delay = 0, motionOk }) {
  if (!motionOk) return <div>{children}</div>
  return (
    <AnimatedContent distance={40} duration={0.6} delay={delay} threshold={0.15}>
      {children}
    </AnimatedContent>
  )
}

/** One small pill. Used for report states and for the rank chips. */
function StateChip({ label, color }) {
  return (
    <span style={{
      background: `${color}1c`, color, border: `1px solid ${color}55`,
      borderRadius: 999, padding: '2px 9px', fontSize: 10.5, fontWeight: 600,
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}

function BadgeBody({ badgeKey, awardedAt, earned }) {
  const b = BADGES[badgeKey]
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 27, lineHeight: 1.1, marginBottom: 8 }}>{b.icon}</div>
      <div className="display" style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>
        {b.label}
      </div>
      <div style={{ fontSize: 10.5, color: 'var(--text-muted)', lineHeight: 1.45 }}>
        {b.description}
      </div>
      {earned && (
        <div className="mono" style={{ fontSize: 9.5, color: 'var(--accent)', marginTop: 8 }}>
          {fmtDate(awardedAt)}
        </div>
      )}
    </div>
  )
}

export default function ImpactPage() {
  const { user } = useAuth()
  const motionOk = useMotionOk()

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [scope, setScope] = useState('city')
  const [board, setBoard] = useState(null)
  const [boardLoading, setBoardLoading] = useState(true)
  const [boardError, setBoardError] = useState(null)

  // ── My impact ───────────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true
    setLoading(true)
    fetchMyImpact()
      .then(res => { if (alive) { setData(res); setError(null) } })
      .catch(err => {
        if (alive) setError(err?.response?.data?.detail || err.message || 'Could not load your impact.')
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [])

  // ── Leaderboard (refetches when the scope toggle changes) ───────────────
  useEffect(() => {
    let alive = true
    setBoardLoading(true)
    const city = scope === 'city' ? (user?.city || null) : null
    fetchLeaderboard(city, 50)
      .then(res => { if (alive) { setBoard(res); setBoardError(null) } })
      .catch(err => {
        if (alive) setBoardError(err?.response?.data?.detail || err.message || 'Could not load the leaderboard.')
      })
      .finally(() => { if (alive) setBoardLoading(false) })
    return () => { alive = false }
  }, [scope, user?.city])

  if (loading) {
    return (
      <div className="page-grid-bg" style={styles.page}>
        <CenterState><Spinner label="Adding up your impact…" /></CenterState>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="page-grid-bg" style={styles.page}>
        <EmptyState
          icon={AlertTriangle}
          title="We could not load your impact"
          sub={String(error)}
          action={<button className="btn btn-accent" onClick={() => window.location.reload()}>Try again</button>}
        />
      </div>
    )
  }

  const stats = data?.stats || {}
  const badges = data?.badges || []
  const earnedMap = badges.reduce((acc, b) => { acc[b.badge_key] = b.awarded_at; return acc }, {})

  const streak = stats.current_streak_days || 0
  const bestStreak = stats.best_streak_days || 0

  const reports = [...(data?.my_reports || [])]
    .sort((a, b) => new Date(b.last_reported) - new Date(a.last_reported))
    .slice(0, 50)

  const boardItems = board?.items || []

  return (
    <div className="page-grid-bg" style={styles.page}>
      <style>{SPOTLIGHT_SKIN}</style>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="anim-fade-up" style={{ marginBottom: 26 }}>
        <div className="overline" style={{ color: 'var(--accent)', marginBottom: 8 }}>
          Your contribution
        </div>
        <h1 className="display" style={styles.h1}>My impact</h1>
        <div className="road-divider" style={{ margin: '14px 0 14px', maxWidth: 260 }} />
        <p style={styles.lead}>
          Points come only from reports that other drivers confirm or that the city fixes.
          Being right matters more than reporting a lot.
        </p>
      </div>

      {/* ── KPIs + streak + rank ───────────────────────────────────────── */}
      <Reveal motionOk={motionOk} delay={0}>
        <div style={styles.kpiGrid}>
          <Kpi delay="delay-1" icon={Award} label="Points" countTo={stats.points_total || 0}
               color="var(--accent)" sub="earned from confirmed and repaired reports" />
          <Kpi delay="delay-2" icon={MapPin} label="Reports sent" countTo={stats.reports_total || 0}
               sub="hazards you flagged" />
          <Kpi delay="delay-3" icon={CheckCircle2} label="Confirmed" countTo={stats.confirmed_total || 0}
               color="var(--green)" sub="other drivers agreed with you" />
          <Kpi delay="delay-4" icon={Wrench} label="Repaired" countTo={stats.fixed_total || 0}
               color="var(--cyan)" sub="fixed by the city after you reported" />
        </div>

        <div className="card anim-fade-up delay-4" style={styles.streakStrip}>
          <div style={styles.streakLeft}>
            <span style={{
              width: 34, height: 34, borderRadius: 10, flexShrink: 0,
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              background: streak > 0 ? 'rgba(255, 159, 67, 0.14)' : 'var(--bg-card2)',
              border: `1px solid ${streak > 0 ? 'rgba(255, 159, 67, 0.4)' : 'var(--border)'}`,
            }}>
              <Flame size={17} style={{ color: streak > 0 ? 'var(--orange)' : 'var(--text-muted)' }} />
            </span>
            {streak > 0 ? (
              <div>
                <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>
                  {streak} day{streak === 1 ? '' : 's'} in a row
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Best so far: {bestStreak} day{bestStreak === 1 ? '' : 's'}
                </div>
              </div>
            ) : (
              <div>
                <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>
                  Report today to start a streak.
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Best so far: {bestStreak} day{bestStreak === 1 ? '' : 's'}
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {user?.city && (
              <StateChip
                label={data?.rank_city ? `Rank #${data.rank_city} in ${user.city}` : `Unranked in ${user.city}`}
                color="var(--accent)"
              />
            )}
            <StateChip
              label={data?.rank_global ? `#${data.rank_global} overall` : 'Unranked overall'}
              color="var(--cyan)"
            />
          </div>
        </div>
      </Reveal>

      {/* ── Badges ─────────────────────────────────────────────────────── */}
      <Reveal motionOk={motionOk} delay={0.1}>
        <div style={styles.section}>
          <SectionTitle
            overline="Milestones"
            title="Badges"
            right={
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {badges.length} / {ALL_BADGE_KEYS.length}
              </span>
            }
          />
          <div style={styles.badgeGrid}>
            {ALL_BADGE_KEYS.map((key) => {
              const awardedAt = earnedMap[key]
              const earned = Boolean(awardedAt)
              if (earned) {
                return (
                  <SpotlightCard key={key} className="impact-badge" spotlightColor="var(--accent-glow)">
                    <BadgeBody badgeKey={key} awardedAt={awardedAt} earned />
                  </SpotlightCard>
                )
              }
              return (
                <div key={key} className="card" style={styles.badgeLocked}>
                  <BadgeBody badgeKey={key} earned={false} />
                </div>
              )
            })}
          </div>
        </div>
      </Reveal>

      {/* ── Leaderboard ────────────────────────────────────────────────── */}
      <Reveal motionOk={motionOk} delay={0.15}>
        <div style={styles.section}>
          <SectionTitle
            overline="Standings"
            title="Leaderboard"
            right={
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="chip"
                  onClick={() => setScope('city')}
                  style={scope === 'city' ? styles.chipOn : undefined}
                  disabled={!user?.city}
                >
                  {user?.city || 'My city'}
                </button>
                <button
                  className="chip"
                  onClick={() => setScope('global')}
                  style={scope === 'global' ? styles.chipOn : undefined}
                >
                  Global
                </button>
              </div>
            }
          />

          <div className="card" style={{ padding: 8 }}>
            {boardError && (
              <div style={styles.errorBanner}>
                <AlertTriangle size={13} /> {String(boardError)}
              </div>
            )}

            {boardLoading ? (
              <CenterState><Spinner size={24} label="Loading standings…" /></CenterState>
            ) : boardItems.length === 0 ? (
              <EmptyState
                icon={Trophy}
                title="No one has scored yet"
                sub="Be the first. Send a report and it starts here."
                action={<Link to="/live" className="btn btn-accent">Report a hazard</Link>}
              />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {boardItems.map((row) => {
                  const isMe = user?.username && row.username === user.username
                  return (
                    <div
                      key={`${row.rank}-${row.username}`}
                      className="table-row-hover"
                      style={{
                        ...styles.lbRow,
                        border: `1px solid ${isMe ? 'var(--border-accent)' : 'transparent'}`,
                        background: isMe ? 'var(--accent-dim)' : 'transparent',
                      }}
                    >
                      <span className="mono" style={{
                        ...styles.lbRank,
                        color: row.rank <= 3 ? 'var(--accent)' : 'var(--text-muted)',
                      }}>
                        {MEDALS[row.rank] || `#${row.rank}`}
                      </span>

                      <span style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={styles.lbName}>{row.username}</span>
                        {isMe && <StateChip label="you" color="var(--accent)" />}
                      </span>

                      <span style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0 }}>
                        {fmtNum(row.confirmed_total)} confirmed
                      </span>

                      <span className="mono" style={styles.lbPoints}>
                        {fmtNum(row.points_total)}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </Reveal>

      {/* ── My reports ─────────────────────────────────────────────────── */}
      <Reveal motionOk={motionOk} delay={0.2}>
        <div style={styles.section}>
          <SectionTitle
            overline="History"
            title="My reports"
            right={
              reports.length > 0 && (
                <Link to="/live" className="btn btn-sm">Report a hazard</Link>
              )
            }
          />

          {reports.length === 0 ? (
            <div className="card" style={{ padding: 8 }}>
              <EmptyState
                icon={Inbox}
                title="You have not reported anything yet"
                sub="Hazards you report show up here with what happened to them."
                action={<Link to="/live" className="btn btn-accent">Report a hazard</Link>}
              />
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {reports.map((r) => {
                const statusColor = LIVE_STATUS_COLORS[r.status] || 'var(--text-muted)'
                const expired = !r.is_active && !r.resolved && !r.promoted
                return (
                  <div key={r.event_id} className="card table-row-hover" style={styles.reportRow}>
                    <ClassDot cls={r.damage_type} size={32} />

                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={styles.reportTop}>
                        <span className="display" style={{ fontWeight: 700, fontSize: 13 }}>
                          {CLASS_LABELS[r.damage_type] || r.damage_type}
                        </span>
                        <StateChip
                          label={LIVE_STATUS_LABELS[r.status] || r.status}
                          color={statusColor}
                        />
                        {r.resolved && <StateChip label="Fixed" color="var(--green)" />}
                        {r.promoted && <StateChip label="Now official" color="var(--cyan)" />}
                        {expired && <StateChip label="Expired" color="var(--text-muted)" />}
                      </div>
                      <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 4 }}>
                        {fmtCoord(r.latitude, r.longitude)}
                      </div>
                    </div>

                    <div style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0, textAlign: 'right' }}>
                      {fmtDate(r.last_reported)}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </Reveal>
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
    maxWidth: 1160,
    margin: '0 auto',
  },
  h1: {
    fontSize: 34,
    fontWeight: 700,
    letterSpacing: '-0.02em',
    lineHeight: 1.1,
  },
  lead: {
    fontSize: 13.5,
    color: 'var(--text-dim)',
    maxWidth: 620,
    lineHeight: 1.6,
  },
  section: {
    marginTop: 34,
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: 14,
  },
  streakStrip: {
    marginTop: 14,
    padding: '12px 16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 14,
    flexWrap: 'wrap',
  },
  streakLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  badgeGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: 12,
  },
  badgeLocked: {
    padding: '16px 14px',
    opacity: 0.38,
    filter: 'grayscale(1)',
  },
  chipOn: {
    borderColor: 'var(--accent)',
    background: 'var(--accent-dim)',
    color: 'var(--accent)',
  },
  lbRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
  },
  lbRank: {
    width: 34,
    flexShrink: 0,
    fontSize: 12,
    fontWeight: 700,
    textAlign: 'center',
  },
  lbName: {
    fontSize: 13,
    fontWeight: 600,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  lbPoints: {
    fontSize: 13,
    fontWeight: 700,
    color: 'var(--accent)',
    width: 68,
    textAlign: 'right',
    flexShrink: 0,
  },
  reportRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '11px 14px',
  },
  reportTop: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    flexWrap: 'wrap',
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    margin: 8,
    padding: '8px 11px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255, 93, 93, 0.1)',
    border: '1px solid rgba(255, 93, 93, 0.35)',
    color: 'var(--red)',
    fontSize: 11.5,
  },
}
