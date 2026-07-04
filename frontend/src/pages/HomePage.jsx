/**
 * frontend/src/pages/HomePage.jsx — "Command Center"
 *
 * Landing page: hero + live city KPIs + severity pulse + quick actions +
 * the 7-stage pipeline strip. Everything degrades gracefully when the API
 * is offline (the page still renders with placeholders).
 */

import React, { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Map, Upload, BarChart2, ListOrdered, ArrowRight, AlertTriangle,
  Radar, ScanLine, Layers, Ruler, Gauge, Copy, Database, CheckCircle2,
  MapPin, Activity, Table, Radio, Film, Users,
} from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'
import { useApi } from '../hooks/useApi'
import { fetchStats } from '../utils/api'
import { fmtNum, fmtDate } from '../utils/format'
import {
  CLASS_COLORS, CLASS_LABELS, SEVERITY_COLORS, SEVERITY_LABELS, PIPELINE_STAGES,
} from '../utils/constants'
import { Kpi, SectionTitle } from '../components/ui'
import { useAuth } from '../context/AuthContext'

const STAGE_ICONS = [ScanLine, Radar, Layers, Ruler, Gauge, Copy, Database]

export default function HomePage() {
  const navigate = useNavigate()
  const { user, isOperator } = useAuth()
  const { data: stats } = useApi(fetchStats, [])
  const cityName = user?.city || 'your city'

  const severityData = useMemo(() => {
    const rows = stats?.severity_breakdown || []
    return [1, 2, 3, 4, 5].map(s => ({
      name: `S${s}`,
      value: rows.find(r => r.severity === s)?.count || 0,
      color: SEVERITY_COLORS[s],
    }))
  }, [stats])

  const classData = useMemo(() => {
    const rows = [...(stats?.damage_type_breakdown || [])]
    return rows
      .sort((a, b) => b.count - a.count)
      .slice(0, 6)
      .map(r => ({
        name: CLASS_LABELS[r.damage_type] || r.damage_type,
        count: r.count,
        color: CLASS_COLORS[r.damage_type] || '#888',
      }))
  }, [stats])

  const total = stats?.total_detections ?? null
  const openIssues = total !== null ? total - (stats?.fixed_count || 0) : null

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>

        {/* ── Hero ─────────────────────────────────────────────────────── */}
        <section style={styles.hero}>
          <div style={{ maxWidth: 640 }}>
            <div className="overline anim-fade-up" style={{ color: 'var(--accent)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
              <MapPin size={12} />
              {(user?.city || 'ROAD INTELLIGENCE NETWORK').toUpperCase()}
            </div>
            <h1 className="display anim-fade-up delay-1" style={styles.heroTitle}>
              Every street.<br />
              Every crack.<br />
              <span className="text-gradient" style={{ textShadow: 'none' }}>Mapped &amp; ranked.</span>
            </h1>
            <div className="road-divider anim-fade-up delay-2" style={{ width: 180, margin: '22px 0' }} />
            <p className="anim-fade-up delay-2" style={styles.heroSub}>
              RIDS turns ordinary dashcam footage into a live, georeferenced map of road
              damage across {cityName} — detected by <strong>RT-DETR</strong>, measured by{' '}
              <strong>SAM&nbsp;2.1</strong> and <strong>Monodepth2</strong>, scored S1–S5, and
              ranked into a repair plan the city can act on.
            </p>
            <div className="anim-fade-up delay-3" style={{ display: 'flex', gap: 10, marginTop: 26, flexWrap: 'wrap' }}>
              {/* Citizens land on the Live map — the operator pages 404 for them */}
              <Link to={isOperator ? '/map' : '/live'} className="btn btn-accent" style={{ padding: '11px 20px', fontSize: 13 }}>
                <Map size={15} /> Open the live map <ArrowRight size={14} />
              </Link>
              {isOperator ? (
                <Link to="/ingest" className="btn" style={{ padding: '11px 20px', fontSize: 13 }}>
                  <Upload size={15} /> Upload a survey
                </Link>
              ) : (
                <Link to="/live" className="btn" style={{ padding: '11px 20px', fontSize: 13 }}>
                  <Radio size={15} /> Report a hazard
                </Link>
              )}
            </div>
          </div>

          {/* Severity pulse card */}
          <div className="card anim-fade-up delay-3" style={styles.pulseCard}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span className="overline">City severity pulse</span>
              <Activity size={14} style={{ color: 'var(--accent)' }} />
            </div>
            <div style={{ height: 170 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={severityData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={48} outerRadius={72}
                    paddingAngle={3} stroke="none"
                  >
                    {severityData.map(d => <Cell key={d.name} fill={d.color} />)}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-bright)', borderRadius: 8, fontSize: 12 }}
                    itemStyle={{ color: 'var(--text)' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 10, flexWrap: 'wrap' }}>
              {severityData.map(d => (
                <span key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10.5, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ width: 7, height: 7, borderRadius: 2, background: d.color }} />
                  {d.name} {d.value}
                </span>
              ))}
            </div>
            <div style={{ textAlign: 'center', marginTop: 12, fontSize: 11, color: 'var(--text-muted)' }}>
              Last survey: <span className="mono" style={{ color: 'var(--text-dim)' }}>{fmtDate(stats?.last_survey_date)}</span>
            </div>
          </div>
        </section>

        {/* ── Mode chooser (survey mode is operator-only) ──────────────── */}
        <section style={styles.modeGrid}>
          {isOperator && (
          <Link to="/ingest" className="card card-accent-hover anim-fade-up delay-1" style={styles.modeCard}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span style={{ ...styles.modeIcon, background: 'var(--accent-dim)', border: '1px solid var(--border-accent)', color: 'var(--accent)' }}>
                <Film size={18} />
              </span>
              <span className="overline" style={{ color: 'var(--accent)' }}>Survey mode</span>
            </div>
            <div className="display" style={styles.modeTitle}>Upload a drive. Get a full audit.</div>
            <p style={styles.modeText}>
              The deep-analysis path: dashcam video + GPS run through the 7-stage GPU pipeline —
              SAM geometry, Monodepth2 depth, S1–S5 severity, spatial dedup, PostGIS. Precise,
              auditable, thesis-grade results.
            </p>
            <span style={styles.modeCta}>Start a survey <ArrowRight size={13} /></span>
          </Link>
          )}

          <Link to="/live" className="card card-accent-hover anim-fade-up delay-2" style={styles.modeCard}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span style={{ ...styles.modeIcon, background: 'rgba(255,93,93,0.1)', border: '1px solid rgba(255,93,93,0.35)', color: 'var(--red)' }}>
                <Radio size={18} />
              </span>
              <span className="overline" style={{ color: 'var(--red)' }}>
                Live mode
                <span style={{
                  display: 'inline-block', width: 6, height: 6, borderRadius: '50%', marginLeft: 7,
                  background: 'var(--red)', boxShadow: '0 0 6px var(--red)',
                  animation: 'pulse 1.6s ease-in-out infinite',
                }} />
              </span>
            </div>
            <div className="display" style={styles.modeTitle}>Every car is a sensor. Waze for road damage.</div>
            <p style={styles.modeText}>
              The real-time path: cameras running the same RT-DETR model (and people tapping the map)
              stream hazards in live. Independent vehicles re-sighting the same spot escalate it —
              unverified → confirmed → verified — and disputes clear it.
            </p>
            <span style={{ ...styles.modeCta, color: 'var(--red)' }}>
              <Users size={13} /> Open the live map <ArrowRight size={13} />
            </span>
          </Link>
        </section>

        {/* ── KPI row (numbers roll in) ───────────────────────────────── */}
        <section style={styles.kpiGrid}>
          <Kpi delay="delay-1" icon={Radar} label="Detections on record" value="—" countTo={total ?? null} sub="de-duplicated physical damage instances" />
          <Kpi delay="delay-2" icon={AlertTriangle} label="Critical (S4–S5)" value="—" countTo={stats ? stats.critical_count : null} sub="urgent or emergency response" color="var(--red)" />
          <Kpi delay="delay-3" icon={Gauge} label="Average severity" value="—" countTo={stats?.avg_severity != null ? Number(stats.avg_severity) : null} decimals={1} sub={`avg confidence ${stats?.avg_confidence != null ? (stats.avg_confidence * 100).toFixed(0) + '%' : '—'}`} color="var(--orange)" />
          <Kpi delay="delay-4" icon={CheckCircle2} label="Repaired" value="—" countTo={stats ? stats.fixed_count : null} sub={openIssues !== null ? `${fmtNum(openIssues)} still open` : ''} color="var(--green)" />
        </section>

        {/* ── Two-column: top classes + quick actions ──────────────────── */}
        <section style={styles.twoCol}>
          <div className="card anim-fade-up" style={{ padding: 22 }}>
            <SectionTitle overline="Damage taxonomy" title="Most frequent damage classes" />
            {classData.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12.5, padding: '30px 0', textAlign: 'center' }}>
                No detections yet — upload a survey to populate the map.
              </div>
            ) : (
              <div style={{ height: 230 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classData} layout="vertical" margin={{ left: 8, right: 24 }}>
                    <XAxis type="number" hide />
                    <YAxis
                      type="category" dataKey="name" width={130}
                      tick={{ fill: 'var(--text-dim)', fontSize: 11.5 }}
                      axisLine={false} tickLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: 'var(--accent-dim)' }}
                      contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-bright)', borderRadius: 8, fontSize: 12 }}
                    />
                    <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={16}>
                      {classData.map(d => <Cell key={d.name} fill={d.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {isOperator ? (
              <>
                <QuickAction
                  to="/priority" icon={ListOrdered} title="Repair planner"
                  sub="Ranked repair queue with cost estimates" delay="delay-1"
                />
                <QuickAction
                  to="/explorer" icon={Table} title="Detection explorer"
                  sub="Filter, sort, export and audit every record" delay="delay-2"
                />
                <QuickAction
                  to="/stats" icon={BarChart2} title="City analytics"
                  sub="Severity distribution, class breakdown, trends" delay="delay-3"
                />
                <QuickAction
                  to="/ingest" icon={Upload} title="Ingest new footage"
                  sub="Upload .mp4 + .gpx and watch the 7-stage pipeline" delay="delay-4"
                />
              </>
            ) : (
              <>
                <QuickAction
                  to="/live" icon={Radio} title="Live hazard map"
                  sub="See and report hazards around you in real time" delay="delay-1"
                />
                <QuickAction
                  to="/live" icon={Users} title="Drive mode"
                  sub="Mount your phone — impacts auto-report as you drive" delay="delay-2"
                />
                <QuickAction
                  to="/about" icon={BarChart2} title="How RIDS works"
                  sub="The detection pipeline, severity scoring and validation" delay="delay-3"
                />
              </>
            )}
          </div>
        </section>

        {/* ── Pipeline strip ───────────────────────────────────────────── */}
        <section style={{ marginTop: 44 }}>
          <SectionTitle
            overline="Under the hood"
            title="From raw dashcam video to a ranked repair plan"
            right={
              <Link to="/about" className="btn btn-sm btn-ghost">
                How it works <ArrowRight size={12} />
              </Link>
            }
          />
          <div style={styles.pipelineStrip}>
            {PIPELINE_STAGES.map((st, i) => {
              const Icon = STAGE_ICONS[i]
              return (
                <React.Fragment key={st.key}>
                  <div
                    className={`card card-accent-hover anim-fade-up delay-${Math.min(i + 1, 6)}`}
                    style={styles.stageCard}
                    onClick={() => navigate('/about')}
                  >
                    <div style={styles.stageNum} className="mono">{String(i + 1).padStart(2, '0')}</div>
                    <Icon size={18} style={{ color: 'var(--accent)', marginBottom: 8 }} />
                    <div className="display" style={{ fontSize: 12.5, fontWeight: 700 }}>{st.label}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.45 }}>{st.sub}</div>
                  </div>
                  {i < PIPELINE_STAGES.length - 1 && (
                    <div style={styles.stageArrow}>
                      <ArrowRight size={13} style={{ color: 'var(--text-muted)' }} />
                    </div>
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </section>

        {/* ── Footer ───────────────────────────────────────────────────── */}
        <footer style={styles.footer}>
          <div className="road-divider" style={{ width: '100%', marginBottom: 22, opacity: 0.35 }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
              RIDS — Road Infrastructure Detection System · Bachelor's thesis, Babeș-Bolyai University · Paraschiv Tudor, 2026
            </span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              RT-DETR-L · SAM 2.1 · Monodepth2 · PostGIS
            </span>
          </div>
        </footer>
      </div>
    </div>
  )
}

function QuickAction({ to, icon: Icon, title, sub, delay }) {
  return (
    <Link to={to} className={`card card-accent-hover anim-fade-up ${delay}`} style={styles.quickAction}>
      <span style={styles.quickIcon}><Icon size={17} /></span>
      <span style={{ flex: 1 }}>
        <span className="display" style={{ display: 'block', fontSize: 13.5, fontWeight: 700, color: 'var(--text)' }}>{title}</span>
        <span style={{ display: 'block', fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</span>
      </span>
      <ArrowRight size={15} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
    </Link>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'var(--nav-h)',
    overflowX: 'hidden',
  },
  inner: {
    maxWidth: 1160,
    margin: '0 auto',
    padding: '0 28px 40px',
  },
  hero: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 40,
    flexWrap: 'wrap',
    padding: '64px 0 44px',
  },
  heroTitle: {
    fontSize: 'clamp(38px, 5.4vw, 60px)',
    fontWeight: 700,
    lineHeight: 1.04,
    letterSpacing: '-0.025em',
  },
  heroAccent: {
    color: 'var(--accent)',
    textShadow: '0 0 42px var(--accent-glow)',
  },
  heroSub: {
    fontSize: 14.5,
    color: 'var(--text-dim)',
    lineHeight: 1.75,
    maxWidth: 540,
  },
  pulseCard: {
    width: 320,
    padding: '18px 20px 16px',
    flexShrink: 0,
  },
  modeGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: 14,
    margin: '4px 0 22px',
  },
  modeCard: {
    padding: '22px 24px',
    textDecoration: 'none',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
  },
  modeIcon: {
    width: 38, height: 38, borderRadius: 10,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modeTitle: {
    fontSize: 17,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '-0.01em',
    lineHeight: 1.3,
  },
  modeText: {
    fontSize: 12.5,
    color: 'var(--text-dim)',
    lineHeight: 1.7,
    margin: '8px 0 14px',
  },
  modeCta: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 12.5,
    fontWeight: 700,
    color: 'var(--accent)',
    marginTop: 'auto',
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 14,
    marginTop: 8,
  },
  twoCol: {
    display: 'grid',
    // min(100%, …) lets the columns collapse to one on narrow screens
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
    gap: 14,
    marginTop: 14,
  },
  quickAction: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '16px 18px',
    textDecoration: 'none',
    flex: 1,
  },
  quickIcon: {
    width: 40, height: 40, borderRadius: 11, flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    color: 'var(--accent)',
  },
  pipelineStrip: {
    display: 'flex',
    alignItems: 'stretch',
    gap: 6,
    overflowX: 'auto',
    paddingBottom: 8,
  },
  stageCard: {
    minWidth: 138,
    flex: 1,
    padding: '14px 14px 12px',
    position: 'relative',
    cursor: 'pointer',
  },
  stageNum: {
    position: 'absolute',
    top: 10, right: 12,
    fontSize: 10,
    color: 'var(--text-muted)',
  },
  stageArrow: {
    display: 'flex',
    alignItems: 'center',
    flexShrink: 0,
  },
  footer: {
    marginTop: 56,
  },
}
