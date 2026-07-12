/**
 * frontend/src/pages/StatsPage.jsx — City analytics dashboard.
 *
 * Server aggregates come from GET /api/stats; richer breakdowns (stacked
 * severity-by-class, survey timeline) are derived client-side from the full
 * detection list, which at city-survey scale (≤ a few thousand rows) is cheap.
 */

import React, { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Radar, AlertTriangle, Gauge, CheckCircle2, TrendingUp, Layers, Target, ArrowRight,
  Clock, Wrench, RotateCcw, Banknote, Printer, Activity,
} from 'lucide-react'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, AreaChart, Area, Legend, ComposedChart, Line,
} from 'recharts'
import { useApi } from '../hooks/useApi'
import { fetchStats, fetchDetections, fetchOpsAnalytics } from '../utils/api'
import { fmtNum, fmtDate, fmtPct, fmtRon } from '../utils/format'
import {
  CLASS_COLORS, CLASS_LABELS, SEVERITY_COLORS, SEVERITY_LABELS,
  WORK_ORDER_LABELS, WORK_ORDER_COLORS,
} from '../utils/constants'
import { Kpi, SectionTitle, Spinner, CenterState, EmptyState, ProgressBar } from '../components/ui'
import { printCityReport } from '../utils/cityReport'
import { useAuth } from '../context/AuthContext'

const tooltipStyle = {
  contentStyle: {
    background: 'var(--bg-card)', border: '1px solid var(--border-bright)',
    borderRadius: 8, fontSize: 12,
  },
  itemStyle: { color: 'var(--text)' },
  labelStyle: { color: 'var(--text-muted)' },
}

export default function StatsPage() {
  const { user } = useAuth()
  const { data: stats, loading: statsLoading, error } = useApi(fetchStats, [])
  const { data: detData } = useApi(() => fetchDetections({ page: 1, page_size: 5000 }), [])
  const { data: ops, loading: opsLoading, error: opsError } = useApi(fetchOpsAnalytics, [])
  const detections = detData?.items || []

  // ── Derived datasets ─────────────────────────────────────────────────────
  const severityData = useMemo(() =>
    [1, 2, 3, 4, 5].map(s => ({
      name: SEVERITY_LABELS[s],
      short: `S${s}`,
      value: (stats?.severity_breakdown || []).find(r => r.severity === s)?.count || 0,
      color: SEVERITY_COLORS[s],
    })), [stats])

  const classData = useMemo(() =>
    [...(stats?.damage_type_breakdown || [])]
      .sort((a, b) => b.count - a.count)
      .map(r => ({
        name: CLASS_LABELS[r.damage_type] || r.damage_type,
        count: r.count,
        color: CLASS_COLORS[r.damage_type] || '#888',
      })), [stats])

  // Stacked severity-per-class matrix
  const stackData = useMemo(() => {
    const byClass = {}
    detections.forEach(d => {
      if (!d.severity) return
      const key = d.damage_type
      if (!byClass[key]) byClass[key] = { name: CLASS_LABELS[key] || key, s1: 0, s2: 0, s3: 0, s4: 0, s5: 0, total: 0 }
      byClass[key][`s${d.severity}`]++
      byClass[key].total++
    })
    return Object.values(byClass).sort((a, b) => b.total - a.total).slice(0, 8)
  }, [detections])

  // Survey timeline (detections grouped by last_detected)
  const timelineData = useMemo(() => {
    const byDate = {}
    detections.forEach(d => {
      const day = d.last_detected
      if (!day) return
      if (!byDate[day]) byDate[day] = { date: day, total: 0, critical: 0 }
      byDate[day].total++
      if (d.severity >= 4) byDate[day].critical++
    })
    return Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  }, [detections])

  const repairedPct = stats && stats.total_detections > 0
    ? (stats.fixed_count / stats.total_detections) * 100
    : 0

  if (statsLoading) {
    return (
      <div style={styles.page} className="page-grid-bg">
        <CenterState><Spinner label="Crunching city statistics…" /></CenterState>
      </div>
    )
  }

  if (error) {
    return (
      <div style={styles.page} className="page-grid-bg">
        <EmptyState
          icon={AlertTriangle}
          title="API unreachable"
          sub={`${error} — make sure the backend stack is running (docker compose up -d).`}
        />
      </div>
    )
  }

  const noData = !stats || stats.total_detections === 0

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>

        <SectionTitle
          overline="City analytics"
          title={user?.city ? `Road condition — ${user.city}` : 'Road condition'}
          right={
            <Link to="/map" className="btn btn-sm">
              View on map <ArrowRight size={12} />
            </Link>
          }
        />

        {noData ? (
          <EmptyState
            icon={Radar}
            title="No detections yet"
            sub="Upload a dashcam survey to populate the statistics."
            action={<Link to="/ingest" className="btn btn-accent">Upload footage</Link>}
          />
        ) : (
          <>
            {/* ── KPI row ─────────────────────────────────────────────── */}
            <div style={styles.kpiGrid}>
              <Kpi delay="delay-1" icon={Radar} label="Total detections" value={fmtNum(stats.total_detections)}
                   sub={`${fmtNum(stats.detections_today)} in the last survey`} />
              <Kpi delay="delay-2" icon={AlertTriangle} label="Critical (S4–S5)" value={fmtNum(stats.critical_count)}
                   sub={`${((stats.critical_count / stats.total_detections) * 100).toFixed(1)}% of all damage`} color="var(--red)" />
              <Kpi delay="delay-3" icon={Gauge} label="Avg severity" value={stats.avg_severity ?? '—'}
                   sub={`avg detector confidence ${stats.avg_confidence != null ? fmtPct(stats.avg_confidence) : '—'}`} color="var(--orange)" />
              <Kpi delay="delay-4" icon={CheckCircle2} label="Repaired" value={fmtNum(stats.fixed_count)}
                   sub={`${repairedPct.toFixed(1)}% of the backlog cleared`} color="var(--green)" />
            </div>

            {/* Repair progress */}
            <div className="card anim-fade-up delay-2" style={{ padding: '16px 20px', marginTop: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span className="overline">Repair progress</span>
                <span className="mono" style={{ fontSize: 12, color: 'var(--green)' }}>
                  {fmtNum(stats.fixed_count)} / {fmtNum(stats.total_detections)}
                </span>
              </div>
              <ProgressBar value={repairedPct} color="var(--green)" height={8} />
            </div>

            {/* ── Row: severity donut + class bars ─────────────────────── */}
            <div style={styles.twoCol}>
              <div className="card anim-fade-up delay-1" style={{ padding: 22 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <Target size={14} style={{ color: 'var(--accent)' }} />
                  <span className="overline">Severity distribution</span>
                </div>
                <div style={{ height: 250 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={severityData} dataKey="value" nameKey="short"
                        cx="50%" cy="50%" innerRadius={58} outerRadius={90}
                        paddingAngle={3} stroke="none"
                      >
                        {severityData.map(d => <Cell key={d.short} fill={d.color} />)}
                      </Pie>
                      <Tooltip {...tooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
                  {severityData.map(d => (
                    <span key={d.short} className="mono" style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10.5, color: 'var(--text-muted)' }}>
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: d.color }} />
                      {d.short} · {d.value}
                    </span>
                  ))}
                </div>
              </div>

              <div className="card anim-fade-up delay-2" style={{ padding: 22 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <Layers size={14} style={{ color: 'var(--accent)' }} />
                  <span className="overline">Damage classes</span>
                </div>
                <div style={{ height: 280 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={classData} layout="vertical" margin={{ left: 4, right: 26 }}>
                      <XAxis type="number" hide />
                      <YAxis
                        type="category" dataKey="name" width={128}
                        tick={{ fill: 'var(--text-dim)', fontSize: 11 }}
                        axisLine={false} tickLine={false}
                      />
                      <Tooltip cursor={{ fill: 'var(--accent-dim)' }} {...tooltipStyle} />
                      <Bar dataKey="count" radius={[0, 5, 5, 0]} barSize={13}>
                        {classData.map(d => <Cell key={d.name} fill={d.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* ── Severity × class stacked matrix ──────────────────────── */}
            <div className="card anim-fade-up delay-3" style={{ padding: 22, marginTop: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <Gauge size={14} style={{ color: 'var(--accent)' }} />
                <span className="overline">Severity make-up per class</span>
              </div>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stackData} margin={{ left: -14, right: 10 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 6" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} interval={0} angle={-14} dy={8} height={46} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: 'var(--accent-dim)' }} {...tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
                    {[1, 2, 3, 4, 5].map(s => (
                      <Bar key={s} dataKey={`s${s}`} name={SEVERITY_LABELS[s]} stackId="sev"
                           fill={SEVERITY_COLORS[s]} radius={s === 5 ? [3, 3, 0, 0] : 0} barSize={26} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* ── Survey timeline ──────────────────────────────────────── */}
            {timelineData.length > 1 && (
              <div className="card anim-fade-up delay-4" style={{ padding: 22, marginTop: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <TrendingUp size={14} style={{ color: 'var(--accent)' }} />
                  <span className="overline">Detections by survey date</span>
                </div>
                <div style={{ height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData} margin={{ left: -14, right: 10 }}>
                      <defs>
                        <linearGradient id="gTotal" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gCrit" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#ff5d5d" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="#ff5d5d" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--border)" strokeDasharray="3 6" vertical={false} />
                      <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={fmtDate} />
                      <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
                      <Tooltip {...tooltipStyle} labelFormatter={fmtDate} />
                      <Area type="monotone" dataKey="total" name="All detections" stroke="var(--accent)" strokeWidth={2} fill="url(#gTotal)" />
                      <Area type="monotone" dataKey="critical" name="Critical (S4+)" stroke="#ff5d5d" strokeWidth={2} fill="url(#gCrit)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <div style={{ textAlign: 'center', marginTop: 26, fontSize: 11, color: 'var(--text-muted)' }}>
              Last survey processed: <span className="mono" style={{ color: 'var(--text-dim)' }}>{fmtDate(stats.last_survey_date)}</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 26px)',
    paddingBottom: 40,
  },
  inner: {
    maxWidth: 1100,
    margin: '0 auto',
    padding: '0 26px',
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 14,
  },
  twoCol: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.2fr)',
    gap: 14,
    marginTop: 14,
  },
}
