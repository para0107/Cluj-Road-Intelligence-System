import React from 'react'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { Activity, AlertTriangle, Map, TrendingUp, ArrowLeft, CheckCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import { fetchStats, fetchPriority } from '../utils/api'
import { CLASS_COLORS, CLASS_LABELS, SEVERITY_COLORS, SEVERITY_LABELS } from '../utils/constants'

// ── Shared custom tooltip ─────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card2)', border: '1px solid var(--border-bright)',
      borderRadius: 'var(--radius)', padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-dim)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.fill || p.color, fontWeight: 600 }}>
          {p.value} {p.name ? `(${p.name})` : ''}
        </div>
      ))}
    </div>
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────
function StatCard({ icon: Icon, label, value, color, sub }) {
  return (
    <div style={styles.card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ ...styles.cardValue, color }}>{value}</div>
          <div style={styles.cardLabel}>{label}</div>
          {sub && <div style={styles.cardSub}>{sub}</div>}
        </div>
        <div style={{ ...styles.cardIcon, background: `${color}18`, color }}>
          <Icon size={18} />
        </div>
      </div>
    </div>
  )
}

// ── Section wrapper ───────────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionTitle}>{title}</div>
      {children}
    </div>
  )
}

export default function StatsPage() {
  const navigate = useNavigate()
  const { data: stats, loading: sLoading, error: sError } = useApi(fetchStats)
  const { data: priority } = useApi(() => fetchPriority(10))

  if (sLoading) return <LoadingState />
  if (sError)   return <ErrorState error={sError} />
  if (!stats)   return null

  // Prepare chart data
  const typeData = stats.damage_type_breakdown
    .sort((a, b) => b.count - a.count)
    .map(d => ({
      name: CLASS_LABELS[d.damage_type] || d.damage_type,
      count: d.count,
      fill: CLASS_COLORS[d.damage_type] || '#888',
    }))

  const sevData = [1, 2, 3, 4, 5].map(s => {
    const found = stats.severity_breakdown.find(r => r.severity === s)
    return {
      name: SEVERITY_LABELS[s],
      value: found?.count || 0,
      fill: SEVERITY_COLORS[s],
    }
  })

  return (
    <div style={styles.page}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.backBtn} onClick={() => navigate('/')}>
            <ArrowLeft size={14} /> MAP
          </button>
          <div>
            <h1 style={styles.title}>Statistics</h1>
            <p style={styles.subtitle}>
              Road damage analysis · {stats.last_survey_date ? `Last survey: ${stats.last_survey_date}` : 'KITTI validation dataset'}
            </p>
          </div>
        </div>
      </div>

      <div style={styles.body}>
        {/* KPI row */}
        <div style={{ ...styles.kpiRow, gridTemplateColumns: 'repeat(4,1fr)' }}>
          <StatCard icon={Map}           label="Total Detections" value={stats.total_detections}               color="var(--accent)" />
          <StatCard icon={AlertTriangle} label="Critical (S4–S5)"  value={stats.critical_count}                color="var(--red)"    />
          <StatCard icon={TrendingUp}    label="Avg Severity"       value={stats.avg_severity?.toFixed(2) ?? '—'} color="var(--orange)"
            sub={stats.avg_severity ? ['Low','Low-Med','Medium','High','Critical'][Math.round(stats.avg_severity)-1] : null}
          />
          <StatCard icon={CheckCircle}   label="Avg Confidence"     value={stats.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(1)}%` : '—'} color="var(--blue)" />
        </div>

        <div style={styles.grid}>
          {/* Damage type bar chart */}
          <Section title="DETECTIONS BY CLASS">
            <div style={styles.chartWrap}>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={typeData} margin={{ top: 4, right: 8, left: 0, bottom: 60 }}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#6b7280', fontSize: 10 }}
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {typeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>

          {/* Severity pie */}
          <Section title="SEVERITY DISTRIBUTION">
            <div style={styles.chartWrap}>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={sevData.filter(d => d.value > 0)}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={3}
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) => `${name.split('·')[0].trim()} ${(percent * 100).toFixed(0)}%`}
                    labelLine={{ stroke: '#2a3040' }}
                  >
                    {sevData.filter(d => d.value > 0).map((d, i) => (
                      <Cell key={i} fill={d.fill} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {/* Legend */}
              <div style={styles.severityLegend}>
                {sevData.filter(d => d.value > 0).map((d, i) => (
                  <div key={i} style={styles.sevLegendRow}>
                    <span style={{
                      display: 'inline-block', width: 8, height: 8,
                      borderRadius: '50%', background: d.fill, flexShrink: 0,
                    }} />
                    <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>{d.name}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: d.fill }}>{d.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </Section>
        </div>

        {/* Priority list */}
        {priority?.items?.length > 0 && (
          <Section title="TOP 10 PRIORITY DETECTIONS">
            <div style={styles.table}>
              <div style={styles.tableHead}>
                {['#', 'Type', 'Severity', 'Priority Score', 'Detections', 'GPS', 'Last Seen'].map(h => (
                  <div key={h} style={styles.th}>{h}</div>
                ))}
              </div>
              {priority.items.map(item => (
                <div key={item.id} style={styles.tableRow}>
                  <div style={{ ...styles.td, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {item.rank}
                  </div>
                  <div style={{ ...styles.td, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%',
                      background: CLASS_COLORS[item.damage_type] || '#888', flexShrink: 0,
                    }} />
                    <span style={{ fontSize: 12 }}>
                      {CLASS_LABELS[item.damage_type] || item.damage_type}
                    </span>
                  </div>
                  <div style={styles.td}>
                    <span style={{
                      background: `${SEVERITY_COLORS[item.severity]}25`,
                      color: SEVERITY_COLORS[item.severity],
                      border: `1px solid ${SEVERITY_COLORS[item.severity]}50`,
                      borderRadius: 4, padding: '1px 6px',
                      fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700,
                    }}>
                      S{item.severity}
                    </span>
                  </div>
                  <div style={{ ...styles.td, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                    {item.priority_score?.toFixed(4)}
                  </div>
                  <div style={{ ...styles.td, fontFamily: 'var(--font-mono)' }}>
                    {item.detection_count}×
                  </div>
                  <div style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                    {item.latitude?.toFixed(4)}, {item.longitude?.toFixed(4)}
                  </div>
                  <div style={{ ...styles.td, fontSize: 11, color: 'var(--text-muted)' }}>
                    {item.last_detected || '—'}
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  )
}

function LoadingState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 48px)', color: 'var(--text-muted)', gap: 12 }}>
      <div style={{ width: 20, height: 20, border: '2px solid var(--border)', borderTop: '2px solid var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      Loading statistics…
    </div>
  )
}
function ErrorState({ error }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 48px)', color: 'var(--red)', gap: 8 }}>
      <AlertTriangle size={16} /> {error}
    </div>
  )
}

const styles = {
  page: { paddingTop: 48, minHeight: '100vh', background: 'var(--bg)' },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '24px 32px 0',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 16 },
  backBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '6px 12px', background: 'transparent',
    border: '1px solid var(--border-bright)', borderRadius: 'var(--radius)',
    color: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)',
    fontWeight: 700, cursor: 'pointer', letterSpacing: '.08em',
  },
  title: { fontSize: 26, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.5px' },
  subtitle: { fontSize: 12, color: 'var(--text-muted)', marginTop: 2 },
  body: { padding: '24px 32px 48px' },
  kpiRow: { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, marginBottom: 24 },
  card: {
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', padding: '20px 24px',
  },
  cardValue: { fontSize: 28, fontWeight: 800, fontFamily: 'var(--font-mono)', lineHeight: 1 },
  cardLabel: { fontSize: 11, color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '.06em' },
  cardSub:   { fontSize: 11, color: 'var(--text-dim)', marginTop: 2 },
  cardIcon: { width: 36, height: 36, borderRadius: 'var(--radius)', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 },
  section: {
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', padding: '20px 24px',
  },
  sectionTitle: {
    fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
    color: 'var(--text-muted)', letterSpacing: '.12em', marginBottom: 16,
  },
  chartWrap: { width: '100%' },
  severityLegend: { display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: 8 },
  sevLegendRow: { display: 'flex', alignItems: 'center', gap: 6 },
  table: { borderRadius: 'var(--radius)', overflow: 'hidden', border: '1px solid var(--border)' },
  tableHead: {
    display: 'grid', gridTemplateColumns: '32px 1.8fr 80px 120px 80px 160px 100px',
    background: 'var(--bg-card2)', borderBottom: '1px solid var(--border)',
  },
  tableRow: {
    display: 'grid', gridTemplateColumns: '32px 1.8fr 80px 120px 80px 160px 100px',
    borderBottom: '1px solid var(--border)',
    transition: 'background 0.1s',
  },
  th: {
    padding: '8px 12px', fontSize: 10, fontFamily: 'var(--font-mono)',
    fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '.08em',
    textTransform: 'uppercase',
  },
  td: { padding: '10px 12px', fontSize: 13, color: 'var(--text)', alignSelf: 'center' },
}