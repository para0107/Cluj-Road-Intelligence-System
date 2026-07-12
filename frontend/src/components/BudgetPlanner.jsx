/**
 * frontend/src/components/BudgetPlanner.jsx
 *
 * Client-side repair budget calculator. Takes the /stats response and turns
 * it into a "what would it cost to fix X% of each severity band" sketch.
 * No API calls: every number here comes from REPAIR_COST_RON and
 * SEVERITY_COST_FACTOR, so it is a planning aid, not a quote.
 */

import React, { useMemo, useState, useRef, useEffect } from 'react'
import { Banknote, Printer } from 'lucide-react'
import { SectionTitle, SevBadge, Spinner, CenterState } from './ui'
import {
  REPAIR_COST_RON, SEVERITY_COST_FACTOR, SEVERITY_LABELS, CLASS_LABELS,
} from '../utils/constants'
import { fmtNum, fmtRon } from '../utils/format'
import { useAuth } from '../context/AuthContext'
import CountUp from '../reactbits/CountUp/CountUp'

const BANDS = [5, 4, 3, 2, 1]
const DEFAULT_PLAN = { 1: 10, 2: 30, 3: 60, 4: 100, 5: 100 }
const FALLBACK_BASE = 700          // RON, used when no class mix is known

export default function BudgetPlanner({ stats }) {
  const { user } = useAuth()
  const [plan, setPlan] = useState(DEFAULT_PLAN)
  const [contingency, setContingency] = useState(10)

  // Average unit cost across the damage mix actually present in the city.
  const avgBase = useMemo(() => {
    const mix = stats?.damage_type_breakdown || []
    let cost = 0
    let n = 0
    mix.forEach((row) => {
      const count = Number(row?.count) || 0
      if (count <= 0) return
      cost += (REPAIR_COST_RON[row.damage_type] ?? FALLBACK_BASE) * count
      n += count
    })
    return n > 0 ? cost / n : FALLBACK_BASE
  }, [stats])

  // One row per severity band: how many sites, what each costs, how many the
  // plan repairs, and the subtotal.
  const rows = useMemo(() => {
    const bySeverity = {}
    ;(stats?.severity_breakdown || []).forEach((row) => {
      bySeverity[Number(row?.severity)] = Number(row?.count) || 0
    })
    return BANDS.map((s) => {
      const count = bySeverity[s] || 0
      const unit = avgBase * (SEVERITY_COST_FACTOR[s] ?? 1)
      const pct = plan[s] ?? 0
      const planned = Math.round((count * pct) / 100)
      return { s, count, unit, pct, planned, subtotal: planned * unit }
    })
  }, [stats, avgBase, plan])

  const subtotal = rows.reduce((acc, r) => acc + r.subtotal, 0)
  const contingencyRon = (subtotal * contingency) / 100
  const total = Math.round(subtotal + contingencyRon)
  const plannedSites = rows.reduce((acc, r) => acc + r.planned, 0)
  const totalSites = rows.reduce((acc, r) => acc + r.count, 0)

  // CountUp animates from the previous total instead of snapping back to 0
  // every time a slider moves.
  const prevTotal = useRef(0)
  useEffect(() => { prevTotal.current = total }, [total])

  const setBand = (s, value) => setPlan(p => ({ ...p, [s]: Number(value) }))

  const printSheet = () => {
    const mix = (stats?.damage_type_breakdown || [])
      .slice()
      .sort((a, b) => (b.count || 0) - (a.count || 0))
      .slice(0, 5)
      .map(r => `${CLASS_LABELS[r.damage_type] || r.damage_type} (${r.count})`)
      .join(' · ') || 'no damage recorded yet'
    const sevColors = { 1: '#3ddc84', 2: '#d4a900', 3: '#e08a30', 4: '#e04848', 5: '#a21caf' }
    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>RDDS — Repair budget</title>
<style>
  body { font-family:'Segoe UI',sans-serif; color:#10141c; margin:0; }
  .cover { background:#05070b; color:#eaff3d; padding:44px 44px 30px; }
  .cover h1 { margin:0 0 6px; font-size:28px; }
  .cover p { margin:0; color:#a8b0c2; font-size:13px; }
  .body { padding:32px 44px; }
  h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:#7a8296; margin:0 0 10px; }
  .assume { background:#f7f8fb; border:1px solid #eef0f4; border-radius:8px; padding:14px 18px; margin-bottom:26px; font-size:12.5px; line-height:1.7; }
  table { width:100%; border-collapse:collapse; }
  th { background:#05070b; color:#eaff3d; text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; padding:10px 13px; }
  td { padding:9px 13px; border-bottom:1px solid #eef0f4; font-size:12.5px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .num { font-family:monospace; text-align:right; }
  .sum { margin-top:18px; font-size:13px; text-align:right; color:#3d4557; }
  .total { margin-top:8px; font-size:19px; font-weight:700; text-align:right; }
</style></head><body>
<div class="cover">
  <h1>Road repair budget${user?.city ? ` — ${user.city}` : ''}</h1>
  <p>RDDS planning sheet · ${plannedSites} of ${totalSites} sites · generated ${new Date().toLocaleString()}</p>
</div>
<div class="body">
  <h2>Assumptions</h2>
  <div class="assume">
    Average unit cost: <b>${Math.round(avgBase).toLocaleString()} RON</b> per site, from the damage mix on record.<br>
    Damage mix: ${mix}.<br>
    Severity multipliers: ${BANDS.slice().reverse().map(s => `S${s} ×${SEVERITY_COST_FACTOR[s]}`).join(' · ')}.<br>
    Contingency: <b>${contingency}%</b> on top of the subtotal.
  </div>

  <h2>Plan</h2>
  <table>
    <thead><tr>
      <th>Band</th><th>Sites on record</th><th>Share planned</th>
      <th>Sites to repair</th><th>Unit cost</th><th>Subtotal</th>
    </tr></thead>
    <tbody>
${rows.map((r) => {
  const sc = sevColors[r.s] || '#888'
  return `      <tr>
        <td><span class="badge" style="background:${sc}22;color:${sc}">S${r.s}</span></td>
        <td class="num">${r.count.toLocaleString()}</td>
        <td class="num">${r.pct}%</td>
        <td class="num">${r.planned.toLocaleString()}</td>
        <td class="num">${Math.round(r.unit).toLocaleString()} RON</td>
        <td class="num">${Math.round(r.subtotal).toLocaleString()} RON</td>
      </tr>`
}).join('\n')}
    </tbody>
  </table>

  <div class="sum">Subtotal: ${Math.round(subtotal).toLocaleString()} RON</div>
  <div class="sum">Contingency (${contingency}%): ${Math.round(contingencyRon).toLocaleString()} RON</div>
  <div class="total">Planned budget: ${total.toLocaleString()} RON</div>

  <p style="font-size:10.5px;color:#7a8296;margin-top:26px">
    These are planning estimates from average unit costs, not a quote.
  </p>
</div>
<script>window.onload = () => window.print()</script>
</body></html>`
    const w = window.open('', '_blank')
    if (!w) return
    w.document.write(html)
    w.document.close()
  }

  if (!stats) {
    return (
      <div className="card" style={styles.card}>
        <CenterState><Spinner label="Loading the numbers…" /></CenterState>
      </div>
    )
  }

  return (
    <div className="card" style={styles.card}>
      <SectionTitle
        overline="Plan"
        title="Repair budget"
        right={
          <button className="btn btn-sm" onClick={printSheet}>
            <Printer size={13} /> Print budget sheet
          </button>
        }
      />

      {/* Headline total */}
      <div style={styles.totalBox}>
        <div>
          <div className="overline" style={{ marginBottom: 6 }}>Planned budget</div>
          <div className="display" style={styles.totalValue}>
            <CountUp from={prevTotal.current} to={total} separator="," duration={0.7} />
            <span style={{ fontSize: 18, color: 'var(--text-dim)', marginLeft: 8 }}>RON</span>
          </div>
          <div style={styles.totalSub}>
            {fmtNum(plannedSites)} of {fmtNum(totalSites)} sites · {fmtRon(avgBase)} average per site
          </div>
        </div>
        <span style={styles.totalIcon}>
          <Banknote size={20} style={{ color: 'var(--accent)' }} />
        </span>
      </div>

      {/* Sliders */}
      <div style={styles.sliders}>
        {rows.map((r) => (
          <div key={r.s} style={styles.sliderRow}>
            <div style={styles.sliderHead}>
              <label htmlFor={`band-${r.s}`} style={styles.sliderLabel}>
                How much of S{r.s} do you plan to repair?
              </label>
              <span className="mono" style={styles.sliderValue}>{r.pct}%</span>
            </div>
            <input
              id={`band-${r.s}`}
              type="range"
              min="0"
              max="100"
              step="5"
              value={r.pct}
              onChange={(e) => setBand(r.s, e.target.value)}
              style={styles.range}
            />
            <div style={styles.sliderSub}>
              {fmtNum(r.count)} on record · {fmtNum(r.planned)} in the plan · {fmtRon(r.subtotal)}
            </div>
          </div>
        ))}

        <div style={{ ...styles.sliderRow, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
          <div style={styles.sliderHead}>
            <label htmlFor="contingency" style={styles.sliderLabel}>Contingency</label>
            <span className="mono" style={styles.sliderValue}>{contingency}%</span>
          </div>
          <input
            id="contingency"
            type="range"
            min="0"
            max="30"
            step="1"
            value={contingency}
            onChange={(e) => setContingency(Number(e.target.value))}
            style={styles.range}
          />
          <div style={styles.sliderSub}>
            Adds {fmtRon(contingencyRon)} on top of a {fmtRon(subtotal)} subtotal.
          </div>
        </div>
      </div>

      {/* Breakdown */}
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={{ ...styles.th, textAlign: 'left' }}>Band</th>
              <th style={styles.th}>Sites</th>
              <th style={styles.th}>Planned</th>
              <th style={styles.th}>Unit cost</th>
              <th style={styles.th}>Subtotal</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.s} className="table-row-hover">
                <td style={{ ...styles.td, textAlign: 'left' }}>
                  <SevBadge s={r.s} compact />
                  <span style={styles.bandName}>{(SEVERITY_LABELS[r.s] || '').split('· ')[1] || ''}</span>
                </td>
                <td className="mono" style={styles.td}>{fmtNum(r.count)}</td>
                <td className="mono" style={{ ...styles.td, color: 'var(--text)' }}>{fmtNum(r.planned)}</td>
                <td className="mono" style={styles.td}>{fmtRon(r.unit)}</td>
                <td className="mono" style={{ ...styles.td, color: 'var(--text)', fontWeight: 700 }}>
                  {fmtRon(r.subtotal)}
                </td>
              </tr>
            ))}
            <tr>
              <td style={{ ...styles.td, textAlign: 'left', color: 'var(--text-dim)' }} colSpan={4}>
                Contingency ({contingency}%)
              </td>
              <td className="mono" style={{ ...styles.td, color: 'var(--text)' }}>{fmtRon(contingencyRon)}</td>
            </tr>
            <tr>
              <td style={{ ...styles.tdTotal, textAlign: 'left' }} colSpan={4}>Planned budget</td>
              <td className="mono" style={{ ...styles.tdTotal, color: 'var(--accent)' }}>{fmtRon(total)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p style={styles.disclaimer}>
        These are planning estimates from average unit costs, not a quote.
      </p>
    </div>
  )
}

const styles = {
  card: {
    padding: '20px 22px 18px',
  },
  totalBox: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
    padding: '16px 18px',
    borderRadius: 'var(--radius-lg)',
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
  },
  totalValue: {
    fontSize: 32,
    fontWeight: 700,
    lineHeight: 1.1,
    color: 'var(--text)',
    fontVariantNumeric: 'tabular-nums',
  },
  totalSub: {
    fontSize: 11.5,
    color: 'var(--text-muted)',
    marginTop: 6,
  },
  totalIcon: {
    width: 44,
    height: 44,
    borderRadius: 12,
    flexShrink: 0,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
  },
  sliders: {
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    margin: '18px 0 4px',
  },
  sliderRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
  },
  sliderHead: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  sliderLabel: {
    fontSize: 12.5,
    color: 'var(--text-dim)',
  },
  sliderValue: {
    fontSize: 12,
    fontWeight: 700,
    color: 'var(--accent)',
  },
  range: {
    width: '100%',
    accentColor: 'var(--accent)',
    cursor: 'pointer',
  },
  sliderSub: {
    fontSize: 10.5,
    color: 'var(--text-muted)',
  },
  tableWrap: {
    marginTop: 18,
    overflowX: 'auto',
  },
  table: {
    width: '100%',
    minWidth: 460,
    borderCollapse: 'collapse',
  },
  th: {
    textAlign: 'right',
    padding: '8px 10px',
    borderBottom: '1px solid var(--border)',
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    whiteSpace: 'nowrap',
  },
  td: {
    textAlign: 'right',
    padding: '9px 10px',
    borderBottom: '1px solid var(--border)',
    fontSize: 12,
    color: 'var(--text-dim)',
    whiteSpace: 'nowrap',
  },
  tdTotal: {
    textAlign: 'right',
    padding: '11px 10px',
    fontSize: 13,
    fontWeight: 700,
    color: 'var(--text)',
    whiteSpace: 'nowrap',
  },
  bandName: {
    marginLeft: 8,
    fontSize: 11.5,
    color: 'var(--text-muted)',
  },
  disclaimer: {
    marginTop: 14,
    fontSize: 10.5,
    color: 'var(--text-muted)',
  },
}
