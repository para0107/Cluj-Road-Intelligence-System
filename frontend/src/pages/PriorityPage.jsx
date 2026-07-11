/**
 * frontend/src/pages/PriorityPage.jsx — Repair planner.
 *
 * Ranked repair queue from GET /api/priority-list, enriched client-side with
 * rough cost estimates (REPAIR_COST_RON heuristics — a planning aid, not a
 * quote). Selecting rows builds a "work order" summary that can be printed.
 */

import React, { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ListOrdered, AlertTriangle, Printer, Banknote, MapPin, HardHat, ArrowRight,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { fetchPriority } from '../utils/api'
import { fmtRon, fmtDate } from '../utils/format'
import {
  CLASS_LABELS, SEVERITY_COLORS, SEVERITY_ACTIONS,
  REPAIR_COST_RON, SEVERITY_COST_FACTOR,
} from '../utils/constants'
import { SevBadge, ClassDot, SectionTitle, Spinner, CenterState, EmptyState, Kpi } from '../components/ui'
import { useAuth } from '../context/AuthContext'

const estimateCost = (item) =>
  (REPAIR_COST_RON[item.damage_type] || 800) * (SEVERITY_COST_FACTOR[item.severity] || 1)

export default function PriorityPage() {
  const { user } = useAuth()
  const { data, loading, error } = useApi(() => fetchPriority(100), [])
  const [selected, setSelected] = useState(new Set())

  const items = useMemo(() =>
    (data?.items || []).map(it => ({ ...it, est_cost: estimateCost(it) })), [data])

  const totalBacklogCost = useMemo(() =>
    items.reduce((acc, it) => acc + it.est_cost, 0), [items])

  const selectedItems = items.filter(it => selected.has(it.id))
  const selectedCost = selectedItems.reduce((acc, it) => acc + it.est_cost, 0)

  const toggle = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const selectTopN = (n) => setSelected(new Set(items.slice(0, n).map(it => it.id)))

  const printWorkOrder = () => {
    const rows = selectedItems.length > 0 ? selectedItems : items.slice(0, 20)
    const sevColors = { 1: '#3ddc84', 2: '#d4a900', 3: '#e08a30', 4: '#e04848', 5: '#a21caf' }
    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>RDDS — Work order</title>
<style>
  body { font-family:'Segoe UI',sans-serif; color:#10141c; margin:0; }
  .cover { background:#05070b; color:#eaff3d; padding:44px 44px 30px; }
  .cover h1 { margin:0 0 6px; font-size:28px; }
  .cover p { margin:0; color:#a8b0c2; font-size:13px; }
  .body { padding:32px 44px; }
  table { width:100%; border-collapse:collapse; }
  th { background:#05070b; color:#eaff3d; text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; padding:10px 13px; }
  td { padding:9px 13px; border-bottom:1px solid #eef0f4; font-size:12.5px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .total { margin-top:18px; font-size:15px; font-weight:700; text-align:right; }
</style></head><body>
<div class="cover">
  <h1>Road repair work order${user?.city ? ` — ${user.city}` : ''}</h1>
  <p>RDDS priority queue · ${rows.length} sites · generated ${new Date().toLocaleString()}</p>
</div>
<div class="body">
<table>
<thead><tr><th>#</th><th>Damage</th><th>Severity</th><th>Priority</th><th>GPS</th><th>Seen</th><th>Est. cost</th></tr></thead>
<tbody>
${rows.map((it, i) => {
  const sc = sevColors[it.severity] || '#888'
  return `<tr>
    <td>${i + 1}</td>
    <td>${CLASS_LABELS[it.damage_type] || it.damage_type}</td>
    <td><span class="badge" style="background:${sc}22;color:${sc}">S${it.severity}</span></td>
    <td style="font-family:monospace">${it.priority_score.toFixed(3)}</td>
    <td style="font-family:monospace;font-size:11px">${it.latitude.toFixed(5)}, ${it.longitude.toFixed(5)}</td>
    <td>${it.detection_count}×</td>
    <td style="font-family:monospace">${Math.round(it.est_cost).toLocaleString()} RON</td>
  </tr>`
}).join('')}
</tbody></table>
<div class="total">Estimated total: ${Math.round(rows.reduce((a, r) => a + r.est_cost, 0)).toLocaleString()} RON</div>
<p style="font-size:10.5px;color:#7a8296;margin-top:26px">
  Cost figures are heuristic planning estimates derived from damage class and severity — not a contractor quote.
</p>
</div>
<script>window.onload = () => window.print()</script>
</body></html>`
    const w = window.open('', '_blank')
    w.document.write(html)
    w.document.close()
  }

  if (loading) {
    return (
      <div style={styles.page} className="page-grid-bg">
        <CenterState><Spinner label="Ranking the repair queue…" /></CenterState>
      </div>
    )
  }

  if (error) {
    return (
      <div style={styles.page} className="page-grid-bg">
        <EmptyState icon={AlertTriangle} title="API unreachable" sub={error} />
      </div>
    )
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>
        <SectionTitle
          overline="Act"
          title="Repair planner"
          right={
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-sm" onClick={() => selectTopN(10)}>Top 10</button>
              <button className="btn btn-sm" onClick={() => selectTopN(25)}>Top 25</button>
              <button className="btn btn-sm btn-accent" onClick={printWorkOrder} disabled={items.length === 0}>
                <Printer size={13} /> Work order
              </button>
            </div>
          }
        />

        {items.length === 0 ? (
          <EmptyState
            icon={ListOrdered}
            title="Nothing to repair (yet)"
            sub="The priority queue fills up as surveys are processed."
            action={<Link to="/ingest" className="btn btn-accent">Upload footage</Link>}
          />
        ) : (
          <>
            {/* ── KPI row ────────────────────────────────────────────── */}
            <div style={styles.kpiGrid}>
              <Kpi delay="delay-1" icon={ListOrdered} label="Queue size" value={items.length}
                   sub="highest-priority open damage sites" />
              <Kpi delay="delay-2" icon={Banknote} label="Backlog estimate" value={fmtRon(totalBacklogCost)}
                   sub="heuristic repair cost of the whole queue" color="var(--orange)" />
              <Kpi delay="delay-3" icon={HardHat} label="Selected for crew" value={selected.size}
                   sub={selected.size ? `≈ ${fmtRon(selectedCost)}` : 'select rows to build a work order'} color="var(--cyan)" />
            </div>

            {/* ── Queue ──────────────────────────────────────────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 16 }}>
              {items.map((it, idx) => {
                const isSel = selected.has(it.id)
                const sevColor = SEVERITY_COLORS[it.severity] || '#888'
                return (
                  <div
                    key={it.id}
                    className={`card anim-fade-up delay-${Math.min(Math.floor(idx / 4) + 1, 6)}`}
                    onClick={() => toggle(it.id)}
                    style={{
                      ...styles.row,
                      borderColor: isSel ? 'var(--border-accent)' : 'var(--border)',
                      background: isSel ? 'var(--accent-dim)' : 'var(--bg-card)',
                    }}
                  >
                    {/* Rank */}
                    <div className="mono" style={{
                      ...styles.rank,
                      color: idx < 3 ? 'var(--accent)' : 'var(--text-muted)',
                      borderColor: idx < 3 ? 'var(--border-accent)' : 'var(--border)',
                    }}>
                      {String(idx + 1).padStart(2, '0')}
                    </div>

                    <ClassDot cls={it.damage_type} size={34} />

                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span className="display" style={{ fontWeight: 700, fontSize: 13.5 }}>
                          {CLASS_LABELS[it.damage_type] || it.damage_type}
                        </span>
                        <SevBadge s={it.severity} compact />
                        <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                          priority {it.priority_score.toFixed(3)}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                        {SEVERITY_ACTIONS[it.severity]}
                      </div>
                    </div>

                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div className="mono" style={{ fontSize: 12.5, fontWeight: 700, color: sevColor }}>
                        {fmtRon(it.est_cost)}
                      </div>
                      <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                        <MapPin size={9} style={{ display: 'inline', marginRight: 3 }} />
                        {it.latitude.toFixed(4)}, {it.longitude.toFixed(4)}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                        seen {it.detection_count}× · {fmtDate(it.last_detected)}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', marginTop: 22 }}>
              <Link to="/map" className="btn">
                See all of it on the map <ArrowRight size={13} />
              </Link>
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
    maxWidth: 900,
    margin: '0 auto',
    padding: '0 26px',
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 14,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '13px 16px',
    cursor: 'pointer',
    userSelect: 'none',
  },
  rank: {
    width: 34, height: 34, borderRadius: 9, flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 12, fontWeight: 700,
    border: '1px solid var(--border)',
    background: 'var(--bg-card2)',
  },
}
