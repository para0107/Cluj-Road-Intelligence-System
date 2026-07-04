/**
 * frontend/src/components/ui.jsx
 *
 * Shared UI primitives — the single source of visual truth for every page.
 * All colours come from the CSS design tokens in index.css.
 */

import React from 'react'
import { SEVERITY_COLORS, SEVERITY_LABELS, CLASS_COLORS, CLASS_LABELS, CLASS_ICONS } from '../utils/constants'
import useCountUp from '../hooks/useCountUp'

// ── Severity badge ────────────────────────────────────────────────────────
export function SevBadge({ s, compact = false }) {
  const color = SEVERITY_COLORS[s] || '#888'
  return (
    <span style={{
      background: `${color}22`, color, border: `1px solid ${color}55`,
      borderRadius: 5, padding: compact ? '1px 6px' : '2px 8px',
      fontSize: compact ? 10 : 11, fontFamily: 'var(--font-mono)', fontWeight: 700,
      whiteSpace: 'nowrap',
    }}>
      {compact ? `S${s}` : (SEVERITY_LABELS[s] || `S${s}`)}
    </span>
  )
}

// ── Damage class chip (with dot + optional count) ────────────────────────
export function ClassChip({ cls, count, active = true, onClick }) {
  const color = CLASS_COLORS[cls] || '#888'
  return (
    <button
      className="chip"
      onClick={onClick}
      style={{
        borderColor: active ? `${color}88` : 'var(--border)',
        background: active ? `${color}16` : 'transparent',
        color: active ? color : 'var(--text-muted)',
      }}
    >
      <span style={{
        width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
        background: active ? color : 'var(--text-muted)',
      }} />
      {CLASS_LABELS[cls] || cls}
      {count !== undefined && (
        <span style={{
          background: active ? `${color}2e` : 'var(--border)',
          borderRadius: 8, padding: '0 6px', fontSize: 10,
          fontFamily: 'var(--font-mono)',
          color: active ? color : 'var(--text-muted)',
        }}>
          {count}
        </span>
      )}
    </button>
  )
}

// ── Class icon dot ────────────────────────────────────────────────────────
export function ClassDot({ cls, size = 26 }) {
  const color = CLASS_COLORS[cls] || '#888'
  return (
    <span style={{
      width: size, height: size, borderRadius: 8, flexShrink: 0,
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      background: `${color}1c`, border: `1px solid ${color}44`,
      color, fontSize: size * 0.55,
    }}>
      {CLASS_ICONS[cls] || '●'}
    </span>
  )
}

// ── KPI stat card ─────────────────────────────────────────────────────────
// Pass `countTo` (a number) to make the value roll in with a count-up
// animation; `value` stays the fallback for non-numeric displays.
export function Kpi({ icon: Icon, label, value, countTo = null, decimals = 0, sub, color = 'var(--accent)', delay = '' }) {
  const animated = useCountUp(countTo ?? NaN, decimals)
  const shown = countTo != null
    ? animated.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
    : value
  return (
    <div className={`card card-accent-hover anim-fade-up ${delay}`} style={{ padding: '18px 20px', position: 'relative', overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, ${color}, transparent 70%)`,
      }} />
      <div style={{
        position: 'absolute', top: -34, right: -34, width: 96, height: 96, borderRadius: '50%',
        background: `radial-gradient(circle, color-mix(in srgb, ${color} 14%, transparent), transparent 70%)`,
        pointerEvents: 'none',
      }} />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span className="overline">{label}</span>
        {Icon && <Icon size={15} style={{ color, opacity: 0.9 }} />}
      </div>
      <div className="display" style={{ fontSize: 30, fontWeight: 700, lineHeight: 1.1, color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
        {shown}
      </div>
      {sub && <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>{sub}</div>}
    </div>
  )
}

// ── Section title with road-marking accent ───────────────────────────────
export function SectionTitle({ overline, title, right }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 16, marginBottom: 18 }}>
      <div>
        {overline && <div className="overline" style={{ color: 'var(--accent)', marginBottom: 6 }}>{overline}</div>}
        <h2 className="display" style={{ fontSize: 21, fontWeight: 700, letterSpacing: '-0.01em' }}>{title}</h2>
      </div>
      {right}
    </div>
  )
}

// ── Spinner ───────────────────────────────────────────────────────────────
export function Spinner({ size = 30, label }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      <div style={{
        width: size, height: size, borderRadius: '50%',
        border: '3px solid var(--border)', borderTopColor: 'var(--accent)',
        animation: 'spin 0.8s linear infinite',
      }} />
      {label && <span className="mono" style={{ color: 'var(--accent)', fontSize: 12.5 }}>{label}</span>}
    </div>
  )
}

// ── Full-area centered state (loading / empty / error) ──────────────────
export function CenterState({ children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      minHeight: 260, width: '100%',
    }}>
      {children}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, sub, action }) {
  return (
    <CenterState>
      <div style={{ textAlign: 'center', maxWidth: 380 }} className="anim-fade-up">
        {Icon && (
          <div style={{
            width: 52, height: 52, borderRadius: 14, margin: '0 auto 14px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'var(--accent-dim)', border: '1px solid var(--border-accent)',
          }}>
            <Icon size={22} style={{ color: 'var(--accent)' }} />
          </div>
        )}
        <div className="display" style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>{title}</div>
        {sub && <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 16 }}>{sub}</div>}
        {action}
      </div>
    </CenterState>
  )
}

// ── Progress bar ──────────────────────────────────────────────────────────
export function ProgressBar({ value, color = 'var(--accent)', height = 6 }) {
  return (
    <div style={{
      width: '100%', height, borderRadius: height, overflow: 'hidden',
      background: 'var(--border)',
    }}>
      <div style={{
        width: `${Math.max(0, Math.min(100, value))}%`, height: '100%',
        background: color, borderRadius: height,
        transition: 'width 0.4s cubic-bezier(0.4,0,0.2,1)',
      }} />
    </div>
  )
}

// ── Simple toggle switch ──────────────────────────────────────────────────
export function Toggle({ checked, onChange, label }) {
  return (
    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
      <span
        onClick={() => onChange(!checked)}
        style={{
          width: 34, height: 19, borderRadius: 999, position: 'relative',
          background: checked ? 'var(--accent)' : 'var(--border-bright)',
          transition: 'var(--transition)', flexShrink: 0,
        }}
      >
        <span style={{
          position: 'absolute', top: 2.5, left: checked ? 17 : 2.5,
          width: 14, height: 14, borderRadius: '50%',
          background: checked ? 'var(--accent-ink)' : 'var(--text-dim)',
          transition: 'var(--transition)',
        }} />
      </span>
      {label && <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{label}</span>}
    </label>
  )
}

// ── Key-value row for detail panels ──────────────────────────────────────
export function KvRow({ k, v, mono = false }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 12,
      padding: '7px 0', borderBottom: '1px solid var(--border)', fontSize: 12,
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{k}</span>
      <span className={mono ? 'mono' : ''} style={{ color: 'var(--text)', textAlign: 'right' }}>{v}</span>
    </div>
  )
}
