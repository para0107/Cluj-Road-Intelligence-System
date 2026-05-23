/**
 * frontend/src/components/Navbar.jsx
 *
 * Top navigation bar — fixed, 48 px tall.
 * Matches the design tokens in index.css exactly:
 *   --bg-card, --border, --accent, --font-mono, --font-sans, etc.
 *
 * Links: MAP · EXPLORER · STATS · UPLOAD
 * The theme toggle (dark/light) is preserved.
 */

import React, { useState, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Map, Table, BarChart2, Upload, Sun, Moon } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/',         label: 'Map',      icon: Map      },
  { to: '/explorer', label: 'Explorer', icon: Table    },
  { to: '/stats',    label: 'Stats',    icon: BarChart2 },
  { to: '/ingest',   label: 'Upload',   icon: Upload   },
]

export default function Navbar() {
  const [dark, setDark] = useState(true)

  // Sync <html> class on mount and on toggle
  useEffect(() => {
    document.documentElement.classList.toggle('light', !dark)
  }, [dark])

  return (
    <nav style={styles.nav}>
      {/* Brand */}
      <div style={styles.brand}>
        <span style={styles.brandDot} />
        <span style={styles.brandName}>RIDS</span>
        <span style={styles.brandSub}>Road Infrastructure Detection System</span>
      </div>

      {/* Links */}
      <div style={styles.links}>
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              ...styles.link,
              color:            isActive ? 'var(--accent)' : 'var(--text-muted)',
              background:       isActive ? 'var(--accent-dim)' : 'transparent',
              border:           isActive ? '1px solid var(--accent)' : '1px solid transparent',
            })}
          >
            <Icon size={13} />
            {label}
            {/* Highlight the Upload link with a subtle badge when on /ingest */}
            {to === '/ingest' && (
              <span style={styles.uploadBadge}>NEW</span>
            )}
          </NavLink>
        ))}
      </div>

      {/* Theme toggle */}
      <button
        style={styles.themeBtn}
        onClick={() => setDark(v => !v)}
        title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {dark ? <Sun size={14} /> : <Moon size={14} />}
      </button>
    </nav>
  )
}

const styles = {
  nav: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    height: 48,
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    padding: '0 20px',
    background: 'rgba(10,12,16,0.96)',
    borderBottom: '1px solid var(--border)',
    backdropFilter: 'blur(12px)',
    gap: 0,
  },

  // Brand
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginRight: 28,
    flexShrink: 0,
  },
  brandDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: 'var(--accent)',
    boxShadow: '0 0 8px var(--accent-glow)',
  },
  brandName: {
    fontFamily: 'var(--font-mono)',
    fontSize: 13,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '.1em',
  },
  brandSub: {
    fontSize: 10,
    color: 'var(--text-muted)',
    letterSpacing: '.04em',
    // hide on very small screens if needed
  },

  // Links
  links: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  link: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '5px 12px',
    borderRadius: 'var(--radius)',
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '.06em',
    textDecoration: 'none',
    transition: 'var(--transition)',
  },
  uploadBadge: {
    fontSize: 8,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '.06em',
    background: 'var(--accent)',
    color: '#0a0c10',
    borderRadius: 3,
    padding: '1px 4px',
    marginLeft: 2,
  },

  // Theme toggle
  themeBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 32,
    height: 32,
    background: 'transparent',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'var(--transition)',
    marginLeft: 'auto',
  },
}
