import React, { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Map, BarChart2, Search, Sun, Moon } from 'lucide-react'

const NAV = [
  { to: '/',          label: 'MAP',       Icon: Map        },
  { to: '/stats',     label: 'STATS',     Icon: BarChart2  },
  { to: '/explorer',  label: 'EXPLORER',  Icon: Search     },
]

export default function Navbar() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark')

  useEffect(() => {
    if (theme === 'light') {
      document.documentElement.classList.add('light')
    } else {
      document.documentElement.classList.remove('light')
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')

  return (
    <nav style={styles.nav}>
      {/* Logo */}
      <div style={styles.logo}>
        <span style={styles.logoAccent}>RIDS</span>
        <span style={styles.logoSub}>Road Infrastructure Detection</span>
      </div>

      {/* Links */}
      <div style={styles.links}>
        {NAV.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              ...styles.link,
              ...(isActive ? styles.linkActive : {}),
            })}
          >
            <Icon size={13} style={{ marginRight: 6 }} />
            {label}
          </NavLink>
        ))}
      </div>

      <div style={styles.rightSection}>
        {/* Theme toggle */}
        <button onClick={toggleTheme} style={styles.themeBtn} title="Toggle light/dark mode">
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>

        {/* Status dot */}
        <div style={styles.status}>
          <span style={styles.dot} />
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>LIVE</span>
        </div>
      </div>
    </nav>
  )
}

const styles = {
  nav: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 48,
    padding: '0 20px',
    background: 'var(--bg-card)',
    borderBottom: '1px solid var(--border)',
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1000,
  },
  logo: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 8,
  },
  logoAccent: {
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    fontSize: 15,
    color: 'var(--accent)',
    letterSpacing: '0.08em',
  },
  logoSub: {
    fontSize: 11,
    color: 'var(--text-muted)',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  },
  links: {
    display: 'flex',
    gap: 4,
  },
  link: {
    display: 'flex',
    alignItems: 'center',
    padding: '5px 12px',
    borderRadius: 'var(--radius)',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.1em',
    color: 'var(--text-muted)',
    textDecoration: 'none',
    transition: 'var(--transition)',
    border: '1px solid transparent',
  },
  linkActive: {
    color: 'var(--accent)',
    background: 'var(--accent-dim)',
    border: '1px solid rgba(232,255,71,0.2)',
  },
  rightSection: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  themeBtn: {
    background: 'transparent',
    border: 'none',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 4,
    borderRadius: 'var(--radius)',
    transition: 'var(--transition)',
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    display: 'inline-block',
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: 'var(--green)',
    boxShadow: '0 0 6px var(--green)',
    animation: 'pulse 2s infinite',
  },
}