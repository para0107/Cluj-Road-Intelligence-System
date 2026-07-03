/**
 * frontend/src/components/Navbar.jsx
 *
 * Fixed top navigation — var(--nav-h) tall.
 * Brand · nav links · live pipeline indicator · API health dot · clock · theme.
 */

import React, { useState, useEffect, useRef } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Map, Table, BarChart2, Upload, ListOrdered,
  Info, Sun, Moon, Activity, Radio, LogOut, Shield, MapPin, ChevronDown,
} from 'lucide-react'
import { fetchHealth } from '../utils/api'
import { useAuth } from '../context/AuthContext'

const ROLE_COLORS = {
  admin: 'var(--red)',
  municipality: 'var(--cyan)',
  user: 'var(--green)',
}

const NAV_ITEMS = [
  { to: '/',         label: 'Command',  icon: LayoutDashboard },
  { to: '/live',     label: 'Live',     icon: Radio, live: true },
  { to: '/map',      label: 'Map',      icon: Map },
  { to: '/explorer', label: 'Explorer', icon: Table },
  { to: '/stats',    label: 'Stats',    icon: BarChart2 },
  { to: '/priority', label: 'Repairs',  icon: ListOrdered },
  { to: '/ingest',   label: 'Upload',   icon: Upload },
  { to: '/about',    label: 'System',   icon: Info },
]

// Road-marking logo mark
function LogoMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--bg-card2)" stroke="var(--border-bright)" />
      <path d="M10 28 L14 4 h4 L22 28 h-4 l-.6-5 h-2.8 L14 28 Z" fill="var(--accent)" opacity=".14" />
      <path d="M15.2 6 h1.6 v4 h-1.6 Z M15 13 h2 v4 h-2 Z M14.8 20 h2.4 v4 h-2.4 Z" fill="var(--accent)" />
    </svg>
  )
}

export default function Navbar() {
  const { user, isAuthed, isAdmin, logout, shareLocation } = useAuth()
  const navigate = useNavigate()
  const [dark, setDark] = useState(() => localStorage.getItem('rids_theme') !== 'light')
  const [now, setNow] = useState(new Date())
  const [health, setHealth] = useState(null)          // null | 'ok' | 'down'
  const [jobActive, setJobActive] = useState(false)   // a pipeline run is in flight
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)

  // Close the user menu on outside click
  useEffect(() => {
    const onClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  // Theme sync
  useEffect(() => {
    document.documentElement.classList.toggle('light', !dark)
    localStorage.setItem('rids_theme', dark ? 'dark' : 'light')
  }, [dark])

  // Clock
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  // Health probe + active job flag (cheap, 30 s cadence)
  useEffect(() => {
    let alive = true
    const probe = async () => {
      const h = await fetchHealth()
      if (alive) setHealth(h.status === 'ok' ? 'ok' : 'down')
      if (alive) setJobActive(Boolean(localStorage.getItem('rids_active_job')))
    }
    probe()
    const t = setInterval(probe, 30_000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const healthColor = health === 'ok' ? 'var(--green)' : health === 'down' ? 'var(--red)' : 'var(--text-muted)'

  return (
    <nav style={styles.nav}>
      {/* Brand */}
      <NavLink to="/" style={styles.brand}>
        <LogoMark />
        <div style={{ lineHeight: 1.15 }}>
          <div className="display" style={styles.brandName}>RIDS</div>
          <div style={styles.brandSub}>CLUJ-NAPOCA · ROAD INTELLIGENCE</div>
        </div>
      </NavLink>

      {/* Links (only when signed in) */}
      <div style={styles.links}>
        {isAuthed && NAV_ITEMS.map(({ to, label, icon: Icon, live }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            <Icon size={13} />
            {label}
            {live && (
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'var(--red)', boxShadow: '0 0 6px var(--red)',
                animation: 'pulse 1.6s ease-in-out infinite',
              }} />
            )}
          </NavLink>
        ))}
      </div>

      {/* Right cluster */}
      <div style={styles.right}>
        {jobActive && (
          <span style={styles.jobBadge} title="A pipeline run is in progress">
            <Activity size={11} style={{ animation: 'pulse 1.4s ease-in-out infinite' }} />
            PIPELINE
          </span>
        )}

        <span style={styles.clock} className="mono" title="Local time — Cluj-Napoca">
          {now.toLocaleTimeString('en-GB')}
        </span>

        <span style={styles.health} title={`API ${health === 'ok' ? 'online' : health === 'down' ? 'offline' : 'checking…'}`}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', background: healthColor,
            boxShadow: `0 0 8px ${healthColor}`,
            animation: health === 'ok' ? 'none' : 'pulse 1.4s ease-in-out infinite',
          }} />
          <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>API</span>
        </span>

        <button
          className="btn btn-ghost btn-sm"
          style={{ width: 32, height: 32, padding: 0 }}
          onClick={() => setDark(v => !v)}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark ? <Sun size={14} /> : <Moon size={14} />}
        </button>

        {/* User menu */}
        {isAuthed ? (
          <div style={{ position: 'relative' }} ref={menuRef}>
            <button className="btn btn-ghost btn-sm" style={styles.userBtn} onClick={() => setMenuOpen(v => !v)}>
              <span style={{ ...styles.avatar, borderColor: `${ROLE_COLORS[user.role]}66`, color: ROLE_COLORS[user.role] }}>
                {(user.username || '?')[0].toUpperCase()}
              </span>
              <span style={{ fontSize: 12, fontWeight: 600, maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {user.username}
              </span>
              <ChevronDown size={11} style={{ color: 'var(--text-muted)' }} />
            </button>

            {menuOpen && (
              <div className="glass anim-fade-in" style={styles.userMenu}>
                <div style={styles.menuHeader}>
                  <div style={{ fontWeight: 700, fontSize: 13 }}>{user.full_name || user.username}</div>
                  <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 2 }}>{user.email}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
                    <span className="mono" style={{
                      fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
                      color: ROLE_COLORS[user.role], border: `1px solid ${ROLE_COLORS[user.role]}55`,
                      borderRadius: 4, padding: '1px 6px',
                    }}>
                      {user.role}
                    </span>
                    {user.city && (
                      <span style={{ fontSize: 10.5, color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 3 }}>
                        <MapPin size={9} /> {user.city}
                      </span>
                    )}
                  </div>
                </div>
                <button className="table-row-hover" style={styles.menuItem}
                  onClick={() => { shareLocation(); setMenuOpen(false) }}>
                  <MapPin size={12} style={{ color: 'var(--cyan)' }} /> Update my location
                </button>
                {isAdmin && (
                  <button className="table-row-hover" style={styles.menuItem}
                    onClick={() => { navigate('/admin'); setMenuOpen(false) }}>
                    <Shield size={12} style={{ color: 'var(--red)' }} /> Manage accounts
                  </button>
                )}
                <button className="table-row-hover" style={{ ...styles.menuItem, borderTop: '1px solid var(--border)' }}
                  onClick={() => { logout(); setMenuOpen(false); navigate('/login') }}>
                  <LogOut size={12} style={{ color: 'var(--text-muted)' }} /> Sign out
                </button>
              </div>
            )}
          </div>
        ) : (
          <NavLink to="/login" className="btn btn-accent btn-sm">Sign in</NavLink>
        )}
      </div>
    </nav>
  )
}

const styles = {
  nav: {
    position: 'fixed',
    top: 0, left: 0, right: 0,
    height: 'var(--nav-h)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    padding: '0 18px',
    background: 'var(--bg-glass)',
    borderBottom: '1px solid var(--border)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    gap: 12,
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginRight: 20,
    flexShrink: 0,
    textDecoration: 'none',
  },
  brandName: {
    fontSize: 15,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '0.12em',
  },
  brandSub: {
    fontSize: 8.5,
    color: 'var(--text-muted)',
    letterSpacing: '0.14em',
    fontFamily: 'var(--font-mono)',
  },
  links: {
    display: 'flex',
    alignItems: 'center',
    gap: 3,
    flex: 1,
    overflowX: 'auto',
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    flexShrink: 0,
    marginLeft: 'auto',
  },
  jobBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    padding: '4px 10px',
    borderRadius: 999,
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    color: 'var(--accent)',
    fontSize: 9.5,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '0.1em',
  },
  clock: {
    fontSize: 11.5,
    color: 'var(--text-dim)',
    letterSpacing: '0.04em',
  },
  health: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
  },
  userBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    padding: '3px 8px 3px 4px',
    height: 34,
    border: '1px solid var(--border)',
  },
  avatar: {
    width: 24, height: 24, borderRadius: 7, flexShrink: 0,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    border: '1px solid', background: 'var(--bg-card2)',
    fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-display)',
  },
  userMenu: {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    width: 230,
    zIndex: 1100,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  menuHeader: {
    padding: '12px 14px',
    borderBottom: '1px solid var(--border)',
  },
  menuItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 9,
    width: '100%',
    padding: '10px 14px',
    background: 'transparent',
    border: 'none',
    color: 'var(--text-dim)',
    fontSize: 12,
    cursor: 'pointer',
    textAlign: 'left',
  },
}
