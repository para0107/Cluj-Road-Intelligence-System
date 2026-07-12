/**
 * frontend/src/components/NotificationsBell.jsx
 *
 * In-app notification bell for the navbar. Polls GET /notifications every
 * 60 s, shows an unread count, and opens a dropdown panel styled like the
 * Navbar user menu. Clicking a notification marks it read and follows its
 * link. Renders nothing for signed-out visitors.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Bell, Award, Medal, Wrench, BadgeCheck, Info, CheckCheck } from 'lucide-react'
import { fetchNotifications, markNotificationsRead } from '../utils/api'
import { useAuth } from '../context/AuthContext'

const POLL_MS = 60_000

// Icon + colour per notification kind (anything unknown falls back to Info).
const KIND_ICONS = {
  points:   { icon: Award,      color: 'var(--accent)' },
  badge:    { icon: Medal,      color: 'var(--yellow)' },
  fixed:    { icon: Wrench,     color: 'var(--green)' },
  promoted: { icon: BadgeCheck, color: 'var(--cyan)' },
}
const FALLBACK_ICON = { icon: Info, color: 'var(--text-muted)' }

/** "just now" · "42 min ago" · "3 h ago" · "5 d ago" · "12 Mar 2026" */
function relTime(iso) {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (secs < 60) return 'just now'
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins} min ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days} d ago`
  const weeks = Math.floor(days / 7)
  if (weeks < 5) return `${weeks} w ago`
  return new Date(then).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
}

export default function NotificationsBell() {
  const { isAuthed } = useAuth()
  const navigate = useNavigate()

  const [open, setOpen] = useState(false)
  const [unread, setUnread] = useState(0)
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)

  const wrapRef = useRef(null)
  const aliveRef = useRef(true)

  const load = useCallback(async () => {
    try {
      const data = await fetchNotifications(1, 20)
      if (!aliveRef.current) return
      setUnread(data?.unread || 0)
      setItems(Array.isArray(data?.items) ? data.items : [])
    } catch {
      // The bell must never break the navbar — a failed poll just keeps
      // the last known list until the next one succeeds.
    } finally {
      if (aliveRef.current) setLoading(false)
    }
  }, [])

  // Poll on mount, then every minute (only while signed in)
  useEffect(() => {
    aliveRef.current = true
    if (!isAuthed) {
      setItems([])
      setUnread(0)
      return () => { aliveRef.current = false }
    }
    load()
    const t = setInterval(load, POLL_MS)
    return () => { aliveRef.current = false; clearInterval(t) }
  }, [isAuthed, load])

  // Close the panel on outside click (same pattern as the Navbar user menu)
  useEffect(() => {
    const onClick = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  const markAll = async () => {
    setItems(prev => prev.map(n => (n.read_at ? n : { ...n, read_at: new Date().toISOString() })))
    setUnread(0)
    try {
      await markNotificationsRead(null, true)
    } catch {
      /* ignore — the next poll re-syncs the true state */
    }
    load()
  }

  const openItem = async (n) => {
    setOpen(false)
    if (!n.read_at) {
      setItems(prev => prev.map(x => (
        x.id === n.id ? { ...x, read_at: new Date().toISOString() } : x
      )))
      setUnread(u => Math.max(0, u - 1))
      try {
        await markNotificationsRead([n.id])
      } catch {
        /* ignore — the next poll re-syncs the true state */
      }
    }
    if (n.link) navigate(n.link)
  }

  if (!isAuthed) return null

  const badgeText = unread > 9 ? '9+' : String(unread)

  return (
    <div style={{ position: 'relative' }} ref={wrapRef}>
      <button
        className="btn btn-ghost btn-sm"
        style={{ width: 32, height: 32, padding: 0, position: 'relative' }}
        onClick={() => setOpen(v => !v)}
        title={unread > 0 ? `${unread} unread notification${unread === 1 ? '' : 's'}` : 'Notifications'}
        aria-label="Notifications"
      >
        <Bell size={14} />
        {unread > 0 && <span style={styles.badge}>{badgeText}</span>}
      </button>

      {open && (
        <div className="glass anim-fade-in" style={styles.panel}>
          <div style={styles.panelHeader}>
            <span className="display" style={{ fontWeight: 700, fontSize: 13 }}>Notifications</span>
            {unread > 0 && (
              <button onClick={markAll} style={styles.markAll}>
                <CheckCheck size={11} /> Mark all read
              </button>
            )}
          </div>

          {loading && items.length === 0 ? (
            <div style={styles.emptyLine}>Loading your notifications…</div>
          ) : items.length === 0 ? (
            <div style={styles.emptyLine}>Nothing new yet. Reports you send will show up here.</div>
          ) : (
            items.map((n) => {
              const { icon: Icon, color } = KIND_ICONS[n.kind] || FALLBACK_ICON
              const isUnread = !n.read_at
              return (
                <button
                  key={n.id}
                  className="table-row-hover"
                  onClick={() => openItem(n)}
                  style={{
                    ...styles.item,
                    background: isUnread ? 'var(--accent-dim)' : 'transparent',
                    cursor: n.link ? 'pointer' : 'default',
                  }}
                >
                  <span style={{ ...styles.itemIcon, color, background: `color-mix(in srgb, ${color} 14%, transparent)` }}>
                    <Icon size={13} />
                  </span>

                  <span style={{ flex: 1, minWidth: 0 }}>
                    <span style={styles.itemTitle}>{n.title}</span>
                    {n.body && <span style={styles.itemBody}>{n.body}</span>}
                    <span className="mono" style={styles.itemTime}>{relTime(n.created_at)}</span>
                  </span>

                  {isUnread && <span style={styles.unreadDot} />}
                </button>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}

const styles = {
  badge: {
    position: 'absolute',
    top: -5,
    right: -5,
    minWidth: 15,
    height: 15,
    padding: '0 4px',
    borderRadius: 999,
    background: 'var(--accent)',
    color: 'var(--accent-ink)',
    border: '1px solid var(--bg)',
    fontFamily: 'var(--font-mono)',
    fontSize: 9,
    fontWeight: 700,
    lineHeight: '13px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  panel: {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    width: 320,
    maxWidth: 'calc(100vw - 24px)',
    maxHeight: 420,
    overflowY: 'auto',
    zIndex: 1100,
    display: 'flex',
    flexDirection: 'column',
    overflowX: 'hidden',
  },
  panelHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    padding: '11px 14px',
    borderBottom: '1px solid var(--border)',
    position: 'sticky',
    top: 0,
    background: 'var(--bg-glass)',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    zIndex: 1,
  },
  markAll: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: 'transparent',
    border: 'none',
    color: 'var(--accent)',
    fontFamily: 'var(--font-sans)',
    fontSize: 11,
    fontWeight: 600,
    cursor: 'pointer',
    padding: 0,
  },
  item: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 10,
    width: '100%',
    padding: '10px 14px',
    border: 'none',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text)',
    textAlign: 'left',
    font: 'inherit',
  },
  itemIcon: {
    width: 26,
    height: 26,
    borderRadius: 8,
    flexShrink: 0,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
    marginTop: 1,
  },
  itemTitle: {
    display: 'block',
    fontSize: 12.5,
    fontWeight: 700,
    color: 'var(--text)',
    lineHeight: 1.35,
  },
  itemBody: {
    display: 'block',
    fontSize: 12,
    color: 'var(--text-dim)',
    lineHeight: 1.4,
    marginTop: 2,
  },
  itemTime: {
    display: 'block',
    fontSize: 10,
    color: 'var(--text-muted)',
    marginTop: 4,
  },
  unreadDot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: 'var(--accent)',
    boxShadow: '0 0 6px var(--accent-glow)',
    flexShrink: 0,
    marginTop: 6,
  },
  emptyLine: {
    padding: '26px 20px',
    textAlign: 'center',
    fontSize: 12,
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },
}
