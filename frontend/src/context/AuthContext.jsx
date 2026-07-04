/**
 * frontend/src/context/AuthContext.jsx
 *
 * The only global state in the app: who is logged in.
 * Token + cached profile live in localStorage; the profile is re-validated
 * against GET /auth/me on mount. After login/refresh the browser's (free,
 * permission-based) geolocation is pushed to the backend once so the map can
 * open on the user's city and municipality scoping works.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { authLogin, authRegister, authGoogle, fetchMe, updateMyLocation, TOKEN_KEY, USER_KEY } from '../utils/api'

const AuthContext = createContext(null)

export function useAuth() {
  return useContext(AuthContext)
}

function readCachedUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') } catch { return null }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(readCachedUser)
  const [booting, setBooting] = useState(Boolean(localStorage.getItem(TOKEN_KEY)))

  const persist = useCallback((token, profile) => {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(USER_KEY, JSON.stringify(profile))
    setUser(profile)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setUser(null)
  }, [])

  // Push the current position once per session (fire-and-forget; the user
  // may simply deny the permission and everything still works).
  const shareLocation = useCallback(() => {
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const updated = await updateMyLocation(pos.coords.latitude, pos.coords.longitude)
          localStorage.setItem(USER_KEY, JSON.stringify(updated))
          setUser(updated)
        } catch { /* offline or logged out — ignore */ }
      },
      () => { /* permission denied — fine */ },
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 },
    )
  }, [])

  // Validate the cached session on mount
  useEffect(() => {
    if (!localStorage.getItem(TOKEN_KEY)) { setBooting(false); return }
    let alive = true
    fetchMe()
      .then((profile) => {
        if (!alive) return
        localStorage.setItem(USER_KEY, JSON.stringify(profile))
        setUser(profile)
        shareLocation()
      })
      .catch(() => { if (alive) logout() })
      .finally(() => { if (alive) setBooting(false) })
    return () => { alive = false }
  }, [logout, shareLocation])

  const login = useCallback(async (identifier, password) => {
    const data = await authLogin(identifier, password)
    persist(data.access_token, data.user)
    shareLocation()
    return data.user
  }, [persist, shareLocation])

  // Registration may return an account+token (status "ok"), or an interim
  // status ("verify_email" / "awaiting_approval") with no session yet —
  // only persist when a token actually came back.
  const register = useCallback(async (payload) => {
    const data = await authRegister(payload)
    if (data.access_token && data.user) {
      persist(data.access_token, data.user)
      shareLocation()
    }
    return data
  }, [persist, shareLocation])

  /** Persist a session from any auth outcome that carries a token. */
  const adoptSession = useCallback((data) => {
    if (data?.access_token && data?.user) {
      persist(data.access_token, data.user)
      shareLocation()
    }
    return data?.user || null
  }, [persist, shareLocation])

  const loginWithGoogle = useCallback(async (idToken) => {
    const data = await authGoogle(idToken)
    persist(data.access_token, data.user)
    shareLocation()
    return data.user
  }, [persist, shareLocation])

  const value = {
    user,
    booting,
    isAuthed: Boolean(user),
    isAdmin: user?.role === 'admin',
    isOperator: user?.role === 'admin' || user?.role === 'municipality',
    login,
    loginWithGoogle,
    register,
    adoptSession,
    logout,
    shareLocation,
    setUser: (u) => { localStorage.setItem(USER_KEY, JSON.stringify(u)); setUser(u) },
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
