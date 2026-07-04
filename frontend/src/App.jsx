import React from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import MapPage from './pages/MapPage'
import StatsPage from './pages/StatsPage'
import ExplorerPage from './pages/ExplorerPage'
import IngestionPage from './pages/IngestionPage'
import PriorityPage from './pages/PriorityPage'
import AboutPage from './pages/AboutPage'
import LivePage from './pages/LivePage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import AdminPage from './pages/AdminPage'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Spinner, CenterState } from './components/ui'

/** Everything except /login and /register requires a session. */
function RequireAuth({ children }) {
  const { isAuthed, booting } = useAuth()
  const location = useLocation()
  if (booting) {
    return (
      <div style={{ paddingTop: 'var(--nav-h)', height: '100%' }}>
        <CenterState><Spinner label="Restoring session…" /></CenterState>
      </div>
    )
  }
  if (!isAuthed) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  return children
}

/**
 * Survey/analytics pages (Map, Explorer, Stats, Repairs, Upload) are for
 * municipality operators and admins only; citizens are sent back to Command.
 * The backend enforces the same rule on the underlying endpoints.
 */
function RequireOperator({ children }) {
  const { isOperator } = useAuth()
  if (!isOperator) return <Navigate to="/" replace />
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <Navbar />
      <Routes>
        <Route path="/login"    element={<LoginPage />}    />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/"         element={<RequireAuth><HomePage /></RequireAuth>}      />
        <Route path="/live"     element={<RequireAuth><LivePage /></RequireAuth>}      />
        <Route path="/map"      element={<RequireAuth><RequireOperator><MapPage /></RequireOperator></RequireAuth>}       />
        <Route path="/stats"    element={<RequireAuth><RequireOperator><StatsPage /></RequireOperator></RequireAuth>}     />
        <Route path="/explorer" element={<RequireAuth><RequireOperator><ExplorerPage /></RequireOperator></RequireAuth>}  />
        <Route path="/priority" element={<RequireAuth><RequireOperator><PriorityPage /></RequireOperator></RequireAuth>}  />
        <Route path="/ingest"   element={<RequireAuth><RequireOperator><IngestionPage /></RequireOperator></RequireAuth>} />
        <Route path="/about"    element={<RequireAuth><AboutPage /></RequireAuth>}     />
        <Route path="/admin"    element={<RequireAuth><AdminPage /></RequireAuth>}     />
        <Route path="*"         element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}
