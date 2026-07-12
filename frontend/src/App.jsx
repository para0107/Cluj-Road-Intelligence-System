import React, { Suspense, lazy } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Navbar from './components/Navbar'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Spinner, CenterState } from './components/ui'
import CityGate from './components/CityGate'
import OnboardingTour from './components/OnboardingTour'

// Every page is a lazy chunk: the shell (navbar + auth) loads instantly and a
// user only downloads the code for pages they actually open. Vite splits the
// heavy vendors (leaflet, recharts, animation libs) into their own chunks too,
// and the assistant's model runtimes are dynamic imports on top of that.
const HomePage       = lazy(() => import('./pages/HomePage'))
const MapPage        = lazy(() => import('./pages/MapPage'))
const StatsPage      = lazy(() => import('./pages/StatsPage'))
const ExplorerPage   = lazy(() => import('./pages/ExplorerPage'))
const IngestionPage  = lazy(() => import('./pages/IngestionPage'))
const PriorityPage   = lazy(() => import('./pages/PriorityPage'))
const AboutPage      = lazy(() => import('./pages/AboutPage'))
const LivePage       = lazy(() => import('./pages/LivePage'))
const LoginPage      = lazy(() => import('./pages/LoginPage'))
const RegisterPage   = lazy(() => import('./pages/RegisterPage'))
const AdminPage      = lazy(() => import('./pages/AdminPage'))
const ImpactPage     = lazy(() => import('./pages/ImpactPage'))
const TriagePage     = lazy(() => import('./pages/TriagePage'))
const WorkOrdersPage = lazy(() => import('./pages/WorkOrdersPage'))
const QualityPage    = lazy(() => import('./pages/QualityPage'))
const AssistantPage  = lazy(() => import('./pages/AssistantPage'))
const PricingPage    = lazy(() => import('./pages/PricingPage'))
const DevelopersPage = lazy(() => import('./pages/DevelopersPage'))

/**
 * Everything except the public pages requires a session. Accounts without a
 * city (Google first login, legacy rows) must pick one before using the app:
 * the maps and municipality scoping depend on it.
 */
function RequireAuth({ children }) {
  const { isAuthed, booting, user } = useAuth()
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
  if (!user?.city) {
    return (
      <>
        {children}
        <CityGate />
      </>
    )
  }
  return (
    <>
      {children}
      <OnboardingTour />
    </>
  )
}

/**
 * Survey and operations pages (Map, Explorer, Stats, Repairs, Upload, Triage,
 * Work orders, Quality) are for municipality operators and admins only;
 * citizens are sent back to Command. The backend enforces the same rule on the
 * underlying endpoints.
 */
function RequireOperator({ children }) {
  const { isOperator } = useAuth()
  if (!isOperator) return <Navigate to="/" replace />
  return children
}

function RouteFallback() {
  return (
    <div style={{ paddingTop: 'var(--nav-h)', height: '100%' }}>
      <CenterState><Spinner label="Loading…" /></CenterState>
    </div>
  )
}

const operatorRoute = (element) => (
  <RequireAuth><RequireOperator>{element}</RequireOperator></RequireAuth>
)

export default function App() {
  return (
    <AuthProvider>
      <Navbar />
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          {/* Public */}
          <Route path="/login"      element={<LoginPage />} />
          <Route path="/register"   element={<RegisterPage />} />
          <Route path="/pricing"    element={<PricingPage />} />
          <Route path="/developers" element={<DevelopersPage />} />

          {/* Any signed-in role */}
          <Route path="/"          element={<RequireAuth><HomePage /></RequireAuth>} />
          <Route path="/live"      element={<RequireAuth><LivePage /></RequireAuth>} />
          <Route path="/impact"    element={<RequireAuth><ImpactPage /></RequireAuth>} />
          <Route path="/assistant" element={<RequireAuth><AssistantPage /></RequireAuth>} />
          <Route path="/about"     element={<RequireAuth><AboutPage /></RequireAuth>} />
          <Route path="/admin"     element={<RequireAuth><AdminPage /></RequireAuth>} />

          {/* Operator only */}
          <Route path="/map"        element={operatorRoute(<MapPage />)} />
          <Route path="/stats"      element={operatorRoute(<StatsPage />)} />
          <Route path="/explorer"   element={operatorRoute(<ExplorerPage />)} />
          <Route path="/priority"   element={operatorRoute(<PriorityPage />)} />
          <Route path="/ingest"     element={operatorRoute(<IngestionPage />)} />
          <Route path="/triage"     element={operatorRoute(<TriagePage />)} />
          <Route path="/workorders" element={operatorRoute(<WorkOrdersPage />)} />
          <Route path="/quality"    element={operatorRoute(<QualityPage />)} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AuthProvider>
  )
}
