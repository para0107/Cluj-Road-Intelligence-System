import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import MapPage from './pages/MapPage'
import StatsPage from './pages/StatsPage'
import ExplorerPage from './pages/ExplorerPage'
import IngestionPage from './pages/IngestionPage'

// Global keyframe animations injected once
const KEYFRAMES = `
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
`

export default function App() {
  return (
    <>
      <style>{KEYFRAMES}</style>
      <Navbar />
      <Routes>
        <Route path="/"         element={<MapPage />}       />
        <Route path="/stats"    element={<StatsPage />}     />
        <Route path="/explorer" element={<ExplorerPage />}  />
        <Route path="/ingest"   element={<IngestionPage />} />
      </Routes>
    </>
  )
}
