import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import MapPage from './pages/MapPage'
import StatsPage from './pages/StatsPage'
import ExplorerPage from './pages/ExplorerPage'
import IngestionPage from './pages/IngestionPage'
import PriorityPage from './pages/PriorityPage'
import AboutPage from './pages/AboutPage'

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/"         element={<HomePage />}      />
        <Route path="/map"      element={<MapPage />}       />
        <Route path="/stats"    element={<StatsPage />}     />
        <Route path="/explorer" element={<ExplorerPage />}  />
        <Route path="/priority" element={<PriorityPage />}  />
        <Route path="/ingest"   element={<IngestionPage />} />
        <Route path="/about"    element={<AboutPage />}     />
      </Routes>
    </>
  )
}
