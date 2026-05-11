import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

// ── Detections ─────────────────────────────────────────────────────────────

export const fetchDetections = (params = {}) =>
  api.get('/detections', { params }).then(r => r.data)

export const fetchDetectionById = (id) =>
  api.get(`/detections/${id}`).then(r => r.data)

export const fetchNearby = (lat, lon, radius_m = 200, limit = 50) =>
  api.get('/detections/nearby', { params: { latitude: lat, longitude: lon, radius_m, limit } })
     .then(r => r.data)

// ── Stats ─────────────────────────────────────────────────────────────────

export const fetchStats = () =>
  api.get('/stats').then(r => r.data)

// ── Heatmap ───────────────────────────────────────────────────────────────

export const fetchHeatmap = () =>
  api.get('/heatmap').then(r => r.data)

// ── Priority ──────────────────────────────────────────────────────────────

export const fetchPriority = (limit = 50) =>
  api.get('/priority-list', { params: { limit } }).then(r => r.data)

export default api