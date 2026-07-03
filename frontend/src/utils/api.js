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

export const updateDetectionStatus = (id, isFixed) =>
  api.patch(`/detections/${id}/status`, { is_fixed: isFixed }).then(r => r.data)

export const deleteDetectionsBulk = (ids, deleteSurveyLog = false) =>
  api.delete('/detections/bulk', {
    data: {
      ids,
      delete_survey_log: deleteSurveyLog,
    },
  }).then(r => r.data)

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

// ── Export ────────────────────────────────────────────────────────────────

/** Trigger a browser download of the full CSV export. */
export const downloadCsv = () => {
  window.open('/api/export/csv', '_blank')
}

// ── Health ────────────────────────────────────────────────────────────────

/**
 * Backend + DB health probe.
 * The bare `/health` route is NOT proxied by nginx (only `/api/` is), so we
 * probe a cheap real API route: 200 → API + DB reachable.
 */
export const fetchHealth = () =>
  api.get('/stats', { timeout: 5000 })
    .then(() => ({ status: 'ok', database: 'connected' }))
    .catch((e) => ({
      status: 'down',
      database: e?.response ? 'unreachable' : 'api offline',
    }))

// ── Ingest ────────────────────────────────────────────────────────────────

/**
 * Upload a survey video and optional GPS file.
 * Uses FormData / multipart — no JSON serialization.
 *
 * @param {File}      videoFile  - .mp4 dashcam footage
 * @param {File|null} gpsFile    - .gpx GPS log (optional, pass null to omit)
 * @param {function}  onProgress - (percent: number) => void  upload progress callback
 * @returns {Promise<{job_id: string, status: string, message: string}>}
 */
export const uploadSurvey = (videoFile, gpsFile, onProgress) => {
  const form = new FormData()
  form.append('video', videoFile)
  if (gpsFile) {
    form.append('gps', gpsFile)
  }
  return api.post('/ingest/upload', form, {
    // Do NOT set Content-Type manually — axios sets multipart/form-data + boundary
    timeout: 120_000,   // large files need more time to upload
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100))
      }
    },
  }).then(r => r.data)
}

/**
 * Poll the status of a running or completed pipeline job.
 * Call this on an interval while status === 'running' | 'initialising'.
 *
 * @param {string} jobId
 * @returns {Promise<{job_id, status, stages, n_frames, n_detections, n_inserted, n_updated, error_message, started_at, finished_at}>}
 */
export const fetchJobStatus = (jobId) =>
  api.get(`/ingest/status/${jobId}`).then(r => r.data)

export default api
