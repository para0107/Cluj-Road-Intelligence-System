import axios from 'axios'

// When the frontend is hosted separately from the backend (e.g. Vercel +
// Oracle VM), set VITE_API_URL to the backend origin at BUILD time, e.g.
//   VITE_API_URL=https://rids-api.example.duckdns.org
// Unset (default) keeps same-origin relative /api — the nginx/Vite proxy path.
export const API_ORIGIN = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')

const api = axios.create({
  baseURL: `${API_ORIGIN}/api`,
  timeout: 15000,
})

// ── Auth wiring ────────────────────────────────────────────────────────────
// Every request carries the JWT; a 401 clears the session and returns the
// user to /login (except on the auth endpoints themselves, where 401 is a
// normal "wrong password" answer the form handles).

export const TOKEN_KEY = 'rids_token'
export const USER_KEY = 'rids_user'

api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (resp) => resp,
  (error) => {
    const status = error?.response?.status
    const url = error?.config?.url || ''
    if (status === 401 && !url.startsWith('/auth/')) {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
      if (!window.location.pathname.startsWith('/login')) {
        window.location.assign('/login')
      }
    }
    return Promise.reject(error)
  },
)

// ── Auth endpoints ─────────────────────────────────────────────────────────

export const authLogin = (identifier, password, extra = {}) =>
  api.post('/auth/login', { identifier, password, ...extra }).then(r => r.data)

/** A fresh proof-of-work challenge for the ALTCHA widget (public endpoint). */
export const fetchCaptchaChallenge = () =>
  api.get('/auth/captcha/challenge').then(r => r.data)

export const authRegister = (payload) =>
  api.post('/auth/register', payload).then(r => r.data)

export const authVerifyEmail = (email, code) =>
  api.post('/auth/verify-email', { email, code }).then(r => r.data)

export const authResendCode = (email) =>
  api.post('/auth/resend-code', { email }).then(r => r.data)

export const deleteMyAccount = (password = null) =>
  api.delete('/auth/me', { data: { password } }).then(r => r.data)

export const adminDeleteUser = (userId) =>
  api.delete(`/auth/users/${userId}`).then(r => r.data)

export const adminSetActive = (userId, isActive) =>
  api.patch(`/auth/users/${userId}/active`, { is_active: isActive }).then(r => r.data)

export const fetchPendingRegistrations = () =>
  api.get('/auth/registrations/pending').then(r => r.data)

export const approveRegistration = (id) =>
  api.post(`/auth/registrations/${id}/approve`).then(r => r.data)

export const denyRegistration = (id) =>
  api.post(`/auth/registrations/${id}/deny`).then(r => r.data)

export const authGoogle = (idToken) =>
  api.post('/auth/oauth/google', { id_token: idToken }).then(r => r.data)

export const fetchAuthConfig = () =>
  api.get('/auth/config').then(r => r.data)

export const fetchMe = () =>
  api.get('/auth/me').then(r => r.data)

export const updateMyLocation = (latitude, longitude, city = null) =>
  api.patch('/auth/me/location', { latitude, longitude, city }).then(r => r.data)

export const fetchUsers = () =>
  api.get('/auth/users').then(r => r.data)

export const setUserRole = (userId, role, city = null) =>
  api.patch(`/auth/users/${userId}/role`, { role, city }).then(r => r.data)

export const updateMyProfile = (payload) =>
  api.patch('/auth/me', payload).then(r => r.data)

// ── City landmarks + centre (free OSM/Nominatim, cached server-side) ───────

export const fetchCityLandmarks = (city, refresh = false) =>
  api.get('/cities/landmarks', {
    params: { city, refresh },
    timeout: 30000,   // first lookup per city is rate-limited by design (~10 s)
  }).then(r => r.data)

export const fetchCityCenter = (city, refresh = false) =>
  api.get('/cities/center', {
    params: { city, refresh },
    timeout: 15000,   // single geocode; first call per city does one OSM query
  }).then(r => r.data)

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

// ── Engagement: points, badges, leaderboard, notifications ────────────────

export const fetchMyImpact = () =>
  api.get('/engagement/me').then(r => r.data)

export const fetchLeaderboard = (city = null, limit = 50) =>
  api.get('/engagement/leaderboard', { params: { city: city || undefined, limit } })
     .then(r => r.data)

export const fetchNotifications = (page = 1, pageSize = 20) =>
  api.get('/notifications', { params: { page, page_size: pageSize } }).then(r => r.data)

export const markNotificationsRead = (ids = null, all = false) =>
  api.post('/notifications/read', { ids, all }).then(r => r.data)

// ── Triage (operator): citizen live events → official detections ──────────

export const fetchTriage = (params = {}) =>
  api.get('/live/triage', { params }).then(r => r.data)

export const promoteEvent = (eventId) =>
  api.post(`/live/events/${eventId}/promote`).then(r => r.data)

export const dismissEvent = (eventId) =>
  api.post(`/live/events/${eventId}/dismiss`).then(r => r.data)

// ── Work orders (operator) ─────────────────────────────────────────────────

export const fetchWorkOrders = (params = {}) =>
  api.get('/work-orders', { params }).then(r => r.data)

export const fetchWorkOrder = (id) =>
  api.get(`/work-orders/${id}`).then(r => r.data)

export const createWorkOrder = (payload) =>
  api.post('/work-orders', payload).then(r => r.data)

export const updateWorkOrder = (id, payload) =>
  api.patch(`/work-orders/${id}`, payload).then(r => r.data)

export const editWorkOrderItems = (id, addIds = [], removeIds = []) =>
  api.post(`/work-orders/${id}/items`, { add_ids: addIds, remove_ids: removeIds })
     .then(r => r.data)

export const deleteWorkOrder = (id) =>
  api.delete(`/work-orders/${id}`).then(r => r.data)

// ── Operations analytics (operator) ────────────────────────────────────────

export const fetchOpsAnalytics = () =>
  api.get('/analytics/ops').then(r => r.data)

// ── Road Quality Index ─────────────────────────────────────────────────────

export const fetchQualityGrid = (bbox, cellM = 120) =>
  api.get('/quality/grid', { params: { bbox, cell_m: cellM } }).then(r => r.data)

/** Trigger a browser download of the quality grid export (operator). */
export const downloadQualityExport = (format = 'csv', bbox = null) => {
  const params = new URLSearchParams({ format })
  if (bbox) params.set('bbox', bbox)
  window.open(`${API_ORIGIN}/api/quality/export?${params.toString()}`, '_blank')
}

// ── Developer API keys ─────────────────────────────────────────────────────

export const fetchApiKeys = () =>
  api.get('/apikeys').then(r => r.data)

export const createApiKey = (name) =>
  api.post('/apikeys', { name }).then(r => r.data)

export const revokeApiKey = (id) =>
  api.delete(`/apikeys/${id}`).then(r => r.data)

// ── Contact sales (public) ─────────────────────────────────────────────────

export const contactSales = (payload) =>
  api.post('/contact/sales', payload).then(r => r.data)

// ── Evidence media ─────────────────────────────────────────────────────────

/**
 * Fetch a detection's evidence photo as an object URL (an <img> tag cannot
 * send the Bearer header, so the JPG comes through axios as a blob).
 * Returns null when the detection has no evidence (404).
 * Caller must URL.revokeObjectURL() the result when done.
 */
export const fetchEvidenceUrl = (detectionId) =>
  api.get(`/media/evidence/${detectionId}`, { responseType: 'blob' })
    .then(r => URL.createObjectURL(r.data))
    .catch(() => null)

// ── Health ────────────────────────────────────────────────────────────────

/**
 * Backend + DB health probe.
 * Uses the public /api/health route (proxied by nginx) so the navbar dot
 * works on the login page too, before any session exists.
 */
export const fetchHealth = () =>
  api.get('/health', { timeout: 5000 })
    .then(r => ({ status: r.data?.status === 'ok' ? 'ok' : 'down', database: r.data?.database }))
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
