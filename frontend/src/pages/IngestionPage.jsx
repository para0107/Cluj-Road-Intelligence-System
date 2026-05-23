/**
 * frontend/src/pages/IngestionPage.jsx
 *
 * Survey ingestion page for RIDS.
 * Allows the operator to upload a dashcam .mp4 and an optional .gpx GPS log,
 * then tracks the pipeline execution stage-by-stage via polling.
 *
 * Architecture:
 *   1. Upload phase  — POST /api/ingest/upload (multipart, with progress bar)
 *   2. Polling phase — GET  /api/ingest/status/{job_id}  every POLL_INTERVAL_MS
 *      Polling stops automatically when status becomes "complete" or "failed".
 *
 * No WebSocket is needed: the orchestrator writes session.json after every
 * stage, and the status endpoint reads it. 10-second polling is fine for a
 * pipeline that takes ~60 minutes end-to-end.
 */

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Upload, FileVideo, FileText as FileGps,
  CheckCircle, XCircle, Loader, AlertTriangle, Map, RefreshCw,
  ChevronRight, Clock,
} from 'lucide-react'
import { uploadSurvey, fetchJobStatus } from '../utils/api'

// ── Constants ─────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 10_000   // poll every 10 s while pipeline is running

// Human-readable stage names matching orchestrator stage keys
const STAGE_META = {
  preprocessor:        { label: 'Preprocessor',        sub: 'Frame extraction · GPS sync · Lighting' },
  detector:            { label: 'RT-DETR Detector',     sub: 'RT-DETR-L inference · Confidence filter' },
  segmentor:           { label: 'SAM Segmentor',        sub: 'SAM 2.1 Tiny · 4 geometry features' },
  depth_estimator:     { label: 'Depth Estimator',      sub: 'Monodepth2 · Relative disparity' },
  severity_classifier: { label: 'Severity Classifier',  sub: 'Rule-based S1–S5 · Weighted multi-signal' },
  deduplicator:        { label: 'Deduplicator',         sub: 'DBSCAN · Haversine · 2 m radius' },
  db_writer:           { label: 'DB Writer',            sub: 'PostGIS upsert · Priority score update' },
}

// Ordered stage keys — used to render the pipeline track in order
const STAGE_ORDER = [
  'preprocessor', 'detector', 'segmentor',
  'depth_estimator', 'severity_classifier', 'deduplicator', 'db_writer',
]

// ── Helper: format elapsed seconds ───────────────────────────────────────

function fmtElapsed(s) {
  if (s == null || s === 0) return null
  if (s < 60)  return `${s.toFixed(1)} s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  return `${m}m ${rem}s`
}

// ── Helper: format ISO timestamp to locale string ─────────────────────────

function fmtTs(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleTimeString() } catch { return iso }
}

// ── File drop zone ────────────────────────────────────────────────────────

function DropZone({ label, accept, file, onFile, icon: Icon, color, hint }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  const handleChange = (e) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        ...styles.dropZone,
        borderColor: dragOver
          ? color
          : file
            ? `${color}80`
            : 'var(--border-bright)',
        background: dragOver
          ? `${color}10`
          : file
            ? `${color}08`
            : 'var(--bg-card2)',
        cursor: 'pointer',
        transition: 'var(--transition)',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={handleChange}
      />
      <div style={{ ...styles.dropIcon, color: file ? color : 'var(--text-muted)', background: file ? `${color}18` : 'var(--border)' }}>
        <Icon size={20} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: file ? 'var(--text)' : 'var(--text-muted)' }}>
          {label}
        </div>
        {file ? (
          <div style={{ fontSize: 11, color, fontFamily: 'var(--font-mono)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
          </div>
        ) : (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{hint}</div>
        )}
      </div>
      {file && (
        <CheckCircle size={16} style={{ color, flexShrink: 0 }} />
      )}
    </div>
  )
}

// ── Stage row ─────────────────────────────────────────────────────────────

function StageRow({ name, stageData, index, totalCompleted }) {
  const meta = STAGE_META[name] || { label: name, sub: '' }

  // Determine visual state
  // stageData is null → not yet reached
  // stageData.skipped → resumed from cache
  // stageData.error   → failed
  // otherwise         → completed

  let state = 'pending'
  if (stageData) {
    if (stageData.error)   state = 'error'
    else if (stageData.skipped) state = 'skipped'
    else state = 'done'
  } else if (index === totalCompleted) {
    state = 'running'
  }

  const stateColor = {
    pending: 'var(--border-bright)',
    running: 'var(--accent)',
    done:    'var(--green)',
    skipped: 'var(--text-muted)',
    error:   'var(--red)',
  }[state]

  const StateIcon = {
    pending: () => <span style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px solid var(--border-bright)', display: 'inline-block' }} />,
    running: () => <div style={{ width: 14, height: 14, border: '2px solid var(--border)', borderTop: `2px solid var(--accent)`, borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />,
    done:    () => <CheckCircle size={14} style={{ color: 'var(--green)' }} />,
    skipped: () => <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />,
    error:   () => <XCircle size={14} style={{ color: 'var(--red)' }} />,
  }[state]

  return (
    <div style={{
      ...styles.stageRow,
      opacity: state === 'pending' ? 0.4 : 1,
      borderLeft: `3px solid ${stateColor}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <StateIcon />
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>
            {meta.label}
            {state === 'skipped' && (
              <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>RESUMED</span>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>{meta.sub}</div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2, flexShrink: 0 }}>
        {stageData?.elapsed_s > 0 && (
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            <Clock size={9} style={{ display: 'inline', marginRight: 3 }} />
            {fmtElapsed(stageData.elapsed_s)}
          </span>
        )}
        {state === 'error' && stageData?.error && (
          <span style={{ fontSize: 10, color: 'var(--red)', maxWidth: 160, textAlign: 'right', lineHeight: 1.3 }}>
            {stageData.error.slice(0, 80)}{stageData.error.length > 80 ? '…' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Pipeline tracker panel ────────────────────────────────────────────────

function PipelineTracker({ job }) {
  if (!job) return null

  const stageMap = {}
  ;(job.stages || []).forEach(s => { stageMap[s.name] = s })

  const completedCount = (job.stages || []).filter(s => !s.error).length

  const statusColor = {
    initialising: 'var(--text-muted)',
    running:      'var(--accent)',
    complete:     'var(--green)',
    failed:       'var(--red)',
    unknown:      'var(--text-muted)',
  }[job.status] || 'var(--text-muted)'

  return (
    <div style={styles.tracker}>
      {/* Header */}
      <div style={styles.trackerHeader}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: statusColor,
              boxShadow: job.status === 'running' ? `0 0 8px ${statusColor}` : 'none',
              animation: job.status === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700, color: statusColor, letterSpacing: '.08em', textTransform: 'uppercase' }}>
              {job.status}
            </span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
            session: {job.job_id}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 20 }}>
          {[
            ['Frames',     job.n_frames],
            ['Detections', job.n_detections],
            ['Inserted',   job.n_inserted],
            ['Updated',    job.n_updated],
          ].map(([label, val]) => (
            <div key={label} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>{val ?? 0}</div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timestamps */}
      {job.started_at && (
        <div style={{ padding: '0 16px 8px', display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          <span>Started: {fmtTs(job.started_at)}</span>
          {job.finished_at && <span>Finished: {fmtTs(job.finished_at)}</span>}
        </div>
      )}

      {/* Progress bar */}
      <div style={{ height: 3, background: 'var(--border)', margin: '0 16px 12px' }}>
        <div style={{
          height: '100%',
          width: `${Math.round((completedCount / STAGE_ORDER.length) * 100)}%`,
          background: job.status === 'failed' ? 'var(--red)' : 'var(--accent)',
          borderRadius: 2,
          transition: 'width 0.6s ease',
        }} />
      </div>

      {/* Stage rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: '0 16px 16px' }}>
        {STAGE_ORDER.map((name, idx) => (
          <StageRow
            key={name}
            name={name}
            stageData={stageMap[name] || null}
            index={idx}
            totalCompleted={completedCount}
          />
        ))}
      </div>

      {/* Error details */}
      {job.status === 'failed' && job.error_message && (
        <div style={styles.errorBox}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
            <AlertTriangle size={13} style={{ color: 'var(--red)', flexShrink: 0 }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--red)' }}>Pipeline error</span>
          </div>
          <pre style={{ fontSize: 10, color: 'var(--text-muted)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0, maxHeight: 160, overflowY: 'auto' }}>
            {job.error_message}
          </pre>
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────

export default function IngestionPage() {
  const navigate = useNavigate()

  // File selection state
  const [videoFile, setVideoFile] = useState(null)
  const [gpsFile,   setGpsFile]   = useState(null)

  // Upload phase
  const [uploading,    setUploading]    = useState(false)
  const [uploadPct,    setUploadPct]    = useState(0)
  const [uploadError,  setUploadError]  = useState(null)

  // Poll phase
  const [jobId,        setJobId]        = useState(null)
  const [jobStatus,    setJobStatus]    = useState(null)   // full status object
  const pollRef = useRef(null)

  // ── Start polling ───────────────────────────────────────────────────────

  const startPolling = useCallback((jid) => {
    if (pollRef.current) clearInterval(pollRef.current)

    const poll = async () => {
      try {
        const data = await fetchJobStatus(jid)
        setJobStatus(data)
        // Stop when terminal state reached
        if (data.status === 'complete' || data.status === 'failed') {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      } catch (err) {
        // Network blip — keep polling, don't clear interval
        console.warn('[RIDS] Status poll error:', err.message)
      }
    }

    // Fire immediately, then repeat
    poll()
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS)
  }, [])

  // Clean up poll on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  // ── Submit handler ──────────────────────────────────────────────────────

  const handleSubmit = async () => {
    if (!videoFile) return

    setUploading(true)
    setUploadError(null)
    setUploadPct(0)
    setJobId(null)
    setJobStatus(null)

    try {
      const result = await uploadSurvey(videoFile, gpsFile, setUploadPct)
      // Persist job_id so MapPage can start live-polling even if the user
      // navigates away from IngestionPage before the pipeline completes.
      localStorage.setItem('rids_active_job', result.job_id)
      setJobId(result.job_id)
      setJobStatus({ job_id: result.job_id, status: 'running', stages: [], n_frames: 0, n_detections: 0, n_inserted: 0, n_updated: 0, error_message: null, started_at: null, finished_at: null })
      startPolling(result.job_id)
    } catch (err) {
      const detail = err?.response?.data?.detail || err.message || 'Upload failed'
      setUploadError(detail)
    } finally {
      setUploading(false)
      setUploadPct(0)
    }
  }

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setVideoFile(null)
    setGpsFile(null)
    setUploadError(null)
    setJobId(null)
    setJobStatus(null)
    setUploading(false)
    setUploadPct(0)
    localStorage.removeItem('rids_active_job')
  }

  const isRunning  = jobStatus?.status === 'running' || jobStatus?.status === 'initialising'
  const isComplete = jobStatus?.status === 'complete'
  const isFailed   = jobStatus?.status === 'failed'
  const hasJob     = jobId !== null

  return (
    <div style={styles.page}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.backBtn} onClick={() => navigate('/')}>
            <ArrowLeft size={14} /> MAP
          </button>
          <div>
            <h1 style={styles.title}>New Survey</h1>
            <p style={styles.subtitle}>
              Upload dashcam footage · Run the RIDS inference pipeline
            </p>
          </div>
        </div>
        {isComplete && (
          <button
            style={{ ...styles.backBtn, color: 'var(--accent)', borderColor: 'var(--accent)' }}
            onClick={() => navigate('/')}
          >
            <Map size={13} /> View on Map
          </button>
        )}
      </div>

      <div style={styles.body}>
        <div style={styles.grid}>

          {/* ── Left column: upload form ──────────────────────────────── */}
          <div style={styles.column}>

            {/* Section: file selection */}
            <div style={styles.section}>
              <div style={styles.sectionTitle}>1 — SELECT FILES</div>

              <DropZone
                label="Dashcam Footage"
                accept=".mp4,video/mp4"
                file={videoFile}
                onFile={setVideoFile}
                icon={FileVideo}
                color="var(--accent)"
                hint="Drag & drop or click to browse · .mp4 required"
              />

              <div style={{ marginTop: 10 }}>
                <DropZone
                  label="GPS Log (optional)"
                  accept=".gpx"
                  file={gpsFile}
                  onFile={setGpsFile}
                  icon={FileGps}
                  color="var(--blue)"
                  hint="Drag & drop .gpx · Required for map placement"
                />
              </div>

              {!gpsFile && (
                <div style={styles.hint}>
                  <AlertTriangle size={11} style={{ flexShrink: 0 }} />
                  Without a GPS file, frames will have no coordinates. Stages 6–7
                  (deduplication + DB write) will be skipped.
                </div>
              )}
            </div>

            {/* Section: submit */}
            <div style={styles.section}>
              <div style={styles.sectionTitle}>2 — RUN PIPELINE</div>

              {uploadError && (
                <div style={styles.errorBanner}>
                  <XCircle size={13} style={{ flexShrink: 0 }} />
                  {uploadError}
                </div>
              )}

              {uploading && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span>Uploading…</span>
                    <span>{uploadPct}%</span>
                  </div>
                  <div style={{ height: 4, background: 'var(--border)', borderRadius: 2 }}>
                    <div style={{
                      height: '100%',
                      width: `${uploadPct}%`,
                      background: 'var(--accent)',
                      borderRadius: 2,
                      transition: 'width 0.2s ease',
                    }} />
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  style={{
                    ...styles.submitBtn,
                    opacity: (!videoFile || uploading || isRunning) ? 0.4 : 1,
                    cursor: (!videoFile || uploading || isRunning) ? 'not-allowed' : 'pointer',
                  }}
                  disabled={!videoFile || uploading || isRunning}
                  onClick={handleSubmit}
                >
                  {uploading ? (
                    <><Loader size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> Uploading…</>
                  ) : isRunning ? (
                    <><Loader size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> Pipeline running…</>
                  ) : (
                    <><Upload size={14} /> Start Pipeline</>
                  )}
                </button>

                {(hasJob || videoFile || gpsFile) && (
                  <button style={styles.resetBtn} onClick={handleReset} disabled={isRunning}>
                    <RefreshCw size={13} /> Reset
                  </button>
                )}
              </div>

              {isRunning && (
                <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
                  Pipeline is running. This page auto-updates every {POLL_INTERVAL_MS / 1000}s — you can leave and come back.
                </div>
              )}
            </div>

            {/* Section: info */}
            <div style={styles.infoBox}>
              <div style={styles.sectionTitle}>PIPELINE STAGES</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                {STAGE_ORDER.map((name, i) => (
                  <div key={name} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                    <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', width: 16, flexShrink: 0, paddingTop: 1 }}>
                      S{i + 1}
                    </span>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-dim)' }}>
                        {STAGE_META[name].label}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {STAGE_META[name].sub}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Right column: live tracker ────────────────────────────── */}
          <div style={styles.column}>
            <div style={styles.section}>
              <div style={styles.sectionTitle}>3 — PIPELINE STATUS</div>

              {!hasJob && !uploading && (
                <div style={styles.emptyTracker}>
                  <Upload size={28} style={{ color: 'var(--border-bright)', marginBottom: 12 }} />
                  <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                    No active job
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    Upload a video to start the pipeline
                  </div>
                </div>
              )}

              {(uploading && !hasJob) && (
                <div style={styles.emptyTracker}>
                  <div style={{ width: 28, height: 28, border: '3px solid var(--border)', borderTop: '3px solid var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', marginBottom: 12 }} />
                  <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Uploading video…</div>
                  <div style={{ fontSize: 11, color: 'var(--accent)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>{uploadPct}%</div>
                </div>
              )}

              {hasJob && <PipelineTracker job={jobStatus} />}
            </div>

            {/* Success summary */}
            {isComplete && jobStatus && (
              <div style={styles.successBox}>
                <CheckCircle size={20} style={{ color: 'var(--green)', flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>
                    Pipeline complete
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                    {jobStatus.n_inserted} new detections inserted ·{' '}
                    {jobStatus.n_updated} updated ·{' '}
                    {jobStatus.n_frames} frames processed
                  </div>
                  <button
                    style={{ ...styles.backBtn, marginTop: 10, color: 'var(--accent)', borderColor: 'var(--accent)' }}
                    onClick={() => navigate('/')}
                  >
                    <Map size={13} /> View on Map
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────

const styles = {
  page: {
    paddingTop: 48,
    minHeight: '100vh',
    background: 'var(--bg)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '24px 32px 0',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  backBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    padding: '6px 12px',
    background: 'transparent',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-muted)',
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    cursor: 'pointer',
    letterSpacing: '.08em',
    transition: 'var(--transition)',
  },
  title: {
    fontSize: 26,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '-0.5px',
  },
  subtitle: {
    fontSize: 12,
    color: 'var(--text-muted)',
    marginTop: 2,
  },
  body: {
    padding: '24px 32px 48px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 20,
    alignItems: 'start',
  },
  column: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  section: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '20px 20px',
  },
  sectionTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    color: 'var(--text-muted)',
    letterSpacing: '.12em',
    marginBottom: 14,
  },

  // Drop zone
  dropZone: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '14px 16px',
    borderRadius: 'var(--radius)',
    border: '1.5px dashed var(--border-bright)',
    userSelect: 'none',
  },
  dropIcon: {
    width: 38,
    height: 38,
    borderRadius: 'var(--radius)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },

  hint: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 6,
    marginTop: 10,
    padding: '8px 12px',
    background: 'rgba(255,140,66,0.08)',
    border: '1px solid rgba(255,140,66,0.2)',
    borderRadius: 'var(--radius)',
    fontSize: 11,
    color: 'var(--orange)',
    lineHeight: 1.4,
  },

  // Buttons
  submitBtn: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: '10px 20px',
    background: 'var(--accent)',
    border: 'none',
    borderRadius: 'var(--radius)',
    color: '#0a0c10',
    fontSize: 13,
    fontWeight: 700,
    transition: 'var(--transition)',
  },
  resetBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '10px 16px',
    background: 'transparent',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-muted)',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'var(--transition)',
  },

  // Error banner
  errorBanner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
    padding: '10px 14px',
    background: 'rgba(255,68,68,0.08)',
    border: '1px solid rgba(255,68,68,0.3)',
    borderRadius: 'var(--radius)',
    fontSize: 12,
    color: 'var(--red)',
    marginBottom: 12,
    lineHeight: 1.4,
  },

  // Info box
  infoBox: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '16px 20px',
    opacity: 0.8,
  },

  // Empty tracker
  emptyTracker: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '48px 20px',
    color: 'var(--text-muted)',
    textAlign: 'center',
  },

  // Pipeline tracker
  tracker: {
    background: 'var(--bg-card2)',
    borderRadius: 'var(--radius)',
    overflow: 'hidden',
  },
  trackerHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: '14px 16px',
    borderBottom: '1px solid var(--border)',
  },

  // Stage row
  stageRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 10px',
    borderRadius: 4,
    background: 'var(--bg-card)',
    transition: 'var(--transition)',
  },

  // Error box inside tracker
  errorBox: {
    margin: '0 16px 16px',
    padding: '10px 14px',
    background: 'rgba(255,68,68,0.06)',
    border: '1px solid rgba(255,68,68,0.25)',
    borderRadius: 'var(--radius)',
  },

  // Success box
  successBox: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 14,
    padding: '18px 20px',
    background: 'rgba(74,222,128,0.06)',
    border: '1px solid rgba(74,222,128,0.25)',
    borderRadius: 'var(--radius-lg)',
  },
}
