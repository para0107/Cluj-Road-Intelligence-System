/**
 * frontend/src/pages/IngestionPage.jsx — Survey upload + live pipeline tracker.
 *
 * Contract with the backend (do not break):
 *  - POST /api/ingest/upload (multipart) → { job_id } (202) or 409 while busy
 *  - GET  /api/ingest/status/{job_id} polled every POLL_INTERVAL_MS
 *  - stage keys must match orchestrator session.json: see PIPELINE_STAGES
 *  - localStorage['rids_active_job'] is the cross-page "pipeline running" flag
 *    (MapPage live-refreshes while it is set; we clear it on terminal states)
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FileVideo, MapPin, CheckCircle, XCircle, ChevronRight, Clock,
  AlertTriangle, Map, Play, RotateCcw, UploadCloud, Satellite,
} from 'lucide-react'
import { uploadSurvey, fetchJobStatus } from '../utils/api'
import { PIPELINE_STAGES } from '../utils/constants'
import { SectionTitle } from '../components/ui'

const POLL_INTERVAL_MS = 10_000

const STAGE_META = Object.fromEntries(PIPELINE_STAGES.map(s => [s.key, s]))
const STAGE_ORDER = PIPELINE_STAGES.map(s => s.key)

// ── Helpers ────────────────────────────────────────────────────────────────

function fmtElapsed(s) {
  if (s == null || s === 0) return null
  if (s < 60) return `${s.toFixed(1)} s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

function fmtTs(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleTimeString() } catch { return iso }
}

// ── File drop zone ─────────────────────────────────────────────────────────

function DropZone({ label, accept, file, onFile, icon: Icon, color, hint }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        ...styles.dropZone,
        borderColor: dragOver ? color : file ? `${color}80` : 'var(--border-bright)',
        background: dragOver ? `${color}10` : file ? `${color}08` : 'var(--bg-card2)',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
      />
      <div style={{
        ...styles.dropIcon,
        color: file ? color : 'var(--text-muted)',
        background: file ? `${color}18` : 'var(--border)',
      }}>
        <Icon size={20} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: file ? 'var(--text)' : 'var(--text-muted)' }}>
          {label}
        </div>
        {file ? (
          <div className="mono" style={{ fontSize: 11, color, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
          </div>
        ) : (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{hint}</div>
        )}
      </div>
      {file && <CheckCircle size={16} style={{ color, flexShrink: 0 }} />}
    </div>
  )
}

// ── Stage row ──────────────────────────────────────────────────────────────

function StageRow({ name, stageData, index, totalCompleted }) {
  const meta = STAGE_META[name] || { label: name, sub: '' }

  // stageData null → not reached · .skipped → resumed/cached · .error → failed
  let state = 'pending'
  if (stageData) {
    if (stageData.error) state = 'error'
    else if (stageData.skipped) state = 'skipped'
    else state = 'done'
  } else if (index === totalCompleted) {
    state = 'running'
  }

  const stateColor = {
    pending: 'var(--border-bright)',
    running: 'var(--accent)',
    done: 'var(--green)',
    skipped: 'var(--text-muted)',
    error: 'var(--red)',
  }[state]

  const StateIcon = {
    pending: () => <span style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px solid var(--border-bright)', display: 'inline-block' }} />,
    running: () => <div style={{ width: 14, height: 14, border: '2px solid var(--border)', borderTop: '2px solid var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />,
    done: () => <CheckCircle size={14} style={{ color: 'var(--green)' }} />,
    skipped: () => <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />,
    error: () => <XCircle size={14} style={{ color: 'var(--red)' }} />,
  }[state]

  return (
    <div style={{
      ...styles.stageRow,
      opacity: state === 'pending' ? 0.45 : 1,
      borderLeft: `3px solid ${stateColor}`,
      background: state === 'running' ? 'var(--accent-dim)' : 'var(--bg-card2)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
        <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', width: 16 }}>
          {String(index + 1).padStart(2, '0')}
        </span>
        <StateIcon />
        <div>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text)' }}>
            {meta.label}
            {state === 'skipped' && (
              <span className="mono" style={{ marginLeft: 6, fontSize: 9.5, color: 'var(--text-muted)' }}>SKIPPED / CACHED</span>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>{meta.sub}</div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2, flexShrink: 0 }}>
        {stageData?.elapsed_s > 0 && (
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            <Clock size={9} style={{ display: 'inline', marginRight: 3 }} />
            {fmtElapsed(stageData.elapsed_s)}
          </span>
        )}
        {state === 'error' && stageData?.error && (
          <span style={{ fontSize: 10, color: 'var(--red)', maxWidth: 170, textAlign: 'right', lineHeight: 1.3 }}>
            {stageData.error.slice(0, 80)}{stageData.error.length > 80 ? '…' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Pipeline tracker panel ─────────────────────────────────────────────────

function PipelineTracker({ job }) {
  if (!job) return null

  const stageMap = {}
  ;(job.stages || []).forEach(s => { stageMap[s.name] = s })
  const completedCount = (job.stages || []).filter(s => !s.error).length

  const statusColor = {
    pending: 'var(--text-muted)',
    initialising: 'var(--cyan)',
    running: 'var(--accent)',
    complete: 'var(--green)',
    failed: 'var(--red)',
    unknown: 'var(--text-muted)',
  }[job.status] || 'var(--text-muted)'

  return (
    <div className="card anim-fade-up" style={{ overflow: 'hidden' }}>
      {/* Header */}
      <div style={styles.trackerHeader}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: statusColor,
              boxShadow: job.status === 'running' ? `0 0 8px ${statusColor}` : 'none',
              animation: (job.status === 'running' || job.status === 'initialising') ? 'pulse 1.5s ease-in-out infinite' : 'none',
              flexShrink: 0,
            }} />
            <span className="mono" style={{ fontSize: 11, fontWeight: 700, color: statusColor, letterSpacing: '.08em', textTransform: 'uppercase' }}>
              {job.status}
            </span>
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            session {job.job_id}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 20 }}>
          {[
            ['Frames', job.n_frames],
            ['Detections', job.n_detections],
            ['Inserted', job.n_inserted],
            ['Updated', job.n_updated],
          ].map(([label, val]) => (
            <div key={label} style={{ textAlign: 'center' }}>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>{val ?? 0}</div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timestamps */}
      {job.started_at && (
        <div className="mono" style={{ padding: '0 18px 8px', display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)' }}>
          <span>Started {fmtTs(job.started_at)}</span>
          {job.finished_at && <span>Finished {fmtTs(job.finished_at)}</span>}
        </div>
      )}

      {/* Progress bar */}
      <div style={{ height: 3, background: 'var(--border)', margin: '0 18px 14px', borderRadius: 2 }}>
        <div style={{
          height: '100%',
          width: `${Math.round((completedCount / STAGE_ORDER.length) * 100)}%`,
          background: job.status === 'failed' ? 'var(--red)' : 'var(--accent)',
          borderRadius: 2,
          transition: 'width 0.6s ease',
        }} />
      </div>

      {/* Stage rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3, padding: '0 18px 18px' }}>
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

// ── Main page ──────────────────────────────────────────────────────────────

export default function IngestionPage() {
  const navigate = useNavigate()

  const [videoFile, setVideoFile] = useState(null)
  const [gpsFile, setGpsFile] = useState(null)

  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [uploadError, setUploadError] = useState(null)

  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const pollRef = useRef(null)

  // ── Polling ──────────────────────────────────────────────────────────────
  const startPolling = useCallback((jid) => {
    if (pollRef.current) clearInterval(pollRef.current)

    const poll = async () => {
      try {
        const data = await fetchJobStatus(jid)
        setJobStatus(data)
        // Terminal state → stop and clear the cross-page flag so a finished
        // run doesn't rehydrate a stale tracker on the next visit.
        if (data.status === 'complete' || data.status === 'failed') {
          clearInterval(pollRef.current)
          pollRef.current = null
          localStorage.removeItem('rids_active_job')
        }
      } catch (err) {
        // 404 → the job no longer exists on the backend (stale localStorage id)
        if (err?.response?.status === 404) {
          clearInterval(pollRef.current)
          pollRef.current = null
          localStorage.removeItem('rids_active_job')
          setJobId(null)
          setJobStatus(null)
          return
        }
        console.warn('[RDDS] Status poll error:', err.message)
      }
    }

    poll()
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS)
  }, [])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  // Rehydrate an in-flight job after navigation / refresh. The pipeline runs
  // on the host and keeps writing session.json regardless of this page being
  // mounted, so reconnect to the saved job_id and resume the live tracker.
  useEffect(() => {
    const savedJobId = localStorage.getItem('rids_active_job')
    if (!savedJobId) return
    setJobId(savedJobId)
    setJobStatus({
      job_id: savedJobId, status: 'running', stages: [],
      n_frames: 0, n_detections: 0, n_inserted: 0, n_updated: 0,
      error_message: null, started_at: null, finished_at: null,
    })
    startPolling(savedJobId)
  }, [startPolling])

  // ── Submit ───────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!videoFile) return
    setUploading(true)
    setUploadError(null)
    setUploadPct(0)
    setJobId(null)
    setJobStatus(null)

    try {
      const result = await uploadSurvey(videoFile, gpsFile, setUploadPct)
      // Persist so MapPage live-refreshes even if the user navigates away.
      localStorage.setItem('rids_active_job', result.job_id)
      setJobId(result.job_id)
      setJobStatus({
        job_id: result.job_id, status: 'pending', stages: [],
        n_frames: 0, n_detections: 0, n_inserted: 0, n_updated: 0,
        error_message: null, started_at: null, finished_at: null,
      })
      startPolling(result.job_id)
    } catch (err) {
      setUploadError(err?.response?.data?.detail || err.message || 'Upload failed')
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

  const isRunning = jobStatus?.status === 'running' || jobStatus?.status === 'initialising' || jobStatus?.status === 'pending'
  const isComplete = jobStatus?.status === 'complete'
  const hasJob = jobId !== null

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>
        <SectionTitle
          overline="Ingest"
          title="New survey"
          right={isComplete && (
            <button className="btn btn-accent btn-sm" onClick={() => navigate('/map')}>
              <Map size={13} /> View results on map
            </button>
          )}
        />

        <div style={styles.grid}>
          {/* ── Left column: upload form ──────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="card anim-fade-up" style={{ padding: 20 }}>
              <div className="overline" style={{ marginBottom: 14 }}>01 — Select files</div>

              <DropZone
                label="Dashcam footage"
                accept=".mp4,video/mp4"
                file={videoFile}
                onFile={setVideoFile}
                icon={FileVideo}
                color="var(--accent)"
                hint="Drag & drop or click to browse · .mp4 required"
              />

              <div style={{ marginTop: 10 }}>
                <DropZone
                  label="GPS log (optional)"
                  accept=".gpx"
                  file={gpsFile}
                  onFile={setGpsFile}
                  icon={Satellite}
                  color="var(--cyan)"
                  hint="Drag & drop .gpx · Required for map placement"
                />
              </div>

              {!gpsFile && (
                <div style={styles.hint}>
                  <AlertTriangle size={11} style={{ flexShrink: 0, marginTop: 1 }} />
                  Without a GPS file, frames have no coordinates: stages 6–7
                  (dedup + DB write) are skipped and nothing appears on the map.
                  The run still completes with detection artifacts.
                </div>
              )}
            </div>

            <div className="card anim-fade-up delay-1" style={{ padding: 20 }}>
              <div className="overline" style={{ marginBottom: 14 }}>02 — Run pipeline</div>

              {uploadError && (
                <div style={styles.errorBanner}>
                  <XCircle size={13} style={{ flexShrink: 0 }} />
                  {uploadError}
                </div>
              )}

              {uploading && (
                <div style={{ marginBottom: 12 }}>
                  <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                    <span>Uploading…</span>
                    <span>{uploadPct}%</span>
                  </div>
                  <div style={{ height: 4, background: 'var(--border)', borderRadius: 2 }}>
                    <div style={{
                      height: '100%', width: `${uploadPct}%`,
                      background: 'var(--accent)', borderRadius: 2,
                      transition: 'width 0.2s ease',
                    }} />
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  className="btn btn-accent"
                  style={{ flex: 1, padding: '11px 0' }}
                  disabled={!videoFile || uploading || isRunning}
                  onClick={handleSubmit}
                >
                  {uploading
                    ? <><UploadCloud size={15} /> Uploading…</>
                    : isRunning
                      ? <><Play size={15} /> Pipeline running…</>
                      : <><Play size={15} /> Start pipeline</>}
                </button>
                {(hasJob || videoFile) && (
                  <button className="btn" onClick={handleReset} disabled={uploading}>
                    <RotateCcw size={14} /> Reset
                  </button>
                )}
              </div>

              <div style={{ marginTop: 14, fontSize: 10.5, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                The upload is handed to the GPU worker on the host
                (<span className="mono">pipeline/job_watcher.py</span> must be running).
                One survey is processed at a time — a second upload is rejected
                with 409 until the current run finishes.
              </div>
            </div>
          </div>

          {/* ── Right column: tracker ─────────────────────────────────── */}
          <div>
            {hasJob ? (
              <PipelineTracker job={jobStatus} />
            ) : (
              <div className="card anim-fade-up delay-2" style={styles.placeholder}>
                <MapPin size={22} style={{ color: 'var(--accent)', marginBottom: 10 }} />
                <div className="display" style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>
                  The 7-stage pipeline will appear here
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7, maxWidth: 330 }}>
                  Frame extraction → RT-DETR detection → SAM segmentation →
                  Monodepth2 depth → severity scoring → spatial dedup → PostGIS write.
                  Live progress updates every {POLL_INTERVAL_MS / 1000} seconds.
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 18, width: '100%' }}>
                  {PIPELINE_STAGES.map((s, i) => (
                    <div key={s.key} style={{ ...styles.ghostStage }}>
                      <span className="mono" style={{ fontSize: 9.5, color: 'var(--text-muted)', width: 16 }}>{String(i + 1).padStart(2, '0')}</span>
                      <span style={{ width: 12, height: 12, borderRadius: '50%', border: '1.5px solid var(--border-bright)' }} />
                      <span style={{ fontSize: 11.5, color: 'var(--text-dim)' }}>{s.label}</span>
                      <span style={{ fontSize: 9.5, color: 'var(--text-muted)', marginLeft: 'auto' }}>{s.sub}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 26px)',
    paddingBottom: 40,
  },
  inner: {
    maxWidth: 1060,
    margin: '0 auto',
    padding: '0 26px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(300px, 5fr) minmax(340px, 6fr)',
    gap: 14,
    alignItems: 'start',
  },

  dropZone: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '16px 16px',
    borderRadius: 'var(--radius-lg)',
    border: '1.5px dashed var(--border-bright)',
    cursor: 'pointer',
    transition: 'var(--transition)',
  },
  dropIcon: {
    width: 44, height: 44, borderRadius: 12, flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'var(--transition)',
  },
  hint: {
    display: 'flex',
    gap: 7,
    marginTop: 12,
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255,159,67,0.08)',
    border: '1px solid rgba(255,159,67,0.25)',
    color: 'var(--orange)',
    fontSize: 10.5,
    lineHeight: 1.55,
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
    padding: '9px 12px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255,93,93,0.1)',
    border: '1px solid rgba(255,93,93,0.35)',
    color: 'var(--red)',
    fontSize: 11.5,
  },

  trackerHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    padding: '16px 18px 12px',
    gap: 12,
    flexWrap: 'wrap',
  },
  stageRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    padding: '9px 12px',
    borderRadius: 8,
    transition: 'var(--transition)',
  },
  errorBox: {
    margin: '0 18px 18px',
    padding: '12px 14px',
    borderRadius: 'var(--radius)',
    background: 'rgba(255,93,93,0.07)',
    border: '1px solid rgba(255,93,93,0.3)',
  },
  placeholder: {
    padding: '26px 24px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
  },
  ghostStage: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '7px 10px',
    borderRadius: 7,
    background: 'var(--bg-card2)',
    opacity: 0.75,
  },
}
