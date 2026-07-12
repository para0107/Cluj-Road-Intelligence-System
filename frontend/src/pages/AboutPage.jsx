/**
 * frontend/src/pages/AboutPage.jsx — "System" page.
 *
 * Explains what RDDS is, the model stack, the severity model, and the
 * execution architecture. Written for a public / municipality demo.
 */

import React from 'react'
import { Link } from 'react-router-dom'
import {
  ScanLine, Radar, Layers, Ruler, Gauge, Copy, Database, ArrowRight,
  Cpu, Container, HardDrive, MapPin, GraduationCap,
} from 'lucide-react'
import { PIPELINE_STAGES, SEVERITY_COLORS, SEVERITY_LABELS, SEVERITY_ACTIONS } from '../utils/constants'
import { SectionTitle } from '../components/ui'
import useMotionOk from '../hooks/useMotionOk'
import SplitText from '../reactbits/SplitText/SplitText'
import FadeContent from '../reactbits/FadeContent/FadeContent'

const STAGE_ICONS = [ScanLine, Radar, Layers, Ruler, Gauge, Copy, Database]

const STAGE_DETAILS = {
  preprocessor: 'Extracts frames at ~2 fps, synchronises each frame to a GPS coordinate by timestamp interpolation from the .gpx log, computes solar elevation and a lighting class per frame.',
  detector: 'RT-DETR-L fine-tuned on the N-RDD2024 dataset detects 10 classes of road damage and road features, with per-class confidence thresholds.',
  segmentor: 'SAM 2.1 Tiny generates a pixel mask for every detection and derives four geometry features: surface area, edge sharpness, interior contrast and mask compactness.',
  depth_estimator: 'Monodepth2 (mono_640x192) estimates relative disparity as a depth proxy inside each mask, with a geometry-based fallback when the estimate is unreliable.',
  severity_classifier: 'A transparent rule-based model combines depth, area, contrast and sharpness with per-class weights into a severity score mapped to S1–S5.',
  deduplicator: 'DBSCAN clustering with a Haversine metric merges repeated sightings of the same physical damage across frames and surveys.',
  db_writer: 'Upserts surviving detections into PostGIS — an existing detection of the same type within the cluster radius is updated instead of duplicated — and recomputes priority scores.',
}

const MODELS = [
  { role: 'Detection', model: 'RT-DETR-L', note: 'fine-tuned on N-RDD2024, 10 damage classes' },
  { role: 'Segmentation', model: 'SAM 2.1 Tiny', note: 'mask geometry features per detection' },
  { role: 'Depth', model: 'Monodepth2', note: 'mono_640x192 · relative disparity proxy' },
  { role: 'Severity', model: 'Rule-based', note: 'weighted multi-signal, fully explainable' },
  { role: 'Dedup', model: 'DBSCAN', note: 'Haversine metric, metres-scale radius' },
]

export default function AboutPage() {
  const motionOk = useMotionOk()

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>

        {/* ── Intro ────────────────────────────────────────────────────── */}
        <div style={{ maxWidth: 760, margin: '0 auto', textAlign: 'center', paddingBottom: 34 }}>
          <div className="overline anim-fade-up" style={{ color: 'var(--accent)', marginBottom: 12 }}>
            <MapPin size={11} style={{ display: 'inline', marginRight: 6 }} />
            THE SYSTEM
          </div>
          {motionOk ? (
            <h1 className="display" style={{ fontSize: 'clamp(26px, 4vw, 40px)', fontWeight: 700, letterSpacing: '-0.02em', lineHeight: 1.15 }}>
              <SplitText
                text="A city-scale road survey for the price of a dashcam."
                tag="span"
                splitType="words"
                delay={40}
                duration={0.7}
                textAlign="center"
                className="display"
              />
            </h1>
          ) : (
            <h1 className="display anim-fade-up delay-1" style={{ fontSize: 'clamp(26px, 4vw, 40px)', fontWeight: 700, letterSpacing: '-0.02em', lineHeight: 1.15 }}>
              A city-scale road survey<br />for the price of a dashcam.
            </h1>
          )}
          <p className="anim-fade-up delay-2" style={{ fontSize: 14, color: 'var(--text-dim)', lineHeight: 1.75, marginTop: 16 }}>
            Traditional road inspections are manual, expensive and infrequent, and a full
            city survey can take months. RDDS replaces them with footage collected during normal
            driving: an overnight processing run turns raw video into a georeferenced damage map with
            severity scores and a ranked repair list.
          </p>
        </div>

        <div className="road-divider anim-fade-up delay-2" style={{ width: 220, margin: '0 auto 44px' }} />

        {/* ── Pipeline ─────────────────────────────────────────────────── */}
        <SectionTitle overline="Pipeline" title="Seven stages, one pass" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 44 }}>
          {PIPELINE_STAGES.map((st, i) => {
            const Icon = STAGE_ICONS[i]
            return (
              <div key={st.key} className={`card anim-fade-up delay-${Math.min(i + 1, 6)}`} style={styles.stageRow}>
                <div className="mono" style={styles.stageNum}>{String(i + 1).padStart(2, '0')}</div>
                <div style={styles.stageIcon}><Icon size={17} /></div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
                    <span className="display" style={{ fontSize: 14, fontWeight: 700 }}>{st.label}</span>
                    <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{st.sub}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.65 }}>
                    {STAGE_DETAILS[st.key]}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* ── Severity model ───────────────────────────────────────────── */}
        <SectionTitle overline="Scoring" title="The S1–S5 severity scale" />
        <div style={styles.sevGrid}>
          {[1, 2, 3, 4, 5].map(s => (
            <div key={s} className={`card anim-fade-up delay-${s}`} style={{ padding: '16px 18px', borderTop: `3px solid ${SEVERITY_COLORS[s]}` }}>
              <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: SEVERITY_COLORS[s] }}>
                {SEVERITY_LABELS[s]}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-dim)', marginTop: 8, lineHeight: 1.6 }}>
                {SEVERITY_ACTIONS[s]}
              </div>
            </div>
          ))}
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '14px 0 44px', lineHeight: 1.7, maxWidth: 760 }}>
          The score combines four normalised signals (relative depth, mask area, interior contrast and
          edge sharpness) with per-class weights, then applies a class-importance factor. Road-marking
          classes are structurally capped at S2: a blurred lane line is never an "emergency". Priority
          for the repair queue is <span className="mono">severity · log(times observed + 1)</span>, so
          persistent damage rises over time.
        </p>

        {/* ── Model stack ──────────────────────────────────────────────── */}
        <SectionTitle overline="Models" title="The model stack" />
        <div className="card anim-fade-up" style={{ overflow: 'hidden', marginBottom: 44 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Role', 'Model', 'Notes'].map(h => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MODELS.map(m => (
                <tr key={m.role} className="table-row-hover">
                  <td style={{ ...styles.td, color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{m.role}</td>
                  <td style={{ ...styles.td, fontWeight: 600 }}>{m.model}</td>
                  <td style={{ ...styles.td, color: 'var(--text-dim)' }}>{m.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* ── Architecture ─────────────────────────────────────────────── */}
        <SectionTitle overline="Architecture" title="Two worlds, one data folder" />
        <div style={styles.archGrid}>
          <div className="card anim-fade-up delay-1" style={styles.archCard}>
            <Container size={18} style={{ color: 'var(--cyan)', marginBottom: 10 }} />
            <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>Docker: the web stack</div>
            <div style={styles.archText}>
              PostGIS, the FastAPI backend and this React frontend run in containers with no GPU and
              no ML dependencies. The backend accepts uploads and serves the map, stats and queue.
            </div>
          </div>
          <div className="card anim-fade-up delay-2" style={styles.archCard}>
            <HardDrive size={18} style={{ color: 'var(--accent)', marginBottom: 10 }} />
            <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>The shared data/ folder</div>
            <div style={styles.archText}>
              The only bridge between the two worlds. The backend drops a job file; the host watcher
              picks it up; the orchestrator writes live progress back after every stage. No sockets,
              no queues, and the design survives restarts on either side.
            </div>
          </div>
          <div className="card anim-fade-up delay-3" style={styles.archCard}>
            <Cpu size={18} style={{ color: 'var(--green)', marginBottom: 10 }} />
            <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>Host: the GPU pipeline</div>
            <div style={styles.archText}>
              A Windows host with CUDA runs the seven-stage pipeline: PyTorch, RT-DETR, SAM 2.1 and
              Monodepth2 process each survey and upsert results into PostGIS.
            </div>
          </div>
        </div>

        {/* ── Credits ──────────────────────────────────────────────────── */}
        <FadeContent duration={700} blur>
          <div className="card" style={{ padding: '22px 24px', marginTop: 44, display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
            <GraduationCap size={26} style={{ color: 'var(--accent)', flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 260 }}>
              <div className="display" style={{ fontSize: 14, fontWeight: 700 }}>
                Open-source project
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>
                Built with RT-DETR, SAM 2.1 and Monodepth2 ·
                Paraschiv Tudor · Cluj-Napoca, 2026
              </div>
            </div>
            <Link to="/ingest" className="btn btn-accent">
              Try it: upload a survey <ArrowRight size={13} />
            </Link>
          </div>
        </FadeContent>
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100%',
    paddingTop: 'calc(var(--nav-h) + 40px)',
    paddingBottom: 48,
  },
  inner: {
    maxWidth: 960,
    margin: '0 auto',
    padding: '0 26px',
  },
  stageRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 14,
    padding: '15px 18px',
  },
  stageNum: {
    fontSize: 11,
    color: 'var(--text-muted)',
    paddingTop: 3,
    width: 20,
    flexShrink: 0,
  },
  stageIcon: {
    width: 38, height: 38, borderRadius: 10, flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    color: 'var(--accent)',
  },
  sevGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
    gap: 10,
  },
  th: {
    textAlign: 'left',
    padding: '11px 16px',
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-bright)',
    background: 'var(--bg-card2)',
  },
  td: {
    padding: '11px 16px',
    fontSize: 12.5,
    borderBottom: '1px solid var(--border)',
  },
  archGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
    gap: 12,
  },
  archCard: {
    padding: '18px 20px',
  },
  archText: {
    fontSize: 12,
    color: 'var(--text-dim)',
    lineHeight: 1.7,
    marginTop: 6,
  },
}
