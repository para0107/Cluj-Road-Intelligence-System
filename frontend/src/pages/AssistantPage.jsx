/**
 * frontend/src/pages/AssistantPage.jsx
 *
 * The RDDS assistant (/assistant).
 *
 * Two modes, both free and both entirely on the visitor's device:
 *
 *   Instant   the default. Searches a built in guide and answers data
 *             questions with real numbers from the API. No download, works on
 *             every phone.
 *
 *   AI mode   opt in. A small language model is downloaded once into the
 *             browser cache and then runs on the device's GPU, so it phrases
 *             answers in its own words. Nothing is sent to any company and
 *             RDDS pays nothing per question.
 *
 * Every AI answer is checked against the retrieved guide text before it is
 * shown (see assistant/guard.js), so the assistant would rather say "I do not
 * know" than invent something.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Send, Sparkles, Cpu, BookOpen, ShieldCheck, ArrowRight, Loader2, Info,
} from 'lucide-react'

import { SectionTitle, Spinner } from '../components/ui'
import { SUGGESTIONS } from '../assistant/knowledge'
import { ask } from '../assistant/graph'
import { isSupported, loadModel, isModelReady, unloadModel, MODEL_ID } from '../assistant/localModel'
import { loadEmbedder } from '../assistant/embedder'
import { buildDenseIndex } from '../assistant/retrieval'

const AI_FLAG = 'rids_assistant_ai'

const GREETING = {
  role: 'assistant',
  mode: 'instant',
  text:
    'Hello. I can explain how RDDS works, what the severity levels mean, how points and badges are earned, and how a city runs repairs. I can also look up live numbers, like how many hazards are open right now. Ask me anything about RDDS.',
  sources: [],
}

export default function AssistantPage() {
  const navigate = useNavigate()

  const [messages, setMessages] = useState([GREETING])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState('')
  const [streaming, setStreaming] = useState('')

  // AI mode
  const supported = isSupported()
  const [aiMode, setAiMode] = useState(false)
  const [showConsent, setShowConsent] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState({ pct: 0, text: '' })
  const [loadError, setLoadError] = useState(null)

  const feedRef = useRef(null)
  const aliveRef = useRef(true)

  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
    }
  }, [])

  // Free the GPU when the user navigates away.
  useEffect(() => () => { unloadModel() }, [])

  // Keep the newest message in view.
  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, streaming, stage])

  // If the model was enabled in a previous visit it is already in the browser
  // cache, so turning it back on is quick. We still ask before downloading.
  useEffect(() => {
    if (supported && localStorage.getItem(AI_FLAG) === '1' && !isModelReady()) {
      startAi(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [supported])

  const startAi = useCallback(async (skipConsent = false) => {
    if (!supported) return
    if (!skipConsent && localStorage.getItem(AI_FLAG) !== '1') {
      setShowConsent(true)
      return
    }

    setShowConsent(false)
    setLoading(true)
    setLoadError(null)

    try {
      // The embedder is small and makes retrieval much better, so it loads
      // first and the chat model follows.
      setProgress({ pct: 0.02, text: 'Loading the search model' })
      await loadEmbedder((p) => {
        if (aliveRef.current && p?.status === 'progress' && p.total) {
          setProgress({ pct: 0.02 + 0.13 * (p.loaded / p.total), text: 'Loading the search model' })
        }
      })
      await buildDenseIndex()

      await loadModel(({ progress: pct, text }) => {
        if (aliveRef.current) {
          setProgress({ pct: 0.15 + 0.85 * (pct || 0), text: text || 'Loading the language model' })
        }
      })

      if (!aliveRef.current) return
      localStorage.setItem(AI_FLAG, '1')
      setAiMode(true)
      setLoading(false)
      setProgress({ pct: 1, text: 'Ready' })
    } catch (e) {
      if (!aliveRef.current) return
      setLoading(false)
      setLoadError(
        'The AI model could not start on this device. The assistant still works in its instant mode.',
      )
      localStorage.removeItem(AI_FLAG)
      setAiMode(false)
    }
  }, [supported])

  const stopAi = () => {
    localStorage.removeItem(AI_FLAG)
    setAiMode(false)
    unloadModel()
  }

  const send = useCallback(async (raw) => {
    const question = (raw ?? input).trim()
    if (!question || busy) return

    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setBusy(true)
    setStreaming('')
    setStage('')

    try {
      const result = await ask(question, {
        aiMode: aiMode && isModelReady(),
        onStage: (s) => aliveRef.current && setStage(s),
        onToken: (_chunk, full) => aliveRef.current && setStreaming(full),
      })

      if (!aliveRef.current) return
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: result.answer,
          sources: result.sources || [],
          mode: result.mode,
          route: result.route || null,
        },
      ])
    } catch (e) {
      if (!aliveRef.current) return
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          mode: 'error',
          text: 'Something went wrong while answering. Try asking again.',
          sources: [],
        },
      ])
    } finally {
      if (aliveRef.current) {
        setBusy(false)
        setStreaming('')
        setStage('')
      }
    }
  }, [input, busy, aiMode])

  return (
    <div className="page-grid-bg" style={styles.page}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="anim-fade-up">
        <SectionTitle
          overline="ASK RDDS"
          title="Assistant"
          right={
            <ModeSwitch
              supported={supported}
              aiMode={aiMode}
              loading={loading}
              onEnable={() => startAi(false)}
              onDisable={stopAi}
            />
          }
        />
        <p style={styles.lede}>
          I answer from a built in guide and from your city's live numbers.{' '}
          {aiMode
            ? 'AI mode is on, so answers are written by a model running inside your browser. I still check every sentence against the guide before showing it.'
            : 'Turn on AI mode for answers written in full sentences. The model runs on your own device, so nothing you type ever leaves your browser.'}
        </p>
      </div>

      {loadError && (
        <div style={styles.errorBanner}>
          <Info size={14} /> {loadError}
        </div>
      )}

      {loading && (
        <div className="card" style={styles.loadingCard}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
            <Loader2 size={15} style={{ color: 'var(--accent)', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>{progress.text}</span>
            <span className="mono" style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
              {Math.round((progress.pct || 0) * 100)}%
            </span>
          </div>
          <div style={styles.progressTrack}>
            <div style={{ ...styles.progressFill, width: `${Math.round((progress.pct || 0) * 100)}%` }} />
          </div>
          <div style={styles.loadingNote}>
            This happens once. The model is saved in your browser, so next time it starts straight away.
          </div>
        </div>
      )}

      {/* ── Chat ───────────────────────────────────────────────────────── */}
      <div className="card" style={styles.chatCard}>
        <div ref={feedRef} style={styles.feed}>
          {messages.map((m, i) => (
            <Message key={i} m={m} onRoute={(r) => navigate(r)} />
          ))}

          {busy && (
            <div style={styles.thinking}>
              {streaming ? (
                <div style={styles.bubbleAssistant}>
                  <div style={styles.bubbleText}>{streaming}</div>
                  <span style={styles.caret} />
                </div>
              ) : (
                <div style={styles.stageRow}>
                  <Spinner size={14} />
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {stage || 'Thinking'}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Suggestions */}
        {messages.length <= 1 && !busy && (
          <div style={styles.suggestions}>
            {SUGGESTIONS.map((s) => (
              <button key={s} className="chip" style={styles.suggestion} onClick={() => send(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        {/* Composer */}
        <form
          style={styles.composer}
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="Ask about reporting, severity, points, or repairs"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
            maxLength={500}
          />
          <button
            type="submit"
            className="btn btn-accent"
            disabled={busy || !input.trim()}
            style={{ flexShrink: 0 }}
          >
            <Send size={14} /> Ask
          </button>
        </form>
      </div>

      {/* ── How it works ───────────────────────────────────────────────── */}
      <div style={styles.notes}>
        <Note icon={BookOpen} title="It answers from a guide">
          Every answer comes from a written guide to RDDS and from live numbers out of the
          API. If the guide does not cover your question, the assistant says so instead of
          guessing.
        </Note>
        <Note icon={Cpu} title="It runs on your device">
          AI mode downloads a small model once and runs it in your browser. Your questions
          never reach an outside company, and RDDS pays nothing to answer them.
        </Note>
        <Note icon={ShieldCheck} title="It is checked before you see it">
          Each sentence the model writes is compared against the guide. Anything it cannot
          support, including any number it made up, is removed.
        </Note>
      </div>

      {/* ── Consent ────────────────────────────────────────────────────── */}
      {showConsent && (
        <div style={styles.overlay} onClick={() => setShowConsent(false)}>
          <div className="card" style={styles.consent} onClick={(e) => e.stopPropagation()}>
            <div className="overline" style={{ marginBottom: 8 }}>ONE TIME DOWNLOAD</div>
            <h3 className="display" style={{ fontSize: 19, marginBottom: 10 }}>Turn on AI answers</h3>
            <p style={styles.consentText}>
              This downloads a language model of about 900 MB once and keeps it in your
              browser. After that it runs on your own device, offline, and nothing you type
              is ever sent anywhere.
            </p>
            <p style={styles.consentText}>
              It needs a recent desktop browser with WebGPU and works best on a machine with
              a dedicated graphics card. On a slow connection the first download takes a few
              minutes.
            </p>
            <div className="mono" style={styles.modelId}>{MODEL_ID}</div>
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => setShowConsent(false)}>
                Not now
              </button>
              <button
                className="btn btn-accent"
                style={{ flex: 1 }}
                onClick={() => {
                  localStorage.setItem(AI_FLAG, '1')
                  startAi(true)
                }}
              >
                Download and start
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Pieces ────────────────────────────────────────────────────────────────

function ModeSwitch({ supported, aiMode, loading, onEnable, onDisable }) {
  if (!supported) {
    return (
      <span style={styles.unsupported} title="AI mode needs a browser with WebGPU, such as Chrome or Edge on a desktop.">
        <Info size={11} /> AI mode needs WebGPU
      </span>
    )
  }
  if (loading) {
    return <span style={styles.unsupported}><Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> Starting</span>
  }
  return aiMode ? (
    <button className="btn btn-sm" style={styles.aiOn} onClick={onDisable} title="Switch back to instant answers">
      <Sparkles size={12} /> AI mode on
    </button>
  ) : (
    <button className="btn btn-ghost btn-sm" onClick={onEnable}>
      <Sparkles size={12} /> Turn on AI mode
    </button>
  )
}

function Message({ m, onRoute }) {
  if (m.role === 'user') {
    return (
      <div style={styles.rowUser}>
        <div style={styles.bubbleUser}>{m.text}</div>
      </div>
    )
  }

  return (
    <div style={styles.rowAssistant} className="anim-fade-in">
      <div style={styles.bubbleAssistant}>
        <div style={styles.bubbleText}>{m.text}</div>

        {m.mode && m.mode !== 'error' && (
          <div style={styles.modeTag}>
            {m.mode === 'ai' && <><Sparkles size={9} /> written by the on-device model</>}
            {m.mode === 'data' && <><Sparkles size={9} /> live numbers from your city</>}
            {m.mode === 'instant' && <><BookOpen size={9} /> from the guide</>}
            {m.mode === 'fallback' && <><Info size={9} /> not in the guide</>}
            {m.mode === 'refused' && <><ShieldCheck size={9} /> off topic</>}
          </div>
        )}

        {m.sources?.length > 0 && (
          <div style={styles.sources}>
            {m.sources.map((s) => (
              <button
                key={s.id}
                className="chip"
                style={styles.source}
                onClick={() => s.route && onRoute(s.route)}
                title={s.route ? `Open ${s.route}` : s.title}
              >
                {s.title}
                {s.route && <ArrowRight size={9} />}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Note({ icon: Icon, title, children }) {
  return (
    <div className="card" style={styles.note}>
      <div style={styles.noteHead}>
        <Icon size={14} style={{ color: 'var(--accent)' }} />
        <span style={{ fontWeight: 700, fontSize: 12.5 }}>{title}</span>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.6 }}>{children}</div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────

const styles = {
  page: {
    paddingTop: 'calc(var(--nav-h) + 28px)',
    paddingBottom: 60,
    paddingLeft: 20,
    paddingRight: 20,
    maxWidth: 900,
    margin: '0 auto',
    minHeight: '100vh',
  },
  lede: {
    fontSize: 13.5,
    color: 'var(--text-dim)',
    lineHeight: 1.7,
    margin: '10px 0 20px',
    maxWidth: 660,
  },
  errorBanner: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 14px', marginBottom: 14,
    borderRadius: 'var(--radius)',
    border: '1px solid var(--orange)',
    background: 'color-mix(in srgb, var(--orange) 10%, transparent)',
    color: 'var(--orange)', fontSize: 12.5,
  },
  loadingCard: { padding: 16, marginBottom: 14 },
  progressTrack: {
    height: 6, borderRadius: 999, background: 'var(--bg-card2)',
    border: '1px solid var(--border)', overflow: 'hidden',
  },
  progressFill: {
    height: '100%', background: 'var(--grad-accent)',
    transition: 'width 0.3s ease',
  },
  loadingNote: { fontSize: 11.5, color: 'var(--text-muted)', marginTop: 8 },

  chatCard: { padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  feed: {
    padding: 18,
    display: 'flex', flexDirection: 'column', gap: 14,
    minHeight: 320, maxHeight: '52vh', overflowY: 'auto',
  },
  rowUser: { display: 'flex', justifyContent: 'flex-end' },
  rowAssistant: { display: 'flex', justifyContent: 'flex-start' },
  bubbleUser: {
    maxWidth: '78%', padding: '10px 14px',
    borderRadius: '14px 14px 4px 14px',
    background: 'var(--accent-dim)',
    border: '1px solid var(--border-accent)',
    color: 'var(--text)', fontSize: 13.5, lineHeight: 1.6,
  },
  bubbleAssistant: {
    maxWidth: '86%', padding: '12px 15px',
    borderRadius: '14px 14px 14px 4px',
    background: 'var(--bg-card2)',
    border: '1px solid var(--border)',
  },
  bubbleText: { fontSize: 13.5, lineHeight: 1.7, color: 'var(--text)', whiteSpace: 'pre-wrap' },
  caret: {
    display: 'inline-block', width: 7, height: 14,
    background: 'var(--accent)', marginLeft: 3, verticalAlign: 'text-bottom',
    animation: 'pulse 1s ease-in-out infinite',
  },
  modeTag: {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    marginTop: 9, fontSize: 9.5, fontFamily: 'var(--font-mono)',
    letterSpacing: '0.06em', textTransform: 'uppercase',
    color: 'var(--text-muted)',
  },
  sources: { display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 9 },
  source: {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    fontSize: 10.5, padding: '3px 8px', cursor: 'pointer',
  },
  thinking: { display: 'flex', justifyContent: 'flex-start' },
  stageRow: {
    display: 'flex', alignItems: 'center', gap: 9,
    padding: '10px 14px', borderRadius: 12,
    background: 'var(--bg-card2)', border: '1px solid var(--border)',
  },
  suggestions: {
    display: 'flex', flexWrap: 'wrap', gap: 6,
    padding: '0 18px 14px',
  },
  suggestion: { fontSize: 11.5, cursor: 'pointer' },
  composer: {
    display: 'flex', gap: 8, padding: 14,
    borderTop: '1px solid var(--border)',
    background: 'var(--bg-card)',
  },
  notes: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 12, marginTop: 18,
  },
  note: { padding: 14 },
  noteHead: { display: 'flex', alignItems: 'center', gap: 7, marginBottom: 7 },
  unsupported: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    fontSize: 11, color: 'var(--text-muted)',
  },
  aiOn: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    border: '1px solid var(--border-accent)',
    background: 'var(--accent-dim)', color: 'var(--accent)',
  },
  overlay: {
    position: 'fixed', inset: 0, zIndex: 2000,
    background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(3px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 20,
  },
  consent: { maxWidth: 440, padding: 22 },
  consentText: { fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.7, marginBottom: 10 },
  modelId: {
    fontSize: 10.5, color: 'var(--text-muted)',
    padding: '6px 9px', borderRadius: 6,
    background: 'var(--bg-card2)', border: '1px solid var(--border)',
    overflowX: 'auto', whiteSpace: 'nowrap',
  },
}
