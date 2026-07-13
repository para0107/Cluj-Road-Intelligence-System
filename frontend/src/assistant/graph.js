/**
 * frontend/src/assistant/graph.js
 *
 * The agentic pipeline, written as an explicit state machine.
 *
 *   guard ─▶ intents ─┬─▶ (data answer, no model involved) ─▶ done
 *                     │
 *                     └─▶ retrieve ─▶ [confident?] ─┬─ yes ─▶ generate ─▶ ground ─▶ done
 *                                                    │
 *                                                    └─ no ──▶ hyde ─▶ re-retrieve
 *                                                                  ─▶ [confident?] ─┬─ yes ─▶ generate ─▶ ground
 *                                                                                   └─ no ──▶ honest fallback
 *
 * Why a hand-rolled graph rather than a framework: the graph is fixed, it has
 * seven nodes, and it must run in a browser next to a 900 MB model download.
 * The framework would add megabytes and a build-time dependency chain to
 * express control flow that is legible as plain code, and the behaviour we
 * actually want (conditional HyDE, retrieval-confidence gating, lexical
 * grounding) lives in the nodes, not in the orchestration.
 *
 * Every step reports progress through `onStage` so the UI can show what the
 * assistant is doing instead of a silent spinner.
 */

import { KNOWLEDGE_BY_ID } from './knowledge'
import { retrieve, confidence } from './retrieval'
import { runIntents } from './intents'
import { guardInput, groundAnswer, REFUSALS } from './guard'
import { embed, isEmbedderReady } from './embedder'
import { generate, hypotheticalAnswer, isModelReady } from './localModel'

// Retrieval-confidence gates, overridable per deployment without a code
// change (VITE_ASSISTANT_CONFIDENT / VITE_ASSISTANT_FLOOR).
const ENV = (typeof import.meta !== 'undefined' && import.meta.env) || {}
// Below this retrieval confidence we do not trust the top hit enough to feed
// it to the model as if it were the answer.
const CONFIDENT = Number(ENV.VITE_ASSISTANT_CONFIDENT) || 0.55
// Below this, even after HyDE, we refuse to generate at all.
const FLOOR = Number(ENV.VITE_ASSISTANT_FLOOR) || 0.28

const MAX_CONTEXT_ENTRIES = 4

function buildContext(hits) {
  return hits
    .map(({ id }) => {
      const entry = KNOWLEDGE_BY_ID[id]
      return entry ? `### ${entry.title}\n${entry.text}` : ''
    })
    .filter(Boolean)
    .join('\n\n')
}

function sourcesOf(hits) {
  return hits
    .map(({ id }) => KNOWLEDGE_BY_ID[id])
    .filter(Boolean)
    .map((e) => ({ id: e.id, title: e.title, route: e.route || null }))
}

/**
 * Run one question through the assistant.
 *
 * @param {string} question
 * @param {object} opts
 * @param {boolean} opts.aiMode     use the in-browser model for phrasing
 * @param {(stage: string) => void} opts.onStage
 * @param {(chunk: string, full: string) => void} opts.onToken
 * @returns {Promise<{answer, sources, mode, refused?, dropped?}>}
 */
export async function ask(question, { aiMode = false, onStage, onToken } = {}) {
  const stage = (s) => onStage?.(s)

  // ── 1. guard ────────────────────────────────────────────────────────────
  stage('Checking the question')
  const guarded = guardInput(question)
  if (!guarded.ok) {
    return {
      answer: REFUSALS[guarded.reason] || REFUSALS.offTopic,
      sources: [],
      mode: 'refused',
      refused: true,
    }
  }
  const q = guarded.question

  // ── 2. intents: numbers come from the API, never from a model ───────────
  stage('Looking up live data')
  const intent = await runIntents(q)

  // ── 3. retrieve ─────────────────────────────────────────────────────────
  stage('Searching the guide')
  const useDense = aiMode && isEmbedderReady()
  let hits = await retrieve(q, { k: MAX_CONTEXT_ENTRIES, useDense })
  let score = confidence(hits)

  // An intent answered it outright. Return the real numbers and stop: there is
  // nothing a language model could add here except the risk of changing them.
  if (intent) {
    const intentSources = (intent.sources || [])
      .map((id) => KNOWLEDGE_BY_ID[id])
      .filter(Boolean)
      .map((e) => ({ id: e.id, title: e.title, route: e.route || null }))

    return {
      answer: intent.answer,
      sources: intentSources.length ? intentSources : sourcesOf(hits.slice(0, 2)),
      mode: 'data',
      route: intent.route,
    }
  }

  // ── 4. instant mode: answer straight from the best entry ────────────────
  if (!aiMode || !isModelReady()) {
    if (!hits.length || score < FLOOR) {
      return {
        answer: `${REFUSALS.unsupported}`,
        sources: sourcesOf(hits.slice(0, 3)),
        mode: 'fallback',
      }
    }
    const best = KNOWLEDGE_BY_ID[hits[0].id]
    return {
      answer: best.text,
      sources: sourcesOf(hits.slice(0, 3)),
      mode: 'instant',
      route: best.route || null,
    }
  }

  // ── 5. HyDE, only when the plain search was weak ────────────────────────
  // Writing a hypothetical answer and searching with THAT costs a small
  // generation, so it is worth it only when we are actually lost. On a
  // confident hit this branch never runs and the answer stays fast.
  if (score < CONFIDENT) {
    stage('Thinking of another way to search')
    try {
      const hypo = await hypotheticalAnswer(q)
      if (hypo && isEmbedderReady()) {
        const vec = await embed(hypo)
        const rewritten = await retrieve(q, {
          k: MAX_CONTEXT_ENTRIES,
          useDense: true,
          hydeVector: vec,
        })
        if (confidence(rewritten) > score) {
          hits = rewritten
          score = confidence(rewritten)
        }
      }
    } catch {
      // HyDE is an optimisation. If it fails we simply carry on with what the
      // first search found.
    }
  }

  // ── 6. refuse rather than invent ────────────────────────────────────────
  if (!hits.length || score < FLOOR) {
    return {
      answer: REFUSALS.unsupported,
      sources: sourcesOf(hits.slice(0, 3)),
      mode: 'fallback',
    }
  }

  // ── 7. generate, then ground ────────────────────────────────────────────
  stage('Writing an answer')
  const context = buildContext(hits)

  let raw
  try {
    raw = await generate({ question: q, context, onToken })
  } catch {
    // The model failed mid-answer (out of memory, tab backgrounded). Fall back
    // to the retrieved text, which is always safe to show.
    const best = KNOWLEDGE_BY_ID[hits[0].id]
    return {
      answer: best.text,
      sources: sourcesOf(hits.slice(0, 3)),
      mode: 'instant',
      route: best.route || null,
    }
  }

  stage('Checking the answer against the guide')
  const { text, dropped, grounded } = groundAnswer(raw, context, [])

  // The model wandered off the context entirely. Show the source text instead
  // of a polished sentence we cannot back up.
  if (!grounded) {
    const best = KNOWLEDGE_BY_ID[hits[0].id]
    return {
      answer: best.text,
      sources: sourcesOf(hits.slice(0, 3)),
      mode: 'instant',
      grounded: false,
      route: best.route || null,
    }
  }

  return {
    answer: text,
    sources: sourcesOf(hits.slice(0, 3)),
    mode: 'ai',
    dropped,
    route: KNOWLEDGE_BY_ID[hits[0].id]?.route || null,
  }
}
