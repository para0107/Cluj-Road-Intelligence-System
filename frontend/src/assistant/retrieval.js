/**
 * frontend/src/assistant/retrieval.js
 *
 * Hybrid retrieval over the knowledge base.
 *
 *   sparse  MiniSearch (BM25-style, fuzzy + prefix). Instant, works on every
 *           device, no download.
 *   dense   Optional. A small sentence-embedding model runs in the browser
 *           (transformers.js) and scores by cosine similarity, which catches
 *           questions that share no words with the entry ("my road is
 *           wrecked, who fixes it?").
 *   fusion  Reciprocal Rank Fusion. It needs no score calibration between the
 *           two systems, which is exactly the problem with mixing BM25 scores
 *           and cosine similarities directly.
 *
 * The dense half is lazy and entirely optional: if the model never loads, the
 * assistant still answers from sparse retrieval alone.
 */

import MiniSearch from 'minisearch'
import { KNOWLEDGE, knowledgeHash } from './knowledge'
import { embed, embedBatch, isEmbedderReady, EMBEDDER_MODEL_ID } from './embedder'

// ── Sparse index ──────────────────────────────────────────────────────────

const mini = new MiniSearch({
  fields: ['title', 'text', 'tags'],
  storeFields: ['id'],
  searchOptions: {
    boost: { title: 3, tags: 2 },
    fuzzy: 0.2,
    prefix: true,
  },
  // tags is an array; MiniSearch wants a string per field.
  extractField: (doc, fieldName) =>
    fieldName === 'tags' ? (doc.tags || []).join(' ') : doc[fieldName],
})

mini.addAll(KNOWLEDGE)

// Everyday words people actually type, mapped to the domain vocabulary the
// knowledge base uses. Sparse search is the only retrieval most users ever
// get (instant mode), so closing this vocabulary gap is the cheapest
// quality win in the whole assistant.
const SYNONYMS = {
  hole: 'pothole', holes: 'pothole', crater: 'pothole', bump: 'pothole',
  broken: 'damage', wrecked: 'damage', ruined: 'damage', destroyed: 'damage',
  reward: 'points', rewards: 'points', prize: 'points', score: 'points',
  money: 'cost pricing', pay: 'cost pricing', fee: 'cost pricing',
  photo: 'evidence picture', picture: 'evidence photo',
  fixing: 'repair', fixes: 'repair', fixed: 'repair',
  alert: 'notification', alerts: 'notification',
  ranking: 'leaderboard', rankings: 'leaderboard',
  car: 'vehicle drive', dashboard: 'stats',
  town: 'city', council: 'municipality city',
  worker: 'crew', workers: 'crew',
  danger: 'hazard severity', dangerous: 'hazard severity',
}

function expandQuery(query) {
  const words = query.toLowerCase().match(/[a-z]+/g) || []
  const extra = new Set()
  for (const w of words) {
    if (SYNONYMS[w]) extra.add(SYNONYMS[w])
  }
  return extra.size ? `${query} ${[...extra].join(' ')}` : query
}

export function sparseSearch(query, k = 6) {
  return mini
    .search(expandQuery(query))
    .slice(0, k)
    .map((r) => r.id)
}

// ── Dense index (built once, in memory) ───────────────────────────────────

let denseVectors = null      // [{ id, vec }]
let densePromise = null

/**
 * Build the dense index. Prefers vectors precomputed at build time by
 * `scripts/embed-knowledge.mjs` (a lazy-loaded ~100 kB JSON chunk), which
 * turns "embed 33 entries on a phone" into "fetch one small file". Falls
 * back to embedding the KB in the browser when the file is missing or stale.
 */
export async function buildDenseIndex() {
  if (denseVectors) return denseVectors
  if (densePromise) return densePromise

  densePromise = (async () => {
    try {
      const { default: pre } = await import('./knowledge_vectors.json')
      // Only trust the file if it was built from THIS knowledge base with
      // THIS embedding model; query and index vectors must share a space.
      if (pre?.hash === knowledgeHash() && pre.model === EMBEDDER_MODEL_ID && pre.vectors) {
        const fromFile = KNOWLEDGE
          .filter((k) => pre.vectors[k.id])
          .map((k) => ({ id: k.id, vec: pre.vectors[k.id] }))
        if (fromFile.length === KNOWLEDGE.length) {
          denseVectors = fromFile
          return denseVectors
        }
      }
    } catch {
      // No precomputed file in this build. Embedding at runtime below.
    }

    const texts = KNOWLEDGE.map((k) => `${k.title}. ${k.text}`)
    const vecs = await embedBatch(texts)
    denseVectors = KNOWLEDGE.map((k, i) => ({ id: k.id, vec: vecs[i] }))
    return denseVectors
  })()

  try {
    return await densePromise
  } catch (e) {
    densePromise = null
    throw e
  }
}

function cosine(a, b) {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export async function denseSearch(query, k = 6) {
  if (!isEmbedderReady()) return []
  const index = await buildDenseIndex()
  const qv = await embed(query)
  return index
    .map(({ id, vec }) => ({ id, score: cosine(qv, vec) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.id)
}

/** Same, but for an already-computed vector (used by the HyDE step). */
export async function denseSearchVector(vec, k = 6) {
  if (!isEmbedderReady() || !vec) return []
  const index = await buildDenseIndex()
  return index
    .map((entry) => ({ id: entry.id, score: cosine(vec, entry.vec) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.id)
}

// ── Fusion ────────────────────────────────────────────────────────────────

const RRF_K = 60

/**
 * Reciprocal Rank Fusion over any number of ranked id lists.
 * Returns [{ id, score }] sorted best first. The score is a fusion score,
 * not a probability: use it only to compare candidates with each other.
 */
export function fuse(...rankedLists) {
  const scores = new Map()
  for (const list of rankedLists) {
    if (!list) continue
    list.forEach((id, rank) => {
      scores.set(id, (scores.get(id) || 0) + 1 / (RRF_K + rank + 1))
    })
  }
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score)
}

/**
 * The retrieval entry point.
 *
 * @param {string} query
 * @param {object} opts  { k, useDense, hydeVector }
 * @returns {Promise<Array<{id, score}>>}
 */
export async function retrieve(query, { k = 4, useDense = false, hydeVector = null } = {}) {
  const sparse = sparseSearch(query, 6)

  let dense = []
  if (useDense) {
    try {
      dense = hydeVector
        ? await denseSearchVector(hydeVector, 6)
        : await denseSearch(query, 6)
    } catch {
      dense = []   // dense retrieval is a bonus, never a hard requirement
    }
  }

  return fuse(sparse, dense).slice(0, k)
}

/**
 * A crude confidence signal for the top hit, used to decide whether to run
 * HyDE and whether to answer at all. Anything at or above ~0.45 of the best
 * possible RRF score (both systems ranking it first) is a solid hit.
 */
export const BEST_POSSIBLE_RRF = 2 / (RRF_K + 1)

export function confidence(fused) {
  if (!fused.length) return 0
  return fused[0].score / BEST_POSSIBLE_RRF
}
