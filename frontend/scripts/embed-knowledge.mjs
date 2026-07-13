/**
 * frontend/scripts/embed-knowledge.mjs
 *
 * Precompute the dense-retrieval vectors for the assistant's knowledge base.
 *
 *   node scripts/embed-knowledge.mjs        (run from frontend/)
 *
 * Run this once after editing src/assistant/knowledge.js. It writes
 * src/assistant/knowledge_vectors.json, which ships in the assistant's lazy
 * chunk so a phone never has to embed 30+ entries itself; it only embeds the
 * user's question. The runtime verifies both the content hash and the model
 * id, so a stale or mismatched file is ignored (with a silent fallback to
 * embedding in the browser), never silently wrong.
 */

import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import { KNOWLEDGE, knowledgeHash } from '../src/assistant/knowledge.js'
import { EMBEDDER_MODEL_ID } from '../src/assistant/embedder.js'

const here = dirname(fileURLToPath(import.meta.url))
const outPath = join(here, '..', 'src', 'assistant', 'knowledge_vectors.json')

const { pipeline } = await import('@huggingface/transformers')

console.log(`Embedding ${KNOWLEDGE.length} entries with ${EMBEDDER_MODEL_ID} ...`)
const extractor = await pipeline('feature-extraction', EMBEDDER_MODEL_ID, { dtype: 'q8' })

const vectors = {}
for (const k of KNOWLEDGE) {
  const out = await extractor(`${k.title}. ${k.text}`, { pooling: 'mean', normalize: true })
  // 6 decimals keeps the file small; cosine ranking is insensitive to the loss.
  vectors[k.id] = Array.from(out.data).map((v) => Number(v.toFixed(6)))
  process.stdout.write('.')
}
console.log()

const payload = {
  model: EMBEDDER_MODEL_ID,
  hash: knowledgeHash(),
  dim: vectors[KNOWLEDGE[0].id].length,
  count: KNOWLEDGE.length,
  vectors,
}

writeFileSync(outPath, JSON.stringify(payload))
console.log(`Wrote ${outPath} (${KNOWLEDGE.length} vectors, dim ${payload.dim}, hash ${payload.hash})`)
