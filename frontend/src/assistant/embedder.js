/**
 * frontend/src/assistant/embedder.js
 *
 * Sentence embeddings, in the browser, for free.
 *
 * Uses transformers.js with a small quantized MiniLM-class model (about
 * 25 MB, cached by the browser after the first load). It runs on WebGPU when
 * the device has it and falls back to WASM on the CPU otherwise.
 *
 * The whole module is optional: nothing imports it eagerly, and every caller
 * treats a failure as "no dense retrieval", not as an error. The assistant's
 * instant mode never touches it.
 */

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2'

let extractor = null
let loadPromise = null
let loadFailed = false

export function isEmbedderReady() {
  return extractor !== null
}

export function embedderFailed() {
  return loadFailed
}

/**
 * Load the embedding model. Safe to call repeatedly: the work happens once.
 * @param {(progress: object) => void} onProgress
 */
export async function loadEmbedder(onProgress) {
  if (extractor) return extractor
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    // Dynamic import keeps transformers.js out of the main bundle entirely.
    const { pipeline, env } = await import('@huggingface/transformers')

    // We ship no local model files; fetch from the hub and let the browser
    // cache them. (Allowed by the connect-src rules in nginx.conf.)
    env.allowLocalModels = false

    const device = (typeof navigator !== 'undefined' && navigator.gpu) ? 'webgpu' : 'wasm'

    extractor = await pipeline('feature-extraction', MODEL_ID, {
      dtype: 'q8',
      device,
      progress_callback: onProgress,
    })
    return extractor
  })()

  try {
    return await loadPromise
  } catch (e) {
    loadFailed = true
    loadPromise = null
    throw e
  }
}

/** Embed one string into a plain number array. */
export async function embed(text) {
  if (!extractor) throw new Error('Embedder is not loaded')
  const out = await extractor(text, { pooling: 'mean', normalize: true })
  return Array.from(out.data)
}

/** Embed many strings. Sequential on purpose: the batch is tiny and this
 *  keeps peak memory low on phones. */
export async function embedBatch(texts) {
  if (!extractor) throw new Error('Embedder is not loaded')
  const vectors = []
  for (const t of texts) {
    vectors.push(await embed(t))
  }
  return vectors
}
