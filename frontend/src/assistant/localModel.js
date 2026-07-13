/**
 * frontend/src/assistant/localModel.js
 *
 * The optional AI answer mode: a small instruct model running entirely inside
 * the user's browser through WebLLM (WebGPU).
 *
 * Cost model: the weights are a one-time download to the visitor's own browser
 * cache, and every token is generated on their GPU. RDDS pays nothing per
 * question and the server does no inference work at all, so the assistant
 * costs the same whether one person or ten thousand people use it.
 *
 * Everything here is dynamically imported so that a visitor who never turns on
 * AI mode downloads none of it.
 */

// About 880 MB of weights. Chosen over larger models because it must fit in
// VRAM on an ordinary laptop and start answering in a few seconds.
// Override per deployment with VITE_WEBLLM_MODEL (any model id from the
// WebLLM prebuilt list, e.g. Qwen2.5-1.5B-Instruct-q4f16_1-MLC for an
// Apache-2.0 model with no attribution clause).
const ENV = (typeof import.meta !== 'undefined' && import.meta.env) || {}
export const MODEL_ID = ENV.VITE_WEBLLM_MODEL || 'Llama-3.2-1B-Instruct-q4f16_1-MLC'

let engine = null
let loadPromise = null

export function isModelReady() {
  return engine !== null
}

/** WebGPU is required. Safari and older browsers will not have it. */
export function isSupported() {
  return typeof navigator !== 'undefined' && Boolean(navigator.gpu)
}

/**
 * Download (once) and start the model.
 * @param {(report: {progress: number, text: string}) => void} onProgress
 */
export async function loadModel(onProgress) {
  if (engine) return engine
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const { CreateMLCEngine } = await import('@mlc-ai/web-llm')
    engine = await CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (report) => {
        onProgress?.({
          progress: report.progress ?? 0,
          text: report.text || 'Loading the model',
        })
      },
    })
    return engine
  })()

  try {
    return await loadPromise
  } catch (e) {
    engine = null
    loadPromise = null
    throw e
  }
}

/** Free the GPU memory (called when the user leaves the page). */
export async function unloadModel() {
  try {
    await engine?.unload?.()
  } catch {
    // Nothing useful to do if teardown fails; the tab is going away anyway.
  }
  engine = null
  loadPromise = null
}

/**
 * Generate an answer, streaming tokens as they arrive.
 *
 * The system prompt is a fixed prefix followed by the retrieved context. The
 * fixed part comes first on purpose: WebLLM can reuse the KV cache for an
 * unchanged prefix across turns, so the constant instructions are not
 * re-processed on every question.
 *
 * @param {object} args
 * @param {string} args.question
 * @param {string} args.context   retrieved knowledge, already assembled
 * @param {string} args.liveLine  one line of live numbers, or ''
 * @param {(chunk: string, full: string) => void} args.onToken
 * @param {number} args.maxTokens
 * @returns {Promise<string>} the full raw answer
 */
export async function generate({ question, context, liveLine = '', onToken, maxTokens = 320 }) {
  if (!engine) throw new Error('The model is not loaded')

  const system = [
    'You are the RDDS assistant. RDDS is a road damage detection system: it finds',
    'road damage in dashcam video, takes hazard reports from drivers, and helps a',
    'city plan repairs.',
    '',
    'Rules you must follow:',
    '1. Answer ONLY using the CONTEXT below. It is the only thing you know.',
    '2. If the context does not answer the question, say you do not know and',
    '   suggest the closest topic that is in the context. Never guess.',
    '3. Never invent numbers, prices, dates, or feature names.',
    '4. Be brief: two to four plain sentences. No lists unless asked.',
    '5. Write plain sentences. Do not use dashes as punctuation.',
    '',
    'CONTEXT:',
    context,
    liveLine ? `\nLIVE NUMBERS (these are current and true):\n${liveLine}` : '',
  ].join('\n')

  const stream = await engine.chat.completions.create({
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: question },
    ],
    temperature: 0.2,
    max_tokens: maxTokens,
    stream: true,
  })

  let full = ''
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content || ''
    if (delta) {
      full += delta
      onToken?.(delta, full)
    }
  }
  return full.trim()
}

/**
 * HyDE: ask the model to write the answer it *imagines*, then embed that and
 * search with it. A hypothetical answer shares far more vocabulary with the
 * right knowledge entry than a terse question does, which is what makes this
 * work. Kept short and cheap, and only used when normal retrieval was weak.
 */
export async function hypotheticalAnswer(question) {
  if (!engine) return null
  const res = await engine.chat.completions.create({
    messages: [
      {
        role: 'system',
        content:
          'Write one short sentence that could plausibly answer the question about a road damage reporting system. Do not say you are unsure. Do not ask a question.',
      },
      { role: 'user', content: question },
    ],
    temperature: 0.3,
    max_tokens: 60,
  })
  return res.choices?.[0]?.message?.content?.trim() || null
}
