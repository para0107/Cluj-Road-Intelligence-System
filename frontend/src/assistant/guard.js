/**
 * frontend/src/assistant/guard.js
 *
 * The assistant's guardrails. Two jobs:
 *
 *   guardInput()   before anything runs: reject prompt-injection attempts and
 *                  questions that have nothing to do with roads or RDDS, and
 *                  cap the length.
 *
 *   groundAnswer() after the model speaks: drop any sentence the retrieved
 *                  context does not support, and strip any number that does
 *                  not appear in the context or in the live data we fetched.
 *
 * There is deliberately NO model-as-judge here. A judge would mean a second
 * generation pass, which on a 1B model running on a phone is both slow and
 * roughly as unreliable as the thing it is judging. Lexical grounding is
 * cheap, deterministic, and it fails in the safe direction: when in doubt it
 * removes the claim rather than keeping it.
 */

const MAX_QUESTION_CHARS = 500

// Classic instruction-override patterns. This is not a complete defence
// against a determined attacker, and it does not need to be: the model runs
// on the user's own device with no secrets and no tools, so the worst case of
// a successful injection is that a user makes their own browser say something
// odd. The check exists to keep normal answers on topic.
const INJECTION_PATTERNS = [
  /ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction)/i,
  /disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above)/i,
  /forget\s+(everything|all|your\s+(instruction|rule|prompt))/i,
  /you\s+are\s+now\s+(a|an|no longer)/i,
  /(reveal|show|print|repeat)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instruction)/i,
  /\bDAN\b|\bjailbreak\b/i,
  /pretend\s+(to\s+be|you\s+are)/i,
  /act\s+as\s+(a|an)\s+(?!road|city|rdds)/i,
]

// Words that mean the question is plausibly about this product's domain.
//
// Deliberately contains NO question words ("how", "what", "why"). An earlier
// version listed them and the gate became useless: "what is the capital of
// France?" matched on "what" and sailed through. The gate must key on the
// subject, never on the sentence being a question.
const ON_TOPIC = [
  'road', 'street', 'pothole', 'crack', 'damage', 'hazard', 'repair', 'fix',
  'severity', 'report', 'detect', 'survey', 'drive', 'driving', 'car', 'vehicle',
  'dashcam', 'gps', 'map', 'live', 'point', 'badge', 'streak', 'leaderboard',
  'rank', 'city', 'municipal', 'crew', 'work order', 'workorder', 'triage',
  'quality', 'rqi', 'api', 'key', 'account', 'sign in', 'signin', 'log in',
  'login', 'register', 'password', 'role', 'admin', 'operator', 'citizen',
  'upload', 'video', 'pipeline', 'stats', 'statistic', 'budget', 'cost',
  'price', 'pricing', 'plan', 'device', 'pair', 'notification', 'rdds',
  'assistant', 'privacy', 'data', 'export', 'csv', 'geojson', 'depth',
  'confirm', 'dispute', 'verify', 'lane', 'manhole', 'rutting', 'asphalt',
  'pavement', 'traffic', 'repaired', 'inspect', 'sensor', 'camera',
]

// Severity codes (S1..S5) are on topic even though they are not words.
const ON_TOPIC_PATTERN = /\bs[1-5]\b/i

// Whole-word matching matters here. A plain substring test looks fine until you
// notice that "capital" contains "api", so "what is the capital of France?"
// counted as an API question and passed the gate. Single words are matched on
// word boundaries; multi-word entries ("work order") stay substring matches.
const ON_TOPIC_WORDS = new Set(ON_TOPIC.filter((w) => !w.includes(' ')))
const ON_TOPIC_PHRASES = ON_TOPIC.filter((w) => w.includes(' '))

function isOnTopic(lower) {
  if (ON_TOPIC_PATTERN.test(lower)) return true
  if (ON_TOPIC_PHRASES.some((p) => lower.includes(p))) return true

  // Match on stems too, so "reporting", "repairs" and "detected" all count.
  const words = lower.match(/[a-z]+/g) || []
  return words.some((w) => {
    if (ON_TOPIC_WORDS.has(w)) return true
    for (const root of ON_TOPIC_WORDS) {
      if (root.length >= 4 && w.startsWith(root)) return true
    }
    return false
  })
}

export const REFUSALS = {
  injection:
    'I can only help with questions about RDDS and road damage. Ask me how reporting works, what a severity level means, or how the repair workflow runs.',
  offTopic:
    'I only know about RDDS: reporting road damage, how severity and points work, and how a city uses the repair tools. Ask me something in that area and I will help.',
  empty: 'Ask me a question about RDDS and I will do my best.',
  unsupported:
    'I do not have that in my guide, so I will not guess. Here are the closest topics I do know about.',
}

/**
 * @returns {{ ok: boolean, reason?: string, question?: string }}
 */
export function guardInput(raw) {
  let question = (raw || '').trim()

  if (!question) return { ok: false, reason: 'empty' }

  // Truncate, then KEEP CHECKING. An early return here once let any input
  // longer than the cap skip the injection and topic gates entirely, so
  // padding a jailbreak past 500 characters walked straight through.
  if (question.length > MAX_QUESTION_CHARS) {
    question = question.slice(0, MAX_QUESTION_CHARS)
  }

  if (INJECTION_PATTERNS.some((re) => re.test(question))) {
    return { ok: false, reason: 'injection' }
  }

  if (!isOnTopic(question.toLowerCase())) {
    return { ok: false, reason: 'offTopic' }
  }

  return { ok: true, question }
}

// ── Grounding ─────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this',
  'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'to',
  'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as', 'it', 'its',
  'you', 'your', 'we', 'our', 'they', 'their', 'can', 'will', 'would', 'may',
  'might', 'do', 'does', 'did', 'not', 'no', 'so', 'up', 'out', 'about',
  'into', 'over', 'also', 'more', 'most', 'other', 'some', 'any', 'each',
  'when', 'where', 'which', 'who', 'how', 'what', 'why', 'there', 'here',
])

/**
 * Crude English stemmer, just enough that "repairs", "repaired" and
 * "repairing" all count as "repair" during the overlap check. Without this
 * the grounding gate dropped honest sentences whose only sin was using a
 * different inflection than the knowledge entry.
 */
function stem(w) {
  if (w.length <= 4) return w
  return w
    .replace(/ies$/, 'y')
    .replace(/(sses|shes|ches|xes)$/, (m) => m.slice(0, -2))
    .replace(/ings?$/, '')
    .replace(/ed$/, '')
    .replace(/s$/, '')
}

function contentWords(text) {
  return (text.toLowerCase().match(/[a-z][a-z-]{2,}/g) || [])
    .filter((w) => !STOPWORDS.has(w))
    .map(stem)
}

function splitSentences(text) {
  return text
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

/** Numbers the answer is allowed to contain: anything present in the context
 *  or in the facts we fetched from the API. */
function allowedNumbers(context, facts) {
  const set = new Set((facts || []).map(String))
  for (const n of context.match(/\d+(?:[.,]\d+)?/g) || []) set.add(n)
  // Small numbers are almost always structural ("two devices", "1 of 5") and
  // are covered by the knowledge base text itself, which is in `context`.
  return set
}

const SUPPORT_THRESHOLD = 0.42

/**
 * Filter a generated answer down to what the context actually supports.
 *
 * @param {string} answer     the model's raw output
 * @param {string} context    the retrieved knowledge text given to the model
 * @param {string[]} facts    numbers fetched from the API this turn
 * @returns {{ text: string, dropped: number, grounded: boolean }}
 */
export function groundAnswer(answer, context, facts = []) {
  const sentences = splitSentences(answer || '')
  if (!sentences.length) return { text: '', dropped: 0, grounded: false }

  const contextWords = new Set(contentWords(context))
  const numbersOk = allowedNumbers(context, facts)

  const kept = []
  let dropped = 0

  for (const sentence of sentences) {
    // A number the context never mentions is a fabrication. Drop the sentence.
    const nums = sentence.match(/\d+(?:[.,]\d+)?/g) || []
    if (nums.some((n) => !numbersOk.has(n))) {
      dropped++
      continue
    }

    // How much of the sentence's meaning comes from the context?
    const words = contentWords(sentence)
    if (words.length === 0) {
      kept.push(sentence)   // "Yes." / "That is correct." carry no new claim
      continue
    }
    const overlap = words.filter((w) => contextWords.has(w)).length / words.length

    if (overlap >= SUPPORT_THRESHOLD) {
      kept.push(sentence)
    } else {
      dropped++
    }
  }

  return {
    text: kept.join(' ').trim(),
    dropped,
    grounded: kept.length > 0,
  }
}
