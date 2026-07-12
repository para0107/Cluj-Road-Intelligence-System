/**
 * frontend/src/assistant/intents.js
 *
 * Structured intents: questions whose answer is a NUMBER get that number from
 * the API, never from a language model. This is the strongest anti-
 * hallucination measure in the assistant, because the class of question a
 * small model is most likely to invent an answer for ("how many potholes are
 * open?") never reaches the model at all.
 *
 * Each handler returns { answer, facts, sources, route } or null if it does
 * not apply. `facts` lists every number in the answer so the grounding check
 * can treat them as supported.
 */

import { fetchStats, fetchMyImpact, fetchLeaderboard } from '../utils/api'
import { fetchLiveStats } from '../utils/live'
import { SEVERITY_LABELS, CLASS_LABELS } from '../utils/constants'

const has = (q, ...words) => words.some((w) => q.includes(w))

function numbersIn(...values) {
  return values.filter((v) => v !== null && v !== undefined).map(String)
}

// ── Live hazards right now ────────────────────────────────────────────────

async function liveIntent(q) {
  const asksLive = has(q, 'hazard', 'live', 'right now', 'currently', 'active', 'open now')
  const asksCount = has(q, 'how many', 'count', 'number of', 'how much')
  if (!(asksLive && asksCount)) return null

  const s = await fetchLiveStats()
  const answer =
    `There are ${s.active_events} active hazards on the live map right now. ` +
    `${s.verified_events} of them are verified by several drivers. ` +
    `In the last hour drivers sent ${s.reports_last_hour} reports from ${s.devices_last_hour} devices.`

  return {
    answer,
    facts: numbersIn(s.active_events, s.verified_events, s.reports_last_hour, s.devices_last_hour),
    sources: ['pipeline-live'],
    route: '/live',
  }
}

// ── Survey statistics ─────────────────────────────────────────────────────

async function statsIntent(q) {
  const asksData = has(q, 'detection', 'damage', 'pothole', 'crack', 'record', 'total', 'backlog', 'repaired', 'fixed', 'critical', 'severity')
  const asksCount = has(q, 'how many', 'count', 'number of', 'how much', 'statistics', 'stats', 'summary')
  if (!(asksData && asksCount)) return null

  const s = await fetchStats()
  const lines = [
    `The city has ${s.total_detections} damage records in total.`,
    `${s.critical_count} are critical (S4 or S5) and ${s.fixed_count} have been repaired.`,
  ]
  if (s.avg_severity != null) lines.push(`The average severity is ${s.avg_severity} out of 5.`)

  const top = [...(s.damage_type_breakdown || [])].sort((a, b) => b.count - a.count)[0]
  if (top) {
    lines.push(`The most common type is ${CLASS_LABELS[top.damage_type] || top.damage_type} with ${top.count} records.`)
  }
  if (s.last_survey_date) lines.push(`The most recent survey was on ${s.last_survey_date}.`)

  return {
    answer: lines.join(' '),
    facts: numbersIn(
      s.total_detections, s.critical_count, s.fixed_count, s.avg_severity,
      top?.count, s.last_survey_date,
    ),
    sources: ['pipeline-what', 'severity-bands'],
    route: '/stats',
  }
}

// ── My own points / rank ──────────────────────────────────────────────────

async function myImpactIntent(q) {
  const asksMine = has(q, 'my ', 'i have', 'do i', 'am i', 'mine')
  const asksScore = has(q, 'point', 'rank', 'badge', 'streak', 'score', 'impact', 'position')
  if (!(asksMine && asksScore)) return null

  const d = await fetchMyImpact()
  const s = d.stats || {}
  const lines = [
    `You have ${s.points_total || 0} points from ${s.reports_total || 0} reports.`,
    `${s.confirmed_total || 0} of your reports were confirmed and ${s.fixed_total || 0} led to a repair.`,
  ]
  if (d.rank_city) lines.push(`You are ranked number ${d.rank_city} in your city.`)
  if (s.current_streak_days) lines.push(`Your current streak is ${s.current_streak_days} days.`)
  const badgeCount = (d.badges || []).length
  lines.push(badgeCount ? `You have earned ${badgeCount} badges.` : 'You have not earned a badge yet.')

  return {
    answer: lines.join(' '),
    facts: numbersIn(
      s.points_total, s.reports_total, s.confirmed_total, s.fixed_total,
      d.rank_city, s.current_streak_days, badgeCount,
    ),
    sources: ['points-how', 'points-impact-page'],
    route: '/impact',
  }
}

// ── Leaderboard ───────────────────────────────────────────────────────────

async function leaderboardIntent(q) {
  if (!has(q, 'leaderboard', 'top reporter', 'who is winning', 'best reporter', 'highest score', 'top of')) return null

  const d = await fetchLeaderboard(null, 5)
  const items = d.items || []
  if (!items.length) {
    return {
      answer: 'Nobody has scored points yet. The first confirmed report takes the top spot.',
      facts: [],
      sources: ['points-leaderboard'],
      route: '/impact',
    }
  }
  const top = items
    .map((r) => `${r.rank}. ${r.username} with ${r.points_total} points`)
    .join(', ')

  return {
    answer: `The top reporters right now are: ${top}. You can see the full list, and your own place in it, on the My impact page.`,
    facts: items.flatMap((r) => numbersIn(r.rank, r.points_total)),
    sources: ['points-leaderboard'],
    route: '/impact',
  }
}

// ── Severity lookup ("what does S4 mean") ─────────────────────────────────

async function severityIntent(q) {
  const m = q.match(/\bs\s*([1-5])\b/)
  if (!m) return null
  const level = Number(m[1])
  const label = SEVERITY_LABELS[level]
  const action = {
    1: 'Monitor it and look again at the next survey.',
    2: 'Schedule it into routine maintenance.',
    3: 'Repair it within the current cycle.',
    4: 'Urgent: send a crew this week.',
    5: 'Emergency: close the lane and repair immediately.',
  }[level]

  return {
    answer: `S${level} is "${label.split('·')[1]?.trim() || label}". ${action} Severity runs from S1, the mildest, to S5, the most dangerous.`,
    facts: [String(level)],
    sources: ['severity-bands', 'severity-how'],
    route: null,
  }
}

const HANDLERS = [severityIntent, myImpactIntent, leaderboardIntent, liveIntent, statsIntent]

/**
 * Try every handler. Returns the first structured answer, or null when the
 * question is not a data question (then the RAG path handles it).
 * A failing API call is not fatal: it just falls through to retrieval.
 */
export async function runIntents(question) {
  const q = ` ${question.toLowerCase().trim()} `
  for (const handler of HANDLERS) {
    try {
      const result = await handler(q)
      if (result) return result
    } catch {
      // The endpoint may be unauthorised or offline. Fall through to the
      // knowledge base rather than showing the user a stack trace.
    }
  }
  return null
}
