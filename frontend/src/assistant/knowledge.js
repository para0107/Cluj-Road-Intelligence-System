/**
 * frontend/src/assistant/knowledge.js
 *
 * The assistant's knowledge base. Every answer the assistant gives in AI mode
 * must be supported by these entries (see guard.js), so this file is the
 * single source of truth for what it is allowed to say.
 *
 * Style rules for entries: plain sentences, no em-dashes, no marketing.
 * If something is not true of RDDS, do not write it here.
 */

export const KNOWLEDGE = [
  // ── Reporting ───────────────────────────────────────────────────────────
  {
    id: 'report-how',
    title: 'How to report a pothole or road damage',
    tags: ['report', 'pothole', 'hazard', 'live', 'how'],
    route: '/live',
    text: `Open the Live page and tap the report button. It uses your phone GPS, so you do not type an address. Pick the type of damage and send. The report appears on the map for other drivers straight away. You need to be signed in to report.`,
  },
  {
    id: 'report-confirm',
    title: 'Confirming and disputing reports',
    tags: ['confirm', 'dispute', 'verify', 'trust'],
    route: '/live',
    text: `When a hazard is already on the map you can confirm it if it is really there, or dispute it if it is gone. RDDS counts confirmations from different devices, not from the same person twice. Two independent devices make a report confirmed, three make it verified. Enough disputes remove it from the map.`,
  },
  {
    id: 'report-drive-mode',
    title: 'Drive mode reports bumps automatically',
    tags: ['drive', 'mode', 'automatic', 'sensor', 'motion', 'phone'],
    route: '/live',
    text: `Drive mode turns your phone into a sensor. Mount the phone in the car and start drive mode from the Live page. When the phone feels a jolt while the GPS says you are moving, it sends one pothole report by itself and then waits a few seconds before it can send another. On iPhone you must start it from a tap because Apple requires the motion permission to come from a real button press.`,
  },
  {
    id: 'report-limits',
    title: 'Limits on how often you can report',
    tags: ['limit', 'cooldown', 'spam', 'rate', 'too many'],
    text: `You can send one report every 15 seconds and up to 40 reports a day. This stops one account from flooding the map. If you hit the limit you get a message asking you to wait, and nothing you already sent is lost.`,
  },

  // ── Points and badges ───────────────────────────────────────────────────
  {
    id: 'points-how',
    title: 'How points work',
    tags: ['points', 'score', 'reward', 'gamification'],
    route: '/impact',
    text: `Sending a report earns no points on its own. Points come only when your report is validated by others or acted on by the city. You get 10 points when other drivers confirm it, 15 more when it reaches verified, 20 when the city accepts it as an official record, and 25 when the damage is repaired. This means accuracy is worth more than volume, and it is why spamming reports does not help anyone.`,
  },
  {
    id: 'points-badges',
    title: 'Badges you can earn',
    tags: ['badge', 'badges', 'achievement', 'award'],
    route: '/impact',
    text: `There are eight badges. First report is for your first report. Road scout is ten confirmed reports and Road guardian is fifty. Triple checked is for a report reaching verified. Week streak is reporting on seven days in a row. Fixer is for one repaired hazard you reported and City changer is for five. Night watch is for reporting late at night.`,
  },
  {
    id: 'points-streak',
    title: 'Streaks',
    tags: ['streak', 'daily', 'consecutive'],
    route: '/impact',
    text: `A streak counts the days in a row on which you sent at least one report. Missing a day resets the current streak, but your best streak is kept. You can see both on the My impact page.`,
  },
  {
    id: 'points-leaderboard',
    title: 'The leaderboard',
    tags: ['leaderboard', 'rank', 'ranking', 'top', 'compete'],
    route: '/impact',
    text: `The leaderboard on the My impact page ranks reporters by points. You can see your city or everyone. It shows usernames only, never real names or e-mail addresses. Your own row is highlighted so you can find yourself quickly.`,
  },
  {
    id: 'points-impact-page',
    title: 'The My impact page',
    tags: ['impact', 'my', 'profile', 'stats'],
    route: '/impact',
    text: `My impact is your personal page. It shows your points, how many reports you sent, how many were confirmed, verified and repaired, your streak, your badges, your rank, and the list of hazards you reported with their current status.`,
  },

  // ── Severity and damage classes ─────────────────────────────────────────
  {
    id: 'severity-bands',
    title: 'What the severity levels S1 to S5 mean',
    tags: ['severity', 's1', 's2', 's3', 's4', 's5', 'level', 'urgent'],
    text: `Every damage record gets a severity from S1 to S5. S1 means monitor it and look again at the next survey. S2 means schedule it into routine maintenance. S3 is a priority repair within the current cycle. S4 is urgent and needs a crew this week. S5 is an emergency: close the lane and repair immediately.`,
  },
  {
    id: 'severity-how',
    title: 'How severity is calculated',
    tags: ['severity', 'calculate', 'score', 'formula', 'how'],
    text: `Severity is rule based, not a guess by a second model. RDDS combines four measured signals: how deep the damage looks, how large its area is, how much its inside contrasts with the road around it, and how sharp its edges are. Each signal is weighted, the damage type applies its own weight, and the result is mapped to the S1 to S5 bands. Painted marking faults are deliberately capped low because they are not a structural danger.`,
  },
  {
    id: 'severity-priority',
    title: 'How repair priority is ranked',
    tags: ['priority', 'rank', 'queue', 'order', 'repair'],
    route: '/priority',
    text: `Priority combines severity with how many times the damage has been seen. Damage that is both severe and repeatedly detected rises to the top of the repair queue. The Repairs page lists sites in this order.`,
  },
  {
    id: 'classes',
    title: 'The types of road damage RDDS detects',
    tags: ['class', 'classes', 'types', 'damage', 'detect'],
    text: `RDDS detects ten types: potholes, longitudinal cracks, transverse cracks, alligator cracks, repaired cracks, patchy road, rutting, manhole covers, faded lane lines and faded pedestrian crossings.`,
  },

  // ── The pipeline ────────────────────────────────────────────────────────
  {
    id: 'pipeline-what',
    title: 'How RDDS finds damage in dashcam video',
    tags: ['pipeline', 'video', 'detect', 'ai', 'model', 'how it works'],
    route: '/about',
    text: `A city operator uploads dashcam video and an optional GPS track. RDDS pulls frames out of the video, finds damage in each frame with an object detector, traces the exact shape of each damage with a segmentation model, estimates how deep it is from a single camera, scores severity with the rule based formula, merges repeated sightings of the same physical damage into one record, and writes the result to the map database. You can watch each stage run on the Upload page.`,
  },
  {
    id: 'pipeline-gps',
    title: 'Video without GPS',
    tags: ['gps', 'no gps', 'location', 'gpx'],
    text: `Damage can be detected without GPS, but it cannot be placed on the map. If a video has no usable GPS track the run still finishes and shows you what it found, and nothing is written to the map. This is expected, not an error.`,
  },
  {
    id: 'pipeline-live',
    title: 'The difference between Live mode and a survey',
    tags: ['live', 'survey', 'difference', 'mode'],
    text: `Live mode is the crowd map: drivers and phones report hazards in real time, other drivers confirm them, and hazards expire on their own if nobody sees them again. A survey is the heavy pass: a city uploads video and the full detection pipeline produces measured, ranked damage records. Live reports can be promoted by an operator into official survey records.`,
  },

  // ── Municipality workflow ───────────────────────────────────────────────
  {
    id: 'muni-triage',
    title: 'The triage inbox',
    tags: ['triage', 'operator', 'promote', 'dismiss', 'municipality'],
    route: '/triage',
    text: `Triage is where a city worker reviews what citizens reported. The most confirmed reports are listed first. The operator can make a report official, which turns it into a damage record on the survey map and in the repair queue, or dismiss it if it is a duplicate or not real. Everyone who reported it is credited and told what happened.`,
  },
  {
    id: 'muni-workorders',
    title: 'Work orders',
    tags: ['work order', 'workorder', 'crew', 'job', 'repair', 'schedule'],
    route: '/workorders',
    text: `A work order groups several damage sites into one job for one crew. It carries a crew name, a scheduled date, a due date and a cost estimate, and it moves through open, scheduled, in progress, repaired and verified. Marking it repaired marks all of its sites as fixed.`,
  },
  {
    id: 'muni-route',
    title: 'Crew route planning',
    tags: ['route', 'plan', 'crew', 'driving', 'order', 'optimize'],
    route: '/workorders',
    text: `A work order can order its sites into a sensible driving route. RDDS uses a nearest neighbour pass and then a 2-opt improvement over straight line distances, so it needs no paid routing service. You get the stop order, the total distance and a printable route sheet for the crew.`,
  },
  {
    id: 'muni-verification',
    title: 'Repair verification and reopened damage',
    tags: ['verify', 'verification', 'reopened', 'failed repair', 'again'],
    route: '/workorders',
    text: `When a site is marked repaired RDDS records the date. If a later survey detects damage at that same site again, the site is flagged as seen again. A work order cannot be closed as verified while any of its sites are in that state, so a repair that did not hold cannot be quietly signed off.`,
  },
  {
    id: 'muni-quality',
    title: 'The Road Quality Index',
    tags: ['quality', 'rqi', 'index', 'score', 'grid', 'band'],
    route: '/quality',
    text: `The Road Quality Index splits the city into squares of about 120 m and scores each one from 0 to 100, where higher is better. The score falls with more damage, worse severity and more recent sightings, and recovers when damage is repaired. Squares are banded A to E. You can export the grid as CSV or GeoJSON.`,
  },
  {
    id: 'muni-analytics',
    title: 'Operations analytics',
    tags: ['analytics', 'stats', 'time to repair', 'backlog', 'budget'],
    route: '/stats',
    text: `The Stats page shows the city how it is doing: the average time from first sighting to repair, the open backlog by severity, how many repairs came back, a weekly trend of new damage against repairs, and what has been spent against what was estimated.`,
  },
  {
    id: 'muni-budget',
    title: 'The budget planner',
    tags: ['budget', 'cost', 'money', 'estimate', 'plan', 'ron'],
    route: '/priority',
    text: `The budget planner estimates what a repair programme would cost. You choose what share of each severity band you intend to repair and it multiplies the counts by average unit costs and a severity factor. These are planning estimates from average costs, not a quote.`,
  },
  {
    id: 'muni-upload',
    title: 'Uploading a survey video',
    tags: ['upload', 'video', 'survey', 'mp4', 'gpx', 'ingest'],
    route: '/ingest',
    text: `Operators upload an MP4 video and an optional GPX track on the Upload page. One survey runs at a time and you can watch each of the seven stages finish. The heavy detection work runs on the machine with the graphics card, not in the browser.`,
  },

  // ── Accounts and roles ──────────────────────────────────────────────────
  {
    id: 'account-roles',
    title: 'Account types',
    tags: ['role', 'account', 'citizen', 'municipality', 'admin', 'operator'],
    text: `There are three kinds of account. A citizen can use the live map, report hazards and see their impact. A municipality account is a city worker and also gets the survey map, triage, work orders, the quality index, analytics and uploads. An admin manages accounts and approves municipality registrations.`,
  },
  {
    id: 'account-register',
    title: 'Creating an account',
    tags: ['register', 'sign up', 'create account', 'join'],
    route: '/register',
    text: `Registration asks for a username, e-mail, password and your city. The city matters because the maps open on it. You confirm your e-mail with a six digit code. A municipality registration also waits for an admin to approve it before it becomes active.`,
  },
  {
    id: 'account-city',
    title: 'Why RDDS asks for your city',
    tags: ['city', 'location', 'why'],
    text: `Every account has a city. It is what the maps open on and what scopes a municipality account to the roads it is responsible for. You can change it from the user menu.`,
  },
  {
    id: 'account-devices',
    title: 'Paired devices',
    tags: ['device', 'pair', 'dashcam', 'edge', 'revoke'],
    route: '/live',
    text: `Reports come from devices linked to your account. A phone or browser links itself when you first report. A dashcam computer is linked with a short pairing code instead, so you never type your password on the vehicle machine. You can disconnect a device at any time and its reports stop being accepted.`,
  },

  // ── Privacy and data ────────────────────────────────────────────────────
  {
    id: 'privacy-location',
    title: 'What location data RDDS stores',
    tags: ['privacy', 'data', 'location', 'street', 'address', 'gdpr'],
    text: `RDDS stores the latitude and longitude of damage and nothing else about where it is. It does not store street names or addresses, and it does not look them up. Damage records are about the road, not about people.`,
  },
  {
    id: 'privacy-assistant',
    title: 'How this assistant handles your questions',
    tags: ['assistant', 'chatbot', 'privacy', 'ai', 'llm'],
    route: '/assistant',
    text: `This assistant runs entirely on your own device. In its normal mode it searches a built in guide and the live numbers from your city. If you turn on the AI answer mode, the language model is downloaded once and then runs inside your browser. Your questions are never sent to an outside company, and RDDS pays nothing to answer them.`,
  },

  // ── Developer API and pricing ───────────────────────────────────────────
  {
    id: 'api-keys',
    title: 'The developer API',
    tags: ['api', 'key', 'developer', 'integrate', 'json'],
    route: '/developers',
    text: `RDDS has a read only API for damage records, the road quality grid and summary statistics. You create a key on the Developers page and send it in the X-API-Key header. The default limit is 60 requests a minute per key, and keys never expose personal data or photos.`,
  },
  {
    id: 'pricing',
    title: 'What RDDS costs',
    tags: ['price', 'pricing', 'cost', 'free', 'buy', 'plan'],
    route: '/pricing',
    text: `The citizen app is free forever: reporting, the live map, drive mode, points and this assistant. The tools a city uses are offered as a free pilot, and a city wide rollout is agreed by e-mail. The developer API is free while it is in preview. No payment is taken on this site.`,
  },

  // ── Navigation ──────────────────────────────────────────────────────────
  {
    id: 'nav-pages',
    title: 'What each page does',
    tags: ['page', 'navigate', 'where', 'find', 'menu'],
    text: `Command is the overview. Live is the real time hazard map where you report. My impact is your points and badges. System explains how RDDS works. City workers also get Map for the survey data, Explorer for the full table, Stats for analytics, Repairs for the ranked queue and budget, Triage for citizen reports, Work orders for repair jobs, Quality for the road quality index, and Upload for new survey video.`,
  },
]

/** Flat lookup by id, used by the grounding check to cite sources. */
export const KNOWLEDGE_BY_ID = Object.fromEntries(KNOWLEDGE.map((k) => [k.id, k]))

/** Suggested opening questions shown as chips on the Assistant page. */
export const SUGGESTIONS = [
  'How do I report a pothole?',
  'How do points work?',
  'What does S4 mean?',
  'How many hazards are open right now?',
  'What is the Road Quality Index?',
  'How do I get an API key?',
]
