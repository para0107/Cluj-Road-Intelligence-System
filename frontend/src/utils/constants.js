// ── Damage class colours ──────────────────────────────────────────────────
export const CLASS_COLORS = {
  longitudinal_crack:        '#6ea8ff',   // blue
  transverse_crack:          '#f472b6',   // pink
  alligator_crack:           '#ff9f43',   // orange
  repaired_crack:            '#3ddc84',   // green
  pothole:                   '#ff5d5d',   // red
  pedestrian_crossing_blur:  '#b388ff',   // purple
  lane_line_blur:            '#ffd60a',   // amber
  manhole_cover:             '#2dd4bf',   // teal
  patchy_road:               '#e879f9',   // fuchsia
  rutting:                   '#94a3b8',   // slate
}

export const CLASS_LABELS = {
  longitudinal_crack:        'Longitudinal Crack',
  transverse_crack:          'Transverse Crack',
  alligator_crack:           'Alligator Crack',
  repaired_crack:            'Repaired Crack',
  pothole:                   'Pothole',
  pedestrian_crossing_blur:  'Crossing Blur',
  lane_line_blur:            'Lane Blur',
  manhole_cover:             'Manhole Cover',
  patchy_road:               'Patchy Road',
  rutting:                   'Rutting',
}

export const CLASS_ICONS = {
  longitudinal_crack:        '〰',
  transverse_crack:          '═',
  alligator_crack:           '⬡',
  repaired_crack:            '✔',
  pothole:                   '⬤',
  pedestrian_crossing_blur:  '⛜',
  lane_line_blur:            '─',
  manhole_cover:             '◎',
  patchy_road:               '▦',
  rutting:                   '∿',
}

export const ALL_CLASSES = Object.keys(CLASS_LABELS)

// ── Severity ──────────────────────────────────────────────────────────────
export const SEVERITY_COLORS = {
  1: '#3ddc84',   // S1 — green
  2: '#ffd60a',   // S2 — amber
  3: '#ff9f43',   // S3 — orange
  4: '#ff5d5d',   // S4 — red
  5: '#c026d3',   // S5 — magenta / critical
}

export const SEVERITY_LABELS = {
  1: 'S1 · Monitor',
  2: 'S2 · Schedule',
  3: 'S3 · Priority',
  4: 'S4 · Urgent',
  5: 'S5 · Emergency',
}

export const SEVERITY_ACTIONS = {
  1: 'Monitor: log it and re-inspect at the next survey.',
  2: 'Schedule: add it to the routine maintenance plan.',
  3: 'Priority repair: schedule it within the current cycle.',
  4: 'Urgent repair: dispatch a crew this week.',
  5: 'Emergency: close the lane and repair immediately.',
}

export const SEVERITY_SHORT = { 1: 'S1', 2: 'S2', 3: 'S3', 4: 'S4', 5: 'S5' }

// ── Map defaults ──────────────────────────────────────────────────────────
// Maps open on the signed-in user's city (see hooks/useCityCenter.js).
// These are the LAST-RESORT fallbacks for the instant before the city
// resolves (or if a legacy account has no city): a country-level view, so
// nothing city-specific is ever hardcoded into the map.
export const DEFAULT_CENTER = [45.9432, 24.9668]   // country centroid fallback
export const DEFAULT_ZOOM   = 7
export const CITY_ZOOM      = 13                   // zoom used once a city resolves

// Landmarks used as an offline fallback for the demo city's fly-to menu
export const CLUJ_LANDMARKS = [
  { name: 'Piața Unirii',        lat: 46.7694, lon: 23.5899 },
  { name: 'Gara CFR',            lat: 46.7847, lon: 23.5867 },
  { name: 'Cluj Arena',          lat: 46.7686, lon: 23.5725 },
  { name: 'FSEGA',               lat: 46.7734, lon: 23.6193 },
  { name: 'Iulius Mall',         lat: 46.7735, lon: 23.6320 },
  { name: 'Aeroport Intl. Cluj', lat: 46.7852, lon: 23.6862 },
  { name: 'Mănăștur',            lat: 46.7568, lon: 23.5567 },
  { name: 'Mărăști',             lat: 46.7830, lon: 23.6180 },
]

// ── Basemaps (all key-free) ───────────────────────────────────────────────
export const BASEMAPS = {
  dark: {
    label: 'Dark',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attr: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a>',
  },
  voyager: {
    label: 'Streets',
    url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    attr: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a>',
  },
  satellite: {
    label: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr: '© <a href="https://www.esri.com/">Esri</a> — World Imagery',
  },
}

export const TILE_URL  = BASEMAPS.dark.url
export const TILE_ATTR = BASEMAPS.dark.attr

// ── Pipeline stages (must match orchestrator + backend session.json) ─────
export const PIPELINE_STAGES = [
  { key: 'preprocessor',        label: 'Preprocessor',        sub: 'Frame extraction · GPS sync · Lighting' },
  { key: 'detector',            label: 'RT-DETR Detector',    sub: 'RT-DETR-L inference · Confidence filter' },
  { key: 'segmentor',           label: 'SAM Segmentor',       sub: 'SAM 2.1 Tiny · 4 geometry features' },
  { key: 'depth_estimator',     label: 'Depth Estimator',     sub: 'Monodepth2 · Relative disparity' },
  { key: 'severity_classifier', label: 'Severity Classifier', sub: 'Rule-based S1–S5 · Weighted multi-signal' },
  { key: 'deduplicator',        label: 'Deduplicator',        sub: 'DBSCAN · Haversine clustering' },
  { key: 'db_writer',           label: 'DB Writer',           sub: 'PostGIS upsert · Priority score update' },
]

// ── Repair planning heuristics (client-side estimates, RON) ──────────────
// Rough unit costs for a repair plan sketch — presentation aid, not a quote.
export const REPAIR_COST_RON = {
  pothole:                   950,
  alligator_crack:           1400,
  longitudinal_crack:        420,
  transverse_crack:          420,
  patchy_road:               1800,
  rutting:                   2200,
  repaired_crack:            120,
  manhole_cover:             600,
  lane_line_blur:            260,
  pedestrian_crossing_blur:  340,
}
export const SEVERITY_COST_FACTOR = { 1: 0.5, 2: 0.8, 3: 1.0, 4: 1.45, 5: 2.1 }
