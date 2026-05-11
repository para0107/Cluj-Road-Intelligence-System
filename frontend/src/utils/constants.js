// ── Damage class colours ──────────────────────────────────────────────────
export const CLASS_COLORS = {
  longitudinal_crack:        '#60a5fa',   // blue
  transverse_crack:          '#f472b6',   // pink
  alligator_crack:           '#fb923c',   // orange
  repaired_crack:            '#4ade80',   // green
  pothole:                   '#f87171',   // red
  pedestrian_crossing_blur:  '#a78bfa',   // purple
  lane_line_blur:            '#fbbf24',   // amber
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

// ── Severity ──────────────────────────────────────────────────────────────
export const SEVERITY_COLORS = {
  1: '#4ade80',   // S1 — green
  2: '#fbbf24',   // S2 — amber
  3: '#fb923c',   // S3 — orange
  4: '#f87171',   // S4 — red
  5: '#a21caf',   // S5 — purple/critical
}

export const SEVERITY_LABELS = {
  1: 'S1 · Monitor',
  2: 'S2 · Schedule',
  3: 'S3 · Priority',
  4: 'S4 · Urgent',
  5: 'S5 · Emergency',
}

export const SEVERITY_SHORT = {
  1: 'S1', 2: 'S2', 3: 'S3', 4: 'S4', 5: 'S5',
}

// ── Map defaults ──────────────────────────────────────────────────────────
// Cluj-Napoca city centre
export const CLUJ_CENTER = [46.7712, 23.6236]
export const CLUJ_ZOOM   = 13

// Map tile — dark Carto tile, no API key required
export const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
export const TILE_ATTR = '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> © <a href="https://carto.com/">CARTO</a>'