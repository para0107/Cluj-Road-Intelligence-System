/**
 * frontend/src/utils/routePlan.js
 *
 * Crew route planning for a work order — pure geometry, no external routing
 * service (the project stays at zero cost, and no third party ever sees the
 * city's repair schedule).
 *
 * Stops are ordered with nearest-neighbour, then improved with 2-opt until no
 * swap helps or the time budget runs out. Distances are straight-line
 * (haversine), so the total is a lower bound on real driving distance: good
 * enough to sequence a day's stops sensibly, which is the whole job here.
 *
 * Work orders are capped at 200 items, so the O(n²) passes stay instant.
 */

const EARTH_RADIUS_KM = 6371

const toRad = (deg) => (deg * Math.PI) / 180

/** Great-circle distance between two {latitude, longitude} points, in km. */
export function haversineKm(a, b) {
  const lat1 = toRad(a.latitude)
  const lat2 = toRad(b.latitude)
  const dLat = lat2 - lat1
  const dLon = toRad(b.longitude - a.longitude)
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(h)))
}

/** Total length of a route given as an ordered array of points, in km. */
export function totalKm(points) {
  let sum = 0
  for (let i = 1; i < points.length; i++) {
    sum += haversineKm(points[i - 1], points[i])
  }
  return sum
}

/** Per-leg distances: legKm[i] is the distance from stop i-1 to stop i (0 for i=0). */
export function legKms(points) {
  return points.map((p, i) => (i === 0 ? 0 : haversineKm(points[i - 1], p)))
}

/**
 * Greedy nearest-neighbour order.
 * @param {Array} points  items with latitude/longitude
 * @param {Object|null} start  optional depot/start position (e.g. the crew's
 *                             current location); defaults to the first item.
 */
export function nearestNeighborOrder(points, start = null) {
  if (points.length <= 2) return [...points]

  const remaining = [...points]
  const route = []
  let current = start

  if (!current) {
    current = remaining.shift()
    route.push(current)
  }

  while (remaining.length > 0) {
    let bestIdx = 0
    let bestDist = Infinity
    for (let i = 0; i < remaining.length; i++) {
      const d = haversineKm(current, remaining[i])
      if (d < bestDist) {
        bestDist = d
        bestIdx = i
      }
    }
    current = remaining.splice(bestIdx, 1)[0]
    route.push(current)
  }
  return route
}

/**
 * 2-opt improvement: repeatedly reverse the segment between two stops when
 * doing so shortens the route. Stops when a full pass finds no improvement
 * or the time budget is spent.
 */
export function twoOpt(points, budgetMs = 200) {
  if (points.length < 4) return [...points]

  const route = [...points]
  const deadline = Date.now() + budgetMs
  let improved = true

  while (improved) {
    improved = false
    for (let i = 0; i < route.length - 2; i++) {
      if (Date.now() > deadline) return route
      for (let k = i + 2; k < route.length; k++) {
        // Current edges (i, i+1) and (k, k+1) vs the swapped pair.
        const a = route[i]
        const b = route[i + 1]
        const c = route[k]
        const d = route[k + 1]

        const before = haversineKm(a, b) + (d ? haversineKm(c, d) : 0)
        const after = haversineKm(a, c) + (d ? haversineKm(b, d) : 0)

        if (after < before - 1e-9) {
          // Reverse the inner segment.
          const segment = route.slice(i + 1, k + 1).reverse()
          route.splice(i + 1, segment.length, ...segment)
          improved = true
        }
      }
    }
  }
  return route
}

/**
 * Plan a route: nearest-neighbour seed, then 2-opt polish.
 * Returns { stops, totalKm, legKm, estimatedMinutes }.
 *
 * The time estimate assumes 25 km/h of city driving plus a fixed time on site
 * per stop; it is a planning aid shown next to the route, not a promise.
 */
export function planRoute(points, { start = null, budgetMs = 200, minutesPerStop = 20, avgSpeedKmh = 25 } = {}) {
  const valid = (points || []).filter(
    (p) => Number.isFinite(p?.latitude) && Number.isFinite(p?.longitude),
  )
  if (valid.length === 0) {
    return { stops: [], totalKm: 0, legKm: [], estimatedMinutes: 0 }
  }

  const seeded = nearestNeighborOrder(valid, start)
  const stops = twoOpt(seeded, budgetMs)
  const km = totalKm(stops)
  const driveMinutes = avgSpeedKmh > 0 ? (km / avgSpeedKmh) * 60 : 0

  return {
    stops,
    totalKm: km,
    legKm: legKms(stops),
    estimatedMinutes: Math.round(driveMinutes + stops.length * minutesPerStop),
  }
}
