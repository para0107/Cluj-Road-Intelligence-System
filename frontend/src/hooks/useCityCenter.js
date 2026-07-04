/**
 * frontend/src/hooks/useCityCenter.js
 *
 * Where should a map open? On the signed-in user's own city — never on a
 * hardcoded one. Resolution order:
 *
 *   1. localStorage cache of the city's geocoded centre  → instant, no flash
 *   2. the user's last known GPS position (profile)      → instant
 *   3. DEFAULT_CENTER (country-level view)               → last resort
 *
 * In parallel, GET /cities/center resolves the city once (server-side
 * Nominatim geocode, cached forever in the DB). When it arrives *after*
 * first paint, `cityCenter` flips from null so the page can glide the map
 * over. The result is also cached locally, so this happens once per browser.
 */

import { useEffect, useMemo, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { fetchCityCenter } from '../utils/api'
import { DEFAULT_CENTER, DEFAULT_ZOOM, CITY_ZOOM } from '../utils/constants'

const CACHE_KEY = 'rids_city_centers'   // { [city.toLowerCase()]: [lat, lon] }

function readCache() {
  try { return JSON.parse(localStorage.getItem(CACHE_KEY) || '{}') } catch { return {} }
}

export default function useCityCenter() {
  const { user } = useAuth()
  const city = user?.city?.trim() || null

  // Synchronous best guess — what the map mounts with.
  const initial = useMemo(() => {
    if (city) {
      const hit = readCache()[city.toLowerCase()]
      if (hit) return { center: hit, zoom: CITY_ZOOM, source: 'city' }
    }
    if (user?.latitude != null && user?.longitude != null) {
      return { center: [user.latitude, user.longitude], zoom: CITY_ZOOM, source: 'gps' }
    }
    return { center: DEFAULT_CENTER, zoom: DEFAULT_ZOOM, source: 'default' }
  }, [city, user?.latitude, user?.longitude])

  // Late-resolved city centre (only set when it was NOT already cached).
  const [cityCenter, setCityCenter] = useState(null)

  useEffect(() => {
    if (!city || readCache()[city.toLowerCase()]) return
    let alive = true
    fetchCityCenter(city)
      .then((res) => {
        if (!alive || res?.latitude == null) return
        const cache = readCache()
        cache[city.toLowerCase()] = [res.latitude, res.longitude]
        localStorage.setItem(CACHE_KEY, JSON.stringify(cache))
        setCityCenter([res.latitude, res.longitude])
      })
      .catch(() => { /* unknown city / offline — the fallback view stands */ })
    return () => { alive = false }
  }, [city])

  return { ...initial, cityCenter, city }
}
