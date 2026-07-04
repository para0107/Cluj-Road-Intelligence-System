/**
 * frontend/src/utils/driveMode.js
 *
 * Phone "drive mode": turns the phone into a free road-damage sensor using
 * only browser APIs — no app store, no cloud service, no cost.
 *
 * How it detects damage (the classic smartphone-accelerometer technique,
 * cf. MIT's Pothole Patrol): while the phone rides in a moving car, hitting a
 * pothole produces a sharp vertical jolt. We watch DeviceMotion linear
 * acceleration; a spike above JOLT_MPS2 while the GPS says the car is
 * actually moving (≥ MIN_SPEED_MPS, so hand-shakes at a red light don't
 * count) fires one auto-report, then a cooldown suppresses the axle's second
 * bounce and the next 8 s of vibration.
 *
 * The caller decides what to do with a hit (LivePage posts a live report
 * with damage_type "pothole" and a confidence mapped from jolt strength).
 *
 * iOS ≥ 13 requires DeviceMotionEvent.requestPermission() from a user
 * gesture — start() must be called from a click handler.
 */

const JOLT_MPS2 = 9          // linear-acceleration spike ⇒ suspected pothole
const MIN_SPEED_MPS = 2.5    // ~9 km/h — ignore jolts while stationary
const COOLDOWN_MS = 8000     // one report per rough patch, not per bounce
const FIX_MAX_AGE_MS = 10000 // GPS fix older than this ⇒ don't report

/** Map jolt strength → detector-style confidence / S1–S5 severity. */
export function joltToSignal(jolt) {
  const confidence = Math.max(0.3, Math.min(0.9, jolt / 25))
  const severity = jolt > 20 ? 4 : jolt > 14 ? 3 : 2
  return { confidence: Number(confidence.toFixed(2)), severity }
}

/**
 * Start drive mode.
 * @param {object}   handlers
 *   onHit({ latitude, longitude, jolt, confidence, severity })  auto-detected damage
 *   onTick({ jolt, speed, hasFix })                             live readout (~4 Hz)
 *   onError(message)                                            fatal setup problem
 * @returns {Promise<() => void>} stop function
 */
export async function startDriveMode({ onHit, onTick, onError }) {
  if (!('geolocation' in navigator)) {
    onError?.('Geolocation is not available in this browser.')
    return () => {}
  }

  // iOS 13+ motion permission — must run inside the click that started us
  if (typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const res = await DeviceMotionEvent.requestPermission()
      if (res !== 'granted') {
        onError?.('Motion sensor permission denied.')
        return () => {}
      }
    } catch {
      onError?.('Motion sensor permission request failed.')
      return () => {}
    }
  }

  let lastFix = null            // { latitude, longitude, speed, ts }
  let lastHitAt = 0
  let peakJolt = 0
  let stopped = false

  const geoWatch = navigator.geolocation.watchPosition(
    (pos) => {
      lastFix = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
        speed: pos.coords.speed,   // m/s or null (browser-dependent)
        ts: Date.now(),
      }
    },
    (err) => {
      // Only a hard permission denial is fatal; timeouts / temporary signal
      // loss just leave hasFix=false until the next good fix.
      if (err.code === err.PERMISSION_DENIED) {
        onError?.('GPS permission denied — drive mode needs location.')
      }
    },
    { enableHighAccuracy: true, maximumAge: 2000, timeout: 15000 },
  )

  const onMotion = (e) => {
    if (stopped) return
    const a = e.acceleration
    if (!a || a.x == null) return                       // some devices only give +gravity
    const jolt = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
    peakJolt = Math.max(peakJolt, jolt)

    if (jolt < JOLT_MPS2) return
    const now = Date.now()
    if (now - lastHitAt < COOLDOWN_MS) return
    if (!lastFix || now - lastFix.ts > FIX_MAX_AGE_MS) return
    // speed===null happens on some browsers — accept the hit but only when a
    // recent fix exists; when speed IS reported, require real movement.
    if (lastFix.speed != null && lastFix.speed < MIN_SPEED_MPS) return

    lastHitAt = now
    onHit?.({
      latitude: lastFix.latitude,
      longitude: lastFix.longitude,
      jolt: Number(jolt.toFixed(1)),
      ...joltToSignal(jolt),
    })
  }
  window.addEventListener('devicemotion', onMotion)

  // Live readout for the UI (peak jolt since last tick, decays each tick)
  const ticker = setInterval(() => {
    onTick?.({
      jolt: Number(peakJolt.toFixed(1)),
      speed: lastFix?.speed ?? null,
      hasFix: Boolean(lastFix && Date.now() - lastFix.ts < FIX_MAX_AGE_MS),
    })
    peakJolt = 0
  }, 250)

  return () => {
    stopped = true
    window.removeEventListener('devicemotion', onMotion)
    navigator.geolocation.clearWatch(geoWatch)
    clearInterval(ticker)
  }
}
