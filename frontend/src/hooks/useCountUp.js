/**
 * frontend/src/hooks/useCountUp.js
 *
 * Animate a number from 0 (or its previous value) to `target` with an
 * ease-out curve. Numbers that arrive from the API "roll in" instead of
 * popping — the classic command-center touch. Free: one rAF loop, no deps.
 *
 * Returns the current animated value (same integer/decimal shape as the
 * target). Non-finite targets are returned untouched.
 */

import { useEffect, useRef, useState } from 'react'

const DURATION_MS = 900
const easeOut = (t) => 1 - (1 - t) ** 3

export default function useCountUp(target, decimals = 0) {
  const [value, setValue] = useState(0)
  const fromRef = useRef(0)
  const rafRef = useRef(null)

  useEffect(() => {
    if (!Number.isFinite(target)) return undefined
    const from = fromRef.current
    const start = performance.now()

    const tick = (now) => {
      const t = Math.min(1, (now - start) / DURATION_MS)
      const v = from + (target - from) * easeOut(t)
      setValue(Number(v.toFixed(decimals)))
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = target
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [target, decimals])

  return Number.isFinite(target) ? value : target
}
