import { useEffect, useState } from 'react'
import useIsMobile from './useIsMobile'

/**
 * Should this page run its heavy decorative motion (WebGL backgrounds, GSAP
 * text intros)?
 *
 * False when the visitor asked for reduced motion, or on phones — where a
 * full-screen WebGL canvas costs frames and battery for decoration nobody
 * asked for. Callers fall back to the static CSS treatment (.page-grid-bg,
 * .anim-fade-up), so the page always looks finished either way.
 */
export default function useMotionOk() {
  const isMobile = useIsMobile()
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  })

  useEffect(() => {
    if (!window.matchMedia) return undefined
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = (e) => setReduced(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  return !isMobile && !reduced
}
