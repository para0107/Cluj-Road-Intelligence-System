/**
 * frontend/src/hooks/useIsMobile.js
 *
 * True below the given viewport width (default 768px — the app's single
 * mobile breakpoint, mirrored by the media queries in index.css).
 */

import { useEffect, useState } from 'react'

export default function useIsMobile(maxWidth = 768) {
  const query = `(max-width: ${maxWidth}px)`
  const [mobile, setMobile] = useState(() => window.matchMedia(query).matches)

  useEffect(() => {
    const mq = window.matchMedia(query)
    const onChange = (e) => setMobile(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])

  return mobile
}
