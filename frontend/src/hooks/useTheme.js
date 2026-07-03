import { useEffect, useState } from 'react'

/**
 * True while the app is in dark mode. The Navbar toggles the `light` class on
 * <html>; a MutationObserver keeps every consumer in sync without a context.
 */
export function useIsDark() {
  const [dark, setDark] = useState(() => !document.documentElement.classList.contains('light'))

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setDark(!document.documentElement.classList.contains('light'))
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  return dark
}
