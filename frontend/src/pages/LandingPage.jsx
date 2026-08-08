/**
 * frontend/src/pages/LandingPage.jsx — the public front door.
 *
 * Shown at "/" to logged-out visitors (signed-in users get the command
 * dashboard instead). It is a self-contained marketing surface: the whole
 * product packaged around one promise — a safer drive for the whole city —
 * with road-damage detection as the mechanism underneath.
 *
 * Design: "the living city." A single dusk photograph of a city opens the page
 * and feathers into a warm paper body. One oxide-brick ink. Bricolage Grotesque
 * over Hanken Grotesk. Styling is scoped under .rdds-landing so it reads the
 * same in either app theme.
 *
 * Live data comes from GET /api/public/landing (stats, recent activity, a road
 * quality grade) and degrades to descriptive copy when the city has no data
 * yet. "Request a pilot" posts to the defended /contact/sales capture.
 */

import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchLanding, requestPilot } from '../utils/api'
import { fmtNum, fmtDate } from '../utils/format'
import useMotionOk from '../hooks/useMotionOk'

const HERO = '/img/hero-a.jpg'

const DAMAGE_LABEL = {
  pothole: 'Pothole', alligator_crack: 'Alligator cracking',
  longitudinal_crack: 'Longitudinal crack', transverse_crack: 'Transverse crack',
  block_crack: 'Block cracking', edge_crack: 'Edge break', rutting: 'Rutting',
  ravelling: 'Ravelling', patch: 'Failed patch',
  lane_line_blur: 'Faded lane line', pedestrian_crossing_blur: 'Faded crossing',
}
const label = (t) => DAMAGE_LABEL[t] || (t ? t.replace(/_/g, ' ') : 'Damage')

export default function LandingPage() {
  const motionOk = useMotionOk()
  const [data, setData] = useState(null)
  const heroImg = useRef(null)

  // pilot form
  const [form, setForm] = useState({ name: '', email: '', city: '', message: '' })
  const [website, setWebsite] = useState('') // honeypot
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let alive = true
    fetchLanding().then(d => { if (alive) setData(d) }).catch(() => {})
    return () => { alive = false }
  }, [])

  // scroll reveals + parallax drift of the hero (motion only, rAF-throttled)
  useEffect(() => {
    const root = document.querySelector('.rdds-landing')
    if (!root) return
    const items = Array.from(root.querySelectorAll('.l-rv'))
    if (!motionOk || !('IntersectionObserver' in window)) {
      items.forEach(el => el.classList.add('in'))
      return
    }
    const io = new IntersectionObserver((es) => {
      es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) } })
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 })
    items.forEach(el => io.observe(el))
    const safety = setTimeout(() => items.forEach(el => el.classList.add('in')), 1800)

    let ticking = false
    const drift = () => {
      const y = window.scrollY
      if (heroImg.current && y < window.innerHeight * 1.1) {
        heroImg.current.style.transform = `translate3d(0, ${(y * 0.26).toFixed(1)}px, 0)`
      }
      ticking = false
    }
    const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(drift) } }
    window.addEventListener('scroll', onScroll, { passive: true })
    drift()
    return () => { io.disconnect(); clearTimeout(safety); window.removeEventListener('scroll', onScroll) }
  }, [motionOk])

  const submitPilot = async (e) => {
    e.preventDefault()
    setErr(null)
    if (!form.name.trim() || !form.email.trim() || !form.message.trim()) {
      setErr('Please fill in your name, email and a short note.')
      return
    }
    setSending(true)
    try {
      await requestPilot({ ...form, website })
      setSent(true)
    } catch (e2) {
      setErr(e2?.response?.data?.detail || 'Could not send just now. Please try again.')
    } finally {
      setSending(false)
    }
  }

  const st = data?.stats
  const hasData = st && st.total_detections > 0
  const q = data?.quality
  const recent = (data?.recent || []).filter(r => r && r.damage_type)

  const arrow = (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M4 12h15M13 6l6 6-6 6" /></svg>
  )

  return (
    <div className={`rdds-landing${motionOk ? ' anim' : ''}`}>
      <style>{CSS}</style>

      {/* ── immersive hero ─────────────────────────────────────────────── */}
      <header className="l-hero" id="top">
        <div className="l-hero-media">
          <img ref={heroImg} src={HERO} alt="A city seen from above at dusk, its avenues threaded with light"
               width="1376" height="768" fetchpriority="high" decoding="async" />
          <span className="l-scrim" aria-hidden="true" />
          <span className="l-pin" style={{ left: '37%', top: '52%' }} aria-hidden="true" />
          <span className="l-pin l-pin-amber" style={{ left: '63%', top: '63%' }} aria-hidden="true" />
        </div>
        <div className="l-hero-inner"><div className="l-wrap">
          <div className="l-tag l-rv"><span className="l-dot" />Every car is a sensor</div>
          <h1 className="l-h1 l-rv l-d1">A safer drive for <span className="l-em">the whole city.</span></h1>
          <p className="l-lead l-rv l-d2">RDDS turns ordinary driving into a live map of road damage, so a
            city can fix the worst hazards first and everyone gets home safer. No new hardware, no cost per report.</p>
          <div className="l-cta l-rv l-d3">
            <a className="l-btn" href="#pilot">Request a pilot {arrow}</a>
            <a className="l-btn l-ghost" href="#loop">See how it works</a>
            <span className="l-note">No account needed to look.</span>
          </div>
        </div></div>
        <div className="l-hero-foot"><div className="l-wrap">
          <span className="l-cue">Scroll <i /></span>
          <span className="l-loc">A road-safety platform for cities</span>
        </div></div>
      </header>

      {/* ── fact ribbon (live when the city has data) ──────────────────── */}
      <section className="l-posthero" aria-label="At a glance"><div className="l-wrap">
        <div className="l-ribbon">
          {hasData ? (
            <>
              <div className="l-stat l-rv"><div className="l-k">{fmtNum(st.total_detections)}</div><div className="l-l">Road faults mapped and on record</div></div>
              <div className="l-stat l-rv l-d1"><div className="l-k">{fmtNum(st.critical_count)}</div><div className="l-l">Urgent hazards flagged (S4–S5)</div></div>
              <div className="l-stat l-rv l-d2"><div className="l-k">{fmtNum(st.fixed_count)}</div><div className="l-l">Repairs verified as fixed</div></div>
              <div className="l-stat l-rv l-d3"><div className="l-k">{q ? q.grade : '—'}</div><div className="l-l">Road quality grade, updated live</div></div>
            </>
          ) : (
            <>
              <div className="l-stat l-rv"><div className="l-k">10<span className="l-u">classes</span></div><div className="l-l">Damage types on the N-RDD2024 schema</div></div>
              <div className="l-stat l-rv l-d1"><div className="l-k">S1–S5</div><div className="l-l">Transparent severity, a readable rule</div></div>
              <div className="l-stat l-rv l-d2"><div className="l-k">0.00<span className="l-u">/report</span></div><div className="l-l">No paid AI, no per-report cost</div></div>
              <div className="l-stat l-rv l-d3"><div className="l-k">7<span className="l-u">stages</span></div><div className="l-l">From dashcam frame to a verified repair</div></div>
            </>
          )}
        </div>
      </div></section>

      {/* ── safety statement ───────────────────────────────────────────── */}
      <section className="l-section" id="why-safety"><div className="l-wrap">
        <div className="l-statement l-rv">
          <div className="l-label">Why it matters</div>
          <h2>A pothole is not a nuisance. It is a <span className="l-em">hazard.</span></h2>
          <p>Broken road surfaces cause blown tyres, sudden swerves, and the worst falls a cyclist can take.
            The damage a city cannot see is the damage that hurts someone. RDDS makes every ordinary drive
            add to a picture of where the road is failing, so the fix arrives before the harm does.</p>
        </div>
      </div></section>

      {/* ── safety, for everyone ───────────────────────────────────────── */}
      <section className="l-section" id="safety"><div className="l-wrap">
        <div className="l-lede l-rv">
          <h2>Safety first. A safer city <span className="l-em">for everyone in it.</span></h2>
          <p className="l-say">Every fault we find and fix is one less way for someone to get hurt on the way home.</p>
        </div>
        <div className="l-safe3">
          <div className="l-safe l-rv"><div className="l-safe-k">For drivers</div><p>No blown tyre, no sudden swerve into the next lane. The worst potholes are found and ranked before they find you.</p></div>
          <div className="l-safe l-rv l-d1"><div className="l-safe-k">For cyclists and pedestrians</div><p>The edge breaks and sunken drains that put a cyclist on the ground are exactly the faults RDDS is built to catch first.</p></div>
          <div className="l-safe l-rv l-d2"><div className="l-safe-k">For the whole city</div><p>Crews spend the budget on the roads that actually endanger people, and the repair is proven to hold. Safety you can audit.</p></div>
        </div>
      </div></section>

      {/* ── the loop ───────────────────────────────────────────────────── */}
      <section className="l-section" id="loop"><div className="l-wrap">
        <div className="l-lede l-rv">
          <h2>From a drive to a repair that holds.</h2>
          <p className="l-say">Five moves, run in order. Each one leaves a record the next one reads, so a
            street is only called safe once the road agrees.</p>
        </div>
        <div className="l-loop">
          <div className="l-step l-rv"><span className="l-no">1</span><h3>Detect</h3><p>Ten kinds of damage read straight from dashcam video, frame by frame: cracks, potholes, rutting, worn markings.</p><span className="l-to">RT-DETR · SAM 2.1</span></div>
          <div className="l-step l-rv l-d1"><span className="l-no">2</span><h3>Score</h3><p>A transparent S1 to S5 severity from measured signals, combined by a published rule. Not a black box that guesses.</p><span className="l-to">depth · area · contrast</span></div>
          <div className="l-step l-rv l-d2"><span className="l-no">3</span><h3>Prioritise</h3><p>A ranked queue your budget can work through. The most dangerous roads first, with how often a fault recurs weighted in.</p><span className="l-to">severity · ln(count+1)</span></div>
          <div className="l-step l-rv l-d3"><span className="l-no">4</span><h3>Repair</h3><p>Group faults into work orders, plan the crew's route on the map, and send them out with the evidence in hand.</p><span className="l-to">work orders · route</span></div>
          <div className="l-step l-rv l-d4"><span className="l-no">5</span><h3>Verify</h3><p>A repair is not done until the road stops showing the damage. If the fault comes back, so does the work order.</p><span className="l-to">reopened-damage guard</span></div>
        </div>
      </div></section>

      {/* ── severity ───────────────────────────────────────────────────── */}
      <section className="l-section" id="severity"><div className="l-wrap">
        <div className="l-lede l-rv">
          <h2>Severity you can argue with.</h2>
          <p className="l-say">One honest scale from monitor to emergency. Faded lane markings are capped low on
            purpose, so a cosmetic fault never outranks a real danger.</p>
        </div>
        <div className="l-sev">
          {[['1', 'Monitor', 'Log it and re-inspect at the next survey.'],
            ['2', 'Schedule', 'Add it to the routine maintenance plan.'],
            ['3', 'Priority', 'Fit it into the current repair cycle.'],
            ['4', 'Urgent', 'Send a crew within the week.'],
            ['5', 'Emergency', 'Close the lane and repair now.']].map(([s, n, a], i) => (
            <div key={s} className={`l-band l-rv l-d${i}`} data-s={s}>
              <div className="l-bar" /><div className="l-c">S{s}</div><div className="l-n">{n}</div><div className="l-a">{a}</div>
            </div>
          ))}
        </div>
      </div></section>

      {/* ── recent activity ticker (live, hidden when empty) ───────────── */}
      {recent.length > 0 && (
        <section className="l-section" id="live-feed"><div className="l-wrap">
          <div className="l-lede l-rv">
            <h2>Live from the field.</h2>
            <p className="l-say">The most recent road faults on record. Every one is a place a driver no longer
              has to find the hard way.</p>
          </div>
          <div className="l-ticker l-rv">
            {recent.slice(0, 8).map((r, i) => (
              <div className="l-tick" key={i}>
                <span className="l-tick-sev" data-s={r.severity || 1}>S{r.severity || 1}</span>
                <span className="l-tick-name">{label(r.damage_type)}</span>
                {r.detection_count > 1 && <span className="l-tick-seen">seen {r.detection_count}×</span>}
                <span className="l-tick-when">{r.last_detected ? fmtDate(r.last_detected) : ''}</span>
              </div>
            ))}
          </div>
        </div></section>
      )}

      {/* ── why it scales + road quality badge ─────────────────────────── */}
      <section className="l-section" id="why"><div className="l-wrap">
        <div className="l-lede l-rv">
          <h2>Built for a whole city, at no cost per report.</h2>
          <p className="l-say">The economics are structural, not a promotion that expires. Nothing here needs a
            vendor contract to run.</p>
        </div>
        <div className="l-scale">
          <div className="l-why">
            <div className="l-reason l-rv"><h3><span className="l-d">01</span>Compute rides with the fleet</h3><p>Every vehicle runs its own detection on its own hardware. The server only stores and ranks, so a bigger survey costs one spatial query, not a bigger bill.</p></div>
            <div className="l-reason l-rv l-d1"><h3><span className="l-d">02</span>No paid AI, anywhere</h3><p>Open models, free map tiles, a self-hosted check for abuse. There is no metered API hiding behind the product.</p></div>
            <div className="l-reason l-rv l-d2"><h3><span className="l-d">03</span>Location is all we keep</h3><p>A detection is a coordinate and a class. No street profiles, no driver identities, nothing resold to anyone.</p></div>
            <div className="l-reason l-rv l-d3"><h3><span className="l-d">04</span>The record stays auditable</h3><p>Every severity score follows a rule you can inspect, and a repair only closes when the road agrees it is fixed.</p></div>
          </div>
          {q && (
            <aside className="l-badge l-rv l-d2" aria-label={`Road quality grade ${q.grade}`}>
              <div className="l-badge-top">Road Quality Index</div>
              <div className="l-badge-grade" data-g={q.grade}>{q.grade}</div>
              <div className="l-badge-score">{q.score}<span>/100</span></div>
              <div className="l-badge-note">A single, honest grade for the network, from live severity and repair state.</div>
            </aside>
          )}
        </div>
      </div></section>

      {/* ── pilot CTA + form ───────────────────────────────────────────── */}
      <section className="l-close" id="pilot" aria-labelledby="pilot-h"><div className="l-wrap">
        <div className="l-close-copy l-rv">
          <h2 id="pilot-h">Bring RDDS to <span className="l-em">your city.</span></h2>
          <p>Start with one vehicle and a week of footage. We hand back a map of your own roads, scored and
            ranked, before you commit to anything.</p>
          <ul className="l-close-list">
            <li>See your worst hazards ranked in days, not months.</li>
            <li>Keep every model and map tile free and open.</li>
            <li>No lock-in: the data and the rules are yours to inspect.</li>
          </ul>
        </div>
        <div className="l-form-wrap l-rv l-d1">
          {sent ? (
            <div className="l-sent">
              <div className="l-sent-mark"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5" /></svg></div>
              <h3>Request received.</h3>
              <p>Thank you. We will be in touch about a pilot for {form.city || 'your city'} shortly.</p>
            </div>
          ) : (
            <form className="l-form" onSubmit={submitPilot} noValidate>
              <h3>Request a pilot</h3>
              {err && <div className="l-err" role="alert">{err}</div>}
              <label>Your name
                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} autoComplete="name" required /></label>
              <label>Work email
                <input type="email" value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} autoComplete="email" required /></label>
              <label>City or road authority
                <input value={form.city} onChange={e => setForm({ ...form, city: e.target.value })} autoComplete="organization" /></label>
              <label>What would you want to see first?
                <textarea rows={3} value={form.message} onChange={e => setForm({ ...form, message: e.target.value })} required /></label>
              {/* honeypot: real people never see or fill this */}
              <input className="l-hp" tabIndex={-1} autoComplete="off" value={website} onChange={e => setWebsite(e.target.value)} aria-hidden="true" />
              <button className="l-btn l-btn-full" disabled={sending}>{sending ? 'Sending…' : <>Request a pilot {arrow}</>}</button>
              <p className="l-form-fine">No obligation. We reply from a person, not a bot.</p>
            </form>
          )}
        </div>
      </div></section>

      <footer className="l-footer"><div className="l-wrap">
        <span><span className="l-br">RDDS</span> · Road Degradation Detection System · 2026</span>
        <span className="l-links">
          <Link to="/pricing">Pricing</Link>
          <Link to="/developers">Developers</Link>
          <Link to="/login">Sign in</Link>
        </span>
        <span>A safer drive for the whole city.</span>
      </div></footer>
    </div>
  )
}

/* ── scoped styles: the whole landing lives under .rdds-landing ─────────── */
const CSS = `
.rdds-landing{
  --paper:#f5f2ec; --paper-2:#efece4; --card:#fbfaf6;
  --ink:#191815; --ink-2:#57544c; --ink-3:#84806f;
  --brick:#a8391f; --brick-deep:#822b14; --brick-lite:#e8825f;
  --line:#1918151a; --line-2:#1918150d;
  --s3:22px; --s4:36px; --s5:clamp(44px,5vw,76px); --s6:clamp(64px,9vw,132px);
  --edge:clamp(20px,5vw,72px); --ease:cubic-bezier(.2,.7,.3,1);
  background:var(--paper); color:var(--ink);
  font-family:"Hanken Grotesk",system-ui,sans-serif; font-size:15px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}
.rdds-landing *{box-sizing:border-box}
.rdds-landing img{max-width:100%;display:block}
.rdds-landing h1,.rdds-landing h2,.rdds-landing h3{font-family:"Bricolage Grotesque","Chivo",sans-serif;margin:0;font-weight:700;letter-spacing:-.02em;line-height:1.0}
.rdds-landing p{margin:0}
.rdds-landing a{color:inherit;text-decoration:none}
.rdds-landing .l-wrap{max-width:1280px;margin:0 auto;padding:0 var(--edge)}
.rdds-landing .l-em{color:var(--brick);font-style:italic;font-weight:600}
.rdds-landing .l-label{font-weight:700;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--brick-deep);margin-bottom:14px}
.rdds-landing .l-br{color:var(--brick-deep)}
.rdds-landing :focus-visible{outline:2px solid var(--brick);outline-offset:3px}

.rdds-landing .l-rv{opacity:1}
.rdds-landing.anim .l-rv{opacity:0;transform:translateY(16px);transition:opacity .8s var(--ease),transform .8s var(--ease)}
.rdds-landing.anim .l-rv.in{opacity:1;transform:none}
.rdds-landing.anim .l-d1{transition-delay:.07s}.rdds-landing.anim .l-d2{transition-delay:.14s}
.rdds-landing.anim .l-d3{transition-delay:.21s}.rdds-landing.anim .l-d4{transition-delay:.28s}

.rdds-landing .l-btn{font-family:"Hanken Grotesk";font-weight:600;font-size:14px;display:inline-flex;align-items:center;gap:9px;padding:12px 20px;border-radius:10px;border:1px solid var(--brick);background:var(--brick);color:#fff;cursor:pointer;transition:background .16s var(--ease),transform .12s var(--ease),box-shadow .16s var(--ease)}
.rdds-landing .l-btn:hover{background:var(--brick-deep);border-color:var(--brick-deep);box-shadow:0 6px 20px -8px #a8391f80}
.rdds-landing .l-btn:active{transform:translateY(1px)}
.rdds-landing .l-btn:disabled{opacity:.6;cursor:default}
.rdds-landing .l-btn svg{width:16px;height:16px}
.rdds-landing .l-btn svg path{transition:transform .18s var(--ease)}
.rdds-landing .l-btn:hover svg path{transform:translateX(3px)}
.rdds-landing .l-ghost{background:#ffffff12;color:#fff;border-color:#ffffff59;backdrop-filter:blur(2px)}
.rdds-landing .l-ghost:hover{background:#fff;color:var(--ink);border-color:#fff;box-shadow:none}
.rdds-landing .l-btn-full{width:100%;justify-content:center;margin-top:6px}

/* hero */
.rdds-landing .l-hero{position:relative;min-height:100svh;display:flex;align-items:flex-end;overflow:hidden}
.rdds-landing .l-hero-media{position:absolute;inset:0;z-index:0;overflow:hidden;background:#14120f}
.rdds-landing .l-hero-media img{position:absolute;left:0;top:-9%;width:100%;height:118%;object-fit:cover;object-position:50% 44%}
.rdds-landing.anim .l-hero-media img{opacity:0;animation:l-heroin 1.6s var(--ease) forwards}
@keyframes l-heroin{to{opacity:1}}
.rdds-landing .l-scrim{position:absolute;inset:0;z-index:1;pointer-events:none;background:linear-gradient(180deg,rgba(18,16,12,.66) 0%,rgba(18,16,12,.2) 20%,rgba(18,16,12,0) 42%,rgba(18,16,12,.3) 62%,rgba(20,18,15,.74) 84%,var(--paper) 100%)}
.rdds-landing .l-pin{position:absolute;z-index:2;width:14px;height:14px;border-radius:50%;background:var(--brick);border:2px solid #fff;box-shadow:0 2px 10px #00000070}
.rdds-landing .l-pin-amber{background:#d6a53c}
.rdds-landing .l-pin::after{content:"";position:absolute;inset:-8px;border-radius:50%;border:1px solid #ffffff85}
.rdds-landing.anim .l-pin::after{animation:l-ping 2.8s var(--ease) infinite}
@keyframes l-ping{0%{transform:scale(.55);opacity:.9}70%,100%{transform:scale(1.9);opacity:0}}
.rdds-landing .l-hero-inner{position:relative;z-index:3;width:100%;padding-top:calc(var(--nav-h) + 46px);padding-bottom:clamp(56px,10vh,124px)}
.rdds-landing .l-tag{display:inline-flex;align-items:center;gap:9px;font-size:12.5px;font-weight:600;color:#f4efe6;margin-bottom:18px}
.rdds-landing .l-dot{position:relative;width:8px;height:8px;border-radius:50%;background:var(--brick-lite)}
.rdds-landing .l-dot::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:1px solid var(--brick-lite);opacity:.6}
.rdds-landing.anim .l-dot::after{animation:l-ping 2.8s var(--ease) infinite}
.rdds-landing .l-h1{color:#fff;font-size:clamp(2.9rem,8vw,6.7rem);letter-spacing:-.038em;max-width:16ch;text-shadow:0 2px 40px rgba(0,0,0,.28)}
.rdds-landing .l-h1 .l-em{color:var(--brick-lite)}
.rdds-landing .l-lead{color:#efe9df;font-size:clamp(15px,1.35vw,17.5px);max-width:50ch;margin-top:var(--s3);line-height:1.62;text-shadow:0 1px 20px rgba(0,0,0,.3)}
.rdds-landing .l-cta{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:var(--s4)}
.rdds-landing .l-note{font-size:13px;color:#e8e2d6c4}
.rdds-landing .l-hero-foot{position:absolute;left:0;right:0;bottom:clamp(18px,3vh,30px);z-index:3;pointer-events:none}
.rdds-landing .l-hero-foot .l-wrap{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;color:#efe9dfcc;font-size:12.5px}
.rdds-landing .l-cue{display:inline-flex;align-items:center;gap:8px;letter-spacing:.08em;text-transform:uppercase;font-size:11px}
.rdds-landing .l-cue i{display:block;width:1px;height:26px;background:linear-gradient(#efe9df00,#efe9dfcc);position:relative;overflow:hidden}
.rdds-landing.anim .l-cue i::after{content:"";position:absolute;top:-26px;left:0;width:1px;height:26px;background:var(--brick-lite);animation:l-drop 2.2s var(--ease) infinite}
@keyframes l-drop{0%{transform:translateY(0)}60%,100%{transform:translateY(52px)}}
@media(max-width:640px){.rdds-landing .l-loc{display:none}}

/* ribbon */
.rdds-landing .l-posthero .l-wrap{padding:var(--s5) var(--edge) 0}
.rdds-landing .l-ribbon{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--s4)}
.rdds-landing .l-stat .l-k{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:clamp(1.7rem,3vw,2.5rem);letter-spacing:-.03em;line-height:1}
.rdds-landing .l-stat .l-k .l-u{font-size:.44em;color:var(--ink-3);font-weight:500;margin-left:5px}
.rdds-landing .l-stat .l-l{font-size:13px;color:var(--ink-2);margin-top:8px;max-width:22ch}
@media(max-width:760px){.rdds-landing .l-ribbon{grid-template-columns:1fr 1fr;gap:var(--s3) var(--s4)}}

/* sections */
.rdds-landing .l-section{margin-top:var(--s6)}
.rdds-landing .l-statement{border-top:3px solid var(--ink);padding-top:var(--s4)}
.rdds-landing .l-statement h2{font-size:clamp(2rem,4.6vw,3.5rem);letter-spacing:-.03em;max-width:18ch}
.rdds-landing .l-statement p{font-size:16px;color:var(--ink-2);max-width:54ch;margin-top:var(--s3);line-height:1.6}
.rdds-landing .l-lede{display:grid;grid-template-columns:1fr 1fr;gap:var(--s5);align-items:end;margin-bottom:var(--s4)}
.rdds-landing .l-lede h2{font-size:clamp(2rem,4.4vw,3.4rem);letter-spacing:-.03em}
.rdds-landing .l-say{font-size:15.5px;color:var(--ink-2);max-width:44ch}
@media(max-width:820px){.rdds-landing .l-lede{grid-template-columns:1fr;gap:var(--s3)}}

/* loop */
.rdds-landing .l-loop{border-top:1px solid var(--ink)}
.rdds-landing .l-step{display:grid;grid-template-columns:64px 200px 1fr auto;gap:var(--s4);align-items:baseline;padding:var(--s3) 0;border-bottom:1px solid var(--line);transition:transform .2s var(--ease)}
.rdds-landing .l-step:hover{transform:translateX(10px)}
.rdds-landing .l-step .l-no{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:22px;color:var(--brick);line-height:1}
.rdds-landing .l-step h3{font-size:clamp(1.3rem,2.2vw,1.7rem);letter-spacing:-.02em}
.rdds-landing .l-step p{font-size:14.5px;color:var(--ink-2);max-width:52ch}
.rdds-landing .l-step .l-to{font-size:12.5px;color:var(--ink-3);white-space:nowrap;opacity:0;transition:opacity .2s var(--ease)}
.rdds-landing .l-step:hover .l-to{opacity:1}
@media(max-width:900px){.rdds-landing .l-step{grid-template-columns:44px 1fr;gap:6px var(--s3)}.rdds-landing .l-step p{grid-column:2}.rdds-landing .l-step .l-to{display:none}}

/* severity */
.rdds-landing .l-sev{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.rdds-landing .l-band{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 16px 18px}
.rdds-landing .l-band .l-bar{height:6px;border-radius:4px;margin-bottom:14px}
.rdds-landing .l-band .l-c{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:20px}
.rdds-landing .l-band .l-n{font-size:12px;font-weight:600;color:var(--ink-2);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
.rdds-landing .l-band .l-a{font-size:13px;color:var(--ink-2);margin-top:10px;line-height:1.5}
.rdds-landing .l-band[data-s="1"] .l-bar{background:#8a9a5b}.rdds-landing .l-band[data-s="1"] .l-c{color:#5f6b3a}
.rdds-landing .l-band[data-s="2"] .l-bar{background:#d6a53c}.rdds-landing .l-band[data-s="2"] .l-c{color:#9a7413}
.rdds-landing .l-band[data-s="3"] .l-bar{background:#cf7a2e}.rdds-landing .l-band[data-s="3"] .l-c{color:#a15718}
.rdds-landing .l-band[data-s="4"] .l-bar{background:#c0472a}.rdds-landing .l-band[data-s="4"] .l-c{color:#a8391f}
.rdds-landing .l-band[data-s="5"] .l-bar{background:#8a2d16}.rdds-landing .l-band[data-s="5"] .l-c{color:#8a2d16}
@media(max-width:820px){.rdds-landing .l-sev{grid-template-columns:1fr 1fr}}
@media(max-width:460px){.rdds-landing .l-sev{grid-template-columns:1fr}}

/* ticker */
.rdds-landing .l-ticker{display:flex;flex-wrap:wrap;gap:10px}
.rdds-landing .l-tick{display:inline-flex;align-items:center;gap:10px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13px}
.rdds-landing .l-tick-sev{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:12px;padding:2px 8px;border-radius:6px;background:#a8391f14;color:var(--brick-deep)}
.rdds-landing .l-tick-sev[data-s="5"]{background:#8a2d1618;color:#8a2d16}
.rdds-landing .l-tick-sev[data-s="4"]{background:#c0472a18;color:#a8391f}
.rdds-landing .l-tick-name{font-weight:600}
.rdds-landing .l-tick-seen{color:var(--ink-3)}
.rdds-landing .l-tick-when{color:var(--ink-3);margin-left:2px}

/* why + badge */
.rdds-landing .l-scale{display:grid;grid-template-columns:1fr 300px;gap:var(--s5);align-items:start}
.rdds-landing .l-why{display:grid;grid-template-columns:1fr 1fr;gap:0 var(--s5)}
.rdds-landing .l-reason{padding:var(--s4) 0;border-top:1px solid var(--line)}
.rdds-landing .l-reason:nth-child(-n+2){border-top:2px solid var(--ink)}
.rdds-landing .l-reason h3{font-size:1.35rem;letter-spacing:-.02em;margin-bottom:8px}
.rdds-landing .l-reason h3 .l-d{color:var(--brick);margin-right:8px}
.rdds-landing .l-reason p{font-size:14.5px;color:var(--ink-2);max-width:46ch}
.rdds-landing .l-badge{background:var(--ink);color:var(--paper);border-radius:16px;padding:24px;position:sticky;top:calc(var(--nav-h) + 16px)}
.rdds-landing .l-badge-top{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#cfcbc0}
.rdds-landing .l-badge-grade{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:800;font-size:5.5rem;line-height:.9;margin:8px 0 4px;color:var(--brick-lite)}
.rdds-landing .l-badge-grade[data-g="A"]{color:#9ec06a}.rdds-landing .l-badge-grade[data-g="B"]{color:#cbd06a}
.rdds-landing .l-badge-grade[data-g="D"]{color:#e59a5a}.rdds-landing .l-badge-grade[data-g="E"]{color:#e07a5a}
.rdds-landing .l-badge-score{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:1.6rem;color:#fff}
.rdds-landing .l-badge-score span{font-size:.6em;color:#a5a196;font-weight:500}
.rdds-landing .l-badge-note{font-size:12.5px;color:#a5a196;margin-top:12px;line-height:1.5}
@media(max-width:820px){.rdds-landing .l-scale{grid-template-columns:1fr}.rdds-landing .l-why{grid-template-columns:1fr}.rdds-landing .l-reason:nth-child(2){border-top:1px solid var(--line)}.rdds-landing .l-badge{position:static;max-width:340px}}

/* close + form */
.rdds-landing .l-close{margin-top:var(--s6);background:var(--ink);color:var(--paper)}
.rdds-landing .l-close .l-wrap{padding:var(--s6) var(--edge);display:grid;grid-template-columns:1.05fr .95fr;gap:var(--s5);align-items:start}
.rdds-landing .l-close h2{font-size:clamp(2.2rem,5.2vw,4rem);letter-spacing:-.035em;color:var(--paper)}
.rdds-landing .l-close h2 .l-em{color:var(--brick-lite)}
.rdds-landing .l-close-copy p{color:#cfcbc0;max-width:42ch;margin-top:var(--s3);font-size:15.5px}
.rdds-landing .l-close-list{list-style:none;margin:var(--s4) 0 0;padding:0;display:flex;flex-direction:column;gap:10px}
.rdds-landing .l-close-list li{position:relative;padding-left:24px;color:#e7e5df;font-size:14.5px}
.rdds-landing .l-close-list li::before{content:"→";position:absolute;left:0;color:var(--brick-lite);font-weight:700}
.rdds-landing .l-form-wrap{background:var(--paper);border-radius:16px;padding:26px;color:var(--ink)}
.rdds-landing .l-form h3{font-size:1.4rem;letter-spacing:-.02em;margin-bottom:16px}
.rdds-landing .l-form label{display:block;font-size:13px;font-weight:600;color:var(--ink-2);margin-bottom:12px}
.rdds-landing .l-form input,.rdds-landing .l-form textarea{width:100%;margin-top:6px;padding:11px 13px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);font:inherit;font-size:14px;font-weight:400}
.rdds-landing .l-form input:focus,.rdds-landing .l-form textarea:focus{outline:none;border-color:var(--brick)}
.rdds-landing .l-form textarea{resize:vertical}
.rdds-landing .l-hp{position:absolute;left:-9999px;width:1px;height:1px;opacity:0}
.rdds-landing .l-err{background:#a8391f12;border:1px solid #a8391f44;color:var(--brick-deep);border-radius:9px;padding:10px 12px;font-size:13px;margin-bottom:12px}
.rdds-landing .l-form-fine{font-size:12px;color:var(--ink-3);margin-top:12px;text-align:center}
.rdds-landing .l-sent{text-align:center;padding:24px 8px}
.rdds-landing .l-sent-mark{width:52px;height:52px;border-radius:50%;background:#8a9a5b22;color:#5f6b3a;display:flex;align-items:center;justify-content:center;font-size:26px;margin:0 auto 14px}
.rdds-landing .l-sent h3{font-size:1.4rem;margin-bottom:8px}
.rdds-landing .l-sent p{color:var(--ink-2);font-size:14.5px}
@media(max-width:820px){.rdds-landing .l-close .l-wrap{grid-template-columns:1fr;gap:var(--s4)}}

/* footer */
.rdds-landing .l-footer{padding:var(--s4) 0 var(--s5)}
.rdds-landing .l-footer .l-wrap{display:flex;flex-wrap:wrap;gap:12px 28px;justify-content:space-between;font-size:13px;color:var(--ink-3)}
.rdds-landing .l-links{display:flex;gap:18px}
.rdds-landing .l-links a:hover{color:var(--brick-deep)}

/* safety-first trio */
.rdds-landing .l-safe3{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--s4) var(--s5);border-top:2px solid var(--ink);padding-top:var(--s4)}
.rdds-landing .l-safe .l-safe-k{font-family:"Bricolage Grotesque","Chivo",sans-serif;font-weight:700;font-size:1.2rem;letter-spacing:-.02em;color:var(--brick-deep);margin-bottom:8px}
.rdds-landing .l-safe p{font-size:14.5px;color:var(--ink-2);max-width:40ch;line-height:1.6}
@media(max-width:820px){.rdds-landing .l-safe3{grid-template-columns:1fr;gap:var(--s3)}}
`
