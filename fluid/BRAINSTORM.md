# Artistic Effects Brainstorm

Parking lot of artistic effect ideas for the fluid sim. Items here have been
discussed but not yet implemented. Pick one (or a small set) to develop into
a focused plan when ready.

The first batch — curl-noise force, vorticity-aligned drive, noise-driven
gravity wells, and Coriolis term — was implemented in the
[plan from 2026-05-24](../../.claude/plans/help-me-brainstorm-some-replicated-kettle.md)
and is not re-listed below.

## A. Force fields (in-kernel additions)

Single-kernel extensions that compose with the existing additive force system.

- **Strain-aligned force** — force along the principal eigenvector of the
  strain-rate tensor, emphasizing shear bands.
- **Audio-reactive coefficient** — modulate any persistent-force coeff by
  mic-input RMS or a single FFT band, pulsing to music.
- **Reaction-diffusion coupled force** — run Gray-Scott on a sibling field,
  use its gradient as a force. Adds biological "spots and stripes" that flow.
  ⭐ *Flagged for follow-up.*

## B. Multi-field advection

Extends `rho` from scalar to vector, or adds sibling fields advected alongside.

- **RGB per-channel velocity offset** — three rho channels, each advected by
  `vel + offset_c`. Flowing chromatic aberration. ⭐ *Flagged for follow-up.*
- **HSV-space advection** — advect H, S, V independently; H drifts on its own
  scalar velocity → constant spectral rotation while shape stays coherent.
- **Dye age field** — scalar `t_since_emission` per cell, advected and
  incremented. Color via age → rainbow tails / time-stamps.
- **Smoke + temperature** — temperature field self-drives buoyancy →
  self-organizing thermals/plumes.
- **Multi-fluid marbling** — N immiscible labels (one-hot per cell), rendered
  with sharp palette → ebru-style Eulerian marbling.

## C. Render-the-flow instead of the dye

Re-use fields already computed; throw `rho` away and visualize the velocity
field itself.

- **Vorticity diverging colormap** — render signed ω with blue/white/red;
  CW/CCW eddies visible directly.
- **Q-criterion / λ₂** — second invariant of ∇u; highlights vortex cores.
- **Schlieren** — `|∇ρ|` with directional shading; wind-tunnel look.
- **LIC (line integral convolution)** — smear a noise texture along velocity
  streamlines. Exposes flow structure without dye.
- **Pressure-field render** — diverging colormap on `p`; shows compression
  zones.

## D. Advected field drives a post-process on a static texture

The "alpha channel / magnification" direction — use the simulated field as
a deformation/lookup applied to a separate image.

- **Heat-haze displacement** — `(∂ρ/∂x, ∂ρ/∂y)` as UV offset into a static
  background image.
- **Magnification field** — `rho` controls per-pixel zoom into background →
  bulging, melting glass.
- **Phase-field iridescence** — `rho` as optical thickness, render thin-film
  interference colors. Oil-slick rainbows that flow.
- **Caustics** — `det(Hessian(ρ))` or `∇²ρ` with bright-line palette;
  underwater-pool patterns.
- **Refractive lens stack** — `rho` displaces samples into a pre-rendered
  fractal/noise; endless ripples on a kaleidoscope.

## E. Domain / coordinate warps (cheap post-process)

Apply only at render time. No simulation change.

- **Polar wrap** — render square domain wrapped around a disc (periodic-x BC
  makes it seamless).
- **Kaleidoscope wedge** — sample π/N wedge, mirror-tile.
- **Droste / recursive zoom** — composite previous output back in with small
  rotation+scale.
- **Log-polar warp** — tunnel/wormhole look; rotation becomes vertical
  scrolling.

## F. Lagrangian sprinkles

Sparse particles advected by the velocity field, rendered as point/line
overlays.

- **Tracer particles** — N points advected by vel, rendered as bright dots.
  "Spark" effects.
- **Streak trails** — particles with fading lines → calligraphy strokes.
- **Lissajous / phyllotactic emitters** — moving dye sources tracing
  parametric curves into the fluid.

## Recommended bundles for future sessions

- **"Living color"** — Multi-channel rendering (B) + vorticity diverging
  colormap toggle (C). Big visual jump.
- **"Dimensional"** — Heat-haze displacement (D) + polar wrap (E) +
  tracer particles (F). Sim becomes a lens.
- **"Generative"** — Reaction-diffusion (A) + HSV advection (B) +
  iridescence (D). Self-organizing & spectral.
