# Dye sharpness, Approach 1: Selle-style MacCormack + optional unsharp pass

**Date:** 2026-05-14
**Status:** Approved for planning
**Author:** Alan (with Claude)

## Goal

Reduce visible numerical diffusion of the dye field `rho` in the interactive
fluid simulation, while keeping the existing Eulerian grid architecture and not
changing the velocity advection. This is "Approach 1" of a longer brainstorm —
intentionally low-risk and drop-in, with Approaches 2 (CIP) and 3 (hybrid
particle-grid) reserved for a later round if this proves insufficient.

## Scope

In scope:
- Replacing the existing finite-difference MacCormack scheme with a
  semi-Lagrangian MacCormack that includes an extrema clamp (Selle et al. 2008).
- Adding an optional per-step Laplacian-based unsharp pass that acts only on
  `rho`, controlled by a new `SimulationConfig.sharpen_strength` field.
- Keeping the new scheme number identical to the old one (`advection_scheme = 2`
  continues to mean "MacCormack", just the better version).
- Keeping `sharpen_strength = 0.0` by default so existing behavior (and the
  convergence test) is unchanged.

Out of scope (deferred to later rounds if needed):
- CIP / gradient-tracking advection (Approach 2).
- Particle-based dye representation (Approach 3).
- Vorticity confinement or any velocity-side anti-diffusion.
- Changes to `vel` advection — velocity continues to use WENO5+SSP-RK3.

## Background

The simulation uses Chorin's projection on a collocated `(res, res)` grid.
Three advection schemes are currently selectable via
`self.advection_scheme`:

- `0` — semi-Lagrangian (bilinear, single back-trace)
- `2` — MacCormack (finite-difference predictor/corrector, no limiter)
- `4` — WENO5 + SSP-RK3

WENO5 is the current default and is the highest-accuracy of the three on smooth
fields, but it still smears initially-sharp features in `rho` over time — both
for the synthetic checkerboard pattern in `init_patterns` and for image inputs
loaded via `init_from_image`.

The existing scheme 2 is unusual for graphics: it's a finite-difference
MacCormack, not the semi-Lagrangian variant that VFX studios actually ship. The
two share a name but behave very differently — the SL variant with an extrema
clamp is, in practice, less diffusive than WENO5 on smooth velocity fields, at
a fraction of the cost. So this work is not a *new* scheme so much as it is
replacing scheme 2 with the version of MacCormack that's actually useful.

## Design

### Piece A: Selle-style MacCormack

The new scheme replaces the contents of the two existing kernels
`advect_maccormack_step1` and `advect_maccormack_step2`. The dispatch in
`step()` (`elif self.advection_scheme == 2: ...`) is updated to use the new
kernel sequence.

**Algorithm (per cell):**

1. Back-trace from cell center `x` along the local velocity by `Δt`:
   `x_back = x − u(x)·Δt / dx` (in grid coordinates).
2. Compute `φ_hat = sample(field, x_back)` using the existing bilinear
   `sample()` function. This is a standard SL forward step.
3. From the *same* cell `x`, forward-trace along the local velocity by `Δt`:
   `x_fwd = x + u(x)·Δt / dx`.
4. Compute `φ_hat_hat = sample(φ_hat_field, x_fwd)` — i.e. read `φ_hat` at the
   forward-traced location. This estimates how much the SL pass smeared the
   original field.
5. Form the corrected estimate:
   `φ_corrected = φ_hat + 0.5 · (field(x) − φ_hat_hat)`.
6. Clamp `φ_corrected` to the min and max of the four grid values
   `field[i0,j0], field[i1,j0], field[i0,j1], field[i1,j1]` surrounding
   `x_back` (the donor cell neighborhood). This is the stability fix that
   prevents the corrector from overshooting near sharp features.
7. Write `φ_corrected` to `new_field`.

The clamp is the part most often skipped in textbook MacCormack writeups —
without it, the scheme is unstable at sharp features and either blows up or
has to fall back to plain SL. The clamp is what makes the whole thing useful
in practice.

**Kernel structure:**

The implementation uses two kernel passes rather than one, because step 4
needs to read `φ_hat` at arbitrary fractional locations and a single Taichi
kernel cannot guarantee that all cells of `φ_hat` are written before any are
read by another thread.

- Kernel `advect_maccormack_predict(field, phi_hat)`: implements steps 1–2.
  Writes `phi_hat` to a scratch field.
- Kernel `advect_maccormack_correct(field, phi_hat, new_field)`: implements
  steps 3–6. Reads `phi_hat` via `sample()` at the forward-traced location,
  computes the corrected value, applies the clamp using the donor neighborhood
  of the back-traced location, and writes `new_field`.

The corrector reads `field` at the cell and at the four donor-neighborhood
cells around the back-traced location. Because of those donor-neighborhood
reads, `new_field` must be a separate field from `field` — writing in-place
would race against other threads of the same kernel still reading the original
`field` values.

We use the existing RK3 intermediate fields (`rho_1`, `vel_1`) as the scratch
`phi_hat` storage, and continue to use `new_rho` / `new_vel` as the corrector
output, copied back into `rho` / `vel` as the final step. No new fields are
added. The dispatch in `step()` becomes:

```
elif self.advection_scheme == 2:
    self.advect_maccormack_predict(self.rho, self.rho_1)
    self.advect_maccormack_correct(self.rho, self.rho_1, self.new_rho)
    self.rho.copy_from(self.new_rho)
    self.advect_maccormack_predict(self.vel, self.vel_1)
    self.advect_maccormack_correct(self.vel, self.vel_1, self.new_vel)
    self.vel.copy_from(self.new_vel)
```

**Boundary conditions:** all three steps rely on the existing `sample()`
function, which already handles periodic vs wall (clamp-to-edge) via
`ti.static(self.bc_wall)`. The new kernels need to do the same `ti.static`
branch when computing the donor-neighborhood indices for the clamp in step 6
— periodic wrap for the default case, clamp-to-edge under wall BC.

### Piece B: Optional unsharp pass on `rho`

A new kernel `sharpen_rho()` applies a single Laplacian-based unsharp mask:

```
rho_new[i,j] = rho[i,j] - sharpen_strength · dt · Δrho[i,j]
```

where `Δrho` is the discrete 5-point Laplacian

```
Δrho[i,j] = (rho[i+1,j] + rho[i-1,j] + rho[i,j+1] + rho[i,j-1] - 4·rho[i,j]) / dx²
```

with the same periodic/wall boundary treatment as the rest of the simulation.

This is anti-diffusion: it is exactly the heat equation run backwards with
coefficient `sharpen_strength`. It is unconditionally unstable in the long
run, but at small per-step strengths it acts as a stable edge enhancement on
features the advection has just smoothed out, before they grow back into the
solution.

**Where in `step()`:** immediately after the advection block and any BC
application on `rho`, before forces and the projection. `rho` doesn't enter
the projection, so the placement within the step is mostly aesthetic, but
doing it right after advection makes the intent clearest.

**Config and default:** add `sharpen_strength: float = 0.0` to
`SimulationConfig`. The kernel is called every step but is a no-op (early
return on `sharpen_strength == 0.0`, evaluated as a `ti.static` branch). Zero
default means existing behavior is preserved — convergence test and current
visuals are unchanged unless the user opts in.

**Stability/bounds:** at zero default this is a no-op. The kernel does not
clamp `rho` to any range — the simulation already lets `rho` be any real
value (e.g. `fill_dye` adds, image init writes [0,1]), so we preserve that.
Users who turn on `sharpen_strength` are responsible for picking a value that
doesn't produce visual artifacts; this is an artistic dial, not a physical
one.

## File-level changes

- `simulation.py`:
  - Add `sharpen_strength: float = 0.0` to `SimulationConfig`.
  - Replace bodies of `advect_maccormack_step1` and `advect_maccormack_step2`
    with `advect_maccormack_predict` and `advect_maccormack_correct` (rename
    to make the change visible and reflect the new algorithm).
  - Add kernel `sharpen_rho()`.
  - Update the `advection_scheme == 2` branch in `step()` to use the new
    kernels.
  - In `step()`, after advection and BC, call `sharpen_rho()` if
    `sharpen_strength != 0.0` (gate on `ti.static` for free at compile time).
  - Update docstrings on the affected kernels and on `SimulationConfig`.

- `CLAUDE.md`: update the architecture section's mention of MacCormack to note
  it is now the Selle-style SL+clamp variant.

- `main.py`: no required changes. (Optional follow-up: bind a key to nudge
  `sharpen_strength` at runtime, similar to the `M` key for recording. Out of
  scope for this design — implement only if trivial.)

- `test_convergence.py`: no changes required; since the default
  `sharpen_strength = 0.0`, the test sees no behavioral change for
  unaffected runs. The convergence test should still pass for scheme 2 with
  improved (or at least non-worse) measured order on smooth fields.

- `test_dye_force.py`: no changes expected; sharpening defaults to off.

## Testing

1. **Convergence test (`test_convergence.py`)** — run with scheme 2 selected.
   Expectation: the new MacCormack should show a measured convergence order
   at least as good as before on smooth advection problems. Sharpening must
   stay disabled here (default is off).

2. **Visual A/B in `main.py`** — load `init_patterns` (the symmetric grid of
   dye) and run with each of schemes 0, 2 (new), and 4 under a simple
   persistent torque. Visually compare crispness of the grid lines after ~5
   seconds of simulated time. Expectation: scheme 2 (new) noticeably crisper
   than scheme 0, comparable to or crisper than scheme 4.

3. **Image input test** — load a sharp image via `init_from_image`, run
   under a mild rotational force, and compare scheme 2 (new) vs scheme 4.
   Expectation: visible edges in the image stay legible longer with the new
   scheme 2.

4. **Sharpening sanity check** — with scheme 2 (new), turn
   `sharpen_strength` up gradually (0.0 → 1.0 → 10.0) and confirm:
   (a) at 0.0, behavior is identical to no-sharpen run;
   (b) at moderate values, dye edges visibly crispen without obvious
       checkerboarding;
   (c) at high values, expected instability eventually appears — this is
       the documented behavior, not a bug.

5. **No-regression smoke** — `test_dye_force.py` continues to pass with no
   code changes to it.

Tests are visual/comparative rather than numeric for items 2–4, in keeping
with the artistic-purpose framing. Item 1 is the only quantitative gate.

## Risks and mitigations

- **MacCormack with clamp is still less accurate on velocity than WENO5 in
  some regimes** — but we're only changing the *option*, not the default.
  Existing WENO5 default (scheme 4) is untouched. Users opt in to scheme 2.

- **Anti-diffusion is unconditionally unstable in theory** — mitigated by
  being default-off, applied only once per step at small strength, and acting
  only on the passive scalar `rho` (which doesn't feed back into velocity).

- **Read-write aliasing in the corrector pass** — avoided by writing the
  corrector output to a separate field (`new_rho` / `new_vel`) and copying
  back, rather than writing in-place. The donor-neighborhood read in the
  clamp would otherwise race against other threads' writes within the same
  kernel.

- **Reuse of `rho_1` / `vel_1` for predictor scratch** — these fields are
  also used as RK3 stage-1 buffers by the WENO5 scheme. The two schemes are
  mutually exclusive at runtime (chosen via `advection_scheme`), so the
  reuse is safe; flagged here only because it's the kind of overload that
  invites confusion later. The kernel docstrings will note this.

## Non-goals / explicitly deferred

- CIP / gradient-tracking advection.
- Particle-based dye representation.
- Vorticity confinement.
- Sharpening on `vel`.
- Adaptive `sharpen_strength` (e.g. as a function of local gradient or time).
- Exposing a key in `main.py` to toggle sharpening at runtime — optional and
  only if trivial.

If Approach 1 doesn't produce enough visual improvement, the next step is to
revisit the Approach 2 / 3 brainstorm conclusions and pick one of those.
