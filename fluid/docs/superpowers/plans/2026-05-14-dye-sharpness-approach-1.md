# Dye Sharpness, Approach 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing finite-difference MacCormack advection (scheme 2) with a Selle-style semi-Lagrangian MacCormack + extrema clamp, and add an optional Laplacian-based unsharp pass on the dye field `rho`. Both target reduced visible diffusion of the dye field while keeping the existing Eulerian grid architecture untouched.

**Architecture:** Two independent changes inside `simulation.py`. The first replaces two existing kernels (`advect_maccormack_step1/2`) with two new ones (`advect_maccormack_predict/correct`) that operate as a predictor-corrector pair on top of the existing bilinear `sample()` function, with an extrema clamp on the corrector for stability. The second adds a `sharpen_rho` kernel that applies a discrete 5-point Laplacian-based unsharp mask to `rho`, gated by a new `SimulationConfig.sharpen_strength` field defaulting to zero. The default behavior of the simulation is unchanged: scheme 4 (WENO5) remains the default advection, and sharpening is off until explicitly enabled.

**Tech Stack:** Python 3, Taichi (GPU kernels), NumPy. Existing test harness: `test_convergence.py`, `test_dye_force.py`. Visualization via `main.py` + `ti.GUI`.

**Design spec:** `fluid/docs/superpowers/specs/2026-05-14-dye-sharpness-approach-1-design.md`

---

## File Inventory

- **Modify** `simulation.py`:
  - `SimulationConfig` dataclass (lines 8–20) — add `sharpen_strength` field.
  - `FluidSimulation.__init__` (lines 44–92) — allocate `self.predict_rho` and `self.predict_vel` scratch fields.
  - `advect_maccormack_step1` / `advect_maccormack_step2` (lines ~361–399) — replace with `advect_maccormack_predict` / `advect_maccormack_correct`.
  - New kernel `sharpen_rho` — added near the other advection kernels.
  - `step()` (lines ~627–end) — update the `advection_scheme == 2` dispatch branch; insert a sharpening call after the rho-advection block.
- **Modify** `test_convergence.py` (lines 75–77) — update the `scheme_id == 2` branch to call the renamed kernels and use the new scratch field, with a copy-back step.
- **Modify** `CLAUDE.md` — update the one-line MacCormack mention in the "Simulation method" section.

---

## Task 1: Add `sharpen_strength` field and predictor scratch fields

**Files:**
- Modify: `simulation.py:8-20` (`SimulationConfig`)
- Modify: `simulation.py:44-92` (`FluidSimulation.__init__`)

- [ ] **Step 1: Add `sharpen_strength` to `SimulationConfig`**

In `simulation.py`, add the field at the end of the dataclass, just below `wall_slip`:

```python
@dataclass
class SimulationConfig:
    res: int = 512
    dt: float = 0.0003
    init_type: str = 'patterns'
    # Persistent force coefficients. Set to non-zero to enable each force.
    # Forces are additive and can be blended freely.
    buoyancy_coeff: float = 0.0   # upward buoyancy proportional to dye density
    torque_coeff: float = 0.0     # counter-clockwise tangential force
    radial_coeff: float = 0.0     # outward radial force
    bc_type: str = 'periodic'   # 'periodic', 'wall', 'absorbing', or 'open'
    # 0.0 = no-slip, 1.0 = free-slip (only used when bc_type='wall').  All the action here happens above 0.99
    wall_slip: float = 0.0
    # Per-step Laplacian-based unsharp mask strength applied to rho. 0 = off
    # (default). Small positive values (single digits) act as edge enhancement;
    # large values are unstable. This is a purely artistic anti-diffusion knob.
    sharpen_strength: float = 0.0
```

- [ ] **Step 2: Allocate `self.predict_rho` and `self.predict_vel` in `__init__`**

In `simulation.py`, inside `FluidSimulation.__init__`, just after the RK3 intermediate-field allocations (around line 82, after `self.dq_vel`), add:

```python
        # Scratch fields for the Selle-style MacCormack predictor (phi_hat).
        # Used only when advection_scheme == 2. Kept separate from the RK3
        # intermediates so the two schemes never share scratch state.
        self.predict_rho = ti.field(float, shape=(self.res, self.res))
        self.predict_vel = ti.Vector.field(2, float, shape=(self.res, self.res))
```

- [ ] **Step 3: Verify the simulation still imports and runs**

Run:

```bash
python -c "from simulation import FluidSimulation, SimulationConfig; s = FluidSimulation(SimulationConfig(res=64)); s.init_patterns(); s.step(); print('ok')"
```

Expected output: `ok` (plus Taichi init banner). No exception. The default `advection_scheme = 4` (WENO5) is unaffected by the new fields.

- [ ] **Step 4: Commit**

```bash
git add fluid/simulation.py
git commit -m "Add sharpen_strength config and predict_rho/vel scratch fields

Preparatory plumbing for Selle-style MacCormack and the optional
sharpening pass; no behavior change at default settings."
```

---

## Task 2: Implement the new MacCormack predictor kernel

**Files:**
- Modify: `simulation.py:361-380` (replace `advect_maccormack_step1`)

- [ ] **Step 1: Replace `advect_maccormack_step1` with `advect_maccormack_predict`**

In `simulation.py`, replace the existing `advect_maccormack_step1` kernel (the predictor that uses forward local differences) with the following:

```python
    @ti.kernel
    def advect_maccormack_predict(self, field: ti.template(), phi_hat: ti.template()):
        """
        Predictor step of the Selle-style semi-Lagrangian MacCormack scheme.

        Performs one semi-Lagrangian back-trace from each grid cell along the
        local velocity and stores the bilinearly interpolated value in phi_hat:

            phi_hat(x) = field(x - u(x) * dt)

        where x = (i + 0.5, j + 0.5) * dx is the cell center. This is
        mathematically identical to one plain semi-Lagrangian advection step.
        It is paired with advect_maccormack_correct, which uses phi_hat to
        estimate and remove the diffusive error of this single SL pass.
        """
        for i, j in field:
            # Back-trace location (positions in grid coordinates)
            p = ti.Vector([i + 0.5, j + 0.5]) - self.dt * self.vel[i, j] / self.dx
            phi_hat[i, j] = self.sample(field, p.x - 0.5, p.y - 0.5)
```

- [ ] **Step 2: Verify the kernel compiles (smoke check)**

The kernel won't be invoked by anything yet (the existing scheme 2 dispatch still references the old `advect_maccormack_step1` name, which will fail at runtime if scheme 2 is selected — that's fine, scheme 2 isn't the default). Just verify the file still imports:

```bash
python -c "from simulation import FluidSimulation; print('ok')"
```

Expected output: `ok`. Any `NameError` or `SyntaxError` here means the rewrite didn't land cleanly.

- [ ] **Step 3: Commit**

```bash
git add fluid/simulation.py
git commit -m "Add Selle-style MacCormack predictor kernel

Replaces advect_maccormack_step1 (finite-difference predictor) with
advect_maccormack_predict (single semi-Lagrangian back-trace into a
scratch phi_hat field). Corrector and dispatch updates follow in the
next commits; scheme 2 is temporarily not invokable end-to-end."
```

---

## Task 3: Implement the new MacCormack corrector kernel with extrema clamp

**Files:**
- Modify: `simulation.py:381-399` (replace `advect_maccormack_step2`)

- [ ] **Step 1: Replace `advect_maccormack_step2` with `advect_maccormack_correct`**

In `simulation.py`, replace the existing `advect_maccormack_step2` with the following:

```python
    @ti.kernel
    def advect_maccormack_correct(self, field: ti.template(),
                                  phi_hat: ti.template(),
                                  new_field: ti.template()):
        """
        Corrector step of the Selle-style semi-Lagrangian MacCormack scheme.

        For each grid cell x = (i + 0.5, j + 0.5) * dx:

        1. Forward-trace from x by +u(x) * dt to obtain x_fwd, and sample
           phi_hat at that location to obtain phi_hat_hat. This estimates the
           value the predictor would have produced if started one step in the
           future and traced backward to x -- i.e. an estimate of the smearing
           introduced by the predictor.

        2. Form the corrected estimate
                  phi_corrected = phi_hat[x] + 0.5 * (field[x] - phi_hat_hat)
           which removes the predictor's leading-order bias.

        3. Clamp phi_corrected to the min and max of the four field values at
           the cells surrounding the back-traced location x_back = x - u(x)*dt
           (i.e. the same donor neighborhood the predictor's bilinear
           interpolation drew from). This clamp is what makes the scheme
           stable around sharp features: it prevents the corrector from
           creating values outside the range of the data it was reconstructed
           from.

        Honors periodic vs wall boundary conditions through the same dispatch
        used elsewhere in the file. Reference: Selle, Fedkiw, Kim, Liu,
        Rossignac (2008), "An Unconditionally Stable MacCormack Method".
        """
        for i, j in field:
            u = self.vel[i, j]

            # Forward-trace location (grid coordinates), used to sample phi_hat
            p_fwd = ti.Vector([i + 0.5, j + 0.5]) + self.dt * u / self.dx
            phi_hat_hat = self.sample(phi_hat, p_fwd.x - 0.5, p_fwd.y - 0.5)

            # Tentative second-order corrected value
            phi_corrected = phi_hat[i, j] + 0.5 * (field[i, j] - phi_hat_hat)

            # Back-trace location, used to find the donor neighborhood for the clamp
            p_back = ti.Vector([i + 0.5, j + 0.5]) - self.dt * u / self.dx
            i_b = int(ti.floor(p_back.x - 0.5))
            j_b = int(ti.floor(p_back.y - 0.5))

            # Default: periodic wrap. Wall: clamp-to-edge.
            i0 = i_b % self.res
            i1 = (i_b + 1) % self.res
            j0 = j_b % self.res
            j1 = (j_b + 1) % self.res
            if ti.static(self.bc_wall):
                i0 = ti.math.clamp(i_b,     0, self.res - 1)
                i1 = ti.math.clamp(i_b + 1, 0, self.res - 1)
                j0 = ti.math.clamp(j_b,     0, self.res - 1)
                j1 = ti.math.clamp(j_b + 1, 0, self.res - 1)

            v00 = field[i0, j0]
            v10 = field[i1, j0]
            v01 = field[i0, j1]
            v11 = field[i1, j1]

            # Component-wise min/max/clamp: works identically for scalar and
            # 2-vector fields, which is how the same kernel can advect both
            # rho and vel.
            lo = ti.min(ti.min(v00, v10), ti.min(v01, v11))
            hi = ti.max(ti.max(v00, v10), ti.max(v01, v11))

            new_field[i, j] = ti.math.clamp(phi_corrected, lo, hi)
```

- [ ] **Step 2: Verify the file still imports**

```bash
python -c "from simulation import FluidSimulation; print('ok')"
```

Expected output: `ok`. The kernel still isn't invokable end-to-end — Task 4 wires it up.

- [ ] **Step 3: Commit**

```bash
git add fluid/simulation.py
git commit -m "Add Selle-style MacCormack corrector with extrema clamp

Replaces advect_maccormack_step2 (finite-difference corrector) with
advect_maccormack_correct: forward-traces, samples phi_hat, forms the
predictor-corrector estimate, then clamps to the donor neighborhood
min/max for stability around sharp features. Works component-wise for
both rho (scalar) and vel (2-vector)."
```

---

## Task 4: Update `step()` dispatch for scheme 2

**Files:**
- Modify: `simulation.py` `step()` method (the `elif self.advection_scheme == 2:` branch)

- [ ] **Step 1: Update the scheme-2 dispatch in `step()`**

In `simulation.py`, find the block in `step()` that begins:

```python
        elif self.advection_scheme == 2:
            # MacCormack
            self.advect_maccormack_step1(self.rho, self.new_rho)
            self.advect_maccormack_step2(self.rho, self.new_rho, self.rho)
            self.advect_maccormack_step1(self.vel, self.new_vel)
            self.advect_maccormack_step2(self.vel, self.new_vel, self.vel)
```

and replace it with:

```python
        elif self.advection_scheme == 2:
            # Selle-style semi-Lagrangian MacCormack with extrema clamp.
            # The corrector cannot be written in-place because it reads field
            # at the donor neighborhood around the back-traced location, so we
            # write into new_rho / new_vel and copy back.
            self.advect_maccormack_predict(self.rho, self.predict_rho)
            self.advect_maccormack_correct(self.rho, self.predict_rho, self.new_rho)
            self.rho.copy_from(self.new_rho)
            self.advect_maccormack_predict(self.vel, self.predict_vel)
            self.advect_maccormack_correct(self.vel, self.predict_vel, self.new_vel)
            self.vel.copy_from(self.new_vel)
```

- [ ] **Step 2: Smoke test scheme 2 end-to-end**

```bash
python -c "
from simulation import FluidSimulation, SimulationConfig
s = FluidSimulation(SimulationConfig(res=64))
s.advection_scheme = 2
s.init_patterns()
for _ in range(20):
    s.step()
print('ok')
"
```

Expected output: `ok`. This confirms the new scheme runs without throwing, advancing 20 steps from the default checkerboard pattern at low resolution.

- [ ] **Step 3: Commit**

```bash
git add fluid/simulation.py
git commit -m "Wire new MacCormack predictor/corrector into step() dispatch

Scheme 2 now invokes advect_maccormack_predict + advect_maccormack_correct
with the dedicated predict_rho / predict_vel scratch fields. Output is
written into new_rho / new_vel and copied back, since the corrector's
donor-neighborhood read forbids in-place output."
```

---

## Task 5: Update `test_convergence.py` to use the renamed kernels

**Files:**
- Modify: `test_convergence.py:75-77`

The convergence test references the old kernel names directly and writes the corrector output back into `sim.rho` in-place. Both need to change.

- [ ] **Step 1: Update the scheme-2 branch in `run_test`**

In `test_convergence.py`, locate the block:

```python
        elif sim.advection_scheme == 2:
            sim.advect_maccormack_step1(sim.rho, sim.new_rho)
            sim.advect_maccormack_step2(sim.rho, sim.new_rho, sim.rho)
```

and replace it with:

```python
        elif sim.advection_scheme == 2:
            sim.advect_maccormack_predict(sim.rho, sim.predict_rho)
            sim.advect_maccormack_correct(sim.rho, sim.predict_rho, sim.new_rho)
            sim.rho.copy_from(sim.new_rho)
```

- [ ] **Step 2: Verify the file still imports**

```bash
python -c "import test_convergence; print('ok')"
```

Run from `/Users/alan/code/math/fluid`. Expected output: `ok`.

- [ ] **Step 3: Commit**

```bash
git add fluid/test_convergence.py
git commit -m "Update convergence test to call renamed MacCormack kernels

Tracks the rename from advect_maccormack_step{1,2} to
advect_maccormack_predict/correct, and switches to the out-of-place
output buffer that the new corrector requires."
```

---

## Task 6: Verify scheme 2 convergence is at least as good as before

**Files:** none modified.

- [ ] **Step 1: Run the convergence test**

From `/Users/alan/code/math/fluid`:

```bash
python test_convergence.py
```

Expected output: tables printing per-scheme L2 errors and convergence rates for smooth / square / triangle initial conditions, across resolutions 32, 64, 128, 256. The "MacCormack" rows should:

- For the smooth IC: report a convergence rate of roughly 2.0 between successive resolutions (the new scheme is nominally 2nd-order). Some variation is expected — anywhere from ~1.7 to ~2.2 is acceptable evidence of correct 2nd-order behavior on smooth data.
- For the square and triangle ICs: report finite, non-NaN L2 errors. The rate will be sub-2 (the clamp activates near discontinuities), and that is expected; the gate here is "doesn't blow up", not a specific rate.

If "MacCormack" rows show NaN, Inf, or rates well below 1.0 on the smooth IC, stop and investigate before proceeding. The most likely root causes are an indexing slip in the clamp donor neighborhood or a sign error in the forward-/back-trace.

The pre-existing scheme rows (Semi-Lagrangian, WENO5) should be unchanged from before this work. Schemes 1 (Upwind) and 3 (TVD) are listed in the test but call kernels that no longer exist in `simulation.py`; they will crash if reached. This is preexisting test-file rot and **out of scope** for this plan — note it for a future cleanup but do not fix it here.

- [ ] **Step 2: No commit** — verification only.

---

## Task 7: Visual A/B test of scheme 2 versus scheme 4

**Files:** temporary edit to `main.py` only; revert before committing.

- [ ] **Step 1: Briefly flip the default to scheme 2 in `main.py`**

Inspect `main.py` to find where the simulation is instantiated and where `advection_scheme` is (implicitly) left at 4. Temporarily set `sim.advection_scheme = 2` right after instantiation. Do **not** commit this change — it's just for the visual test.

- [ ] **Step 2: Run the interactive simulation**

```bash
python main.py
```

Watch the checkerboard `init_patterns` evolve for ~10 seconds of real time. Use the existing mouse / keyboard controls (right-click to inject dye, etc.) to stir the field. Take note of how crisp the dye edges remain.

- [ ] **Step 3: Run again with scheme 4 for comparison**

Switch the temporary line back to `sim.advection_scheme = 4` and re-run. Compare visually. Expectation: scheme 2 produces edges that stay noticeably crisper for longer than scheme 4 under the same flow. If scheme 2 looks visibly *worse* than scheme 4, or shows obvious artifacts (checkerboarding, banding, asymmetry that wasn't there before), stop and investigate.

- [ ] **Step 4: Image-input visual check (if a test image is at hand)**

The spec calls out image inputs as a separate target for sharpness. If a sample image is available, temporarily wire it in via `sim.init_from_image('path/to/image.png')` immediately after construction (replacing or in addition to `init_patterns`) and rerun the simulation under both scheme 2 and scheme 4. Expectation: image features (edges, fine detail) stay legible longer with scheme 2 than with scheme 4 under the same flow.

If no test image is available, skip — `init_patterns` is the canonical sharpness target and Step 3 is sufficient. Note that any image visible to the user will work; the test is qualitative.

- [ ] **Step 5: Revert the `main.py` change**

Make sure the temporary edit to `main.py` is reverted (`git diff main.py` should be empty for this directory). No commit needed for this task.

---

## Task 8: Implement the `sharpen_rho` kernel

**Files:**
- Modify: `simulation.py` — add a new kernel near the other advection kernels (suggested location: just after `advect_maccormack_correct`).

- [ ] **Step 1: Add the `sharpen_rho` kernel**

In `simulation.py`, add the following kernel:

```python
    @ti.kernel
    def sharpen_rho(self, strength: float):
        """
        Applies a single Laplacian-based unsharp pass to the dye field rho.

        Computes the discrete 5-point Laplacian of rho and writes
            new_rho[i, j] = rho[i, j] - strength * dt * laplacian(rho)[i, j]
        into self.new_rho. The caller is responsible for copying new_rho back
        into rho.

        This is mathematically the heat equation run backwards in time with
        coefficient `strength`. Anti-diffusion is unconditionally unstable in
        the long run; this kernel is intended as a per-step edge enhancer on
        features that advection has just smoothed, not as a standalone PDE
        solver. Keep `strength` modest (single-digit values are typical) and
        leave it at zero when sharpening isn't wanted (see SimulationConfig).

        Honors the same periodic vs wall boundary convention as the rest of
        the simulation.
        """
        for i, j in self.rho:
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            if ti.static(self.bc_wall):
                im1 = ti.math.clamp(i - 1, 0, self.res - 1)
                ip1 = ti.math.clamp(i + 1, 0, self.res - 1)
                jm1 = ti.math.clamp(j - 1, 0, self.res - 1)
                jp1 = ti.math.clamp(j + 1, 0, self.res - 1)
            laplacian = (self.rho[ip1, j] + self.rho[im1, j]
                         + self.rho[i, jp1] + self.rho[i, jm1]
                         - 4.0 * self.rho[i, j]) / (self.dx * self.dx)
            self.new_rho[i, j] = self.rho[i, j] - strength * self.dt * laplacian
```

- [ ] **Step 2: Smoke-test the kernel in isolation**

```bash
python -c "
from simulation import FluidSimulation, SimulationConfig
s = FluidSimulation(SimulationConfig(res=64))
s.init_patterns()
s.sharpen_rho(1.0)
s.rho.copy_from(s.new_rho)
print('ok')
"
```

Expected output: `ok`. This confirms the kernel runs without raising. It also leaves `rho` slightly sharpened, but we don't assert anything about the result yet — that's Task 10.

- [ ] **Step 3: Commit**

```bash
git add fluid/simulation.py
git commit -m "Add Laplacian-based sharpen_rho kernel

Writes a per-step unsharp pass on rho into new_rho; the caller copies
back. Intended as an optional artistic anti-diffusion knob, not a
physical operator -- documented as such in the kernel docstring."
```

---

## Task 9: Wire `sharpen_rho` into `step()` with a config gate

**Files:**
- Modify: `simulation.py` `step()` method (insert call after the advection block).

- [ ] **Step 1: Add the sharpening call in `step()`**

In `simulation.py`, in `step()`, locate the advection block (the `if/elif` chain over `self.advection_scheme`). Immediately after that block (after the WENO5 branch and before the `if self.bc_wall and not self.bc_open:` block that applies velocity BC), insert:

```python
        # Optional per-step Laplacian-based unsharp pass on the dye field.
        # Default-off via SimulationConfig.sharpen_strength = 0.0. When on,
        # acts as artistic anti-diffusion -- see the sharpen_rho docstring.
        if self.config.sharpen_strength != 0.0:
            self.sharpen_rho(self.config.sharpen_strength)
            self.rho.copy_from(self.new_rho)
```

The Python-level `if` keeps the kernel call entirely out of the hot path when sharpening is disabled (the default), so there is no perf cost in that case. Strength is passed as a kernel argument so it can later be changed at runtime without recompiling.

- [ ] **Step 2: Verify default-off path is unaffected**

Run a quick smoke check that the default config (sharpening off) still runs:

```bash
python -c "
from simulation import FluidSimulation, SimulationConfig
s = FluidSimulation(SimulationConfig(res=64))
s.init_patterns()
for _ in range(20):
    s.step()
print('ok')
"
```

Expected output: `ok`.

- [ ] **Step 3: Verify sharpening-on path runs**

```bash
python -c "
from simulation import FluidSimulation, SimulationConfig
s = FluidSimulation(SimulationConfig(res=64, sharpen_strength=5.0))
s.init_patterns()
for _ in range(20):
    s.step()
print('ok')
"
```

Expected output: `ok`. Just checks the on-path runs; we'll inspect output quality in Task 10.

- [ ] **Step 4: Commit**

```bash
git add fluid/simulation.py
git commit -m "Wire optional sharpening pass into step()

Calls sharpen_rho after the advection block when sharpen_strength is
non-zero, and copies the result back. Default config leaves it disabled,
so existing behavior is preserved."
```

---

## Task 10: Verify default-off is bit-identical to pre-change behavior, then visual check

**Files:** temporary edit to `main.py` only; revert.

- [ ] **Step 1: Regression check via `test_dye_force.py`**

```bash
python test_dye_force.py
```

Run from `/Users/alan/code/math/fluid`. Expected: same output as before Task 8 (sharpening default-off means no behavior change here).

- [ ] **Step 2: Visual no-op check at `sharpen_strength = 0`**

Temporarily set `SimulationConfig(sharpen_strength=0.0)` (or rely on the default) in `main.py`, run the simulation for 5–10 seconds, and confirm visually it looks identical to a pre-change run. No `main.py` commit.

- [ ] **Step 3: Visual moderate-strength check**

Temporarily set `SimulationConfig(sharpen_strength=10.0)` in `main.py` (and `advection_scheme = 2` if you want the full Approach 1 effect). Run for 5–10 seconds. Expectation: visibly crisper dye edges than the no-op run; no obvious checkerboarding or runaway values.

- [ ] **Step 4: High-strength stability awareness**

Set `sharpen_strength = 200.0` and run briefly. Expectation: instability (overshoots, oscillations, possible blow-up). This is *documented behavior*, not a bug — anti-diffusion is unstable in general. Confirm the simulation doesn't take down the GUI / kernel, just produces visibly bad output. If it crashes Taichi or hangs, that *is* a bug — file a note but it's out of scope to fix here.

- [ ] **Step 5: Revert any temporary `main.py` edits**

```bash
git diff fluid/main.py
```

Expected: empty.

---

## Task 11: Update `CLAUDE.md` MacCormack mention

**Files:**
- Modify: `CLAUDE.md` — the "Simulation method" section's advection step description.

- [ ] **Step 1: Update the architecture note**

Inspect `CLAUDE.md` and find any line referencing "MacCormack" in the architecture / simulation-method section. (There may or may not be one — check before editing.) If a line exists describing scheme 2 as "MacCormack", update it to clarify it is the Selle-style semi-Lagrangian + extrema clamp variant. Sample wording:

> Scheme `2` is the semi-Lagrangian MacCormack with extrema clamp (Selle et al. 2008), suitable when less-diffusive dye transport is wanted without paying the WENO5 cost.

If `CLAUDE.md` already documents the advection schemes in detail elsewhere, prefer extending that section over inserting a stray note. If no MacCormack reference exists in `CLAUDE.md`, skip this task — the spec is the source of truth.

- [ ] **Step 2: Commit (only if a change was made)**

```bash
git add fluid/CLAUDE.md
git commit -m "Note Selle-style MacCormack in CLAUDE.md architecture section"
```

---

## Out of scope (deferred)

- Vorticity confinement on `vel` (deferred per design spec).
- CIP / gradient-tracking advection (Approach 2).
- Hybrid particle-grid representation (Approach 3).
- Cleaning up stale scheme 1 (Upwind) and scheme 3 (TVD) entries in `test_convergence.py` — they crash if their branches are reached but predate this work and aren't the user's current concern.
- Interactive `sharpen_strength` adjustment (slider or key binding in `main.py`).

## Done criteria

1. `python test_convergence.py` runs to completion. Scheme 2's row on the smooth IC shows ~2nd-order convergence; on square/triangle ICs it produces finite errors. The Semi-Lagrangian and WENO5 rows are unchanged from before.
2. `python main.py` with `advection_scheme = 2` keeps dye edges noticeably crisper than with `advection_scheme = 4`, under the same flow, over ~10 seconds of real time.
3. With `SimulationConfig.sharpen_strength = 0.0` (default), simulation behavior is identical to before this work; `test_dye_force.py` output is unchanged.
4. With `sharpen_strength` set to a moderate single-digit value, dye edges visibly crispen further without obvious checkerboarding.
5. The spec at `fluid/docs/superpowers/specs/2026-05-14-dye-sharpness-approach-1-design.md` accurately describes the shipped code (with the one known discrepancy that `test_convergence.py` *did* need updates, addressed in Task 5).
