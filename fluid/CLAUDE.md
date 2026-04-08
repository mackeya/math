# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# Install dependencies
pip install taichi numpy pillow

# Run interactive simulation
python main.py

# Run convergence tests (benchmarks all advection schemes across resolutions)
python test_convergence.py

# Run dye force test
python test_dye_force.py
```

## Architecture

This is a 2D incompressible fluid simulation using [Taichi](https://taichi-lang.org/) for GPU-accelerated computation.

**Core files:**
- `simulation.py` — All simulation logic in `FluidSimulation` class with `SimulationConfig` dataclass
- `main.py` — Interactive GUI loop using `ti.GUI`

**Simulation method (Chorin's Projection):**
1. **Advection** — advects `rho` (dye density) and `vel` (velocity) using the selected scheme
2. **External forces** — applies mouse/keyboard-triggered or persistent forces
3. **Pressure solve** — computes divergence, then solves ∇²p = ∇·u* via Jacobi iteration (100 steps)
4. **Projection** — subtracts ∇p from velocity to enforce incompressibility (∇·u = 0)

**Advection schemes** (set via `sim.advection_scheme`):
- `0` — Semi-Lagrangian (stable, diffusive)
- `1` — Upwind (first-order, very diffusive)
- `2` — MacCormack (second-order predictor-corrector)
- `3` — TVD with Minmod limiter
- `4` — WENO5 + SSP-RK3 (highest quality, default)

**Force system:**
- One-shot forces: `apply_force()`, `apply_bottom_force()`
- Dye injection: `fill_dye()` (adds density at a point, bound to right-click in `main.py`)
- Gradual forces (applied over `force_duration`): `apply_image_gradient_torque()`, `apply_dye_gradient_torque()`
- Persistent forces (per-step, until toggled off): buoyancy or torque based on `config.force_type`

**Key Taichi details:**
- `ti.init(arch=ti.gpu)` is called at module import time in `simulation.py` — GPU falls back to CPU automatically
- Methods decorated with `@ti.kernel` run on GPU and cannot call Python functions; `@ti.func` are GPU-side helpers
- `ti.static()` is used for compile-time branching (e.g., force type selection in `_apply_persistent_force_kernel`)
- All fields use periodic boundary conditions (modular indexing)
- The `FluidSimulation` constructor in `test_convergence.py` uses a different signature (`FluidSimulation(res, dt=dt)`) than in `main.py` — the class accepts both styles via `SimulationConfig`

**Grid layout:** Collocated grid, `(res, res)` cells, `dx = 1/res`, coordinates in [0, 1]. Taichi uses (x, y) = (i, j) indexing (column-major relative to display). Images are flipped vertically and transposed on load to match Taichi's origin convention.
