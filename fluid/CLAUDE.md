# CLAUDE.md
Always write comments in the code, and docstrings for every function and class. Keep these concise and informative. No need for full sentences or paragraphs, but do describe the key components and functionality.
Use informative function names, even if they have to be a bit longer. Example: `select_candidate_words` rather than just `select_candidates`.
Use pure functions and simple classes whenever possible.
Rather than running git commit, ask the user to commit the code themselves.

## Running
The system contains no default `python`. Use the `python3` command instead.

```bash
# Install dependencies
pip3 install taichi numpy pillow imageio[ffmpeg]

# Run interactive simulation
python3 main.py

# Run convergence tests (benchmarks all advection schemes across resolutions)
python3 test_convergence.py

# Run dye force test
python3 test_dye_force.py
```

## Architecture

This is a 2D incompressible fluid simulation using [Taichi](https://taichi-lang.org/) for GPU-accelerated computation.
The main goal of the simulation is for artistic purposes, so results do not need to be totally scientifically accurate as long as they faithfully visually represent Eulerian, crisp, non-difussive fluid dynamics.

**Core files:**
- `simulation.py` — All simulation logic in `FluidSimulation` class with `SimulationConfig` dataclass
- `main.py` — Interactive GUI loop using `ti.GUI`. Press `M` in the GUI to toggle MP4 recording (output goes to `./recordings/`).
- `video_recorder.py` — `VideoRecorder` class that streams `sim.rho` frames to MP4 via `imageio[ffmpeg]`

**Simulation method (Chorin's Projection):**
1. **Advection** — advects `rho` (dye density) and `vel` (velocity) using the selected scheme
2. **External forces** — applies mouse/keyboard-triggered or persistent forces
3. **Pressure solve** — computes divergence, then solves ∇²p = ∇·u* via Jacobi iteration (100 steps)
4. **Projection** — subtracts ∇p from velocity to enforce incompressibility (∇·u = 0)

**Force system:**
- One-shot forces: `apply_force()`, `apply_bottom_force()`
- Dye injection: `fill_dye()` (adds density at a point, bound to right-click in `main.py`)
- Gradual forces (applied over `force_duration`): `apply_image_gradient_torque()`, `apply_dye_gradient_torque()`
- Persistent forces (per-step, until toggled off): buoyancy or torque based on `config.force_type`

**Grid layout:** Collocated grid, `(res, res)` cells, `dx = 1/res`, coordinates in [0, 1]. Taichi uses (x, y) = (i, j) indexing (column-major relative to display). Images are flipped vertically and transposed on load to match Taichi's origin convention.
