# type: ignore
# pylint: skip-file
# flake8: noqa
import taichi as ti
import numpy as np
from dataclasses import dataclass

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
    # (default). The effective per-step coefficient is strength * dt / dx^2,
    # so usable values scale with resolution and timestep. At res=128 and
    # dt=3e-4, strengths around 1e-5 to 1e-4 give mild edge enhancement;
    # values above ~1e-3 blow up over a few hundred steps. Purely artistic
    # anti-diffusion -- has no physical meaning.
    sharpen_strength: float = 0.0
    # Advection scheme (also settable as a runtime attribute on the sim
    # instance). 4 = WENO5 + SSP-RK3, 8 = WENO5 on vel + Bidirectional
    # Characteristic Mapping on rho (bilinear render), 9 = WENO-Z + SSP-RK3,
    # 10 = TENO5 + SSP-RK3.
    advection_scheme: int = 10
    # Remap trigger threshold for the Characteristic Mapping Method
    # (advection_scheme == 8). When the maximum self-consistency error
    # max |X(Y(x)) - x| of the bidirectional maps exceeds this many cell
    # widths, the simulation bakes the current rho into rho_source and
    # resets both maps to identity. Smaller = more frequent remaps (more
    # grid-quantization, less visible map distortion); larger = less
    # frequent (preserves smooth image quality longer, but eventually the
    # map gets noticeably warped before snapping back). Units: cells (dx).
    cmm_remap_threshold_cells: float = 1.0
    # Pressure solver: 'jacobi' (iterative, works with all BCs) or 'fft'
    # (exact spectral solve, periodic BC only — automatically falls back to
    # Jacobi for wall/absorbing/open boundaries).
    pressure_solver: str = 'jacobi'
    # Number of multigrid V-cycles per pressure solve. 2–4 is typically
    # sufficient for near-exact convergence; more gives diminishing returns.
    mg_v_cycles: int = 4
    # Vorticity confinement strength (ε in Fedkiw 2001). At 0.0 the feature
    # is disabled. Typical useful range is 0.1–5.0; larger values increasingly
    # over-energise vortex cores and can cause instability.
    vorticity_confinement_strength: float = 0.0

ti.init(arch=ti.gpu) # Taichi will automatically fall back to CPU if GPU is not available

@ti.data_oriented
class FluidSimulation:
    """
    Simulates a 2D incompressible fluid using the Navier-Stokes equations
    (specifically the Euler equations, as viscosity is omitted here).

    The main equations governed are:
    1. Momentum equation: ∂u/∂t + (u · ∇)u = -∇p + f
       (Describes how velocity 'u' changes over time due to advection (u · ∇)u,
       pressure gradient ∇p, and external forces 'f')
    2. Incompressibility constraint: ∇ · u = 0
       (Ensures the fluid volume is conserved and the velocity field is divergence-free)

    The simulation advances using Chorin's Projection Method:
    - Step 1. Advection: Solve ∂q/∂t + (u · ∇)q = 0 for velocity and dye to get intermediate fields.
    - Step 2. External Forces: Apply forces 'f' to update the field.
    - Step 3. Compute Divergence: Find ∇ · u* of the intermediate velocity.
    - Step 4. Solve Pressure Poisson Equation: Solve ∇²p = ∇ · u* for pressure 'p'.
    - Step 5. Projection: Subtract the pressure gradient (u -= ∇p) to ensure ∇ · u = 0.
    """
    def __init__(self, config_or_res=None, dt: float = None, **kwargs):
        if isinstance(config_or_res, int):
            config = SimulationConfig(res=config_or_res)
            if dt is not None:
                config.dt = dt
        elif config_or_res is None:
            config = SimulationConfig(**kwargs)
        else:
            config = config_or_res
        self.config = config
        self.res = config.res
        self.dx = 1.0 / self.res
        self.dt = config.dt
        self.time = 0.0

        # Velocity fields (staggered grid or collocated? Let's use collocated for simplicity in demo)
        self.vel = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.new_vel = ti.Vector.field(2, float, shape=(self.res, self.res))

        # Density/Dye fields
        self.rho = ti.field(float, shape=(self.res, self.res))
        self.new_rho = ti.field(float, shape=(self.res, self.res))

        # Pressure fields
        self.p = ti.field(float, shape=(self.res, self.res))
        self.p_temp = ti.field(float, shape=(self.res, self.res))
        self.div = ti.field(float, shape=(self.res, self.res))

        # Geometric Multigrid fields for the pressure Poisson solver.
        # Level 0 = finest (res × res), level L-1 = coarsest (4 × 4).
        # Kernels using ti.template() compile a separate specialization
        # per unique field shape, so passing fields of different sizes to
        # the same kernel is safe and expected in Taichi.
        # mg_p[l]: pressure approximation / error correction at level l.
        # mg_f[l]: RHS — divergence at level 0, restricted residual below.
        self._mg_num_levels = int(np.log2(self.res)) - 1  # e.g. 8 for res=512
        self.mg_p = [
            ti.field(float, shape=(self.res >> l, self.res >> l))
            for l in range(self._mg_num_levels)
        ]
        self.mg_f = [
            ti.field(float, shape=(self.res >> l, self.res >> l))
            for l in range(self._mg_num_levels)
        ]

        self.advection_scheme = config.advection_scheme

        # RK3 intermediate fields
        self.rho_1 = ti.field(float, shape=(self.res, self.res))
        self.rho_2 = ti.field(float, shape=(self.res, self.res))
        self.dq_rho = ti.field(float, shape=(self.res, self.res))

        self.vel_1 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.vel_2 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.dq_vel = ti.Vector.field(2, float, shape=(self.res, self.res))

        # Bidirectional Characteristic Mapping (advection_scheme == 8):
        # rho_source is the frozen dye snapshot at the most recent remap.
        # backward_map (X) maps current grid points back to their initial
        # positions; advected by the velocity field each step. forward_map
        # (Y) is the per-cell forward trajectory from the remap moment;
        # used purely to detect when the maps have drifted enough that a
        # remap is needed. Both maps are reset to the identity at remap.
        self.rho_source = ti.field(float, shape=(self.res, self.res))
        self.backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.forward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.new_backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        # WENO5 + SSP-RK3 scratch for advecting the backward_map delta.
        # Kept separate from vel_1 / vel_2 / dq_vel so vel's own WENO step
        # never collides.
        self.delta_1 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.delta_2 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.dq_delta = ti.Vector.field(2, float, shape=(self.res, self.res))
        # 0-d accumulator for the max-self-consistency-error reduction.
        # Mirrors the pattern of self._max_vel_norm.
        self._map_distortion = ti.field(float, shape=())

        # Scalar 0-d field used as the accumulator for the max-CFL reduction
        # in max_cfl(). Lives on the simulation so we don't allocate per call.
        self._max_vel_norm = ti.field(float, shape=())

        # Scalar vorticity ω = ∂v/∂x - ∂u/∂y, used by vorticity confinement.
        self.vorticity = ti.field(float, shape=(self.res, self.res))

        # Gradual force application
        self.image_grad = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.force_duration = 0.0
        self.force_scale = 0.0
        self.dye_force_active = False

        self.bc_wall = (config.bc_type in ('wall', 'absorbing', 'open'))
        self.bc_absorbing = (config.bc_type == 'absorbing')
        self.bc_open = (config.bc_type == 'open')

    @ti.kernel
    def _init_patterns_kernel(self):
        """
        Kernel body for `init_patterns`. Sets velocity, pressure, and dye
        fields to zero, then paints the starting dye pattern (a grid of
        bright squares symmetric around the center). Kept separate from the
        Python `init_patterns` wrapper so that wrapper can seed the CMM state
        after the pattern is written.
        """
        self.rho.fill(0)
        self.vel.fill(0)
        self.p.fill(0)
        for i, j in self.rho:
            # Create a grid of dye symmetric around the center
            if ((i + 16) // 32) % 2 == 0 and ((j + 16) // 32) % 2 == 0:
                self.rho[i, j] = 1.0

            # # Add a central circle
            # dist = (ti.Vector([i * self.dx, j * self.dx]) - ti.Vector([0.5, 0.5])).norm()
            # if dist < 0.1:
            #     self.rho[i, j] = 1.0

            # # Sine wave boundary
            # boundary = self.res / 2 + ti.cos((i * self.dx * 2.0 - 1.0) * np.pi) * self.res * 0.1
            # if j < boundary:
            #     self.rho[i, j] = 1.0

    def init_patterns(self):
        """
        Initialize the dye field with the built-in checkerboard pattern and
        seed the CMM source + maps so scheme 8 can be selected immediately.
        """
        self._init_patterns_kernel()
        self._init_cmm_state_from_rho()


    def init_from_image(self, image_path: str):
        """
        Initializes the dye configuration (rho) from an external image file.
        The image is converted to grayscale (black and white), resized to the
        simulation resolution, and mapped into the density field.
        """
        from PIL import Image

        # Load image, convert to grayscale
        img = Image.open(image_path).convert('L')
        # Resize to match the simulation resolution
        img = img.resize((self.res, self.res))

        # Convert to a NumPy array, normalize the pixel values to [0.0, 1.0],
        # flip vertically (since Taichi typically uses a bottom-left origin),
        # and transpose to align with Taichi's (x, y) memory layout
        dye_np = np.array(img, dtype=np.float32) / 255.0
        dye_np = np.flipud(dye_np)
        dye_np = dye_np.T

        # Clear fields
        self.rho.fill(0)
        self.vel.fill(0)
        self.p.fill(0)

        # Transfer NumPy data to the Taichi field
        self.rho.from_numpy(dye_np)
        if self.bc_absorbing:
            self.apply_absorbing_rho_bc()
        self._init_cmm_state_from_rho()

    @ti.kernel
    def _fill_dye_kernel(self, x: float, y: float, radius: float, amount: float):
        """Kernel body for `fill_dye`. Adds dye in a circular region."""
        for i, j in self.rho:
            dist_x = abs(i * self.dx - x)
            dist_y = abs(j * self.dx - y)
            if dist_x > 0.5: dist_x = 1.0 - dist_x
            if dist_y > 0.5: dist_y = 1.0 - dist_y
            if ti.static(self.bc_wall):
                dist_x = i * self.dx - x
                dist_y = j * self.dx - y
            dist = ti.sqrt(dist_x * dist_x + dist_y * dist_y)
            if dist < radius:
                self.rho[i, j] += amount

    def fill_dye(self, x: float, y: float, radius: float, amount: float):
        """
        Adds dye in a circular region around (x, y). In Eulerian modes
        (schemes 4, 9, 10) this writes to the grid rho field directly. In
        CMM mode (scheme 8) it projects each affected current-frame cell
        through the backward map and deposits the dye into rho_source at
        the back-traced location, so the new dye participates in the
        existing map without forcing a remap.
        """
        if self.advection_scheme == 8:
            self._fill_dye_cmm_kernel(x, y, radius, amount)
        else:
            self._fill_dye_kernel(x, y, radius, amount)

    @ti.kernel
    def apply_force(self, x: float, y: float, f_x: float, f_y: float, radius: float):
        """
        Applies an external force field 'f' to the fluid within a circular region.
        Updates the momentum equation with force integration:
        u(t + Δt) = u(t) + f * Δt
        """
        for i, j in self.vel:
            dist_x = abs(i * self.dx - x)
            dist_y = abs(j * self.dx - y)
            if dist_x > 0.5: dist_x = 1.0 - dist_x
            if dist_y > 0.5: dist_y = 1.0 - dist_y
            if ti.static(self.bc_wall):
                dist_x = i * self.dx - x
                dist_y = j * self.dx - y
            dist = ti.sqrt(dist_x * dist_x + dist_y * dist_y)
            if dist < radius:
                self.vel[i, j] += ti.Vector([f_x, f_y]) * self.dt

    @ti.kernel
    def apply_bottom_force(self, f_x: float, f_y: float):
        """
        Applies a horizontal force to the bottom half of the fluid.
        """
        for i, j in self.vel:
            if j < self.res / 2:
                self.vel[i, j] += ti.Vector([f_x, f_y]) * self.dt

    @ti.kernel
    def _apply_persistent_force_kernel(self, b_coeff: float, t_coeff: float, r_coeff: float):
        """
        Applies persistent forces proportional to the dye density.
        Values are passed as arguments to ensure reactivity in Taichi.
        Forces are additive.
        """
        for i, j in self.vel:
            force = ti.Vector([0.0, 0.0])
            rho = self.rho[i, j]
            r = ti.Vector([i * self.dx - 0.5, j * self.dx - 0.5])
            dist = r.norm()

            # Buoyancy: upward force proportional to dye density
            force += ti.Vector([0.0, b_coeff * rho])

            # Torque: counter-clockwise tangential force
            if dist > 1e-6:
                force_dir = ti.Vector([-r.y, r.x]) / dist
                force += force_dir * t_coeff * (rho - 0.5)

            # Radial: outward force from center
            if dist > 1e-6:
                force += (r / (dist + 0.1)) * r_coeff * rho

            self.vel[i, j] += force * self.dt


    def apply_image_gradient_torque(self, image_path: str, scale: float = 1.0, duration: float = 0.1, blur_sigma: float = 0.0):
        """
        Reads an image and sets up a force to be applied to the fluid equal to
        the gradient of the image, spread over a certain duration.
        The image can be blurred to reduce noise in the gradient calculation.
        """
        from PIL import Image, ImageFilter
        import numpy as np

        img = Image.open(image_path).convert('L')
        img = img.resize((self.res, self.res))

        if blur_sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.flipud(img_np)
        img_np = img_np.T

        # Apply a vignette (edge fade) to avoid boundary artifacts with periodic wrapping
        # We fade the outer 5% of the image to zero
        edge_width = 0.05
        x = np.linspace(0, 1, self.res)
        y = np.linspace(0, 1, self.res)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        mask = np.ones((self.res, self.res), dtype=np.float32)

        def get_mask(coord):
            m = np.ones_like(coord)
            m = np.where(coord < edge_width, 0.5 - 0.5 * np.cos(np.pi * coord / edge_width), m)
            m = np.where(coord > 1.0 - edge_width, 0.5 - 0.5 * np.cos(np.pi * (1.0 - coord) / edge_width), m)
            return m

        mask *= get_mask(xv)
        mask *= get_mask(yv)

        img_np *= mask

        # Use p_temp as a temporary field to hold the image
        self.p_temp.from_numpy(img_np)
        self._precompute_gradient_perp(self.p_temp)
        self.force_scale = scale
        self.force_duration = duration
        self.dye_force_active = False

    def apply_dye_gradient_torque(self, scale: float = 1.0, duration: float = 0.1):
        """
        Sets up a force to be applied to the fluid proportional to the gradient
        of the current dye concentration (rho). This force is dynamic and
        recalculated at each step as the dye field evolves.
        """
        self.force_scale = scale
        self.force_duration = duration
        self.dye_force_active = True

    @ti.kernel
    def _precompute_gradient_perp(self, x: ti.template()):
        for i, j in self.image_grad:
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            if ti.static(self.bc_wall):
                im1 = ti.math.clamp(i - 1, 0, self.res - 1)
                ip1 = ti.math.clamp(i + 1, 0, self.res - 1)
                jm1 = ti.math.clamp(j - 1, 0, self.res - 1)
                jp1 = ti.math.clamp(j + 1, 0, self.res - 1)

            grad_x = (x[ip1, j] - x[im1, j]) * 0.5 / self.dx
            grad_y = (x[i, jp1] - x[i, jm1]) * 0.5 / self.dx

            self.image_grad[i, j] = ti.Vector([grad_y, -grad_x])

    @ti.kernel
    def _apply_stored_force(self):
        for i, j in self.vel:
            self.vel[i, j] += self.image_grad[i, j] * self.force_scale * self.dt

    @ti.func
    def sample(self, q, u, v):
        """
        Samples a field `q` at a fractional coordinate (u, v) using bilinear interpolation.
        """
        i, j = int(ti.floor(u)), int(ti.floor(v))
        f, g = u - i, v - j

        # Default: periodic wrap. Wall: clamp-to-edge (zero-flux Neumann).
        i0, j0 = i % self.res, j % self.res
        i1, j1 = (i + 1) % self.res, (j + 1) % self.res
        if ti.static(self.bc_wall):
            i0 = ti.math.clamp(i,     0, self.res - 1)
            i1 = ti.math.clamp(i + 1, 0, self.res - 1)
            j0 = ti.math.clamp(j,     0, self.res - 1)
            j1 = ti.math.clamp(j + 1, 0, self.res - 1)

        return (1 - f) * (1 - g) * q[i0, j0] + \
               f * (1 - g) * q[i1, j0] + \
               (1 - f) * g * q[i0, j1] + \
               f * g * q[i1, j1]

    @ti.kernel
    def _reduce_max_vel_norm(self):
        """
        Reduction kernel that fills self._max_vel_norm[None] with the maximum
        |vel| value across the grid. Used by max_cfl() as a building block.
        """
        self._max_vel_norm[None] = 0.0
        for i, j in self.vel:
            ti.atomic_max(self._max_vel_norm[None], self.vel[i, j].norm())

    def max_cfl(self) -> float:
        """
        Returns the current maximum CFL number across the grid:
            CFL = max(|u|) * dt / dx
        Values above ~1.0 indicate that per-step advection trajectories cross
        more than one cell. Useful as a diagnostic for whether observed
        artifacts are CFL-induced.
        """
        self._reduce_max_vel_norm()
        return float(self._max_vel_norm[None]) * self.dt / self.dx

    @ti.kernel
    def _init_cmm_state_from_rho(self):
        """
        Seeds the Bidirectional CMM state from the current rho field:
          - rho_source <- rho
          - backward_map[i, j] = (0, 0)         (zero deformation)
          - forward_map[i, j] = cell_center(i, j)  (identity)
        backward_map stores the deformation delta(x) = X(x) - x rather
        than the absolute back-traced coordinate; this makes the field
        continuous across periodic boundaries (no seam-discontinuity in
        the delta field under WENO5 advection).
        """
        for i, j in self.rho:
            self.rho_source[i, j] = self.rho[i, j]
            self.backward_map[i, j] = ti.Vector([0.0, 0.0])
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            self.forward_map[i, j] = ti.Vector([cx, cy])

    @ti.kernel
    def _init_cmm_maps_to_identity(self):
        """
        Resets the bidirectional CMM maps after a remap. backward_map
        (the deformation) goes to zero; forward_map (the absolute
        Lagrangian position) goes to cell center. rho_source is left
        untouched; the caller has already updated it.
        """
        for i, j in self.backward_map:
            self.backward_map[i, j] = ti.Vector([0.0, 0.0])
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            self.forward_map[i, j] = ti.Vector([cx, cy])

    @ti.kernel
    def _finalize_backward_map_step(self):
        """
        Post-advection step for the backward map: copies the WENO5+RK3
        output into backward_map AND subtracts the per-step source term u*dt.
        This source term comes from rewriting the standard backward-map update
            X^{n+1}(x) = X^n(x - u dt)
        in terms of delta = X - x:
            delta^{n+1}(x) = delta^n(x - u dt) - u(x) dt
        where the first part is plain advection of delta (handled by step_weno)
        and the second part is the per-cell source term applied here.
        """
        for i, j in self.backward_map:
            self.backward_map[i, j] = (self.new_backward_map[i, j]
                                        - self.dt * self.vel[i, j])

    @ti.kernel
    def advect_forward_map_rk2(self):
        """
        Advance the forward map Y one step via RK2 midpoint integration.
        Y[i, j] tracks the current position of the particle that was at
        cell (i, j) at the most recent remap.

        Under periodic BCs we do NOT wrap Y back into [0, 1) -- letting
        the absolute value drift over time makes the consistency check
        cleaner, because the backward delta is sampled at Y modulo 1
        anyway (sample() wraps the index), and the rho_source sampling
        composes naturally without modular arithmetic. Under wall BCs we
        clamp Y to the domain like everything else.
        """
        for i, j in self.forward_map:
            y = self.forward_map[i, j]
            u0 = self.sample(self.vel, y.x / self.dx - 0.5, y.y / self.dx - 0.5)
            y_mid = y + 0.5 * self.dt * u0
            u_mid = self.sample(self.vel,
                                y_mid.x / self.dx - 0.5,
                                y_mid.y / self.dx - 0.5)
            y_new = y + self.dt * u_mid
            if ti.static(self.bc_wall):
                y_new.x = ti.math.clamp(y_new.x, 0.0, 1.0)
                y_new.y = ti.math.clamp(y_new.y, 0.0, 1.0)
            # Periodic case: do not wrap; let absolute position accumulate.
            self.forward_map[i, j] = y_new

    @ti.kernel
    def _render_dye_bilinear(self):
        """Render variant: bilinear sampling of rho_source at X(x). Cheap."""
        for i, j in self.rho:
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            src = ti.Vector([cx, cy]) + self.backward_map[i, j]
            self.rho[i, j] = self.sample(self.rho_source,
                                          src.x / self.dx - 0.5,
                                          src.y / self.dx - 0.5)

    def render_dye_from_backward_map(self):
        """
        Reconstructs rho for the current frame by sampling rho_source at
        X(x) = x + delta(x) via bilinear interpolation. One interpolation
        per frame regardless of how many advection steps have passed since
        the last remap, so the rendered rho carries only single-interpolation
        error.
        """
        self._render_dye_bilinear()

    @ti.kernel
    def _compute_map_distortion(self):
        """
        Reduction kernel: max over all cells of ||X(Y(x)) - x||, the
        bidirectional-map self-consistency error. With X = id + delta:
            X(Y(x)) - x = (Y(x) + delta(Y(x))) - x
        where delta(Y) is bilinearly sampled from backward_map at Y mod 1
        (via sample()). When the maps are perfect inverses, this is zero.
        Drift past the configured threshold triggers a remap.
        """
        self._map_distortion[None] = 0.0
        for i, j in self.forward_map:
            x_grid = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx])
            y_fwd = self.forward_map[i, j]
            delta_at_yfwd = self.sample(self.backward_map,
                                         y_fwd.x / self.dx - 0.5,
                                         y_fwd.y / self.dx - 0.5)
            x_recovered = y_fwd + delta_at_yfwd
            err = (x_recovered - x_grid).norm()
            ti.atomic_max(self._map_distortion[None], err)

    @ti.kernel
    def _fill_dye_cmm_kernel(self, x: float, y: float,
                             radius: float, amount: float):
        """
        CMM-mode dye injection. The click location is in current (post-
        deformation) world space, but rho_source lives in pre-deformation
        space. For each cell within radius of the click, project its
        centre through the backward map to get the corresponding source-
        space location, then bilinearly deposit `amount` into rho_source
        at that location. After this kernel returns, normal CMM rendering
        produces dye at the click location -- and the new dye is then
        carried by the same backward map as everything else.
        """
        for i, j in self.rho:
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            dist_x = ti.abs(cx - x)
            dist_y = ti.abs(cy - y)
            if ti.static(not self.bc_wall):
                if dist_x > 0.5: dist_x = 1.0 - dist_x
                if dist_y > 0.5: dist_y = 1.0 - dist_y
            dist = ti.sqrt(dist_x * dist_x + dist_y * dist_y)
            if dist < radius:
                src = ti.Vector([cx, cy]) + self.backward_map[i, j]
                u = src.x / self.dx - 0.5
                v = src.y / self.dx - 0.5
                i_b = int(ti.floor(u))
                j_b = int(ti.floor(v))
                fx = u - i_b
                fy = v - j_b

                i0 = i_b % self.res
                i1 = (i_b + 1) % self.res
                j0 = j_b % self.res
                j1 = (j_b + 1) % self.res
                if ti.static(self.bc_wall):
                    i0 = ti.math.clamp(i_b,     0, self.res - 1)
                    i1 = ti.math.clamp(i_b + 1, 0, self.res - 1)
                    j0 = ti.math.clamp(j_b,     0, self.res - 1)
                    j1 = ti.math.clamp(j_b + 1, 0, self.res - 1)

                ti.atomic_add(self.rho_source[i0, j0], (1.0 - fx) * (1.0 - fy) * amount)
                ti.atomic_add(self.rho_source[i1, j0], fx         * (1.0 - fy) * amount)
                ti.atomic_add(self.rho_source[i0, j1], (1.0 - fx) * fy         * amount)
                ti.atomic_add(self.rho_source[i1, j1], fx         * fy         * amount)

    def cmm_remap(self):
        """
        Bake the current rho into rho_source and reset both maps to the
        identity. Called from `step()` when the bidirectional consistency
        check indicates the maps have drifted past the configured
        threshold. Visually this is a one-frame "snapshot" of the current
        dye state at grid resolution.
        """
        self.rho_source.copy_from(self.rho)
        self._init_cmm_maps_to_identity()


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
        solver. The effective per-step amplification at the Nyquist
        wavelength is ~(1 + 4 * strength * dt / dx^2), so usable strengths
        depend on resolution and timestep. At res=128 with dt=3e-4, useful
        values are around 1e-5 to 1e-4; values above ~1e-3 blow up over a
        few hundred steps. Leave at zero when sharpening isn't wanted (see
        SimulationConfig).

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


    @ti.func
    def weno5_reconstruct(self, v1, v2, v3, v4, v5):
        """WENO5 5th-order reconstruction (Jiang & Shu 1996).

        Computes the weighted combination of three candidate stencil polynomials
        (p0, p1, p2) using smoothness indicators beta_k. Weights alpha_k are
        inversely proportional to (eps + beta_k)^2, so smooth stencils receive
        higher weight. In smooth flow the result approaches the optimal 5th-order
        upwind reconstruction; near discontinuities weight falls on smoother
        stencils, limiting oscillation.
        """
        eps = 1e-6
        p0 = (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0
        p1 = (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0
        p2 = (2.0 * v3 + 5.0 * v4 - v5) / 6.0
        beta0 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3)**2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3)**2
        beta1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4)**2 + 0.25 * (v2 - v4)**2
        beta2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5)**2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5)**2
        alpha0 = 0.1 / (eps + beta0)**2
        alpha1 = 0.6 / (eps + beta1)**2
        alpha2 = 0.3 / (eps + beta2)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / sum_alpha

    @ti.func
    def wenoz_reconstruct(self, v1, v2, v3, v4, v5):
        """WENO-Z 5th-order reconstruction (Borges et al. 2008).

        Replaces the standard alpha_k = d_k / (eps + beta_k)^2 weights with
        alpha_k = d_k * (1 + tau5 / (beta_k + eps))^2 where tau5 = |beta0 - beta2|
        is a higher-order global smoothness indicator. In smooth flow tau5 = O(h^4)
        while beta_k = O(h^2), so the ratio -> 0 and the weights collapse to the
        ideal linear values d_k, recovering full 5th-order accuracy with less
        numerical diffusion than standard WENO5.
        """
        eps = 1e-6
        p0 = (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0
        p1 = (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0
        p2 = (2.0 * v3 + 5.0 * v4 - v5) / 6.0
        beta0 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3)**2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3)**2
        beta1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4)**2 + 0.25 * (v2 - v4)**2
        beta2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5)**2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5)**2
        tau5 = ti.abs(beta0 - beta2)
        alpha0 = 0.1 * (1.0 + (tau5 / (beta0 + eps))**2)
        alpha1 = 0.6 * (1.0 + (tau5 / (beta1 + eps))**2)
        alpha2 = 0.3 * (1.0 + (tau5 / (beta2 + eps))**2)
        sum_alpha = alpha0 + alpha1 + alpha2
        return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / sum_alpha

    @ti.func
    def teno5_reconstruct(self, v1, v2, v3, v4, v5):
        """TENO5 reconstruction (Fu, Hu, Adams 2016).

        Uses the global smoothness indicator tau5 = |beta0 - beta2| to classify
        each candidate stencil as smooth or non-smooth via binary thresholding.
        Smooth stencils get their exact ideal linear weights (d_k), non-smooth
        stencils are zeroed out. In smooth flow all three stencils pass the
        threshold and the result is exactly the 5th-order upwind reconstruction --
        sharper than both WENO5 and WENO-Z since no nonlinear weight distortion
        remains.
        """
        eps = 1e-6
        C_T = 1e-5
        p0 = (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0
        p1 = (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0
        p2 = (2.0 * v3 + 5.0 * v4 - v5) / 6.0
        beta0 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3)**2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3)**2
        beta1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4)**2 + 0.25 * (v2 - v4)**2
        beta2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5)**2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5)**2
        tau5 = ti.abs(beta0 - beta2)
        # gamma_k grows large for stencils far from a discontinuity (smooth stencils)
        # and stays near 1 for stencils spanning a discontinuity (bad stencils).
        # Clamp the ratio before the ^6 to prevent float32 overflow when tau5 >> beta_k.
        gamma0 = (1.0 + ti.math.min(tau5 / (beta0 + eps), 1e5)) ** 6
        gamma1 = (1.0 + ti.math.min(tau5 / (beta1 + eps), 1e5)) ** 6
        gamma2 = (1.0 + ti.math.min(tau5 / (beta2 + eps), 1e5)) ** 6
        sum_gamma = gamma0 + gamma1 + gamma2
        chi0 = gamma0 / sum_gamma
        chi1 = gamma1 / sum_gamma
        chi2 = gamma2 / sum_gamma
        # step function: 1.0 if chi > C_T, else 0.0.
        # sign+max avoids if/else, which fails for vector (vec2 velocity) fields.
        delta0 = ti.math.max(ti.math.sign(chi0 - C_T), 0.0)
        delta1 = ti.math.max(ti.math.sign(chi1 - C_T), 0.0)
        delta2 = ti.math.max(ti.math.sign(chi2 - C_T), 0.0)
        w0 = 0.1 * delta0
        w1 = 0.6 * delta1
        w2 = 0.3 * delta2
        # chi0+chi1+chi2 = 1 guarantees max(chi_k) >= 1/3 >> C_T, so sum_w > 0 always.
        return (w0 * p0 + w1 * p1 + w2 * p2) / (w0 + w1 + w2)

    @ti.kernel
    def advect_weno_rhs(self, field: ti.template(), dq: ti.template()):
        """Computes WENO5 flux divergence RHS; paired with the SSP-RK3 kernels
        to advance a field by one full time step via step_weno / _step_rk3."""
        for i, j in field:
            u = self.vel[i, j]

            flux_x = field[i, j] * 0.0
            flux_y = field[i, j] * 0.0

            # Neighbor indices: default periodic, overridden to clamp for wall BC
            im3 = (i - 3) % self.res
            im2 = (i - 2) % self.res
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            ip2 = (i + 2) % self.res
            ip3 = (i + 3) % self.res
            jm3 = (j - 3) % self.res
            jm2 = (j - 2) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            jp2 = (j + 2) % self.res
            jp3 = (j + 3) % self.res
            if ti.static(self.bc_wall):
                im3 = ti.math.clamp(i - 3, 0, self.res - 1)
                im2 = ti.math.clamp(i - 2, 0, self.res - 1)
                im1 = ti.math.clamp(i - 1, 0, self.res - 1)
                ip1 = ti.math.clamp(i + 1, 0, self.res - 1)
                ip2 = ti.math.clamp(i + 2, 0, self.res - 1)
                ip3 = ti.math.clamp(i + 3, 0, self.res - 1)
                jm3 = ti.math.clamp(j - 3, 0, self.res - 1)
                jm2 = ti.math.clamp(j - 2, 0, self.res - 1)
                jm1 = ti.math.clamp(j - 1, 0, self.res - 1)
                jp1 = ti.math.clamp(j + 1, 0, self.res - 1)
                jp2 = ti.math.clamp(j + 2, 0, self.res - 1)
                jp3 = ti.math.clamp(j + 3, 0, self.res - 1)

            # x flux
            if u.x > 0:
                q_R = self.weno5_reconstruct(field[im2, j], field[im1, j], field[i, j], field[ip1, j], field[ip2, j])
                q_L = self.weno5_reconstruct(field[im3, j], field[im2, j], field[im1, j], field[i, j], field[ip1, j])
                flux_x = u.x * (q_R - q_L)
            else:
                q_R = self.weno5_reconstruct(field[ip3, j], field[ip2, j], field[ip1, j], field[i, j], field[im1, j])
                q_L = self.weno5_reconstruct(field[ip2, j], field[ip1, j], field[i, j], field[im1, j], field[im2, j])
                flux_x = u.x * (q_R - q_L)

            # y flux
            if u.y > 0:
                q_T = self.weno5_reconstruct(field[i, jm2], field[i, jm1], field[i, j], field[i, jp1], field[i, jp2])
                q_B = self.weno5_reconstruct(field[i, jm3], field[i, jm2], field[i, jm1], field[i, j], field[i, jp1])
                flux_y = u.y * (q_T - q_B)
            else:
                q_T = self.weno5_reconstruct(field[i, jp3], field[i, jp2], field[i, jp1], field[i, j], field[i, jm1])
                q_B = self.weno5_reconstruct(field[i, jp2], field[i, jp1], field[i, j], field[i, jm1], field[i, jm2])
                flux_y = u.y * (q_T - q_B)

            dq[i, j] = -(flux_x + flux_y) / self.dx

    @ti.kernel
    def rk3_step1(self, field: ti.template(), field_1: ti.template(), dq: ti.template()):
        for i, j in field:
            field_1[i, j] = field[i, j] + self.dt * dq[i, j]

    @ti.kernel
    def rk3_step2(self, field: ti.template(), field_1: ti.template(), field_2: ti.template(), dq: ti.template()):
        for i, j in field:
            field_2[i, j] = 0.75 * field[i, j] + 0.25 * field_1[i, j] + 0.25 * self.dt * dq[i, j]

    @ti.kernel
    def rk3_step3(self, field: ti.template(), field_2: ti.template(), new_field: ti.template(), dq: ti.template()):
        for i, j in field:
            new_field[i, j] = (1.0 / 3.0) * field[i, j] + (2.0 / 3.0) * field_2[i, j] + (2.0 / 3.0) * self.dt * dq[i, j]

    def _step_rk3(self, rhs_fn, field, field_1, field_2, new_field, dq):
        """SSP-RK3 time integration with an arbitrary WENO-variant RHS kernel."""
        rhs_fn(field, dq)
        self.rk3_step1(field, field_1, dq)
        rhs_fn(field_1, dq)
        self.rk3_step2(field, field_1, field_2, dq)
        rhs_fn(field_2, dq)
        self.rk3_step3(field, field_2, new_field, dq)

    def _step_eulerian(self, rhs_fn):
        """Advect both rho and vel using one WENO-variant RHS function."""
        self._step_rk3(rhs_fn, self.rho, self.rho_1, self.rho_2, self.new_rho, self.dq_rho)
        self.rho.copy_from(self.new_rho)
        self._step_rk3(rhs_fn, self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
        self.vel.copy_from(self.new_vel)

    def step_weno(self, field, field_1, field_2, new_field, dq):
        """SSP-RK3 using WENO5 reconstruction."""
        self._step_rk3(self.advect_weno_rhs, field, field_1, field_2, new_field, dq)

    @ti.kernel
    def advect_wenoz_rhs(self, field: ti.template(), dq: ti.template()):
        """Computes WENO-Z flux divergence RHS; identical structure to
        advect_weno_rhs but calls wenoz_reconstruct for the face values."""
        for i, j in field:
            u = self.vel[i, j]
            flux_x = field[i, j] * 0.0
            flux_y = field[i, j] * 0.0
            im3 = (i - 3) % self.res
            im2 = (i - 2) % self.res
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            ip2 = (i + 2) % self.res
            ip3 = (i + 3) % self.res
            jm3 = (j - 3) % self.res
            jm2 = (j - 2) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            jp2 = (j + 2) % self.res
            jp3 = (j + 3) % self.res
            if ti.static(self.bc_wall):
                im3 = ti.math.clamp(i - 3, 0, self.res - 1)
                im2 = ti.math.clamp(i - 2, 0, self.res - 1)
                im1 = ti.math.clamp(i - 1, 0, self.res - 1)
                ip1 = ti.math.clamp(i + 1, 0, self.res - 1)
                ip2 = ti.math.clamp(i + 2, 0, self.res - 1)
                ip3 = ti.math.clamp(i + 3, 0, self.res - 1)
                jm3 = ti.math.clamp(j - 3, 0, self.res - 1)
                jm2 = ti.math.clamp(j - 2, 0, self.res - 1)
                jm1 = ti.math.clamp(j - 1, 0, self.res - 1)
                jp1 = ti.math.clamp(j + 1, 0, self.res - 1)
                jp2 = ti.math.clamp(j + 2, 0, self.res - 1)
                jp3 = ti.math.clamp(j + 3, 0, self.res - 1)
            if u.x > 0:
                q_R = self.wenoz_reconstruct(field[im2, j], field[im1, j], field[i, j], field[ip1, j], field[ip2, j])
                q_L = self.wenoz_reconstruct(field[im3, j], field[im2, j], field[im1, j], field[i, j], field[ip1, j])
                flux_x = u.x * (q_R - q_L)
            else:
                q_R = self.wenoz_reconstruct(field[ip3, j], field[ip2, j], field[ip1, j], field[i, j], field[im1, j])
                q_L = self.wenoz_reconstruct(field[ip2, j], field[ip1, j], field[i, j], field[im1, j], field[im2, j])
                flux_x = u.x * (q_R - q_L)
            if u.y > 0:
                q_T = self.wenoz_reconstruct(field[i, jm2], field[i, jm1], field[i, j], field[i, jp1], field[i, jp2])
                q_B = self.wenoz_reconstruct(field[i, jm3], field[i, jm2], field[i, jm1], field[i, j], field[i, jp1])
                flux_y = u.y * (q_T - q_B)
            else:
                q_T = self.wenoz_reconstruct(field[i, jp3], field[i, jp2], field[i, jp1], field[i, j], field[i, jm1])
                q_B = self.wenoz_reconstruct(field[i, jp2], field[i, jp1], field[i, j], field[i, jm1], field[i, jm2])
                flux_y = u.y * (q_T - q_B)
            dq[i, j] = -(flux_x + flux_y) / self.dx

    def step_wenoz(self, field, field_1, field_2, new_field, dq):
        """SSP-RK3 using WENO-Z reconstruction."""
        self._step_rk3(self.advect_wenoz_rhs, field, field_1, field_2, new_field, dq)

    @ti.kernel
    def advect_teno5_rhs(self, field: ti.template(), dq: ti.template()):
        """Computes TENO5 flux divergence RHS; identical structure to
        advect_weno_rhs but calls teno5_reconstruct for the face values."""
        for i, j in field:
            u = self.vel[i, j]
            flux_x = field[i, j] * 0.0
            flux_y = field[i, j] * 0.0
            im3 = (i - 3) % self.res
            im2 = (i - 2) % self.res
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            ip2 = (i + 2) % self.res
            ip3 = (i + 3) % self.res
            jm3 = (j - 3) % self.res
            jm2 = (j - 2) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            jp2 = (j + 2) % self.res
            jp3 = (j + 3) % self.res
            if ti.static(self.bc_wall):
                im3 = ti.math.clamp(i - 3, 0, self.res - 1)
                im2 = ti.math.clamp(i - 2, 0, self.res - 1)
                im1 = ti.math.clamp(i - 1, 0, self.res - 1)
                ip1 = ti.math.clamp(i + 1, 0, self.res - 1)
                ip2 = ti.math.clamp(i + 2, 0, self.res - 1)
                ip3 = ti.math.clamp(i + 3, 0, self.res - 1)
                jm3 = ti.math.clamp(j - 3, 0, self.res - 1)
                jm2 = ti.math.clamp(j - 2, 0, self.res - 1)
                jm1 = ti.math.clamp(j - 1, 0, self.res - 1)
                jp1 = ti.math.clamp(j + 1, 0, self.res - 1)
                jp2 = ti.math.clamp(j + 2, 0, self.res - 1)
                jp3 = ti.math.clamp(j + 3, 0, self.res - 1)
            if u.x > 0:
                q_R = self.teno5_reconstruct(field[im2, j], field[im1, j], field[i, j], field[ip1, j], field[ip2, j])
                q_L = self.teno5_reconstruct(field[im3, j], field[im2, j], field[im1, j], field[i, j], field[ip1, j])
                flux_x = u.x * (q_R - q_L)
            else:
                q_R = self.teno5_reconstruct(field[ip3, j], field[ip2, j], field[ip1, j], field[i, j], field[im1, j])
                q_L = self.teno5_reconstruct(field[ip2, j], field[ip1, j], field[i, j], field[im1, j], field[im2, j])
                flux_x = u.x * (q_R - q_L)
            if u.y > 0:
                q_T = self.teno5_reconstruct(field[i, jm2], field[i, jm1], field[i, j], field[i, jp1], field[i, jp2])
                q_B = self.teno5_reconstruct(field[i, jm3], field[i, jm2], field[i, jm1], field[i, j], field[i, jp1])
                flux_y = u.y * (q_T - q_B)
            else:
                q_T = self.teno5_reconstruct(field[i, jp3], field[i, jp2], field[i, jp1], field[i, j], field[i, jm1])
                q_B = self.teno5_reconstruct(field[i, jp2], field[i, jp1], field[i, j], field[i, jm1], field[i, jm2])
                flux_y = u.y * (q_T - q_B)
            dq[i, j] = -(flux_x + flux_y) / self.dx

    def step_teno5(self, field, field_1, field_2, new_field, dq):
        """SSP-RK3 using TENO5 reconstruction."""
        self._step_rk3(self.advect_teno5_rhs, field, field_1, field_2, new_field, dq)

    @ti.kernel
    def compute_divergence(self):
        """
        Computes the divergence of the intermediate velocity field.

        Mathematical detail:
        ∇ · u = ∂u/∂x + ∂v/∂y
        Approximated using central differences:
        (∇ · u)_{i,j} = (u_{i+1,j} - u_{i-1,j}) / (2*dx) + (v_{i,j+1} - v_{i,j-1}) / (2*dx)
        """
        for i, j in self.vel:
            vl = self.vel[(i - 1) % self.res, j].x
            vr = self.vel[(i + 1) % self.res, j].x
            vb = self.vel[i, (j - 1) % self.res].y
            vt = self.vel[i, (j + 1) % self.res].y
            if ti.static(self.bc_wall):
                vl = self.vel[ti.math.clamp(i - 1, 0, self.res - 1), j].x
                vr = self.vel[ti.math.clamp(i + 1, 0, self.res - 1), j].x
                vb = self.vel[i, ti.math.clamp(j - 1, 0, self.res - 1)].y
                vt = self.vel[i, ti.math.clamp(j + 1, 0, self.res - 1)].y
            self.div[i, j] = (vr - vl + vt - vb) * 0.5 / self.dx

    @ti.kernel
    def pressure_solve_jacobi(self, p: ti.template(), new_p: ti.template()):
        """
        Solves the Pressure Poisson Equation: ∇²p = (∇ · u*) / Δt
        (Implementation note: The division by Δt and density ρ is implicitly
        bundled in the computation, maintaining algorithmic simplicity)

        Uses the Jacobi iterative method to solve the linear system.

        Mathematical detail:
        Discretizing the Laplacian ∇²p with central differences gives:
        (p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}) / dx² = (∇ · u*)_{i,j}

        Rearranging for p_{i,j} gives the Jacobi iteration step:
        p_{i,j}^{(k+1)} = 0.25 * [ p_{i+1,j}^{(k)} + p_{i-1,j}^{(k)} + p_{i,j+1}^{(k)} + p_{i,j-1}^{(k)} - dx² * (∇ · u*)_{i,j} ]
        """
        for i, j in p:
            pl = p[(i - 1) % self.res, j]
            pr = p[(i + 1) % self.res, j]
            pb = p[i, (j - 1) % self.res]
            pt = p[i, (j + 1) % self.res]
            if ti.static(self.bc_wall):
                # Neumann BC: ghost cell = self (∂p/∂n = 0)
                pl = p[ti.math.clamp(i - 1, 0, self.res - 1), j]
                pr = p[ti.math.clamp(i + 1, 0, self.res - 1), j]
                pb = p[i, ti.math.clamp(j - 1, 0, self.res - 1)]
                pt = p[i, ti.math.clamp(j + 1, 0, self.res - 1)]
            new_p[i, j] = (pl + pr + pb + pt - self.div[i, j] * self.dx * self.dx) * 0.25

    @ti.kernel
    def _mg_smooth_rb_kernel(self,
                              p: ti.template(),
                              f: ti.template(),
                              color: int):
        """
        One Red-Black Gauss-Seidel smoothing pass over cells of the given
        color: 0 = even cells where (i+j)%2==0, 1 = odd cells.

        Solves one in-place sweep of L*p = f:
            p[i,j] <- (pl + pr + pb + pt - f[i,j] * dx_l²) / 4
        where dx_l = 1.0 / p.shape[0] is the grid spacing at this level.

        RBGS is safe for parallel execution: all writes target cells of
        `color`, all reads come from the opposite color (5-point stencil
        neighbors of an even cell are always odd, and vice versa).

        Boundary conditions controlled by ti.static(self.bc_wall):
          - periodic (False): modular wrap using (x + n) % n
          - Neumann/wall (True): clamp to [0, n-1], enforcing ∂p/∂n = 0
        """
        n = p.shape[0]
        dx_l = 1.0 / n
        dx2 = dx_l * dx_l
        for i, j in p:
            if (i + j) % 2 == color:
                im1 = (i - 1 + n) % n
                ip1 = (i + 1) % n
                jm1 = (j - 1 + n) % n
                jp1 = (j + 1) % n
                if ti.static(self.bc_wall):
                    im1 = ti.math.clamp(i - 1, 0, n - 1)
                    ip1 = ti.math.clamp(i + 1, 0, n - 1)
                    jm1 = ti.math.clamp(j - 1, 0, n - 1)
                    jp1 = ti.math.clamp(j + 1, 0, n - 1)
                pl = p[im1, j]
                pr = p[ip1, j]
                pb = p[i, jm1]
                pt = p[i, jp1]
                p[i, j] = (pl + pr + pb + pt - f[i, j] * dx2) * 0.25

    @ti.kernel
    def _mg_restrict_residual_kernel(self,
                                      fine_p: ti.template(),
                                      fine_f: ti.template(),
                                      coarse_f: ti.template()):
        """
        Computes the residual r = f - L*p on the fine grid, then restricts
        it to the coarse grid using 9-point full-weighting:
            coarse_f[I,J] = Σ_{di,dj∈{-1,0,1}} w(di,dj)/16 · r[2I+di, 2J+dj]
        where w(di,dj) = (2-|di|)*(2-|dj|) gives weights 4,2,2,1 for
        center, face, edge neighbors respectively (sum = 16).

        fine_p and fine_f have shape (n, n); coarse_f has shape (n//2, n//2).
        The discrete Laplacian uses dx_l = 1.0 / n (fine grid spacing).

        Boundary conditions are handled the same way as _mg_smooth_rb_kernel.
        """
        n  = fine_p.shape[0]
        dx_l = 1.0 / n
        dx2  = dx_l * dx_l
        for I, J in coarse_f:
            r_sum = 0.0
            for di, dj in ti.static([(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]):
                fi = 2 * I + di
                fj = 2 * J + dj
                # Wrap fine index: fi may be -1 when I=0, di=-1
                fi_w = (fi + n) % n
                fj_w = (fj + n) % n
                if ti.static(self.bc_wall):
                    fi_w = ti.math.clamp(fi, 0, n - 1)
                    fj_w = ti.math.clamp(fj, 0, n - 1)
                # Neighbors of the fine cell for its Laplacian
                fi_m = (fi_w - 1 + n) % n
                fi_p = (fi_w + 1) % n
                fj_m = (fj_w - 1 + n) % n
                fj_p = (fj_w + 1) % n
                if ti.static(self.bc_wall):
                    fi_m = ti.math.clamp(fi_w - 1, 0, n - 1)
                    fi_p = ti.math.clamp(fi_w + 1, 0, n - 1)
                    fj_m = ti.math.clamp(fj_w - 1, 0, n - 1)
                    fj_p = ti.math.clamp(fj_w + 1, 0, n - 1)
                lap_p = (fine_p[fi_m, fj_w] + fine_p[fi_p, fj_w]
                       + fine_p[fi_w, fj_m] + fine_p[fi_w, fj_p]
                       - 4.0 * fine_p[fi_w, fj_w]) / dx2
                r_ij = fine_f[fi_w, fj_w] - lap_p
                w = float((2 - abs(di)) * (2 - abs(dj)))
                r_sum += w * r_ij
            coarse_f[I, J] = r_sum / 16.0

    @ti.kernel
    def _mg_prolongate_add_kernel(self,
                                   coarse_p: ti.template(),
                                   fine_p:   ti.template()):
        """
        Bilinear prolongation: interpolates coarse_p onto the fine grid
        and ADDS the result to fine_p (fine_p += P * coarse_p).

        Each fine cell (i, j) maps to coarse cell I = i//2, J = j//2.
        The fractional offset into the coarse cell is:
            fx = 0.5 if i is odd (fine cell straddles coarse boundary)
            fx = 0.0 if i is even (fine cell aligns with coarse center)
        Note: this bilinear prolongation is not the exact adjoint of the
        9-point full-weighting restriction; non-adjoint R/P pairs are
        standard practice and do not compromise multigrid convergence.

        Boundary conditions: periodic wrap or clamp for I+1, J+1.
        """
        n_c = coarse_p.shape[0]
        for i, j in fine_p:
            I  = i // 2
            J  = j // 2
            fx = float(i % 2) * 0.5
            fy = float(j % 2) * 0.5
            I1 = (I + 1) % n_c
            J1 = (J + 1) % n_c
            if ti.static(self.bc_wall):
                I1 = ti.math.clamp(I + 1, 0, n_c - 1)
                J1 = ti.math.clamp(J + 1, 0, n_c - 1)
            val = ((1.0 - fx) * (1.0 - fy) * coarse_p[I,  J ]
                 + fx          * (1.0 - fy) * coarse_p[I1, J ]
                 + (1.0 - fx) * fy          * coarse_p[I,  J1]
                 + fx          * fy          * coarse_p[I1, J1])
            fine_p[i, j] += val

    def _solve_pressure_fft(self):
        """
        Solves the Pressure Poisson Equation ∇²p = ∇·u* exactly via the
        Discrete Fourier Transform (periodic BCs only).

        For an N×N periodic grid with cell size dx, the discrete Laplacian
        has known eigenvalues:
            λ_{k,l} = (2cos(2πk/N) - 2 + 2cos(2πl/N) - 2) / dx²

        The solution is:
            P̂_{k,l} = F̂_{k,l} / λ_{k,l}    for (k,l) ≠ (0,0)
            P̂_{0,0} = 0                       (zero-mean pressure gauge fix)
            p = IFFT(P̂)

        This gives the exact solution in O(N² log N) rather than the O(N²·iter)
        approximation of Jacobi iteration.
        """
        N = self.res
        dx = self.dx

        f = self.div.to_numpy()

        F_hat = np.fft.rfft2(f)  # shape (N, N//2+1), complex

        kx = np.arange(N, dtype=np.float64)
        ky = np.arange(N // 2 + 1, dtype=np.float64)
        lam_x = (2.0 * np.cos(2.0 * np.pi * kx / N) - 2.0) / (dx * dx)
        lam_y = (2.0 * np.cos(2.0 * np.pi * ky / N) - 2.0) / (dx * dx)
        lam = lam_x[:, None] + lam_y[None, :]  # (N, N//2+1)

        # Avoid division by zero at the DC mode; zero it out after solving.
        lam[0, 0] = 1.0
        P_hat = F_hat / lam
        P_hat[0, 0] = 0.0  # zero mean pressure

        p = np.fft.irfft2(P_hat, s=(N, N)).astype(np.float32)
        self.p.from_numpy(p)

    def _mg_v_cycle(self, level: int):
        """
        Executes one multigrid V-cycle starting at `level`.

        V-cycle outline:
          1. Pre-smooth:   2 RBGS sweeps on mg_p[level] given mg_f[level]
          2. Restrict:     compute r = mg_f[level] - L*mg_p[level], restrict
                           into mg_f[level+1]
          3. Zero coarse:  mg_p[level+1] = 0 (solving for the error correction)
          4. Recurse:      _mg_v_cycle(level+1)  — or exact-solve at coarsest
          5. Prolongate:   mg_p[level] += P * mg_p[level+1]
          6. Post-smooth:  2 more RBGS sweeps

        At the coarsest level (4×4), 50 RBGS passes give a near-exact solve.
        """
        p = self.mg_p[level]
        f = self.mg_f[level]

        # Pre-smooth: 2 Red-Black GS sweeps
        for _ in range(2):
            self._mg_smooth_rb_kernel(p, f, 0)
            self._mg_smooth_rb_kernel(p, f, 1)

        # At the coarsest level: solve with many iterations and return
        if level == self._mg_num_levels - 1:
            for _ in range(50):
                self._mg_smooth_rb_kernel(p, f, 0)
                self._mg_smooth_rb_kernel(p, f, 1)
            return

        # Restrict residual to next coarser level
        self._mg_restrict_residual_kernel(p, f, self.mg_f[level + 1])

        # Zero the coarse correction (solving L*e = R(r), start from e=0)
        self.mg_p[level + 1].fill(0.0)

        # Recursive V-cycle on the coarser level
        self._mg_v_cycle(level + 1)

        # Prolongate coarse correction and add to fine solution
        self._mg_prolongate_add_kernel(self.mg_p[level + 1], p)

        # Post-smooth: 2 more RBGS sweeps
        for _ in range(2):
            self._mg_smooth_rb_kernel(p, f, 0)
            self._mg_smooth_rb_kernel(p, f, 1)

    def _solve_pressure_multigrid(self):
        """
        Solves ∇²p = ∇·u* using geometric multigrid V-cycles.

        Steps:
          1. Warm-start from self.p (previous timestep's pressure). Temporal
             coherence means the initial residual is already small, so fewer
             V-cycles are needed to converge.
          2. Copy self.div into mg_f[0] as the RHS.
          3. Run config.mg_v_cycles V-cycles.
          4. Copy the result back into self.p for pressure_project().

        Supports periodic BCs (bc_wall=False) and Neumann/wall BCs
        (bc_wall=True). Does NOT support open/Dirichlet BCs — step() falls
        back to Jacobi in that case via the `use_multigrid` check.
        """
        self.mg_p[0].copy_from(self.p)
        self.mg_f[0].copy_from(self.div)
        for _ in range(self.config.mg_v_cycles):
            self._mg_v_cycle(0)
        self.p.copy_from(self.mg_p[0])

    @ti.kernel
    def _compute_vorticity(self):
        """
        Computes the scalar vorticity ω = ∂v/∂x - ∂u/∂y at every cell centre
        using second-order central differences.  Result is stored in self.vorticity.
        """
        for i, j in self.vorticity:
            vr = self.vel[(i + 1) % self.res, j].y
            vl = self.vel[(i - 1) % self.res, j].y
            ut = self.vel[i, (j + 1) % self.res].x
            ub = self.vel[i, (j - 1) % self.res].x
            if ti.static(self.bc_wall):
                vr = self.vel[ti.math.clamp(i + 1, 0, self.res - 1), j].y
                vl = self.vel[ti.math.clamp(i - 1, 0, self.res - 1), j].y
                ut = self.vel[i, ti.math.clamp(j + 1, 0, self.res - 1)].x
                ub = self.vel[i, ti.math.clamp(j - 1, 0, self.res - 1)].x
            self.vorticity[i, j] = (vr - vl - ut + ub) * 0.5 / self.dx

    @ti.kernel
    def _apply_vorticity_confinement(self, strength: float):
        """
        Applies vorticity confinement (Fedkiw et al. 2001) to counteract
        the numerical diffusion of rotational structures.

        The confinement force points tangentially around each vortex core:
            η  = ∇|ω| / (|∇|ω|| + ε)     (unit normal toward vortex core)
            F  = strength · |ω| · (−η_y, η_x)
            vel += F · dt

        Must be called after _compute_vorticity().  Strength (ε_conf) is
        typically in the range 0.1–5.0.
        """
        for i, j in self.vel:
            om_r = ti.abs(self.vorticity[(i + 1) % self.res, j])
            om_l = ti.abs(self.vorticity[(i - 1) % self.res, j])
            om_t = ti.abs(self.vorticity[i, (j + 1) % self.res])
            om_b = ti.abs(self.vorticity[i, (j - 1) % self.res])
            if ti.static(self.bc_wall):
                om_r = ti.abs(self.vorticity[ti.math.clamp(i + 1, 0, self.res - 1), j])
                om_l = ti.abs(self.vorticity[ti.math.clamp(i - 1, 0, self.res - 1), j])
                om_t = ti.abs(self.vorticity[i, ti.math.clamp(j + 1, 0, self.res - 1)])
                om_b = ti.abs(self.vorticity[i, ti.math.clamp(j - 1, 0, self.res - 1)])

            eta = ti.Vector([(om_r - om_l) * 0.5 / self.dx,
                             (om_t - om_b) * 0.5 / self.dx])
            eta_norm = eta.norm() + 1e-6
            eta = eta / eta_norm

            om = self.vorticity[i, j]
            force = strength * ti.abs(om) * ti.Vector([-eta.y, eta.x])
            self.vel[i, j] += force * self.dt

    @ti.kernel
    def pressure_project(self):
        """
        Projects the intermediate velocity field to make it divergence-free
        by subtracting the pressure gradient, satisfying the incompressibility constraint.

        Mathematical detail:
        u^{n+1} = u^* - ∇p
        Using central differences for the gradient ∇p:
        u_{i,j} -= (p_{i+1,j} - p_{i-1,j}) / (2*dx)
        v_{i,j} -= (p_{i,j+1} - p_{i,j-1}) / (2*dx)
        """
        for i, j in self.vel:
            pl = self.p[(i - 1) % self.res, j]
            pr = self.p[(i + 1) % self.res, j]
            pb = self.p[i, (j - 1) % self.res]
            pt = self.p[i, (j + 1) % self.res]
            if ti.static(self.bc_wall):
                pl = self.p[ti.math.clamp(i - 1, 0, self.res - 1), j]
                pr = self.p[ti.math.clamp(i + 1, 0, self.res - 1), j]
                pb = self.p[i, ti.math.clamp(j - 1, 0, self.res - 1)]
                pt = self.p[i, ti.math.clamp(j + 1, 0, self.res - 1)]
            grad_p = ti.Vector([(pr - pl) * 0.5 / self.dx, (pt - pb) * 0.5 / self.dx])
            self.vel[i, j] -= grad_p

    @ti.kernel
    def apply_velocity_bc(self):
        """
        Enforces wall boundary conditions.
        Normal component is always zeroed. Tangential component is multiplied by
        wall_slip: 0.0 = no-slip (tangential zeroed), 1.0 = free-slip (tangential unchanged).
        Corners always get both components zeroed regardless of wall_slip.
        """
        for i, j in self.vel:
            if i == 0 or i == self.res - 1:
                self.vel[i, j].x = 0.0
                self.vel[i, j].y *= self.config.wall_slip
            if j == 0 or j == self.res - 1:
                self.vel[i, j].y = 0.0
                self.vel[i, j].x *= self.config.wall_slip

    @ti.kernel
    def apply_absorbing_rho_bc(self):
        """
        Absorbing boundary: zeroes the boundary rows/columns of the dye field.
        Dye that reaches the wall is removed rather than reflected or accumulated.
        """
        for i, j in self.rho:
            if i == 0 or i == self.res - 1 or j == 0 or j == self.res - 1:
                self.rho[i, j] = 0.0

    @ti.kernel
    def apply_open_pressure_bc(self):
        """
        Open boundary: Dirichlet p=0 at domain edges (ambient pressure outlet).
        Applied after each Jacobi step so interior cells see p=0 at boundary neighbors.
        """
        for i, j in self.p_temp:
            if i == 0 or i == self.res - 1 or j == 0 or j == self.res - 1:
                self.p_temp[i, j] = 0.0

    @ti.kernel
    def apply_open_dye_bc(self):
        """
        Open boundary: zeroes dye at boundary cells where velocity is directed inward.
        Inflow carries no dye; outflow cells are left alone (zero-gradient via clamping).
        """
        for i, j in self.rho:
            if i == 0 and self.vel[i, j].x > 0:              # left wall, inflow
                self.rho[i, j] = 0.0
            if i == self.res - 1 and self.vel[i, j].x < 0:  # right wall, inflow
                self.rho[i, j] = 0.0
            if j == 0 and self.vel[i, j].y > 0:              # bottom wall, inflow
                self.rho[i, j] = 0.0
            if j == self.res - 1 and self.vel[i, j].y < 0:  # top wall, inflow
                self.rho[i, j] = 0.0

    def step(self):
        """
        Advances the fluid simulation by one time step (Δt) using Operator Splitting.

        Steps:
        1. Advection: Solve for advection of density and velocity fields using chosen scheme.
        2. Projection: Compute divergence of the advected intermediate velocity field.
        3. Pressure Solve: Iteratively solve the Poisson equation for pressure using Jacobi.
        4. Project: Subtract the pressure gradient to enforce a divergence-free velocity field.
        """
        self.time += self.dt
        # Advection
        if self.advection_scheme == 4:
            self._step_eulerian(self.advect_weno_rhs)

        elif self.advection_scheme == 8:
            # WENO5 on vel + Bidirectional CMM on rho (bilinear render).
            # rho is rendered from the backward map rather than advected directly,
            # so only the map delta and vel are stepped here.
            # Remap-trigger check happens at the end of step().
            self._step_rk3(self.advect_weno_rhs, self.backward_map, self.delta_1,
                           self.delta_2, self.new_backward_map, self.dq_delta)
            self._finalize_backward_map_step()
            self.advect_forward_map_rk2()
            self.render_dye_from_backward_map()
            self._step_rk3(self.advect_weno_rhs, self.vel, self.vel_1, self.vel_2,
                           self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 9:
            self._step_eulerian(self.advect_wenoz_rhs)

        elif self.advection_scheme == 10:
            self._step_eulerian(self.advect_teno5_rhs)

        # Optional per-step Laplacian-based unsharp pass on the dye field.
        # Default-off via SimulationConfig.sharpen_strength = 0.0. When on,
        # acts as artistic anti-diffusion -- see the sharpen_rho docstring.
        if self.config.sharpen_strength != 0.0:
            self.sharpen_rho(self.config.sharpen_strength)
            self.rho.copy_from(self.new_rho)

        if self.bc_wall and not self.bc_open:
            self.apply_velocity_bc()
        if self.bc_absorbing:
            self.apply_absorbing_rho_bc()
        elif self.bc_open:
            self.apply_open_dye_bc()

        # Apply external forces (e.g. image gradient or dye gradient)
        if self.force_duration > 0:
            if self.dye_force_active:
                # Dynamic force: update gradient from current dye field
                self._precompute_gradient_perp(self.rho)
            self._apply_stored_force()
            self.force_duration -= self.dt
        else:
            self.dye_force_active = False

        self._apply_persistent_force_kernel(
            self.config.buoyancy_coeff,
            self.config.torque_coeff,
            self.config.radial_coeff
        )

        if self.bc_wall and not self.bc_open:
            self.apply_velocity_bc()

        # Vorticity confinement: counteracts numerical diffusion of rotational
        # structures.  Applied after forces, before pressure solve, so the
        # confinement force is included in the divergence correction.
        if self.config.vorticity_confinement_strength != 0.0:
            self._compute_vorticity()
            self._apply_vorticity_confinement(self.config.vorticity_confinement_strength)

        # Projection (Chorin's Projection Method)
        self.compute_divergence()

        # Pressure solve: FFT (exact, periodic only) or Jacobi (iterative).
        use_fft = (self.config.pressure_solver == 'fft' and not self.bc_wall and not self.bc_open)
        if use_fft:
            self._solve_pressure_fft()
        else:
            for _ in range(100):
                self.pressure_solve_jacobi(self.p, self.p_temp)
                if self.bc_open:
                    self.apply_open_pressure_bc()
                self.p.copy_from(self.p_temp)

        self.pressure_project()

        if self.bc_wall and not self.bc_open:
            self.apply_velocity_bc()

        # CMM remap trigger. After the full step finishes, check the
        # bidirectional consistency error; if the maps have drifted more
        # than `cmm_remap_threshold_cells * dx`, bake the current rho into
        # rho_source and reset both maps to identity. This is the only
        # time the dye gets re-quantized to grid resolution; between
        # remaps the source image is read pixel-perfect via one bilinear.
        if self.advection_scheme == 8:
            self._compute_map_distortion()
            threshold = self.config.cmm_remap_threshold_cells * self.dx
            if float(self._map_distortion[None]) > threshold:
                self.cmm_remap()
