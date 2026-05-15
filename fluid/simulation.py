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
    # instance). 0 = Semi-Lagrangian, 2 = Selle-style MacCormack + clamp,
    # 4 = WENO5 + SSP-RK3, 5 = Hybrid (WENO5 on vel, MacCormack on rho),
    # 6 = Hybrid (WENO5 on vel, CIP cubic-Hermite SL on rho with FC limit),
    # 7 = Hybrid (WENO5 on vel, passive Lagrangian particles on rho),
    # 8 = Hybrid (WENO5 on vel, Bidirectional Characteristic Mapping on rho).
    advection_scheme: int = 4
    # Remap trigger threshold for the Characteristic Mapping Method
    # (advection_scheme == 8). When the maximum self-consistency error
    # max |X(Y(x)) - x| of the bidirectional maps exceeds this many cell
    # widths, the simulation bakes the current rho into rho_source and
    # resets both maps to identity. Smaller = more frequent remaps (more
    # grid-quantization, less visible map distortion); larger = less
    # frequent (preserves smooth image quality longer, but eventually the
    # map gets noticeably warped before snapping back). Units: cells (dx).
    cmm_remap_threshold_cells: float = 1.0

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

        # Temp pressure field for Jacobi
        # Advection scheme: 0 = Semi-Lagrangian, 2 = Selle-style MacCormack,
        # 4 = WENO5, 5 = Hybrid (WENO on vel, MacCormack on rho).
        self.advection_scheme = config.advection_scheme

        # RK3 intermediate fields
        self.rho_1 = ti.field(float, shape=(self.res, self.res))
        self.rho_2 = ti.field(float, shape=(self.res, self.res))
        self.dq_rho = ti.field(float, shape=(self.res, self.res))

        self.vel_1 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.vel_2 = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.dq_vel = ti.Vector.field(2, float, shape=(self.res, self.res))

        # Scratch fields for the Selle-style MacCormack predictor (phi_hat).
        # Used only when advection_scheme == 2. Kept separate from the RK3
        # intermediates so the two schemes never share scratch state.
        self.predict_rho = ti.field(float, shape=(self.res, self.res))
        self.predict_vel = ti.Vector.field(2, float, shape=(self.res, self.res))

        # CIP (advection_scheme == 6): gradient of rho is transported alongside
        # rho itself, enabling cubic-Hermite reconstruction during advection.
        # Vector field with .x = d rho / dx, .y = d rho / dy. new_grad_rho is
        # the corrector scratch; the caller swaps after each CIP step.
        self.grad_rho = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.new_grad_rho = ti.Vector.field(2, float, shape=(self.res, self.res))

        # Passive Lagrangian dye (advection_scheme == 7): one particle per
        # grid cell at init. Each particle carries a position (physical, in
        # [0, 1]^2) and a weight (the dye intensity it transports). The
        # cells are re-built from particles each step via bilinear splatting.
        self.n_particles = self.res * self.res
        self.particle_pos = ti.Vector.field(2, float, shape=(self.n_particles,))
        self.particle_weight = ti.field(float, shape=(self.n_particles,))

        # Bidirectional Characteristic Mapping (advection_scheme == 8):
        # rho_source is the frozen dye snapshot at the most recent remap.
        # backward_map (X) maps current grid points back to their initial
        # positions; advected by the velocity field each step. forward_map
        # (Y) is the per-cell forward trajectory from the remap moment;
        # used purely to detect when the maps have drifted enough that a
        # remap is needed. Both maps are reset to the identity at remap.
        # MacCormack predictor/corrector scratch for advecting X reuses
        # the same kernels as the rest of the advection schemes.
        self.rho_source = ti.field(float, shape=(self.res, self.res))
        self.backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.forward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.predict_backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        self.new_backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
        # 0-d accumulator for the max-self-consistency-error reduction.
        # Mirrors the pattern of self._max_vel_norm.
        self._map_distortion = ti.field(float, shape=())

        # Scalar 0-d field used as the accumulator for the max-CFL reduction
        # in max_cfl(). Lives on the simulation so we don't allocate per call.
        self._max_vel_norm = ti.field(float, shape=())

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
        Python `init_patterns` wrapper so that wrapper can also seed the CIP
        gradient field by calling another kernel after this one.
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
        seed the per-scheme auxiliary state (CIP gradient, particle pool,
        CMM source + maps) so any scheme can be selected immediately. The
        extra kernel calls are cheap when the targeted scheme is off.
        """
        self._init_patterns_kernel()
        self.init_grad_rho_from_rho()
        self._init_particles_from_rho()
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
        # Seed per-scheme auxiliary state from the loaded rho. No-op cost
        # for schemes that don't read these.
        self.init_grad_rho_from_rho()
        self._init_particles_from_rho()
        self._init_cmm_state_from_rho()

    @ti.kernel
    def _fill_dye_kernel(self, x: float, y: float, radius: float, amount: float):
        """
        Kernel body for `fill_dye`. Adds dye in a circular region without
        touching grad_rho; the Python wrapper refreshes the CIP gradient
        afterward.
        """
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
        Adds dye in a circular region around (x, y). In Eulerian modes this
        writes to the grid rho field and refreshes the CIP gradient. In
        particle mode (scheme 7) this adds to the weight of every particle
        in the affected disk. In CMM mode (scheme 8) it projects each
        affected current-frame cell through the backward map and deposits
        the dye into rho_source at the back-traced location, so the new
        dye participates in the existing map without forcing a remap.
        """
        if self.advection_scheme == 8:
            self._fill_dye_cmm_kernel(x, y, radius, amount)
        elif self.advection_scheme == 7:
            self._fill_dye_particles_kernel(x, y, radius, amount)
        else:
            self._fill_dye_kernel(x, y, radius, amount)
            self.init_grad_rho_from_rho()

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
    def _precompute_gradient(self, img: ti.template()):
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

            grad_x = (img[ip1, j] - img[im1, j]) * 0.5 / self.dx
            grad_y = (img[i, jp1] - img[i, jm1]) * 0.5 / self.dx

            self.image_grad[i, j] = ti.Vector([grad_x, grad_y])

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
        Values above ~1.0 indicate the per-step advection trajectory is
        crossing more than one cell, at which point linear back-traces
        (semi-Lagrangian, MacCormack) lose accuracy. Useful as a diagnostic
        for whether observed artifacts are CFL-induced.
        """
        self._reduce_max_vel_norm()
        return float(self._max_vel_norm[None]) * self.dt / self.dx

    @ti.kernel
    def advect_semi_lagrangian(self, field: ti.template(), new_field: ti.template()):
        """
        Solves the advection equation ∂q/∂t + (u · ∇)q = 0 using the Semi-Lagrangian method.

        Mathematical detail:
        For every grid point x, we trace back along the velocity field to find where the
        particle came from (assuming constant velocity over Δt):
        x_prev = x - u(x) * Δt
        Then we interpolate the field at x_prev to update the current grid cell:
        q^{n+1}(x) = q^n(x_prev)
        """
        for i, j in field:
            # Backtrace coordinate
            p = ti.Vector([i + 0.5, j + 0.5]) - self.dt * self.vel[i, j] / self.dx
            # Periodic wrap-around happens inside sample
            new_field[i, j] = self.sample(field, p.x - 0.5, p.y - 0.5)


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

        3. Compute the donor neighborhood [lo, hi] over the bilinear support
           (4 cells) at the back-traced location x_back = x - u(x)*dt.

        4. Modified MacCormack: if phi_corrected lies inside [lo, hi], use
           it; otherwise fall back to phi_hat[x] (the safe SL value). The
           fallback avoids the stair-step / halo artifacts that hard
           clamping to the [lo, hi] boundary produces.

        Honors periodic vs wall boundary conditions through the same dispatch
        used elsewhere in the file. Reference: Selle, Fedkiw, Kim, Liu,
        Rossignac (2008), "An Unconditionally Stable MacCormack Method";
        Bridson, "Fluid Simulation for Computer Graphics", 2nd ed., Ch. 5
        (modified MacCormack fallback).
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

            # 2x2 donor stencil: the predictor's bilinear support.
            # Component-wise min/max works for scalar and 2-vector fields,
            # which is how the same kernel can advect both rho and vel.
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
            lo = ti.min(ti.min(v00, v10), ti.min(v01, v11))
            hi = ti.max(ti.max(v00, v10), ti.max(v01, v11))

            # Modified MacCormack: if the corrected value would have been
            # clamped, fall back to the safe semi-Lagrangian predictor
            # value instead of pinning to the boundary. The pinned-boundary
            # variant produces stair-step / halo artifacts on sharp dye
            # features; the fallback is smoother.
            clamped = ti.math.clamp(phi_corrected, lo, hi)
            new_field[i, j] = ti.select(
                clamped == phi_corrected, phi_corrected, phi_hat[i, j]
            )


    @ti.kernel
    def init_grad_rho_from_rho(self):
        """
        Computes grad_rho = nabla rho via central differences. Used to seed
        the CIP gradient field after rho is set by an init method or by
        fill_dye, so the CIP advection has a sensible starting gradient.
        Honors periodic vs wall boundary conditions.
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
            gx = (self.rho[ip1, j] - self.rho[im1, j]) * 0.5 / self.dx
            gy = (self.rho[i, jp1] - self.rho[i, jm1]) * 0.5 / self.dx
            self.grad_rho[i, j] = ti.Vector([gx, gy])


    @ti.func
    def fc_limit(self, g: float, secant: float) -> float:
        """
        Fritsch-Carlson monotone-cubic limit for a Hermite tangent.

        Given a cell-coord secant slope `secant` between two adjacent samples
        and a stored tangent `g` at one of them (both in cell-coord units),
        returns g clamped so the cubic Hermite over the cell is provably
        monotone: tangent must share sign with the secant and its magnitude
        must not exceed 3 * |secant|. If the secant is zero or the tangent
        sign disagrees with it, the tangent is zeroed (this enforces local
        extrema at sample points and prevents the cubic from creating new
        ones). Reference: Fritsch, Carlson (1980) "Monotone Piecewise Cubic
        Interpolation", SIAM J. Numer. Anal.
        """
        result = 0.0
        if secant * g > 0.0:
            # Tangent shares sign with the secant; cap magnitude at 3|secant|.
            max_g = 3.0 * ti.abs(secant)
            if ti.abs(g) > max_g:
                # Preserve sign of secant (which equals sign of g here).
                if secant > 0.0:
                    result = max_g
                else:
                    result = -max_g
            else:
                result = g
        return result

    @ti.kernel
    def advect_cip(self, field: ti.template(),
                   grad_field: ti.template(),
                   new_field: ti.template(),
                   new_grad_field: ti.template()):
        """
        One CIP advection step on a scalar field with its gradient tracked.

        For each grid cell x = (i + 0.5, j + 0.5) * dx:

        1. Back-trace x_back = x - u(x) * dt and find the 2x2 donor
           neighborhood (i0..i1, j0..j1). Local coords (u_loc, v_loc) in
           [0, 1] give the fractional position inside this cell.

        2. Read the 12 corner samples: field at the four cells (f00, f10,
           f01, f11), grad_field.x at the same cells (gx00..gx11), and
           grad_field.y at the same cells (gy00..gy11).

        3. Evaluate the tensor-product cubic Hermite surface at
           (u_loc, v_loc). With cell spacing 1 (cell-coordinate units),
           the tangent basis functions multiply gradients converted to
           cell units, so each gradient is scaled by dx before being
           plugged into the Hermite basis.

        4. The new field value is the surface value at (u_loc, v_loc); the
           new gradient is the analytic partials of the same surface,
           scaled back to physical units (divided by dx).

        Cross-derivative d^2 rho / dx dy is NOT tracked; the bicubic
        Hermite is therefore a tensor product with the cross term implicitly
        set to zero at each corner. This is the standard simplification in
        graphics CIP implementations; the resulting surface is C^1 along
        cell edges and produces dramatically less diffusion than any value-
        only scheme. Reference: Yabe, Aoki (1991), "A universal solver for
        hyperbolic equations by cubic-polynomial interpolation".
        """
        # Standard Hermite basis on [0, 1] in cell-coordinate units.
        # h00(t) = 2t^3 - 3t^2 + 1     (value at left endpoint)
        # h10(t) = t^3 - 2t^2 + t      (tangent at left endpoint)
        # h01(t) = -2t^3 + 3t^2        (value at right endpoint)
        # h11(t) = t^3 - t^2           (tangent at right endpoint)
        # Derivatives:
        # h00'(t) = 6t^2 - 6t
        # h10'(t) = 3t^2 - 4t + 1
        # h01'(t) = -6t^2 + 6t
        # h11'(t) = 3t^2 - 2t
        for i, j in field:
            # Back-trace location (cell-coordinate units).
            p = ti.Vector([i + 0.5, j + 0.5]) - self.dt * self.vel[i, j] / self.dx
            # Convert to sample()-style fractional position (0,0) at corner of cell (0,0)
            u = p.x - 0.5
            v = p.y - 0.5
            i_b = int(ti.floor(u))
            j_b = int(ti.floor(v))
            tx = u - i_b
            ty = v - j_b

            i0 = i_b % self.res
            i1 = (i_b + 1) % self.res
            j0 = j_b % self.res
            j1 = (j_b + 1) % self.res
            if ti.static(self.bc_wall):
                i0 = ti.math.clamp(i_b,     0, self.res - 1)
                i1 = ti.math.clamp(i_b + 1, 0, self.res - 1)
                j0 = ti.math.clamp(j_b,     0, self.res - 1)
                j1 = ti.math.clamp(j_b + 1, 0, self.res - 1)

            # Corner samples
            f00 = field[i0, j0]; f10 = field[i1, j0]; f01 = field[i0, j1]; f11 = field[i1, j1]
            g00 = grad_field[i0, j0]; g10 = grad_field[i1, j0]
            g01 = grad_field[i0, j1]; g11 = grad_field[i1, j1]
            # Gradients in cell-coordinate units (multiply by dx).
            gx00 = g00.x * self.dx; gx10 = g10.x * self.dx
            gx01 = g01.x * self.dx; gx11 = g11.x * self.dx
            gy00 = g00.y * self.dx; gy10 = g10.y * self.dx
            gy01 = g01.y * self.dx; gy11 = g11.y * self.dx

            # Fritsch-Carlson monotone-cubic limiting. The Hermite cubic
            # between two adjacent samples is provably monotone iff each
            # tangent shares sign with the secant slope between the samples
            # and has magnitude <= 3 |secant|. Limiting the corner gradients
            # locally (per back-traced cell) before the Hermite eval kills
            # the new-extremum overshoots that plain CIP otherwise produces
            # everywhere -- at the cost of degrading toward plain SL near
            # local extrema (which is the right trade-off).
            #
            # Each gradient component is constrained by exactly one edge of
            # the donor cell: gx values by the two horizontal edges, gy
            # values by the two vertical edges. Secants are in cell-coord
            # units (cells are 1 unit apart by construction).
            S_x_bot   = f10 - f00
            S_x_top   = f11 - f01
            S_y_left  = f01 - f00
            S_y_right = f11 - f10
            gx00 = self.fc_limit(gx00, S_x_bot)
            gx10 = self.fc_limit(gx10, S_x_bot)
            gx01 = self.fc_limit(gx01, S_x_top)
            gx11 = self.fc_limit(gx11, S_x_top)
            gy00 = self.fc_limit(gy00, S_y_left)
            gy01 = self.fc_limit(gy01, S_y_left)
            gy10 = self.fc_limit(gy10, S_y_right)
            gy11 = self.fc_limit(gy11, S_y_right)

            # Hermite basis values + derivatives at (tx, ty).
            Hx0 = 2.0 * tx * tx * tx - 3.0 * tx * tx + 1.0
            Hx1 = tx * tx * tx - 2.0 * tx * tx + tx           # multiplies gx (left tangent)
            Hx2 = -2.0 * tx * tx * tx + 3.0 * tx * tx
            Hx3 = tx * tx * tx - tx * tx                       # multiplies gx (right tangent)
            dHx0 = 6.0 * tx * tx - 6.0 * tx
            dHx1 = 3.0 * tx * tx - 4.0 * tx + 1.0
            dHx2 = -6.0 * tx * tx + 6.0 * tx
            dHx3 = 3.0 * tx * tx - 2.0 * tx

            Hy0 = 2.0 * ty * ty * ty - 3.0 * ty * ty + 1.0
            Hy1 = ty * ty * ty - 2.0 * ty * ty + ty
            Hy2 = -2.0 * ty * ty * ty + 3.0 * ty * ty
            Hy3 = ty * ty * ty - ty * ty
            dHy0 = 6.0 * ty * ty - 6.0 * ty
            dHy1 = 3.0 * ty * ty - 4.0 * ty + 1.0
            dHy2 = -6.0 * ty * ty + 6.0 * ty
            dHy3 = 3.0 * ty * ty - 2.0 * ty

            # Helper: along each row j_fixed in {0, 1}, build the 1D cubic
            # Hermite of f and of gy at the row. Then combine across rows
            # with another 1D cubic Hermite in y using f-values + gy-values
            # at the endpoints.
            #
            # Row at y=0: F_row0(tx) = f00 * Hx0 + gx00 * Hx1 + f10 * Hx2 + gx10 * Hx3
            # Row at y=1: F_row1(tx) = f01 * Hx0 + gx01 * Hx1 + f11 * Hx2 + gx11 * Hx3
            # Y-gradient at row y=0 evaluated at tx: G_row0(tx) =
            #     gy00 * Hx0 + 0 * Hx1 + gy10 * Hx2 + 0 * Hx3      (cross-term = 0)
            # Y-gradient at row y=1 evaluated at tx: G_row1(tx) similarly.
            # Then sample value: F(tx, ty) = F_row0 * Hy0 + G_row0 * Hy1 + F_row1 * Hy2 + G_row1 * Hy3
            F_row0 = f00 * Hx0 + gx00 * Hx1 + f10 * Hx2 + gx10 * Hx3
            F_row1 = f01 * Hx0 + gx01 * Hx1 + f11 * Hx2 + gx11 * Hx3
            G_row0 = gy00 * Hx0 + gy10 * Hx2
            G_row1 = gy01 * Hx0 + gy11 * Hx2

            new_f = F_row0 * Hy0 + G_row0 * Hy1 + F_row1 * Hy2 + G_row1 * Hy3

            # New x-gradient: derivative of the surface in x at (tx, ty).
            # In cell units: d/d(tx) of the surface above.
            # d F_row / d(tx) = f * dHx0 + gx * dHx1 + f * dHx2 + gx * dHx3
            dFdtx_row0 = f00 * dHx0 + gx00 * dHx1 + f10 * dHx2 + gx10 * dHx3
            dFdtx_row1 = f01 * dHx0 + gx01 * dHx1 + f11 * dHx2 + gx11 * dHx3
            # d G_row / d(tx) uses gy values (cross term zero so only h0/h2 basis)
            dGdtx_row0 = gy00 * dHx0 + gy10 * dHx2
            dGdtx_row1 = gy01 * dHx0 + gy11 * dHx2
            new_gx_cell = (dFdtx_row0 * Hy0 + dGdtx_row0 * Hy1
                           + dFdtx_row1 * Hy2 + dGdtx_row1 * Hy3)

            # New y-gradient: derivative of the surface in y at (tx, ty).
            # d F(tx, ty) / d(ty) = F_row0 * dHy0 + G_row0 * dHy1 + F_row1 * dHy2 + G_row1 * dHy3
            new_gy_cell = (F_row0 * dHy0 + G_row0 * dHy1
                           + F_row1 * dHy2 + G_row1 * dHy3)

            new_field[i, j] = new_f
            # Convert gradients back from cell-coordinate units to physical units.
            new_grad_field[i, j] = ti.Vector([new_gx_cell / self.dx,
                                              new_gy_cell / self.dx])


    @ti.kernel
    def _init_particles_from_rho(self):
        """
        Initializes the passive-Lagrangian dye particles from the current
        rho field. One particle per grid cell: position = cell center,
        weight = rho value at that cell. Called from init_patterns,
        init_from_image, and any other code path that authoritatively sets
        rho. The particle layout is the same independent of bc_type; only
        the per-step advection differs.
        """
        for i, j in self.rho:
            p_idx = i * self.res + j
            self.particle_pos[p_idx] = ti.Vector([(i + 0.5) * self.dx,
                                                   (j + 0.5) * self.dx])
            self.particle_weight[p_idx] = self.rho[i, j]

    @ti.kernel
    def advect_particles_rk2(self):
        """
        Advances every particle one step via the RK2 midpoint method:
            u0    = vel(x)
            x_mid = x + 0.5 * dt * u0
            u_mid = vel(x_mid)
            x_new = x + dt * u_mid
        Velocity is sampled from the simulation's grid via bilinear
        interpolation (the existing `sample()` helper).

        Boundary conditions: periodic wrap when bc_type='periodic',
        clamp-to-edge otherwise. Particles are never removed; "absorbing"
        only kills dye when the splat step writes the boundary rows of
        rho to zero in apply_absorbing_rho_bc.
        """
        for p in range(self.n_particles):
            x = self.particle_pos[p]
            # sample() expects coords where integer == cell index;
            # particle.pos is in physical units. Convert: cell-center coord
            # = pos / dx, sample arg = cell-center coord - 0.5.
            u0 = self.sample(self.vel, x.x / self.dx - 0.5, x.y / self.dx - 0.5)
            x_mid = x + 0.5 * self.dt * u0
            u_mid = self.sample(self.vel,
                                x_mid.x / self.dx - 0.5,
                                x_mid.y / self.dx - 0.5)
            x_new = x + self.dt * u_mid

            if ti.static(self.bc_wall):
                x_new.x = ti.math.clamp(x_new.x, 0.0, 1.0)
                x_new.y = ti.math.clamp(x_new.y, 0.0, 1.0)
            else:
                # Periodic wrap to [0, 1). Subtracting floor handles negative
                # values correctly (Taichi's % follows the dividend's sign).
                x_new.x = x_new.x - ti.floor(x_new.x)
                x_new.y = x_new.y - ti.floor(x_new.y)
            self.particle_pos[p] = x_new

    @ti.kernel
    def splat_particles_to_rho(self):
        """
        Bilinearly deposits each particle's weight into the four nearest
        rho cells. Caller is responsible for zeroing rho before calling
        (so multiple sources can be splatted into the same field if
        desired). Atomic adds make the splat parallel-safe across particles
        that happen to land in the same cell.
        """
        for p in range(self.n_particles):
            x = self.particle_pos[p]
            w = self.particle_weight[p]
            u = x.x / self.dx - 0.5
            v = x.y / self.dx - 0.5
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

            ti.atomic_add(self.rho[i0, j0], (1.0 - fx) * (1.0 - fy) * w)
            ti.atomic_add(self.rho[i1, j0], fx         * (1.0 - fy) * w)
            ti.atomic_add(self.rho[i0, j1], (1.0 - fx) * fy         * w)
            ti.atomic_add(self.rho[i1, j1], fx         * fy         * w)

    @ti.kernel
    def _fill_dye_particles_kernel(self, x: float, y: float,
                                   radius: float, amount: float):
        """
        Particle-mode equivalent of `_fill_dye_kernel`. Adds `amount` to
        the weight of every particle whose current position is within
        `radius` of (x, y). Total dye injected per click depends on local
        particle density, which under sustained flow can vary somewhat
        from the uniform grid-equivalent. Acceptable trade-off for v1.
        """
        for p in range(self.n_particles):
            pos = self.particle_pos[p]
            dist_x = ti.abs(pos.x - x)
            dist_y = ti.abs(pos.y - y)
            if ti.static(not self.bc_wall):
                if dist_x > 0.5: dist_x = 1.0 - dist_x
                if dist_y > 0.5: dist_y = 1.0 - dist_y
            dist = ti.sqrt(dist_x * dist_x + dist_y * dist_y)
            if dist < radius:
                self.particle_weight[p] += amount


    @ti.kernel
    def _init_cmm_state_from_rho(self):
        """
        Seeds the Bidirectional CMM state from the current rho field:
          - rho_source <- rho
          - backward_map[i, j] = (0, 0)         (zero deformation)
          - forward_map[i, j] = cell_center(i, j)  (identity)
        backward_map stores the deformation delta(x) = X(x) - x rather
        than the absolute back-traced coordinate; this makes the field
        continuous across periodic boundaries (otherwise the field would
        have a step jump of size 1 at the seam, which MacCormack's
        bilinear interpolation cannot handle correctly).
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
        Post-advection step for the backward map: copy MacCormack output
        into backward_map AND subtract the source term u*dt. This source
        term comes from rewriting the standard backward-map update
            X^{n+1}(x) = X^n(x - u dt)
        in terms of delta = X - x:
            delta^{n+1}(x) = delta^n(x - u dt) - u(x) dt
        where the first part is plain advection of delta (handled by the
        preceding MacCormack pair) and the second part is the per-cell
        source term applied here.
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
    def render_dye_from_backward_map(self):
        """
        Reconstructs rho for the current frame by sampling rho_source at
        X(x) = x + delta(x). This is the single bilinear interpolation
        that makes CMM non-accumulating: regardless of how many advection
        steps have passed since the last remap, the rendered rho is one
        bilinear sample of the frozen source image. All inaccuracy lives
        in the deformation field, not in the dye field.
        """
        for i, j in self.rho:
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            src = ti.Vector([cx, cy]) + self.backward_map[i, j]
            self.rho[i, j] = self.sample(self.rho_source,
                                          src.x / self.dx - 0.5,
                                          src.y / self.dx - 0.5)

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

    @ti.kernel
    def advect_weno_rhs(self, field: ti.template(), dq: ti.template()):
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

    def step_weno(self, field, field_1, field_2, new_field, dq):
        self.advect_weno_rhs(field, dq)
        self.rk3_step1(field, field_1, dq)

        self.advect_weno_rhs(field_1, dq)
        self.rk3_step2(field, field_1, field_2, dq)

        self.advect_weno_rhs(field_2, dq)
        self.rk3_step3(field, field_2, new_field, dq)

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
        if self.advection_scheme == 0:
            self.advect_semi_lagrangian(self.rho, self.new_rho)
            self.rho.copy_from(self.new_rho)
            self.advect_semi_lagrangian(self.vel, self.new_vel)
            self.vel.copy_from(self.new_vel)
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

        elif self.advection_scheme == 4:
            # WENO5 + SSP-RK3
            self.step_weno(self.rho, self.rho_1, self.rho_2, self.new_rho, self.dq_rho)
            self.rho.copy_from(self.new_rho)
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 5:
            # Hybrid: WENO5 advects velocity (best fine-structure resolution),
            # MacCormack-SL advects rho (clamp keeps sharp dye edges crisp).
            # The two transports are independent; rho is done first so that
            # it is advected by the current velocity, not the just-updated
            # one (consistent with the other scheme dispatches).
            self.advect_maccormack_predict(self.rho, self.predict_rho)
            self.advect_maccormack_correct(self.rho, self.predict_rho, self.new_rho)
            self.rho.copy_from(self.new_rho)
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 6:
            # Hybrid: WENO5 on vel + CIP cubic-Hermite SL on rho. Same
            # rho-first ordering as scheme 5. CIP also advects grad_rho
            # alongside rho; we swap both fields back after the step.
            self.advect_cip(self.rho, self.grad_rho,
                            self.new_rho, self.new_grad_rho)
            self.rho.copy_from(self.new_rho)
            self.grad_rho.copy_from(self.new_grad_rho)
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 7:
            # Hybrid: WENO5 on vel + passive Lagrangian particles on rho.
            # Particles are advected by sampling the grid velocity (RK2
            # midpoint), then bilinear-splatted back into rho so that
            # rendering and rho-dependent forces continue to work
            # unchanged. The rho.fill(0) is necessary because the splat
            # accumulates atomically; otherwise old values would persist.
            self.advect_particles_rk2()
            self.rho.fill(0.0)
            self.splat_particles_to_rho()
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 8:
            # Hybrid: WENO5 on vel + Bidirectional CMM on rho.
            # 1) Advect the backward map X using MacCormack-FC. Reuses the
            #    existing scheme-2 kernels, which are templated on field
            #    type and work for vec2 inputs.
            # 2) Advance the forward map Y by RK2 (Lagrangian trajectory).
            # 3) Render rho by sampling rho_source at X. This is the
            #    non-accumulating step: no matter how many frames have
            #    passed since the last remap, rho is one bilinear sample
            #    of the frozen source image.
            # 4) Velocity as usual.
            # Remap-trigger check happens at the end of step().
            self.advect_maccormack_predict(self.backward_map, self.predict_backward_map)
            self.advect_maccormack_correct(self.backward_map, self.predict_backward_map,
                                            self.new_backward_map)
            # Combine the copy-back with the per-step source term u*dt
            # that the delta-formulation of the backward-map update
            # requires (see _finalize_backward_map_step docstring).
            self._finalize_backward_map_step()
            self.advect_forward_map_rk2()
            self.render_dye_from_backward_map()
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
            self.vel.copy_from(self.new_vel)

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

        # Projection (Chorin's Projection Method)
        self.compute_divergence()

        # Jacobi iterative solver for Pressure
        for _ in range(100): # Increased iterations for better convergence
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
