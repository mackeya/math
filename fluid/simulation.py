import taichi as ti
import numpy as np

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
    def __init__(self, res, dt=0.0003):
        self.res = res
        self.dx = 1.0 / res
        self.dt = dt
        self.time = 0.0

        # Velocity fields (staggered grid or collocated? Let's use collocated for simplicity in demo)
        self.vel = ti.Vector.field(2, float, shape=(res, res))
        self.new_vel = ti.Vector.field(2, float, shape=(res, res))

        # Density/Dye fields
        self.rho = ti.field(float, shape=(res, res))
        self.new_rho = ti.field(float, shape=(res, res))

        # Pressure fields
        self.p = ti.field(float, shape=(res, res))
        self.p_temp = ti.field(float, shape=(res, res))
        self.div = ti.field(float, shape=(res, res))

        # Temp pressure field for Jacobi
        self.advection_scheme = 4 # 0: Semi-Lagrangian, 1: Upwind, 2: MacCormack, 3: TVD, 4: WENO5

        # RK3 intermediate fields
        self.rho_1 = ti.field(float, shape=(res, res))
        self.rho_2 = ti.field(float, shape=(res, res))
        self.dq_rho = ti.field(float, shape=(res, res))

        self.vel_1 = ti.Vector.field(2, float, shape=(res, res))
        self.vel_2 = ti.Vector.field(2, float, shape=(res, res))
        self.dq_vel = ti.Vector.field(2, float, shape=(res, res))

        # Gradual force application
        self.image_grad = ti.Vector.field(2, float, shape=(res, res))
        self.force_duration = 0.0
        self.force_scale = 0.0
        self.dye_force_active = False

    @ti.kernel
    def init_patterns(self):
        """
        Initializes the velocity, pressure, and density (dye) fields to zero,
        and sets up a starting dye pattern consisting of a grid and a central circle.
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

    @ti.kernel
    def fill_dye(self, x: float, y: float, radius: float, amount: float):
        """
        Adds dye (or a density scalar field) to the fluid in a circular region.
        Equation: ρ(x, y) = ρ(x, y) + amount (for points within the specified radius)
        """
        for i, j in self.rho:
            dx = abs(i * self.dx - x)
            dy = abs(j * self.dx - y)
            # Periodic distance
            if dx > 0.5: dx = 1.0 - dx
            if dy > 0.5: dy = 1.0 - dy
            dist = ti.sqrt(dx*dx + dy*dy)
            if dist < radius:
                self.rho[i, j] += amount

    @ti.kernel
    def apply_force(self, x: float, y: float, f_x: float, f_y: float, radius: float):
        """
        Applies an external force field 'f' to the fluid within a circular region.
        Updates the momentum equation with force integration:
        u(t + Δt) = u(t) + f * Δt
        """
        for i, j in self.vel:
            dx = abs(i * self.dx - x)
            dy = abs(j * self.dx - y)
            if dx > 0.5: dx = 1.0 - dx
            if dy > 0.5: dy = 1.0 - dy
            dist = ti.sqrt(dx*dx + dy*dy)
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
        Applies periodic boundary conditions for coordinates out of bounds.
        """
        # Bi-linear interpolation with periodic boundaries
        i, j = int(ti.floor(u)), int(ti.floor(v))
        f, g = u - i, v - j

        # Periodic wrap-around for indices
        i0, j0 = i % self.res, j % self.res
        i1, j1 = (i + 1) % self.res, (j + 1) % self.res

        return (1 - f) * (1 - g) * q[i0, j0] + \
               f * (1 - g) * q[i1, j0] + \
               (1 - f) * g * q[i0, j1] + \
               f * g * q[i1, j1]

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
    def advect_upwind(self, field: ti.template(), new_field: ti.template()):
        """
        Solves the advection equation using a first-order upwind finite difference scheme.

        Mathematical detail:
        q^{n+1}_{i,j} = q^n_{i,j} - Δt * (u * q_x + v * q_y)
        Where spatial derivatives q_x and q_y are approximated based on the direction
        of the local velocity to ensure numerical stability (upwind biasing):
        If u > 0: q_x ≈ (q_{i,j} - q_{i-1,j}) / dx
        If u < 0: q_x ≈ (q_{i+1,j} - q_{i,j}) / dx
        """
        for i, j in field:
            u = self.vel[i, j]
            val = field[i, j]
            # Wrap around neighbors
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res

            if u.x > 0:
                val -= (self.dt / self.dx) * u.x * (field[i, j] - field[im1, j])
            else:
                val -= (self.dt / self.dx) * u.x * (field[ip1, j] - field[i, j])

            if u.y > 0:
                val -= (self.dt / self.dx) * u.y * (field[i, j] - field[i, jm1])
            else:
                val -= (self.dt / self.dx) * u.y * (field[i, jp1] - field[i, j])
            new_field[i, j] = val

    @ti.kernel
    def advect_maccormack_step1(self, field: ti.template(), temp_field: ti.template()):
        """
        Step 1 (Predictor) of the MacCormack method for advection.
        Calculates an intermediate field using forward local differences.

        Mathematical detail:
        q^*_{i,j} = q^n_{i,j} - Δt/dx * [u * (q^n_{i+1,j} - q^n_{i,j}) + v * (q^n_{i,j+1} - q^n_{i,j})]
        """
        # Predictor with wrapping
        for i, j in field:
            u = self.vel[i, j]
            val = field[i, j]
            ip1 = (i + 1) % self.res
            jp1 = (j + 1) % self.res
            val -= (self.dt / self.dx) * (u.x * (field[ip1, j] - field[i, j]) + u.y * (field[i, jp1] - field[i, j]))
            temp_field[i, j] = val

    @ti.kernel
    def advect_maccormack_step2(self, field: ti.template(), temp_field: ti.template(), new_field: ti.template()):
        """
        Step 2 (Corrector) of the MacCormack method for advection.
        Uses backward differences on the intermediate field to correct the solution,
        resulting in a scheme that is 2nd-order accurate in both time and space.

        Mathematical detail:
        q^{n+1}_{i,j} = 0.5 * (q^n_{i,j} + q^*_{i,j} - Δt/dx * [u * (q^*_{i,j} - q^*_{i-1,j}) + v * (q^*_{i,j} - q^*_{i,j-1})])
        """
        for i, j in field:
            u = self.vel[i, j]
            im1 = (i - 1) % self.res
            jm1 = (j - 1) % self.res
            val_corr = temp_field[i, j] - (self.dt / self.dx) * (u.x * (temp_field[i, j] - temp_field[im1, j]) + u.y * (temp_field[i, j] - temp_field[i, jm1]))
            new_field[i, j] = 0.5 * (field[i, j] + val_corr)

    @ti.func
    def minmod_scalar(self, a, b):
        res = 0.0
        if a * b > 0.0:
            if ti.abs(a) < ti.abs(b):
                res = a
            else:
                res = b
        return res

    @ti.func
    def minmod_vec2(self, a, b):
        return ti.Vector([self.minmod_scalar(a.x, b.x), self.minmod_scalar(a.y, b.y)])

    @ti.kernel
    def advect_tvd_scalar(self, field: ti.template(), new_field: ti.template()):
        """
        Solves the advection equation using a 2D TVD scheme with Minmod limiter (second-order).
        """
        for i, j in field:
            u = self.vel[i, j]
            val = field[i, j]

            # x direction flux difference
            if u.x > 0:
                im1 = (i - 1) % self.res
                im2 = (i - 2) % self.res
                ip1 = (i + 1) % self.res
                c = u.x * self.dt / self.dx
                dq_i_plus_half = field[ip1, j] - field[i, j]
                dq_i_minus_half = field[i, j] - field[im1, j]
                dq_i_minus_3_half = field[im1, j] - field[im2, j]
                flux_r = u.x * field[i, j] + 0.5 * u.x * (1.0 - c) * self.minmod_scalar(dq_i_plus_half, dq_i_minus_half)
                flux_l = u.x * field[im1, j] + 0.5 * u.x * (1.0 - c) * self.minmod_scalar(dq_i_minus_half, dq_i_minus_3_half)
                val -= (self.dt / self.dx) * (flux_r - flux_l)
            else:
                ip1 = (i + 1) % self.res
                ip2 = (i + 2) % self.res
                im1 = (i - 1) % self.res
                c = -u.x * self.dt / self.dx
                dq_i_minus_half = field[i, j] - field[im1, j]
                dq_i_plus_half = field[ip1, j] - field[i, j]
                dq_i_plus_3_half = field[ip2, j] - field[ip1, j]
                flux_r = u.x * field[ip1, j] - 0.5 * u.x * (1.0 - c) * self.minmod_scalar(dq_i_plus_half, dq_i_plus_3_half)
                flux_l = u.x * field[i, j] - 0.5 * u.x * (1.0 - c) * self.minmod_scalar(dq_i_minus_half, dq_i_plus_half)
                val -= (self.dt / self.dx) * (flux_r - flux_l)

            # y direction flux difference
            if u.y > 0:
                jm1 = (j - 1) % self.res
                jm2 = (j - 2) % self.res
                jp1 = (j + 1) % self.res
                c = u.y * self.dt / self.dx
                dq_j_plus_half = field[i, jp1] - field[i, j]
                dq_j_minus_half = field[i, j] - field[i, jm1]
                dq_j_minus_3_half = field[i, jm1] - field[i, jm2]
                flux_t = u.y * field[i, j] + 0.5 * u.y * (1.0 - c) * self.minmod_scalar(dq_j_plus_half, dq_j_minus_half)
                flux_b = u.y * field[i, jm1] + 0.5 * u.y * (1.0 - c) * self.minmod_scalar(dq_j_minus_half, dq_j_minus_3_half)
                val -= (self.dt / self.dx) * (flux_t - flux_b)
            else:
                jp1 = (j + 1) % self.res
                jp2 = (j + 2) % self.res
                jm1 = (j - 1) % self.res
                c = -u.y * self.dt / self.dx
                dq_j_minus_half = field[i, j] - field[i, jm1]
                dq_j_plus_half = field[i, jp1] - field[i, j]
                dq_j_plus_3_half = field[i, jp2] - field[i, jp1]
                flux_t = u.y * field[i, jp1] - 0.5 * u.y * (1.0 - c) * self.minmod_scalar(dq_j_plus_half, dq_j_plus_3_half)
                flux_b = u.y * field[i, j] - 0.5 * u.y * (1.0 - c) * self.minmod_scalar(dq_j_minus_half, dq_j_plus_half)
                val -= (self.dt / self.dx) * (flux_t - flux_b)

            new_field[i, j] = val

    @ti.kernel
    def advect_tvd_vec(self, field: ti.template(), new_field: ti.template()):
        """
        Solves the advection equation using a 2D TVD scheme with Minmod limiter (second-order) for vector fields.
        """
        for i, j in field:
            u = self.vel[i, j]
            val = field[i, j]

            # x direction flux difference
            if u.x > 0:
                im1 = (i - 1) % self.res
                im2 = (i - 2) % self.res
                ip1 = (i + 1) % self.res
                c = u.x * self.dt / self.dx
                dq_i_plus_half = field[ip1, j] - field[i, j]
                dq_i_minus_half = field[i, j] - field[im1, j]
                dq_i_minus_3_half = field[im1, j] - field[im2, j]
                flux_r = u.x * field[i, j] + 0.5 * u.x * (1.0 - c) * self.minmod_vec2(dq_i_plus_half, dq_i_minus_half)
                flux_l = u.x * field[im1, j] + 0.5 * u.x * (1.0 - c) * self.minmod_vec2(dq_i_minus_half, dq_i_minus_3_half)
                val -= (self.dt / self.dx) * (flux_r - flux_l)
            else:
                ip1 = (i + 1) % self.res
                ip2 = (i + 2) % self.res
                im1 = (i - 1) % self.res
                c = -u.x * self.dt / self.dx
                dq_i_minus_half = field[i, j] - field[im1, j]
                dq_i_plus_half = field[ip1, j] - field[i, j]
                dq_i_plus_3_half = field[ip2, j] - field[ip1, j]
                flux_r = u.x * field[ip1, j] - 0.5 * u.x * (1.0 - c) * self.minmod_vec2(dq_i_plus_half, dq_i_plus_3_half)
                flux_l = u.x * field[i, j] - 0.5 * u.x * (1.0 - c) * self.minmod_vec2(dq_i_minus_half, dq_i_plus_half)
                val -= (self.dt / self.dx) * (flux_r - flux_l)

            # y direction flux difference
            if u.y > 0:
                jm1 = (j - 1) % self.res
                jm2 = (j - 2) % self.res
                jp1 = (j + 1) % self.res
                c = u.y * self.dt / self.dx
                dq_j_plus_half = field[i, jp1] - field[i, j]
                dq_j_minus_half = field[i, j] - field[i, jm1]
                dq_j_minus_3_half = field[i, jm1] - field[i, jm2]
                flux_t = u.y * field[i, j] + 0.5 * u.y * (1.0 - c) * self.minmod_vec2(dq_j_plus_half, dq_j_minus_half)
                flux_b = u.y * field[i, jm1] + 0.5 * u.y * (1.0 - c) * self.minmod_vec2(dq_j_minus_half, dq_j_minus_3_half)
                val -= (self.dt / self.dx) * (flux_t - flux_b)
            else:
                jp1 = (j + 1) % self.res
                jp2 = (j + 2) % self.res
                jm1 = (j - 1) % self.res
                c = -u.y * self.dt / self.dx
                dq_j_minus_half = field[i, j] - field[i, jm1]
                dq_j_plus_half = field[i, jp1] - field[i, j]
                dq_j_plus_3_half = field[i, jp2] - field[i, jp1]
                flux_t = u.y * field[i, jp1] - 0.5 * u.y * (1.0 - c) * self.minmod_vec2(dq_j_plus_half, dq_j_plus_3_half)
                flux_b = u.y * field[i, j] - 0.5 * u.y * (1.0 - c) * self.minmod_vec2(dq_j_minus_half, dq_j_plus_half)
                val -= (self.dt / self.dx) * (flux_t - flux_b)

            new_field[i, j] = val

    @ti.func
    def weno5_reconstruct_scalar(self, v1, v2, v3, v4, v5):
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
    def weno5_reconstruct_vec2(self, v1, v2, v3, v4, v5):
        return ti.Vector([
            self.weno5_reconstruct_scalar(v1.x, v2.x, v3.x, v4.x, v5.x),
            self.weno5_reconstruct_scalar(v1.y, v2.y, v3.y, v4.y, v5.y)
        ])

    @ti.kernel
    def advect_weno_rhs_scalar(self, field: ti.template(), dq: ti.template()):
        for i, j in field:
            u = self.vel[i, j]
            flux_x = 0.0
            flux_y = 0.0

            # x flux
            im3 = (i - 3) % self.res
            im2 = (i - 2) % self.res
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            ip2 = (i + 2) % self.res
            ip3 = (i + 3) % self.res
            if u.x > 0:
                q_R = self.weno5_reconstruct_scalar(field[im2, j], field[im1, j], field[i, j], field[ip1, j], field[ip2, j])
                q_L = self.weno5_reconstruct_scalar(field[im3, j], field[im2, j], field[im1, j], field[i, j], field[ip1, j])
                flux_x = u.x * (q_R - q_L)
            else:
                q_R = self.weno5_reconstruct_scalar(field[ip3, j], field[ip2, j], field[ip1, j], field[i, j], field[im1, j])
                q_L = self.weno5_reconstruct_scalar(field[ip2, j], field[ip1, j], field[i, j], field[im1, j], field[im2, j])
                flux_x = u.x * (q_R - q_L)

            # y flux
            jm3 = (j - 3) % self.res
            jm2 = (j - 2) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            jp2 = (j + 2) % self.res
            jp3 = (j + 3) % self.res
            if u.y > 0:
                q_T = self.weno5_reconstruct_scalar(field[i, jm2], field[i, jm1], field[i, j], field[i, jp1], field[i, jp2])
                q_B = self.weno5_reconstruct_scalar(field[i, jm3], field[i, jm2], field[i, jm1], field[i, j], field[i, jp1])
                flux_y = u.y * (q_T - q_B)
            else:
                q_T = self.weno5_reconstruct_scalar(field[i, jp3], field[i, jp2], field[i, jp1], field[i, j], field[i, jm1])
                q_B = self.weno5_reconstruct_scalar(field[i, jp2], field[i, jp1], field[i, j], field[i, jm1], field[i, jm2])
                flux_y = u.y * (q_T - q_B)

            dq[i, j] = -(flux_x + flux_y) / self.dx

    @ti.kernel
    def advect_weno_rhs_vec(self, field: ti.template(), dq: ti.template()):
        for i, j in field:
            u = self.vel[i, j]
            flux_x = ti.Vector([0.0, 0.0])
            flux_y = ti.Vector([0.0, 0.0])

            # x flux
            im3 = (i - 3) % self.res
            im2 = (i - 2) % self.res
            im1 = (i - 1) % self.res
            ip1 = (i + 1) % self.res
            ip2 = (i + 2) % self.res
            ip3 = (i + 3) % self.res
            if u.x > 0:
                q_R = self.weno5_reconstruct_vec2(field[im2, j], field[im1, j], field[i, j], field[ip1, j], field[ip2, j])
                q_L = self.weno5_reconstruct_vec2(field[im3, j], field[im2, j], field[im1, j], field[i, j], field[ip1, j])
                flux_x = u.x * (q_R - q_L)
            else:
                q_R = self.weno5_reconstruct_vec2(field[ip3, j], field[ip2, j], field[ip1, j], field[i, j], field[im1, j])
                q_L = self.weno5_reconstruct_vec2(field[ip2, j], field[ip1, j], field[i, j], field[im1, j], field[im2, j])
                flux_x = u.x * (q_R - q_L)

            # y flux
            jm3 = (j - 3) % self.res
            jm2 = (j - 2) % self.res
            jm1 = (j - 1) % self.res
            jp1 = (j + 1) % self.res
            jp2 = (j + 2) % self.res
            jp3 = (j + 3) % self.res
            if u.y > 0:
                q_T = self.weno5_reconstruct_vec2(field[i, jm2], field[i, jm1], field[i, j], field[i, jp1], field[i, jp2])
                q_B = self.weno5_reconstruct_vec2(field[i, jm3], field[i, jm2], field[i, jm1], field[i, j], field[i, jp1])
                flux_y = u.y * (q_T - q_B)
            else:
                q_T = self.weno5_reconstruct_vec2(field[i, jp3], field[i, jp2], field[i, jp1], field[i, j], field[i, jm1])
                q_B = self.weno5_reconstruct_vec2(field[i, jp2], field[i, jp1], field[i, j], field[i, jm1], field[i, jm2])
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

    def step_weno(self, field, field_1, field_2, new_field, dq, is_vec=False):
        if is_vec:
            self.advect_weno_rhs_vec(field, dq)
        else:
            self.advect_weno_rhs_scalar(field, dq)
        self.rk3_step1(field, field_1, dq)

        if is_vec:
            self.advect_weno_rhs_vec(field_1, dq)
        else:
            self.advect_weno_rhs_scalar(field_1, dq)
        self.rk3_step2(field, field_1, field_2, dq)

        if is_vec:
            self.advect_weno_rhs_vec(field_2, dq)
        else:
            self.advect_weno_rhs_scalar(field_2, dq)
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
        # One iteration of Jacobi with wrapping
        for i, j in p:
            pl = p[(i - 1) % self.res, j]
            pr = p[(i + 1) % self.res, j]
            pb = p[i, (j - 1) % self.res]
            pt = p[i, (j + 1) % self.res]
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
            grad_p = ti.Vector([(pr - pl) * 0.5 / self.dx, (pt - pb) * 0.5 / self.dx])
            self.vel[i, j] -= grad_p

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
        elif self.advection_scheme == 1:
            self.advect_upwind(self.rho, self.new_rho)
            self.rho.copy_from(self.new_rho)
            self.advect_upwind(self.vel, self.new_vel)
            self.vel.copy_from(self.new_vel)
        elif self.advection_scheme == 2:
            # MacCormack
            self.advect_maccormack_step1(self.rho, self.new_rho)
            self.advect_maccormack_step2(self.rho, self.new_rho, self.rho)
            self.advect_maccormack_step1(self.vel, self.new_vel)
            self.advect_maccormack_step2(self.vel, self.new_vel, self.vel)
        elif self.advection_scheme == 3:
            # TVD
            self.advect_tvd_scalar(self.rho, self.new_rho)
            self.rho.copy_from(self.new_rho)
            self.advect_tvd_vec(self.vel, self.new_vel)
            self.vel.copy_from(self.new_vel)

        elif self.advection_scheme == 4:
            # WENO5 + SSP-RK3
            self.step_weno(self.rho, self.rho_1, self.rho_2, self.new_rho, self.dq_rho, False)
            self.rho.copy_from(self.new_rho)
            self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel, True)
            self.vel.copy_from(self.new_vel)

        # Apply external forces (e.g. image gradient or dye gradient)
        if self.force_duration > 0:
            if self.dye_force_active:
                # Dynamic force: update gradient from current dye field
                self._precompute_gradient_perp(self.rho)
            self._apply_stored_force()
            self.force_duration -= self.dt
        else:
            self.dye_force_active = False

        # Projection (Chorin's Projection Method)
        self.compute_divergence()

        # Jacobi iterative solver for Pressure
        for _ in range(100): # Increased iterations for better convergence
            self.pressure_solve_jacobi(self.p, self.p_temp)
            self.p.copy_from(self.p_temp)

        self.pressure_project()
