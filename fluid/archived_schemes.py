# type: ignore
# pylint: skip-file
# flake8: noqa
"""
Archived advection schemes removed from simulation.py.

These are methods of FluidSimulation that were dropped because WENO5, WENO-Z,
TENO5, and WENO+bilinear CMM proved sufficient. They are preserved here for
reference in case any should be revisited.

Each section lists:
  - The Taichi fields / instance attributes that scheme exclusively used.
  - The verbatim method code as it appeared in simulation.py.

None of this file is imported anywhere; it is documentation only.
"""

import taichi as ti
import numpy as np

# ---------------------------------------------------------------------------
# SCHEME 0 — Semi-Lagrangian
# ---------------------------------------------------------------------------
# Exclusive fields: none (uses only self.rho, self.new_rho, self.vel,
#                         self.new_vel, which are kept by other schemes).
#
# Dispatch in step():
#   if self.advection_scheme == 0:
#       self.advect_semi_lagrangian(self.rho, self.new_rho)
#       self.rho.copy_from(self.new_rho)
#       self.advect_semi_lagrangian(self.vel, self.new_vel)
#       self.vel.copy_from(self.new_vel)

class _SemiLagrangianArchive:
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


# ---------------------------------------------------------------------------
# SCHEME 2 — Selle-style Semi-Lagrangian MacCormack + extrema clamp
# ---------------------------------------------------------------------------
# Exclusive fields:
#   self.predict_rho = ti.field(float, shape=(self.res, self.res))
#   self.predict_vel = ti.Vector.field(2, float, shape=(self.res, self.res))
#
# Dispatch in step():
#   elif self.advection_scheme == 2:
#       self.advect_maccormack_predict(self.rho, self.predict_rho)
#       self.advect_maccormack_correct(self.rho, self.predict_rho, self.new_rho)
#       self.rho.copy_from(self.new_rho)
#       self.advect_maccormack_predict(self.vel, self.predict_vel)
#       self.advect_maccormack_correct(self.vel, self.predict_vel, self.new_vel)
#       self.vel.copy_from(self.new_vel)

class _MacCormackArchive:
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


# ---------------------------------------------------------------------------
# SCHEME 5 — Hybrid WENO5 velocity + MacCormack rho
# ---------------------------------------------------------------------------
# No exclusive fields (reuses predict_rho/predict_vel from scheme 2 and
# rho_1/rho_2/dq_rho/vel_1/vel_2/dq_vel from scheme 4).
#
# Dispatch in step():
#   elif self.advection_scheme == 5:
#       self.advect_maccormack_predict(self.rho, self.predict_rho)
#       self.advect_maccormack_correct(self.rho, self.predict_rho, self.new_rho)
#       self.rho.copy_from(self.new_rho)
#       self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
#       self.vel.copy_from(self.new_vel)


# ---------------------------------------------------------------------------
# SCHEME 6 — Hybrid WENO5 velocity + CIP cubic-Hermite SL rho
# ---------------------------------------------------------------------------
# Exclusive fields:
#   self.grad_rho     = ti.Vector.field(2, float, shape=(self.res, self.res))
#   self.new_grad_rho = ti.Vector.field(2, float, shape=(self.res, self.res))
#
# Also required fc_limit (archived below) and init_grad_rho_from_rho (archived below).
#
# Dispatch in step():
#   elif self.advection_scheme == 6:
#       self.advect_cip(self.rho, self.grad_rho, self.new_rho, self.new_grad_rho)
#       self.rho.copy_from(self.new_rho)
#       self.grad_rho.copy_from(self.new_grad_rho)
#       self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
#       self.vel.copy_from(self.new_vel)
#
# fill_dye() re-seeds grad_rho after every injection:
#   else:
#       self._fill_dye_kernel(x, y, radius, amount)
#       self.init_grad_rho_from_rho()
#
# init_patterns() and init_from_image() also seed grad_rho:
#   self.init_grad_rho_from_rho()

class _CIPArchive:
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

            # Fritsch-Carlson monotone-cubic limiting.
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
            Hx1 = tx * tx * tx - 2.0 * tx * tx + tx
            Hx2 = -2.0 * tx * tx * tx + 3.0 * tx * tx
            Hx3 = tx * tx * tx - tx * tx
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

            F_row0 = f00 * Hx0 + gx00 * Hx1 + f10 * Hx2 + gx10 * Hx3
            F_row1 = f01 * Hx0 + gx01 * Hx1 + f11 * Hx2 + gx11 * Hx3
            G_row0 = gy00 * Hx0 + gy10 * Hx2
            G_row1 = gy01 * Hx0 + gy11 * Hx2

            new_f = F_row0 * Hy0 + G_row0 * Hy1 + F_row1 * Hy2 + G_row1 * Hy3

            dFdtx_row0 = f00 * dHx0 + gx00 * dHx1 + f10 * dHx2 + gx10 * dHx3
            dFdtx_row1 = f01 * dHx0 + gx01 * dHx1 + f11 * dHx2 + gx11 * dHx3
            dGdtx_row0 = gy00 * dHx0 + gy10 * dHx2
            dGdtx_row1 = gy01 * dHx0 + gy11 * dHx2
            new_gx_cell = (dFdtx_row0 * Hy0 + dGdtx_row0 * Hy1
                           + dFdtx_row1 * Hy2 + dGdtx_row1 * Hy3)

            new_gy_cell = (F_row0 * dHy0 + G_row0 * dHy1
                           + F_row1 * dHy2 + G_row1 * dHy3)

            new_field[i, j] = new_f
            # Convert gradients back from cell-coordinate units to physical units.
            new_grad_field[i, j] = ti.Vector([new_gx_cell / self.dx,
                                              new_gy_cell / self.dx])


# ---------------------------------------------------------------------------
# SCHEME 7 — Hybrid WENO5 velocity + passive Lagrangian particles rho
# ---------------------------------------------------------------------------
# Exclusive fields:
#   self.n_particles    = self.res * self.res
#   self.particle_pos   = ti.Vector.field(2, float, shape=(self.n_particles,))
#   self.particle_weight = ti.field(float, shape=(self.n_particles,))
#
# fill_dye() dispatched to _fill_dye_particles_kernel for scheme 7.
# init_patterns() and init_from_image() called _init_particles_from_rho().
#
# Dispatch in step():
#   elif self.advection_scheme == 7:
#       self.advect_particles_rk2()
#       self.rho.fill(0.0)
#       self.splat_particles_to_rho()
#       self.step_weno(self.vel, self.vel_1, self.vel_2, self.new_vel, self.dq_vel)
#       self.vel.copy_from(self.new_vel)

class _ParticleArchive:
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


# ---------------------------------------------------------------------------
# CMM — Removed render modes and MacCormack backward-map advection path
# ---------------------------------------------------------------------------
# These were sub-options within scheme 8 (WENO+CMM) that were removed when
# CMM was simplified to bilinear rendering + WENO5 map advection only.
#
# Removed exclusive fields:
#   self.predict_backward_map = ti.Vector.field(2, float, shape=(self.res, self.res))
#   self.new_backward_map     = ti.Vector.field(2, float, shape=(self.res, self.res))
#     (new_backward_map is kept; only the MacCormack path that wrote to it is removed)
#
# Removed instance attributes:
#   self.cmm_render_mode          = 2   # 0 bilinear, 1 catmull-rom, 2 monotone-cubic
#   self.cmm_use_weno_map_advection = True
#
# MacCormack backward-map path (was gated on cmm_use_weno_map_advection == False):
#   self.advect_maccormack_predict(self.backward_map, self.predict_backward_map)
#   self.advect_maccormack_correct(self.backward_map, self.predict_backward_map,
#                                   self.new_backward_map)
#
# Removed cubic sampling helpers (used only by the two removed render kernels):
#   cubic_sample()          -- Catmull-Rom 4x4 bicubic tensor product
#   monotone_cubic_sample() -- PCHIP-style monotone cubic tensor product

class _CMMRenderArchive:
    @ti.kernel
    def _render_dye_bicubic(self):
        """Render variant: Catmull-Rom bicubic sampling at X(x). Sharper."""
        for i, j in self.rho:
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            src = ti.Vector([cx, cy]) + self.backward_map[i, j]
            self.rho[i, j] = self.cubic_sample(self.rho_source,
                                                src.x / self.dx - 0.5,
                                                src.y / self.dx - 0.5)

    @ti.kernel
    def _render_dye_monotone_cubic(self):
        """Render variant: PCHIP-style monotone cubic at X(x). Cubic-
        crispness without Catmull-Rom's ringing dips."""
        for i, j in self.rho:
            cx = (i + 0.5) * self.dx
            cy = (j + 0.5) * self.dx
            src = ti.Vector([cx, cy]) + self.backward_map[i, j]
            self.rho[i, j] = self.monotone_cubic_sample(self.rho_source,
                                                        src.x / self.dx - 0.5,
                                                        src.y / self.dx - 0.5)

    # cubic_sample and monotone_cubic_sample are not reproduced here because they
    # are long (50-150 lines each) and their full text is preserved in git history.
    # The render kernels above are sufficient context for any future revival.
    # See commits prior to the scheme-cleanup commit for the full implementations.
