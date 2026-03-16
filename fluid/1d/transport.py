import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# Parameters
res = 512
dx = 1.0 / res
dt = 0.0006
velocity = 0.5

# Fields
q_upwind = ti.field(dtype=float, shape=res)
new_q_upwind = ti.field(dtype=float, shape=res)

q_semi_lag = ti.field(dtype=float, shape=res)
new_q_semi_lag = ti.field(dtype=float, shape=res)

q_upwind2 = ti.field(dtype=float, shape=res)
new_q_upwind2 = ti.field(dtype=float, shape=res)

q_semi_lag2 = ti.field(dtype=float, shape=res)
new_q_semi_lag2 = ti.field(dtype=float, shape=res)

q_spectral = ti.field(dtype=float, shape=res)

q_tvd = ti.field(dtype=float, shape=res)
new_q_tvd = ti.field(dtype=float, shape=res)

q_weno = ti.field(dtype=float, shape=res)
q_weno_1 = ti.field(dtype=float, shape=res)
q_weno_2 = ti.field(dtype=float, shape=res)
dq_weno = ti.field(dtype=float, shape=res)
new_q_weno = ti.field(dtype=float, shape=res)

@ti.kernel
def init():
    for i in q_upwind:
        x = i * dx
        if 0.3 < x < 0.5:
            q_upwind[i] = 1.0
            q_semi_lag[i] = 1.0
            q_upwind2[i] = 1.0
            q_semi_lag2[i] = 1.0
            q_spectral[i] = 1.0
            q_tvd[i] = 1.0
            q_weno[i] = 1.0
        else:
            q_upwind[i] = 0.0
            q_semi_lag[i] = 0.0
            q_upwind2[i] = 0.0
            q_semi_lag2[i] = 0.0
            q_spectral[i] = 0.0
            q_tvd[i] = 0.0
            q_weno[i] = 0.0

@ti.kernel
def step_upwind():
    for i in q_upwind:
        im1 = (i - 1) % res
        # Upwind scheme (assuming u > 0)
        new_q_upwind[i] = q_upwind[i] - velocity * (dt / dx) * (q_upwind[i] - q_upwind[im1])

@ti.kernel
def step_semi_lag():
    for i in q_semi_lag:
        # Trace back
        x_prev = i * dx - velocity * dt

        # Periodic boundaries
        x_prev = x_prev % 1.0
        if x_prev < 0:
            x_prev += 1.0

        # Interpolate
        idx = x_prev / dx
        idx_0 = int(ti.floor(idx))
        idx_1 = (idx_0 + 1) % res
        weight_1 = idx - idx_0
        weight_0 = 1.0 - weight_1

        # Wrap idx_0
        idx_0 = idx_0 % res

        new_q_semi_lag[i] = weight_0 * q_semi_lag[idx_0] + weight_1 * q_semi_lag[idx_1]

@ti.kernel
def step_upwind2():
    for i in q_upwind2:
        im1 = (i - 1) % res
        im2 = (i - 2) % res
        # 2nd order upwind (assuming u > 0)
        # q_x approx (3*q_i - 4*q_{i-1} + q_{i-2}) / (2*dx)
        # Note: can be oscillatory
        grad = (3.0 * q_upwind2[i] - 4.0 * q_upwind2[im1] + q_upwind2[im2]) / (2.0 * dx)
        new_q_upwind2[i] = q_upwind2[i] - velocity * dt * grad

@ti.kernel
def step_semi_lag2():
    for i in q_semi_lag2:
        # Trace back
        x_prev = i * dx - velocity * dt

        # Periodic boundaries
        x_prev = x_prev % 1.0
        if x_prev < 0:
            x_prev += 1.0

        # Interpolate
        idx = x_prev / dx
        idx_0 = int(ti.floor(idx))
        t = idx - idx_0

        # Catmull-Rom Cubic Spline Interpolation
        p0 = q_semi_lag2[(idx_0 - 1) % res]
        p1 = q_semi_lag2[idx_0 % res]
        p2 = q_semi_lag2[(idx_0 + 1) % res]
        p3 = q_semi_lag2[(idx_0 + 2) % res]

        new_q_semi_lag2[i] = 0.5 * (
            2.0 * p1 +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
        )

@ti.func
def minmod(a, b):
    """
    Minmod flux limiter function.
    Returns the argument with the smallest absolute value if they have the same sign.
    Returns 0 if they have opposite signs (extrema detection).

    Mathematical basis:
    It acts as a non-linear switch. If the slopes (differences) 'a' and 'b' have the same sign,
    it picks the smaller one to prevent overshooting near sharp gradients.
    If they have opposite signs, it indicates a local extremum, so it returns 0,
    effectively reverting the scheme to the stable 1st-order upwind method at that point.
    """
    res = 0.0
    if a * b > 0.0:
        if ti.abs(a) < ti.abs(b):
            res = a
        else:
            res = b
    return res

@ti.kernel
def step_tvd():
    """
    Solves the advection equation using a Total Variation Diminishing (TVD) scheme.

    Mathematical basis:
    It uses a high-resolution flux formulation: F_{i+1/2} = F_{low} + limiter * (F_{high} - F_{low})
    Here, F_{low} is the 1st-order upwind flux, and F_{high} corresponds to the 2nd-order Lax-Wendroff flux.
    The limiter (Minmod) ensures that the total variation of the solution does not increase,
    preventing the spurious oscillations (Gibbs phenomenon) that purely 2nd-order schemes produce
    near discontinuities (like the sharp edges of our square wave).

    c = u * dt / dx (Courant number)
    The extra term represents the anti-diffusive correction limited by the 'minmod' function
    based on consecutive spatial gradients.
    """
    c = velocity * dt / dx
    for i in q_tvd:
        im1 = (i - 1) % res
        im2 = (i - 2) % res
        ip1 = (i + 1) % res

        # Backward differences at consecutive locations
        dq_i_plus_half = q_tvd[ip1] - q_tvd[i]
        dq_i_minus_half = q_tvd[i] - q_tvd[im1]
        dq_i_minus_3_half = q_tvd[im1] - q_tvd[im2]

        # Flux at the right cell boundary (i+1/2) using upwind + limited anti-diffusive correction
        flux_i_plus_half = velocity * q_tvd[i] + 0.5 * velocity * (1.0 - c) * minmod(dq_i_plus_half, dq_i_minus_half)
        # Flux at the left cell boundary (i-1/2)
        flux_i_minus_half = velocity * q_tvd[im1] + 0.5 * velocity * (1.0 - c) * minmod(dq_i_minus_half, dq_i_minus_3_half)

        # Update using Conservative Finite-Volume formulation
        new_q_tvd[i] = q_tvd[i] - (dt / dx) * (flux_i_plus_half - flux_i_minus_half)

@ti.func
def weno5_reconstruct(v1, v2, v3, v4, v5):
    """
    WENO-5 Spatial Reconstruction for q_{i+1/2}^-
    Assuming velocity > 0, we use a left-biased 5-point stencil (v1..v5).
    """
    eps = 1e-6

    # Polynomials (3rd order approximations on 3-point substencils)
    p0 = (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0
    p1 = (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0
    p2 = (2.0 * v3 + 5.0 * v4 - v5) / 6.0

    # Smoothness indicators (Jiang and Shu)
    beta0 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3)**2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3)**2
    beta1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4)**2 + 0.25 * (v2 - v4)**2
    beta2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5)**2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5)**2

    # Linear weights combined with smoothness indicators to get nonlinear weights
    alpha0 = 0.1 / (eps + beta0)**2
    alpha1 = 0.6 / (eps + beta1)**2
    alpha2 = 0.3 / (eps + beta2)**2

    sum_alpha = alpha0 + alpha1 + alpha2

    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha

    return w0 * p0 + w1 * p1 + w2 * p2

@ti.kernel
def weno_rhs(q_in: ti.template()):
    for i in q_in:
        im3 = (i - 3) % res
        im2 = (i - 2) % res
        im1 = (i - 1) % res
        ip1 = (i + 1) % res
        ip2 = (i + 2) % res

        # Reconstruct state at right boundary (i+1/2) left side
        q_R = weno5_reconstruct(q_in[im2], q_in[im1], q_in[i], q_in[ip1], q_in[ip2])
        flux_R = velocity * q_R

        # Reconstruct state at left boundary (i-1/2) left side -> shift indices by -1
        q_L = weno5_reconstruct(q_in[im3], q_in[im2], q_in[im1], q_in[i], q_in[ip1])
        flux_L = velocity * q_L

        dq_weno[i] = -(flux_R - flux_L) / dx

@ti.kernel
def weno_rk3_step1():
    for i in q_weno:
        q_weno_1[i] = q_weno[i] + dt * dq_weno[i]

@ti.kernel
def weno_rk3_step2():
    for i in q_weno:
        q_weno_2[i] = 0.75 * q_weno[i] + 0.25 * q_weno_1[i] + 0.25 * dt * dq_weno[i]

@ti.kernel
def weno_rk3_step3():
    for i in q_weno:
        new_q_weno[i] = (1.0 / 3.0) * q_weno[i] + (2.0 / 3.0) * q_weno_2[i] + (2.0 / 3.0) * dt * dq_weno[i]

def step_weno():
    """
    Solves advection using the 5th order Weighted Essentially Non-Oscillatory scheme,
    integrated in time using the 3rd order Strong Stability Preserving Runge-Kutta (SSP-RK3) method.
    """
    weno_rhs(q_weno)
    weno_rk3_step1()

    weno_rhs(q_weno_1)
    weno_rk3_step2()

    weno_rhs(q_weno_2)
    weno_rk3_step3()

@ti.kernel
def copy_fields():
    for i in q_upwind:
        q_upwind[i] = new_q_upwind[i]
        q_semi_lag[i] = new_q_semi_lag[i]
        q_upwind2[i] = new_q_upwind2[i]
        q_semi_lag2[i] = new_q_semi_lag2[i]
        q_tvd[i] = new_q_tvd[i]
        q_weno[i] = new_q_weno[i]

def main():
    init()
    gui = ti.GUI("1D Transport (Comparing Schemes)", res=(800, 600))

    pos_upwind = np.zeros((res, 2), dtype=np.float32)
    pos_semi = np.zeros((res, 2), dtype=np.float32)
    pos_upwind2 = np.zeros((res, 2), dtype=np.float32)
    pos_semi2 = np.zeros((res, 2), dtype=np.float32)
    pos_spectral = np.zeros((res, 2), dtype=np.float32)
    pos_tvd = np.zeros((res, 2), dtype=np.float32)
    pos_weno = np.zeros((res, 2), dtype=np.float32)

    # The x coordinates are fixed
    xs = np.linspace(0, 1, res)
    pos_upwind[:, 0] = xs
    pos_semi[:, 0] = xs
    pos_upwind2[:, 0] = xs
    pos_semi2[:, 0] = xs
    pos_spectral[:, 0] = xs
    pos_tvd[:, 0] = xs
    pos_weno[:, 0] = xs

    steps_per_frame = 10

    # Spectral method precomputations
    frequencies = np.fft.fftfreq(res, d=dx)
    k = 2 * np.pi * frequencies
    phase_shift = np.exp(-1j * velocity * dt * k)

    q_spec_np = q_spectral.to_numpy()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'r':
                init()
                q_spec_np = q_spectral.to_numpy()

        for _ in range(steps_per_frame):
            step_upwind()
            step_semi_lag()
            step_upwind2()
            step_semi_lag2()
            step_tvd()
            step_weno()
            copy_fields()

            # Step spectral method in frequency domain
            q_hat = np.fft.fft(q_spec_np)
            q_hat *= phase_shift
            q_spec_np = np.real(np.fft.ifft(q_hat))

        q_up = q_upwind.to_numpy()
        q_sl = q_semi_lag.to_numpy()
        q_up2 = q_upwind2.to_numpy()
        q_sl2 = q_semi_lag2.to_numpy()
        q_t = q_tvd.to_numpy()
        q_w = q_weno.to_numpy()

        # Scale and offset for visualization
        # Range [0, 1] maps to [0.1, 0.9] of the screen height
        pos_upwind[:, 1] = 0.1 + q_up * 0.8
        pos_semi[:, 1] = 0.1 + q_sl * 0.8
        pos_upwind2[:, 1] = 0.1 + q_up2 * 0.8
        pos_semi2[:, 1] = 0.1 + q_sl2 * 0.8
        pos_spectral[:, 1] = 0.1 + q_spec_np * 0.8
        pos_tvd[:, 1] = 0.1 + q_t * 0.8
        pos_weno[:, 1] = 0.1 + q_w * 0.8

        gui.circles(pos_upwind, radius=2, color=0xFF0000)  # Red for Upwind (1st order)
        # gui.circles(pos_semi, radius=2, color=0x0000FF)    # Blue for Semi-Lag (1st order)
        # gui.circles(pos_upwind2, radius=2, color=0x00FF00) # Green for Upwind (2nd order)
        # gui.circles(pos_semi2, radius=2, color=0xFFFF00)   # Yellow for Semi-Lag (2nd order cubic)
        # gui.circles(pos_spectral, radius=2, color=0xFF00FF) # Magenta for Spectral
        gui.circles(pos_tvd, radius=2, color=0x00FFFF)     # Cyan for TVD
        gui.circles(pos_weno, radius=2, color=0xFF8800)    # Orange for WENO-5

        gui.text("Red: Upwind (1st Order)", pos=(0.05, 0.95), color=0xFF0000)
        gui.text("Blue: Semi-Lagrangian (1st Order)", pos=(0.05, 0.90), color=0x0000FF)
        gui.text("Green: Upwind (2nd Order)", pos=(0.05, 0.85), color=0x00FF00)
        gui.text("Yellow: Semi-Lagrangian (Cubic)", pos=(0.05, 0.80), color=0xFFFF00)
        gui.text("Magenta: Spectral Method", pos=(0.05, 0.75), color=0xFF00FF)
        gui.text("Cyan: TVD (Minmod)", pos=(0.05, 0.70), color=0x00FFFF)
        gui.text("Orange: WENO-5", pos=(0.05, 0.65), color=0xFF8800)

        gui.show()

if __name__ == "__main__":
    main()
