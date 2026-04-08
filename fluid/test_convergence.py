# type: ignore
# pylint: skip-file
import taichi as ti
import numpy as np
import time
from simulation import FluidSimulation

def get_exact_solution(res, t, u, v, ic_type='smooth'):
    """
    Returns the exact solution for a 2D advection equation with constant velocity.
    """
    x = np.linspace(0, 1, res, endpoint=False)
    y = np.linspace(0, 1, res, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Calculate shifted coordinates for periodic advection
    X_shifted = (X - u * t) % 1.0
    Y_shifted = (Y - v * t) % 1.0

    if ic_type == 'smooth':
        return np.sin(2 * np.pi * X_shifted) * np.sin(2 * np.pi * Y_shifted)
    elif ic_type == 'square':
        # Square wave / Box discontinuity
        mask = (X_shifted > 0.4) & (X_shifted < 0.6) & (Y_shifted > 0.4) & (Y_shifted < 0.6)
        return mask.astype(np.float64)
    elif ic_type == 'triangle':
        # Triangle / Cone (discontinuity in derivative)
        r = np.sqrt((X_shifted - 0.5)**2 + (Y_shifted - 0.5)**2)
        return np.maximum(0, 1 - 5 * r)
    return np.zeros((res, res))

@ti.kernel
def set_uniform_velocity(sim: ti.template(), u: float, v: float):
    for i, j in sim.vel:
        sim.vel[i, j] = ti.Vector([u, v])

@ti.kernel
def set_initial_condition(sim: ti.template(), ic_type: int):
    # ic_type: 0 for smooth, 1 for square, 2 for triangle
    for i, j in sim.rho:
        x = i * sim.dx
        y = j * sim.dx
        if ic_type == 0:
            sim.rho[i, j] = ti.sin(2 * ti.math.pi * x) * ti.sin(2 * ti.math.pi * y)
        elif ic_type == 1:
            if 0.4 < x < 0.6 and 0.4 < y < 0.6:
                sim.rho[i, j] = 1.0
            else:
                sim.rho[i, j] = 0.0
        elif ic_type == 2:
            r = ti.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            sim.rho[i, j] = ti.max(0.0, 1.0 - 5.0 * r)

def run_test(scheme_id, res, T, dt, ic_type='smooth'):
    sim = FluidSimulation(res, dt=dt)
    sim.advection_scheme = scheme_id

    # Mapping ic_type string to int for kernel
    ic_map = {'smooth': 0, 'square': 1, 'triangle': 2}

    # Initialize fields
    set_initial_condition(sim, ic_map[ic_type])
    u_vel, v_vel = 1.0, 1.0
    set_uniform_velocity(sim, u_vel, v_vel)

    steps = int(T / dt)
    for _ in range(steps):
        # Only advect, skip projection as we have fixed velocity
        if sim.advection_scheme == 0:
            sim.advect_semi_lagrangian(sim.rho, sim.new_rho)
            sim.rho.copy_from(sim.new_rho)
        elif sim.advection_scheme == 1:
            sim.advect_upwind(sim.rho, sim.new_rho)
            sim.rho.copy_from(sim.new_rho)
        elif sim.advection_scheme == 2:
            sim.advect_maccormack_step1(sim.rho, sim.new_rho)
            sim.advect_maccormack_step2(sim.rho, sim.new_rho, sim.rho)
        elif sim.advection_scheme == 3:
            sim.advect_tvd(sim.rho, sim.new_rho)
            sim.rho.copy_from(sim.new_rho)
        elif sim.advection_scheme == 4:
            sim.step_weno(sim.rho, sim.rho_1, sim.rho_2, sim.new_rho, sim.dq_rho)
            sim.rho.copy_from(sim.new_rho)

    # Compute error
    rho_num = sim.rho.to_numpy()
    rho_exact = get_exact_solution(res, T, u_vel, v_vel, ic_type)

    l2_error = np.sqrt(np.mean((rho_num - rho_exact)**2))
    return l2_error

def main():
    schemes = {
        0: "Semi-Lagrangian",
        1: "Upwind",
        2: "MacCormack",
        3: "TVD (Minmod)",
        4: "WENO5"
    }

    resolutions = [32, 64, 128, 256]
    ic_types = ["smooth", "square", "triangle"]
    T = 0.1

    for ic in ic_types:
        print(f"\nTest Case: {ic.upper()} Initial Condition")
        print(f"{'Scheme':<20} | {'Res':<5} | {'L2 Error':<10} | {'Rate':<5}")
        print("-" * 50)

        for sid, name in schemes.items():
            errors = []
            for i, res in enumerate(resolutions):
                # dt = 0.05 / res for WENO5, 0.1 / res others
                dt = 0.05 / res if sid == 4 else 0.1 / res
                err = run_test(sid, res, T, dt, ic_type=ic)
                errors.append(err)

                rate = ""
                if i > 0:
                    rate = np.log2(errors[i-1] / errors[i])
                    rate = f"{rate:.2f}"

                print(f"{name:<20} | {res:<5} | {err:<10.3e} | {rate}")
            print("-" * 50)

if __name__ == "__main__":
    main()
