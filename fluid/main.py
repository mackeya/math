import taichi as ti
import numpy as np
from simulation import FluidSimulation, SimulationConfig

def main():
    config = SimulationConfig()
    config.res = 512
    # Dye initialization
    config.init_type = 'patterns'
    config.init_type = 'image'
    # Boundary conditions: 'periodic', 'wall', 'absorbing', or 'open'
    config.bc_type = 'open'
    # config.bc_type = 'absorbing'
    # config.bc_type = 'wall'
    # config.bc_type = 'periodic'

    sim = FluidSimulation(config)
    if config.init_type == 'patterns':
        sim.init_patterns()
    elif config.init_type == 'image':
        sim.init_from_image("./lenna.png")

    gui = ti.GUI("2D Fluid Simulation Demo", res=(config.res, config.res))

    # Simple UI state
    advection_names = {0: "Semi-Lagrangian", 4: "WENO-5"}

    print("Controls:")
    print("  Mouse Left: Add Force (Drag)")
    print("  Mouse Right: Add Dye")
    print("  Key 1: Semi-Lagrangian (Stable, Smooth)")
    print("  Key 2: WENO-5")
    print("  Key R: Reset Patterns")
    print("  Key F: Apply force to bottom half")
    print("  Key B: Toggle dye gravity (persistent)")
    print("  Key G: Apply image gradient force (gradual)")
    print("  Key D: Apply dye gradient force (gradual/dynamic)")
    print("  Key V: Toggle dye vortex (persistent)")
    print("  Key C: Toggle dye radial (persistent)")

    prev_mouse = None
    DEFAULT_FORCE = 3.0  # default coefficient strength when toggling a force on

    while gui.running:
        # Handle events
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == '1':
                sim.advection_scheme = 0
            elif gui.event.key == '2':
                sim.advection_scheme = 4
            elif gui.event.key == 'r':
                sim.time = 0.0
                if config.init_type == 'patterns':
                    sim.init_patterns()
                elif config.init_type == 'image':
                    sim.init_from_image("./lenna.png")
            elif gui.event.key == 'g':
                sim.apply_image_gradient_torque("./lenna.png", scale=1.0, duration=0.1, blur_sigma=1.0)
            elif gui.event.key == 'd':
                sim.apply_dye_gradient_torque(scale=0.1, duration=0.1)
            elif gui.event.key == 'b':
                # Toggle buoyancy: set coefficient to default or zero it off
                if sim.config.buoyancy_coeff < 1.0:
                    sim.config.buoyancy_coeff = DEFAULT_FORCE
                else:
                    sim.config.buoyancy_coeff = 0.0
            elif gui.event.key == 'v':
                # Toggle torque
                if sim.config.torque_coeff < 1.0:
                    sim.config.torque_coeff = DEFAULT_FORCE
                else:
                    sim.config.torque_coeff = 0.0
            elif gui.event.key == 'c':
                # Toggle radial
                if sim.config.radial_coeff < 1.0:
                    sim.config.radial_coeff = DEFAULT_FORCE
                else:
                    sim.config.radial_coeff = 0.0

        # Handle mouse interaction
        curr_mouse = gui.get_cursor_pos()
        if prev_mouse is None:
            prev_mouse = curr_mouse

        if gui.is_pressed(ti.GUI.LMB):
            # Apply force proportional to mouse movement
            dx, dy = curr_mouse[0] - prev_mouse[0], curr_mouse[1] - prev_mouse[1]
            sim.apply_force(curr_mouse[0], curr_mouse[1], dx * 40000, dy * 40000, 0.03)

        if gui.is_pressed(ti.GUI.RMB):
            sim.fill_dye(curr_mouse[0], curr_mouse[1], 0.02, 5.0)

        if gui.is_pressed('f'):
            sim.apply_bottom_force(1000.0, 200.0)

        prev_mouse = curr_mouse

        # Step simulation with substepping for stability
        substeps = 10
        for _ in range(substeps):
            sim.step()

        # Render
        dye_img = sim.rho.to_numpy()
        gui.set_image(dye_img)

        # Show current scheme info
        gui.text(f"Scheme: {advection_names[sim.advection_scheme]}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Time: {sim.time:.2f}s", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"dt: {sim.dt}, substeps: {substeps}", pos=(0.05, 0.85), color=0xFFFFFF)
        bc_label = config.bc_type if config.bc_type in ('periodic', 'open') else f"{config.bc_type} (slip={config.wall_slip:.1f})"
        gui.text(f"BC: {bc_label}", pos=(0.05, 0.80), color=0xFFFFFF)

        gui.show()

if __name__ == "__main__":
    main()