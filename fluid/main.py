import taichi as ti
import numpy as np
from simulation import FluidSimulation, SimulationConfig
from video_recorder import VideoRecorder

def main():
    config = SimulationConfig()
    config.res = 512
    config.sharpen_strength = 0e-6

    # Dye initialization
    config.init_type = 'patterns'
    # config.init_type = 'image'

    # Boundary conditions:
    # - 'periodic'
    # - 'wall' and 'absorbing': same for the fluid itself, different for dye:
    #       absorbing zeros out dye reaching the boundary
    # - 'open': dye region surrounded by clean fluid at zero pressure,
    #      velocity unconstrained at boundary
    config.bc_type = 'open'
    # config.bc_type = 'absorbing'
    config.bc_type = 'periodic'

    config.vorticity_confinement_strength = 0.0

    sim = FluidSimulation(config)
    if config.init_type == 'patterns':
        sim.init_patterns()
    elif config.init_type == 'image':
        sim.init_from_image("./lenna.png")

    gui = ti.GUI("2D Fluid Simulation Demo", res=(config.res, config.res))

    # Simple UI state
    advection_names = {
        4: "WENO-5",
        8: "WENO-5 + Bidirectional CMM (bilinear)",
        9: "WENO-Z",
        10: "TENO5",
    }

    print("Controls:")
    print("  Mouse Left: Add Force (Drag)")
    print("  Mouse Right: Add Dye")
    print("  Key 1: WENO-5")
    print("  Key 2: WENO-Z")
    print("  Key 3: TENO5")
    print("  Key 4: WENO-5 + Bidirectional CMM (bilinear)")
    print("  Key R: Reset Patterns")
    print("  Key F: Apply force to bottom half (held)")
    print("  Key B: Toggle dye gravity (persistent)")
    print("  Key G: Apply image gradient force (gradual)")
    print("  Key D: Apply dye gradient force (gradual/dynamic)")
    print("  Key V: Toggle dye vortex (persistent)")
    print("  Key C: Toggle dye radial (persistent)")
    print("  Key P: Toggle pressure solver (Jacobi ↔ FFT, periodic BC only)")
    print("  Key K: Decrease vorticity confinement strength")
    print("  Key L: Increase vorticity confinement strength")
    print("  Key M: Toggle video recording (writes to ./recordings/)")

    prev_mouse = None
    DEFAULT_FORCE = 3.0  # default coefficient strength when toggling a force on

    # Video recorder: starts in the stopped state. The 'm' key toggles it.
    # One frame per rendered tick is appended at the configured fps, so
    # playback is realtime-ish relative to the user's interactive session.
    recorder = VideoRecorder(output_dir="recordings", fps=30)

    try:
        while gui.running:
            # Handle events
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == '1':
                    sim.advection_scheme = 4
                elif gui.event.key == '2':
                    sim.advection_scheme = 9
                elif gui.event.key == '3':
                    sim.advection_scheme = 10
                elif gui.event.key == '4':
                    sim.advection_scheme = 8
                elif gui.event.key == 'r':
                    sim.time = 0.0
                    if config.init_type == 'patterns':
                        sim.init_patterns()
                    elif config.init_type == 'image':
                        sim.init_from_image("./lenna.png")
                elif gui.event.key == 'g':
                    sim.apply_image_gradient_torque("./lenna.png", scale=1.0, duration=0.05, blur_sigma=1.0)
                elif gui.event.key == 'd':
                    sim.apply_dye_gradient_torque(scale=0.1, duration=0.03)
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
                elif gui.event.key == 'p':
                    # Toggle pressure solver between Jacobi and FFT.
                    # FFT is exact for periodic BCs and silently ignored otherwise.
                    if sim.config.pressure_solver == 'jacobi':
                        sim.config.pressure_solver = 'fft'
                        print("Pressure solver: FFT (exact, periodic BC)")
                    else:
                        sim.config.pressure_solver = 'jacobi'
                        print("Pressure solver: Jacobi (100 iterations)")
                elif gui.event.key == 'k':
                    # Decrease vorticity confinement strength (min 0).
                    sim.config.vorticity_confinement_strength = max(
                        0.0, sim.config.vorticity_confinement_strength - 0.01)
                    print(f"Vorticity confinement: {sim.config.vorticity_confinement_strength:.3f}")
                elif gui.event.key == 'l':
                    # Increase vorticity confinement strength.
                    sim.config.vorticity_confinement_strength += 0.01
                    print(f"Vorticity confinement: {sim.config.vorticity_confinement_strength:.3f}")
                elif gui.event.key == 'm':
                    # Toggle video recording. Each press starts a new clip
                    # (with a fresh timestamped filename) or stops the
                    # in-progress one. The HUD overlay indicates state.
                    if recorder.is_recording:
                        path = recorder.stop()
                        print(f"Stopped recording: {path} ({recorder.frames_written} frames)")
                    else:
                        path = recorder.start()
                        print(f"Recording to: {path}")

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

            # Render. Capture the same numpy array that gets handed to the
            # GUI so the recorded video matches exactly what the user sees,
            # minus the HUD overlays which are drawn separately by gui.text.
            dye_img = sim.rho.to_numpy()
            recorder.add_frame(dye_img)
            gui.set_image(dye_img)

            # Show current scheme info
            gui.text(f"Scheme: {advection_names[sim.advection_scheme]}", pos=(0.05, 0.95), color=0xFFFFFF)
            gui.text(f"Time: {sim.time:.2f}s", pos=(0.05, 0.90), color=0xFFFFFF)
            gui.text(f"dt: {sim.dt}, substeps: {substeps}", pos=(0.05, 0.85), color=0xFFFFFF)
            bc_label = config.bc_type if config.bc_type in ('periodic', 'open') else f"{config.bc_type} (slip={config.wall_slip:.1f})"
            gui.text(f"BC: {bc_label}", pos=(0.05, 0.80), color=0xFFFFFF)
            # CFL diagnostic: max(|u|) * dt / dx. Highlighted red when above 1.0;
            # high CFL reduces advection accuracy across all schemes.
            cfl = sim.max_cfl()
            cfl_color = 0xFF3333 if cfl > 1.0 else 0xFFFFFF
            gui.text(f"max CFL: {cfl:.2f}", pos=(0.05, 0.75), color=cfl_color)
            # Pressure solver and vorticity confinement state.
            solver_label = sim.config.pressure_solver.upper()
            if sim.config.pressure_solver == 'fft' and (sim.bc_wall or sim.bc_open):
                solver_label += " (fallback: Jacobi)"
            gui.text(f"Pressure: {solver_label}  [P to toggle]", pos=(0.05, 0.70), color=0xFFFFFF)
            vc_str = f"{sim.config.vorticity_confinement_strength:.2f}"
            gui.text(f"Vortex confinement: {vc_str}  [K / L]", pos=(0.05, 0.65), color=0xFFFFFF)

            # Recording indicator. Drawn after set_image so it appears in the
            # GUI window only; never enters the recorded video.
            if recorder.is_recording:
                gui.text(f"REC ● {recorder.frames_written}f", pos=(0.80, 0.95), color=0xFF3333)

            gui.show()
    finally:
        # Flush any in-progress recording on shutdown (normal exit or
        # exception) so the MP4 trailer is written and the file is playable.
        final_path = recorder.stop()
        if final_path is not None:
            print(f"Stopped recording on exit: {final_path} ({recorder.frames_written} frames)")

if __name__ == "__main__":
    main()