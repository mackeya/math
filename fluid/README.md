# 2D Fluid Simulation Demo

This project implements a 2D fluid simulation using the **Taichi** library, focusing on comparing different numerical schemes for advection.

## Features
- **GPU Acceleration**: High-performance simulation using Taichi's JIT compiler.
- **Multiple Advection Schemes**:
    - **Semi-Lagrangian**: Unconditionally stable, suitable for interactive applications, but prone to numerical smoothing.
    - **Upwind**: First-order stable, but very diffusive (sharp features blur quickly).
    - **MacCormack**: Second-order accurate predictor-corrector; sharper than Upwind but can show dispersive oscillations ("ringing").
- **Real-time Interaction**: Click and drag to add dye or apply force.

## How to Run
1.  Ensure you have `taichi` and `numpy` installed:
    ```bash
    pip install taichi numpy
    ```
2.  Run the main script:
    ```bash
    python fluid/main.py
    ```

## Controls
- **Mouse Left Button**: Add dye (density).
- **Mouse Right Button**: Add force.
- **Key 1**: Switch to Semi-Lagrangian scheme.
- **Key 2**: Switch to Upwind scheme.
- **Key 3**: Switch to MacCormack scheme.
- **Key R**: Reset simulation.

## Implementation Details
The simulation follows Chorin's projection method:
1.  **Advection**: Update the velocity and density fields using one of the available schemes.
2.  **External Forces**: Apply forces from mouse interaction.
3.  **Pressure Projection**:
    - Compute the divergence of the velocity field.
    - Solve the Poisson equation for pressure using Jacobi iteration.
    - Subtract the pressure gradient to enforce zero divergence (incompressibility).
