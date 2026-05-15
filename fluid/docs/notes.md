# TODO:
-- sharpening is interesting but is applied even where the velocity field is zero, creating problems over time. should only be applied when there's flow, or maybe it makes no sense


## For avoiding numerical diffusion
-- go lagrangian to some degree
-- try vortex confinement

Gemini suggests
1. The "Coordinate Map" Trick (Lagrangian Texture Mapping)
2. Lagrangian Particles (The "FLIP" Approach)
3. Non-Linear "Sharpening" Kernels (The Anti-Diffusion Term)
    A common method is to add a term to your advection that pushes values toward the nearest "limit" (either 0 or 255):
    ∂t∂D​=⋯+λ⋅∇2D⋅(1−∣∇D∣)
