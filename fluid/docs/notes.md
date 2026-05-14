# TODO:
-- maccormack looks nice but has obvious visual issues. see if we can fix it
-- sharpening is interesting but is applied even where the velocity field is zero, creating problems over time. should only be applied when there's flow, or maybe it makes no sense


## For avoiding numerical diffusion
Gemini suggests
1. The "Coordinate Map" Trick (Lagrangian Texture Mapping)
2. BFECC (Back-and-Forth Error Compensation and Correction)
3. Lagrangian Particles (The "FLIP" Approach)
4. Non-Linear "Sharpening" Kernels (The Anti-Diffusion Term)
    A common method is to add a term to your advection that pushes values toward the nearest "limit" (either 0 or 255):
    ∂t∂D​=⋯+λ⋅∇2D⋅(1−∣∇D∣)
