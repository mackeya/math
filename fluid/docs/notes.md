# TODO:
-- sharpening is interesting but is applied even where the velocity field is zero, creating problems over time. should only be applied when there's flow, or maybe it makes no sense

Help me brainstorm some more artistic effects which could be applied in this fluid simulation. Anything which would create a beautiful or trippy visual effect is fair game. To give you the general idea of what I'm going for, a few general areas which could be pursued include:
-- more or varied types of forces that can be applied to the fluid, along the lines of those currently in _apply_persistent_force_kernel, e.g. applying force based on perlin or simplex noise etc
-- the flow could advect an alpha channel, magnification, or some other effect rather than the dye representing the image itself
-- the simulation could be upgraded to handle all 3 color channels, and advect them in different ways

What suggestions do you have?

1. The "Coordinate Map" Trick (Lagrangian Texture Mapping)
    Going further with this: submap?
3. Non-Linear "Sharpening" Kernels (The Anti-Diffusion Term)
    A common method is to add a term to your advection that pushes values toward the nearest "limit" (either 0 or 255):
    ∂t∂D​=⋯+λ⋅∇2D⋅(1−∣∇D∣)