# TODO:
-- sharpening is interesting but is applied even where the velocity field is zero, creating problems over time. should only be applied when there's flow, or maybe it makes no sense


## For avoiding numerical diffusion
-- try vortex confinement

Gemini suggests
1. The "Coordinate Map" Trick (Lagrangian Texture Mapping)
    Holy shit it's really good!
    Going further with this: submap
3. Non-Linear "Sharpening" Kernels (The Anti-Diffusion Term)
    A common method is to add a term to your advection that pushes values toward the nearest "limit" (either 0 or 255):
    ∂t∂D​=⋯+λ⋅∇2D⋅(1−∣∇D∣)


Let's pause and consider the approach. Here is an example implementation of the Chebyshev iteration in MATLAB. Your implementation should match it. After translating this implementation in to Python and Taichi as as appropriate for the codebase, test it to see how it works.

function [x] = SolChebyshev002(A, b, x0, iterNum, lMax, lMin)

  d = (lMax + lMin) / 2;
  c = (lMax - lMin) / 2;
  preCond = eye(size(A)); % Preconditioner
  x = x0;
  r = b - A * x;

  for i = 1:iterNum % size(A, 1)
      z = linsolve(preCond, r);
      if (i == 1)
          p = z;
          alpha = 1/d;
      else if (i == 2)
          beta = (1/2) * (c * alpha)^2
          alpha = 1/(d - beta / alpha);
          p = z + beta * p;
      else
          beta = (c * alpha / 2)^2;
          alpha = 1/(d - beta / alpha);
          p = z + beta * p;
      end;

      x = x + alpha * p;
      r = b - A * x; %(= r - alpha * A * p)
      if (norm(r) < 1e-15), break; end; % stop if necessary
  end;
end