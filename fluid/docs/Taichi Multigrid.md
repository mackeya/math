# **Design Document: Geometric Multigrid (GMG) Pressure Solver in Taichi**

## **1\. Overview & Objective**

This document outlines the architectural and mathematical implementation details for a high-performance **Geometric Multigrid (GMG)** Poisson solver within an existing 2D incompressible fluid mechanics simulation.  
To maximize the visual capabilities of high-order advection schemes like **TENO-5**, the velocity field must be kept strictly divergence-free ($\\nabla \\cdot \\mathbf{u} \= 0$). While relaxation methods like Jacobi scale poorly ($O(N^2)$) and fail to clear low-frequency divergence errors, this GMG implementation delivers $O(N)$ linear scaling. It resolves global pressure waves efficiently across a geometric grid hierarchy, driving the divergence residual to near-machine epsilon in a handful of iterations.

## **2\. Mathematical Framework**

During Chorin’s splitting method (the pressure-projection step), we solve the Poisson equation for pressure:

$$\\nabla^2 p \= \\frac{\\rho}{\\Delta t} \\nabla \\cdot \\mathbf{u}^\* \\quad \\equiv \\quad A p \= b$$  
Where $u^\*$ is the intermediate divergent velocity field, and $A$ is the discrete Laplacian operator.

### **2.1 The Multigrid V-Cycle**

Instead of clearing low-frequency errors on a fine grid (which requires hundreds of iterations), GMG shifts the smooth, low-frequency error to a coarser grid, where it reappears as high-frequency error and can be cleared instantly via simple relaxation.

Fine Grid (L0)  \--- Smooth \---\> Restrict \-----------------------\> Prolongate \---\> Smooth \---\> Corrected L0  
                     \\                                              /  
Coarse Grid (L1)      \--- Smooth \---\> Restrict \-------\> Prolongate \---  
                                       \\                  /  
Coarsest Grid (L2)                      \---- Base Solve \---

## **3\. Structural & JIT Compile-Once Architecture**

Taichi utilizes a Just-In-Time (JIT) compiler. Dynamically selecting fields using runtime indices (e.g., grids\[level\]) forces the compiler to generate unoptimized kernels or trigger compilation storms during the execution loop.

### **3.1 Structural Parameterization**

To achieve optimal GPU performance, the multi-level grid tree must be pre-allocated as distinct structural templates. This allows Taichi to build explicit, highly optimized static code pipelines for each level exactly once at startup.

Python  
import taichi as ti

class GeometricMultigrid:  
    def \_\_init\_\_(self, top\_res: int, num\_levels: int):  
        self.N \= top\_res  
        self.levels \= num\_levels  
          
        \# Pre-allocate explicit structural hierarchies  
        \# Level 0 is finest, Level (num\_levels-1) is coarsest  
        self.p \= \[ti.field(dtype=ti.f32, shape=(self.N \>\> l, self.N \>\> l)) for l in range(num\_levels)\]  
        self.r \= \[ti.field(dtype=ti.f32, shape=(self.N \>\> l, self.N \>\> l)) for l in range(num\_levels)\]  
        self.b \= \[ti.field(dtype=ti.f32, shape=(self.N \>\> l, self.N \>\> l)) for l in range(num\_levels)\]

## **4\. Boundary Condition Invariance: Clamped Stencils**

A solid box domain requires a **Zero-Gradient (Neumann)** boundary condition:

$$\\frac{\\partial p}{\\partial n} \= 0$$  
To prevent boundary misalignment across coarse levels—which injects massive artificial high-frequency divergence back into the system—all kernels must treat boundaries using a **clamped stencil invariant**.

Python  
@ti.func  
def safe\_sample(field: ti.template(), i: ti.i32, j: ti.i32, nx: ti.i32, ny: ti.i32) \-\> ti.f32:  
    """  
    Implicitly enforces a Neumann boundary condition at any grid level   
    by clamping out-of-bound requests to the edge pixel.  
    """  
    clamped\_i \= ti.max(0, ti.min(i, nx \- 1))  
    clamped\_j \= ti.max(0, ti.min(j, ny \- 1))  
    return field\[clamped\_i, clamped\_j\]

## **5\. Kernel Specifications**

### **5.1 Smoother (Red-Black Gauss-Seidel)**

To maximize GPU thread execution and double relaxation efficiency over Jacobi, the smoothing pass uses a checkerboard layout.

Python  
@ti.kernel  
def smooth\_level(p\_f: ti.template(), b\_f: ti.template(), color: ti.i32):  
    \# Pass grid dimensions statically via metadata  
    nx, ny \= p\_f.shape\[0\], p\_f.shape\[1\]  
    dx \= 1.0 / nx  
      
    for i, j in p\_f:  
        if (i \+ j) % 2 \== color:  
            p\_left  \= safe\_sample(p\_f, i \- 1, j, nx, ny)  
            p\_right \= safe\_sample(p\_f, i \+ 1, j, nx, ny)  
            p\_down  \= safe\_sample(p\_f, i, j \- 1, nx, ny)  
            p\_up    \= safe\_sample(p\_f, i, j \+ 1, nx, ny)  
              
            \# Discrete Poisson relaxation  
            p\_f\[i, j\] \= 0.25 \* (p\_left \+ p\_right \+ p\_down \+ p\_up \- (dx \* dx) \* b\_f\[i, j\])

### **5.2 Residual Evaluation**

Computes the current algebraic error $r \= b \- Ap$ before downsampling.

Python  
@ti.kernel  
def compute\_residual(p\_f: ti.template(), b\_f: ti.template(), r\_f: ti.template()):  
    nx, ny \= p\_f.shape\[0\], p\_f.shape\[1\]  
    inv\_dx2 \= float(nx \* nx) \# 1 / dx^2  
      
    for i, j in p\_f:  
        p\_c     \= p\_f\[i, j\]  
        p\_left  \= safe\_sample(p\_f, i \- 1, j, nx, ny)  
        p\_right \= safe\_sample(p\_f, i \+ 1, j, nx, ny)  
        p\_down  \= safe\_sample(p\_f, i, j \- 1, nx, ny)  
        p\_up    \= safe\_sample(p\_f, i, j \+ 1, nx, ny)  
          
        laplacian \= (p\_left \+ p\_right \+ p\_down \+ p\_up \- 4.0 \* p\_c) \* inv\_dx2  
        r\_f\[i, j\] \= b\_f\[i, j\] \- laplacian

### **5.3 Restriction (Fine-to-Coarse)**

Transfers the residual from a fine grid to a coarse grid using a 4-pixel spatial block average.

Python  
@ti.kernel  
def restrict\_level(r\_fine: ti.template(), b\_coarse: ti.template()):  
    for i, j in b\_coarse:  
        \# Map coarse cell coordinates to the 4 matching fine cells  
        fi, fj \= i \* 2, j \* 2  
          
        v00 \= r\_fine\[fi,     fj\]  
        v10 \= r\_fine\[fi \+ 1, fj\]  
        v01 \= r\_fine\[fi,     fj \+ 1\]  
        v11 \= r\_fine\[fi \+ 1, fj \+ 1\]  
          
        b\_coarse\[i, j\] \= 0.25 \* (v00 \+ v10 \+ v01 \+ v11)

### **5.4 Prolongation and Accumulation (Coarse-to-Fine)**

Interpolates the coarse error correction back up to the fine grid using bilinear interpolation, adding it to the existing fine pressure field.

Python  
@ti.kernel  
def prolongate\_level(p\_coarse: ti.template(), p\_fine: ti.template()):  
    nc\_x, nc\_y \= p\_coarse.shape\[0\], p\_coarse.shape\[1\]  
      
    for i, j in p\_fine:  
        \# Calculate continuous fractional coordinates on the coarse grid  
        ci\_float \= (float(i) \+ 0.5) / 2.0 \- 0.5  
        cj\_float \= (float(j) \+ 0.5) / 2.0 \- 0.5  
          
        \# Base discrete index coordinates  
        ci \= ti.i32(ti.floor(ci\_float))  
        cj \= ti.i32(ti.floor(cj\_float))  
          
        \# Linear weights  
        wx \= ci\_float \- float(ci)  
        wy \= cj\_float \- float(cj)  
          
        \# Sample using the edge-clamping invariant  
        c00 \= safe\_sample(p\_coarse, ci,     cj,     nc\_x, nc\_y)  
        c10 \= safe\_sample(p\_coarse, ci \+ 1, cj,     nc\_x, nc\_y)  
        c01 \= safe\_sample(p\_coarse, ci,     cj \+ 1, nc\_x, nc\_y)  
        c11 \= safe\_sample(p\_coarse, ci \+ 1, cj \+ 1, nc\_x, nc\_y)  
          
        \# Bilinear interpolation product  
        correction \= (1.0 \- wx) \* (1.0 \- wy) \* c00 \+ \\  
                     wx         \* (1.0 \- wy) \* c10 \+ \\  
                     (1.0 \- wx) \* wy         \* c01 \+ \\  
                     wx         \* wy         \* c11  
                       
        p\_fine\[i, j\] \+= correction

## **6\. Execution Control Flow (Python Side)**

The orchestration of the system happens in standard Python loops. Because the fields are unrolled into lists, the sequential calls invoke compiled template variants without spawning any runtime JIT artifacts.

Python  
class GeometricMultigrid:  
    \# ... init fields ...

    def solve\_v\_cycle(self, divergence\_input: ti.template()):  
        \# Copy fine divergence directly into the L0 RHS buffer  
        self.b\[0\].copy\_from(divergence\_input)  
        self.p\[0\].fill(0.0) \# Reset finest initial guess

        \# \--- 1\. DOWNWARD PASS (Restriction) \---  
        for l in range(self.levels \- 1):  
            \# Pre-smoothing (2 iterations of Red-Black Gauss-Seidel)  
            smooth\_level(self.p\[l\], self.b\[l\], 0\) \# Red  
            smooth\_level(self.p\[l\], self.b\[l\], 1\) \# Black  
              
            \# Evaluate algebraic error  
            compute\_residual(self.p\[l\], self.b\[l\], self.r\[l\])  
              
            \# Downsample error to become the next level's RHS source  
            restrict\_level(self.r\[l\], self.b\[l \+ 1\])  
            self.p\[l \+ 1\].fill(0.0) \# Clear lower level guess

        \# \--- 2\. COARSEST BASE SOLVE \---  
        \# At deepest resolution (e.g., 16x16), run exhaustive relaxation  
        coarsest\_lvl \= self.levels \- 1  
        for \_ in range(32):  
            smooth\_level(self.p\[coarsest\_lvl\], self.b\[coarsest\_lvl\], 0\)  
            smooth\_level(self.p\[self.levels-1\], self.b\[self.levels-1\], 1\)

        \# \--- 3\. UPWARD PASS (Prolongation & Correction) \---  
        for l in range(self.levels \- 2, \-1, \-1):  
            \# Up-sample correction from (l+1) and accumulate onto level (l)  
            prolongate\_level(self.p\[l \+ 1\], self.p\[l\])  
              
            \# Post-smoothing to remove high-frequency interpolation errors  
            smooth\_level(self.p\[l\], self.b\[l\], 0\)  
            smooth\_level(self.p\[l\], self.b\[l\], 1\)

        \# Output solution is located inside self.p\[0\]

## **7\. Verification and Diagnostics Checklist**

To ensure the solver is operating at peak correctness, implement the following testing hooks:

* \[ \] **The Divergence Test:** Render a real-time viewport tracking ti.abs(b\_field \- Laplacian(p\_0)). The residual must decrease uniformly across the entire grid surface. If geometric artifacts manifest near the boundaries, the clamp invariants inside prolongate\_level or safe\_sample are tracking out-of-bounds indices incorrectly.  
* \[ \] **Two-Iteration Baseline:** Run the solver at a single, fixed level with an all-zero configuration. The Red-Black Gauss-Seidel kernel must match a standalone reference Jacobi pass byte-for-byte on its very first iteration step.  
* \[ \] **Compilation Validation:** Monitor the terminal output during execution. Taichi should run its structural JIT engine *exactly once* during the first execution frame. If compilation logging reappears during subsequent cycles, check that no runtime variables or non-templated primitives are leaking into the structural kernel headers.