import taichi as ti
from PIL import Image
import numpy as np


@ti.kernel
def topple(cells: ti.template(), next_cells: ti.template(), was_modified: ti.template()): # type: ignore
    # Reset the flag at the start of the kernel
    was_modified[None] = 0

    # Reset next_cells to current values
    for i, j in cells:
        next_cells[i, j] = cells[i, j]

    # Parallel toppling logic
    for i, j in cells:
        if cells[i, j] >= 4:
            ti.atomic_max(was_modified[None], 1)
            # We topple 4 grains
            next_cells[i, j] -= 4

            # Add grains to neighbors (handling boundaries)
            if i-1 >= 0:   ti.atomic_add(next_cells[i-1, j], 1)
            if i+1 < next_cells.shape[0]: ti.atomic_add(next_cells[i+1, j], 1)
            if j-1 >= 0:   ti.atomic_add(next_cells[i, j-1], 1)
            if j+1 < next_cells.shape[1]: ti.atomic_add(next_cells[i, j+1], 1)

    # Update main field
    for i, j in cells:
        cells[i, j] = next_cells[i, j]


@ti.kernel
def topple_multi(grid: ti.template(), next_grid: ti.template(), was_modified: ti.template()): # type: ignore
    # Reset the flag at the start of the kernel
    was_modified[None] = 0

    # Reset next_grid for the double-buffer logic
    for i, j in next_grid:
        next_grid[i, j] = grid[i, j]

    for i, j in grid:
        if grid[i, j] >= 4:
            ti.atomic_max(was_modified[None], 1)
            # Note: This is a parallel kernel.
            # We use atomic adds to prevent race conditions.

            # These next two lines appear to be a fancy way of only moving
            # grains away if there are more than 4.
            # This also moves many grains at once.
            num = grid[i, j] // 4
            next_grid[i, j] -= num * 4

            # Spread to neighbors
            if i + 1 < grid.shape[0]: ti.atomic_add(next_grid[i + 1, j], num)
            if i - 1 >= 0: ti.atomic_add(next_grid[i - 1, j], num)
            if j + 1 < grid.shape[1]: ti.atomic_add(next_grid[i, j + 1], num)
            if j - 1 >= 0: ti.atomic_add(next_grid[i, j - 1], num)

    # Swap buffers
    for i, j in grid:
        grid[i, j] = next_grid[i, j]


@ti.kernel
def topple_kernel(src: ti.template(), dst: ti.template(), was_modified: ti.template()): # type: ignore
    was_modified[None] = 0

    for i, j in src:
        out_grains = 0
        if src[i, j] >= 4:
            out_grains = (src[i, j] // 4)
            ti.atomic_max(was_modified[None], 1)

        # Atomic Update the destination
        ti.atomic_add(dst[i, j], src[i, j] - (out_grains * 4))

        if out_grains > 0:
            if i + 1 < src.shape[0]: ti.atomic_add(dst[i + 1, j], out_grains)
            if i - 1 >= 0:           ti.atomic_add(dst[i - 1, j], out_grains)
            if j + 1 < src.shape[1]: ti.atomic_add(dst[i, j + 1], out_grains)
            if j - 1 >= 0:           ti.atomic_add(dst[i, j - 1], out_grains)


@ti.kernel
def topple8(cells: ti.template(), next_cells: ti.template(), was_modified: ti.template()): # type: ignore
    # Topple function with diagonal spreading.
    # Reset the flag at the start of the kernel
    was_modified[None] = 0

    # Reset next_cells to current values
    for i, j in cells:
        next_cells[i, j] = cells[i, j]

    # Parallel toppling logic
    for i, j in cells:
        if cells[i, j] >= 8:
            ti.atomic_max(was_modified[None], 1)
            # We topple 8 grains
            next_cells[i, j] -= 8

            # Add grains to neighbors (handling boundaries)
            if i-1 >= 0:
                ti.atomic_add(next_cells[i-1, j], 1)
                if j-1 >= 0:   ti.atomic_add(next_cells[i-1, j-1], 1)
                if j+1 < next_cells.shape[1]: ti.atomic_add(next_cells[i-1, j+1], 1)
            if i+1 < next_cells.shape[0]:
                ti.atomic_add(next_cells[i+1, j], 1)
                if j-1 >= 0:   ti.atomic_add(next_cells[i+1, j-1], 1)
                if j+1 < next_cells.shape[1]: ti.atomic_add(next_cells[i+1, j+1], 1)
            if j-1 >= 0:   ti.atomic_add(next_cells[i, j-1], 1)
            if j+1 < next_cells.shape[1]: ti.atomic_add(next_cells[i, j+1], 1)

    # Update main field
    for i, j in cells:
        cells[i, j] = next_cells[i, j]


def load_as_grayscale(filepath):
    # 'L' mode stands for Luminance (8-bit pixels, black and white)
    img = Image.open(filepath).convert('L')

    # Convert to a standard NumPy array
    grid_np = np.array(img)

    print(f"Loaded grayscale image: {grid_np.shape}")
    return grid_np