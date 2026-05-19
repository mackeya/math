"""
Headless scheme-comparison utility for the 2D fluid simulation.

Runs the same initial dye field through identical persistent forcing for a
fixed number of steps under each named configuration, then saves a still
PNG of the resulting dye field for each. Also produces a single contact-
sheet PNG with all variants tiled and labelled.

Intentionally a scratch tool: parameters and the list of configurations
live as constants at the top of the file -- edit and re-run.

Usage:
    cd fluid/
    python3 compare_schemes.py

Output:
    comparisons/<prefix>_<label>.png  -- one PNG per configuration
    comparisons/<prefix>_contact.png  -- tiled labelled side-by-side
"""

import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulation import FluidSimulation, SimulationConfig


# -- Tunable parameters --------------------------------------------------

RES = 512                       # simulation grid resolution
DT = 3e-4                       # time step
INIT = 'image'                  # 'patterns' or 'image'
INIT = 'patterns'
IMAGE_PATH = './lenna.png'
N_STEPS = 2000                  # simulated time = N_STEPS * DT
# BC_TYPE = 'periodic'           # matches main.py default
BC_TYPE = 'absorbing'           # matches main.py default
PERSISTENT_TORQUE = 0.0         # equivalent of pressing 'v' at startup
PERSISTENT_BUOYANCY = 3.0       # equivalent of pressing 'b'
PERSISTENT_RADIAL = 0.0         # equivalent of pressing 'c'
PRESSURE_SOLVER = 'fft'         # jacobi, fft

OUTPUT_DIR = 'comparisons'
OUTPUT_PREFIX = f't{N_STEPS * DT:.3f}s'

# Configuration list. Each entry is (label, setup_fn). setup_fn takes a
# constructed sim and is responsible for setting the advection scheme and
# any scheme-specific toggles. The label is used in the PNG filename and
# in the contact-sheet tile header.

CONFIGURATIONS = [
    ("WENO5",     lambda s: setattr(s, 'advection_scheme', 4)),
    ("WENO-Z",    lambda s: setattr(s, 'advection_scheme', 9)),
    ("TENO5",     lambda s: setattr(s, 'advection_scheme', 10)),
    ("CMM",       lambda s: setattr(s, 'advection_scheme', 8)),
]


# -- Image helpers -------------------------------------------------------

def _rho_to_image_array(rho):
    """
    Converts a (res, res) sim rho field into a (res, res) image-orientation
    uint8 array. Matches init_from_image's inverse transform so saved PNGs
    look upright (origin at top-left, y increasing downward).
    """
    img = np.clip(rho, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    # sim stores rho[i, j] = (col, row) after init_from_image's flipud+T;
    # inverse is T then flipud.
    return np.flipud(img.T)


def _save_grayscale(rho, path):
    """Save a single sim rho field as an 8-bit grayscale PNG."""
    Image.fromarray(_rho_to_image_array(rho), mode='L').save(path)


def _save_contact_sheet(frames, path, tile_label_height=28, gap=8):
    """
    Tile all frames into a roughly-square grid and save as one PNG. Each
    tile is preceded by a black band with the label printed in white.
    """
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    tile_w, tile_h = RES, RES
    cell_w = tile_w + gap
    cell_h = tile_h + tile_label_height + gap
    sheet_w = cols * cell_w + gap
    sheet_h = rows * cell_h + gap

    sheet = Image.new('L', (sheet_w, sheet_h), color=0)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, rho) in enumerate(frames):
        col = idx % cols
        row = idx // cols
        x0 = gap + col * cell_w
        y0 = gap + row * cell_h
        # Label band
        draw.text((x0 + 4, y0 + 4), label, fill=255, font=font)
        # Image tile (paste underneath the label band)
        tile_img = Image.fromarray(_rho_to_image_array(rho), mode='L')
        sheet.paste(tile_img, (x0, y0 + tile_label_height))

    sheet.save(path)


# -- Main loop -----------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim_time = N_STEPS * DT
    print(f'Comparing {len(CONFIGURATIONS)} schemes at res={RES}, '
          f'{N_STEPS} steps = {sim_time:.3f}s simulated time')
    print(f'Init: {INIT}, BC: {BC_TYPE}, '
          f'persistent torque={PERSISTENT_TORQUE}, '
          f'buoyancy={PERSISTENT_BUOYANCY}, radial={PERSISTENT_RADIAL}, '
          f'pressure_solver={PRESSURE_SOLVER}')
    print()

    frames = []
    for label, setup_fn in CONFIGURATIONS:
        # Each config gets its own sim instance. Per-config memory is
        # modest (~10 MB at res=512); seven configs ~= 70 MB.
        sim = FluidSimulation(SimulationConfig(
            res=RES, dt=DT, bc_type=BC_TYPE,
            torque_coeff=PERSISTENT_TORQUE,
            buoyancy_coeff=PERSISTENT_BUOYANCY,
            radial_coeff=PERSISTENT_RADIAL,
            pressure_solver=PRESSURE_SOLVER,
        ))
        if INIT == 'image':
            sim.init_from_image(IMAGE_PATH)
        elif INIT == 'patterns':
            sim.init_patterns()
        else:
            raise ValueError(f'unknown INIT: {INIT}')
        # Set scheme + toggles AFTER init (init may overwrite some state
        # such as the CMM source snapshot, which is what we want).
        setup_fn(sim)

        t0 = time.time()
        for _ in range(N_STEPS):
            sim.step()
        elapsed = time.time() - t0

        rho = sim.rho.to_numpy()
        print(f'  {label:>30} | {N_STEPS}/{elapsed:5.2f}s '
              f'= {N_STEPS / elapsed:5.0f} steps/sec | '
              f'rho range [{rho.min():+.3f}, {rho.max():+.3f}]')

        out_path = os.path.join(
            OUTPUT_DIR,
            f'{OUTPUT_PREFIX}_{label.replace(" ", "_").replace("+", "")}.png',
        )
        _save_grayscale(rho, out_path)
        frames.append((label, rho))

    contact_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_contact.png')
    _save_contact_sheet(frames, contact_path)
    print()
    print(f'Wrote {len(frames)} individual PNGs and {contact_path}')


if __name__ == '__main__':
    main()
