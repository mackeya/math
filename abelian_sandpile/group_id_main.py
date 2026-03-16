import taichi as ti
import zarr
import numpy as np
import threading
import queue
import time
from PIL import Image
import sandpile_utils.utils as su

ti.init(arch=ti.gpu)


### Simulation parameters
N = 455
topples_per_frame = 16
save_data = False
color_scheme = 'custom'  # 'custom' or 'grayscale'


### Simulation variables
cells = ti.field(dtype=ti.i32, shape=(N, N))
next_cells = ti.field(dtype=ti.i32, shape=(N, N))

# Track whether the grid (cells) was modified in the last step
was_modified = ti.field(dtype=ti.i32, shape=())
was_modified[None] = 1

# Zarr data storage setup
if save_data:
    store = zarr.DirectoryStore('/Users/alan/Python/sandpile_data/group_id_data/data.zarr')
    root = zarr.group(store=store, overwrite=True)
    z_array = root.create_dataset('history', shape=(0, N, N),
                                chunks=(1, N, N), dtype='i4')

# Queue to pass data from Taichi (GPU/Main Thread) to Saver (CPU/Background)
save_queue = queue.Queue()


@ti.kernel
def initialize():
    for i, j in cells:
        # # Setting to initialize for video v0, N = 501
        # cells[i, j] = 5
        # # Diagonal lines
        # if i-j == 4*N//5 or j-i == 4*N//5:
        #     cells[i, j] = 100
        # # Center region
        # c = 110
        # if np.abs(i-N//2) < c and np.abs(j-N//2) < c:
        #     cells[i, j] = 0

        # Maybe better than video
        cells[i, j] = 5
        # Diagonal lines
        if np.abs(i-j) == 400:
            cells[i, j] = 100
        # Center region
        c = 100
        if np.abs(i-N//2) < c and np.abs(j-N//2) < c:
            cells[i, j] = 0

        # Other initialization ideas:
        # # Border
        # border_size = 50
        # if (i < border_size or i >= N - border_size) or (j < border_size or j >= N - border_size):
        #     cells[i, j] = 7
        # # Diagonal lines
        # if np.abs(i-j) == 450:
        #     # cells[i, j] = 50 + np.abs(i-j)
        #     cells[i, j] = 50
        # # Center region
        # c = 100
        # if np.abs(i-N//2) < c and np.abs(j-N//2) < c:
        #     cells[i, j] = 0
        # # Center cell
        # if (i == N//2 and j == N//2):
        #     cells[i, j] = 1000

        # # Initialize from photo
        # lennarr = su.load_as_grayscale("/Users/alan/Python/sandpile_data/group_id_data/lenna.png")
        # resized = np.array(Image.fromarray(lennarr).resize((N, N), Image.NEAREST))
        # resized = np.fliplr(np.transpose(resized))
        # init = np.array(np.round((resized / 20))).astype(np.int32)
        # cells.from_numpy(init)

def saver_worker():
    """ This function runs in a separate thread. """
    print("Saver thread started.")
    while True:
        data = save_queue.get()
        if data is None: break  # Signal to stop

        # Append the new frame to the Zarr array
        current_idx = z_array.shape[0]
        z_array.resize(current_idx + 1, N, N)
        z_array[current_idx] = data

        print(f"Frame {current_idx} saved to Zarr.")
        save_queue.task_done()

@ti.kernel
def render(pixels: ti.template()): # type: ignore
    """Map cell values to colors for visualization."""
    for i, j in cells:
        val = cells[i, j]

        # Color mapping (Normalized 0.0 to 1.0)
        color = ti.Vector([0.0, 0.0, 0.0])
        if val == 0: color = ti.Vector([0.1, 0.1, 0.1]) # Dark
        elif val == 1: color = ti.Vector([0.2, 0.6, 0.8]) # Blue
        elif val == 2: color = ti.Vector([0.9, 0.8, 0.1]) # Yellow
        elif val == 3: color = ti.Vector([0.99, 0.0, 0.0]) # Red
        # elif val >= 4: color = ti.Vector([245, 2, 245]) / 255

        pixels[i, j] = color

@ti.kernel
def render_grayscale(pixels: ti.template()): # type: ignore
    """Map cell values to grayscale for visualization."""
    for i, j in cells:
        gray = ti.cast(cells[i, j], ti.f32) / 25
        pixels[i, j] = gray


### Simulation
# Start the background data-saving thread
thread = threading.Thread(target=saver_worker, daemon=True)
thread.start()

# Visualization Setup
gui = ti.GUI("Sandpile finite grid", res=(N, N))
initialize()
# initialize_lena()

step = 0
ti.sync()
prev_time = time.perf_counter()

snapshot = cells.to_numpy()
save_queue.put(snapshot)

while gui.running:
    # Sleep to avoid overwhelming rendering and crashing
    time.sleep(0.005)
    # Run multiple steps per frame
    for _ in range(topples_per_frame):
        su.topple(cells, next_cells, was_modified)

    if save_data and was_modified[None] == 1:
        # 1. Capture GPU data to CPU
        snapshot = cells.to_numpy()
        # 2. Push to queue and immediately continue simulation
        save_queue.put(snapshot)

    if step % 100 == 0 and was_modified[None] == 1:
        print(f"Step {step}")
        ti.sync() # Wait for all topples to actually finish on the M4
        curr_time = time.perf_counter()
        print(f"Step {step}: {curr_time - prev_time:.2f} seconds elapsed.")
        prev_time = curr_time

    step += 1

    if color_scheme == 'custom':
        # Display values 0-3 with specific colors
        pixels = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
        render(pixels)
        gui.set_image(pixels)
    elif color_scheme == 'grayscale':
        # Display grid in grayscale
        pixels = ti.field(dtype=ti.f32, shape=(N, N))
        render_grayscale(pixels)
        gui.set_image(pixels)
    else:
        raise ValueError(f"Unknown color scheme: {color_scheme}")

    gui.show()


# Save final snapshot and clean up
if save_data:
    snapshot = cells.to_numpy()
    save_queue.put(snapshot)
    save_queue.put(None)
    thread.join()