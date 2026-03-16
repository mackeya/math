import taichi as ti
import zarr
import numpy as np
import threading
import queue
import time
import sandpile_utils.utils as su

ti.init(arch=ti.metal)

# Simulation parameters
n = 1001  # Initial grid resolution
tot_grains = 16000000
canvas_res = 800
display_animation = False
save_data = True
if not display_animation and not save_data:
    print("Warning: Both display_animation and save_data are False. Nothing to do!")

grid = ti.field(dtype=ti.i32, shape=(n, n))
next_grid = ti.field(dtype=ti.i32, shape=(n, n))
colors = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))

def expand_grid(n, grid):
    old_n = n
    new_n = int(old_n * 1.1) # Expand by 10%
    # TODO: consider expand by fixed amount, since growth slows near
    # the end just when multiplicative increases get big
    # new_n = int(old_n + 200)
    print(f"Expanding grid: {old_n}x{old_n} -> {new_n}x{new_n}")

    # Create new fields
    new_grid = ti.field(dtype=ti.i32, shape=(new_n, new_n))
    new_next_grid = ti.field(dtype=ti.i32, shape=(new_n, new_n))

    # Calculate offsets to place old grid in the center
    offset = (new_n - old_n) // 2

    # Use a simple NumPy transfer for the resize (safe and easy)
    old_data = grid.to_numpy()
    new_data = np.zeros((new_n, new_n), dtype=np.int32)
    new_data[offset:offset+old_n, offset:offset+old_n] = old_data

    # Load into the new Taichi fields
    new_grid.from_numpy(new_data)

    return new_n, new_grid, new_next_grid

# Zarr setup -- absolute path example
if save_data:
    store = zarr.DirectoryStore(f'/Users/alan/Python/sandpile_data/infinite_grid_data/data{tot_grains}.zarr')
    root = zarr.group(store=store, overwrite=True)
    # Create a dataset to hold multiple frames: [frame_index, x, y]
    # Chunks should be tuned; (1, n, n) means one full grid per chunk
    z_array = root.create_dataset('history', shape=(0, n, n),
                                chunks=(1, n, n), dtype='i4')

    # Queue to pass data from Taichi (GPU/Main Thread) to Saver (CPU/Background)
    save_queue = queue.Queue()

    def saver_worker():
        """ This function runs in a separate thread. """
        print("Saver thread started.")
        while True:
            data = save_queue.get()
            if data is None: break  # Signal to stop

            # Append the new frame to the Zarr array
            current_idx = z_array.shape[0]
            z_array.resize(current_idx + 1, n, n)
            z_array[current_idx] = data

            print(f"Frame {current_idx} saved to Zarr.")
            save_queue.task_done()

@ti.kernel
def init_sand():
    grid[n // 2, n // 2] = tot_grains

@ti.kernel
def check_boundary(cgrid: ti.template(), n: ti.i32) -> ti.i32: # type: ignore
    # We check the top/bottom and left/right edges for any sand > 0
    hit = 0
    for i in range(n):
        if cgrid[i, 10] > 0 or cgrid[i, n-11] > 0: hit = 1
        if cgrid[10, i] > 0 or cgrid[n-11, i] > 0: hit = 1
    return hit

# Track whether the grid was modified in the last step
was_modified = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def update_colors():
    for i, j in grid:
        val = grid[i, j]
        if val == 0:
            colors[i, j] = ti.Vector([0.05, 0.05, 0.1]) # Dark Blue
        elif val == 1:
            colors[i, j] = ti.Vector([0.3, 0.0, 0.5])  # Indigo
        elif val == 2:
            colors[i, j] = ti.Vector([0.0, 0.8, 0.8])  # Cyan
        elif val == 3:
            colors[i, j] = ti.Vector([1.0, 0.0, 0.6])  # Pink
        else:
            # colors[i, j] = ti.Vector([1.0, 1.0, 1.0])  # White
            # colors[i, j] = ti.Vector([252, 186, 3]) / 255.0 * 0.5
            colors[i, j] = ti.Vector([1.0, 0, 0])


# Start the background thread
if save_data:
    thread = threading.Thread(target=saver_worker, daemon=True)
    thread.start()

# Setup canvas and window (optional)
window = None
canvas = None
if display_animation:
    window = ti.ui.Window("M4 Powered Sandpile", (canvas_res, canvas_res))
    canvas = window.get_canvas()

init_sand()

# Main loop
step = 0
ti.sync()
prev_time = time.perf_counter()
start_time = prev_time

# Initial save of the starting state
if save_data:
    snapshot = grid.to_numpy()
    save_queue.put(snapshot)

running = True
topple_func = su.topple_kernel
try:
    while running:
        # Run multiple steps per frame for speed
        for _ in range(1000):
            # Fancty toppling logic with pointer swap -- a little faster
            next_grid.fill(0)
            topple_func(grid, next_grid, was_modified)
            grid, next_grid = next_grid, grid

        if was_modified[None] == 0:
            print("Grid stabilized. Stopping simulation.")
            running = False
            break

        # Expand grid if necessary
        if check_boundary(grid, n):
            print(f"boundary at {n} hit on step {step}")
            n, grid, next_grid = expand_grid(n, grid)

        # Save occasionally
        if (step % 100 == 0) and (was_modified[None] == 1):
            if save_data:
                # 1. Capture GPU data to CPU (the 'bottleneck' step)
                snapshot = grid.to_numpy()
                # 2. Push to queue and immediately continue simulation
                save_queue.put(snapshot)

            # Performance logging
            ti.sync() # Wait for all topples to actually finish on the M4
            curr_time = time.perf_counter()
            print(f"Step {step}: {curr_time - prev_time:.2f} seconds elapsed.")
            prev_time = curr_time

        step += 1

        # Rendering (only if display_animation is True)
        if display_animation:
            update_colors()
            canvas.set_image(colors)
            window.show()


except KeyboardInterrupt:
    print("\n[Ctrl+C] detected. Initiating graceful shutdown...")

finally:
    # This block runs NO MATTER WHAT (Natural finish or Ctrl+C)
    # One more loop to finish stabilizing the grid
    running = True
    while running:
        for _ in range(1000):
            next_grid.fill(0)
            topple_func(grid, next_grid, was_modified)
            grid, next_grid = next_grid, grid
        if was_modified[None] == 0:
            print("Final stabilization complete.")
            running = False

        step += 1
        if step % 100 == 0:
            print(f"Final stabilization step {step}...")

    print(f"Cleaning up at step {step}...")

    if save_data:
        # 1. Take a final snapshot of the current state
        ti.sync()
        final_snapshot = grid.to_numpy()
        save_queue.put(final_snapshot)

        # 2. Signal the saver thread that we are done
        save_queue.put(None)

        # 3. Wait for the thread to finish writing the remaining queue to the SSD
        print("Writing remaining data to Zarr... please do not force close.")
        thread.join()

    print("Done. All data safely stored.")
    end_time = time.perf_counter()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")