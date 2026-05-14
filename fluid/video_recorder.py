"""
Video recording for the interactive fluid simulation.

Streams simulation frames to an MP4 file via imageio + ffmpeg. The recorder is
intentionally decoupled from Taichi: it accepts a plain numpy array each frame,
so it can be reused from any entry point (interactive GUI, offline script,
test harness).

Typical usage:
    recorder = VideoRecorder(output_dir="recordings", fps=30)
    recorder.start()                          # opens recordings/fluid_<timestamp>.mp4
    while running:
        recorder.add_frame(sim.rho.to_numpy())
    recorder.stop()                           # flushes and closes the file
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import imageio.v2 as imageio


class VideoRecorder:
    """
    Streams a sequence of grayscale dye-density frames to an MP4 file.

    The recorder is created in a stopped state. Call ``start()`` to open a new
    timestamped MP4, ``add_frame()`` once per simulation tick while recording,
    and ``stop()`` to flush and close the file. Multiple start/stop cycles are
    supported within a single recorder instance — each cycle produces a
    separate output file.

    Frames arrive in Taichi's (i, j) layout (i = x to the right, j = y upward,
    float values nominally in [0, 1]) and are reoriented and quantized to
    standard 8-bit RGB before being handed to ffmpeg.
    """

    def __init__(self, output_dir: str = "recordings", fps: int = 30,
                 quality: int = 8):
        """
        Configure where and how recordings will be written.

        Parameters
        ----------
        output_dir : str
            Directory where MP4 files will be saved. Created lazily on the
            first call to ``start()``; not touched at construction time.
        fps : int
            Output video frame rate. The recorder writes one video frame per
            call to ``add_frame()``, so ``fps`` controls playback speed
            relative to the user's interactive session, not capture rate.
        quality : int
            Encoder quality on imageio's 0 (worst) - 10 (best) scale. 8 is a
            good visually-lossless-ish default with reasonable file sizes.
        """
        # Directory where MP4s are written; created lazily on start().
        self.output_dir = Path(output_dir)
        # Output frame rate baked into the MP4 container.
        self.fps = fps
        # Encoder quality; passed straight through to imageio's writer.
        self.quality = quality
        # imageio writer when active, None when stopped.
        self._writer = None
        # Path of the in-progress (or most recently completed) recording.
        self.current_path: Path | None = None
        # Frames written so far in the current recording (0 when stopped).
        self.frames_written = 0

    @property
    def is_recording(self) -> bool:
        """Return True iff a writer is currently open and accepting frames."""
        return self._writer is not None

    def start(self) -> Path:
        """
        Open a new MP4 file with a timestamped name and begin recording.

        If a recording is already in progress, this is a no-op and returns the
        path of the existing file. Otherwise, the output directory is created
        if needed and a fresh writer is opened with broad-compatibility
        settings (libx264 + yuv420p).

        Returns
        -------
        Path
            The path of the file that frames will be written to.
        """
        # Idempotent: if already recording, just report the active path.
        if self.is_recording:
            return self.current_path
        # Create the output directory on demand so we don't litter empty dirs
        # for users who never actually record anything.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Timestamp the filename so successive clips in one session don't
        # collide.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.output_dir / f"fluid_{stamp}.mp4"
        # macro_block_size=1 lets us pass arbitrary dimensions (e.g. 512) to
        # x264 without imageio padding the frame to a 16-pixel multiple.
        # imageio's libx264 path already defaults to yuv420p, which is what
        # every common player (QuickTime, browser, VLC) expects.
        self._writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec="libx264",
            quality=self.quality,
            macro_block_size=1,
        )
        self.current_path = path
        self.frames_written = 0
        return path

    def add_frame(self, dye_field: np.ndarray) -> None:
        """
        Append one frame to the current recording.

        No-op when not recording, so the caller can unconditionally invoke
        this every simulation tick.

        Parameters
        ----------
        dye_field : np.ndarray
            A 2D float array (typically ``sim.rho.to_numpy()``) with shape
            ``(res, res)`` and values nominally in ``[0, 1]``. Indexed in
            Taichi's (i, j) convention with i = x rightward and j = y upward;
            this method handles the reorientation to standard image layout.
        """
        # Silently do nothing when not recording, so callers can always invoke
        # add_frame() without guarding on is_recording.
        if not self.is_recording:
            return
        # Reorient from Taichi's (i=x-right, j=y-up) to standard image layout
        # (row=y-down, col=x-right). Transposing swaps axes so the first axis
        # becomes j (rows, but still y-up); flipud then flips j to y-down.
        oriented = np.flipud(dye_field.T)
        # Quantize float [0, 1] -> uint8 [0, 255]. Clip in case advection has
        # produced slight overshoot beyond [0, 1].
        gray = np.clip(oriented * 255.0, 0, 255).astype(np.uint8)
        # Stack the single grayscale channel into RGB. libx264 with yuv420p
        # expects a 3-channel input; a grayscale-only frame would be rejected
        # or implicitly converted in a way that isn't worth depending on.
        rgb = np.stack([gray, gray, gray], axis=-1)
        self._writer.append_data(rgb)
        self.frames_written += 1

    def stop(self) -> Path | None:
        """
        Flush any buffered frames and close the current recording.

        Safe to call when not recording (returns ``None``). Always safe to
        call from a ``finally`` block on shutdown.

        Returns
        -------
        Path or None
            The path of the file just finalized, or ``None`` if no recording
            was in progress.
        """
        # No-op when not recording; callers can put this in a finally block
        # without worrying about double-close.
        if not self.is_recording:
            return None
        path = self.current_path
        # Closing the writer flushes ffmpeg's internal buffer and writes the
        # MP4 trailer; without this the file may be unplayable.
        self._writer.close()
        self._writer = None
        self.current_path = None
        return path
