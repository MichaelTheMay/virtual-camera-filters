"""Webcam capture thread — reads frames from a video device into a deque."""

import collections
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CaptureThread(threading.Thread):
    """Continuously captures frames from a V4L2 video device.

    Frames are stored in a fixed-size deque and the most recent one is
    exposed via the :pyattr:`latest_frame` property (thread-safe).
    """

    def __init__(self, device_path: str = "/dev/video0") -> None:
        super().__init__(daemon=True)
        self._device_path: str = device_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=2)
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()

        # FPS tracking
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._fps_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame, or *None*."""
        with self._lock:
            return self._buffer[-1].copy() if self._buffer else None

    @property
    def fps(self) -> float:
        """Return the measured capture FPS."""
        return self._fps

    def set_device(self, path: str) -> None:
        """Switch to a different video device (takes effect on next loop iteration)."""
        logger.info("Capture device changing to %s", path)
        with self._lock:
            self._device_path = path
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("CaptureThread started for %s", self._device_path)
        self._fps_start_time = time.monotonic()

        while not self._stop_event.is_set():
            # (Re-)open capture if necessary
            if self._cap is None or not self._cap.isOpened():
                self._open_device()
                if self._cap is None:
                    # Wait a bit before retrying
                    self._stop_event.wait(1.0)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from %s", self._device_path)
                self._stop_event.wait(0.1)
                continue

            with self._lock:
                self._buffer.append(frame)

            # FPS calculation
            self._frame_count += 1
            elapsed = time.monotonic() - self._fps_start_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.monotonic()

        self._release()
        logger.info("CaptureThread stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_device(self) -> None:
        try:
            cap = cv2.VideoCapture(self._device_path)
            if cap.isOpened():
                self._cap = cap
                logger.info("Opened capture device %s", self._device_path)
            else:
                cap.release()
                logger.error("Cannot open device %s", self._device_path)
        except Exception:
            logger.exception("Error opening device %s", self._device_path)

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
