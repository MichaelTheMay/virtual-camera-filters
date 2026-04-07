"""Filter processing thread — pulls frames, applies filter chain, exposes result."""

import logging
import threading
from typing import Any, List, Optional, Protocol

import numpy as np

from core.capture import CaptureThread

logger = logging.getLogger(__name__)


class Filter(Protocol):
    """Minimal protocol every filter callable must satisfy."""

    enabled: bool

    def process(self, frame: np.ndarray) -> np.ndarray: ...


class ProcessorThread(threading.Thread):
    """Reads the latest captured frame, runs it through an ordered filter chain,
    and stores the processed result for downstream consumers."""

    def __init__(self, capture: CaptureThread) -> None:
        super().__init__(daemon=True)
        self._capture: CaptureThread = capture
        self._filters: List[Any] = []
        self._filters_lock: threading.Lock = threading.Lock()
        self._frame_lock: threading.Lock = threading.Lock()
        self._processed_frame: Optional[np.ndarray] = None
        self._stop_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def processed_frame(self) -> Optional[np.ndarray]:
        """Return the most recently processed frame, or *None*."""
        with self._frame_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

    def set_filters(self, filters: List[Any]) -> None:
        """Replace the current filter chain."""
        with self._filters_lock:
            self._filters = list(filters)
        logger.info("Filter chain updated (%d filters)", len(filters))

    def stop(self) -> None:
        """Signal the processing loop to stop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("ProcessorThread started")

        while not self._stop_event.is_set():
            frame = self._capture.latest_frame
            if frame is None:
                self._stop_event.wait(0.01)
                continue

            frame = self._apply_filters(frame)

            with self._frame_lock:
                self._processed_frame = frame

            # Yield to avoid busy-spinning
            self._stop_event.wait(0.001)

        logger.info("ProcessorThread stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filters(self, frame: np.ndarray) -> np.ndarray:
        with self._filters_lock:
            filters = list(self._filters)

        for filt in filters:
            if not getattr(filt, "enabled", False):
                continue
            try:
                frame = filt.process(frame)
            except Exception:
                logger.exception(
                    "Filter %r raised an exception — skipping",
                    getattr(filt, "__class__", filt),
                )
        return frame
