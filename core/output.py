"""Virtual camera output thread — writes processed frames to v4l2loopback."""

import logging
import threading
from typing import Optional

import cv2
import numpy as np
import pyfakewebcam  # type: ignore[import-untyped]

from core.processor import ProcessorThread

logger = logging.getLogger(__name__)


class OutputThread(threading.Thread):
    """Reads processed frames and writes them to a pyfakewebcam device."""

    def __init__(
        self,
        processor: ProcessorThread,
        device: str = "/dev/video10",
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__(daemon=True)
        self._processor: ProcessorThread = processor
        self._device: str = device
        self._width: int = width
        self._height: int = height
        self._stop_event: threading.Event = threading.Event()
        self._fake_cam: Optional[pyfakewebcam.FakeWebcam] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the output loop to stop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            "OutputThread started — writing to %s (%dx%d)",
            self._device,
            self._width,
            self._height,
        )
        try:
            self._fake_cam = pyfakewebcam.FakeWebcam(
                self._device, self._width, self._height
            )
        except Exception:
            logger.exception("Failed to open fake webcam device %s", self._device)
            return

        while not self._stop_event.is_set():
            frame = self._processor.processed_frame
            if frame is None:
                self._stop_event.wait(0.01)
                continue

            try:
                # Ensure frame matches expected dimensions
                if (frame.shape[1], frame.shape[0]) != (self._width, self._height):
                    frame = cv2.resize(frame, (self._width, self._height))

                # pyfakewebcam expects RGB
                rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._fake_cam.schedule_frame(rgb_frame)
            except Exception:
                logger.exception("Error writing frame to virtual camera")

            # Small sleep to limit output rate (~30 fps ceiling)
            self._stop_event.wait(0.033)

        logger.info("OutputThread stopped")
