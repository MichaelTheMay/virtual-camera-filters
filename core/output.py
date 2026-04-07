"""Virtual camera output thread — writes processed frames to v4l2loopback."""

import ctypes
import fcntl
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

from core.processor import ProcessorThread

logger = logging.getLogger(__name__)

# v4l2 constants
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_PIX_FMT_YUYV = 0x56595559
V4L2_COLORSPACE_JPEG = 7
VIDIOC_S_FMT = 0xC0D05605


class _v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
    ]


class _v4l2_format(ctypes.Structure):
    class _fmt(ctypes.Union):
        class _pix(ctypes.Structure):
            _fields_ = _v4l2_pix_format._fields_
        _fields_ = [("pix", _pix), ("raw", ctypes.c_ubyte * 200)]
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _fmt)]


class V4L2Writer:
    """Write YUYV frames to a v4l2loopback device (no pyfakewebcam dependency)."""

    def __init__(self, device: str, width: int, height: int) -> None:
        self._fd = os.open(device, os.O_WRONLY | os.O_SYNC)
        self._width = width
        self._height = height

        fmt = _v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        fmt.fmt.pix.bytesperline = width * 2
        fmt.fmt.pix.sizeimage = width * height * 2
        fmt.fmt.pix.colorspace = V4L2_COLORSPACE_JPEG
        fcntl.ioctl(self._fd, VIDIOC_S_FMT, fmt)

        self._buffer = np.zeros((height, width * 2), dtype=np.uint8)

    def write_frame(self, rgb_frame: np.ndarray) -> None:
        """Convert RGB frame to YUYV and write to device."""
        yuv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YUV)
        for i in range(self._height):
            self._buffer[i, ::2] = yuv[i, :, 0]
            self._buffer[i, 1::4] = yuv[i, ::2, 1]
            self._buffer[i, 3::4] = yuv[i, ::2, 2]
        os.write(self._fd, self._buffer.tobytes())

    def close(self) -> None:
        os.close(self._fd)


class OutputThread(threading.Thread):
    """Reads processed frames and writes them to a v4l2loopback device."""

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
        self._writer: Optional[V4L2Writer] = None

    def stop(self) -> None:
        """Signal the output loop to stop."""
        self._stop_event.set()

    def run(self) -> None:
        logger.info(
            "OutputThread started — writing to %s (%dx%d)",
            self._device, self._width, self._height,
        )
        try:
            self._writer = V4L2Writer(self._device, self._width, self._height)
        except Exception:
            logger.exception("Failed to open v4l2 device %s", self._device)
            return

        while not self._stop_event.is_set():
            frame = self._processor.processed_frame
            if frame is None:
                self._stop_event.wait(0.01)
                continue

            try:
                if (frame.shape[1], frame.shape[0]) != (self._width, self._height):
                    frame = cv2.resize(frame, (self._width, self._height))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._writer.write_frame(rgb_frame)
            except Exception:
                logger.exception("Error writing frame to virtual camera")

            self._stop_event.wait(0.033)

        if self._writer:
            self._writer.close()
        logger.info("OutputThread stopped")
