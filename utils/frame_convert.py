"""Frame format conversion utilities for OpenCV / Qt interop."""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage, QPixmap


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to BGR."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def frame_to_qpixmap(
    frame: np.ndarray, target_size: Optional[QSize] = None
) -> QPixmap:
    """Convert a BGR numpy array to a :class:`QPixmap`.

    Parameters
    ----------
    frame:
        An OpenCV image in BGR colour order.
    target_size:
        If provided the pixmap is scaled to this size using smooth
        transformation while keeping the aspect ratio.
    """
    rgb = bgr_to_rgb(frame)
    h, w, ch = rgb.shape
    bytes_per_line: int = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimage)

    if target_size is not None:
        from PyQt6.QtCore import Qt

        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    return pixmap
