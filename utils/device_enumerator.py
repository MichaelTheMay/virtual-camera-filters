"""Video device enumeration — discovers V4L2 cameras on the system."""

import glob
import logging
import re
from typing import Dict, List

import cv2

logger = logging.getLogger(__name__)

_VIRTUAL_CAM_PATH: str = "/dev/video10"


def enumerate_cameras() -> List[Dict[str, str]]:
    """Return a sorted list of available V4L2 camera devices.

    Each entry is a dict with keys ``'name'`` and ``'path'``.
    The virtual camera device (``/dev/video10``) is excluded.
    """
    devices: List[Dict[str, str]] = []

    paths = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.search(r"\d+$", p).group()) if re.search(r"\d+$", p) else 0,
    )

    for path in paths:
        if path == _VIRTUAL_CAM_PATH:
            continue

        try:
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                backend_name: str = cap.getBackendName()
                devices.append({"name": backend_name, "path": path})
                cap.release()
            else:
                cap.release()
        except Exception:
            logger.debug("Skipping %s — could not open", path, exc_info=True)

    logger.info("Enumerated %d camera(s)", len(devices))
    return devices
