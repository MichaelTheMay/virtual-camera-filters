"""Face melt filter — makes your face drip downward like melting wax."""

import logging
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from filters.base import BaseFilter
from utils.model_downloader import get_model_path

logger = logging.getLogger(__name__)

# Key face mesh landmark groups for the melt effect
# Jawline + lower face gets the strongest melt
JAWLINE: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
# Nose + mouth area — moderate melt
NOSE_MOUTH: list[int] = [
    1, 2, 3, 4, 5, 6, 195, 197,  # nose bridge/tip
    0, 11, 12, 13, 14, 15, 16, 17,  # upper lip
    267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185,  # lips
]
# Cheeks — moderate melt
CHEEKS: list[int] = [
    116, 117, 118, 119, 100, 36, 205, 187, 123, 147,  # left cheek
    345, 346, 347, 348, 329, 266, 425, 411, 352, 376,  # right cheek
]
# Forehead/brows — minimal melt (anchor points)
FOREHEAD: list[int] = [
    10, 67, 69, 104, 108, 109, 151, 299, 337, 338,
]


class FaceMeltFilter(BaseFilter):
    """Make faces drip downward like melting wax."""

    name: str = "Face Melt"

    def __init__(self) -> None:
        super().__init__()

        self._register_param(
            "speed",
            value=0.5,
            min_v=0.1,
            max_v=2.0,
            step=0.1,
            default=0.5,
            param_type="float",
        )
        self._register_param(
            "intensity",
            value=0.5,
            min_v=0.0,
            max_v=1.0,
            step=0.05,
            default=0.5,
            param_type="float",
        )
        self._register_param(
            "drip_length",
            value=40,
            min_v=10,
            max_v=150,
            step=2,
            default=40,
            param_type="int",
        )

        # MediaPipe Face Landmarker
        model_path = get_model_path("face_landmarker")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self._start_time: Optional[float] = None
        self._prev_landmarks: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            self._start_time = None
            self._prev_landmarks = None
            return frame

        try:
            if self._start_time is None:
                self._start_time = time.monotonic()

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                self._prev_landmarks = None
                return frame

            face_lm = result.face_landmarks[0]
            pts = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lm],
                dtype=np.float32,
            )

            # Smooth landmarks temporally
            if self._prev_landmarks is not None and self._prev_landmarks.shape == pts.shape:
                pts = 0.4 * pts + 0.6 * self._prev_landmarks
            self._prev_landmarks = pts.copy()

            speed: float = self._params["speed"]["value"]
            intensity: float = self._params["intensity"]["value"]
            drip_length: int = self._params["drip_length"]["value"]

            elapsed = time.monotonic() - self._start_time
            # Oscillating melt phase so it breathes
            phase = (np.sin(elapsed * speed * 1.5) + 1.0) / 2.0  # 0..1

            # Build per-landmark drip amount
            # Face center Y for reference
            face_center_y = pts[:, 1].mean()
            face_top_y = pts[:, 1].min()
            face_height = pts[:, 1].max() - face_top_y

            # Each landmark gets a downward displacement based on its vertical position
            # Lower landmarks drip more, upper landmarks act as anchors
            norm_y = (pts[:, 1] - face_top_y) / max(face_height, 1.0)  # 0=top, 1=bottom

            # Weight by landmark group
            drip_weight = np.zeros(len(pts), dtype=np.float32)
            for idx in JAWLINE:
                if idx < len(pts):
                    drip_weight[idx] = 1.0
            for idx in NOSE_MOUTH:
                if idx < len(pts):
                    drip_weight[idx] = 0.7
            for idx in CHEEKS:
                if idx < len(pts):
                    drip_weight[idx] = 0.6
            for idx in FOREHEAD:
                if idx < len(pts):
                    drip_weight[idx] = 0.05

            # Remaining landmarks: interpolate by vertical position
            for i in range(len(pts)):
                if drip_weight[i] == 0.0:
                    drip_weight[i] = norm_y[i] * 0.5

            # Per-landmark drip in pixels
            drip_amount = drip_weight * drip_length * intensity * phase

            # Add some waviness per landmark for organic look
            wave = np.sin(pts[:, 0] * 0.05 + elapsed * speed * 3.0) * 0.3 + 1.0
            drip_amount *= wave

            # Build displacement map using radial basis interpolation
            # For each pixel, compute displacement from nearby landmarks
            map_y_offset = np.zeros((h, w), dtype=np.float32)
            map_x_offset = np.zeros((h, w), dtype=np.float32)

            # For efficiency, use a grid-based approach with scattered landmark influence
            # Sample a sparse grid then upsample
            grid_step = 8
            small_h = h // grid_step + 1
            small_w = w // grid_step + 1
            small_dy = np.zeros((small_h, small_w), dtype=np.float32)
            small_dx = np.zeros((small_h, small_w), dtype=np.float32)

            # Grid coordinates
            gy = np.arange(small_h) * grid_step
            gx = np.arange(small_w) * grid_step
            grid_gx, grid_gy = np.meshgrid(gx, gy)

            # Influence radius — proportional to face size
            sigma = face_height * 0.25

            for i in range(len(pts)):
                if drip_amount[i] < 0.5:
                    continue
                px, py = pts[i]
                dist_sq = (grid_gx - px) ** 2 + (grid_gy - py) ** 2
                weight = np.exp(-dist_sq / (2.0 * sigma * sigma))
                # Drip is downward (positive y), with slight horizontal spread
                small_dy += weight * drip_amount[i]
                # Slight horizontal wobble
                small_dx += weight * np.sin(py * 0.03 + elapsed * speed * 2) * drip_amount[i] * 0.15

            # Upsample to full resolution
            map_y_offset = cv2.resize(small_dy, (w, h), interpolation=cv2.INTER_LINEAR)
            map_x_offset = cv2.resize(small_dx, (w, h), interpolation=cv2.INTER_LINEAR)

            # Build remap arrays
            base_x = np.arange(w, dtype=np.float32)
            base_y = np.arange(h, dtype=np.float32)
            map_x, map_y = np.meshgrid(base_x, base_y)

            # To drip downward, we sample from *above* (negative displacement in source)
            map_x = (map_x - map_x_offset).astype(np.float32)
            map_y = (map_y - map_y_offset).astype(np.float32)

            output = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            return output

        except Exception:
            logger.exception("FaceMeltFilter.process failed")
            return frame
