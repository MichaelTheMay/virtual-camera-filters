"""Eye contact correction filter using MediaPipe Face Landmarker."""

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from filters.base import BaseFilter
from utils.model_downloader import get_model_path

logger = logging.getLogger(__name__)

# Landmark indices for iris detection (refined face mesh)
LEFT_IRIS: list[int] = [468, 469, 470, 471]
RIGHT_IRIS: list[int] = [473, 474, 475, 476]

# Eye contour landmarks for bounding the warp region
LEFT_EYE_CONTOUR: list[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_CONTOUR: list[int] = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]


class EyeContactFilter(BaseFilter):
    """Subtly shift the irises toward the camera to simulate eye contact."""

    name: str = "Eye Contact"

    def __init__(self) -> None:
        super().__init__()

        self._register_param(
            "strength",
            value=0.3,
            min_v=0.0,
            max_v=1.0,
            step=0.05,
            default=0.3,
            param_type="float",
        )
        self._register_param(
            "vertical",
            value=0.15,
            min_v=0.0,
            max_v=1.0,
            step=0.05,
            default=0.15,
            param_type="float",
        )
        self._register_param(
            "smoothing",
            value=5,
            min_v=1,
            max_v=10,
            step=1,
            default=5,
            param_type="int",
        )

        # Initialise MediaPipe Face Landmarker (tasks API)
        model_path = get_model_path("face_landmarker")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # Temporal smoothing state — smooth the iris centers, not all landmarks
        self._prev_left_iris: Optional[np.ndarray] = None
        self._prev_right_iris: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_array(
        landmarks: list, h: int, w: int
    ) -> np.ndarray:
        """Convert MediaPipe landmark list to an (N, 2) float32 array in pixel coords."""
        return np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks],
            dtype=np.float32,
        )

    @staticmethod
    def _iris_center(pts: np.ndarray, indices: list[int]) -> np.ndarray:
        """Return the mean (x, y) of the given iris landmark indices."""
        return pts[indices].mean(axis=0)

    def _smooth_point(
        self, current: np.ndarray, prev: Optional[np.ndarray]
    ) -> np.ndarray:
        """EMA smooth a single 2D point."""
        alpha = 1.0 / max(self._params["smoothing"]["value"], 1)
        if prev is None:
            return current.copy()
        return (alpha * current + (1.0 - alpha) * prev).astype(np.float32)

    @staticmethod
    def _eye_bounding_box(
        pts: np.ndarray, contour_indices: list[int], margin: int = 5
    ) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) tight around eye contour."""
        contour = pts[contour_indices]
        x_min, y_min = contour.min(axis=0).astype(int)
        x_max, y_max = contour.max(axis=0).astype(int)
        return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)

    @staticmethod
    def _warp_eye_region(
        frame: np.ndarray,
        iris_center: np.ndarray,
        shift_x: float,
        shift_y: float,
        bbox: tuple[int, int, int, int],
        iris_radius: float,
    ) -> np.ndarray:
        """Warp the iris region by (shift_x, shift_y) pixels with smooth falloff.

        Uses a wide Gaussian (~40% of bbox size) so the displacement tapers
        to zero at the bounding-box edges.  No separate blend mask is needed —
        at the borders the remap is an identity transform, making the seam
        invisible.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        region_h = y2 - y1
        region_w = x2 - x1
        if region_h <= 2 or region_w <= 2:
            return frame

        # Build coordinate grids in absolute frame coordinates
        cols = np.arange(region_w, dtype=np.float32) + x1
        rows = np.arange(region_h, dtype=np.float32) + y1
        grid_x, grid_y = np.meshgrid(cols, rows)

        # Distance from iris center
        dist_sq = (grid_x - iris_center[0]) ** 2 + (grid_y - iris_center[1]) ** 2

        # Wide Gaussian: sigma = 40% of the eye bbox size so the warp
        # spreads naturally across the whole eye and tapers to ~0 at edges.
        sigma = max(region_w, region_h) * 0.4
        weight = np.exp(-dist_sq / (2.0 * sigma * sigma))

        # remap maps tell where to *sample from*.
        # To shift the iris in the +shift direction, sample from -shift offset.
        map_x = (grid_x + shift_x * weight).astype(np.float32)
        map_y = (grid_y + shift_y * weight).astype(np.float32)

        warped_region = cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Paste directly — Gaussian weight ≈ 0 at edges so no seam.
        output = frame.copy()
        output[y1:y2, x1:x2] = warped_region

        return output

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Shift irises toward camera to simulate eye contact."""
        if not self.enabled:
            return frame

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                self._prev_left_iris = None
                self._prev_right_iris = None
                return frame

            face_lm = result.face_landmarks[0]
            pts = self._landmarks_to_array(face_lm, h, w)

            strength: float = self._params["strength"]["value"]
            vertical: float = self._params["vertical"]["value"]
            image_center_x = w / 2.0

            output = frame.copy()

            for iris_ids, contour_ids, is_left in [
                (LEFT_IRIS, LEFT_EYE_CONTOUR, True),
                (RIGHT_IRIS, RIGHT_EYE_CONTOUR, False),
            ]:
                iris_c = self._iris_center(pts, iris_ids)

                # Temporal smoothing on the iris center
                if is_left:
                    iris_c = self._smooth_point(iris_c, self._prev_left_iris)
                    self._prev_left_iris = iris_c.copy()
                else:
                    iris_c = self._smooth_point(iris_c, self._prev_right_iris)
                    self._prev_right_iris = iris_c.copy()

                # Estimate iris radius from the 4 iris landmarks
                iris_pts = pts[iris_ids]
                iris_radius = np.linalg.norm(iris_pts - iris_c, axis=1).mean()

                # How many pixels to shift the iris
                # Horizontal: toward image center
                shift_x = (image_center_x - iris_c[0]) * strength * 0.3
                # Vertical: upward (toward camera above screen) — negative y
                shift_y = -iris_radius * vertical * 1.5

                # Clamp to avoid crazy warps
                max_shift = iris_radius * 1.2
                shift_x = np.clip(shift_x, -max_shift, max_shift)
                shift_y = np.clip(shift_y, -max_shift, max_shift)

                bbox = self._eye_bounding_box(pts, contour_ids, margin=8)
                output = self._warp_eye_region(
                    output, iris_c, shift_x, shift_y, bbox, iris_radius
                )

            return output

        except Exception:
            logger.exception("EyeContactFilter.process failed")
            return frame
