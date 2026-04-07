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
        """Warp the iris region by (shift_x, shift_y) pixels with tight falloff."""
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

        # Tight Gaussian falloff centered on the iris — sigma = iris_radius
        # This keeps the warp localised to the iris and doesn't distort eyelids
        sigma = max(iris_radius, 3.0)
        weight = np.exp(-dist_sq / (2.0 * sigma * sigma))

        # For cv2.remap: map tells where to *sample from*.
        # To move the iris in the +shift direction, we sample from -shift.
        map_x = (grid_x + shift_x * weight).astype(np.float32)
        map_y = (grid_y + shift_y * weight).astype(np.float32)

        warped_region = cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Soft circular blend mask centered on the iris (not the bbox center)
        iris_local_x = iris_center[0] - x1
        iris_local_y = iris_center[1] - y1
        local_cols = np.arange(region_w, dtype=np.float32)
        local_rows = np.arange(region_h, dtype=np.float32)
        lx, ly = np.meshgrid(local_cols, local_rows)
        local_dist = np.sqrt((lx - iris_local_x) ** 2 + (ly - iris_local_y) ** 2)

        # Blend radius slightly larger than iris
        blend_r = sigma * 1.8
        blend_mask = np.clip(1.0 - (local_dist / blend_r), 0.0, 1.0)
        blend_mask = cv2.GaussianBlur(blend_mask, (5, 5), sigmaX=0)
        blend_3ch = np.stack([blend_mask] * 3, axis=-1)

        output = frame.copy()
        roi = output[y1:y2, x1:x2].astype(np.float32)
        warped_f = warped_region[y1:y2, x1:x2].astype(np.float32) if warped_region.shape[0] > region_h else warped_region.astype(np.float32)
        # warped_region from remap is already the right size
        warped_f = warped_region.astype(np.float32)
        blended = warped_f * blend_3ch + roi * (1.0 - blend_3ch)
        output[y1:y2, x1:x2] = blended.astype(np.uint8)

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
