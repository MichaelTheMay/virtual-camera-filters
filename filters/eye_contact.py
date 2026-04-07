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
            value=0.5,
            min_v=0.0,
            max_v=1.0,
            step=0.05,
            default=0.5,
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

        # Temporal smoothing state
        self._prev_landmarks: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_array(
        landmarks: list, h: int, w: int
    ) -> np.ndarray:
        """Convert MediaPipe landmark list to an (N, 2) float32 array in pixel coords."""
        pts = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks],
            dtype=np.float32,
        )
        return pts

    @staticmethod
    def _iris_center(pts: np.ndarray, indices: list[int]) -> np.ndarray:
        """Return the mean (x, y) of the given iris landmark indices."""
        return pts[indices].mean(axis=0)

    def _smooth_landmarks(self, pts: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to landmark positions."""
        alpha: float = 1.0 / self._params["smoothing"]["value"]
        if self._prev_landmarks is None or self._prev_landmarks.shape != pts.shape:
            self._prev_landmarks = pts.copy()
            return pts
        smoothed = alpha * pts + (1.0 - alpha) * self._prev_landmarks
        self._prev_landmarks = smoothed.copy()
        return smoothed

    @staticmethod
    def _eye_bounding_box(
        pts: np.ndarray, contour_indices: list[int], margin: int = 10
    ) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box around eye contour landmarks."""
        contour = pts[contour_indices]
        x_min, y_min = contour.min(axis=0).astype(int)
        x_max, y_max = contour.max(axis=0).astype(int)
        return (
            max(x_min - margin, 0),
            max(y_min - margin, 0),
            x_max + margin,
            y_max + margin,
        )

    @staticmethod
    def _warp_eye_region(
        frame: np.ndarray,
        iris_center: np.ndarray,
        target_center: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Warp an eye region so the iris moves toward *target_center*."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x2 = min(x2, w)
        y2 = min(y2, h)

        region_h = y2 - y1
        region_w = x2 - x1
        if region_h <= 0 or region_w <= 0:
            return frame

        # Displacement vector (pixels)
        dx = target_center[0] - iris_center[0]
        dy = target_center[1] - iris_center[1]

        # Radius of influence
        radius = max(region_w, region_h) / 2.0

        # Build coordinate grids for the region (vectorised)
        cols = np.arange(region_w, dtype=np.float32) + x1
        rows = np.arange(region_h, dtype=np.float32) + y1
        grid_x, grid_y = np.meshgrid(cols, rows)

        dist = np.sqrt(
            (grid_x - iris_center[0]) ** 2
            + (grid_y - iris_center[1]) ** 2
        )
        weight = np.exp(-(dist ** 2) / (2.0 * (radius / 2.0) ** 2))

        map_x = (grid_x - dx * weight).astype(np.float32)
        map_y = (grid_y - dy * weight).astype(np.float32)

        warped_region = cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR
        )

        # Blend back using a soft elliptical mask
        blend_mask = np.zeros((region_h, region_w), dtype=np.float32)
        cv2.ellipse(
            blend_mask,
            center=(region_w // 2, region_h // 2),
            axes=(region_w // 2 - 2, region_h // 2 - 2),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )
        blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), sigmaX=0)
        blend_3ch = np.stack([blend_mask] * 3, axis=-1)

        output = frame.copy()
        roi = output[y1:y2, x1:x2].astype(np.float32)
        warped_f = warped_region.astype(np.float32)
        blended = warped_f * blend_3ch + roi * (1.0 - blend_3ch)
        output[y1:y2, x1:x2] = blended.astype(np.uint8)

        return output

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Shift irises toward the image centre to simulate eye contact."""
        if not self.enabled:
            return frame

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                self._prev_landmarks = None
                return frame

            face_lm = result.face_landmarks[0]
            pts = self._landmarks_to_array(face_lm, h, w)
            pts = self._smooth_landmarks(pts)

            strength: float = self._params["strength"]["value"]
            image_center_x = w / 2.0

            output = frame.copy()

            for iris_ids, contour_ids in [
                (LEFT_IRIS, LEFT_EYE_CONTOUR),
                (RIGHT_IRIS, RIGHT_EYE_CONTOUR),
            ]:
                iris_c = self._iris_center(pts, iris_ids)
                target = iris_c.copy()
                target[0] += (image_center_x - iris_c[0]) * strength

                bbox = self._eye_bounding_box(pts, contour_ids)
                bbox = (
                    max(bbox[0], 0),
                    max(bbox[1], 0),
                    min(bbox[2], w),
                    min(bbox[3], h),
                )

                output = self._warp_eye_region(output, iris_c, target, bbox)

            return output

        except Exception:
            logger.exception("EyeContactFilter.process failed")
            return frame
