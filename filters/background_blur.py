"""Background blur filter using MediaPipe selfie segmentation."""

import logging

import cv2
import mediapipe as mp
import numpy as np

from filters.base import BaseFilter

logger = logging.getLogger(__name__)


class BackgroundBlurFilter(BaseFilter):
    """Blur the background while keeping the foreground subject sharp."""

    name: str = "Background Blur"

    def __init__(self) -> None:
        super().__init__()

        self._register_param(
            "threshold",
            value=0.6,
            min_v=0.0,
            max_v=1.0,
            step=0.05,
            default=0.6,
            param_type="float",
        )
        self._register_param(
            "blur_strength",
            value=55,
            min_v=1,
            max_v=199,
            step=2,
            default=55,
            param_type="int",
        )
        self._register_param(
            "edge_smoothing",
            value=7,
            min_v=1,
            max_v=21,
            step=2,
            default=7,
            param_type="int",
        )

        # Initialise MediaPipe selfie segmentation once
        self._selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1,
        )

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply background blur to *frame* (BGR, uint8)."""
        if not self.enabled:
            return frame

        try:
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._selfie_seg.process(rgb)
            raw_mask = result.segmentation_mask  # float32, [0..1]

            # Binary threshold
            threshold: float = self._params["threshold"]["value"]
            binary_mask = (raw_mask > threshold).astype(np.float32)

            # Smooth edges
            edge_k: int = self._params["edge_smoothing"]["value"]
            binary_mask = cv2.GaussianBlur(
                binary_mask, (edge_k, edge_k), sigmaX=0
            )

            # Expand to 3-channel mask for element-wise blending
            mask_3ch = np.stack([binary_mask] * 3, axis=-1)  # (h, w, 3) float32

            # Blurred background
            blur_k: int = self._params["blur_strength"]["value"]
            blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), sigmaX=0)

            # Blend: foreground * mask + background * (1 - mask)
            frame_f = frame.astype(np.float32)
            blurred_f = blurred.astype(np.float32)
            output = frame_f * mask_3ch + blurred_f * (1.0 - mask_3ch)

            return output.astype(np.uint8)

        except Exception:
            logger.exception("BackgroundBlurFilter.process failed")
            return frame
