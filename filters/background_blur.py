"""Background blur filter using MediaPipe selfie segmentation."""

import logging

import cv2
import mediapipe as mp
import numpy as np

from filters.base import BaseFilter
from utils.model_downloader import get_model_path

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

        # Initialise MediaPipe selfie segmentation (tasks API)
        model_path = get_model_path("selfie_segmenter")
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_confidence_masks=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply background blur to *frame* (BGR, uint8)."""
        if not self.enabled:
            return frame

        try:
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._segmenter.segment(mp_image)

            # confidence_masks[0] is the person mask
            raw_mask = result.confidence_masks[0].numpy_view()
            # Squeeze if needed (may be h,w,1)
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[:, :, 0]

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
