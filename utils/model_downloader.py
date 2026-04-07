"""Download MediaPipe model files on first use."""

import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.expanduser("~/.cache/vcam-studio/models")

_MODELS = {
    "selfie_segmenter": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def get_model_path(name: str) -> str:
    """Return local path for *name*, downloading if necessary."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = _MODELS[name]
    filename = url.rsplit("/", 1)[-1]
    # Use the key name as prefix to avoid collisions
    local_path = os.path.join(MODEL_DIR, f"{name}.{'tflite' if name == 'selfie_segmenter' else 'task'}")
    if not os.path.exists(local_path):
        logger.info("Downloading %s model from %s ...", name, url)
        urllib.request.urlretrieve(url, local_path)
        logger.info("Saved to %s", local_path)
    return local_path
