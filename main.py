"""VirtualCam Studio – integration layer.

Wires together capture, processing, output, GUI, and filter modules
into a running PyQt6 application with live webcam preview and
virtual-camera output via v4l2loopback.
"""

import logging
import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.capture import CaptureThread
from core.config_manager import ConfigManager
from core.output import OutputThread
from core.processor import ProcessorThread
from filters.background_blur import BackgroundBlurFilter
from filters.eye_contact import EyeContactFilter
from gui.main_window import MainWindow
from gui.theme import get_dark_stylesheet
from utils.device_enumerator import enumerate_cameras
from utils.frame_convert import frame_to_qpixmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("virtualcam")

VIRTUAL_DEVICE = "/dev/video10"


def _build_filter_info(filters: list) -> list[dict]:
    """Return a list of dicts describing each filter for the GUI."""
    return [
        {
            "name": f.name,
            "enabled": f.enabled,
            "params": f.get_params(),
        }
        for f in filters
    ]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Check for the v4l2loopback virtual camera device
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)

    if not os.path.exists(VIRTUAL_DEVICE):
        QMessageBox.warning(
            None,
            "Virtual Camera Not Found",
            "Virtual camera not found.\n\n"
            "Run: sudo modprobe v4l2loopback devices=1 video_nr=10 "
            "card_label='VirtualCam' exclusive_caps=1",
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Apply dark theme
    # ------------------------------------------------------------------
    app.setStyleSheet(get_dark_stylesheet())

    # ------------------------------------------------------------------
    # 3. Load persisted configuration
    # ------------------------------------------------------------------
    config_manager = ConfigManager()
    try:
        saved_state = config_manager.load()
    except Exception:
        logger.exception("Failed to load config – using defaults")
        saved_state = {}

    # ------------------------------------------------------------------
    # 4. Instantiate filters and restore saved state
    # ------------------------------------------------------------------
    bg_blur = BackgroundBlurFilter()
    eye_contact = EyeContactFilter()
    filters: list = [bg_blur, eye_contact]

    for f in filters:
        state = saved_state.get(f.name)
        if state is not None:
            try:
                f.load_state(state)
            except Exception:
                logger.exception("Failed to restore state for filter %s", f.name)

    # ------------------------------------------------------------------
    # 5. Build the main window
    # ------------------------------------------------------------------
    cameras = enumerate_cameras()
    window = MainWindow()
    window.set_cameras(cameras)
    window.setup_filters(_build_filter_info(filters))

    # ------------------------------------------------------------------
    # 6. Pipeline thread references
    # ------------------------------------------------------------------
    capture_thread: CaptureThread | None = None
    processor_thread: ProcessorThread | None = None
    output_thread: OutputThread | None = None

    current_device: str = saved_state.get("current_device", "")
    if not current_device and cameras:
        current_device = cameras[0]["path"]

    # ------------------------------------------------------------------
    # Helper: start / stop the pipeline
    # ------------------------------------------------------------------
    def _start_pipeline() -> None:
        nonlocal capture_thread, processor_thread, output_thread

        if not current_device:
            window.show_error("No Camera", "No camera device selected.")
            return

        try:
            capture_thread = CaptureThread(current_device)
            processor_thread = ProcessorThread(capture_thread)
            processor_thread.set_filters(
                [f for f in filters if f.enabled]
            )
            output_thread = OutputThread(processor_thread, device=VIRTUAL_DEVICE)

            capture_thread.start()
            processor_thread.start()
            output_thread.start()

            window.refresh_timer.start()
            logger.info("Pipeline started on %s", current_device)
        except Exception:
            logger.exception("Failed to start pipeline")
            _stop_pipeline()
            window.show_error(
                "Pipeline Error",
                "Failed to start the camera pipeline. Check logs for details.",
            )

    def _stop_pipeline() -> None:
        nonlocal capture_thread, processor_thread, output_thread

        window.refresh_timer.stop()

        for thread, label in [
            (output_thread, "output"),
            (processor_thread, "processor"),
            (capture_thread, "capture"),
        ]:
            if thread is not None:
                try:
                    thread.stop()
                except Exception:
                    logger.exception("Error stopping %s thread", label)

        output_thread = None
        processor_thread = None
        capture_thread = None
        logger.info("Pipeline stopped")

    pipeline_running = False

    def _toggle_pipeline() -> None:
        nonlocal pipeline_running
        if pipeline_running:
            _stop_pipeline()
            pipeline_running = False
        else:
            _start_pipeline()
            pipeline_running = True

    # ------------------------------------------------------------------
    # 7. Wire signals
    # ------------------------------------------------------------------
    def _on_camera_changed(device_path: str) -> None:
        nonlocal current_device, pipeline_running
        current_device = device_path
        logger.info("Camera changed to %s", device_path)
        if pipeline_running:
            _stop_pipeline()
            _start_pipeline()

    def _on_filter_toggled(filter_name: str, enabled: bool) -> None:
        for f in filters:
            if f.name == filter_name:
                f.enabled = enabled
                break
        if processor_thread is not None:
            try:
                processor_thread.set_filters(
                    [f for f in filters if f.enabled]
                )
            except Exception:
                logger.exception("Error updating filters on processor")

    def _on_param_changed(filter_name: str, param_name: str, value: float) -> None:
        for f in filters:
            if f.name == filter_name:
                try:
                    f.set_param(param_name, value)
                except Exception:
                    logger.exception(
                        "Error setting param %s on %s", param_name, filter_name
                    )
                break

    window.start_stop_clicked.connect(_toggle_pipeline)
    window.camera_changed.connect(_on_camera_changed)
    window.filter_toggled.connect(_on_filter_toggled)
    window.param_changed.connect(_on_param_changed)

    # ------------------------------------------------------------------
    # 8. Refresh timer → preview + FPS
    # ------------------------------------------------------------------
    def _on_refresh() -> None:
        if processor_thread is None:
            return
        try:
            with processor_thread.lock:
                frame = processor_thread.processed_frame
            if frame is not None:
                pixmap = frame_to_qpixmap(frame)
                window.set_preview(pixmap)
            if capture_thread is not None:
                window.set_fps(capture_thread.fps)
        except Exception:
            logger.exception("Error during preview refresh")

    window.refresh_timer.timeout.connect(_on_refresh)

    # ------------------------------------------------------------------
    # 9. Cleanup on quit – stop pipeline & save config
    # ------------------------------------------------------------------
    def _on_quit() -> None:
        nonlocal pipeline_running
        if pipeline_running:
            _stop_pipeline()
            pipeline_running = False

        state_to_save: dict = {"current_device": current_device}
        for f in filters:
            try:
                state_to_save[f.name] = f.get_state()
            except Exception:
                logger.exception("Error getting state for filter %s", f.name)
        try:
            config_manager.save(state_to_save)
        except Exception:
            logger.exception("Failed to save config")

    app.aboutToQuit.connect(_on_quit)

    # ------------------------------------------------------------------
    # 10. Show window and run
    # ------------------------------------------------------------------
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
