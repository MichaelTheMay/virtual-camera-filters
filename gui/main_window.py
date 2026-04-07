"""Main application window for VirtualCam Studio."""

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.filter_toggles import FilterTogglePanel
from gui.preview_widget import PreviewWidget
from gui.theme import get_dark_stylesheet


class MainWindow(QMainWindow):
    """Main window for VirtualCam Studio with preview, controls, and filters."""

    start_stop_clicked = pyqtSignal()
    camera_changed = pyqtSignal(str)
    filter_toggled = pyqtSignal(str, bool)
    param_changed = pyqtSignal(str, str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_running = False

        self.setWindowTitle("VirtualCam Studio")
        self.setMinimumSize(QSize(800, 600))
        self.setStyleSheet(get_dark_stylesheet())

        self._setup_ui()

        # Refresh timer (created but NOT started)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(33)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- Top bar ---
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(200)
        self._camera_combo.currentTextChanged.connect(self._on_camera_changed)
        top_layout.addWidget(self._camera_combo)

        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet(
            "QPushButton { border-radius: 12px; padding: 6px 16px; }"
        )
        self._start_btn.clicked.connect(self._on_start_stop)
        top_layout.addWidget(self._start_btn)

        top_layout.addStretch()

        self._fps_label = QLabel("0 FPS")
        self._fps_label.setFixedWidth(80)
        self._fps_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        top_layout.addWidget(self._fps_label)

        main_layout.addWidget(top_bar)

        # --- Preview ---
        self._preview = PreviewWidget()
        main_layout.addWidget(self._preview, stretch=1)

        # --- Filter panel ---
        self._filter_panel = FilterTogglePanel()
        self._filter_panel.toggled.connect(self.filter_toggled)
        self._filter_panel.param_changed.connect(self.param_changed)
        main_layout.addWidget(self._filter_panel)

    def _on_camera_changed(self, text: str) -> None:
        """Handle camera selection change."""
        idx = self._camera_combo.currentIndex()
        path = self._camera_combo.itemData(idx)
        self.camera_changed.emit(path if path else text)

    def _on_start_stop(self) -> None:
        """Toggle between start and stop states."""
        self._is_running = not self._is_running
        if self._is_running:
            self._start_btn.setText("Stop")
            self._start_btn.setStyleSheet(
                "QPushButton { background-color: #4caf50; color: #ffffff; "
                "border-radius: 12px; padding: 6px 16px; }"
            )
        else:
            self._start_btn.setText("Start")
            self._start_btn.setStyleSheet(
                "QPushButton { border-radius: 12px; padding: 6px 16px; }"
            )
        self.start_stop_clicked.emit()

    def set_preview(self, pixmap: QPixmap) -> None:
        """Update the preview widget with a new frame."""
        self._preview.update_frame(pixmap)

    def set_fps(self, fps: float) -> None:
        """Update the FPS display label."""
        self._fps_label.setText(f"{fps:.0f} FPS")

    def set_cameras(self, cameras: list[dict]) -> None:
        """Populate the camera combo box.

        Each dict should have 'name' and 'path' keys.
        """
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        for cam in cameras:
            self._camera_combo.addItem(cam["name"], cam["path"])
        self._camera_combo.blockSignals(False)
        if cameras:
            self._camera_combo.setCurrentIndex(0)

    def setup_filters(self, filter_list: list[dict]) -> None:
        """Set up filter toggles in the bottom panel."""
        self._filter_panel.setup_filters(filter_list)

    def show_error(self, title: str, message: str) -> None:
        """Display an error message dialog."""
        QMessageBox.critical(self, title, message)
