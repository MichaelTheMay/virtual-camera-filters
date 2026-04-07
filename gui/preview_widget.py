"""Preview widget for displaying camera/filter output frames."""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLabel


class PreviewWidget(QLabel):
    """Widget that displays a camera preview frame, scaled to fit."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None

        self.setStyleSheet("background-color: #000000;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)

    def update_frame(self, pixmap: QPixmap) -> None:
        """Store the original pixmap and display it scaled to widget size."""
        self._original_pixmap = pixmap
        self._scale_and_display()

    def _scale_and_display(self) -> None:
        """Scale the stored pixmap to the current widget size and display it."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Rescale the stored pixmap when the widget is resized."""
        super().resizeEvent(event)
        self._scale_and_display()

    def sizeHint(self) -> QSize:  # noqa: N802
        """Return the preferred size for the preview widget."""
        return QSize(640, 480)
