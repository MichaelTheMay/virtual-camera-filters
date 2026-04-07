"""Dark theme stylesheet for VirtualCam Studio."""


def get_dark_stylesheet() -> str:
    """Return a comprehensive Qt dark theme stylesheet string."""
    return """
    QMainWindow {
        background-color: #1e1e1e;
    }

    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: "Segoe UI", "Ubuntu", sans-serif;
        font-size: 13px;
    }

    QPushButton {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3d3d3d;
        border-radius: 12px;
        padding: 6px 16px;
        min-height: 24px;
    }

    QPushButton:hover {
        background-color: #3d3d3d;
        border-color: #4a9eff;
    }

    QPushButton:pressed {
        background-color: #4a9eff;
        color: #ffffff;
    }

    QPushButton:checked {
        background-color: #4caf50;
        color: #ffffff;
        border-color: #4caf50;
    }

    QComboBox {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        padding: 4px 8px;
        min-height: 24px;
    }

    QComboBox:hover {
        border-color: #4a9eff;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #3d3d3d;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #e0e0e0;
        margin-right: 5px;
    }

    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3d3d3d;
        selection-background-color: #4a9eff;
        selection-color: #ffffff;
    }

    QSlider::groove:horizontal {
        border: none;
        height: 4px;
        background: #3d3d3d;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        background: #4a9eff;
        border: none;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }

    QSlider::handle:horizontal:hover {
        background: #6ab4ff;
    }

    QSlider::sub-page:horizontal {
        background: #4a9eff;
        border-radius: 2px;
    }

    QSlider::add-page:horizontal {
        background: #3d3d3d;
        border-radius: 2px;
    }

    QLabel {
        background-color: transparent;
        color: #e0e0e0;
        border: none;
    }

    QGroupBox {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        color: #4a9eff;
    }

    QFrame {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 4px;
    }

    QMessageBox {
        background-color: #1e1e1e;
    }

    QMessageBox QLabel {
        color: #e0e0e0;
    }

    QMessageBox QPushButton {
        min-width: 80px;
    }
    """
