"""Filter toggle widgets with collapsible parameter sliders."""

from PyQt6.QtCore import (
    QPropertyAnimation,
    Qt,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ParamSlider(QWidget):
    """A labeled slider for a single filter parameter."""

    value_changed = pyqtSignal(str, float)

    def __init__(
        self,
        param_name: str,
        param_config: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._param_name = param_name
        self._config = param_config

        self._min_val: float = param_config.get("min", 0)
        self._max_val: float = param_config.get("max", 100)
        self._step: float = param_config.get("step", 1)
        self._param_type: str = param_config.get("type", "int")
        self._default: float = param_config.get("default", self._min_val)
        self._value: float = param_config.get("value", self._default)

        # Determine if this is an odd-only int param (step == 2)
        self._odd_only = self._param_type == "int" and self._step == 2

        # Calculate scale factor for float params
        if self._param_type == "float" and self._step > 0:
            self._scale = round(1.0 / self._step)
        else:
            self._scale = 1

        self._setup_ui()
        self._set_slider_from_value(self._value)

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Parameter name label
        self._name_label = QLabel(self._param_name)
        self._name_label.setFixedWidth(100)
        layout.addWidget(self._name_label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._configure_slider()
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        # Value label
        self._value_label = QLabel()
        self._value_label.setFixedWidth(50)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._update_value_label(self._value)
        layout.addWidget(self._value_label)

    def _configure_slider(self) -> None:
        if self._odd_only:
            # Map odd values: 1,3,5,... to slider indices 0,1,2,...
            min_idx = int(self._min_val) // 2
            max_idx = int(self._max_val) // 2
            self._slider.setRange(min_idx, max_idx)
            self._slider.setSingleStep(1)
        elif self._param_type == "float":
            self._slider.setRange(
                int(self._min_val * self._scale),
                int(self._max_val * self._scale),
            )
            self._slider.setSingleStep(1)
        else:
            self._slider.setRange(int(self._min_val), int(self._max_val))
            self._slider.setSingleStep(int(self._step))

    def _set_slider_from_value(self, value: float) -> None:
        if self._odd_only:
            self._slider.setValue(int(value) // 2)
        elif self._param_type == "float":
            self._slider.setValue(int(value * self._scale))
        else:
            self._slider.setValue(int(value))

    def _slider_to_value(self, slider_val: int) -> float:
        if self._odd_only:
            return float(slider_val * 2 + 1)
        elif self._param_type == "float":
            return slider_val / self._scale
        else:
            return float(slider_val)

    def _on_slider_changed(self, slider_val: int) -> None:
        actual = self._slider_to_value(slider_val)
        self._value = actual
        self._update_value_label(actual)
        self.value_changed.emit(self._param_name, actual)

    def _update_value_label(self, value: float) -> None:
        if self._param_type == "float":
            self._value_label.setText(f"{value:.2f}")
        else:
            self._value_label.setText(str(int(value)))

    def set_value(self, value: float) -> None:
        """Programmatically set the slider value."""
        self._value = value
        self._set_slider_from_value(value)


class FilterToggle(QWidget):
    """A toggle button for a filter with collapsible parameter sliders."""

    toggled = pyqtSignal(str, bool)
    param_changed = pyqtSignal(str, str, float)

    def __init__(
        self,
        filter_name: str,
        params: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._filter_name = filter_name
        self._params = params
        self._param_sliders: dict[str, ParamSlider] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Toggle button
        self._toggle_btn = QPushButton(self._filter_name)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setStyleSheet(self._button_style(False))
        self._toggle_btn.clicked.connect(self._on_toggled)
        layout.addWidget(self._toggle_btn)

        # Collapsible content frame
        self._content_frame = QFrame()
        self._content_frame.setMaximumHeight(0)
        self._content_frame.setStyleSheet("border: none;")
        content_layout = QVBoxLayout(self._content_frame)
        content_layout.setContentsMargins(0, 4, 0, 0)
        content_layout.setSpacing(2)

        for name, config in self._params.items():
            slider = ParamSlider(name, config)
            slider.value_changed.connect(self._on_param_changed)
            content_layout.addWidget(slider)
            self._param_sliders[name] = slider

        layout.addWidget(self._content_frame)

        # Animation for collapsing/expanding
        self._animation = QPropertyAnimation(self._content_frame, b"maximumHeight")
        self._animation.setDuration(200)

    @staticmethod
    def _button_style(active: bool) -> str:
        if active:
            return (
                "QPushButton { background-color: #4caf50; color: #ffffff; "
                "border-radius: 12px; padding: 6px 16px; border: none; }"
            )
        return (
            "QPushButton { background-color: #555555; color: #e0e0e0; "
            "border-radius: 12px; padding: 6px 16px; border: none; }"
        )

    def _on_toggled(self, checked: bool) -> None:
        self._toggle_btn.setStyleSheet(self._button_style(checked))
        self._animate_collapse(checked)
        self.toggled.emit(self._filter_name, checked)

    def _animate_collapse(self, expand: bool) -> None:
        content_height = self._content_frame.sizeHint().height()
        self._animation.stop()
        if expand:
            self._animation.setStartValue(0)
            self._animation.setEndValue(content_height)
        else:
            self._animation.setStartValue(self._content_frame.maximumHeight())
            self._animation.setEndValue(0)
        self._animation.start()

    def _on_param_changed(self, param_name: str, value: float) -> None:
        self.param_changed.emit(self._filter_name, param_name, value)

    def set_state(self, enabled: bool, params: dict | None = None) -> None:
        """Programmatically set the toggle state and parameter values."""
        self._toggle_btn.setChecked(enabled)
        self._toggle_btn.setStyleSheet(self._button_style(enabled))
        self._animate_collapse(enabled)
        if params:
            for name, value in params.items():
                if name in self._param_sliders:
                    self._param_sliders[name].set_value(value)


class FilterTogglePanel(QWidget):
    """Horizontal panel holding filter toggle widgets."""

    toggled = pyqtSignal(str, bool)
    param_changed = pyqtSignal(str, str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._toggles: dict[str, FilterToggle] = {}

    def setup_filters(self, filter_list: list[dict]) -> None:
        """Build filter toggle widgets from a list of filter definitions.

        Each dict must have 'name' (str) and 'params' (dict) keys.
        """
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._toggles.clear()

        for fdef in filter_list:
            name = fdef["name"]
            params = fdef.get("params", {})
            toggle = FilterToggle(name, params)
            toggle.toggled.connect(self.toggled)
            toggle.param_changed.connect(self.param_changed)
            self._layout.addWidget(toggle)
            self._toggles[name] = toggle

        self._layout.addStretch()

    def set_filter_state(
        self,
        filter_name: str,
        enabled: bool,
        params: dict | None = None,
    ) -> None:
        """Programmatically set a filter's toggle state and parameters."""
        if filter_name in self._toggles:
            self._toggles[filter_name].set_state(enabled, params)
