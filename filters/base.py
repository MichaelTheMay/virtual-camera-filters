"""Base filter class for VirtualCam Studio."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ParamSpec = dict[str, Any]  # {"value", "min", "max", "step", "default", "type"}


class BaseFilter(ABC):
    """Abstract base class for all video filters."""

    name: str = "Unnamed Filter"

    def __init__(self) -> None:
        self.enabled: bool = False
        self._params: dict[str, ParamSpec] = {}

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _register_param(
        self,
        name: str,
        *,
        value: int | float,
        min_v: int | float,
        max_v: int | float,
        step: int | float,
        default: int | float,
        param_type: str,
    ) -> None:
        """Register a tunable parameter with validation metadata."""
        self._params[name] = {
            "value": value,
            "min": min_v,
            "max": max_v,
            "step": step,
            "default": default,
            "type": param_type,
        }

    def set_param(self, name: str, value: int | float) -> None:
        """Set a parameter value, clamping to bounds and snapping to step."""
        if name not in self._params:
            logger.error("Unknown parameter '%s' for filter '%s'", name, self.name)
            return

        spec = self._params[name]
        min_v = spec["min"]
        max_v = spec["max"]
        step = spec["step"]

        # Clamp
        value = max(min_v, min(max_v, value))

        # Snap to step (relative to min)
        steps = round((value - min_v) / step)
        value = min_v + steps * step

        # Clamp again in case rounding pushed past max
        value = max(min_v, min(max_v, value))

        if spec["type"] == "int":
            value = int(value)
        else:
            value = float(value)

        spec["value"] = value

    def get_params(self) -> dict[str, ParamSpec]:
        """Return the full parameter dictionary."""
        return self._params

    def get_state(self) -> dict[str, Any]:
        """Return serialisable state for persistence."""
        return {
            "enabled": self.enabled,
            "params": {name: spec["value"] for name, spec in self._params.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore filter state from a previously saved dict."""
        if "enabled" in state:
            self.enabled = bool(state["enabled"])
        for param_name, param_value in state.get("params", {}).items():
            self.set_param(param_name, param_value)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame and return the result."""
        ...
