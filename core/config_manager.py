"""Persistent configuration manager — reads/writes YAML session state."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_STATE: Dict[str, Any] = {
    "device_path": "/dev/video0",
    "filters": {},
}

_CONFIG_DIR: Path = Path.home() / ".config" / "vcam-studio"
_CONFIG_FILE: Path = _CONFIG_DIR / "session.yaml"


class ConfigManager:
    """Save and load application state to ``~/.config/vcam-studio/session.yaml``."""

    def __init__(self, config_path: Path = _CONFIG_FILE) -> None:
        self._path: Path = config_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: Dict[str, Any]) -> None:
        """Persist *state* to the YAML config file."""
        os.makedirs(self._path.parent, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(state, fh, default_flow_style=False)
            logger.info("Configuration saved to %s", self._path)
        except Exception:
            logger.exception("Failed to save configuration to %s", self._path)

    def load(self) -> Dict[str, Any]:
        """Load state from the YAML config file.

        Returns default values when the file is missing or corrupt.
        """
        if not self._path.exists():
            logger.info("No config file found — returning defaults")
            return dict(_DEFAULT_STATE)

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise ValueError("Config root is not a mapping")
            logger.info("Configuration loaded from %s", self._path)
            return data
        except Exception:
            logger.exception(
                "Corrupt or unreadable config at %s — returning defaults", self._path
            )
            return dict(_DEFAULT_STATE)
