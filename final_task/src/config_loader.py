"""Configuration loader with project-relative paths and override support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"


class ConfigError(RuntimeError):
    """Raised when the runtime configuration is missing or malformed."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load default config and optional override file using relative-safe paths."""
    try:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Default config not found: {DEFAULT_CONFIG_PATH}") from exc

    if not config_path:
        return config

    override_path = Path(config_path)
    if not override_path.is_absolute():
        override_path = (PROJECT_ROOT / override_path).resolve()

    try:
        with override_path.open("r", encoding="utf-8") as handle:
            override = json.load(handle)
    except FileNotFoundError as exc:
        hint = ""
        if override_path == (PROJECT_ROOT / "config" / "local.json").resolve():
            hint = (
                " (create it with: cp config/hardware.example.json config/local.json)"
            )
        raise ConfigError(f"Override config not found: {override_path}{hint}") from exc

    if not isinstance(override, dict):
        raise ConfigError("Override config must be a JSON object")

    return _deep_merge(config, override)
