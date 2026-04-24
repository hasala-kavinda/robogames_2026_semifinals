"""Lightweight diagnostics for quick preflight verification on the drone."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config


def main() -> None:
    config = load_config("config/defaults.json")
    print("[diagnostics] Loaded configuration:")
    print(json.dumps(config, indent=2))

    host = config["camera"]["host"]
    port = int(config["camera"]["port"])
    try:
        with socket.create_connection((host, port), timeout=2.0):
            print(f"[diagnostics] Camera socket reachable at {host}:{port}")
    except OSError as exc:
        print(f"[diagnostics] Camera socket not reachable: {exc}")

    print("[diagnostics] Run scripts/test_mavlink.py for MAVLink heartbeat check")


if __name__ == "__main__":
    main()
