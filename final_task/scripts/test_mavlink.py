"""Quick MAVLink connectivity check for onboard verification."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.mavlink_client import MavlinkClient


def main() -> None:
    config = load_config("config/defaults.json")
    mavlink = MavlinkClient(config["mavlink"])
    mavlink.connect()

    altitude = mavlink.get_relative_altitude_m(timeout_s=2.0)
    if altitude is None:
        print("[test_mavlink] Connected, but no altitude sample received yet")
    else:
        print(f"[test_mavlink] Connected. Relative altitude: {altitude:.2f}m")


if __name__ == "__main__":
    main()
