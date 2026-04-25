#!/usr/bin/env python3
"""Single-file MAVLink takeoff and land check script.

Default flow:
1) Connect to MAVLink endpoint
2) Set GUIDED mode
3) Arm
4) Take off to target altitude
5) Hover for configured seconds
6) Land and wait for disarm

Examples:
  python3 Checks/takeoff_land.py
  python3 Checks/takeoff_land.py --connection udp:0.0.0.0:14550 --altitude 1.2 --hover 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from pymavlink import mavutil


DEFAULT_ALTITUDE_M = 1.0
DEFAULT_HOVER_S = 5.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 10.0
DEFAULT_COMMAND_TIMEOUT_S = 30.0
DEFAULT_CONNECTION = "udp:192.168.1.71:14550"


def load_defaults() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    defaults_path = repo_root / "final_task" / "config" / "defaults.json"
    if not defaults_path.exists():
        return {}
    try:
        with defaults_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def parse_args() -> argparse.Namespace:
    defaults = load_defaults()
    mav_cfg = defaults.get("mavlink", {}) if isinstance(defaults, dict) else {}

    parser = argparse.ArgumentParser(description="MAVLink arm/takeoff/land quick check")
    parser.add_argument(
        "--connection",
        default=mav_cfg.get("connection", DEFAULT_CONNECTION),
        help="MAVLink connection string (e.g. udp:0.0.0.0:14550)",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=DEFAULT_ALTITUDE_M,
        help="Takeoff altitude in meters",
    )
    parser.add_argument(
        "--hover",
        type=float,
        default=DEFAULT_HOVER_S,
        help="Hover duration in seconds after reaching altitude",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=float(mav_cfg.get("heartbeat_timeout_s", DEFAULT_HEARTBEAT_TIMEOUT_S)),
        help="Heartbeat wait timeout in seconds",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=float(mav_cfg.get("command_timeout_s", DEFAULT_COMMAND_TIMEOUT_S)),
        help="Timeout for arm/takeoff/land waits in seconds",
    )
    parser.add_argument(
        "--altitude-tolerance",
        type=float,
        default=0.10,
        help="Altitude acceptance tolerance in meters",
    )
    return parser.parse_args()


def wait_heartbeat(master: mavutil.mavfile, timeout_s: float) -> None:
    print(f"[flight] Waiting heartbeat (timeout={timeout_s:.1f}s)")
    master.wait_heartbeat(timeout=timeout_s)
    print(
        f"[flight] Connected. system={master.target_system} component={master.target_component}"
    )


def mode_id_for(master: mavutil.mavfile, mode: str) -> int:
    mapping = master.mode_mapping()
    if not mapping or mode not in mapping:
        raise RuntimeError(f"Mode '{mode}' is not available on this vehicle")
    return int(mapping[mode])


def set_mode(master: mavutil.mavfile, mode: str, timeout_s: float) -> None:
    mode_id = mode_id_for(master, mode)
    print(f"[flight] Setting mode to {mode}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if hb is None:
            continue
        if int(hb.custom_mode) == mode_id:
            print(f"[flight] Mode confirmed: {mode}")
            return
    raise TimeoutError(f"Timed out while waiting for mode {mode}")


def is_armed(master: mavutil.mavfile, timeout_s: float = 1.0) -> bool:
    hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout_s)
    if hb is None:
        return False
    armed_flag = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    return bool(int(hb.base_mode) & int(armed_flag))


def arm_vehicle(master: mavutil.mavfile, timeout_s: float) -> None:
    print("[flight] Arming vehicle")
    master.arducopter_arm()

    deadline = time.time() + timeout_s
    forced_arm_sent = False
    while time.time() < deadline:
        if is_armed(master, timeout_s=1.0):
            print("[flight] Vehicle armed")
            return

        # Some ArduPilot setups require a force-arm fallback.
        if not forced_arm_sent and (deadline - time.time()) < (timeout_s / 2.0):
            print("[flight] Standard arm pending, sending force-arm fallback")
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                21196,
                0,
                0,
                0,
                0,
                0,
            )
            forced_arm_sent = True

    raise TimeoutError("Timed out while arming")


def relative_altitude_m(
    master: mavutil.mavfile, timeout_s: float = 1.0
) -> Optional[float]:
    msg = master.recv_match(
        type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout_s
    )
    if msg is None:
        return None
    return float(msg.relative_alt) / 1000.0


def takeoff(
    master: mavutil.mavfile, altitude_m: float, timeout_s: float, tolerance_m: float
) -> None:
    print(f"[flight] Takeoff to {altitude_m:.2f} m")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        float(altitude_m),
    )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alt = relative_altitude_m(master, timeout_s=1.0)
        if alt is None:
            continue
        print(f"[flight] Current altitude: {alt:.2f} m")
        if alt >= altitude_m - tolerance_m:
            print("[flight] Target altitude reached")
            return
    raise TimeoutError("Timed out while reaching target altitude")


def land(master: mavutil.mavfile) -> None:
    print("[flight] Sending LAND command")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def wait_disarmed(master: mavutil.mavfile, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not is_armed(master, timeout_s=1.0):
            print("[flight] Vehicle disarmed")
            return
    raise TimeoutError("Timed out while waiting for disarm")


def safe_land(master: Optional[mavutil.mavfile], timeout_s: float) -> None:
    if master is None:
        return
    try:
        if is_armed(master, timeout_s=0.5):
            print("[flight] Safety path: attempting LAND")
            land(master)
            wait_disarmed(master, timeout_s=min(timeout_s, 30.0))
    except Exception as exc:
        print(f"[flight] Safety LAND attempt failed: {exc}")


def main() -> int:
    args = parse_args()
    if args.altitude <= 0.0:
        print("[flight] --altitude must be > 0")
        return 2
    if args.hover < 0.0:
        print("[flight] --hover must be >= 0")
        return 2

    print(f"[flight] Connecting to: {args.connection}")
    master: Optional[mavutil.mavfile] = None

    try:
        master = mavutil.mavlink_connection(args.connection)
        wait_heartbeat(master, timeout_s=args.heartbeat_timeout)

        set_mode(master, "GUIDED", timeout_s=args.command_timeout)
        arm_vehicle(master, timeout_s=args.command_timeout)
        takeoff(
            master,
            altitude_m=args.altitude,
            timeout_s=args.command_timeout,
            tolerance_m=max(0.02, args.altitude_tolerance),
        )

        if args.hover > 0.0:
            print(f"[flight] Hovering for {args.hover:.1f} s")
            time.sleep(args.hover)

        land(master)
        wait_disarmed(master, timeout_s=args.command_timeout)
        print("[flight] Takeoff/land sequence complete")
        return 0

    except KeyboardInterrupt:
        print("[flight] Interrupted by user")
        safe_land(master, timeout_s=args.command_timeout)
        return 130

    except Exception as exc:
        print(f"[flight] Error: {exc}")
        safe_land(master, timeout_s=args.command_timeout)
        return 1

    finally:
        if master is not None:
            master.close()


if __name__ == "__main__":
    sys.exit(main())
