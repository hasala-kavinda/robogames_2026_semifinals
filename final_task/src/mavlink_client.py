"""Lightweight MAVLink helper optimized for Raspberry Pi field execution."""

from __future__ import annotations

import time
from typing import Optional

from pymavlink import mavutil


class MavlinkClient:
    """Wrap pymavlink operations with safe defaults and timeout handling."""

    def __init__(self, config: dict):
        self._cfg = config
        self.master: Optional[mavutil.mavfile] = None

    def connect(self) -> None:
        """Connect to MAVLink endpoint and wait for heartbeat."""
        connection = self._cfg["connection"]
        heartbeat_timeout_s = float(self._cfg.get("heartbeat_timeout_s", 10.0))
        print(f"[mavlink] Connecting to {connection}")
        self.master = mavutil.mavlink_connection(connection)
        self.master.wait_heartbeat(timeout=heartbeat_timeout_s)
        print("[mavlink] Heartbeat received")

    def _require_master(self):
        if self.master is None:
            raise RuntimeError("MAVLink client is not connected")
        return self.master

    def set_mode(self, mode: str) -> None:
        """Set autopilot mode; for example GUIDED or LAND."""
        master = self._require_master()
        mapping = master.mode_mapping()
        if mode not in mapping:
            raise ValueError(f"Mode {mode} is not available on this vehicle")

        mode_id = mapping[mode]
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        time.sleep(0.5)

    def is_armed(self, timeout_s: float = 1.0) -> bool:
        """Read armed state from heartbeat."""
        master = self._require_master()
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout_s)
        if msg is None:
            return False
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def arm(self, timeout_s: float = 5.0) -> None:
        """Arm motors and wait for armed state confirmation."""
        master = self._require_master()
        master.arducopter_arm()
        started = time.time()
        while time.time() - started < timeout_s:
            if self.is_armed(timeout_s=1.0):
                print("[mavlink] Vehicle armed")
                return
            time.sleep(0.2)
        raise TimeoutError("Timed out while arming")

    def get_relative_altitude_m(self, timeout_s: float = 1.0) -> Optional[float]:
        """Read relative altitude from GLOBAL_POSITION_INT, meters."""
        master = self._require_master()
        msg = master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=timeout_s,
        )
        if msg is None:
            return None
        return float(msg.relative_alt) / 1000.0

    def takeoff(self, target_altitude_m: float) -> None:
        """Issue takeoff command and wait until target altitude is reached."""
        master = self._require_master()
        max_altitude_m = float(self._cfg.get("max_altitude_m", 3.0))
        timeout_s = float(self._cfg.get("command_timeout_s", 30.0))
        if target_altitude_m > max_altitude_m:
            raise ValueError("Requested altitude exceeds configured max altitude")

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
            target_altitude_m,
        )

        started = time.time()
        while time.time() - started < timeout_s:
            altitude = self.get_relative_altitude_m(timeout_s=1.0)
            if altitude is None:
                continue
            if abs(altitude - target_altitude_m) < 0.08:
                print(f"[mavlink] Reached target altitude {target_altitude_m:.2f}m")
                return
            time.sleep(0.15)

        raise TimeoutError("Timed out while reaching takeoff altitude")

    def send_velocity_body(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        """Send velocity in body NED frame; positive vz means descending."""
        master = self._require_master()
        max_altitude_m = float(self._cfg.get("max_altitude_m", 3.0))

        current_alt = self.get_relative_altitude_m(timeout_s=0.15)
        # Clamp upward commands near ceiling to reduce altitude overshoot in gusty air.
        if current_alt is not None and current_alt >= max_altitude_m and vz < 0.0:
            vz = 0.0

        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000011111000111,
            0,
            0,
            0,
            float(vx),
            float(vy),
            float(vz),
            0,
            0,
            0,
            0,
            float(yaw_rate),
        )

    def stop_motion(self) -> None:
        """Command zero body velocity to stabilize before state transitions."""
        self.send_velocity_body(0.0, 0.0, 0.0, 0.0)

    def land(self) -> None:
        """Issue land command and wait for disarm."""
        master = self._require_master()
        timeout_s = float(self._cfg.get("command_timeout_s", 30.0))
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

        started = time.time()
        while time.time() - started < timeout_s:
            if not self.is_armed(timeout_s=1.0):
                print("[mavlink] Landing complete")
                return
            time.sleep(0.2)
        raise TimeoutError("Timed out while waiting for landing/disarm")
