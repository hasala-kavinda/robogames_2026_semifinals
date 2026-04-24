"""Landing command fusion using AprilTag alignment plus line-follow assist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .apriltag_detector import TagDetection
from .line_follower import LineResult


@dataclass
class LandingCommand:
    """Velocity command proposal for the mission loop."""

    vx: float
    vy: float
    vz: float
    yaw_rate: float
    should_land: bool
    reason: str


class LandingController:
    """Track target AprilTag and trigger landing when alignment is stable."""

    def __init__(self, config: dict):
        self._cfg = config
        self._stable_counter = 0

    def reset(self) -> None:
        """Reset stability counter when mission state is re-entered."""
        self._stable_counter = 0

    def compute_command(
        self,
        target_tag: Optional[TagDetection],
        line_result: LineResult,
        frame_width: int,
        frame_height: int,
    ) -> LandingCommand:
        """Fuse tag centering and line guidance to produce safe approach command."""
        if target_tag is None:
            self._stable_counter = 0
            return LandingCommand(0.0, 0.0, 0.0, 0.15, False, "tag_missing")

        target_x = frame_width / 2.0
        target_y = frame_height / 2.0

        dx = (target_tag.center[0] - target_x) / max(target_x, 1.0)
        dy = (target_tag.center[1] - target_y) / max(target_y, 1.0)

        lateral_gain = float(self._cfg.get("lateral_gain", 0.30))
        line_assist_gain = float(self._cfg.get("line_assist_gain", 0.20))
        max_lateral = float(self._cfg.get("max_lateral_speed_mps", 0.30))

        # vy in body frame steers left/right; blend line error to reduce drift on wind gusts.
        vy = (dx * lateral_gain)
        if line_result.visible:
            vy += float(line_result.error) * line_assist_gain
        vy = max(min(vy, max_lateral), -max_lateral)

        # Keep gentle forward speed while tag is above center to avoid stalling short.
        approach_speed = float(self._cfg.get("approach_speed_mps", 0.15))
        vx = approach_speed if dy < 0.35 else 0.05

        tolerance_px = int(self._cfg.get("tag_center_tolerance_px", 70))
        stable_needed = int(self._cfg.get("stable_frames_required", 8))
        centered = (
            abs(target_tag.center[0] - int(target_x)) <= tolerance_px
            and abs(target_tag.center[1] - int(target_y)) <= tolerance_px
        )

        if centered:
            self._stable_counter += 1
        else:
            self._stable_counter = 0

        if self._stable_counter >= stable_needed:
            return LandingCommand(0.0, 0.0, 0.0, 0.0, True, "stable_alignment")

        descent_rate = float(self._cfg.get("descent_rate_mps", 0.20))
        return LandingCommand(vx=vx, vy=vy, vz=descent_rate, yaw_rate=0.0, should_land=False, reason="aligning")
