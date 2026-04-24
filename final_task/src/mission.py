"""Mission entrypoint for final-round onboard autonomy."""

from __future__ import annotations

import argparse
import time
from enum import Enum
from typing import Optional

from .apriltag_detector import AprilTagDetector
from .camera_stream import CameraStream
from .config_loader import load_config
from .landing_controller import LandingController
from .line_follower import YellowLineFollower
from .mavlink_client import MavlinkClient


class MissionState(Enum):
    INIT = "INIT"
    FOLLOW_LINE = "FOLLOW_LINE"
    TAG_ALIGN = "TAG_ALIGN"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    FAILSAFE = "FAILSAFE"


class MissionRunner:
    """Integrates MAVLink + camera + perception + landing logic in one loop."""

    def __init__(self, config: dict, target_tag_id: Optional[int], debug: bool = False):
        self.config = config
        self.target_tag_id = target_tag_id
        self.debug = debug

        self.mavlink = MavlinkClient(config["mavlink"])
        self.camera = CameraStream(config["camera"])
        self.apriltag = AprilTagDetector(config["apriltag"])
        self.line_follower = YellowLineFollower(config["line_follow"])
        self.landing = LandingController(config["landing"])

        self.state = MissionState.INIT
        self.prev_line_error = 0.0

    def run(self) -> None:
        """Run mission until successful landing or failsafe timeout."""
        loop_hz = float(self.config["mission"].get("loop_hz", 15.0))
        dt_target = 1.0 / max(loop_hz, 1.0)
        timeout_s = float(self.config["mission"].get("mission_timeout_s", 300.0))
        target_altitude = float(self.config["mission"].get("target_altitude_m", 1.6))
        mode = str(self.config["mission"].get("mode", "GUIDED"))

        started_at = time.time()
        self.camera.start()

        try:
            self.mavlink.connect()
            self.mavlink.set_mode(mode)
            self.mavlink.arm()
            self.mavlink.takeoff(target_altitude)
            self.state = MissionState.FOLLOW_LINE

            while True:
                loop_started = time.time()

                elapsed = loop_started - started_at
                if elapsed > timeout_s:
                    print("[mission] Timeout reached, entering failsafe landing")
                    self.state = MissionState.FAILSAFE

                if self.state == MissionState.FAILSAFE:
                    self.mavlink.stop_motion()
                    self.mavlink.land()
                    self.state = MissionState.COMPLETE
                    break

                frame = self.camera.get_latest_frame(timeout_s=dt_target)
                if frame is None:
                    # Missing frames can happen due to network jitter; hold with slow search yaw.
                    self.mavlink.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.2)
                    self._sleep_to_rate(loop_started, dt_target)
                    continue

                line = self.line_follower.detect(frame)
                detections = self.apriltag.detect(frame)
                selected = self.apriltag.select_target(
                    detections,
                    self.target_tag_id if self.target_tag_id is not None else self.config["apriltag"].get("target_tag_id"),
                    frame.shape[1],
                )

                if selected is not None:
                    self.state = MissionState.TAG_ALIGN
                    cmd = self.landing.compute_command(selected, line, frame.shape[1], frame.shape[0])
                    if cmd.should_land:
                        print("[mission] Landing criteria satisfied")
                        self.state = MissionState.LANDING
                        self.mavlink.stop_motion()
                        self.mavlink.land()
                        self.state = MissionState.COMPLETE
                        break

                    self.mavlink.send_velocity_body(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)

                elif line.visible:
                    self.state = MissionState.FOLLOW_LINE
                    line_cfg = self.config["line_follow"]
                    kp = float(line_cfg.get("kp", 0.95))
                    kd = float(line_cfg.get("kd", 0.30))
                    speed = float(line_cfg.get("forward_speed_mps", 0.30))
                    derivative = (line.error - self.prev_line_error) / max(dt_target, 1e-6)
                    self.prev_line_error = line.error

                    # Positive error means path is to the right; yaw right to center the path.
                    yaw_rate = (kp * line.error) + (kd * derivative)
                    self.mavlink.send_velocity_body(speed, 0.0, 0.0, yaw_rate)
                else:
                    # If both line and tags are missing, keep scanning with conservative yaw.
                    search_rate = float(self.config["line_follow"].get("search_yaw_rate_rps", 0.25))
                    self.mavlink.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=search_rate)

                if self.debug:
                    visible_tags = len(detections)
                    print(
                        f"[mission] state={self.state.value} "
                        f"line_visible={line.visible} line_error={line.error:.3f} tags={visible_tags}"
                    )

                self._sleep_to_rate(loop_started, dt_target)

            print("[mission] Mission completed")
        finally:
            self.camera.stop()

    @staticmethod
    def _sleep_to_rate(started: float, period_s: float) -> None:
        remaining = period_s - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-round drone mission runner")
    parser.add_argument(
        "--config",
        default="config/defaults.json",
        help="Relative or absolute path to JSON config override",
    )
    parser.add_argument(
        "--target-tag-id",
        type=int,
        default=None,
        help="Optional tag ID to prioritize during final landing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for state transitions and perception",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = MissionRunner(config, target_tag_id=args.target_tag_id, debug=args.debug)
    runner.run()


if __name__ == "__main__":
    main()
