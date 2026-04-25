"""Mission entrypoint preserving semi-final behavior for final round."""

from __future__ import annotations

import argparse
from collections import deque
from enum import Enum
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple

from .apriltag_detector import AprilTagDetector, TagDetection
from .camera_stream import CameraStream
from .config_loader import PROJECT_ROOT, ConfigError, load_config
from .line_follower import LineResult, YellowLineFollower
from .mavlink_client import MavlinkClient


class MissionState(Enum):
    INIT = "INIT"
    FOLLOW_LINE = "FOLLOW_LINE"
    VISUAL_SERVO = "VISUAL_SERVO"
    LAND_WAIT = "LAND_WAIT"
    RETAKEOFF = "RETAKEOFF"
    COMPLETE = "COMPLETE"
    FAILSAFE = "FAILSAFE"


class MissionRunner:
    """Semi-final style multi-airport mission using modular components."""

    def __init__(self, config: dict, target_countries: List[int], debug: bool):
        self.config = config
        self.debug = debug

        self.mavlink = MavlinkClient(config["mavlink"])
        self.camera = CameraStream(config["camera"])
        self.line_detector = YellowLineFollower(config["line_follow"])
        self.tag_detector = AprilTagDetector(config["apriltag"])

        self.state = MissionState.INIT
        self.mission_start_time = 0.0
        self.mission_timeout_s = float(
            config["mission"].get("mission_timeout_s", 300.0)
        )

        self.kp = float(config["line_follow"].get("kp", 0.95))
        self.kd = float(config["line_follow"].get("kd", 0.30))
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.target_speed = float(config["line_follow"].get("forward_speed_mps", 0.30))
        self.search_yaw_rate = float(
            config["line_follow"].get("search_yaw_rate_rps", 0.25)
        )

        self.target_countries = [
            country for country in target_countries if country != 0
        ]
        self.target_index = 0

        self.airport_info: Dict[int, TagDetection] = {}
        self.airport_graph: Dict[int, Set[int]] = {}
        self.last_seen_airport: Optional[int] = None
        self.last_landed_airport: Optional[int] = None
        self.graph_revision = 0

        self.active_planned_path: List[int] = []
        self.expected_next_airport: Optional[int] = None
        self.route_country: Optional[int] = None
        self.route_start_airport: Optional[int] = None
        self.route_graph_revision = -1
        self.last_skip_log_key: Optional[Tuple[str, int]] = None

        self.completed_countries: Set[int] = set()
        self.landed_airports: Set[int] = set()
        self.last_land_attempt_time: Dict[int, float] = {}
        self.land_retry_cooldown_s = 8.0

        self.servo_kp = float(config["landing"].get("lateral_gain", 0.30))
        self.servo_kd = float(config["landing"].get("line_assist_gain", 0.20))
        self.servo_tolerance_px = int(
            config["landing"].get("tag_center_tolerance_px", 70)
        )
        self.servo_timeout_s = 300.0
        self.max_servo_velocity = float(
            config["landing"].get("max_lateral_speed_mps", 0.30)
        )
        self.servo_start_time = 0.0
        self.servo_target_tag: Optional[TagDetection] = None
        self.prev_servo_error: Tuple[float, float] = (0.0, 0.0)

    def _current_target_country(self) -> Optional[int]:
        if self.target_index >= len(self.target_countries):
            return None
        return self.target_countries[self.target_index]

    def _mission_elapsed(self) -> float:
        return time.time() - self.mission_start_time

    def _timed_out(self) -> bool:
        return self._mission_elapsed() >= self.mission_timeout_s

    def _record_edge(self, a: int, b: int):
        neighbors_a = self.airport_graph.setdefault(a, set())
        neighbors_b = self.airport_graph.setdefault(b, set())

        changed = False
        if b not in neighbors_a:
            neighbors_a.add(b)
            changed = True
        if a not in neighbors_b:
            neighbors_b.add(a)
            changed = True

        if changed:
            self.graph_revision += 1

    def _update_airport_knowledge(self, tags: List[TagDetection]):
        if not tags:
            return

        primary = tags[0]
        if primary.tag_id not in self.airport_graph:
            self.airport_graph[primary.tag_id] = set()
            self.graph_revision += 1

        self.airport_info[primary.tag_id] = primary

        if (
            self.last_seen_airport is not None
            and self.last_seen_airport != primary.tag_id
        ):
            self._record_edge(self.last_seen_airport, primary.tag_id)

        self.last_seen_airport = primary.tag_id

    def _bfs_path(self, start: int, goals: Set[int]) -> Optional[List[int]]:
        if start in goals:
            return [start]
        if start not in self.airport_graph:
            return None

        visited = {start}
        frontier: deque[Tuple[int, List[int]]] = deque([(start, [start])])
        while frontier:
            node, path = frontier.popleft()
            for nxt in self.airport_graph.get(node, set()):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt in goals:
                    return new_path
                visited.add(nxt)
                frontier.append((nxt, new_path))
        return None

    def _dfs_path(self, start: int, goals: Set[int]) -> Optional[List[int]]:
        if start in goals:
            return [start]
        if start not in self.airport_graph:
            return None

        visited = {start}
        stack: List[Tuple[int, List[int]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in self.airport_graph.get(node, set()):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt in goals:
                    return new_path
                visited.add(nxt)
                stack.append((nxt, new_path))
        return None

    def _find_candidate_path(self, country: int) -> Optional[List[int]]:
        goals = {
            tag_id
            for tag_id, info in self.airport_info.items()
            if info.country_code == country and info.airport_status == 1
        }
        if not goals:
            return None

        start = self.last_landed_airport or self.last_seen_airport
        if start is None:
            return None

        return self._bfs_path(start, goals) or self._dfs_path(start, goals)

    def _route_start_airport(self) -> Optional[int]:
        return self.last_seen_airport or self.last_landed_airport

    def _log_skip_once(self, reason: str, tag: TagDetection, detail: str):
        key = (reason, tag.tag_id)
        if self.last_skip_log_key == key:
            return
        print(detail)
        self.last_skip_log_key = key

    def _advance_route_through_transit(self, transit_tag: TagDetection):
        self.last_seen_airport = transit_tag.tag_id
        self._refresh_route_plan(force=True)

    def _derive_expected_next_airport(
        self,
        path: List[int],
        start_airport: Optional[int],
    ) -> Optional[int]:
        if not path:
            return None
        if start_airport is None:
            return path[0]

        if path[0] == start_airport:
            return path[1] if len(path) > 1 else path[0]

        if start_airport in path:
            idx = path.index(start_airport)
            return path[idx + 1] if idx + 1 < len(path) else path[idx]

        return path[0]

    def _clear_active_route(self):
        self.active_planned_path = []
        self.expected_next_airport = None

    def _refresh_route_plan(self, force: bool = False):
        target_country = self._current_target_country()
        start_airport = self._route_start_airport()

        if target_country is None:
            self._clear_active_route()
            self.route_country = None
            self.route_start_airport = None
            self.route_graph_revision = self.graph_revision
            return

        if start_airport is None:
            self._clear_active_route()
            self.route_country = target_country
            self.route_start_airport = None
            self.route_graph_revision = self.graph_revision
            return

        should_recompute = (
            force
            or self.route_country != target_country
            or self.route_start_airport != start_airport
            or self.route_graph_revision != self.graph_revision
        )
        if not should_recompute:
            return

        planned_path = self._find_candidate_path(target_country)
        if planned_path:
            self.active_planned_path = planned_path
            self.expected_next_airport = self._derive_expected_next_airport(
                planned_path,
                start_airport,
            )
            print(
                f"Route plan country {target_country}: {planned_path}, "
                f"next={self.expected_next_airport}"
            )
        else:
            self._clear_active_route()
            print(
                f"No known route yet for country {target_country} "
                f"from airport {start_airport}."
            )

        self.route_country = target_country
        self.route_start_airport = start_airport
        self.route_graph_revision = self.graph_revision

    def _should_land_here(self, tag: TagDetection) -> bool:
        target_country = self._current_target_country()
        if target_country is None:
            return False

        if (
            self.expected_next_airport is not None
            and tag.tag_id != self.expected_next_airport
        ):
            return False
        if tag.airport_status != 1:
            return False
        if tag.country_code != target_country:
            return False

        last_attempt = self.last_land_attempt_time.get(tag.tag_id, 0.0)
        if time.time() - last_attempt < self.land_retry_cooldown_s:
            return False

        if tag.tag_id in self.landed_airports:
            return False

        return True

    def _apply_line_following(self, line: LineResult):
        if line.visible:
            self.filtered_error = (0.7 * self.filtered_error) + (0.3 * line.error)
            derivative = self.filtered_error - self.prev_error
            yaw_rate = (self.filtered_error * self.kp) + (derivative * self.kd)
            yaw_rate = max(min(yaw_rate, 1.2), -1.2)
            self.prev_error = self.filtered_error
            self.mavlink.send_velocity_body(
                self.target_speed,
                0.0,
                0.0,
                yaw_rate,
            )
        else:
            self.mavlink.send_velocity_body(
                0.0,
                0.0,
                0.0,
                self.search_yaw_rate,
            )

    @staticmethod
    def _select_primary_tag(
        tags: List[TagDetection],
        frame_width: int,
    ) -> TagDetection:
        center_x = frame_width / 2.0
        return min(tags, key=lambda tag: abs(tag.center[0] - center_x))

    def _handle_tag_targeting(self, tag: TagDetection) -> None:
        if tag.airport_status == 0:
            self._handle_unsafe_tag(tag)
            return

        target_country = self._current_target_country()
        if target_country is not None and tag.country_code != target_country:
            self._handle_country_mismatch(tag, target_country)
            return

        if not self._should_land_here(tag):
            if self.expected_next_airport is not None:
                self._log_skip_once(
                    "not-expected",
                    tag,
                    (
                        f"Ignoring tag {tag.tag_id}; expected next airport "
                        f"is {self.expected_next_airport}."
                    ),
                )
            return

        print(f"Targeting airport tag {tag.tag_id} for country {tag.country_code}.")
        self.last_skip_log_key = None
        self.servo_target_tag = tag
        self.servo_start_time = time.time()
        self.prev_servo_error = (0.0, 0.0)
        self.state = MissionState.VISUAL_SERVO

    def _handle_unsafe_tag(self, tag: TagDetection) -> None:
        if (
            self.expected_next_airport is not None
            and tag.tag_id == self.expected_next_airport
        ):
            self._log_skip_once(
                "transit-unsafe",
                tag,
                (
                    f"Transit via unsafe airport tag {tag.tag_id}; "
                    "replanning route forward."
                ),
            )
            self._advance_route_through_transit(tag)
            return

        self._log_skip_once(
            "unsafe",
            tag,
            f"Skipping unsafe airport tag {tag.tag_id}.",
        )

    def _handle_country_mismatch(
        self,
        tag: TagDetection,
        target_country: int,
    ) -> None:
        if (
            self.expected_next_airport is not None
            and tag.tag_id == self.expected_next_airport
        ):
            self._log_skip_once(
                "transit-country",
                tag,
                (
                    f"Transit via tag {tag.tag_id} country "
                    f"{tag.country_code}; replanning to target "
                    f"country {target_country}."
                ),
            )
            self._advance_route_through_transit(tag)
            return

        self._log_skip_once(
            "country-mismatch",
            tag,
            (
                f"Skipping tag {tag.tag_id}: country "
                f"{tag.country_code} != target {target_country}."
            ),
        )

    def _handle_detected_tags(
        self,
        tags: List[TagDetection],
        frame_width: int,
    ):
        if not tags:
            return

        self._refresh_route_plan()

        tag: Optional[TagDetection] = None
        if self.expected_next_airport is not None:
            tag = next(
                (item for item in tags if item.tag_id == self.expected_next_airport),
                None,
            )
            if tag is None:
                return

        if tag is None:
            tag = self._select_primary_tag(tags, frame_width)
        self._handle_tag_targeting(tag)

    def _servo_to_tag(
        self,
        matching: Optional[TagDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[bool, bool]:
        if matching is None:
            self.mavlink.send_velocity_body(0.0, 0.0, 0.0, 0.0)
            return False, False

        error_x = (matching.center[0] - (frame_width / 2.0)) / (frame_width / 2.0)
        error_y = (matching.center[1] - (frame_height / 2.0)) / (frame_height / 2.0)

        deriv_x = error_x - self.prev_servo_error[0]
        deriv_y = error_y - self.prev_servo_error[1]

        cmd_x = (self.servo_kp * error_x) + (self.servo_kd * deriv_x)
        cmd_y = (self.servo_kp * error_y) + (self.servo_kd * deriv_y)

        cmd_x = max(
            min(cmd_x, self.max_servo_velocity),
            -self.max_servo_velocity,
        )
        cmd_y = max(
            min(cmd_y, self.max_servo_velocity),
            -self.max_servo_velocity,
        )

        self.prev_servo_error = (error_x, error_y)

        vx_cmd = -cmd_y
        vy_cmd = cmd_x
        self.mavlink.send_velocity_body(vx_cmd, vy_cmd, 0.0, 0.0)

        pixel_dx = matching.center[0] - (frame_width / 2.0)
        pixel_dy = matching.center[1] - (frame_height / 2.0)
        converged = (
            abs(pixel_dx) <= self.servo_tolerance_px
            and abs(pixel_dy) <= self.servo_tolerance_px
        )
        timed_out = (time.time() - self.servo_start_time) >= self.servo_timeout_s
        return converged, timed_out

    def _frame_loop_step(self, frame):
        line = self.line_detector.detect(frame)
        tags = self.tag_detector.detect(frame)
        self._update_airport_knowledge(tags)

        if self.state == MissionState.FOLLOW_LINE:
            self._apply_line_following(line)
            self._refresh_route_plan()
            self._handle_detected_tags(tags, frame.shape[1])
        elif self.state == MissionState.VISUAL_SERVO:
            self._handle_visual_servo(tags, frame)

        return line, tags

    def _handle_visual_servo(self, tags, frame):
        matching = None
        if self.servo_target_tag is not None:
            matching = next(
                (tag for tag in tags if tag.tag_id == self.servo_target_tag.tag_id),
                None,
            )

        converged, timed_out = self._servo_to_tag(
            matching,
            frame.shape[1],
            frame.shape[0],
        )

        if converged and self.servo_target_tag is not None:
            if not self._land_wait_takeoff_cycle(self.servo_target_tag):
                self._failsafe_land()
                return
            self.servo_target_tag = None

        elif timed_out:
            print("Visual servo timed out. Triggering failsafe landing.")
            self._failsafe_land()

    def _run_loop(self, dt_target: float):
        while True:
            loop_started = time.time()

            if self._timed_out():
                print("Mission timeout reached. Triggering failsafe landing.")
                self._failsafe_land()
                break

            if self.state == MissionState.COMPLETE:
                print("Mission complete: landed on requested airports.")
                break

            frame = self.camera.get_latest_frame(timeout_s=dt_target)
            if frame is None:
                self.mavlink.send_velocity_body(
                    0.0,
                    0.0,
                    0.0,
                    self.search_yaw_rate,
                )
                self._sleep_to_rate(loop_started, dt_target)
                continue

            line, tags = self._frame_loop_step(frame)

            if self.debug:
                print(
                    f"[mission] state={self.state.value} "
                    f"target_country={self._current_target_country()} "
                    f"tags={len(tags)} line_visible={line.visible}"
                )

            self._sleep_to_rate(loop_started, dt_target)

    def _land_wait_takeoff_cycle(self, landing_tag: TagDetection) -> bool:
        self.state = MissionState.LAND_WAIT
        self.last_land_attempt_time[landing_tag.tag_id] = time.time()
        self.mavlink.stop_motion()
        time.sleep(0.5)

        try:
            self.mavlink.land()
        except TimeoutError:
            return False

        self.landed_airports.add(landing_tag.tag_id)
        self.last_landed_airport = landing_tag.tag_id
        self.completed_countries.add(landing_tag.country_code)

        time.sleep(4.0)

        if self._current_target_country() == landing_tag.country_code:
            self.target_index += 1

        self._refresh_route_plan(force=True)

        if self._current_target_country() is None:
            self.state = MissionState.COMPLETE
            return True

        self.state = MissionState.RETAKEOFF
        self.mavlink.set_mode("GUIDED")
        try:
            self.mavlink.arm()
            target_altitude = float(
                self.config["mission"].get("target_altitude_m", 1.6)
            )
            self.mavlink.takeoff(target_altitude)
        except TimeoutError:
            return False

        self.state = MissionState.FOLLOW_LINE
        return True

    def _failsafe_land(self):
        self.state = MissionState.FAILSAFE
        self.mavlink.stop_motion()
        try:
            self.mavlink.land()
        except TimeoutError:
            print("Failsafe landing timed out.")

    @staticmethod
    def _sleep_to_rate(started: float, period_s: float) -> None:
        remaining = period_s - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)

    def run(self):
        if not self.target_countries:
            raise ValueError("No requested countries configured for mission.")

        loop_hz = float(self.config["mission"].get("loop_hz", 15.0))
        dt_target = 1.0 / max(loop_hz, 1.0)
        target_altitude = float(self.config["mission"].get("target_altitude_m", 1.6))
        mode = str(self.config["mission"].get("mode", "GUIDED"))

        self.camera.start()

        try:
            self.mavlink.connect()
            self.mavlink.set_mode(mode)
            self.mavlink.arm()
            self.mavlink.takeoff(target_altitude)

            self.state = MissionState.FOLLOW_LINE
            self.mission_start_time = time.time()

            self._run_loop(dt_target)
        finally:
            self.camera.stop()


def _parse_countries(value: str) -> List[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Countries list cannot be empty")
    return [int(part) for part in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final-round mission runner (semi-final behavior)",
    )
    parser.add_argument(
        "--config",
        default="config/defaults.json",
        help="Relative or absolute path to JSON config override",
    )
    parser.add_argument(
        "--countries",
        default=None,
        help="Comma separated target countries (example: 1,2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for mission state and targeting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override_path = Path(args.config)
    if not override_path.is_absolute():
        override_path = (PROJECT_ROOT / override_path).resolve()

    if not override_path.exists():
        hint = ""
        if override_path == (PROJECT_ROOT / "config" / "local.json").resolve():
            hint = (
                " (create it with: cp config/hardware.example.json config/local.json)"
            )
        raise SystemExit(f"Config file not found: {override_path}{hint}")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if args.countries:
        target_countries = _parse_countries(args.countries)
    else:
        target_countries = list(config["mission"].get("requested_countries", [1, 2]))

    runner = MissionRunner(
        config,
        target_countries=target_countries,
        debug=args.debug,
    )
    runner.run()


if __name__ == "__main__":
    main()
