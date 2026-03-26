from collections import deque
from enum import Enum
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from control import Control
from sensor import Camera
from vision import (
    AprilTagDetector,
    LineDetectionResult,
    TagDetectionResult,
    YellowLineDetector,
    draw_debug_overlays,
)


# Requested-country configuration.
# If only one country is needed, set country2 to 0.
country1 = 1
country2 = 2
Airports = [country1, country2]


class MissionState(Enum):
    INIT = "INIT"
    FOLLOW_LINE = "FOLLOW_LINE"
    VISUAL_SERVO = "VISUAL_SERVO"
    LAND_WAIT = "LAND_WAIT"
    RETAKEOFF = "RETAKEOFF"
    COMPLETE = "COMPLETE"
    FAILSAFE = "FAILSAFE"


class Brain:
    def __init__(self):
        self.control = Control(max_altitude=3.0)
        self.camera = Camera()

        # Perception modules for line tracking and AprilTag metadata parsing.
        self.line_detector = YellowLineDetector()
        self.tag_detector = AprilTagDetector()

        # Thread-shared perception state.
        self._lock = threading.Lock()
        self.latest_line = LineDetectionResult(False, 0.0, 0.0, (0, 0), None)
        self.latest_tags: List[TagDetectionResult] = []
        self.debug_frame: Optional[np.ndarray] = None

        # Mission state.
        self.state = MissionState.INIT
        self.mission_start_time = 0.0
        self.mission_timeout_s = 240.0

        # PD tuning and command shaping for curved line following.
        self.kp = 1.1
        self.kd = 0.4
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.target_speed = 0.3
        self.search_yaw_rate = 0.35

        # Requested-country order. Zero is ignored by definition.
        self.target_countries = [c for c in Airports if c != 0]
        self.target_index = 0

        # Airport knowledge base discovered dynamically at runtime.
        self.airport_info: Dict[int, TagDetectionResult] = {}
        self.airport_graph: Dict[int, Set[int]] = {}
        self.last_seen_airport: Optional[int] = None
        self.last_landed_airport: Optional[int] = None
        self.graph_revision = 0

        # Active route state produced by BFS/DFS planning.
        self.active_planned_path: List[int] = []
        self.expected_next_airport: Optional[int] = None
        self.route_country: Optional[int] = None
        self.route_start_airport: Optional[int] = None
        self.route_graph_revision = -1

        # Completion and de-duplication tracking.
        self.completed_countries: Set[int] = set()
        self.landed_airports: Set[int] = set()
        self.last_land_attempt_time: Dict[int, float] = {}
        self.land_retry_cooldown_s = 8.0

        # Visual servoing tuning and state.
        self.servo_kp = 0.3
        self.servo_kd = 0.2
        self.servo_tolerance_px = 60
        self.servo_timeout_s = 300.0
        self.max_servo_velocity = 0.1
        self.servo_start_time = 0.0
        self.servo_target_tag: Optional[TagDetectionResult] = None
        self.prev_servo_error: Tuple[float, float] = (0.0, 0.0)
        self.servo_result: Optional[Dict[str, object]] = None

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

    def _update_airport_knowledge(self, tags: List[TagDetectionResult]):
        """Update airport nodes and infer edges from sequential detections."""
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
        """Breadth-first search for shortest known route to any goal."""
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
        """Depth-first search fallback used when BFS has no known route."""
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
        """Find a known route to a safe airport in the requested country."""
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

        # Prefer BFS for shortest known path, then DFS fallback if needed.
        return self._bfs_path(start, goals) or self._dfs_path(start, goals)

    def _route_start_airport(self) -> Optional[int]:
        return self.last_landed_airport or self.last_seen_airport

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

    def _should_land_here(self, tag: TagDetectionResult) -> bool:
        """Validate airport safety and country constraints before landing."""
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
            # Avoid repeatedly landing at the same airport during a mission.
            return False

        return True

    def _apply_line_following(self, line: LineDetectionResult):
        """Run PD line following with smoothing for curved segments."""
        if line.visible:
            self.filtered_error = (0.7 * self.filtered_error) + (0.3 * line.error)
            derivative = self.filtered_error - self.prev_error
            yaw_rate = (self.filtered_error * self.kp) + (derivative * self.kd)
            yaw_rate = float(np.clip(yaw_rate, -1.2, 1.2))
            self.prev_error = self.filtered_error

            self.control.set_velocity_body(self.target_speed, 0, 0, yaw_rate)
        else:
            # If the line is temporarily lost, gently rotate to reacquire it.
            self.control.set_velocity_body(0.0, 0.0, 0.0, self.search_yaw_rate)

    def process_frame(self, frame_raw: np.ndarray):
        """Camera callback: run detectors and update mission knowledge."""
        frame_bgr = frame_raw.copy()

        line_result = self.line_detector.detect(frame_bgr)
        tags = self.tag_detector.detect(frame_bgr)

        servo_result: Optional[Dict[str, object]] = None
        if self.state == MissionState.VISUAL_SERVO and self.servo_target_tag:
            # Find the fresh detection of the current target tag in this frame.
            matching = next(
                (t for t in tags if t.tag_id == self.servo_target_tag.tag_id),
                None,
            )
            if matching is not None:
                h, w = frame_bgr.shape[:2]
                dx_px = matching.center[0] - (w / 2.0)
                dy_px = matching.center[1] - (h / 2.0)
                servo_result = {
                    "visible": True,
                    "error_x": dx_px / (w / 2.0),
                    "error_y": dy_px / (h / 2.0),
                    "pixel_error": (dx_px, dy_px),
                }

        with self._lock:
            self.latest_line = line_result
            self.latest_tags = tags
            self.servo_result = servo_result
            self._update_airport_knowledge(tags)

            self.debug_frame = draw_debug_overlays(
                frame_bgr,
                line_result,
                tags,
                self._current_target_country(),
                self.state.value,
                (self._mission_elapsed() if self.mission_start_time > 0 else 0.0),
            )

    def _land_wait_takeoff_cycle(
        self,
        landing_tag: TagDetectionResult,
    ) -> bool:
        """Land, wait 4 seconds, then take off when mission continues."""
        self.state = MissionState.LAND_WAIT
        self.last_land_attempt_time[landing_tag.tag_id] = time.time()
        self.control.set_velocity_body(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.5)

        if not self.control.land(timeout=40):
            return False

        self.landed_airports.add(landing_tag.tag_id)
        self.last_landed_airport = landing_tag.tag_id
        self.completed_countries.add(landing_tag.country_code)

        # Required dwell duration at airport.
        time.sleep(4.0)

        if self._current_target_country() == landing_tag.country_code:
            self.target_index += 1

        self._refresh_route_plan(force=True)

        if self._current_target_country() is None:
            self.state = MissionState.COMPLETE
            return True

        self.state = MissionState.RETAKEOFF
        self.control.set_mode("GUIDED")
        self.control.arm_motors()
        if not self.control.takeoff(2.0, timeout=30):
            return False

        self.state = MissionState.FOLLOW_LINE
        return True

    def _failsafe_land(self):
        """Stop movement and land safely when mission must terminate."""
        self.state = MissionState.FAILSAFE
        self.control.set_velocity_body(0.0, 0.0, 0.0, 0.0)
        self.control.land(timeout=40)

    def _snapshot_perception(
        self,
    ) -> Tuple[LineDetectionResult, List[TagDetectionResult], Optional[np.ndarray]]:
        """Read latest perception state atomically from camera callback thread."""
        with self._lock:
            line = self.latest_line
            tags = list(self.latest_tags)
            frame = self.debug_frame.copy() if self.debug_frame is not None else None
        return line, tags, frame

    def _render_debug_and_check_stop(self, frame: Optional[np.ndarray]) -> bool:
        """Render debug frame and return True if manual stop is requested."""
        if frame is None:
            return False

        cv2.imshow("Drone Vision", frame)
        return cv2.waitKey(1) == ord("q")

    @staticmethod
    def _select_primary_tag(
        tags: List[TagDetectionResult],
        frame: Optional[np.ndarray],
    ) -> TagDetectionResult:
        """Choose the tag nearest to image center as current airport candidate."""
        _, width = frame.shape[:2] if frame is not None else (0, 0)
        center_x = width / 2 if width > 0 else 0

        def _center_distance(tag: TagDetectionResult) -> float:
            return abs(tag.center[0] - center_x)

        return min(tags, key=_center_distance)

    def _handle_detected_tags(
        self,
        tags: List[TagDetectionResult],
        frame: Optional[np.ndarray],
    ):
        """Handle tag-driven decisions while line following continues."""
        if not tags:
            return

        self._refresh_route_plan()

        tag: Optional[TagDetectionResult] = None
        if self.expected_next_airport is not None:
            tag = next(
                (t for t in tags if t.tag_id == self.expected_next_airport),
                None,
            )
            if tag is None:
                # Planned next airport is not visible yet; keep exploring forward.
                return

        if tag is None:
            tag = self._select_primary_tag(tags, frame)
        if tag.airport_status == 0:
            print(f"Skipping unsafe airport tag {tag.tag_id}.")
            return

        target_country = self._current_target_country()
        if target_country is not None and tag.country_code != target_country:
            print(
                f"Skipping tag {tag.tag_id}: country {tag.country_code} "
                f"!= target {target_country}."
            )
            return

        if not self._should_land_here(tag):
            if self.expected_next_airport is not None:
                print(
                    f"Ignoring tag {tag.tag_id}; expected next airport "
                    f"is {self.expected_next_airport}."
                )
            return

        print(f"Targeting airport tag {tag.tag_id} for country {tag.country_code}.")
        self.servo_target_tag = tag
        self.servo_start_time = time.time()
        self.prev_servo_error = (0.0, 0.0)
        self.state = MissionState.VISUAL_SERVO

    def _servo_to_tag(self) -> Tuple[bool, bool]:
        """
        Drive body-frame velocities to center the target tag.

        Returns (converged, timed_out).
        """
        with self._lock:
            servo_result = self.servo_result

        if servo_result is None or not servo_result.get("visible", False):
            # Hold position when the tag is temporarily lost to avoid drift.
            self.control.set_velocity_body(0.0, 0.0, 0.0, 0.0)
            return False, False

        raw_error_x = servo_result.get("error_x", 0.0)
        raw_error_y = servo_result.get("error_y", 0.0)
        if not isinstance(raw_error_x, (int, float)):
            raw_error_x = 0.0
        if not isinstance(raw_error_y, (int, float)):
            raw_error_y = 0.0

        error_x = float(raw_error_x)
        error_y = float(raw_error_y)

        pixel_error_raw = servo_result.get("pixel_error", (0.0, 0.0))
        if (
            isinstance(pixel_error_raw, tuple)
            and len(pixel_error_raw) == 2
            and isinstance(pixel_error_raw[0], (int, float))
            and isinstance(pixel_error_raw[1], (int, float))
        ):
            pixel_dx = float(pixel_error_raw[0])
            pixel_dy = float(pixel_error_raw[1])
        else:
            pixel_dx, pixel_dy = 0.0, 0.0

        deriv_x = error_x - self.prev_servo_error[0]
        deriv_y = error_y - self.prev_servo_error[1]

        cmd_x = float(
            np.clip(
                (self.servo_kp * error_x) + (self.servo_kd * deriv_x),
                -self.max_servo_velocity,
                self.max_servo_velocity,
            )
        )
        cmd_y = float(
            np.clip(
                (self.servo_kp * error_y) + (self.servo_kd * deriv_y),
                -self.max_servo_velocity,
                self.max_servo_velocity,
            )
        )

        self.prev_servo_error = (error_x, error_y)

        # Map image-plane error to body-frame velocity commands.
        vx_cmd = -cmd_y  # forward/back to correct vertical image error
        vy_cmd = cmd_x  # lateral to correct horizontal image error
        self.control.set_velocity_body(vx_cmd, vy_cmd, 0.0, 0.0)

        converged = (
            abs(pixel_dx) <= self.servo_tolerance_px
            and abs(pixel_dy) <= self.servo_tolerance_px
        )
        timed_out = (time.time() - self.servo_start_time) >= self.servo_timeout_s
        return converged, timed_out

    def _log_planned_path(self):
        """Compatibility wrapper: keep route plan refreshed during mission."""
        self._refresh_route_plan()

    def start(self):
        """Run until all requested countries are serviced or timeout occurs."""
        if not self.target_countries:
            raise ValueError("Airports contains no requested countries.")

        self.camera.start_thread(self.process_frame)
        self.control.set_mode("GUIDED")
        self.control.arm_motors()

        if not self.control.takeoff(2.0, timeout=30):
            raise RuntimeError("Initial takeoff failed")

        self.state = MissionState.FOLLOW_LINE
        self.mission_start_time = time.time()

        while True:
            if self._timed_out():
                print("Mission timeout reached. Triggering failsafe landing.")
                self._failsafe_land()
                break

            line, tags, frame = self._snapshot_perception()

            if self._render_debug_and_check_stop(frame):
                print("Manual stop requested. Triggering failsafe landing.")
                self._failsafe_land()
                break

            if self.state == MissionState.COMPLETE:
                print("Mission complete: landed on all requested airports.")
                break

            if self.state == MissionState.FOLLOW_LINE:
                self._apply_line_following(line)
                self._refresh_route_plan()
                self._handle_detected_tags(tags, frame)

            elif self.state == MissionState.VISUAL_SERVO:
                converged, timed_out = self._servo_to_tag()
                if converged and self.servo_target_tag is not None:
                    if not self._land_wait_takeoff_cycle(self.servo_target_tag):
                        break
                    self.servo_target_tag = None
                    self.servo_result = None
                elif timed_out:
                    print("Visual servoing timed out. Triggering failsafe landing.")
                    self._failsafe_land()
                    break

            time.sleep(0.05)

    def stop(self):
        """Explicit shutdown to ensure clean camera and OpenCV teardown."""
        self.camera.stop_thread()
        cv2.destroyAllWindows()

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
        brain._failsafe_land()
    finally:
        brain.stop()
