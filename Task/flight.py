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

        # Completion and de-duplication tracking.
        self.completed_countries: Set[int] = set()
        self.landed_airports: Set[int] = set()
        self.last_land_attempt_time: Dict[int, float] = {}
        self.land_retry_cooldown_s = 8.0

    def _current_target_country(self) -> Optional[int]:
        if self.target_index >= len(self.target_countries):
            return None
        return self.target_countries[self.target_index]

    def _mission_elapsed(self) -> float:
        return time.time() - self.mission_start_time

    def _timed_out(self) -> bool:
        return self._mission_elapsed() >= self.mission_timeout_s

    def _record_edge(self, a: int, b: int):
        self.airport_graph.setdefault(a, set()).add(b)
        self.airport_graph.setdefault(b, set()).add(a)

    def _update_airport_knowledge(self, tags: List[TagDetectionResult]):
        """Update airport nodes and infer edges from sequential detections."""
        if not tags:
            return

        primary = tags[0]
        self.airport_info[primary.tag_id] = primary
        self.airport_graph.setdefault(primary.tag_id, set())

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

    def _should_land_here(self, tag: TagDetectionResult) -> bool:
        """Validate airport safety and country constraints before landing."""
        target_country = self._current_target_country()
        if target_country is None:
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
        # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = frame_raw.copy()

        line_result = self.line_detector.detect(frame_bgr)
        tags = self.tag_detector.detect(frame_bgr)

        with self._lock:
            self.latest_line = line_result
            self.latest_tags = tags
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
        return min(tags, key=lambda t, cx=center_x: abs(t.center[0] - cx))

    def _handle_detected_tags(
        self,
        tags: List[TagDetectionResult],
        frame: Optional[np.ndarray],
    ) -> bool:
        """Handle tag-driven decisions. Returns False when mission should stop."""
        if not tags:
            return True

        tag = self._select_primary_tag(tags, frame)
        if tag.airport_status == 0:
            print(f"Skipping unsafe airport tag {tag.tag_id}.")
            return True

        if not self._should_land_here(tag):
            return True

        print(f"Landing at airport tag {tag.tag_id} for country {tag.country_code}.")
        if self._land_wait_takeoff_cycle(tag):
            return True

        print("Landing cycle failed. Triggering failsafe.")
        self._failsafe_land()
        return False

    def _log_planned_path(self):
        """Run BFS/DFS planning on the discovered graph for current target."""
        target_country = self._current_target_country()
        if target_country is None:
            return

        planned_path = self._find_candidate_path(target_country)
        if planned_path is not None:
            print(f"Planned path to country {target_country}: {planned_path}")

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

            self._apply_line_following(line)
            self._log_planned_path()

            if not self._handle_detected_tags(tags, frame):
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
