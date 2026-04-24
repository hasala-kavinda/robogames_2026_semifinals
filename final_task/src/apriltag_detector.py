"""AprilTag detection and airport metadata decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TagDetection:
    """Structured AprilTag detection used by mission logic."""

    tag_id: int
    country_code: int
    airport_status: int
    reachable_airports: int
    center: Tuple[int, int]
    perimeter_px: float
    corners: np.ndarray


class AprilTagDetector:
    """Detect AprilTags and decode airport metadata from tag IDs."""

    def __init__(self, config: dict):
        self._cfg = config
        dictionary_name = str(config.get("dictionary", "DICT_APRILTAG_36h11"))
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._params = cv2.aruco.DetectorParameters()
        self._use_aruco_detector = hasattr(cv2.aruco, "ArucoDetector")
        self._legacy_detect_markers = getattr(cv2.aruco, "detectMarkers", None)
        if self._use_aruco_detector:
            self._detector: Any = cv2.aruco.ArucoDetector(
                self._dictionary,
                self._params,
            )
        else:
            self._detector = None
            if self._legacy_detect_markers is None:
                raise RuntimeError(
                    "OpenCV ArUco detectMarkers API is unavailable"
                )

    @staticmethod
    def decode_airport_metadata(tag_id: int) -> Optional[Dict[str, int]]:
        """Decode airport metadata from a 3-digit airport tag ID."""
        if tag_id < 100 or tag_id > 999:
            return None

        country_code = tag_id // 100
        airport_status = (tag_id // 10) % 10
        reachable_airports = tag_id % 10

        if airport_status not in (0, 1):
            return None

        return {
            "country_code": country_code,
            "airport_status": airport_status,
            "reachable_airports": reachable_airports,
        }

    def detect(self, frame_bgr: np.ndarray) -> List[TagDetection]:
        """
        Return filtered detections from the region where ground tags are
        expected.
        """
        height = frame_bgr.shape[0]
        top = int(height * float(self._cfg.get("roi_top_ratio", 0.15)))
        bottom = int(height * float(self._cfg.get("roi_bottom_ratio", 0.95)))

        roi = frame_bgr[top:bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self._use_aruco_detector and self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            detect_markers = self._legacy_detect_markers
            if detect_markers is None:
                return []
            corners, ids, _ = detect_markers(
                gray,
                self._dictionary,
                parameters=self._params,
            )

        if ids is None:
            return []

        min_perimeter = float(self._cfg.get("min_perimeter_px", 50.0))
        detections: List[TagDetection] = []

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            tag_id = int(marker_id)
            metadata = self.decode_airport_metadata(tag_id)
            if metadata is None:
                continue

            pts = marker_corners[0]
            perimeter = cv2.arcLength(pts.astype(np.float32), True)
            if perimeter < min_perimeter:
                # Tiny detections are often distant noise and unstable under
                # motion blur.
                continue

            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1])) + top
            shifted_corners = pts.copy()
            shifted_corners[:, 1] += top

            detections.append(
                TagDetection(
                    tag_id=tag_id,
                    country_code=metadata["country_code"],
                    airport_status=metadata["airport_status"],
                    reachable_airports=metadata["reachable_airports"],
                    center=(center_x, center_y),
                    perimeter_px=float(perimeter),
                    corners=shifted_corners,
                )
            )

        return detections

    def select_target(
        self,
        detections: List[TagDetection],
        target_tag_id: Optional[int],
        frame_width: int,
    ) -> Optional[TagDetection]:
        """Pick target tag by ID or nearest to image center."""
        if not detections:
            return None

        if target_tag_id is not None:
            for item in detections:
                if item.tag_id == target_tag_id:
                    return item
            return None

        center_x = frame_width // 2
        return min(detections, key=lambda d: abs(d.center[0] - center_x))
