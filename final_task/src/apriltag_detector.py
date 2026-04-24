"""AprilTag detection tuned for ground-path tags in forward camera feed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TagDetection:
    """Structured tag detection result used by mission and landing logic."""

    tag_id: int
    center: Tuple[int, int]
    perimeter_px: float
    corners: np.ndarray


class AprilTagDetector:
    """Detect AprilTags using OpenCV ArUco AprilTag dictionary."""

    def __init__(self, config: dict):
        self._cfg = config
        dictionary_name = str(config.get("dictionary", "DICT_APRILTAG_36h11"))
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._detect_fn = self._detector.detectMarkers
            self._dictionary = None
            self._params = None
        else:
            # Compatibility path for OpenCV versions without ArucoDetector class.
            self._detector = None
            self._detect_fn = cv2.aruco.detectMarkers
            self._dictionary = dictionary
            self._params = params

    def detect(self, frame_bgr: np.ndarray) -> List[TagDetection]:
        """Return filtered detections from region where ground tags are expected."""
        height = frame_bgr.shape[0]
        top = int(height * float(self._cfg.get("roi_top_ratio", 0.15)))
        bottom = int(height * float(self._cfg.get("roi_bottom_ratio", 0.95)))

        roi = frame_bgr[top:bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self._dictionary is None:
            corners, ids, _ = self._detect_fn(gray)
        else:
            corners, ids, _ = self._detect_fn(gray, self._dictionary, parameters=self._params)

        if ids is None:
            return []

        min_perimeter = float(self._cfg.get("min_perimeter_px", 50.0))
        detections: List[TagDetection] = []

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners[0]
            perimeter = cv2.arcLength(pts.astype(np.float32), True)
            if perimeter < min_perimeter:
                # Tiny detections are often distant noise and unstable under motion blur.
                continue

            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1])) + top
            shifted_corners = pts.copy()
            shifted_corners[:, 1] += top

            detections.append(
                TagDetection(
                    tag_id=int(marker_id),
                    center=(center_x, center_y),
                    perimeter_px=float(perimeter),
                    corners=shifted_corners,
                )
            )

        return detections

    def select_target(self, detections: List[TagDetection], target_tag_id: Optional[int], frame_width: int) -> Optional[TagDetection]:
        """Pick target tag by configured ID, otherwise the one nearest image center."""
        if not detections:
            return None

        if target_tag_id is not None:
            for item in detections:
                if item.tag_id == target_tag_id:
                    return item
            return None

        center_x = frame_width // 2
        return min(detections, key=lambda d: abs(d.center[0] - center_x))
