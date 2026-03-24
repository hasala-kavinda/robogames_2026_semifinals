"""
Vision utilities for yellow line following and AprilTag airport decoding.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LineDetectionResult:
    """Result from yellow line detection in the look-ahead ROI."""

    visible: bool
    error: float
    confidence: float
    roi: Tuple[int, int]
    center_x: Optional[int]


@dataclass
class TagDetectionResult:
    """AprilTag metadata decoded from the numeric tag ID."""

    tag_id: int
    country_code: int
    airport_status: int
    reachable_airports: int
    center: Tuple[int, int]


class YellowLineDetector:
    """Detect yellow line center in the upper-middle look-ahead region."""

    def __init__(self):
        # HSV bounds tuned for common bright yellow road markings.
        self.lower_yellow = np.array([18, 70, 70], dtype=np.uint8)
        self.upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> LineDetectionResult:
        """Return normalized lateral error in [-1, 1] and confidence."""
        height, width = frame_bgr.shape[:2]

        search_top = int(height * 0.2)
        search_bottom = int(height * 0.45)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Focus on look-ahead band to avoid floor noise near the drone.
        roi_mask = np.zeros_like(mask)
        roi_mask[search_top:search_bottom, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        # Clean binary mask for stable contour extraction on curves.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return LineDetectionResult(
                visible=False,
                error=0.0,
                confidence=0.0,
                roi=(search_top, search_bottom),
                center_x=None,
            )

        # Largest contour is assumed to represent the dominant line segment.
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return LineDetectionResult(
                visible=False,
                error=0.0,
                confidence=0.0,
                roi=(search_top, search_bottom),
                center_x=None,
            )

        center_x = int(moments["m10"] / moments["m00"])
        error = (center_x - (width / 2.0)) / (width / 2.0)

        # Confidence scales with occupied ROI area and is capped in [0, 1].
        roi_area = max((search_bottom - search_top) * width, 1)
        confidence = float(min(area / (roi_area * 0.25), 1.0))

        return LineDetectionResult(
            visible=True,
            error=float(np.clip(error, -1.0, 1.0)),
            confidence=confidence,
            roi=(search_top, search_bottom),
            center_x=center_x,
        )


class AprilTagDetector:
    """Detect AprilTags and decode airport metadata from tag ID digits."""

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        parameters = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self.detect_markers = self.detector.detectMarkers
        else:
            self.dictionary = dictionary
            self.parameters = parameters
            detect_markers = getattr(cv2.aruco, "detectMarkers", None)
            if detect_markers is None:
                raise RuntimeError(
                    "OpenCV aruco.detectMarkers API is unavailable"
                )
            self.detect_markers = detect_markers

    def detect(self, frame_bgr: np.ndarray) -> List[TagDetectionResult]:
        """Return decoded tags found in the frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if hasattr(self, "dictionary"):
            corners, ids, _ = self.detect_markers(
                gray,
                self.dictionary,
                parameters=self.parameters,
            )
        else:
            corners, ids, _ = self.detect_markers(gray)

        results: List[TagDetectionResult] = []
        if ids is None:
            return results

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            c = marker_corners[0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))

            decoded = self.decode_airport_metadata(int(marker_id))
            if decoded is None:
                continue

            results.append(
                TagDetectionResult(
                    tag_id=int(marker_id),
                    country_code=decoded["country_code"],
                    airport_status=decoded["airport_status"],
                    reachable_airports=decoded["reachable_airports"],
                    center=(center_x, center_y),
                )
            )

        return results

    @staticmethod
    def decode_airport_metadata(tag_id: int) -> Optional[Dict[str, int]]:
        """
        Decode airport metadata from a three-digit tag ID:
        digit1=country, digit2=status(1 safe / 0 unsafe),
        digit3=reachable count.
        """
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


def draw_debug_overlays(
    frame_bgr: np.ndarray,
    line_result: LineDetectionResult,
    tags: List[TagDetectionResult],
    target_country: Optional[int],
    state_name: str,
    elapsed_s: float,
) -> np.ndarray:
    """Render overlays so perception and mission decisions are visible."""
    out = frame_bgr.copy()
    _, w = out.shape[:2]

    top, bottom = line_result.roi
    cv2.rectangle(out, (0, top), (w - 1, bottom), (0, 255, 0), 2)

    if line_result.visible and line_result.center_x is not None:
        cy = (top + bottom) // 2
        cv2.circle(out, (line_result.center_x, cy), 8, (0, 0, 255), -1)
        cv2.line(out, (w // 2, top), (w // 2, bottom), (255, 255, 0), 1)

    for tag in tags:
        cv2.circle(out, tag.center, 6, (255, 0, 0), -1)
        label = (
            f"ID:{tag.tag_id} C:{tag.country_code} "
            f"S:{tag.airport_status} R:{tag.reachable_airports}"
        )
        cv2.putText(
            out,
            label,
            (tag.center[0] + 8, max(tag.center[1] - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    header = (
        f"State:{state_name} "
        f"TargetCountry:{target_country} T:{elapsed_s:.1f}s"
    )
    cv2.putText(
        out,
        header,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out
