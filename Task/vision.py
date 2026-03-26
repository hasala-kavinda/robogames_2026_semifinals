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
    roi: Tuple[int, int, int, int]  # top, bottom, left, right
    center_x: Optional[int]


@dataclass
class TagDetectionResult:
    """AprilTag metadata decoded from the numeric tag ID."""

    tag_id: int
    country_code: int
    airport_status: int
    reachable_airports: int
    center: Tuple[int, int]


@dataclass
class ServoingResult:
    """Result from visual servoing to center tag in frame."""

    visible: bool
    error_x: float  # Normalized horizontal error in [-1, 1]
    error_y: float  # Normalized vertical error in [-1, 1]
    pixel_error: Tuple[int, int]  # Raw pixel offset (dx, dy)
    confidence: float  # Detection confidence [0, 1]
    tag_size_pixels: int  # Approximate tag width in pixels


class YellowLineDetector:
    """Detect yellow line center in the upper-middle look-ahead region."""

    def __init__(self):
        # HSV bounds tuned for common bright yellow road markings.
        self.lower_yellow = np.array([18, 70, 70], dtype=np.uint8)
        self.upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

        # Memory for Momentum Tracking (Solution 2)
        self.last_cx: Optional[int] = None

    def detect(self, frame_bgr: np.ndarray) -> LineDetectionResult:
        """Return normalized lateral error in [-1, 1] and confidence."""
        height, width = frame_bgr.shape[:2]

        # Define the vertical AND horizontal ROI
        search_top = int(height * 0.2)
        search_bottom = int(height * 0.45)

        # Chop off the outer 25% on both sides to avoid peripheral noise
        search_left = int(width * 0.15)
        search_right = int(width * 0.85)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Focus on the narrowed look-ahead box
        roi_mask = np.zeros_like(mask)
        roi_mask[search_top:search_bottom, search_left:search_right] = 255
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

        # Pre-calculate valid contours with valid moments to avoid zero-division
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 50:  # Ignore tiny noise artifacts
                M = cv2.moments(c)
                if M["m00"] > 0:
                    valid_contours.append((c, M, area))

        if not valid_contours:
            self.last_cx = None  # Reset memory if the line is completely lost
            return LineDetectionResult(
                visible=False,
                error=0.0,
                confidence=0.0,
                roi=(search_top, search_bottom, search_left, search_right),
                center_x=None,
            )

        # --- MOMENTUM TRACKING LOGIC ---
        if self.last_cx is None:
            best_data = max(valid_contours, key=lambda item: item[2])
        else:
            best_data = min(
                valid_contours,
                key=lambda item: abs(
                    int(item[1]["m10"] / item[1]["m00"]) - self.last_cx
                ),
            )

        best_contour, moments, area = best_data
        center_x = int(moments["m10"] / moments["m00"])
        self.last_cx = center_x
        # -------------------------------------------

        error = (center_x - (width / 2.0)) / (width / 2.0)

        roi_area = max((search_bottom - search_top) * (search_right - search_left), 1)
        confidence = float(min(area / (roi_area * 0.25), 1.0))

        return LineDetectionResult(
            visible=True,
            error=float(np.clip(error, -1.0, 1.0)),
            confidence=confidence,
            roi=(search_top, search_bottom, search_left, search_right),
            center_x=center_x,
        )


class AprilTagDetector:
    """Detect AprilTags and decode airport metadata from tag ID digits."""

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self.detect_markers = self.detector.detectMarkers
        else:
            self.dictionary = dictionary
            self.parameters = parameters
            detect_markers = getattr(cv2.aruco, "detectMarkers", None)
            if detect_markers is None:
                raise RuntimeError("OpenCV aruco.detectMarkers API is unavailable")
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

    def calculate_servoing_error(
        self,
        tag: TagDetectionResult,
        frame_shape: Tuple[int, int, int],
    ) -> ServoingResult:
        """
        Calculate normalized errors for visual servoing (tag centering).

        Args:
            tag: TagDetectionResult with tag center coordinates
            frame_shape: (height, width, channels) tuple

        Returns:
            ServoingResult with normalized errors and tag size estimate
        """
        height, width, _ = frame_shape
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        tag_cx, tag_cy = tag.center

        # Pixel offset from frame center
        pixel_error_x = tag_cx - frame_center_x
        pixel_error_y = tag_cy - frame_center_y

        # Normalize to [-1, 1] range (similar to line following)
        error_x = pixel_error_x / (width / 2.0)
        error_y = pixel_error_y / (height / 2.0)

        # Clamp to valid range
        error_x = float(np.clip(error_x, -1.0, 1.0))
        error_y = float(np.clip(error_y, -1.0, 1.0))

        # Estimate tag size (assume roughly square in image plane)
        # This is a coarse estimate; ideally we'd track corner points
        # For now, assume detection means tag is reasonably visible
        tag_size_estimate = (
            50  # pixels (placeholder; could enhance with corner tracking)
        )

        # Confidence: tag inside frame and visible
        visible = 0 <= tag_cx < width and 0 <= tag_cy < height
        confidence = 1.0 if visible else 0.0

        return ServoingResult(
            visible=visible,
            error_x=error_x,
            error_y=error_y,
            pixel_error=(int(pixel_error_x), int(pixel_error_y)),
            confidence=confidence,
            tag_size_pixels=tag_size_estimate,
        )

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
    servo_result: Optional["ServoingResult"] = None,
    servo_timeout_s: Optional[float] = None,
) -> np.ndarray:
    """Render overlays so perception and mission decisions are visible."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    top, bottom, left, right = line_result.roi
    cv2.rectangle(out, (left, top), (right - 1, bottom), (0, 255, 0), 2)

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

    # Visual servoing overlay: draw target crosshair and error vector
    if servo_result is not None and servo_result.visible:
        frame_center_x = w // 2
        frame_center_y = h // 2

        # Draw target crosshair (frame center)
        crosshair_size = 20
        cv2.line(
            out,
            (frame_center_x - crosshair_size, frame_center_y),
            (frame_center_x + crosshair_size, frame_center_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            out,
            (frame_center_x, frame_center_y - crosshair_size),
            (frame_center_x, frame_center_y + crosshair_size),
            (0, 255, 0),
            2,
        )

        # Draw error vector from target to tag center (green line)
        tag_cx, tag_cy = servo_result.pixel_error
        actual_tag_x = frame_center_x + tag_cx
        actual_tag_y = frame_center_y + tag_cy
        cv2.arrowedLine(
            out,
            (frame_center_x, frame_center_y),
            (actual_tag_x, actual_tag_y),
            (0, 255, 0),
            2,
            tipLength=0.3,
        )

        # Label servo errors and timeout
        servo_label = (
            f"Servo Error: ({servo_result.error_x:+.2f}, {servo_result.error_y:+.2f}) "
            f"Px:({servo_result.pixel_error[0]:+d}, {servo_result.pixel_error[1]:+d})"
        )
        cv2.putText(
            out,
            servo_label,
            (8, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if servo_timeout_s is not None:
            timeout_label = f"Servo Timeout: {elapsed_s:.1f}s / {servo_timeout_s:.1f}s"
            cv2.putText(
                out,
                timeout_label,
                (8, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    header = f"State:{state_name} TargetCountry:{target_country} T:{elapsed_s:.1f}s"
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
