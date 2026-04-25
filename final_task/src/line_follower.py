"""Yellow-path detection with adaptive filtering for changing outdoor lighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LineResult:
    """Line tracking result used by mission and landing assist."""

    visible: bool
    error: float
    confidence: float
    center_x: int
    roi: Tuple[int, int]


class YellowLineFollower:
    """Estimate lateral path error from yellow lane-like markings."""

    def __init__(self, config: dict):
        self._cfg = config
        self._base_lower = np.array(config["hsv_lower"], dtype=np.uint8)
        self._base_upper = np.array(config["hsv_upper"], dtype=np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> LineResult:
        """Return normalized lateral error in [-1, 1] and detection confidence."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
            return LineResult(False, 0.0, 0.0, 0, (0, 0))

        height, width = frame_bgr.shape[:2]
        top = int(height * float(self._cfg.get("roi_top_ratio", 0.2)))
        bottom = int(height * float(self._cfg.get("roi_bottom_ratio", 0.55)))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower, upper = self._adaptive_bounds(hsv)

        mask = cv2.inRange(hsv, lower, upper)
        roi_mask = np.zeros_like(mask)
        roi_mask[top:bottom, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        kernel_size = int(self._cfg.get("morph_kernel", 5))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return LineResult(False, 0.0, 0.0, width // 2, (top, bottom))

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < float(self._cfg.get("min_contour_area_px", 250.0)):
            return LineResult(False, 0.0, 0.0, width // 2, (top, bottom))

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return LineResult(False, 0.0, 0.0, width // 2, (top, bottom))

        center_x = int(moments["m10"] / moments["m00"])
        error = (center_x - (width / 2.0)) / (width / 2.0)

        roi_area = max((bottom - top) * width, 1)
        confidence = float(min(area / (roi_area * 0.30), 1.0))

        return LineResult(
            visible=True,
            error=float(np.clip(error, -1.0, 1.0)),
            confidence=confidence,
            center_x=center_x,
            roi=(top, bottom),
        )

    def _adaptive_bounds(self, hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust yellow mask lower bounds when exposure shifts in real flight."""
        if not bool(self._cfg.get("adaptive_filtering", True)):
            return self._base_lower, self._base_upper

        # Use median brightness to avoid overreacting to short glare highlights.
        median_v = float(np.median(hsv[:, :, 2]))
        median_s = float(np.median(hsv[:, :, 1]))

        dynamic_lower = self._base_lower.copy()

        # In low light, permit darker and less saturated yellow pixels.
        if median_v < 80:
            dynamic_lower[2] = max(
                int(self._cfg.get("adaptive_min_value_floor", 45)),
                int(self._base_lower[2] * 0.7),
            )
            dynamic_lower[1] = max(
                int(self._cfg.get("adaptive_min_saturation_floor", 60)),
                int(self._base_lower[1] * 0.8),
            )

        # In haze/glare, saturation can drop; reduce saturation floor moderately.
        if median_s < 70:
            dynamic_lower[1] = max(int(dynamic_lower[1] * 0.8), 45)

        return dynamic_lower, self._base_upper
