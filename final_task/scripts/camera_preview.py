"""Live camera preview with line and AprilTag overlays for field tuning."""

from __future__ import annotations

import sys
import time
import importlib
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    april_tag_module = importlib.import_module("src.apriltag_detector")
    camera_module = importlib.import_module("src.camera_stream")
    config_module = importlib.import_module("src.config_loader")
    line_module = importlib.import_module("src.line_follower")

    april_tag_detector_cls = april_tag_module.AprilTagDetector
    camera_stream_cls = camera_module.CameraStream
    load_config = config_module.load_config
    yellow_line_follower_cls = line_module.YellowLineFollower

    config = load_config("config/defaults.json")

    camera = camera_stream_cls(config["camera"])
    line = yellow_line_follower_cls(config["line_follow"])
    tags = april_tag_detector_cls(config["apriltag"])

    camera.start()
    started = time.time()
    frames = 0

    try:
        while True:
            frame = camera.get_latest_frame(timeout_s=0.2)
            if frame is None:
                continue

            line_result = line.detect(frame)
            detections = tags.detect(frame)
            selected = tags.select_target(detections, None, frame.shape[1])

            top, bottom = line_result.roi
            cv2.rectangle(
                frame,
                (0, top),
                (frame.shape[1] - 1, bottom),
                (0, 255, 0),
                2,
            )
            if line_result.visible:
                cv2.circle(
                    frame,
                    (line_result.center_x, (top + bottom) // 2),
                    6,
                    (0, 0, 255),
                    -1,
                )

            for detection in detections:
                pts = detection.corners.astype(int)
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {detection.tag_id}",
                    (detection.center[0] + 8, detection.center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if selected is not None:
                cv2.circle(frame, selected.center, 8, (0, 255, 255), -1)

            frames += 1
            fps = frames / max(time.time() - started, 1e-6)
            cv2.putText(
                frame,
                (
                    f"fps={fps:.1f} line={line_result.visible} "
                    f"tags={len(detections)}"
                ),
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("final_task_camera_preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
