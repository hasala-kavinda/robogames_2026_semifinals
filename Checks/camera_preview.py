"""Single-file camera preview with simple red-color HSV detection.

Usage examples:
  python3 Checks/camera_preview.py
  python3 Checks/camera_preview.py --source tcp --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np


# =========================
# TUNING: CHANGE VALUES HERE
# =========================
# OpenCV HSV ranges:
# - H: 0..179
# - S: 0..255
# - V: 0..255
#
# Current defaults target RED (low-hue red range).
DEFAULT_H_MIN = 0
DEFAULT_S_MIN = 120
DEFAULT_V_MIN = 70
DEFAULT_H_MAX = 10
DEFAULT_S_MAX = 255
DEFAULT_V_MAX = 255

# Ignore tiny blobs/noise below this contour area in pixels.
DEFAULT_MIN_AREA = 100

# Morphology kernel size for mask cleanup (odd values like 3, 5, 7 are typical).
DEFAULT_KERNEL_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera preview and color detection")
    parser.add_argument("--source", choices=["webcam", "tcp"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", default="192.168.1.60")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--channels", type=int, default=3)

    parser.add_argument(
        "--h-min",
        type=int,
        default=DEFAULT_H_MIN,
        help="HSV lower hue bound (0..179)",
    )
    parser.add_argument(
        "--s-min",
        type=int,
        default=DEFAULT_S_MIN,
        help="HSV lower saturation bound (0..255)",
    )
    parser.add_argument(
        "--v-min",
        type=int,
        default=DEFAULT_V_MIN,
        help="HSV lower value bound (0..255)",
    )
    parser.add_argument(
        "--h-max",
        type=int,
        default=DEFAULT_H_MAX,
        help="HSV upper hue bound (0..179)",
    )
    parser.add_argument(
        "--s-max",
        type=int,
        default=DEFAULT_S_MAX,
        help="HSV upper saturation bound (0..255)",
    )
    parser.add_argument(
        "--v-max",
        type=int,
        default=DEFAULT_V_MAX,
        help="HSV upper value bound (0..255)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum contour area in pixels to accept detection",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help="Morphology kernel size for mask cleanup",
    )

    return parser.parse_args()


def open_ip_stream(host: str, port: int) -> Optional[cv2.VideoCapture]:
    stream_urls = [
        f"http://{host}:{port}/video",
        f"http://{host}:{port}/stream",
        f"http://{host}:{port}/live",
        f"http://{host}:{port}/mjpeg",
        f"http://{host}:{port}/video.mjpg",
        f"rtsp://{host}:{port}/stream",
        f"rtsp://{host}:{port}/video",
        f"http://{host}:{port}",
    ]

    for url in stream_urls:
        print(f"[camera] Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"[camera] Connected to: {url}")
            return cap
        cap.release()

    return None


def draw_color_detection(
    frame_bgr: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_area: int,
    kernel_size: int,
) -> tuple[np.ndarray, int, Optional[tuple[int, int]]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    area = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(largest))
        if area > min_area:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            moments = cv2.moments(largest)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center = (cx, cy)
                cv2.circle(frame_bgr, center, 6, (0, 0, 255), -1)

    return mask, area, center


def run_webcam(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    lower = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
    upper = np.array([args.h_max, args.s_max, args.v_max], dtype=np.uint8)
    kernel_size = max(1, int(args.kernel_size))

    started = time.time()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            mask, area, center = draw_color_detection(
                frame,
                lower,
                upper,
                min_area=args.min_area,
                kernel_size=kernel_size,
            )

            frames += 1
            fps = frames / max(time.time() - started, 1e-6)
            cv2.putText(
                frame,
                f"fps={fps:.1f} area={area} center={center}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("camera_preview", frame)
            cv2.imshow("color_mask", mask)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_tcp(args: argparse.Namespace) -> None:
    lower = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
    upper = np.array([args.h_max, args.s_max, args.v_max], dtype=np.uint8)
    kernel_size = max(1, int(args.kernel_size))

    started = time.time()
    frames = 0

    cap = open_ip_stream(args.host, args.port)
    if cap is None:
        raise RuntimeError(
            f"Could not open stream for host={args.host} port={args.port}"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] Frame read failed. Reconnecting...")
                cap.release()
                time.sleep(0.5)
                cap = open_ip_stream(args.host, args.port)
                if cap is None:
                    print("[camera] Reconnect failed")
                    break
                continue

            mask, area, center = draw_color_detection(
                frame,
                lower,
                upper,
                min_area=args.min_area,
                kernel_size=kernel_size,
            )

            frames += 1
            fps = frames / max(time.time() - started, 1e-6)
            cv2.putText(
                frame,
                f"fps={fps:.1f} area={area} center={center}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("camera_preview", frame)
            cv2.imshow("color_mask", mask)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.source == "webcam":
        run_webcam(args)
    else:
        run_tcp(args)


if __name__ == "__main__":
    main()
