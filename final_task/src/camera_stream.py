"""TCP camera client with low-latency frame buffering for onboard use."""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Continuously receive frames from camera server and keep latest samples."""

    def __init__(self, config: dict):
        self.host = str(config["host"])
        self.port = int(config["port"])
        self.socket_timeout_s = float(config.get("socket_timeout_s", 1.0))
        self.reconnect_delay_s = float(config.get("reconnect_delay_s", 1.0))
        self.channels = int(config.get("channels", 3))
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(config.get("max_queue_size", 2)))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop camera thread and release UI resources if any were used."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    def get_latest_frame(self, timeout_s: float = 0.0) -> Optional[np.ndarray]:
        """Return the latest frame; old frames are dropped to preserve control latency."""
        try:
            if timeout_s > 0.0:
                return self._queue.get(timeout=timeout_s)
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.socket_timeout_s)
                    sock.connect((self.host, self.port))
                    print(f"[camera] Connected to {self.host}:{self.port}")
                    while not self._stop_event.is_set():
                        frame = self._read_frame(sock)
                        if frame is None:
                            break
                        # Drop stale samples under load so controller always sees latest scene.
                        while self._queue.full():
                            try:
                                self._queue.get_nowait()
                            except queue.Empty:
                                break
                        self._queue.put(frame)
            except OSError as exc:
                print(f"[camera] Connection error: {exc}")

            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay_s)

    def _read_frame(self, sock: socket.socket) -> Optional[np.ndarray]:
        # Header format is width+height as unsigned short each.
        header = self._recv_exact(sock, 4)
        if header is None:
            return None

        width, height = struct.unpack("=HH", header)
        payload_len = width * height * self.channels
        payload = self._recv_exact(sock, payload_len)
        if payload is None:
            return None

        frame_rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, self.channels))
        # Convert to BGR because OpenCV processing and display expect BGR by default.
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < num_bytes and not self._stop_event.is_set():
            try:
                chunk = sock.recv(num_bytes - len(data))
            except socket.timeout:
                continue
            except ConnectionResetError:
                return None

            if not chunk:
                return None
            data.extend(chunk)

        if len(data) != num_bytes:
            return None
        return bytes(data)
