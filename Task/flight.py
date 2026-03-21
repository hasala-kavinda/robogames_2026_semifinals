import cv2
import numpy as np
import time
from control import Control
from sensor import Camera


class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()

        # --- Shared State Variables (Thread Safe-ish) ---
        self.current_error = 0.0
        self.line_visible = False
        self.debug_frame = None

        # --- PD Controller Tuning ---
        self.kp = 1.5  # Proportional: Steers towards the line
        self.kd = 0.8  # Derivative: Dampens wobbling on curves
        self.prev_error = 0.0
        self.target_speed = 0.5  # m/s

    def process_frame(self, frame):
        """
        Callback from the camera thread.
        Slices the grayscale image to look ahead and updates self.current_error.
        """
        # 1. Blur and Threshold (using your original values)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

        height, width = mask.shape

        # 2. Define Look-Ahead ROI (e.g., top 20% to 40% of the screen)
        search_top = int(height * 0.2)
        search_bottom = int(height * 0.4)

        # Black out everything outside the ROI
        mask[0:search_top, 0:width] = 0
        mask[search_bottom:height, 0:width] = 0

        # 3. Find contours in the sliced mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Create a color version of the frame JUST for debugging drawings
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])

                # Normalize error between -1 (left) and 1 (right)
                self.current_error = (cx - (width / 2)) / (width / 2)
                self.line_visible = True

                # Draw a red dot on the detected center
                cv2.circle(
                    display_frame,
                    (cx, int((search_top + search_bottom) / 2)),
                    8,
                    (0, 0, 255),
                    -1,
                )
        else:
            self.line_visible = False

        # Draw green ROI box for debugging
        cv2.rectangle(
            display_frame, (0, search_top), (width, search_bottom), (0, 255, 0), 2
        )

        # Pass the BGR frame to the main thread for display
        self.debug_frame = display_frame

    def line_follow(self):
        """
        Main control loop running at ~20Hz.
        Reads the latest error and applies PD control.
        """
        print("Starting PD Line Following...")

        while True:
            # 1. Display the vision feed in the main thread (prevents OpenCV crashes)
            if self.debug_frame is not None:
                cv2.imshow("Drone Vision", self.debug_frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            # 2. Control Logic
            if self.line_visible:
                # PD Math
                error = self.current_error
                derivative = error - self.prev_error

                yaw_rate = (error * self.kp) + (derivative * self.kd)
                self.prev_error = error

                # Send velocity command
                self.control.set_velocity_body(self.target_speed, 0, 0, yaw_rate)
            else:
                # Hover in place if the line is completely lost
                self.control.set_velocity_body(0, 0, 0, 0)

            # Run loop at roughly 20Hz (0.05s)
            time.sleep(0.05)

    def start(self):
        """Start the processing"""
        self.camera.start_thread(self.process_frame)
        self.control.set_mode("GUIDED")
        self.control.arm_motors()

        # Wait for takeoff to finish before following line
        self.control.takeoff(2.0)  # Using 2.0m as flight alt
        time.sleep(1)  # Brief pause to stabilize

        self.line_follow()

    def __del__(self):
        self.camera.stop_thread()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain
