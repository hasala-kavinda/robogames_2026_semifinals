import cv2
import numpy as np
import sys
import time


def get_video_feed(ip_address, port):
    """
    Get video feed from specified IP and port using OpenCV

    Args:
        ip_address (str): IP address of the stream
        port (int): Port number

    Returns:
        cv2.VideoCapture: Video capture object
    """
    # Construct the stream URL
    # Common stream formats - try different ones if needed
    stream_urls = [
        f"http://{ip_address}:{port}/video",
        f"http://{ip_address}:{port}/stream",
        f"http://{ip_address}:{port}/live",
        f"http://{ip_address}:{port}/mjpeg",
        f"http://{ip_address}:{port}/video.mjpg",
        f"rtsp://{ip_address}:{port}/stream",
        f"rtsp://{ip_address}:{port}/video",
    ]

    # Try each URL format
    for url in stream_urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)

        # Check if stream opened successfully
        if cap.isOpened():
            print(f"Successfully connected to: {url}")
            return cap

    # If no URL worked, try just the IP:port
    print(f"Trying default connection to {ip_address}:{port}")
    cap = cv2.VideoCapture(f"http://{ip_address}:{port}")

    if cap.isOpened():
        return cap
    else:
        print("Failed to connect to any stream URL")
        return None


def main():
    # Configuration
    IP_ADDRESS = "192.168.1.60"
    PORT = 9000

    print(f"Attempting to connect to {IP_ADDRESS}:{PORT}")

    # Get video feed
    cap = get_video_feed(IP_ADDRESS, PORT)

    if cap is None or not cap.isOpened():
        print("Error: Could not open video stream")
        print("\nTroubleshooting tips:")
        print("1. Check if the IP address is correct")
        print("2. Verify the device is reachable (ping command)")
        print("3. Check if the port is correct")
        print("4. Ensure the stream is active on the device")
        print("5. Check firewall settings")
        sys.exit(1)

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nStream Properties:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print("\nControls:")
    print("Press 'q' or 'ESC' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to record video")
    print("-" * 40)

    # Initialize video recording
    recording = False
    out = None
    frame_count = 0

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Reconnecting...")
            time.sleep(1)
            cap = get_video_feed(IP_ADDRESS, PORT)
            if cap is None or not cap.isOpened():
                print("Failed to reconnect")
                break
            continue

        # Display FPS on frame
        current_time = time.time()
        if not hasattr(main, "last_time"):
            main.last_time = current_time
            main.frame_counter = 0
        main.frame_counter += 1

        if current_time - main.last_time >= 1.0:
            fps_text = f"FPS: {main.frame_counter}"
            main.frame_counter = 0
            main.last_time = current_time
        else:
            fps_text = f"FPS: Calculating..."

        # Add text overlay
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Stream: {IP_ADDRESS}:{PORT}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if recording:
            cv2.putText(
                frame,
                "RECORDING...",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Display the frame
        cv2.imshow("IP Camera Feed", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # 'q' or ESC
            print("\nQuitting...")
            break

        elif key == ord("s"):  # Save screenshot
            screenshot_name = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved as: {screenshot_name}")

        elif key == ord("r"):  # Toggle recording
            if not recording:
                # Start recording
                video_name = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
                recording = True
                print(f"Recording started: {video_name}")
            else:
                # Stop recording
                recording = False
                if out:
                    out.release()
                print("Recording stopped")

    # Cleanup
    if recording and out:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
