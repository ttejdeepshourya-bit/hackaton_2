"""
Desktop OpenCV Driver Drowsiness Monitor
----------------------------------------
Standalone desktop application using OpenCV and MediaPipe.
Runs with minimal overhead and real-time Heads-Up Display (HUD).

Keyboard Controls:
    'q' : Quit application
    'r' : Reset alert counters
    'm' : Toggle audio beep mute/unmute
    '+' / '=' : Increase EAR threshold by 0.01
    '-' : Decrease EAR threshold by 0.01
"""

import sys
import time
import cv2

from drowsiness_detector import DrowsinessDetector
from audio_alert import get_audio_alert


def run_desktop_app(
    camera_index: int = 0,
    ear_threshold: float = 0.24,
    consecutive_frames: int = 25,
    enable_sound: bool = True
):
    print("=" * 65)
    print(" AI DRIVER DROWSINESS DETECTION SYSTEM (DESKTOP MODE) ")
    print(" B.Tech CSE AIML College Project Prototype")
    print("=" * 65)
    print(f"[INFO] Initializing webcam on device index {camera_index}...")

    # Open webcam with DirectShow on Windows for fast startup
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARNING] cv2.CAP_DSHOW failed. Trying default backend for index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("\n" + "!" * 65)
        print(f"[ERROR] Could not access webcam at index {camera_index}!")
        print("TROUBLESHOOTING TIPS:")
        print(" 1. Check if another app is using the webcam (Zoom, Teams, Discord, Browser).")
        print(" 2. In Windows Settings -> Privacy & Security -> Camera, make sure")
        print("    'Let apps access your camera' is turned ON.")
        print(" 3. If using an external USB webcam, try camera index 1 or 2:")
        print("    py desktop_app.py --camera 1")
        print("!" * 65 + "\n")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Initializing MediaPipe FaceLandmarker...")
    detector = DrowsinessDetector(
        ear_threshold=ear_threshold,
        consecutive_frames_threshold=consecutive_frames,
        time_threshold_seconds=1.5
    )
    audio = get_audio_alert(cooldown_seconds=1.0, enabled=enable_sound)

    print("\n[SUCCESS] Monitoring started! Press 'q' in the video window to quit.")
    print("Keyboard shortcuts:")
    print("  'q' : Quit")
    print("  'r' : Reset alert counter")
    print("  'm' : Toggle sound mute/unmute")
    print("  '+' / '-' : Adjust EAR threshold\n")

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame capture returned empty. Camera may have disconnected.")
                time.sleep(0.1)
                continue

            # Mirror frame horizontally for natural webcam experience
            frame = cv2.flip(frame, 1)

            # Process frame through AI detector
            result = detector.process_frame(frame, draw_landmarks=True)
            annotated_frame = result["annotated_frame"]

            # Trigger audio alert if drowsiness detected
            if result["is_drowsy"] and audio.enabled:
                audio.trigger(frequency=1400, duration_ms=300)

            # Calculate live FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Display FPS on top-right
            h, w = annotated_frame.shape[:2]
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f}",
                (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("Driver Drowsiness Detection System - Live Feed", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] User requested exit.")
                break
            elif key == ord('r'):
                detector.reset_state()
                print("[INFO] Alert counters reset to zero.")
            elif key == ord('m'):
                audio.set_enabled(not audio.enabled)
                status = "ENABLED" if audio.enabled else "MUTED"
                print(f"[INFO] Sound alert: {status}")
            elif key in [ord('+'), ord('=')]:
                detector.ear_threshold = round(detector.ear_threshold + 0.01, 2)
                print(f"[INFO] EAR Threshold increased to: {detector.ear_threshold}")
            elif key == ord('-'):
                detector.ear_threshold = max(0.10, round(detector.ear_threshold - 0.01, 2))
                print(f"[INFO] EAR Threshold decreased to: {detector.ear_threshold}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released and windows closed cleanly. Goodbye!")


if __name__ == "__main__":
    camera_idx = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        camera_idx = int(sys.argv[1])
    run_desktop_app(camera_index=camera_idx)
