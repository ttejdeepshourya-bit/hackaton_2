"""
Drowsiness Detection Engine
---------------------------
Core computer vision module using MediaPipe FaceLandmarker and Eye Aspect Ratio (EAR)
to detect prolonged eye closure as an indicator of potential driver drowsiness.

Mathematical Foundation:
    The Eye Aspect Ratio (EAR) is defined as:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    where:
        p1, p4: horizontal eye corner landmark coordinates
        p2, p6 and p3, p5: vertical eyelid landmark pairs

When eyes are open, EAR is approximately 0.28 - 0.38.
When eyes close, EAR drops sharply below 0.20 - 0.24.
Prolonged closure (exceeding consecutive frame / time threshold) triggers an alert.
"""

import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from download_model import ensure_model_exists


# MediaPipe 468/478 Face Mesh Landmark Indices for Eyes
# 6 landmarks per eye used for Eye Aspect Ratio calculation:
# [P1: outer_corner, P2: top_1, P3: top_2, P4: inner_corner, P5: bottom_2, P6: bottom_1]
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Complete perimeter contour indices for visual rendering
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def calculate_euclidean_distance(point_a, point_b) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def calculate_ear(eye_points) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) from 6 2D landmark coordinates.
    Points order: [P1, P2, P3, P4, P5, P6]
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    # Vertical eyelid distances
    vertical_1 = calculate_euclidean_distance(p2, p6)
    vertical_2 = calculate_euclidean_distance(p3, p5)

    # Horizontal eye length
    horizontal = calculate_euclidean_distance(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class DrowsinessDetector:
    """
    Processes video frames to extract face mesh landmarks, compute EAR,
    and monitor eye closure duration for drowsiness detection.
    """

    def __init__(
        self,
        ear_threshold: float = 0.24,
        consecutive_frames_threshold: int = 25,
        time_threshold_seconds: float = 1.5,
        model_path: str = None
    ):
        """
        Initialize the detector.

        Args:
            ear_threshold: Threshold below which an eye is considered closed.
            consecutive_frames_threshold: Number of consecutive closed frames before triggering alert.
            time_threshold_seconds: Minimum time (seconds) of sustained closure to trigger alert.
            model_path: Path to face_landmarker.task model.
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.time_threshold_seconds = time_threshold_seconds

        # Ensure MediaPipe model is present
        if model_path is None:
            model_path = ensure_model_exists()
        self.model_path = model_path

        # Configure MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # State tracking variables
        self.closed_frames_count = 0
        self.closed_start_time = None
        self.closed_duration = 0.0
        self.is_drowsy = False
        self.total_alerts = 0
        self.alert_was_triggered = False

    def reset_state(self):
        """Resets all real-time counters and state tracking."""
        self.closed_frames_count = 0
        self.closed_start_time = None
        self.closed_duration = 0.0
        self.is_drowsy = False
        self.total_alerts = 0
        self.alert_was_triggered = False

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = True) -> dict:
        """
        Analyzes a single BGR camera frame and updates drowsiness metrics.

        Args:
            frame: OpenCV BGR image as numpy array.
            draw_landmarks: Whether to draw eye outlines and annotations on the frame.

        Returns:
            Dictionary containing processed frame, metrics, and alert status.
        """
        if frame is None or frame.size == 0:
            return {
                "annotated_frame": frame,
                "face_detected": False,
                "ear_avg": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "eye_status": "No Feed",
                "system_status": "Camera Error",
                "is_drowsy": False,
                "alert_triggered": False,
                "closed_frames": 0,
                "closed_duration": 0.0,
                "total_alerts": self.total_alerts
            }

        height, width = frame.shape[:2]
        output_frame = frame.copy()

        # Convert OpenCV BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform landmark detection
        try:
            detection_result = self.landmarker.detect(mp_image)
        except Exception as e:
            # Catch inference exceptions gracefully without crashing
            return {
                "annotated_frame": output_frame,
                "face_detected": False,
                "ear_avg": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "eye_status": "Unknown",
                "system_status": f"Inference Error: {str(e)[:30]}",
                "is_drowsy": False,
                "alert_triggered": False,
                "closed_frames": 0,
                "closed_duration": 0.0,
                "total_alerts": self.total_alerts
            }

        face_landmarks_list = detection_result.face_landmarks

        # Handle missing face detection gracefully
        if not face_landmarks_list or len(face_landmarks_list) == 0:
            self.closed_frames_count = 0
            self.closed_start_time = None
            self.closed_duration = 0.0
            self.is_drowsy = False
            self.alert_was_triggered = False

            # Draw visual indicator for missing face
            cv2.putText(
                output_frame,
                "NO FACE DETECTED",
                (30, 45),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            return {
                "annotated_frame": output_frame,
                "face_detected": False,
                "ear_avg": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "eye_status": "Unknown",
                "system_status": "No Face Detected",
                "is_drowsy": False,
                "alert_triggered": False,
                "closed_frames": 0,
                "closed_duration": 0.0,
                "total_alerts": self.total_alerts
            }

        # Face detected: extract landmarks for the primary face
        face_landmarks = face_landmarks_list[0]

        # Extract 2D pixel coordinates for eye landmarks
        left_eye_pts = []
        for idx in LEFT_EYE_LANDMARKS:
            lm = face_landmarks[idx]
            left_eye_pts.append((int(lm.x * width), int(lm.y * height)))

        right_eye_pts = []
        for idx in RIGHT_EYE_LANDMARKS:
            lm = face_landmarks[idx]
            right_eye_pts.append((int(lm.x * width), int(lm.y * height)))

        # Calculate EAR for both eyes
        ear_left = calculate_ear(left_eye_pts)
        ear_right = calculate_ear(right_eye_pts)
        ear_avg = (ear_left + ear_right) / 2.0

        current_time = time.time()
        just_triggered_alert = False

        # Evaluate eye state against threshold
        if ear_avg < self.ear_threshold:
            # Eyes are closed
            self.closed_frames_count += 1
            if self.closed_start_time is None:
                self.closed_start_time = current_time
            self.closed_duration = current_time - self.closed_start_time
            eye_status = "Closed"

            # Check if duration or frame threshold is reached
            threshold_reached = (
                self.closed_frames_count >= self.consecutive_frames_threshold
                or self.closed_duration >= self.time_threshold_seconds
            )

            if threshold_reached:
                if not self.is_drowsy:
                    # Rising edge: new drowsiness event
                    self.is_drowsy = True
                    self.total_alerts += 1
                    just_triggered_alert = True
                system_status = "DROWSINESS DETECTED!"
            else:
                system_status = f"Eyes Closing ({self.closed_duration:.1f}s)"
        else:
            # Eyes are open: reset closure counters
            self.closed_frames_count = 0
            self.closed_start_time = None
            self.closed_duration = 0.0
            self.is_drowsy = False
            self.alert_was_triggered = False
            eye_status = "Open"
            system_status = "Attentive / Normal"

        # Optional visual annotations on frame
        if draw_landmarks:
            self._draw_annotations(
                output_frame,
                face_landmarks,
                width,
                height,
                left_eye_pts,
                right_eye_pts,
                ear_avg,
                eye_status,
                system_status
            )

        return {
            "annotated_frame": output_frame,
            "face_detected": True,
            "ear_avg": float(ear_avg),
            "ear_left": float(ear_left),
            "ear_right": float(ear_right),
            "eye_status": eye_status,
            "system_status": system_status,
            "is_drowsy": self.is_drowsy,
            "alert_triggered": just_triggered_alert,
            "closed_frames": self.closed_frames_count,
            "closed_duration": float(self.closed_duration),
            "total_alerts": self.total_alerts
        }

    def _draw_annotations(
        self,
        frame: np.ndarray,
        landmarks,
        width: int,
        height: int,
        left_pts,
        right_pts,
        ear_avg: float,
        eye_status: str,
        system_status: str
    ):
        """Draws professional eye contours, landmarks, and status HUD on the frame."""
        # 1. Draw smooth eye contours
        left_contour_pts = np.array(
            [[int(landmarks[idx].x * width), int(landmarks[idx].y * height)] for idx in LEFT_EYE_CONTOUR],
            dtype=np.int32
        )
        right_contour_pts = np.array(
            [[int(landmarks[idx].x * width), int(landmarks[idx].y * height)] for idx in RIGHT_EYE_CONTOUR],
            dtype=np.int32
        )

        contour_color = (0, 0, 255) if self.is_drowsy else (0, 255, 0)
        cv2.polylines(frame, [left_contour_pts], isClosed=True, color=contour_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [right_contour_pts], isClosed=True, color=contour_color, thickness=1, lineType=cv2.LINE_AA)

        # 2. Draw dots on key measurement points
        for pt in left_pts + right_pts:
            cv2.circle(frame, pt, 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)

        # 3. Status Banner at Top
        if self.is_drowsy:
            # Red warning banner
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 220), -1)
            cv2.putText(
                frame,
                "** WARNING: DROWSINESS DETECTED! WAKE UP! **",
                (int(width * 0.05), 34),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        else:
            # Semi-transparent dark header banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 45), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            status_color = (0, 255, 120) if eye_status == "Open" else (0, 200, 255)
            cv2.putText(
                frame,
                f"Eye Status: {eye_status.upper()}",
                (20, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA
            )

        # 4. Bottom HUD telemetry card
        hud_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - hud_h), (width, height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        ear_text = f"EAR: {ear_avg:.2f} (Thresh: {self.ear_threshold:.2f})"
        alerts_text = f"Total Alerts: {self.total_alerts}"

        cv2.putText(
            frame,
            ear_text,
            (20, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            alerts_text,
            (width - 200, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )


if __name__ == "__main__":
    print("[TEST] Testing DrowsinessDetector initialization...")
    detector = DrowsinessDetector()
    print("[TEST] Detector initialized successfully!")
    
    # Test on synthetic frame
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.process_frame(dummy)
    print("[TEST] Dummy frame result - Face Detected:", res["face_detected"])
    print("[TEST] Dummy frame result - System Status:", res["system_status"])
