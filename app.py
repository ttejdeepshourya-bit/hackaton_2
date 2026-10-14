"""
AI-Based Driver Drowsiness Detection System
-------------------------------------------
College Capstone Prototype - B.Tech CSE AIML
Built with Python, OpenCV, MediaPipe, and Streamlit.

Monitors real-time Eye Aspect Ratio (EAR) via laptop webcam.
Identifies sustained eye closure and triggers multi-modal audio/visual alerts.
Supports Live Laptop Stream, Browser Camera Snapshot, and Image Upload.
"""

import os
import sys
import time
import warnings
import collections

# Suppress underlying C++/absl/TF logging noise
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from drowsiness_detector import DrowsinessDetector
from audio_alert import get_audio_alert
from download_model import ensure_model_exists


# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection | AI System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphic styling, badges, and warning animations
st.markdown("""
<style>
    /* Global styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        margin-right: 0.5rem;
    }
    .badge-aiml {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    
    /* Metrics card container */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-lbl {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Warning banner styles */
    .alert-banner-drowsy {
        background: linear-gradient(90deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        border: 2px solid #ef4444;
        animation: pulse-red 1.2s infinite;
        margin-bottom: 1rem;
    }
    .alert-banner-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 0.75rem 1.2rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-bottom: 1rem;
    }
    @keyframes pulse-red {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.88; transform: scale(1.01); }
    }
    
    /* Disclaimer card */
    .disclaimer-card {
        background: #0f172a;
        border-left: 4px solid #f59e0b;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if "total_alerts" not in st.session_state:
    st.session_state.total_alerts = 0

if "ear_history" not in st.session_state:
    st.session_state.ear_history = collections.deque(maxlen=100)

if "event_logs" not in st.session_state:
    st.session_state.event_logs = []

if "model_ready" not in st.session_state:
    with st.spinner("Verifying MediaPipe AI Face Landmark model..."):
        try:
            model_path = ensure_model_exists()
            st.session_state.model_ready = True
            st.session_state.model_path = model_path
        except Exception as e:
            st.error(f"Failed to load AI model: {e}")
            st.session_state.model_ready = False


# ---------------------------------------------------------
# Sidebar Controls & Parameters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Operation Mode")
    app_mode = st.radio(
        "Choose Mode:",
        options=[
            "🎥 Live Webcam Stream (Laptop Demo)",
            "📸 Browser Camera Snapshot (Cloud Demo)",
            "📁 Upload Face Image"
        ],
        index=0,
        help="Use 'Live Webcam Stream' for continuous real-time monitoring on your laptop."
    )

    st.divider()

    if app_mode == "🎥 Live Webcam Stream (Laptop Demo)":
        st.markdown("### 🚦 Stream Controls")
        col_start, col_stop = st.columns(2)
        start_clicked = col_start.button("▶️ Start", type="primary", width="stretch", disabled=st.session_state.monitoring)
        stop_clicked = col_stop.button("⏹️ Stop", width="stretch", disabled=not st.session_state.monitoring)

        if start_clicked:
            st.session_state.monitoring = True
            st.rerun()

        if stop_clicked:
            st.session_state.monitoring = False
            st.rerun()

        st.divider()

    st.markdown("### ⚙️ Detection Parameters")
    
    ear_threshold = st.slider(
        "EAR Threshold (Eye Closure)",
        min_value=0.15,
        max_value=0.35,
        value=0.24,
        step=0.01,
        help="Eye Aspect Ratio below this value triggers eye closure detection."
    )

    consecutive_frames = st.slider(
        "Consecutive Closed Frames",
        min_value=10,
        max_value=60,
        value=25,
        step=5,
        help="Number of consecutive frames eyes must remain closed before drowsiness warning."
    )

    time_threshold = st.slider(
        "Sustained Duration (Seconds)",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.1,
        help="Minimum sustained closure duration to trigger warning."
    )

    st.divider()

    st.markdown("### 📷 Camera & Visuals")
    if app_mode == "🎥 Live Webcam Stream (Laptop Demo)":
        camera_index = st.selectbox(
            "Select Camera Index",
            options=[0, 1, 2],
            index=0,
            help="0 is usually your laptop's built-in webcam. 1 or 2 is an external USB camera."
        )

    show_landmarks = st.checkbox("Show Eye Landmark Contours", value=True)
    enable_sound = st.checkbox("Enable Audio Warning Beep", value=True)

    st.divider()

    if st.button("🔄 Reset Total Alerts & Logs", width="stretch"):
        st.session_state.total_alerts = 0
        st.session_state.ear_history.clear()
        st.session_state.event_logs.clear()
        st.success("Counters reset!")
        st.rerun()


# ---------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🚗 AI Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
st.markdown("""
<div>
    <span class="badge-pill">MediaPipe Face Mesh</span>
    <span class="badge-pill badge-aiml">Eye Aspect Ratio (EAR)</span>
    <span class="badge-pill">Real-Time Computer Vision</span>
    <span class="badge-pill">OpenCV + Streamlit</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Educational Prototype Notice:</strong> This project is developed for B.Tech CSE AIML academic presentation and demonstration purposes.
    It is not a certified automotive safety device. Never test or operate software while actively driving a motor vehicle.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dashboard Layout: Top Metric Cards
# ---------------------------------------------------------
metric_cols = st.columns(4)
metric_eye_placeholder = metric_cols[0].empty()
metric_ear_placeholder = metric_cols[1].empty()
metric_alerts_placeholder = metric_cols[2].empty()
metric_status_placeholder = metric_cols[3].empty()


def update_metrics(eye_status="Unknown", ear_val=0.0, total_alerts=0, monitoring=False, is_drowsy=False):
    """Renders the 4 live KPI metric cards."""
    # 1. Eye Status Card
    if eye_status == "Open":
        eye_color = "#10b981"
        eye_icon = "👁️ OPEN"
    elif eye_status == "Closed":
        eye_color = "#ef4444"
        eye_icon = "😴 CLOSED"
    else:
        eye_color = "#94a3b8"
        eye_icon = "❓ UNKNOWN"

    metric_eye_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Eye State</div>
        <div class="metric-val" style="color: {eye_color};">{eye_icon}</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Live EAR Value
    ear_color = "#ef4444" if (ear_val > 0 and ear_val < ear_threshold) else "#38bdf8"
    metric_ear_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Current EAR (Thresh: {ear_threshold:.2f})</div>
        <div class="metric-val" style="color: {ear_color};">{ear_val:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Total Alerts Count
    alerts_color = "#ef4444" if total_alerts > 0 else "#e2e8f0"
    metric_alerts_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Alerts Triggered</div>
        <div class="metric-val" style="color: {alerts_color};">🚨 {total_alerts}</div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Monitoring Status
    if monitoring:
        status_lbl = "ACTIVE"
        status_color = "#10b981"
    else:
        status_lbl = "STOPPED"
        status_color = "#64748b"

    metric_status_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">System Status</div>
        <div class="metric-val" style="color: {status_color};">{status_lbl}</div>
    </div>
    """, unsafe_allow_html=True)


# Render initial metric state
update_metrics(
    eye_status="Unknown",
    ear_val=0.0,
    total_alerts=st.session_state.total_alerts,
    monitoring=st.session_state.monitoring
)

# Placeholders for dynamic content
alert_banner_placeholder = st.empty()


# ---------------------------------------------------------
# MODE 1: LIVE WEBCAM STREAM (LAPTOP RUNNER)
# ---------------------------------------------------------
if app_mode == "🎥 Live Webcam Stream (Laptop Demo)":
    stop_bar_placeholder = st.empty()
    video_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.session_state.monitoring:
        # Quick Stop button right above the video feed for instant user response
        if stop_bar_placeholder.button("⏹️ Stop Live Monitoring", type="secondary", width="stretch", key="main_stop_btn"):
            st.session_state.monitoring = False
            st.rerun()

        # Initialize Detector and Audio
        try:
            detector = DrowsinessDetector(
                ear_threshold=ear_threshold,
                consecutive_frames_threshold=consecutive_frames,
                time_threshold_seconds=time_threshold,
                model_path=st.session_state.get("model_path")
            )
            audio = get_audio_alert(cooldown_seconds=1.0, enabled=enable_sound)
        except Exception as e:
            st.error(f"Failed to initialize computer vision pipeline: {e}")
            st.session_state.monitoring = False
            st.stop()

        # Attempt to open webcam
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(f"❌ Could not open webcam at Camera Index {camera_index}!")
            st.markdown("""
            **Troubleshooting suggestions:**
            - Ensure your webcam is not being used by another application (Zoom, Teams, Skype, or browser tab).
            - Verify camera permissions in **Windows Settings > Privacy & security > Camera**.
            - If using an external webcam, try changing **Select Camera Index** to `1` or `2` in the sidebar.
            """)
            st.session_state.monitoring = False
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        alert_banner_placeholder.markdown("""
        <div class="alert-banner-safe">
            🟢 Live Monitoring Active — Focus on the road. Eyes are being analyzed in real-time.
        </div>
        """, unsafe_allow_html=True)

        try:
            frame_counter = 0
            while st.session_state.monitoring:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("Webcam frame capture returned empty. Camera may be disconnected.")
                    time.sleep(0.1)
                    break

                # Mirror horizontally for natural webcam experience
                frame = cv2.flip(frame, 1)

                # AI Inference
                result = detector.process_frame(frame, draw_landmarks=show_landmarks)
                annotated_frame = result["annotated_frame"]

                # Update session total alerts
                st.session_state.total_alerts = result["total_alerts"]

                # Handle Audio and Banner Alerts
                if result["is_drowsy"]:
                    alert_banner_placeholder.markdown("""
                    <div class="alert-banner-drowsy">
                        ⚠️ ALERT: PROLONGED EYE CLOSURE DETECTED! WAKE UP!
                    </div>
                    """, unsafe_allow_html=True)

                    if enable_sound:
                        audio.trigger(frequency=1400, duration_ms=300)

                    # Log event if rising edge
                    if result["alert_triggered"]:
                        st.session_state.event_logs.append({
                            "Time": time.strftime("%H:%M:%S"),
                            "Event": "Drowsiness Warning",
                            "EAR": round(result["ear_avg"], 3),
                            "Duration (s)": round(result["closed_duration"], 1)
                        })
                else:
                    if result["face_detected"]:
                        alert_banner_placeholder.markdown("""
                        <div class="alert-banner-safe">
                            🟢 Driver Attentive — Normal Eye Aspect Ratio Detected.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        alert_banner_placeholder.markdown("""
                        <div class="alert-banner-safe" style="color: #f59e0b; border-color: rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.1);">
                            ⚠️ Notice: No face detected in frame. Please align with the webcam.
                        </div>
                        """, unsafe_allow_html=True)

                # Update KPI cards every 2 frames for smooth UI performance
                frame_counter += 1
                if frame_counter % 2 == 0:
                    update_metrics(
                        eye_status=result["eye_status"],
                        ear_val=result["ear_avg"],
                        total_alerts=st.session_state.total_alerts,
                        monitoring=True,
                        is_drowsy=result["is_drowsy"]
                    )

                    # Append to history buffer
                    if result["face_detected"]:
                        st.session_state.ear_history.append(result["ear_avg"])

                # Render live video frame using width="stretch"
                rgb_view = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_view, channels="RGB", width="stretch")

                # Smooth frame rate limiter (~30 FPS)
                time.sleep(0.02)

        finally:
            cap.release()
            cv2.destroyAllWindows()

    else:
        # When monitoring is stopped / idle
        alert_banner_placeholder.markdown("""
        <div class="alert-banner-safe" style="color: #94a3b8; border-color: #334155; background: #0f172a;">
            ⏸️ System Idle. Click <strong>'▶️ Start'</strong> in the sidebar to begin real-time drowsiness monitoring.
        </div>
        """, unsafe_allow_html=True)

        col_guide1, col_guide2 = st.columns(2)
        with col_guide1:
            st.markdown("### 🔬 How It Works")
            st.markdown("""
            1. **Facial Landmark Mesh**: The laptop webcam captures real-time video frames. Google MediaPipe's deep-learning model maps **468 3D facial landmarks**.
            2. **Eye Aspect Ratio (EAR)**: The 6 landmark coordinates surrounding each eye are extracted:
               $$\\text{EAR} = \\frac{||P_2 - P_6|| + ||P_3 - P_5||}{2 \\cdot ||P_1 - P_4||}$$
            3. **Closure Threshold**: When eyes open, EAR is $\\approx 0.28 - 0.38$. When closed, EAR plummets below $0.20$.
            4. **Temporal Filter**: A normal human blink lasts $100 - 300\\text{ ms}$. If eye closure persists for $> 1.5\\text{ seconds}$ (or $\\ge 25$ frames), a **Drowsiness Alert** triggers.
            """)

        with col_guide2:
            st.markdown("### 📋 Quick Demonstration Checklist")
            st.markdown("""
            - [x] Click **'▶️ Start'** in the left sidebar.
            - [x] Allow webcam access when prompted by the OS.
            - [x] Verify green eye landmark contours surround your eyes.
            - [x] **Test 1 - Normal Blink**: Blink naturally; verify system remains **Normal**.
            - [x] **Test 2 - Sustained Eye Closure**: Close your eyes for $> 2\\text{ seconds}$; verify the red warning banner and sound alert trigger.
            - [x] Open your eyes again; verify the system resets immediately to **Normal**.
            - [x] Click **'⏹️ Stop'** when finished.
            """)


# ---------------------------------------------------------
# MODE 2: BROWSER CAMERA SNAPSHOT (WORKS ON CLOUD / MOBILE)
# ---------------------------------------------------------
elif app_mode == "📸 Browser Camera Snapshot (Cloud Demo)":
    st.markdown("### 📸 Browser Camera Snapshot Mode")
    st.info("💡 This mode uses your web browser's camera. It works anywhere, including cloud deployments like Streamlit Community Cloud!")

    snapshot = st.camera_input("Capture a photo of your face (Test Open vs. Closed Eyes)")

    if snapshot is not None:
        # Convert buffer to OpenCV image
        file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            consecutive_frames_threshold=1,
            time_threshold_seconds=0.0,
            model_path=st.session_state.get("model_path")
        )
        result = detector.process_frame(frame, draw_landmarks=show_landmarks)

        # Update metrics
        update_metrics(
            eye_status=result["eye_status"],
            ear_val=result["ear_avg"],
            total_alerts=st.session_state.total_alerts,
            monitoring=False
        )

        col_img1, col_img2 = st.columns([3, 2])
        with col_img1:
            rgb_result = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            st.image(rgb_result, caption="AI Landmark & EAR Analysis", width="stretch")

        with col_img2:
            st.markdown("#### Snapshot Diagnostic Results")
            if result["face_detected"]:
                if result["ear_avg"] < ear_threshold:
                    st.error(f"😴 **EYES CLOSED DETECTED!**\n\nEAR: **{result['ear_avg']:.3f}** (Below threshold {ear_threshold:.2f})")
                else:
                    st.success(f"👁️ **EYES OPEN & ATTENTIVE!**\n\nEAR: **{result['ear_avg']:.3f}** (Above threshold {ear_threshold:.2f})")

                st.write(f"- **Left Eye EAR:** {result['ear_left']:.3f}")
                st.write(f"- **Right Eye EAR:** {result['ear_right']:.3f}")
                st.write(f"- **Average EAR:** {result['ear_avg']:.3f}")
            else:
                st.warning("⚠️ No face was detected in this photo. Please face the camera and try again.")


# ---------------------------------------------------------
# MODE 3: UPLOAD FACE IMAGE
# ---------------------------------------------------------
elif app_mode == "📁 Upload Face Image":
    st.markdown("### 📁 Upload Face Photo for EAR Analysis")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            model_path=st.session_state.get("model_path")
        )
        result = detector.process_frame(frame, draw_landmarks=show_landmarks)

        update_metrics(
            eye_status=result["eye_status"],
            ear_val=result["ear_avg"],
            total_alerts=st.session_state.total_alerts,
            monitoring=False
        )

        col_up1, col_up2 = st.columns([3, 2])
        with col_up1:
            rgb_res = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            st.image(rgb_res, caption="Analyzed Image with Facial Mesh", width="stretch")

        with col_up2:
            st.markdown("#### Diagnostic Report")
            if result["face_detected"]:
                if result["ear_avg"] < ear_threshold:
                    st.error(f"😴 **EYES CLOSED** (EAR: {result['ear_avg']:.3f})")
                else:
                    st.success(f"👁️ **EYES OPEN** (EAR: {result['ear_avg']:.3f})")
                st.write(f"- **Left Eye EAR:** {result['ear_left']:.3f}")
                st.write(f"- **Right Eye EAR:** {result['ear_right']:.3f}")
                st.write(f"- **Average EAR:** {result['ear_avg']:.3f}")
            else:
                st.warning("No face detected in this image.")


# ---------------------------------------------------------
# Telemetry Section: EAR Signal Chart & Event Log
# ---------------------------------------------------------
if len(st.session_state.ear_history) > 5:
    st.markdown("### 📈 Live Eye Aspect Ratio (EAR) Telemetry")
    chart_data = pd.DataFrame({
        "Live EAR": list(st.session_state.ear_history),
        "Threshold": [ear_threshold] * len(st.session_state.ear_history)
    })
    st.line_chart(chart_data, color=["#38bdf8", "#ef4444"])

if len(st.session_state.event_logs) > 0:
    st.markdown("### 📑 Session Alert History")
    log_df = pd.DataFrame(st.session_state.event_logs)
    st.dataframe(log_df, width="stretch")
