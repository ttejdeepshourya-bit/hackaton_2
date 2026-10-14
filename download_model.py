"""
Model Downloader Utility
------------------------
Ensures that the official MediaPipe FaceLandmarker task bundle
is downloaded and ready for real-time landmark inference.
"""

import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_model_exists(target_path: str = MODEL_PATH) -> str:
    """
    Checks if the MediaPipe model file exists. If missing, downloads it.
    Returns the absolute path to the verified model file.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        return target_path

    print(f"[INFO] Face Landmarker model not found at: {target_path}")
    print(f"[INFO] Downloading official MediaPipe model (~3.7 MB) from:\n       {MODEL_URL}...")

    def progress_callback(blocks, block_size, total_size):
        if total_size > 0:
            percent = (blocks * block_size / total_size) * 100
            sys.stdout.write(f"\r[INFO] Download progress: {percent:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, target_path, reporthook=progress_callback)
        print("\n[INFO] Model successfully downloaded and verified!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        raise RuntimeError(
            f"Could not download MediaPipe FaceLandmarker model. "
            f"Please check your internet connection or manually place {MODEL_FILENAME} in the 'models/' directory."
        )

    return target_path


if __name__ == "__main__":
    path = ensure_model_exists()
    print(f"[SUCCESS] Face Landmarker model ready at: {path} (Size: {os.path.getsize(path):,} bytes)")
