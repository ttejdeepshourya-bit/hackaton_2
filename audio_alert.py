"""
Audio Alert Manager
-------------------
Provides non-blocking, thread-safe audible warnings for driver drowsiness detection.
Supports native Windows beep (winsound) with automatic cooldown to avoid system audio lag.
"""

import threading
import time
import os

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AudioAlert:
    """
    Manages non-blocking audio alerts for drowsiness warnings.
    """

    def __init__(self, cooldown_seconds: float = 1.0, enabled: bool = True):
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self._last_alert_time = 0.0
        self._lock = threading.Lock()
        self._is_playing = False
        
        # Path to pre-generated wav alert chime
        self.wav_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "alert.wav"
        )

    def set_enabled(self, enabled: bool):
        """Enable or disable audio alert playback."""
        self.enabled = enabled

    def trigger(self, frequency: int = 1400, duration_ms: int = 400):
        """
        Triggers a warning beep in a separate daemon thread if cooldown has passed.
        Non-blocking: returns immediately to preserve video frame rate.
        """
        if not self.enabled:
            return

        current_time = time.time()
        with self._lock:
            if self._is_playing:
                return
            if (current_time - self._last_alert_time) < self.cooldown_seconds:
                return
            self._is_playing = True
            self._last_alert_time = current_time

        thread = threading.Thread(
            target=self._play_sound_worker,
            args=(frequency, duration_ms),
            daemon=True
        )
        thread.start()

    def _play_sound_worker(self, frequency: int, duration_ms: int):
        """Worker executing the sound playback on a background thread."""
        try:
            if HAS_WINSOUND:
                # Windows hardware buzzer / beep
                try:
                    winsound.Beep(frequency, duration_ms)
                except Exception:
                    # Fallback to wav file if beep is muted in OS settings
                    if os.path.exists(self.wav_path):
                        winsound.PlaySound(
                            self.wav_path,
                            winsound.SND_FILENAME | winsound.SND_NODEFAULT
                        )
            else:
                # Terminal bell character as cross-platform fallback
                print("\a", end="", flush=True)
        except Exception as e:
            # Silently catch audio errors so computer vision never crashes
            pass
        finally:
            with self._lock:
                self._is_playing = False


# Module-level singleton helper for convenient global usage
_default_alert = None


def get_audio_alert(cooldown_seconds: float = 1.0, enabled: bool = True) -> AudioAlert:
    global _default_alert
    if _default_alert is None:
        _default_alert = AudioAlert(cooldown_seconds=cooldown_seconds, enabled=enabled)
    else:
        _default_alert.cooldown_seconds = cooldown_seconds
        _default_alert.enabled = enabled
    return _default_alert


if __name__ == "__main__":
    print("[TEST] Initializing AudioAlert system...")
    alert = AudioAlert(cooldown_seconds=0.5, enabled=True)
    print("[TEST] Triggering non-blocking test alert beep...")
    alert.trigger()
    time.sleep(1.0)
    print("[TEST] AudioAlert execution complete.")
