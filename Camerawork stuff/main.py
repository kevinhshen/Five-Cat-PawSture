#Open a terminal and run this to download the libraries
#pip3 install numpy opencv-python ultralytics pyserial


"""
posture_monitor.py
-------------------
Host-side posture monitoring backend for the "Stats & Emotion Cube" project.

Pipeline:
    Webcam (side view) -> YOLO26-Pose keypoints -> vector angle math
    -> EMA smoothing -> baseline calibration comparison
    -> state classification (HAPPY / NEUTRAL / SAD) -> Serial to microcontroller

Hardware assumptions:
    - Side-view webcam, 3-5 ft away, shoulder height, subject facing left or right.
    - ESP32/Arduino listening on a serial port for single-word state strings.

Controls:
    'c' -> calibrate baseline (hold upright posture, then press)
    'q' -> quit

Dependencies:
    pip install ultralytics opencv-python pyserial numpy
"""

import time
import math
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import serial  # pyserial
except ImportError:
    serial = None  # allow running without hardware connected


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # -- Model / capture --
    model_path: str = "yolo26n-pose.pt"
    camera_index: int = 0
    sample_interval_s: float = 2.5          # process ~1 frame every 2-3s (load optimization)
    conf_threshold: float = 0.5             # min keypoint confidence to trust a point

    # -- Smoothing --
    ema_alpha: float = 0.3                  # 0 = no update, 1 = no smoothing at all

    # -- Classification thresholds (degrees of DEVIATION from calibrated baseline) --
    neutral_deviation_deg: float = 8.0      # up to this = still HAPPY
    sad_deviation_deg: float = 16.0         # beyond this = SAD, between = NEUTRAL

    # -- Serial --
    serial_port: Optional[str] = None       # e.g. "COM5" or "/dev/ttyUSB0"; None = disabled
    serial_baud: int = 115200
    send_min_interval_s: float = 1.0        # avoid spamming the microcontroller

    # -- YOLO26-Pose COCO keypoint indices we use (same 17-point layout as v8/11) --
    # 0: nose, 3: left ear, 4: right ear, 5: left shoulder, 6: right shoulder,
    # 11: left hip, 12: right hip
    KP_L_EAR: int = 3
    KP_R_EAR: int = 4
    KP_L_SHOULDER: int = 5
    KP_R_SHOULDER: int = 6
    KP_L_HIP: int = 11
    KP_R_HIP: int = 12


# --------------------------------------------------------------------------- #
# Vector / angle math
# --------------------------------------------------------------------------- #

def angle_from_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    """
    Angle (degrees) between the vector p_bottom->p_top and the true vertical (0,-1)
    in image coordinates (y grows downward).

    0 deg  = perfectly vertical (upright)
    +/-90  = fully horizontal

    Sign convention: positive = leaning "forward" in +x image direction,
    negative = leaning "backward" in -x direction. Which physical direction
    that corresponds to depends on which way the subject faces the camera;
    calibration removes the need to worry about that.
    """
    vec = p_top - p_bottom
    # Vertical reference vector (pointing up in image space)
    vertical = np.array([0.0, -1.0])
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    dot = np.clip(np.dot(vec_norm, vertical), -1.0, 1.0)
    angle = math.degrees(math.acos(dot))
    # signed via x-component of the vector
    if vec[0] < 0:
        angle = -angle
    return angle


def pick_side(keypoints: np.ndarray, confs: np.ndarray, cfg: Config
              ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    From a side-view webcam, only one side of the body (L or R) will be visible
    with high confidence. Pick whichever side (ear/shoulder/hip) is most confident.
    Returns (ear_xy, shoulder_xy, hip_xy) or None if not enough confidence.
    """
    left_idxs = (cfg.KP_L_EAR, cfg.KP_L_SHOULDER, cfg.KP_L_HIP)
    right_idxs = (cfg.KP_R_EAR, cfg.KP_R_SHOULDER, cfg.KP_R_HIP)

    left_conf = confs[list(left_idxs)].mean()
    right_conf = confs[list(right_idxs)].mean()

    chosen_idxs = left_idxs if left_conf >= right_conf else right_idxs
    chosen_conf = max(left_conf, right_conf)

    if chosen_conf < cfg.conf_threshold:
        return None

    ear, shoulder, hip = (keypoints[i] for i in chosen_idxs)
    return ear, shoulder, hip


# --------------------------------------------------------------------------- #
# EMA smoothing
# --------------------------------------------------------------------------- #

class EMASmoother:
    """Simple exponential moving average for scalar values (e.g. angles)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

@dataclass
class Baseline:
    neck_angle: Optional[float] = None      # ear-shoulder vector vs vertical
    trunk_angle: Optional[float] = None     # shoulder-hip vector vs vertical
    calibrated: bool = False

    def set(self, neck_angle: float, trunk_angle: float):
        self.neck_angle = neck_angle
        self.trunk_angle = trunk_angle
        self.calibrated = True


# --------------------------------------------------------------------------- #
# Serial link
# --------------------------------------------------------------------------- #

class SerialLink:
    """Thin wrapper around pyserial with graceful no-op fallback."""

    def __init__(self, port: Optional[str], baud: int, min_interval_s: float):
        self.enabled = port is not None and serial is not None
        self.min_interval_s = min_interval_s
        self._last_sent = 0.0
        self._last_state = None
        self.conn = None

        if self.enabled:
            try:
                self.conn = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # allow ESP32/Arduino to reset after port open
                print(f"[serial] Connected on {port} @ {baud} baud")
            except Exception as e:
                print(f"[serial] Failed to open {port}: {e}. Running without hardware output.")
                self.enabled = False

    def send_state(self, state: str):
        now = time.time()
        # Only send on state change OR after min interval, to avoid flooding the MCU
        if state == self._last_state and (now - self._last_sent) < self.min_interval_s:
            return
        self._last_sent = now
        self._last_state = state

        if self.enabled and self.conn is not None:
            try:
                self.conn.write(f"{state}\n".encode("utf-8"))
            except Exception as e:
                print(f"[serial] Write failed: {e}")
        else:
            # Placeholder output when no hardware attached
            print(f"[serial->cube] {state}")

    def close(self):
        if self.enabled and self.conn is not None:
            self.conn.close()


# --------------------------------------------------------------------------- #
# Posture classification
# --------------------------------------------------------------------------- #

def classify_state(neck_dev: float, trunk_dev: float, cfg: Config) -> str:
    """
    Combine forward-head (neck) and slouch (trunk) deviations from baseline
    into a single comfort state. Worst-of-two-signals logic: whichever
    metric is most out of range drives the state.
    """
    worst_dev = max(abs(neck_dev), abs(trunk_dev))

    if worst_dev <= cfg.neutral_deviation_deg:
        return "HAPPY"
    elif worst_dev <= cfg.sad_deviation_deg:
        return "NEUTRAL"
    else:
        return "SAD"


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #

class PostureMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"[model] Loading {cfg.model_path} ...")
        self.model = YOLO(cfg.model_path)

        self.cap = cv2.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cfg.camera_index}")

        self.neck_smoother = EMASmoother(cfg.ema_alpha)
        self.trunk_smoother = EMASmoother(cfg.ema_alpha)
        self.baseline = Baseline()
        self.serial_link = SerialLink(cfg.serial_port, cfg.serial_baud, cfg.send_min_interval_s)

        self._last_sample_t = 0.0
        self._last_state = "NEUTRAL"
        # small rolling history purely for on-screen debug/plotting if desired
        self._history = deque(maxlen=100)

    # ---------------------------------------------------------------- #
    def _extract_angles(self, frame) -> Optional[Tuple[float, float]]:
        """Run pose model on one frame, return (neck_angle, trunk_angle) or None."""
        results = self.model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None

        # Use the first detected person (assume single-student desk setup)
        keypoints = results.keypoints.xy[0].cpu().numpy()      # (17, 2)
        confs = results.keypoints.conf[0].cpu().numpy()        # (17,)

        picked = pick_side(keypoints, confs, self.cfg)
        if picked is None:
            return None

        ear, shoulder, hip = picked

        neck_angle = angle_from_vertical(ear, shoulder)     # forward head posture
        trunk_angle = angle_from_vertical(shoulder, hip)    # slouch

        return neck_angle, trunk_angle

    # ---------------------------------------------------------------- #
    def _handle_calibration(self, neck_angle: float, trunk_angle: float):
        self.baseline.set(neck_angle, trunk_angle)
        self.neck_smoother.reset()
        self.trunk_smoother.reset()
        print(f"[calibration] Baseline set -> neck: {neck_angle:.1f} deg, "
              f"trunk: {trunk_angle:.1f} deg")

    # ---------------------------------------------------------------- #
    def _draw_overlay(self, frame, neck_angle, trunk_angle, state):
        y = 30
        lines = [
            f"State: {state}",
            f"Neck angle: {neck_angle:.1f} deg" if neck_angle is not None else "Neck angle: --",
            f"Trunk angle: {trunk_angle:.1f} deg" if trunk_angle is not None else "Trunk angle: --",
            "Baseline: SET" if self.baseline.calibrated else "Baseline: NOT SET (press 'c')",
        ]
        color = {"HAPPY": (0, 200, 0), "NEUTRAL": (0, 200, 200), "SAD": (0, 0, 220)}.get(state, (255, 255, 255))
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        return frame

    # ---------------------------------------------------------------- #
    def run(self):
        print("Press 'c' to calibrate baseline (sit upright first), 'q' to quit.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[capture] Frame grab failed, retrying...")
                    time.sleep(0.5)
                    continue

                now = time.time()
                do_inference = (now - self._last_sample_t) >= self.cfg.sample_interval_s

                neck_angle_raw = trunk_angle_raw = None
                neck_smoothed = self.neck_smoother.value
                trunk_smoothed = self.trunk_smoother.value

                if do_inference:
                    self._last_sample_t = now
                    angles = self._extract_angles(frame)
                    if angles is not None:
                        neck_angle_raw, trunk_angle_raw = angles
                        neck_smoothed = self.neck_smoother.update(neck_angle_raw)
                        trunk_smoothed = self.trunk_smoother.update(trunk_angle_raw)

                        if self.baseline.calibrated:
                            neck_dev = neck_smoothed - self.baseline.neck_angle
                            trunk_dev = trunk_smoothed - self.baseline.trunk_angle
                            state = classify_state(neck_dev, trunk_dev, self.cfg)
                            self._last_state = state
                            self._history.append((now, neck_dev, trunk_dev, state))
                            self.serial_link.send_state(state)
                        else:
                            # No baseline yet: report NEUTRAL as a safe default
                            self._last_state = "NEUTRAL"

                # Overlay + display every rendered frame (cheap), inference only every N seconds
                frame = self._draw_overlay(
                    frame,
                    neck_smoothed,
                    trunk_smoothed,
                    self._last_state,
                )
                cv2.imshow("Posture Monitor (press c=calibrate, q=quit)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Force an immediate inference pass for calibration, don't wait for sample timer
                    angles = self._extract_angles(frame)
                    if angles is not None:
                        self._handle_calibration(*angles)
                    else:
                        print("[calibration] No confident keypoints detected — adjust position and retry.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.serial_link.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Posture monitor backend for Stats & Emotion Cube")
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=2.5, help="Seconds between inference samples")
    p.add_argument("--port", type=str, default=None, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    return Config(
        model_path=args.model,
        camera_index=args.camera,
        sample_interval_s=args.interval,
        serial_port=args.port,
        serial_baud=args.baud,
    )


if __name__ == "__main__":
    cfg = parse_args()
    app = PostureMonitor(cfg)
    app.run()