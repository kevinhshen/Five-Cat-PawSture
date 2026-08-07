"""
PawSture
-------------------
Host-side posture monitoring backend for the "Stats & Emotion Cube" project.

Pipeline:
    Webcam -> YOLOv8-Pose keypoints -> persistent object tracking -> target student filtering
    -> automatic front/side view selection -> calibrated view-specific posture metrics
    -> EMA smoothing -> state classification -> Serial to microcontroller -> Audio Alert

Dependencies:
    pip install ultralytics opencv-python pyserial numpy
"""
import argparse
import json
import time
import threading
import platform
import subprocess
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import serial  # pyserial
    from serial.tools import list_ports
except ImportError:
    serial = None  # allow running without hardware connected
    list_ports = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # -- Model / capture --
    model_path: str = "yolov8n-pose.pt"
    camera_index: int = 0
    sample_interval_s: float = 0.5
    model_imgsz: int = 320
    conf_threshold: float = 0.45

    # -- Smoothing --
    ema_alpha: float = 0.3

    # -- Baseline calibration --
    auto_calibrate: bool = True
    auto_calibrate_samples: int = 6

    # -- Front-view geometry checks --
    front_min_eye_shoulder_ratio: float = 0.16
    front_max_eye_tilt_ratio: float = 0.8  # Increased to allow head tilting
    front_min_shoulder_head_ratio: float = 2.2 

    # -- Classification thresholds for front-view posture strain score (0-100) --
    neutral_score: float = 22.0
    sad_score: float = 42.0

    # -- Classification thresholds for side-view angle deviations --
    side_neutral_deviation_deg: float = 10.0
    side_sad_deviation_deg: float = 20.0

    # -- Automatic view selection & hysteresis --
    switch_confirm_samples: int = 5      
    switch_margin: float = 0.15          
    side_mode_bias: float = 0.15         

    # -- Audio Alerts --
    alert_sound_path: str = "alert.wav"  
    audio_enabled_default: bool = True
    bad_posture_trigger_s: float = 3.0   
    mode_switch_grace_s: float = 10.0     
    alert_cooldown_s: float = 5.0        

    # -- Serial --
    serial_port: Optional[str] = None
    serial_baud: int = 115200
    send_min_interval_s: float = 1.0

    # -- Web console --
    web_enabled: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    web_open_browser: bool = True
    web_client_timeout_s: float = 3.0

    # -- YOLOv8-Pose COCO keypoint indices --
    KP_NOSE: int = 0
    KP_L_EYE: int = 1
    KP_R_EYE: int = 2
    KP_L_EAR: int = 3
    KP_R_EAR: int = 4
    KP_L_SHOULDER: int = 5
    KP_R_SHOULDER: int = 6
    KP_L_HIP: int = 11
    KP_R_HIP: int = 12


# --------------------------------------------------------------------------- #
# Audio Helper (Cross-Platform)
# --------------------------------------------------------------------------- #

def play_alert_sound(sound_path: str):
    """Plays an audio file using native OS tools (no external libraries required)."""
    current_os = platform.system()
    
    try:
        if current_os == "Darwin":
            # macOS built-in audio player
            subprocess.run(["afplay", sound_path], check=True)
        elif current_os == "Windows":
            # Windows built-in audio library
            import winsound
            winsound.PlaySound(sound_path, winsound.SND_FILENAME)
        elif current_os == "Linux":
            # standard Linux audio player
            subprocess.run(["aplay", sound_path], check=True)
        else:
            print(f"[audio] Unsupported OS for audio playback: {current_os}")
    except Exception as e:
        print(f"[audio] Failed to play sound: {e}")


# --------------------------------------------------------------------------- #
# Front-facing posture metrics
# --------------------------------------------------------------------------- #

@dataclass
class FrontPostureMetrics:
    eye_width_px: float
    eye_drop: float
    eye_tilt: float
    eye_shift: float
    neck_height: float
    shoulder_slope: float
    hip_shift: float
    shoulder_width_px: float
    nose_shift: float = 0.0


@dataclass
class PostureReading:
    metrics: Optional[FrontPostureMetrics]
    score: Optional[float] = None
    compression_pct: Optional[float] = None


@dataclass
class Baseline:
    metrics: Optional[FrontPostureMetrics] = None
    calibrated: bool = False

    def set(self, metrics: FrontPostureMetrics):
        self.metrics = metrics
        self.calibrated = True

    def clear(self):
        self.metrics = None
        self.calibrated = False


@dataclass
class SideBaseline:
    neck_angle: Optional[float] = None
    trunk_angle: Optional[float] = None
    calibrated: bool = False

    def set(self, neck_angle: float, trunk_angle: float):
        self.neck_angle = neck_angle
        self.trunk_angle = trunk_angle
        self.calibrated = True

    def clear(self):
        self.neck_angle = None
        self.trunk_angle = None
        self.calibrated = False


def point_visible(confs: np.ndarray, idx: int, cfg: Config) -> bool:
    return idx < len(confs) and confs[idx] >= cfg.conf_threshold


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def normalized_vertical_gap(top: np.ndarray, bottom: np.ndarray, scale: float) -> float:
    return abs(float(bottom[1] - top[1])) / scale


def normalized_x_gap(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return abs(float(a[0] - b[0])) / scale


def normalized_y_gap(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return abs(float(a[1] - b[1])) / scale


def choose_eye_pair(keypoints: np.ndarray, confs: np.ndarray, cfg: Config) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if point_visible(confs, cfg.KP_L_EYE, cfg) and point_visible(confs, cfg.KP_R_EYE, cfg):
        return keypoints[cfg.KP_L_EYE], keypoints[cfg.KP_R_EYE]
    return None


def choose_head_center(keypoints: np.ndarray, confs: np.ndarray, cfg: Config) -> Optional[Tuple[np.ndarray, str]]:
    eye_pair = choose_eye_pair(keypoints, confs, cfg)
    if eye_pair is not None:
        return midpoint(*eye_pair), "eyes"

    if point_visible(confs, cfg.KP_L_EAR, cfg) and point_visible(confs, cfg.KP_R_EAR, cfg):
        return midpoint(keypoints[cfg.KP_L_EAR], keypoints[cfg.KP_R_EAR]), "ears"

    if point_visible(confs, cfg.KP_NOSE, cfg):
        return keypoints[cfg.KP_NOSE], "nose"

    return None


def extract_front_posture_metrics(
    keypoints: np.ndarray,
    confs: np.ndarray,
    cfg: Config,
) -> Optional[Tuple[FrontPostureMetrics, Dict[str, np.ndarray]]]:
    required = (cfg.KP_L_SHOULDER, cfg.KP_R_SHOULDER)
    if not all(point_visible(confs, idx, cfg) for idx in required):
        return None

    head_pick = choose_head_center(keypoints, confs, cfg)
    if head_pick is None:
        return None

    head_center, head_source = head_pick
    left_shoulder = keypoints[cfg.KP_L_SHOULDER]
    right_shoulder = keypoints[cfg.KP_R_SHOULDER]
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    shoulder_width_px = float(np.linalg.norm(left_shoulder - right_shoulder))

    if shoulder_width_px < 20:
        return None

    eye_pair = choose_eye_pair(keypoints, confs, cfg)
    if eye_pair is None:
        return None

    left_eye, right_eye = eye_pair
    eye_center = midpoint(left_eye, right_eye)
    eye_width_px = float(np.linalg.norm(left_eye - right_eye))
    if eye_width_px < 8:
        return None

    nose_shift = 0.0

    # Geometry check 1.5: Profile Face Check
    if point_visible(confs, cfg.KP_NOSE, cfg):
        nose = keypoints[cfg.KP_NOSE]
        nose_to_eye_dist = float(np.linalg.norm(nose - eye_center))
        nose_shift = normalized_x_gap(nose, eye_center, eye_width_px)
        if eye_width_px < (nose_to_eye_dist * 0.9):
            return None

    if (eye_width_px / shoulder_width_px) < cfg.front_min_eye_shoulder_ratio:
        return None

    eye_tilt = normalized_y_gap(left_eye, right_eye, eye_width_px)
    if eye_tilt > cfg.front_max_eye_tilt_ratio:
        return None

    shoulder_head_ratio = shoulder_width_px / eye_width_px
    if shoulder_head_ratio < cfg.front_min_shoulder_head_ratio:
        return None

    hip_center = None
    hip_shift = 0.0
    if point_visible(confs, cfg.KP_L_HIP, cfg) and point_visible(confs, cfg.KP_R_HIP, cfg):
        hip_center = midpoint(keypoints[cfg.KP_L_HIP], keypoints[cfg.KP_R_HIP])
        hip_shift = normalized_x_gap(shoulder_center, hip_center, shoulder_width_px)

    metrics = FrontPostureMetrics(
        eye_width_px=eye_width_px,
        eye_drop=float(eye_center[1] - shoulder_center[1]) / shoulder_width_px,
        eye_tilt=eye_tilt,
        eye_shift=normalized_x_gap(eye_center, shoulder_center, shoulder_width_px),
        neck_height=normalized_vertical_gap(head_center, shoulder_center, shoulder_width_px),
        shoulder_slope=normalized_y_gap(left_shoulder, right_shoulder, shoulder_width_px),
        hip_shift=hip_shift,
        shoulder_width_px=shoulder_width_px,
        nose_shift=nose_shift,
    )

    tracking_points = {
        "head": head_center,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_center": shoulder_center,
        "head_source": head_source,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "eye_center": eye_center,
    }
    if point_visible(confs, cfg.KP_NOSE, cfg):
        tracking_points["nose"] = keypoints[cfg.KP_NOSE]
    if hip_center is not None:
        tracking_points["hip_center"] = hip_center
        tracking_points["left_hip"] = keypoints[cfg.KP_L_HIP]
        tracking_points["right_hip"] = keypoints[cfg.KP_R_HIP]

    return metrics, tracking_points


def score_front_posture(metrics: FrontPostureMetrics, baseline: FrontPostureMetrics) -> Tuple[float, float]:
    compression = max(0.0, (baseline.neck_height - metrics.neck_height) / max(baseline.neck_height, 1e-6))
    eye_drop_extra = max(0.0, metrics.eye_drop - baseline.eye_drop)
    eye_tilt_extra = max(0.0, metrics.eye_tilt - baseline.eye_tilt)
    eye_shift_extra = max(0.0, metrics.eye_shift - baseline.eye_shift)
    face_close_extra = max(0.0, (metrics.eye_width_px / max(baseline.eye_width_px, 1e-6)) - 1.0)
    hip_shift_extra = max(0.0, metrics.hip_shift - baseline.hip_shift)
    shoulder_slope_extra = max(0.0, metrics.shoulder_slope - baseline.shoulder_slope)
    nose_shift_extra = max(0.0, metrics.nose_shift - baseline.nose_shift)

    strain_score = (
        eye_drop_extra * 145.0
        + eye_tilt_extra * 115.0
        + eye_shift_extra * 95.0
        + face_close_extra * 90.0
        + compression * 80.0
        + hip_shift_extra * 55.0
        + shoulder_slope_extra * 25.0
        + nose_shift_extra * 70.0
    )
    posture_score = 100.0 - min(strain_score, 100.0)
    return posture_score, compression * 100.0


def average_front_metrics(samples) -> FrontPostureMetrics:
    return FrontPostureMetrics(
        eye_width_px=float(np.mean([sample.eye_width_px for sample in samples])),
        eye_drop=float(np.mean([sample.eye_drop for sample in samples])),
        eye_tilt=float(np.mean([sample.eye_tilt for sample in samples])),
        eye_shift=float(np.mean([sample.eye_shift for sample in samples])),
        neck_height=float(np.mean([sample.neck_height for sample in samples])),
        shoulder_slope=float(np.mean([sample.shoulder_slope for sample in samples])),
        hip_shift=float(np.mean([sample.hip_shift for sample in samples])),
        shoulder_width_px=float(np.mean([sample.shoulder_width_px for sample in samples])),
        nose_shift=float(np.mean([sample.nose_shift for sample in samples])),
    )


def classify_state(score: float, cfg: Config) -> str:
    if score >= 100.0 - cfg.neutral_score:
        return "HAPPY"
    if score >= 100.0 - cfg.sad_score:
        return "NEUTRAL"
    return "SAD"


# --------------------------------------------------------------------------- #
# Side-facing posture metrics
# --------------------------------------------------------------------------- #

def angle_from_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    vec = p_top - p_bottom
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(vec_norm, [0.0, -1.0]), -1.0, 1.0))))
    return -angle if vec[0] < 0 else angle


def choose_side_chain(
    keypoints: np.ndarray,
    confs: np.ndarray,
    cfg: Config,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]]:
    choices = (
        ("left", (cfg.KP_L_EAR, cfg.KP_L_SHOULDER, cfg.KP_L_HIP)),
        ("right", (cfg.KP_R_EAR, cfg.KP_R_SHOULDER, cfg.KP_R_HIP)),
    )
    side_name, indices = max(choices, key=lambda choice: float(np.mean(confs[list(choice[1])])))
    quality = float(np.mean(confs[list(indices)]))
    if quality < cfg.conf_threshold:
        return None
    ear, shoulder, hip = (keypoints[index] for index in indices)
    return ear, shoulder, hip, quality, side_name


def side_state(neck_deviation: float, trunk_deviation: float, cfg: Config) -> str:
    worst_deviation = max(abs(neck_deviation), abs(trunk_deviation))
    if worst_deviation <= cfg.side_neutral_deviation_deg:
        return "HAPPY"
    if worst_deviation <= cfg.side_sad_deviation_deg:
        return "NEUTRAL"
    return "SAD"


def point_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_from_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    vec = b - a
    return float(np.degrees(np.arctan2(-vec[1], vec[0])))


def segment_midpoint(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    mid = midpoint(a, b).astype(int)
    return int(mid[0]), int(mid[1])


def draw_text_label(frame, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)


def draw_segment_label(
    frame,
    text: str,
    a: np.ndarray,
    b: np.ndarray,
    color: Tuple[int, int, int],
    offset: Tuple[int, int] = (8, -8),
):
    mid_x, mid_y = segment_midpoint(a, b)
    draw_text_label(frame, text, (mid_x + offset[0], mid_y + offset[1]), color)


# --------------------------------------------------------------------------- #
# EMA smoothing & Serial link
# --------------------------------------------------------------------------- #

class EMASmoother:
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


class MetricSmoother:
    def __init__(self, alpha: float):
        self.smoothers = {
            "eye_width_px": EMASmoother(alpha),
            "eye_drop": EMASmoother(alpha),
            "eye_tilt": EMASmoother(alpha),
            "eye_shift": EMASmoother(alpha),
            "neck_height": EMASmoother(alpha),
            "shoulder_slope": EMASmoother(alpha),
            "hip_shift": EMASmoother(alpha),
            "shoulder_width_px": EMASmoother(alpha),
            "nose_shift": EMASmoother(alpha),
        }

    def update(self, metrics: FrontPostureMetrics) -> FrontPostureMetrics:
        values = {
            name: smoother.update(getattr(metrics, name))
            for name, smoother in self.smoothers.items()
        }
        return FrontPostureMetrics(**values)

    def reset(self):
        for smoother in self.smoothers.values():
            smoother.reset()


def auto_detect_port() -> Optional[str]:
    if list_ports is None:
        return None
    ports = list(list_ports.comports())
    if not ports:
        return None
    keywords = ("usbmodem", "usbserial", "arduino", "ch340", "wchusbserial", "ttyusb", "ttyacm")
    for port in ports:
        device = port.device.lower()
        description = (port.description or "").lower()
        if any(keyword in device or keyword in description for keyword in keywords):
            return port.device
    return ports[0].device if len(ports) == 1 else None


class SerialLink:
    def __init__(self, port: Optional[str], baud: int, min_interval_s: float):
        if port is None:
            port = auto_detect_port()
        self.enabled = port is not None and serial is not None
        self.min_interval_s = min_interval_s
        self._last_sent = 0.0
        self._last_state = None
        self.conn = None
        if self.enabled:
            try:
                self.conn = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print(f"[serial] Connected on {port} @ {baud} baud")
            except Exception as e:
                print(f"[serial] Failed to open {port}: {e}")
                self.enabled = False

    def send_state(self, state: str):
        now = time.time()
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
            print(f"[serial->cube] {state}")

    def close(self):
        if self.enabled and self.conn is not None:
            self.conn.close()


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

        self.front_metric_smoother = MetricSmoother(cfg.ema_alpha)
        self.front_baseline = Baseline()
        self.side_neck_smoother = EMASmoother(cfg.ema_alpha)
        self.side_trunk_smoother = EMASmoother(cfg.ema_alpha)
        self.side_baseline = SideBaseline()
        self._front_calibration_samples = deque(maxlen=cfg.auto_calibrate_samples)
        self._side_calibration_samples = deque(maxlen=cfg.auto_calibrate_samples)
        self.serial_link = SerialLink(cfg.serial_port, cfg.serial_baud, cfg.send_min_interval_s)

        self._last_sample_t = 0.0
        self._last_state = "NEUTRAL"
        self._active_mode: Optional[str] = None
        self._candidate_mode: Optional[str] = None
        self._candidate_count = 0
        self._last_tracking_points = None
        self._last_side_points = None
        self._last_reading = PostureReading(metrics=None)
        self._last_side_angles: Tuple[Optional[float], Optional[float]] = (None, None)
        self.target_track_id: Optional[int] = None
        
        self.audio_on = self.cfg.audio_enabled_default
        self._bad_state_start_t = None
        self._last_alert_t = 0.0
        self._last_switch_t = 0.0
        self._running = True
        self._latest_jpeg: Optional[bytes] = None
        self._web_client_seen = False
        self._last_web_client_t = time.time()
        self._state_lock = threading.Lock()

    def _extract_candidates(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=self.cfg.model_imgsz
        )[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return {}

        track_ids = None
        if results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().numpy()

        target_idx = None

        if self.target_track_id is not None and track_ids is not None:
            matches = np.where(track_ids == self.target_track_id)[0]
            if len(matches) > 0:
                target_idx = matches[0]

        if target_idx is None:
            frame_h, frame_w = frame.shape[:2]
            frame_center = np.array([frame_w / 2.0, frame_h / 2.0])
            min_dist = float("inf")

            for i, kpts in enumerate(results.keypoints.xy):
                kpts_np = kpts.cpu().numpy()
                visible = kpts_np[kpts_np.any(axis=1)]
                if len(visible) == 0:
                    continue
                center = visible.mean(axis=0)
                dist = np.linalg.norm(center - frame_center)
                if dist < min_dist:
                    min_dist = dist
                    target_idx = i

            if target_idx is not None and track_ids is not None:
                self.target_track_id = int(track_ids[target_idx])

        if target_idx is None:
            return {}

        keypoints = results.keypoints.xy[target_idx].cpu().numpy()
        confs = results.keypoints.conf[target_idx].cpu().numpy()
        candidates = {}

        front = extract_front_posture_metrics(keypoints, confs, self.cfg)
        if front is not None:
            required = (self.cfg.KP_L_EYE, self.cfg.KP_R_EYE, self.cfg.KP_L_SHOULDER, self.cfg.KP_R_SHOULDER)
            quality = float(np.mean(confs[list(required)]))
            candidates["FRONT"] = (quality, front)

        side = choose_side_chain(keypoints, confs, self.cfg)
        if side is not None:
            ear, shoulder, hip, quality, side_name = side
            candidates["SIDE"] = (quality, (ear, shoulder, hip, side_name))

        return candidates

    def _select_mode(self, candidates) -> Optional[str]:
        if not candidates:
            self._candidate_mode = None
            self._candidate_count = 0
            return self._active_mode

        weighted_candidates = {}
        for mode, (qual, data) in candidates.items():
            score = qual
            if mode == self._active_mode:
                score += self.cfg.side_mode_bias if mode == "SIDE" else (self.cfg.switch_margin / 2)
            weighted_candidates[mode] = score

        best_mode = max(weighted_candidates, key=lambda m: weighted_candidates[m])

        if self._active_mode is None:
            self._active_mode = best_mode
            self._candidate_mode = None
            self._candidate_count = 0
            return self._active_mode

        if best_mode == self._active_mode:
            self._candidate_mode = None
            self._candidate_count = 0
            return self._active_mode

        if best_mode == self._candidate_mode:
            self._candidate_count += 1
        else:
            self._candidate_mode = best_mode
            self._candidate_count = 1

        if self._candidate_count >= self.cfg.switch_confirm_samples:
            self._active_mode = best_mode
            self._candidate_mode = None
            self._candidate_count = 0
            self._last_switch_t = time.time()
            print(f"[view] Confirmed view switch to {self._active_mode} tracking.")

        return self._active_mode

    def _calibrate_front(self, metrics: FrontPostureMetrics):
        self.front_baseline.set(metrics)
        self._front_calibration_samples.clear()
        self.front_metric_smoother.reset()
        print(f"[calibration] Front baseline set -> eye width: {metrics.eye_width_px:.1f}px")

    def _calibrate_side(self, neck_angle: float, trunk_angle: float):
        self.side_baseline.set(neck_angle, trunk_angle)
        self._side_calibration_samples.clear()
        self.side_neck_smoother.reset()
        self.side_trunk_smoother.reset()
        print(f"[calibration] Side baseline set -> neck: {neck_angle:.1f} deg, trunk: {trunk_angle:.1f} deg")

    def _maybe_auto_calibrate_front(self, metrics: FrontPostureMetrics):
        if not self.cfg.auto_calibrate or self.front_baseline.calibrated:
            return
        self._front_calibration_samples.append(metrics)
        if len(self._front_calibration_samples) < self.cfg.auto_calibrate_samples:
            return
        self._calibrate_front(average_front_metrics(self._front_calibration_samples))
        print("[calibration] Front baseline auto-captured.")

    def _maybe_auto_calibrate_side(self, neck_angle: float, trunk_angle: float):
        if not self.cfg.auto_calibrate or self.side_baseline.calibrated:
            return
        self._side_calibration_samples.append((neck_angle, trunk_angle))
        if len(self._side_calibration_samples) < self.cfg.auto_calibrate_samples:
            return
        neck_avg = float(np.mean([sample[0] for sample in self._side_calibration_samples]))
        trunk_avg = float(np.mean([sample[1] for sample in self._side_calibration_samples]))
        self._calibrate_side(neck_avg, trunk_avg)
        print("[calibration] Side baseline auto-captured.")

    def _update_front(self, extracted):
        metrics, tracking_points = extracted
        self._last_tracking_points = tracking_points
        self._last_side_points = None
        smoothed_metrics = self.front_metric_smoother.update(metrics)
        self._maybe_auto_calibrate_front(smoothed_metrics)
        score = None
        compression = None
        if self.front_baseline.calibrated and self.front_baseline.metrics is not None:
            score, compression = score_front_posture(smoothed_metrics, self.front_baseline.metrics)
            self._last_state = classify_state(score, self.cfg)
            self.serial_link.send_state(self._last_state)
        else:
            self._last_state = "NEUTRAL"
        self._last_reading = PostureReading(smoothed_metrics, score, compression)
        self._last_side_angles = (None, None)

    def _update_side(self, extracted):
        ear, shoulder, hip, side_name = extracted
        neck_angle = angle_from_vertical(ear, shoulder)
        trunk_angle = angle_from_vertical(shoulder, hip)
        neck_smoothed = self.side_neck_smoother.update(neck_angle)
        trunk_smoothed = self.side_trunk_smoother.update(trunk_angle)
        self._last_side_points = (ear, shoulder, hip, side_name)
        self._last_tracking_points = None
        self._last_reading = PostureReading(metrics=None)
        self._last_side_angles = (neck_smoothed, trunk_smoothed)
        self._maybe_auto_calibrate_side(neck_smoothed, trunk_smoothed)
        if self.side_baseline.calibrated:
            neck_dev = neck_smoothed - self.side_baseline.neck_angle
            trunk_dev = trunk_smoothed - self.side_baseline.trunk_angle
            self._last_state = side_state(neck_dev, trunk_dev, self.cfg)
            self.serial_link.send_state(self._last_state)
        else:
            self._last_state = "NEUTRAL"

    def _draw_front_tracking(self, frame):
        points = self._last_tracking_points
        if points is None:
            return frame
        left_eye, right_eye = tuple(points["left_eye"].astype(int)), tuple(points["right_eye"].astype(int))
        eye_center = tuple(points["eye_center"].astype(int))
        left_shoulder, right_shoulder = tuple(points["left_shoulder"].astype(int)), tuple(points["right_shoulder"].astype(int))
        shoulder_center = tuple(points["shoulder_center"].astype(int))

        cv2.line(frame, left_eye, right_eye, (0, 255, 255), 3)
        cv2.line(frame, left_shoulder, right_shoulder, (255, 180, 0), 3)
        cv2.line(frame, shoulder_center, eye_center, (255, 180, 0), 3)

        eye_distance = point_distance_px(points["left_eye"], points["right_eye"])
        eye_angle = angle_from_horizontal(points["left_eye"], points["right_eye"])
        shoulder_distance = point_distance_px(points["left_shoulder"], points["right_shoulder"])
        shoulder_angle = angle_from_horizontal(points["left_shoulder"], points["right_shoulder"])
        head_distance = point_distance_px(points["eye_center"], points["shoulder_center"])
        head_angle = angle_from_vertical(points["eye_center"], points["shoulder_center"])

        if "nose" in points:
            nose = points["nose"]
            nose_pt = tuple(nose.astype(int))
            nose_distance = point_distance_px(points["eye_center"], nose)
            nose_angle = angle_from_horizontal(points["eye_center"], nose)
            cv2.line(frame, eye_center, nose_pt, (0, 160, 255), 2)
            draw_segment_label(frame, f"{nose_distance:.0f}px, {nose_angle:+.1f} deg", points["eye_center"], nose, (0, 160, 255), (10, 18))

        draw_segment_label(frame, f"{eye_distance:.0f}px, {eye_angle:+.1f} deg", points["left_eye"], points["right_eye"], (0, 255, 255), (8, -10))
        draw_segment_label(frame, f"{shoulder_distance:.0f}px, {shoulder_angle:+.1f} deg", points["left_shoulder"], points["right_shoulder"], (255, 180, 0), (8, 18))
        draw_segment_label(frame, f"{head_distance:.0f}px, {head_angle:+.1f} deg", points["shoulder_center"], points["eye_center"], (255, 180, 0), (10, -10))

        for point, color in ((left_eye, (0, 255, 255)), (right_eye, (0, 255, 255)), (eye_center, (0, 220, 220)), (left_shoulder, (255, 0, 255)), (right_shoulder, (255, 0, 255)), (shoulder_center, (255, 180, 0))):
            cv2.circle(frame, point, 7, color, -1)
            cv2.circle(frame, point, 10, (0, 0, 0), 2)
        if "nose" in points:
            nose_pt = tuple(points["nose"].astype(int))
            cv2.circle(frame, nose_pt, 7, (0, 160, 255), -1)
            cv2.circle(frame, nose_pt, 10, (0, 0, 0), 2)
            draw_text_label(frame, "Nose", (nose_pt[0] + 10, nose_pt[1] - 10), (0, 160, 255))
        return frame

    def _draw_side_tracking(self, frame):
        if self._last_side_points is None:
            return frame
        ear, shoulder, hip, side_name = self._last_side_points
        ear_pt, shoulder_pt, hip_pt = tuple(ear.astype(int)), tuple(shoulder.astype(int)), tuple(hip.astype(int))
        cv2.line(frame, shoulder_pt, ear_pt, (255, 200, 0), 3)
        cv2.line(frame, hip_pt, shoulder_pt, (0, 165, 255), 3)

        neck_distance = point_distance_px(ear, shoulder)
        trunk_distance = point_distance_px(shoulder, hip)
        neck_angle = angle_from_vertical(ear, shoulder)
        trunk_angle = angle_from_vertical(shoulder, hip)
        draw_segment_label(frame, f"{neck_distance:.0f}px, {neck_angle:+.1f} deg", shoulder, ear, (255, 200, 0), (10, -10))
        draw_segment_label(frame, f"{trunk_distance:.0f}px, {trunk_angle:+.1f} deg", hip, shoulder, (0, 165, 255), (10, 18))

        for point, color, label in ((ear_pt, (255, 0, 0), "Ear"), (shoulder_pt, (0, 255, 0), "Shoulder"), (hip_pt, (0, 0, 255), "Hip")):
            cv2.circle(frame, point, 7, color, -1)
            draw_text_label(frame, label, (point[0] + 10, point[1] - 10), color)
        draw_text_label(frame, f"Tracked side: {side_name}", (10, frame.shape[0] - 18), (230, 230, 230))
        return frame

    def _draw_overlay(self, frame, reading: PostureReading, state: str, show_keyboard_hints: bool = True):
        if self._active_mode == "FRONT":
            frame = self._draw_front_tracking(frame)
        elif self._active_mode == "SIDE":
            frame = self._draw_side_tracking(frame)

        metrics = reading.metrics
        if self._active_mode == "FRONT":
            lines = [
                "Mode: FRONT",
                f"State: {state}",
                f"Posture score: {reading.score:.0f}/100" if reading.score is not None else "Posture score: --",
                f"Eye drop: {metrics.eye_drop:.2f}" if metrics is not None else "Eye drop: --",
                f"Eye tilt: {metrics.eye_tilt:.2f}" if metrics is not None else "Eye tilt: --",
                f"Eye shift: {metrics.eye_shift:.2f}" if metrics is not None else "Eye shift: --",
                f"Nose shift: {metrics.nose_shift:.2f}" if metrics is not None else "Nose shift: --",
                f"Face distance: {metrics.eye_width_px:.0f}px" if metrics is not None else "Face distance: --",
                f"Shoulder slope: {metrics.shoulder_slope:.2f}" if metrics is not None else "Shoulder slope: --",
                f"Neck compression: {reading.compression_pct:.0f}%" if reading.compression_pct is not None else "Neck compression: --",
                "Front baseline: SET" if self.front_baseline.calibrated else (
                    "Front baseline: AUTO..." if self.cfg.auto_calibrate else (
                        "Front baseline: NOT SET (press 'c')" if show_keyboard_hints else "Front baseline: NOT SET"
                    )
                ),
            ]
        elif self._active_mode == "SIDE":
            neck_angle, trunk_angle = self._last_side_angles
            lines = [
                "Mode: SIDE",
                f"State: {state}",
                f"Neck angle: {neck_angle:.1f} deg" if neck_angle is not None else "Neck angle: --",
                f"Trunk angle: {trunk_angle:.1f} deg" if trunk_angle is not None else "Trunk angle: --",
                "Side baseline: SET" if self.side_baseline.calibrated else (
                    "Side baseline: AUTO..." if self.cfg.auto_calibrate else (
                        "Side baseline: NOT SET (press 'c')" if show_keyboard_hints else "Side baseline: NOT SET"
                    )
                ),
            ]
        else:
            lines = ["Mode: FINDING VIEW", "Show face & shoulders, or side profile."]
        color = {"HAPPY": (0, 200, 0), "NEUTRAL": (0, 200, 200), "SAD": (0, 0, 220)}.get(state, (255, 255, 255))
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        return frame

    def _process_frame(self, frame, show_keyboard_hints: bool = True):
        now = time.time()
        do_inference = (now - self._last_sample_t) >= self.cfg.sample_interval_s

        if do_inference:
            self._last_sample_t = now
            candidates = self._extract_candidates(frame)
            active_mode = self._select_mode(candidates)
            if active_mode is not None and active_mode in candidates:
                _, extracted = candidates[active_mode]
                if active_mode == "FRONT":
                    self._update_front(extracted)
                else:
                    self._update_side(extracted)

        if self._last_state == "SAD":
            if self._bad_state_start_t is None:
                self._bad_state_start_t = now

            time_in_bad = now - self._bad_state_start_t
            time_since_switch = now - self._last_switch_t
            time_since_alert = now - self._last_alert_t

            if (self.audio_on and
                time_in_bad >= self.cfg.bad_posture_trigger_s and
                time_since_switch >= self.cfg.mode_switch_grace_s and
                time_since_alert >= self.cfg.alert_cooldown_s):

                threading.Thread(
                    target=play_alert_sound,
                    args=(self.cfg.alert_sound_path,),
                    daemon=True
                ).start()
                print("[audio] Alert played!")

                self._last_alert_t = now
        else:
            self._bad_state_start_t = None

        frame = self._draw_overlay(frame, self._last_reading, self._last_state, show_keyboard_hints)

        audio_status = "ON" if self.audio_on else "MUTED"
        audio_hint = f"Audio: {audio_status}"
        if show_keyboard_hints:
            audio_hint += " (press 'm')"
        cv2.putText(frame, audio_hint, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with self._state_lock:
                self._latest_jpeg = encoded.tobytes()

        return frame

    def calibrate_current(self) -> Tuple[bool, str]:
        with self._state_lock:
            active_mode = self._active_mode
            if active_mode == "FRONT" and self._last_reading.metrics is not None:
                self._calibrate_front(self._last_reading.metrics)
                return True, "Front baseline set."
            if active_mode == "SIDE":
                neck_angle, trunk_angle = self._last_side_angles
                if neck_angle is not None and trunk_angle is not None:
                    self._calibrate_side(neck_angle, trunk_angle)
                    return True, "Side baseline set."
            return False, "Wait for front or side tracking before calibrating."

    def toggle_audio(self) -> Dict[str, object]:
        with self._state_lock:
            self.audio_on = not self.audio_on
            print(f"[audio] Alerts {'enabled' if self.audio_on else 'muted'}")
            return self.status_payload()

    def reset_target(self) -> Dict[str, object]:
        with self._state_lock:
            self.target_track_id = None
            self.front_baseline.clear()
            self.side_baseline.clear()
            self._front_calibration_samples.clear()
            self._side_calibration_samples.clear()
            self.front_metric_smoother.reset()
            self.side_neck_smoother.reset()
            self.side_trunk_smoother.reset()
            print("[tracking] Target and baselines reset; the centered student will be selected next.")
            return self.status_payload()

    def stop(self) -> Dict[str, object]:
        with self._state_lock:
            self._running = False
            return self.status_payload()

    def record_web_client(self):
        with self._state_lock:
            self._web_client_seen = True
            self._last_web_client_t = time.time()

    def web_client_timed_out(self) -> bool:
        with self._state_lock:
            if not self._web_client_seen:
                return False
            return (time.time() - self._last_web_client_t) > self.cfg.web_client_timeout_s

    def status_payload(self) -> Dict[str, object]:
        mode = self._active_mode or "FINDING"
        score = self._last_reading.score
        compression = self._last_reading.compression_pct
        neck_angle, trunk_angle = self._last_side_angles
        return {
            "state": self._last_state,
            "mode": mode,
            "target": self.target_track_id,
            "audioOn": self.audio_on,
            "autoCalibrate": self.cfg.auto_calibrate,
            "frontBaseline": self.front_baseline.calibrated,
            "sideBaseline": self.side_baseline.calibrated,
            "serialConnected": bool(self.serial_link.enabled and self.serial_link.conn is not None),
            "score": round(score, 1) if score is not None else None,
            "compression": round(compression, 1) if compression is not None else None,
            "neckAngle": round(neck_angle, 1) if neck_angle is not None else None,
            "trunkAngle": round(trunk_angle, 1) if trunk_angle is not None else None,
            "running": self._running,
        }

    def latest_jpeg(self) -> Optional[bytes]:
        with self._state_lock:
            return self._latest_jpeg

    def run_web(self):
        server = make_web_server(self)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://{self.cfg.web_host}:{self.cfg.web_port}"
        print("PawSture web console is running.")
        print(f"Open {url}")
        if self.cfg.web_open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue
                self._process_frame(frame, show_keyboard_hints=False)
                if self.web_client_timed_out():
                    print("[web] Browser closed or disconnected; stopping PawSture.")
                    self.stop()
                time.sleep(0.01)
        finally:
            self._running = False
            server.shutdown()
            server.server_close()
            self.cap.release()
            self.serial_link.close()

    def run(self):
        print("Automatic front/side posture mode with student targeting.")
        print("Baselines auto-capture for each view; press 'c' to recalibrate active view, 'm' to mute, 'q' to quit.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue

                frame = self._process_frame(frame)
                cv2.imshow("Posture Monitor - Auto Front/Side", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    ok, message = self.calibrate_current()
                    if not ok:
                        print(f"[calibration] {message}")
                if key == ord("m"):
                    self.toggle_audio()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.serial_link.close()


WEB_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PawSture Console</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101216;
      --panel: #181c22;
      --panel-2: #202631;
      --line: #323946;
      --text: #f4f7fb;
      --muted: #aab4c2;
      --green: #49d17d;
      --yellow: #f3c969;
      --red: #ff6f6f;
      --blue: #66b7ff;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: #151920;
    }

    .brand {
      display: flex;
      align-items: baseline;
      gap: 12px;
      min-width: 0;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1;
      letter-spacing: 0;
    }

    .subtitle {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .top-status {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
      min-width: 0;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--red);
      box-shadow: 0 0 16px currentColor;
      flex: 0 0 auto;
    }

    .dot.connected {
      background: var(--green);
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      padding: 18px;
      min-height: 0;
    }

    .viewer {
      min-height: 0;
      border: 1px solid var(--line);
      background: #050608;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 8px;
    }

    .viewer img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    aside {
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }

    .state {
      display: grid;
      gap: 8px;
      padding: 16px;
      background: var(--panel-2);
      border-radius: 8px;
      border: 1px solid var(--line);
    }

    .state-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }

    .state-value {
      font-size: 34px;
      line-height: 1;
      font-weight: 800;
      color: var(--yellow);
    }

    .state-value.happy {
      color: var(--green);
    }

    .state-value.sad {
      color: var(--red);
    }

    .stats {
      display: grid;
      gap: 10px;
    }

    .row {
      min-height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, .07);
    }

    .row:last-child {
      border-bottom: 0;
    }

    .row span:first-child {
      color: var(--muted);
      font-size: 13px;
    }

    .row strong {
      font-size: 14px;
      text-align: right;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    button {
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #edf3fa;
      color: #11151b;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }

    button.secondary {
      background: #222a35;
      color: var(--text);
    }

    button.danger {
      background: #3a2024;
      color: #ffd7d7;
      border-color: #684049;
    }

    button:active {
      transform: translateY(1px);
    }

    .message {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
    }

    footer {
      min-height: 46px;
      display: flex;
      align-items: center;
      padding: 0 22px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      background: #151920;
    }

    @media (max-width: 900px) {
      header {
        height: auto;
        min-height: 64px;
        align-items: flex-start;
        flex-direction: column;
        padding: 14px;
        gap: 8px;
      }

      main {
        grid-template-columns: 1fr;
        padding: 12px;
      }

      .viewer {
        aspect-ratio: 4 / 3;
      }

      aside {
        min-height: auto;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <h1>PawSture</h1>
        <div class="subtitle">Live posture monitor</div>
      </div>
      <div class="top-status">
        <span id="deviceDot" class="dot"></span>
        <span id="deviceText">Device disconnected</span>
        <span id="audioText">Audio on</span>
      </div>
    </header>

    <main>
      <section class="viewer" aria-label="Live camera tracking view">
        <img src="/video_feed" alt="Live camera feed with posture tracking overlay">
      </section>

      <aside>
        <section class="state">
          <div class="state-label">Posture State</div>
          <div id="stateValue" class="state-value">NEUTRAL</div>
        </section>

        <section class="panel stats">
          <div class="row"><span>Mode</span><strong id="modeValue">Finding</strong></div>
          <div class="row"><span>Score</span><strong id="scoreValue">--</strong></div>
          <div class="row"><span>Neck angle</span><strong id="neckValue">--</strong></div>
          <div class="row"><span>Trunk angle</span><strong id="trunkValue">--</strong></div>
          <div class="row"><span>Front baseline</span><strong id="frontValue">Not set</strong></div>
          <div class="row"><span>Side baseline</span><strong id="sideValue">Not set</strong></div>
        </section>

        <section class="panel controls">
          <button id="calibrateBtn">Recalibrate [C]</button>
          <button id="audioBtn" class="secondary">Mute Alerts [M]</button>
          <button id="resetBtn" class="secondary">Reset Target [R]</button>
          <button id="stopBtn" class="danger">Stop Session [Q]</button>
          <div id="message" class="message"></div>
        </section>
      </aside>
    </main>

    <footer id="footerStatus">Waiting for live tracking...</footer>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const stateValue = $("stateValue");
    const message = $("message");

    function formatNumber(value, suffix = "") {
      return value === null || value === undefined ? "--" : `${value}${suffix}`;
    }

    function titleCase(value) {
      if (!value) return "--";
      return value.charAt(0) + value.slice(1).toLowerCase();
    }

    function renderStatus(data) {
      stateValue.textContent = data.state || "NEUTRAL";
      stateValue.classList.toggle("happy", data.state === "HAPPY");
      stateValue.classList.toggle("sad", data.state === "SAD");

      $("modeValue").textContent = titleCase(data.mode);
      $("scoreValue").textContent = formatNumber(data.score, "/100");
      $("neckValue").textContent = formatNumber(data.neckAngle, " deg");
      $("trunkValue").textContent = formatNumber(data.trunkAngle, " deg");
      $("frontValue").textContent = data.frontBaseline ? "Set" : (data.autoCalibrate ? "Auto..." : "Not set");
      $("sideValue").textContent = data.sideBaseline ? "Set" : (data.autoCalibrate ? "Auto..." : "Not set");
      $("audioText").textContent = data.audioOn ? "Audio on" : "Audio muted";
      $("audioBtn").textContent = data.audioOn ? "Mute Alerts [M]" : "Unmute Alerts [M]";

      $("deviceDot").classList.toggle("connected", data.serialConnected);
      $("deviceText").textContent = data.serialConnected ? "Device connected" : "Device disconnected";
      $("footerStatus").textContent = data.running
        ? `Tracking ${titleCase(data.mode)} view`
        : "Session stopped";
    }

    async function refreshStatus() {
      try {
        const res = await fetch("/status", { cache: "no-store" });
        renderStatus(await res.json());
      } catch (error) {
        $("footerStatus").textContent = "Connection lost";
      }
    }

    async function postAction(path, label) {
      message.textContent = label;
      try {
        const res = await fetch(path, { method: "POST" });
        const data = await res.json();
        renderStatus(data.status || data);
        message.textContent = data.message || "";
      } catch (error) {
        message.textContent = "Command failed.";
      }
    }

    function closeBrowserWindow() {
      window.open("", "_self");
      window.close();
    }

    async function stopSession() {
      await postAction("/stop", "Stopping session...");
      setTimeout(closeBrowserWindow, 150);
    }

    $("calibrateBtn").addEventListener("click", () => postAction("/calibrate", "Calibrating..."));
    $("audioBtn").addEventListener("click", () => postAction("/toggle-audio", "Updating audio..."));
    $("resetBtn").addEventListener("click", () => postAction("/reset-target", "Resetting target..."));
    $("stopBtn").addEventListener("click", stopSession);

    document.addEventListener("keydown", (event) => {
      if (event.repeat) return;
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const key = event.key.toLowerCase();
      if (key === "c") {
        event.preventDefault();
        postAction("/calibrate", "Calibrating...");
      } else if (key === "m") {
        event.preventDefault();
        postAction("/toggle-audio", "Updating audio...");
      } else if (key === "r") {
        event.preventDefault();
        postAction("/reset-target", "Resetting target...");
      } else if (key === "q") {
        event.preventDefault();
        stopSession();
      }
    });

    refreshStatus();
    setInterval(refreshStatus, 750);
  </script>
</body>
</html>
"""


def make_web_server(monitor: PostureMonitor) -> ThreadingHTTPServer:
    class PawStureRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK):
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(WEB_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/status":
                monitor.record_web_client()
                self._send_json(monitor.status_payload())
                return
            if path == "/video_feed":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while monitor._running:
                    frame = monitor.latest_jpeg()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(0.08)
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/calibrate":
                ok, message = monitor.calibrate_current()
                status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
                self._send_json({"ok": ok, "message": message, "status": monitor.status_payload()}, status)
                return
            if path == "/toggle-audio":
                self._send_json({"ok": True, "message": "", "status": monitor.toggle_audio()})
                return
            if path == "/reset-target":
                self._send_json({"ok": True, "message": "Target reset.", "status": monitor.reset_target()})
                return
            if path == "/stop":
                self._send_json({"ok": True, "message": "Session stopped.", "status": monitor.stop()})
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((monitor.cfg.web_host, monitor.cfg.web_port), PawStureRequestHandler)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Automatic front/side posture monitor backend for Stats & Emotion Cube")
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--web", action="store_true", help="Start the local browser console instead of the OpenCV window.")
    p.add_argument("--web-host", default="127.0.0.1", help="Host for the local web console.")
    p.add_argument("--web-port", type=int, default=8000, help="Port for the local web console.")
    p.add_argument("--no-open", action="store_true", help="Do not automatically open the browser in web mode.")
    p.add_argument("--manual-calibration", action="store_true", help="Require pressing C/button to set front and side baselines.")
    args = p.parse_args()

    return Config(
        model_path=args.model,
        camera_index=args.camera,
        sample_interval_s=args.interval,
        model_imgsz=args.imgsz,
        serial_port=args.port,
        serial_baud=args.baud,
        web_enabled=args.web,
        web_host=args.web_host,
        web_port=args.web_port,
        web_open_browser=not args.no_open,
        auto_calibrate=not args.manual_calibration,
    )


if __name__ == "__main__":
    cfg = parse_args()
    app = PostureMonitor(cfg)
    if cfg.web_enabled:
        app.run_web()
    else:
        app.run()
