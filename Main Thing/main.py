"""
PawSture
-------------------
Host-side posture monitoring backend for the "Stats & Emotion Cube" project.

Pipeline:
    Webcam -> YOLOv8-Pose keypoints -> persistent object tracking -> target student filtering
    -> automatic front/side view selection -> calibrated view-specific posture metrics
    -> EMA smoothing -> state classification -> Serial to microcontroller -> Audio Alert

Dependencies:
    pip install ultralytics opencv-python pyserial numpy playsound==1.2.2
"""
import argparse
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import serial  # pyserial
    from serial.tools import list_ports
except ImportError:
    serial = None  # allow running without hardware connected
    list_ports = None

try:
    from playsound import playsound
except ImportError:
    playsound = None


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

    # -- Front-view geometry checks --
    front_min_eye_shoulder_ratio: float = 0.16
    front_max_eye_tilt_ratio: float = 0.8  # Increased to allow head tilting
    front_min_shoulder_head_ratio: float = 2.2 

    # -- Classification thresholds for front-view posture score (0-100) --
    neutral_score: float = 28.0
    sad_score: float = 55.0

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
    mode_switch_grace_s: float = 4.0     
    alert_cooldown_s: float = 5.0        

    # -- Serial --
    serial_port: Optional[str] = None
    serial_baud: int = 115200
    send_min_interval_s: float = 1.0

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


@dataclass
class SideBaseline:
    neck_angle: Optional[float] = None
    trunk_angle: Optional[float] = None
    calibrated: bool = False

    def set(self, neck_angle: float, trunk_angle: float):
        self.neck_angle = neck_angle
        self.trunk_angle = trunk_angle
        self.calibrated = True


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

    # Geometry check 1.5: Profile Face Check
    if point_visible(confs, cfg.KP_NOSE, cfg):
        nose = keypoints[cfg.KP_NOSE]
        nose_to_eye_dist = float(np.linalg.norm(nose - eye_center))
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

    score = (
        eye_drop_extra * 115.0
        + eye_tilt_extra * 85.0
        + eye_shift_extra * 70.0
        + face_close_extra * 65.0
        + compression * 45.0
        + hip_shift_extra * 30.0
        + shoulder_slope_extra * 12.0
    )
    return min(score, 100.0), compression * 100.0


def classify_state(score: float, cfg: Config) -> str:
    if score <= cfg.neutral_score:
        return "HAPPY"
    if score <= cfg.sad_score:
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
        self.front_metric_smoother.reset()
        print(f"[calibration] Front baseline set -> eye width: {metrics.eye_width_px:.1f}px")

    def _calibrate_side(self, neck_angle: float, trunk_angle: float):
        self.side_baseline.set(neck_angle, trunk_angle)
        self.side_neck_smoother.reset()
        self.side_trunk_smoother.reset()
        print(f"[calibration] Side baseline set -> neck: {neck_angle:.1f} deg, trunk: {trunk_angle:.1f} deg")

    def _update_front(self, extracted):
        metrics, tracking_points = extracted
        self._last_tracking_points = tracking_points
        self._last_side_points = None
        smoothed_metrics = self.front_metric_smoother.update(metrics)
        score = None
        compression = None
        if self.front_baseline.calibrated and self.front_baseline.metrics is not None:
            score, compression = score_front_posture(metrics, self.front_baseline.metrics)
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
        for point, color in ((left_eye, (0, 255, 255)), (right_eye, (0, 255, 255)), (eye_center, (0, 220, 220)), (left_shoulder, (255, 0, 255)), (right_shoulder, (255, 0, 255)), (shoulder_center, (255, 180, 0))):
            cv2.circle(frame, point, 7, color, -1)
            cv2.circle(frame, point, 10, (0, 0, 0), 2)
        return frame

    def _draw_side_tracking(self, frame):
        if self._last_side_points is None:
            return frame
        ear, shoulder, hip, side_name = self._last_side_points
        ear_pt, shoulder_pt, hip_pt = tuple(ear.astype(int)), tuple(shoulder.astype(int)), tuple(hip.astype(int))
        cv2.line(frame, shoulder_pt, ear_pt, (255, 200, 0), 3)
        cv2.line(frame, hip_pt, shoulder_pt, (0, 165, 255), 3)
        for point, color, label in ((ear_pt, (255, 0, 0), "Ear"), (shoulder_pt, (0, 255, 0), "Shoulder"), (hip_pt, (0, 0, 255), "Hip")):
            cv2.circle(frame, point, 7, color, -1)
            cv2.putText(frame, label, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Tracked side: {side_name}", (10, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 2)
        return frame

    def _draw_overlay(self, frame, reading: PostureReading, state: str):
        if self._active_mode == "FRONT":
            frame = self._draw_front_tracking(frame)
        elif self._active_mode == "SIDE":
            frame = self._draw_side_tracking(frame)

        metrics = reading.metrics
        if self._active_mode == "FRONT":
            lines = [
                f"Mode: FRONT (Target #{self.target_track_id})",
                f"State: {state}",
                f"Posture score: {reading.score:.0f}/100" if reading.score is not None else "Posture score: --",
                f"Eye drop: {metrics.eye_drop:.2f}" if metrics is not None else "Eye drop: --",
                f"Eye tilt: {metrics.eye_tilt:.2f}" if metrics is not None else "Eye tilt: --",
                f"Eye shift: {metrics.eye_shift:.2f}" if metrics is not None else "Eye shift: --",
                f"Face distance: {metrics.eye_width_px:.0f}px" if metrics is not None else "Face distance: --",
                f"Shoulder slope: {metrics.shoulder_slope:.2f}" if metrics is not None else "Shoulder slope: --",
                f"Neck compression: {reading.compression_pct:.0f}%" if reading.compression_pct is not None else "Neck compression: --",
                "Front baseline: SET" if self.front_baseline.calibrated else "Front baseline: NOT SET (press 'c')",
            ]
        elif self._active_mode == "SIDE":
            neck_angle, trunk_angle = self._last_side_angles
            lines = [
                f"Mode: SIDE (Target #{self.target_track_id})",
                f"State: {state}",
                f"Neck angle: {neck_angle:.1f} deg" if neck_angle is not None else "Neck angle: --",
                f"Trunk angle: {trunk_angle:.1f} deg" if trunk_angle is not None else "Trunk angle: --",
                "Side baseline: SET" if self.side_baseline.calibrated else "Side baseline: NOT SET (press 'c')",
            ]
        else:
            lines = ["Mode: FINDING VIEW", "Show face & shoulders, or side profile."]
        color = {"HAPPY": (0, 200, 0), "NEUTRAL": (0, 200, 200), "SAD": (0, 0, 220)}.get(state, (255, 255, 255))
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        return frame

    def run(self):
        print("Automatic front/side posture mode with student targeting.")
        print("Press 'c' to calibrate & lock active target, 'm' to mute, 'q' to quit.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue

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
                        
                        if playsound is not None:
                            threading.Thread(
                                target=playsound, 
                                args=(self.cfg.alert_sound_path,), 
                                daemon=True
                            ).start()
                            print("[audio] Alert played!")
                        
                        self._last_alert_t = now
                else:
                    self._bad_state_start_t = None

                frame = self._draw_overlay(frame, self._last_reading, self._last_state)
                
                audio_status = "ON" if self.audio_on else "MUTED"
                cv2.putText(frame, f"Audio: {audio_status} (press 'm')", (10, frame.shape[0] - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                cv2.imshow("Posture Monitor - Auto Front/Side", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.target_track_id = None
                    candidates = self._extract_candidates(frame)
                    active_mode = self._select_mode(candidates)

                    if active_mode == "FRONT" and active_mode in candidates:
                        _, extracted = candidates[active_mode]
                        metrics, _ = extracted
                        self._calibrate_front(metrics)
                        print(f"[tracking] Locked onto target student #{self.target_track_id}")
                    elif active_mode == "SIDE" and active_mode in candidates:
                        _, extracted = candidates[active_mode]
                        ear, shoulder, hip, _ = extracted
                        self._calibrate_side(angle_from_vertical(ear, shoulder), angle_from_vertical(shoulder, hip))
                        print(f"[tracking] Locked onto target student #{self.target_track_id}")
                    else:
                        print("[calibration] Wait for FRONT or SIDE mode before calibrating.")
                if key == ord("m"):
                    self.audio_on = not self.audio_on
                    print(f"[audio] Alerts {'enabled' if self.audio_on else 'muted'}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.serial_link.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Automatic front/side posture monitor backend for Stats & Emotion Cube")
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    return Config(
        model_path=args.model,
        camera_index=args.camera,
        sample_interval_s=args.interval,
        model_imgsz=args.imgsz,
        serial_port=args.port,
        serial_baud=args.baud,
    )


if __name__ == "__main__":
    cfg = parse_args()
    app = PostureMonitor(cfg)
    app.run()