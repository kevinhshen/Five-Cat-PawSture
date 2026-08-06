"""
PawSture
-------------------
Host-side posture monitoring backend for the "Stats & Emotion Cube" project.

Pipeline:
    Front-facing webcam -> YOLOv8-Pose keypoints -> calibrated front-view
    posture metrics -> EMA smoothing -> state classification
    -> Serial to microcontroller

Hardware assumptions:
    - Front-facing webcam or laptop camera aimed at the student from the front.
    - Camera should see the face, both shoulders, and ideally both hips.
    - Arduino listening on a serial port for single-word state strings.

Controls:
    'c' -> calibrate baseline (sit upright first, then press)
    'q' -> quit

Dependencies:
    pip install ultralytics opencv-python pyserial numpy
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    model_path: str = "yolov8n-pose.pt"
    camera_index: int = 0
    sample_interval_s: float = 1.0
    model_imgsz: int = 320
    conf_threshold: float = 0.45

    # -- Smoothing --
    ema_alpha: float = 0.3

    # -- Classification thresholds for a combined 0-100 front-view posture score --
    neutral_score: float = 28.0
    sad_score: float = 55.0

    # -- Serial --
    serial_port: Optional[str] = None
    serial_baud: int = 115200
    send_min_interval_s: float = 1.0

    # -- YOLOv8-Pose COCO keypoint indices --
    # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear,
    # 5: left shoulder, 6: right shoulder, 11: left hip, 12: right hip
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
    """
    All values are normalized by shoulder width, so they are less sensitive to
    camera resolution and distance than raw pixel measurements.
    """

    neck_height: float          # head center to shoulder center vertical gap
    shoulder_slope: float       # left/right shoulder height mismatch
    head_tilt: float            # left/right eye or ear height mismatch
    center_shift: float         # head center x offset from shoulder center
    hip_shift: float            # shoulder center x offset from hip center
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


def choose_head_center(keypoints: np.ndarray, confs: np.ndarray, cfg: Config) -> Optional[Tuple[np.ndarray, str]]:
    """
    Prefer ears because they line up naturally with shoulder height. Fall back to
    eyes, then nose, because front-facing webcams often lose ears under hair or
    headphones.
    """
    if point_visible(confs, cfg.KP_L_EAR, cfg) and point_visible(confs, cfg.KP_R_EAR, cfg):
        return midpoint(keypoints[cfg.KP_L_EAR], keypoints[cfg.KP_R_EAR]), "ears"

    if point_visible(confs, cfg.KP_L_EYE, cfg) and point_visible(confs, cfg.KP_R_EYE, cfg):
        return midpoint(keypoints[cfg.KP_L_EYE], keypoints[cfg.KP_R_EYE]), "eyes"

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

    head_tilt = 0.0
    if point_visible(confs, cfg.KP_L_EAR, cfg) and point_visible(confs, cfg.KP_R_EAR, cfg):
        head_tilt = normalized_y_gap(keypoints[cfg.KP_L_EAR], keypoints[cfg.KP_R_EAR], shoulder_width_px)
    elif point_visible(confs, cfg.KP_L_EYE, cfg) and point_visible(confs, cfg.KP_R_EYE, cfg):
        head_tilt = normalized_y_gap(keypoints[cfg.KP_L_EYE], keypoints[cfg.KP_R_EYE], shoulder_width_px)

    hip_center = None
    hip_shift = 0.0
    if point_visible(confs, cfg.KP_L_HIP, cfg) and point_visible(confs, cfg.KP_R_HIP, cfg):
        hip_center = midpoint(keypoints[cfg.KP_L_HIP], keypoints[cfg.KP_R_HIP])
        hip_shift = normalized_x_gap(shoulder_center, hip_center, shoulder_width_px)

    metrics = FrontPostureMetrics(
        neck_height=normalized_vertical_gap(head_center, shoulder_center, shoulder_width_px),
        shoulder_slope=normalized_y_gap(left_shoulder, right_shoulder, shoulder_width_px),
        head_tilt=head_tilt,
        center_shift=normalized_x_gap(head_center, shoulder_center, shoulder_width_px),
        hip_shift=hip_shift,
        shoulder_width_px=shoulder_width_px,
    )

    tracking_points = {
        "head": head_center,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_center": shoulder_center,
        "head_source": head_source,
    }
    if hip_center is not None:
        tracking_points["hip_center"] = hip_center
        tracking_points["left_hip"] = keypoints[cfg.KP_L_HIP]
        tracking_points["right_hip"] = keypoints[cfg.KP_R_HIP]

    return metrics, tracking_points


def score_front_posture(metrics: FrontPostureMetrics, baseline: FrontPostureMetrics) -> Tuple[float, float]:
    """
    Build a practical front-facing posture score.

    The strongest signal is neck compression: if head-to-shoulder height gets
    shorter than the calibrated upright baseline, the student is probably
    dropping the head, hunching, or collapsing the shoulders. Side-to-side tilt
    and center shifts catch leaning and asymmetry.
    """
    compression = max(0.0, (baseline.neck_height - metrics.neck_height) / max(baseline.neck_height, 1e-6))
    shoulder_slope_extra = max(0.0, metrics.shoulder_slope - baseline.shoulder_slope)
    head_tilt_extra = max(0.0, metrics.head_tilt - baseline.head_tilt)
    center_shift_extra = max(0.0, metrics.center_shift - baseline.center_shift)
    hip_shift_extra = max(0.0, metrics.hip_shift - baseline.hip_shift)

    score = (
        compression * 95.0
        + shoulder_slope_extra * 95.0
        + head_tilt_extra * 85.0
        + center_shift_extra * 70.0
        + hip_shift_extra * 45.0
    )
    return min(score, 100.0), compression * 100.0


def classify_state(score: float, cfg: Config) -> str:
    if score <= cfg.neutral_score:
        return "HAPPY"
    if score <= cfg.sad_score:
        return "NEUTRAL"
    return "SAD"


# --------------------------------------------------------------------------- #
# EMA smoothing
# --------------------------------------------------------------------------- #

class EMASmoother:
    """Simple exponential moving average for scalar values."""

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
    """EMA smoothing for the group of front-facing posture metrics."""

    def __init__(self, alpha: float):
        self.smoothers = {
            "neck_height": EMASmoother(alpha),
            "shoulder_slope": EMASmoother(alpha),
            "head_tilt": EMASmoother(alpha),
            "center_shift": EMASmoother(alpha),
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

    @property
    def value(self) -> Optional[FrontPostureMetrics]:
        if self.smoothers["neck_height"].value is None:
            return None
        return FrontPostureMetrics(
            neck_height=self.smoothers["neck_height"].value,
            shoulder_slope=self.smoothers["shoulder_slope"].value,
            head_tilt=self.smoothers["head_tilt"].value,
            center_shift=self.smoothers["center_shift"].value,
            hip_shift=self.smoothers["hip_shift"].value,
            shoulder_width_px=self.smoothers["shoulder_width_px"].value,
        )


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
                time.sleep(2)
                print(f"[serial] Connected on {port} @ {baud} baud")
            except Exception as e:
                print(f"[serial] Failed to open {port}: {e}. Running without hardware output.")
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

        self.metric_smoother = MetricSmoother(cfg.ema_alpha)
        self.score_smoother = EMASmoother(cfg.ema_alpha)
        self.baseline = Baseline()
        self.serial_link = SerialLink(cfg.serial_port, cfg.serial_baud, cfg.send_min_interval_s)

        self._last_sample_t = 0.0
        self._last_state = "NEUTRAL"
        self._last_tracking_points = None
        self._last_reading = PostureReading(metrics=None)
        self._history = deque(maxlen=100)

    # ---------------------------------------------------------------- #
    def _extract_metrics(self, frame) -> Optional[Tuple[FrontPostureMetrics, Dict[str, np.ndarray]]]:
        """Run pose model on one frame, return front-facing metrics or None."""
        results = self.model(frame, verbose=False, imgsz=self.cfg.model_imgsz)[0]

        if results.keypoints is None or len(results.keypoints.xy) == 0:
            self._last_tracking_points = None
            return None

        keypoints = results.keypoints.xy[0].cpu().numpy()
        confs = results.keypoints.conf[0].cpu().numpy()

        extracted = extract_front_posture_metrics(keypoints, confs, self.cfg)
        if extracted is None:
            self._last_tracking_points = None
            return None

        metrics, tracking_points = extracted
        self._last_tracking_points = tracking_points
        return metrics, tracking_points

    # ---------------------------------------------------------------- #
    def _handle_calibration(self, metrics: FrontPostureMetrics):
        self.baseline.set(metrics)
        self.metric_smoother.reset()
        self.score_smoother.reset()
        print(
            "[calibration] Baseline set -> "
            f"neck height: {metrics.neck_height:.2f}, "
            f"shoulder slope: {metrics.shoulder_slope:.2f}, "
            f"head tilt: {metrics.head_tilt:.2f}, "
            f"center shift: {metrics.center_shift:.2f}"
        )

    # ---------------------------------------------------------------- #
    def _draw_overlay(self, frame, reading: PostureReading, state: str):
        frame = self._draw_tracking_overlay(frame)

        metrics = reading.metrics
        score = reading.score
        compression = reading.compression_pct

        y = 30
        lines = [
            f"State: {state}",
            f"Posture score: {score:.0f}/100" if score is not None else "Posture score: --",
            f"Neck compression: {compression:.0f}%" if compression is not None else "Neck compression: --",
            f"Shoulder slope: {metrics.shoulder_slope:.2f}" if metrics is not None else "Shoulder slope: --",
            f"Head/torso shift: {metrics.center_shift:.2f}" if metrics is not None else "Head/torso shift: --",
            "Baseline: SET" if self.baseline.calibrated else "Baseline: NOT SET (press 'c')",
        ]
        color = {"HAPPY": (0, 200, 0), "NEUTRAL": (0, 200, 200), "SAD": (0, 0, 220)}.get(state, (255, 255, 255))
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        return frame

    # ---------------------------------------------------------------- #
    def _draw_tracking_overlay(self, frame):
        """Draw front-facing keypoints and posture guides."""
        points = self._last_tracking_points
        if points is None:
            return frame

        head = tuple(points["head"].astype(int))
        left_shoulder = tuple(points["left_shoulder"].astype(int))
        right_shoulder = tuple(points["right_shoulder"].astype(int))
        shoulder_center = tuple(points["shoulder_center"].astype(int))

        cv2.line(frame, left_shoulder, right_shoulder, (255, 180, 0), 3)
        cv2.line(frame, shoulder_center, head, (255, 180, 0), 3)
        cv2.line(frame, (head[0], shoulder_center[1]), shoulder_center, (180, 180, 180), 2)

        if "left_hip" in points and "right_hip" in points:
            left_hip = tuple(points["left_hip"].astype(int))
            right_hip = tuple(points["right_hip"].astype(int))
            hip_center = tuple(points["hip_center"].astype(int))
            cv2.line(frame, left_hip, right_hip, (0, 180, 255), 2)
            cv2.line(frame, hip_center, shoulder_center, (0, 180, 255), 2)
            cv2.circle(frame, hip_center, 6, (0, 180, 255), -1)

        draw_points = [
            ("Head", head, (0, 255, 255)),
            ("L Shoulder", left_shoulder, (255, 0, 255)),
            ("R Shoulder", right_shoulder, (255, 0, 255)),
            ("Shoulders", shoulder_center, (255, 180, 0)),
        ]
        for label, point, color in draw_points:
            cv2.circle(frame, point, 7, color, -1)
            cv2.circle(frame, point, 10, (0, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        source = points.get("head_source", "head")
        cv2.putText(
            frame,
            f"Head source: {source}",
            (10, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            2,
        )
        return frame

    # ---------------------------------------------------------------- #
    def run(self):
        print("Front-facing posture mode.")
        print("Sit upright with face and shoulders visible, press 'c' to calibrate, 'q' to quit.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[capture] Frame grab failed, retrying...")
                    time.sleep(0.5)
                    continue

                now = time.time()
                do_inference = (now - self._last_sample_t) >= self.cfg.sample_interval_s

                if do_inference:
                    self._last_sample_t = now
                    extracted = self._extract_metrics(frame)
                    if extracted is not None:
                        metrics, _ = extracted
                        smoothed_metrics = self.metric_smoother.update(metrics)
                        score = None
                        compression = None

                        if self.baseline.calibrated and self.baseline.metrics is not None:
                            raw_score, compression = score_front_posture(smoothed_metrics, self.baseline.metrics)
                            score = self.score_smoother.update(raw_score)
                            state = classify_state(score, self.cfg)
                            self._last_state = state
                            self._history.append((now, score, compression, state))
                            self.serial_link.send_state(state)
                        else:
                            self._last_state = "NEUTRAL"

                        self._last_reading = PostureReading(
                            metrics=smoothed_metrics,
                            score=score,
                            compression_pct=compression,
                        )

                frame = self._draw_overlay(frame, self._last_reading, self._last_state)
                cv2.imshow("Posture Monitor (press c=calibrate, q=quit)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    extracted = self._extract_metrics(frame)
                    if extracted is not None:
                        metrics, _ = extracted
                        self._handle_calibration(metrics)
                    else:
                        print("[calibration] Need a clear front view of face and both shoulders.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.serial_link.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Front-facing posture monitor backend for Stats & Emotion Cube")
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between inference samples")
    p.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size; lower is faster")
    p.add_argument("--port", type=str, default=None, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
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
