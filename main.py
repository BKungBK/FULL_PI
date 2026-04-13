"""
SmartBin V2 Engine — Raspberry Pi 5
=====================================
Automated waste sorting:
  Stage-1  NCNN YOLO  → detect + crop object
  Stage-2  TFLite INT8 → classify waste type
  Hardware GPIO servos  → sort into correct bin

Detection timing (from hand-withdrawal):
  t=0.00s  Hand withdrawn   → Servo1 pre-emptive → 120°
  t=0.50s  Grab 5 frames    → rolling frame buffer
  t~0.55s  Inference done   → label determined (5-frame vote)
  t~0.65s  Sort servo       → Servo2 → target bin angle
  t=1.00s  Drop             → Servo1 → 45°
  t=1.50s  Reset            → all servos home

Usage:
    python main.py
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────
import cv2
import lgpio
import ncnn
import numpy as np
import requests
import tflite_runtime.interpreter as tflite
from concurrent.futures import ThreadPoolExecutor

# ── Suppress known cosmetic bug in tflite_runtime Delegate.__del__ ────
# When load_delegate() fails (e.g. XNNPACK not available), the Delegate
# object is only partially constructed.  Python's GC then calls __del__
# which crashes on "object has no attribute '_library'".  We patch it
# once here so the AttributeError is silently absorbed.
_orig_delegate_del = getattr(tflite.Delegate, "__del__", None)
def _safe_delegate_del(self: tflite.Delegate) -> None:
    try:
        if _orig_delegate_del:
            _orig_delegate_del(self)
    except AttributeError:
        pass
tflite.Delegate.__del__ = _safe_delegate_del  # type: ignore[method-assign]

# ── shared state (thread-safe singleton for UI + server) ──────────────
try:
    from shared_state import system_state
except ImportError:
    system_state = None  # graceful fallback when running without shared_state.py

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SmartBin")


# =====================================================================
# 1. CONFIGURATION  (class-level constants — no instantiation needed)
# =====================================================================
class Config:
    """All tunable constants in one place. Edit here only."""

    # ── ESP32-CAM ─────────────────────────────────────────────────────
    ESP32_IP_OVERRIDE: Optional[str] = "10.42.0.177"   # None → auto-scan
    ESP32_STREAM_PORT: int = 81
    ESP32_CONTROL_PORT: int = 80
    ESP32_SCAN_TIMEOUT: float = 0.5
    ESP32_SCAN_WORKERS: int = 100
    ESP32_SCAN_PORTS: Tuple[int, ...] = (81, 80)
    FLASH_INTENSITY_ON: int = 200
    FLASH_INTENSITY_OFF: int = 0

    # ── Teachable Machine Classifier ───────────────────────────────────
    CLASSIFIER_MODEL_PATH: str = "model.tflite"
    CLASSIFIER_LABELS_PATH: str = "labels.txt"
    TFLITE_THREADS: int = 4
    VOTE_FRAMES: int = 5
    # Class labels will be loaded dynamically from labels.txt.
    CLASS_NAMES: List[str] = field(default_factory=list)

    # ── ROI window (Ignored in TM mode, but kept for compatibility) ────
    ROI_ENABLED: bool = True
    ROI_CENTER_X: int = 320
    ROI_CENTER_Y: int = 240
    ROI_SIZE: int = 320

    # ── GPIO pins ─────────────────────────────────────────────────────
    TRIG_PIN: int = 5
    ECHO_PIN: int = 6
    SERVO1_PIN: int = 18   # capture / tipping arm
    SERVO2_PIN: int = 19   # bin selector
    BIN_TRIG_PINS: Tuple[int, ...] = (13, 16, 20, 21)
    BIN_ECHO_PINS: Tuple[int, ...] = (12, 25, 26, 27)

    # ── Servo angles (degrees) ────────────────────────────────────────
    CAP_HOME_ANGLE: int = 99
    SORT_HOME_ANGLE: int = 86
    PHOTO_ANGLE: int = 120   # pre-emptive capture tilt
    SWEEP_ANGLE: int = 45    # drop / tip angle
    PLASTIC_ANGLE: int = 112  # PET / HDPEM
    GLASS_ANGLE: int = 157
    METAL_ANGLE: int = 67    # AluCan (default)

    # ── Timing (seconds) ─────────────────────────────────────────────
    DETECT_DISTANCE_CM: float = 15.0
    HAND_WITHDRAWN_DISTANCE_CM: float = 20.0   # hysteresis for release
    BIN_FULL_DISTANCE_CM: float = 10.0
    COOLDOWN_SECONDS: float = 5.0
    MIN_CONFIDENCE_THRESHOLD: float = 0.50     # minimum score before reject
    CAMERA_RECONNECT_DELAY: float = 1.0
    ULTRASONIC_TIMEOUT: float = 0.1
    CAMERA_BUFFER_SIZE: int = 10  # rolling frame buffer depth

    # ── Derived URL helpers ───────────────────────────────────────────
    @classmethod
    def stream_url(cls, ip: str) -> str:
        return f"http://{ip}:{cls.ESP32_STREAM_PORT}/stream"

    @classmethod
    def flash_url(cls, ip: str, on: bool) -> str:
        val = cls.FLASH_INTENSITY_ON if on else cls.FLASH_INTENSITY_OFF
        return (
            f"http://{ip}:{cls.ESP32_CONTROL_PORT}"
            f"/control?var=led_intensity&val={val}"
        )


# =====================================================================
# 2. GLOBAL STATE  (shared between engine functions)
# =====================================================================
@dataclass
class GlobalState:
    """Mutable singletons shared by inference functions."""
    classifier_interpreter: Optional[tflite.Interpreter] = None
    last_roi_coords: Optional[Tuple[int, int, int, int]] = None


g_state = GlobalState()


# =====================================================================
# 3. DETECTION STATE MACHINE
# =====================================================================
class DetectionState(Enum):
    IDLE         = "idle"
    HAND_PRESENT = "hand_present"
    PROCESSING   = "processing"
    COOLDOWN     = "cooldown"


# =====================================================================
# 4. ROI HELPER
# =====================================================================
def crop_roi(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop ROI from *frame* using current Config settings.

    Returns:
        (roi_crop, (x1, y1, x2, y2)) in original frame coordinates.
    """
    h, w = frame.shape[:2]
    if not Config.ROI_ENABLED:
        return frame, (0, 0, w, h)

    cx = Config.ROI_CENTER_X
    cy = Config.ROI_CENTER_Y
    half = Config.ROI_SIZE // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# =====================================================================
# 5. MODEL LOADING
# =====================================================================


def load_tflite_model(model_path: str) -> tflite.Interpreter:
    """Load a TFLite model and use XNNPACK when the delegate is available."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    delegates = []
    try:
        _d = tflite.load_delegate("libtensorflow-lite-delegate-xnnpack.so")
        delegates.append(_d)
        logging.info("XNNPACK delegate loaded.")
    except Exception:
        # XNNPACK .so not present on this platform (Pi 5 headless) — run on CPU.
        # Note: tflite_runtime may print "Exception ignored in Delegate.__del__"
        # above — this is a known cosmetic bug in some tflite builds; it is safe.
        pass

    interpreter = tflite.Interpreter(
        model_path=model_path,
        num_threads=Config.TFLITE_THREADS,
        experimental_delegates=delegates,
    )
    interpreter.allocate_tensors()
    return interpreter


def initialize_models() -> None:
    """Load Teachable Machine TFLite classifier and labels.txt."""
    try:
        with open(Config.CLASSIFIER_LABELS_PATH, 'r') as f:
            Config.CLASS_NAMES = f.read().splitlines()
        logger.info("Class names loaded: %s", Config.CLASS_NAMES)
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        # Fallback to defaults
        Config.CLASS_NAMES = ["plastic", "metal", "glass"]

    logger.info("Loading classifier from %s...", Config.CLASSIFIER_MODEL_PATH)
    g_state.classifier_interpreter = load_tflite_model(Config.CLASSIFIER_MODEL_PATH)
    input_detail  = g_state.classifier_interpreter.get_input_details()[0]
    output_detail = g_state.classifier_interpreter.get_output_details()[0]
    shape = input_detail.get("shape", [])
    input_h = int(shape[1]) if len(shape) > 1 else 224
    input_w = int(shape[2]) if len(shape) > 2 else 224

    quant = input_detail.get("quantization", (0.0, 0))
    try:
        scale = float(quant[0] if isinstance(quant[0], (int, float)) else quant[0][0])
        zp = int(quant[1] if isinstance(quant[1], (int, float)) else quant[1][0])
    except Exception:
        scale, zp = 0.0, 0

    logger.info(
        "Classifier loaded. Input: %dx%d  dtype=%s  quant=(scale=%.6f, zp=%d)  "
        "Output shape: %s",
        input_h, input_w,
        input_detail["dtype"].__name__,
        scale, zp,
        output_detail["shape"],
    )


# =====================================================================
# 6. AI INFERENCE FUNCTIONS  (verbatim from spec)
# =====================================================================

def preprocess_yolo(frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
    """Preprocess frame for YOLO NCNN input (letterbox resize, RGB CHW float32)."""
    h, w = frame.shape[:2]
    size = Config.YOLO_INPUT_SIZE

    # Calculate scale and padding
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image (letterbox)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_tensor = np.ascontiguousarray(
        np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    )
    return input_tensor, LetterboxInfo(
        scale=scale,
        pad_left=left,
        pad_top=top,
        orig_h=h,
        orig_w=w,
    )


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE adaptive contrast enhancement. (Matches dataset_tool.py)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare_image(img: np.ndarray) -> np.ndarray:
    """Crop center square and resize specifically for Teachable Machine models."""
    h_img, w_img = img.shape[:2]
    size = min(h_img, w_img)
    x = (w_img - size) // 2
    y = (h_img - size) // 2
    
    square = img[y : y + size, x : x + size]
    resized240 = cv2.resize(square, (240, 240))
    margin = 8
    roi224 = resized240[margin : 240 - margin, margin : 240 - margin]
    return roi224


def predict(img: np.ndarray) -> Tuple[str, float]:
    """Run Teachable Machine float32 standard inference."""
    try:
        interpreter = g_state.classifier_interpreter
        if interpreter is None:
            raise RuntimeError("Classifier is not initialized")
            
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # ปรับให้ตรงกับขนาด input model ทั่วไป
        img_resized = cv2.resize(img, (224, 224)) 
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        image_array = np.asarray(img_rgb)
        
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
        img_final = np.expand_dims(normalized_image_array, 0)
        
        interpreter.set_tensor(input_details[0]['index'], img_final)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        conf = float(output[idx])
        
        # reject คือน้อยกว่าค่าที่ตั้งไว้ทั้งหมด
        if conf < Config.MIN_CONFIDENCE_THRESHOLD:
            return "reject", conf
            
        class_name_raw = (
            Config.CLASS_NAMES[idx]
            if idx < len(Config.CLASS_NAMES)
            else "reject"
        )
        
        # Normalize class name for UI tracking + color matching
        ll = class_name_raw.lower()
        if "plastic" in ll or "pet" in ll or "hdpe" in ll:
            class_name = "plastic"
        elif "metal" in ll or "can" in ll:
            class_name = "metal"
        elif "glass" in ll:
            class_name = "glass"
        else:
            class_name = "reject"
            
        return class_name, conf
    except Exception as e:
        logger.error("Predict error: %s", e)
        return "reject", 0.0


def classify_single_frame(frame: np.ndarray) -> Tuple[str, float]:
    """Full pipeline: extract ROI + prepare image + classify one frame."""
    roi_frame, roi_coords = crop_roi(frame)
    g_state.last_roi_coords = roi_coords
    roi224 = prepare_image(roi_frame)
    
    # ── Match exact dataset_tool.py preprocessing ──
    processed = apply_clahe(roi224)
    
    return predict(processed)


def classify_with_voting(frames: List[np.ndarray]) -> Tuple[str, float, float]:
    """
    Run 5-shot voting classification using sequential execution.

    OPTIMIZATION: Sequential is faster than ThreadPoolExecutor because:
    1. GIL prevents true parallelism for CPU-bound TFLite inference
    2. ThreadPoolExecutor has overhead (thread creation, context switching)
    3. TFLite already uses num_threads=4 for intra-op parallelism
    4. Sequential avoids the overhead of 5 thread creations

    Returns (majority_vote_class, avg_confidence, elapsed_ms).
    """
    if len(frames) < Config.VOTE_FRAMES:
        logging.warning(
            "Not enough frames for voting: %d < %d", len(frames), Config.VOTE_FRAMES
        )
        # Pad with last frame
        while len(frames) < Config.VOTE_FRAMES:
            frames.append(frames[-1])

    # OPTIMIZATION: Sequential execution with TFLite intra-op parallelism
    # This is faster than ThreadPoolExecutor due to GIL limitations
    votes: List[str] = []
    confidences: List[float] = []
    start_time = time.time()

    for i in range(Config.VOTE_FRAMES):
        try:
            cls, conf = classify_single_frame(frames[i])
            votes.append(cls)
            confidences.append(conf)
            logging.debug("Frame %d: %s (%.2f)", i, cls, conf)
        except Exception as e:
            logging.error("Inference failed for frame %d: %s", i, e)
            votes.append("reject")  # Fail safe
            confidences.append(0.0)

    elapsed_ms = (time.time() - start_time) * 1000

    # Majority vote
    if not votes:
        return "reject", 0.0, elapsed_ms

    vote_counts = Counter(votes)
    winner = vote_counts.most_common(1)[0][0]

    # Average confidence of the winning class
    winner_confs = [c for v, c in zip(votes, confidences) if v == winner]
    avg_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0

    logging.info(
        "Voting result: %s (conf=%.2f) in %.1fms (distribution: %s)",
        winner, avg_conf, elapsed_ms, dict(vote_counts),
    )
    return winner, avg_conf, elapsed_ms


# =====================================================================
# 7. NETWORK SCANNER
# =====================================================================
class NetworkScanner:
    """Discover the ESP32-CAM on the local /24 subnet."""

    def find_esp32(self) -> str:
        """Return the ESP32-CAM IP, retrying indefinitely until found."""
        if Config.ESP32_IP_OVERRIDE:
            logger.info("Using configured ESP32 IP: %s", Config.ESP32_IP_OVERRIDE)
            return Config.ESP32_IP_OVERRIDE

        logger.info("Searching for ESP32-CAM on the local network …")
        while True:
            ip = self._scan_subnet()
            if ip is not None:
                return ip
            logger.warning("ESP32 not found — retrying in 2 s …")
            time.sleep(2)

    @staticmethod
    def _get_local_ip() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()

    def _check_host(self, ip: str) -> Optional[str]:
        for port in Config.ESP32_SCAN_PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(Config.ESP32_SCAN_TIMEOUT)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    return ip
            except OSError:
                pass
        return None

    def _scan_subnet(self) -> Optional[str]:
        local_ip = self._get_local_ip()
        if local_ip == "127.0.0.1":
            logger.warning("No network interface detected")
            return None
        subnet = ".".join(local_ip.split(".")[:3])
        logger.info("Scanning subnet %s.0/24 …", subnet)
        candidates = [f"{subnet}.{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=Config.ESP32_SCAN_WORKERS) as pool:
            for result in pool.map(self._check_host, candidates):
                if result is not None:
                    return result
        return None


# =====================================================================
# 8. HARDWARE CONTROLLER
# =====================================================================
class HardwareController:
    """GPIO, servos, and HC-SR04 ultrasonic sensors."""

    def __init__(self, chip: int) -> None:
        self._h = chip
        self._last_angle: Dict[int, float] = {
            Config.SERVO1_PIN: Config.CAP_HOME_ANGLE,
            Config.SERVO2_PIN: Config.SORT_HOME_ANGLE,
        }
        self._init_pins()

    def _init_pins(self) -> None:
        output_pins = (
            [Config.TRIG_PIN, Config.SERVO1_PIN, Config.SERVO2_PIN]
            + list(Config.BIN_TRIG_PINS)
        )
        input_pins = [Config.ECHO_PIN] + list(Config.BIN_ECHO_PINS)
        for pin in output_pins:
            lgpio.gpio_claim_output(self._h, pin)
        for pin in input_pins:
            lgpio.gpio_claim_input(self._h, pin)
        logger.info("GPIO initialised (out=%s, in=%s)", output_pins, input_pins)

    # ── servo ─────────────────────────────────────────────────────────

    def move_servo(self, pin: int, angle: float) -> None:
        """Set servo to *angle* degrees (clamped 0–180)."""
        angle = max(0.0, min(180.0, angle))
        pulse_us = int(500 + (angle / 180.0) * 2000)
        lgpio.tx_servo(self._h, pin, pulse_us)
        self._last_angle[pin] = angle
    def smooth_move(
        self,
        pin: int,
        target: float,
        speed: float = 30.0,
    ) -> None:
        """Interpolate servo using a Delta-Time loop and Quartic Easing.
        This provides perfect 100% fluid movement immune to OS stutter,
        with an extremely long braking phase so the bin doesn't sway.
        """
        home_angle = Config.CAP_HOME_ANGLE if pin == Config.SERVO1_PIN else Config.SORT_HOME_ANGLE
        start = self._last_angle.get(pin, home_angle)
        distance = abs(target - start)
        
        if distance < 0.5:
            self.move_servo(pin, target)
            return

        duration = distance / speed
        duration = max(0.15, duration)

        start_time = time.monotonic()
        
        while True:
            elapsed = time.monotonic() - start_time
            t = elapsed / duration
            
            if t >= 1.0:
                # Force final strict angle to stop micro-chatter
                self.move_servo(pin, target)
                break
                
            # Ease-In-Out Quart Formula
            # Extremely soft start -> fast middle -> Extremely soft landing (slowest at end)
            if t < 0.5:
                eased_t = 8.0 * (t ** 4)
            else:
                p = (-2.0 * t) + 2.0
                eased_t = 1.0 - ((p ** 4) / 2.0)
                
            current = start + (target - start) * eased_t
            self.move_servo(pin, current)
            
            # Tiny sleep to yield CPU, but position is tied to pure time, not loop cycles
            time.sleep(0.005)

    def idle_servos(self, force: bool = False) -> None:
        """Stop PWM signal to reduce servo heat. 
        Only idles if near Home (92°) unless force=True to prevent dropping heavy loads.
        """
        for pin in (Config.SERVO1_PIN, Config.SERVO2_PIN):
            home_angle = Config.CAP_HOME_ANGLE if pin == Config.SERVO1_PIN else Config.SORT_HOME_ANGLE
            angle = self._last_angle.get(pin, home_angle)
            # Safe to idle ONLY if at Home or Forced
            if force or abs(angle - home_angle) < 2.0:
                try:
                    lgpio.tx_servo(self._h, pin, 0)
                except Exception:
                    try:
                        lgpio.tx_pwm(self._h, pin, 50, 0)
                    except Exception as exc:
                        logger.debug("idle_servos pin %d: %s", pin, exc)

    # ── ultrasonic ────────────────────────────────────────────────────

    def read_distance(self, trig: int, echo: int) -> float:
        """Trigger HC-SR04 and return distance in cm. Returns 999.0 on error."""
        timeout = Config.ULTRASONIC_TIMEOUT
        try:
            lgpio.gpio_write(self._h, trig, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(self._h, trig, 0)

            deadline = time.time() + timeout
            while lgpio.gpio_read(self._h, echo) == 0:
                if time.time() > deadline:
                    return 999.0

            t_start = time.time()
            deadline = t_start + timeout
            while lgpio.gpio_read(self._h, echo) == 1:
                if time.time() > deadline:
                    return 999.0
            t_end = time.time()

            return (t_end - t_start) * 34300.0 / 2.0
        except OSError as exc:
            logger.error("Ultrasonic read error (trig=%d): %s", trig, exc)
            return 999.0

    def check_bins_full(self) -> Optional[int]:
        """Return 1-based bin index if any bin is full, else None."""
        for i, (trig, echo) in enumerate(
            zip(Config.BIN_TRIG_PINS, Config.BIN_ECHO_PINS)
        ):
            if self.read_distance(trig, echo) < Config.BIN_FULL_DISTANCE_CM:
                return i + 1
        return None

    def cleanup(self) -> None:
        try:
            self.idle_servos()
        except Exception as exc:
            logger.debug("idle_servos during cleanup: %s", exc)
        try:
            lgpio.gpiochip_close(self._h)
            logger.info("GPIO released")
        except Exception as exc:
            logger.error("GPIO cleanup error: %s", exc)


# =====================================================================
# 9. CAMERA STREAM  (thread-safe, rolling frame buffer)
# =====================================================================
class CameraStream(threading.Thread):
    """Background MJPEG grabber with a fixed-size rolling frame buffer.

    - ``get_frame()`` — latest single frame (for live preview)
    - ``get_recent_frames(n)`` — last n frames (for voting inference)
    """

    def __init__(self, stream_url: str) -> None:
        super().__init__(daemon=True, name="CameraStream")
        self._url = stream_url
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._buffer: deque = deque(maxlen=Config.CAMERA_BUFFER_SIZE)
        self._running = True

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_recent_frames(self, n: int = 5) -> List[np.ndarray]:
        """Return copies of the last *n* frames from the rolling buffer."""
        with self._lock:
            frames = list(self._buffer)
        tail = frames[-n:] if len(frames) >= n else frames
        return [f.copy() for f in tail]

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            cap = cv2.VideoCapture(self._url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("Camera stream opened: %s", self._url)
            while self._running:
                ret, img = cap.read()
                if not ret:
                    logger.warning("Camera frame dropped — reconnecting …")
                    break
                with self._lock:
                    self._frame = img
                    self._buffer.append(img)
                # Push JPEG to shared_state for server MJPEG proxy
                # (avoids server opening a competing stream to ESP32)
                if system_state:
                    try:
                        _, jpeg = cv2.imencode(
                            '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        system_state.set_frame(jpeg.tobytes())
                    except Exception:
                        pass
            cap.release()
            if self._running:
                time.sleep(Config.CAMERA_RECONNECT_DELAY)


# =====================================================================
# 10. SMARTBIN ENGINE
# =====================================================================
class SmartBinEngine:
    """Top-level orchestrator.

    Detection timing (from hand-withdrawal moment t=0):
        t=0.00  Hand withdrawn   → Servo1 → PHOTO_ANGLE (120°)  [pre-emptive]
        t=0.50  End countdown    → grab 5 frames from buffer
        t~0.55  Inference done   → label + confidence
        t~0.65  Sort command     → Servo2 → target bin angle
        t=1.00  Drop             → Servo1 → SWEEP_ANGLE (45°)
        t=1.50  Reset            → all servos home → idle
    """

    def __init__(self) -> None:
        self._running  = False
        self._hw:  Optional[HardwareController] = None
        self._cam: Optional[CameraStream] = None
        self._esp32_ip: str = ""
        self._cv2_ok: bool = True   # flipped False on first headless cv2.error
        self._load_roi_config()

    def _load_roi_config(self) -> None:
        import os, json
        try:
            if os.path.exists("roi_config.json"):
                with open("roi_config.json", "r") as f:
                    data = json.load(f)
                    Config.ROI_CENTER_X = data.get("cx", 320)
                    Config.ROI_CENTER_Y = data.get("cy", 240)
                    Config.ROI_SIZE     = data.get("size", 320)
                logger.info("Loaded ROI from config: cx=%d cy=%d size=%d", 
                            Config.ROI_CENTER_X, Config.ROI_CENTER_Y, Config.ROI_SIZE)
        except Exception as e:
            logger.error("Failed to load ROI config: %s", e)

    def _save_roi_config(self) -> None:
        import json
        try:
            with open("roi_config.json", "w") as f:
                json.dump({
                    "cx": Config.ROI_CENTER_X,
                    "cy": Config.ROI_CENTER_Y,
                    "size": Config.ROI_SIZE
                }, f)
        except Exception as e:
            logger.error("Failed to save ROI config: %s", e)

    # ── lifecycle ─────────────────────────────────────────────────────

    def setup(self) -> None:
        """Discover ESP32, load models, init GPIO, start camera thread."""
        # ESP32
        scanner = NetworkScanner()
        self._esp32_ip = scanner.find_esp32()
        logger.info("ESP32-CAM at %s", self._esp32_ip)
        if system_state:
            system_state.update(esp32_ip=self._esp32_ip, engine_running=True)

        # AI models
        initialize_models()

        # GPIO
        chip = lgpio.gpiochip_open(0)
        self._hw = HardwareController(chip)
        logger.info("Initializing servos to Home...")
        self._hw.move_servo(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE)
        self._hw.move_servo(Config.SERVO2_PIN, Config.SORT_HOME_ANGLE)
        time.sleep(1)
        self._hw.idle_servos()

        # Camera
        stream_url = Config.stream_url(self._esp32_ip)
        self._cam = CameraStream(stream_url)
        self._cam.start()

        logger.info("System online — entering main loop")

    def run(self) -> None:
        """Main sensor loop with hand-detection state machine."""
        self._running = True
        hw = self._hw
        cam = self._cam
        assert hw and cam

        state = DetectionState.IDLE
        last_cooldown_end: float = 0.0
        hand_present_start_time: float = 0.0

        try:
            while self._running:
                frame = cam.get_frame()

                # ── distance readings ────────────────────────────────
                dist_main = hw.read_distance(Config.TRIG_PIN, Config.ECHO_PIN)
                # ── bin distances (all 4) ─────────────────────────────
                bin_dists = [
                    hw.read_distance(Config.BIN_TRIG_PINS[i], Config.BIN_ECHO_PINS[i])
                    for i in range(len(Config.BIN_TRIG_PINS))
                ]
                
                # Check fullness without polling sensors a second time (prevents lag)
                full_bin = next((i + 1 for i, d in enumerate(bin_dists) if d < Config.BIN_FULL_DISTANCE_CM), None)

                # ── push to shared state ──────────────────────────────
                if system_state:
                    system_state.update(
                        detection_state=state.value,
                        main_distance_cm=dist_main,
                        bin_distances_cm=bin_dists,
                    )

                # ── poll UI command queue ─────────────────────────────
                if system_state:
                    cmd = system_state.get_command()
                    if cmd:
                        self._handle_command(cmd, hw)

                # ── OpenCV preview (headless-safe) ────────────────────
                if self._cv2_ok and frame is not None:
                    try:
                        preview = self._draw_preview(frame, dist_main, full_bin, state)
                        cv2.imshow("Smart Trash Bin", preview)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        # No display available (headless / no GTK) — disable preview
                        self._cv2_ok = False
                        logger.info("Headless mode: OpenCV preview disabled")

                # ── state machine ─────────────────────────────────────
                now          = time.monotonic()
                cooldown_ok  = (now - last_cooldown_end) > Config.COOLDOWN_SECONDS
                is_full      = full_bin is not None

                if not is_full and cooldown_ok and state == DetectionState.IDLE:
                    if dist_main < Config.DETECT_DISTANCE_CM:
                        if hand_present_start_time == 0.0:
                            hand_present_start_time = now
                        elif (now - hand_present_start_time) >= 0.5:
                            state = DetectionState.HAND_PRESENT
                            logger.info("Hand detected robustly (present >= 0.5s)")
                    else:
                        hand_present_start_time = 0.0

                elif state == DetectionState.HAND_PRESENT:
                    # Clear timer once we move out of IDLE
                    hand_present_start_time = 0.0
                    
                    if dist_main > Config.HAND_WITHDRAWN_DISTANCE_CM:
                        state = DetectionState.PROCESSING
                        logger.info("Hand withdrawn — starting pipeline")
                        self._process_detection()
                        last_cooldown_end = time.monotonic()
                        state = DetectionState.COOLDOWN
                        if system_state:
                            system_state.update(detection_state="cooldown")
                        time.sleep(Config.COOLDOWN_SECONDS)
                        state = DetectionState.IDLE


        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    # ── detection pipeline ────────────────────────────────────────────

    def _handle_command(self, cmd: dict, hw: HardwareController) -> None:
        """Execute a control command received from the UI/server."""
        action = cmd.get("action", "")
        if action == "stop":
            self._running = False
        elif action == "servo":
            name  = cmd.get("name", "")
            # Determine appropriate home angle based on name
            home_angle = Config.CAP_HOME_ANGLE if name == "capture" else Config.SORT_HOME_ANGLE
            angle = int(cmd.get("angle", home_angle))
            if name == "capture":
                # Increase speed to 25 deg/sec for smoother movement
                hw.smooth_move(Config.SERVO1_PIN, angle, speed=25.0)
                time.sleep(0.3)
                hw.idle_servos()
            elif name == "sort":
                hw.smooth_move(Config.SERVO2_PIN, angle, speed=25.0)
                time.sleep(0.3)
                hw.idle_servos()
            elif name == "all":
                # Manual 'all' command (e.g. from Home button)
                hw.smooth_move(Config.SERVO1_PIN, angle, speed=35.0)
                hw.smooth_move(Config.SERVO2_PIN, angle, speed=35.0)
                time.sleep(0.3)
                hw.idle_servos()
        elif action == "flash":
            self._send_flash(on=bool(cmd.get("on", False)))
        elif action == "mode":
            if system_state:
                system_state.update(mode=cmd.get("mode", "auto"))
        elif action == "set_roi":
            Config.ROI_CENTER_X = int(cmd.get("cx", 320))
            Config.ROI_CENTER_Y = int(cmd.get("cy", 240))
            Config.ROI_SIZE     = int(cmd.get("size", 320))
            self._save_roi_config()
            logger.info("ROI updated & saved: cx=%d cy=%d size=%d", Config.ROI_CENTER_X, Config.ROI_CENTER_Y, Config.ROI_SIZE)
        elif action == "update_config":
            if "detect_dist"    in cmd: Config.DETECT_DISTANCE_CM          = float(cmd["detect_dist"])
            if "withdraw_dist"  in cmd: Config.HAND_WITHDRAWN_DISTANCE_CM  = float(cmd["withdraw_dist"])
            if "cooldown"       in cmd: Config.COOLDOWN_SECONDS             = float(cmd["cooldown"])
            if "min_conf"       in cmd: Config.MIN_CONFIDENCE_THRESHOLD     = float(cmd["min_conf"])
        elif action == "manual_sort":
            cls = cmd.get("class", "reject")
            logger.info("Manual sort triggered: %s", cls)
            target = self._label_to_angle(cls)
            
            # Serve 2 (Sort): Slow and steady to stop swaying
            hw.smooth_move(Config.SERVO2_PIN, target, speed=30.0) 
            time.sleep(0.3)
            
            # Servo 1 (Capture): Needs high speed for torque/lifting power
            hw.smooth_move(Config.SERVO1_PIN, Config.SWEEP_ANGLE, speed=60.0)
            time.sleep(0.5)
            
            # Reset both back to Home
            hw.smooth_move(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE, speed=50.0)
            hw.smooth_move(Config.SERVO2_PIN, Config.SORT_HOME_ANGLE, speed=35.0)
            time.sleep(0.2)
            hw.idle_servos()
            if system_state:
                system_state.add_sort_event(cls, 1.0, 0.0)

    def _capture_http_frame(self) -> Optional[np.ndarray]:
        """Grab a single high-quality frame directly from ESP32 /capture endpoint."""
        url = f"http://{self._esp32_ip}/capture"
        try:
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            else:
                logger.error("HTTP capture returned status %d", resp.status_code)
        except Exception as exc:
            logger.error("HTTP capture error: %s", exc)
        return None

    def _process_detection(self) -> None:
        """Execute the full detection pipeline with guaranteed Reset."""
        t0 = time.monotonic()
        hw = self._hw
        assert hw

        try:
            # t=0.00 — Tilt capture arm to photo position (120°)
            # High speed here guarantees it has the torque to lift against gravity
            hw.smooth_move(Config.SERVO1_PIN, Config.PHOTO_ANGLE, speed=55.0)
            
            # Flash + Capture
            self._send_flash(on=True)
            time.sleep(0.5) 
            frame = self._capture_http_frame()
            self._send_flash(on=False) 
            
            if frame is None:
                logger.warning("Failed capture — aborting pipeline")
                return

            # Inference
            start_ms = time.time() * 1000.0
            label, conf = classify_single_frame(frame)
            elapsed_ms = (time.time() * 1000.0) - start_ms
            
            logger.info("[t=%.3f] Result: %s (conf=%.2f)", 
                        time.monotonic() - t0, label, conf)
            self._record_sort(label, conf, elapsed_ms)

            # Stage 1: Reset Arm to Neutral
            hw.smooth_move(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE, speed=50.0)

            # Stage 2: Rotate Bin (Servo 2)
            target_angle = self._label_to_angle(label)
            hw.smooth_move(Config.SERVO2_PIN, target_angle, speed=35.0)

            # Stage 3: Drop (Servo 1)
            time.sleep(0.2) 
            hw.smooth_move(Config.SERVO1_PIN, Config.SWEEP_ANGLE, speed=65.0)
            time.sleep(0.6)

        except Exception as exc:
            logger.error("Detection pipeline crashed: %s", exc)
        finally:
            # CRITICAL: Always return both servos to Home / Center
            hw.smooth_move(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE, speed=45.0)
            hw.smooth_move(Config.SERVO2_PIN, Config.SORT_HOME_ANGLE, speed=45.0)
            time.sleep(0.2)
            hw.idle_servos(force=True)
            logger.info("[t=%.3f] Pipeline finished & Reset", time.monotonic() - t0)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _label_to_angle(label: str) -> int:
        """Map predicted class name → Servo2 angle."""
        ll = label.lower()
        if "reject" in ll:
            return Config.SORT_HOME_ANGLE   # stay home, drop in default bin
        if "pet" in ll or "hdpe" in ll or "plastic" in ll:
            return Config.PLASTIC_ANGLE  # 112°
        if "glass" in ll:
            return Config.GLASS_ANGLE    # 157°
        return Config.METAL_ANGLE        # 67° (metal / AluCan / default)

    def _send_flash(self, on: bool) -> None:
        url = Config.flash_url(self._esp32_ip, on)
        try:
            requests.get(url, timeout=0.5)
        except requests.RequestException as exc:
            logger.debug("Flash command ignored: %s", exc)

    def _record_sort(self, label: str, conf: float, inference_ms: float = 0.0) -> None:
        """Record sort result to shared_state (used by UI + server)."""
        if system_state:
            system_state.add_sort_event(label, conf, inference_ms)
        else:
            logger.info("Sort: %s conf=%.2f inference=%.1fms", label, conf, inference_ms)

    @staticmethod
    def _draw_preview(
        frame: np.ndarray,
        dist_cm: float,
        full_bin: Optional[int],
        state: DetectionState,
    ) -> np.ndarray:
        """Overlay status text and ROI box on a preview frame."""
        preview = frame.copy()
        h, w = preview.shape[:2]

        # ROI rectangle
        roi = g_state.last_roi_coords
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 200, 0), 1)

        # Status text
        state_colors = {
            DetectionState.IDLE:         (0, 200, 100),
            DetectionState.HAND_PRESENT: (0, 200, 255),
            DetectionState.PROCESSING:   (0, 120, 255),
            DetectionState.COOLDOWN:     (100, 100, 100),
        }
        color = (0, 0, 255) if full_bin else state_colors.get(state, (255, 255, 255))
        status = (
            f"FULL: BIN {full_bin}" if full_bin
            else f"{state.value.upper()}  {dist_cm:.1f}cm"
        )
        cv2.putText(
            preview, status, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        return preview

    def shutdown(self) -> None:
        """Release all resources gracefully."""
        self._running = False
        if system_state:
            system_state.update(engine_running=False, detection_state="idle")
        if self._cam:
            self._cam.stop()
        if self._hw:
            self._hw.cleanup()
        if self._cv2_ok:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        logger.info("System shut down cleanly")


# =====================================================================
# 11. ENTRY POINT
# =====================================================================
def main() -> None:
    """Launch SmartBin V2 engine from CLI."""
    engine = SmartBinEngine()

    def _signal_handler(sig: int, _frame: object) -> None:
        logger.info("Signal %d received — shutting down …", sig)
        engine.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    engine.setup()
    engine.run()


if __name__ == "__main__":
    main()