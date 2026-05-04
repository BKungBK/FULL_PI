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
import base64
import math
import os
import signal
from dataclasses import dataclass
from typing import Callable, List
from collections import deque
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import field
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
    # ESP32 IP configuration:
    # - If Pi is a Wi-Fi hotspot (recommended for production): keep the hardcoded IP.
    #   The Pi's hotspot DHCP will always assign the same /24 range.
    #   Verify the ESP32's assigned IP with: cat /var/lib/misc/dnsmasq.leases
    # - If using a home router: set DHCP reservation on the router for ESP32's MAC,
    #   then update this value to match. Or set to None to auto-scan (slower boot).
    ESP32_IP_OVERRIDE: Optional[str] = "10.42.0.177"
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
    DETECT_DISTANCE_CM: float = 10.0
    HAND_WITHDRAWN_DISTANCE_CM: float = 20.0   # hysteresis for release
    BIN_FULL_DISTANCE_CM: float = 10.0
    COOLDOWN_SECONDS: float = 5.0
    MIN_CONFIDENCE_THRESHOLD: float = 0.50     # minimum score before reject
    UNCERTAINTY_REJECT_THRESHOLD: float = 0.30  # if 2+ classes exceed this → reject
    CAMERA_RECONNECT_DELAY: float = 1.0
    ULTRASONIC_TIMEOUT: float = 0.1
    YOLO_INPUT_SIZE: int = 640  # reserved for future NCNN YOLO pipeline
    CAMERA_BUFFER_SIZE: int = 10  # rolling frame buffer depth

    # ── Servo timing (seconds) ───────────────────────────────────────
    SERVO_UPDATE_HZ: float = 50.0            # match PWM period (20 ms)
    SERVO_SETTLE_TIME_S: float = 0.20        # post-move mechanical damping
    SERVO_HOLD_TIME_S: float = 0.30          # keep PWM alive after reaching target
    HOME_VERIFY_DELAY_S: float = 0.40        # extra hold after re-command home (prevents sag)
    PHOTO_SETTLE_DELAY: float = 0.25         # wait after servo reaches PHOTO_ANGLE
    FLASH_SETTLE_DELAY: float = 0.20         # wait after flash ON before capture
    HOME_ALL_DELAY_S: float = 0.80           # worst-case full-sweep homing at startup

    # ── Reject bin servo angle ───────────────────────────────────────
    REJECT_SERVO_ANGLE: int = 86   # default = SORT_HOME_ANGLE; set to 4th bin angle if hardware supports

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
    last_ai_image_b64: str = ""   # base64 JPEG of the image actually fed to predict()


g_state = GlobalState()


# =====================================================================
# 3. LOG BROADCAST HANDLER  (send logs to WebSocket)
# =====================================================================
class WebSocketLogHandler(logging.Handler):
    """Custom log handler that sends logs to WebSocket clients.
    
    Queues recent logs and broadcasts them via shared_state.
    """
    def __init__(self, max_logs: int = 100):
        super().__init__()
        self._log_queue: deque = deque(maxlen=max_logs)
        self._log_cbs: List[Callable] = []
        
    def emit(self, record: logging.LogRecord) -> None:
        """Called whenever a log record is created."""
        try:
            log_entry = {
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "message": self.format(record),
                "source": record.name
            }
            self._log_queue.append(log_entry)
            # Notify callbacks (server will register one)
            for cb in self._log_cbs:
                try:
                    cb(log_entry)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback to receive log updates."""
        if callback not in self._log_cbs:
            self._log_cbs.append(callback)
    
    def get_recent_logs(self, n: int = 50) -> List[dict]:
        """Get recent n log entries."""
        return list(self._log_queue)[-n:]


# Global log handler instance
_ws_log_handler: Optional[WebSocketLogHandler] = None


def setup_ws_logging() -> WebSocketLogHandler:
    """Setup WebSocket log handler and attach to root logger."""
    global _ws_log_handler
    if _ws_log_handler is None:
        _ws_log_handler = WebSocketLogHandler(max_logs=200)
        _ws_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        _ws_log_handler.setFormatter(formatter)
        
        # Attach to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(_ws_log_handler)
        
    return _ws_log_handler


def get_ws_log_handler() -> Optional[WebSocketLogHandler]:
    """Get the WebSocket log handler instance."""
    return _ws_log_handler


# =====================================================================
# 4. SIMPLE SERVO LOCK
# =====================================================================
class SimpleServoLock:
    """
    Simple position lock - refreshes servo command periodically.
    
    Reference: https://www.pololu.com/blog/17/servo-control-interface-in-detail
    - Analog servos need regular pulses to maintain position
    - 200ms refresh rate is sufficient for holding (5Hz)
    """
    
    def __init__(self, hw_controller: "HardwareController"):
        self._hw = hw_controller
        self._lock_active = False
        self._lock_thread: Optional[threading.Thread] = None
        self._stop_lock = threading.Event()
        
    def start_lock(self, pin: int, angle: int) -> None:
        """Start simple lock - refreshes position every 200ms."""
        if self._lock_active:
            self.stop_lock()
        
        self._lock_active = True
        self._stop_lock.clear()
        
        # Set initial position
        self._hw.move_servo(pin, angle)
        
        # Start lock thread
        self._lock_thread = threading.Thread(
            target=self._lock_loop,
            args=(pin, int(angle)),
            daemon=True
        )
        self._lock_thread.start()
        logger.info("Lock: Started on pin=%d at %d°", pin, angle)
    
    def _lock_loop(self, pin: int, angle: int) -> None:
        """Simple lock loop - refresh every 200ms."""
        while not self._stop_lock.is_set():
            self._hw.move_servo(pin, angle)
            self._stop_lock.wait(timeout=0.2)  # 200ms refresh
    
    def stop_lock(self) -> None:
        """Stop the lock."""
        if not self._lock_active:
            return
        self._stop_lock.set()
        if self._lock_thread and self._lock_thread.is_alive():
            self._lock_thread.join(timeout=1.0)
        self._lock_active = False
        logger.info("Lock: Stopped")
    
    def is_locked(self) -> bool:
        """Check if lock is active."""
        return self._lock_active and self._lock_thread and self._lock_thread.is_alive()


# =====================================================================
# 5. SERVO SPECIFICATION & CONTROL  (Professional PWM-based)
# =====================================================================
# Step interval = 20ms — matches servo refresh rate (50Hz)
# Sending pulses faster than this has no benefit
_STEP_S = 0.020


def ease_in_out_cubic(t: float) -> float:
    """Smooth start & end — ไม่กระตุก"""
    return 3 * t**2 - 2 * t**3


@dataclass(frozen=True)
class ServoSpec:
    """Servo hardware specification with calibrated pulse widths."""
    name:    str
    pin:     int
    min_pw:  int    # µs (pulsewidth ต่ำสุด)
    max_pw:  int    # µs (pulsewidth สูงสุด)
    min_deg: float  # องศาต่ำสุด
    max_deg: float  # องศาสูงสุด
    freq_hz: int = 50  # Standard servo frequency


# Calibrated specs — tune these if mechanical range is off
SERVO_SPECS: dict[str, ServoSpec] = {
    "MG995": ServoSpec(
        "MG995", 
        pin=Config.SERVO1_PIN, 
        min_pw=1000, max_pw=2000, 
        min_deg=0, max_deg=180
    ),
    "DS5180SSG": ServoSpec(
        "DS5180SSG", 
        pin=Config.SERVO2_PIN, 
        min_pw=500, max_pw=2500, 
        min_deg=0, max_deg=270
    ),
}


def _deg_to_pw(spec: ServoSpec, deg: float) -> int:
    """Convert degrees to pulse width in microseconds."""
    deg = max(spec.min_deg, min(spec.max_deg, deg))
    ratio = (deg - spec.min_deg) / (spec.max_deg - spec.min_deg)
    return round(spec.min_pw + ratio * (spec.max_pw - spec.min_pw))


@dataclass
class _ServoState:
    """Internal per-servo state."""
    spec:        ServoSpec
    current_deg: Optional[float] = None
    _cancel:     threading.Event = field(default_factory=threading.Event)
    _lock:       threading.Lock = field(default_factory=threading.Lock)
    _active_thread: Optional[threading.Thread] = None


# =====================================================================
# 4. DETECTION STATE MACHINE
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

# ── NCNN YOLO (reserved for future use) ──────────────────────────────

@dataclass
class LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int
    orig_h: int
    orig_w: int


def preprocess_yolo(frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
    """Preprocess frame for YOLO NCNN input (letterbox resize, RGB CHW float32).

    NOTE: This function is reserved for future NCNN pipeline integration.
    The current detection pipeline uses classify_single_frame() → TFLite only.
    """
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

        # Tier 1 — absolute low confidence
        if conf < Config.MIN_CONFIDENCE_THRESHOLD:
            logger.info("Softmax %s | top=%s %.3f → Tier-1 reject (low conf)",
                        np.round(output, 3).tolist(),
                        Config.CLASS_NAMES[idx] if idx < len(Config.CLASS_NAMES) else "?", conf)
            return "reject", conf

        # Tier 2 — model is confused between multiple classes
        ambiguous_count = int(np.sum(output >= Config.UNCERTAINTY_REJECT_THRESHOLD))
        if ambiguous_count >= 2:
            top_name = Config.CLASS_NAMES[idx] if idx < len(Config.CLASS_NAMES) else "?"
            logger.info("Softmax %s | top=%s %.3f | ambiguous=%d → Tier-2 reject (confused)",
                        np.round(output, 3).tolist(), top_name, conf, ambiguous_count)
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

        logger.info("Softmax %s | top=%s %.3f | ambiguous=%d → %s",
                    np.round(output, 3).tolist(),
                    class_name_raw, conf, ambiguous_count, class_name)
        return class_name, conf
    except Exception as e:
        logger.error("Predict error: %s", e)
        return "reject", 0.0


def _encode_jpg_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV BGR image to base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def classify_single_frame(frame: np.ndarray) -> Tuple[str, float]:
    """Full pipeline: extract ROI + prepare image + classify one frame."""
    roi_frame, roi_coords = crop_roi(frame)
    g_state.last_roi_coords = roi_coords
    roi224 = prepare_image(roi_frame)

    # ── Match exact dataset_tool.py preprocessing ──
    processed = apply_clahe(roi224)

    # ── Keep base64 snapshot of the exact AI input image ──
    g_state.last_ai_image_b64 = _encode_jpg_b64(processed)

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
        # Initialize servo states from specs
        self._servos: dict[str, _ServoState] = {}
        for key, spec in SERVO_SPECS.items():
            self._servos[key] = _ServoState(spec=spec)
        self._smart_lock = SimpleServoLock(self)
        self._init_pins()
        
    def _pin_to_key(self, pin: int) -> str:
        """Map GPIO pin to servo key."""
        if pin == Config.SERVO1_PIN:
            return "MG995"
        elif pin == Config.SERVO2_PIN:
            return "DS5180SSG"
        raise ValueError(f"Unknown servo pin: {pin}")
        
    def _key_to_pin(self, key: str) -> int:
        """Map servo key to GPIO pin."""
        return SERVO_SPECS[key].pin

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

    # ── servo (Professional ServoSpec-based control) ─────────────────────

    def move_servo(self, pin: int, angle: float) -> None:
        """Low-level: Send servo to angle using calibrated pulse width."""
        key = self._pin_to_key(pin)
        state = self._servos[key]
        pw = _deg_to_pw(state.spec, angle)
        with state._lock:
            lgpio.tx_servo(self._h, pin, pw, state.spec.freq_hz)
            state.current_deg = angle

    def move_instant(self, pin: int, angle: float) -> None:
        """Instant move - no easing, immediate positioning."""
        key = self._pin_to_key(pin)
        state = self._servos[key]
        self._cancel_and_wait(state)
        self.move_servo(pin, angle)
        logger.debug("[%s] instant → %.1f°", key, angle)

    def move_smooth(
        self,
        pin: int,
        target: float,
        duration: float = 0.5,
    ) -> None:
        """
        Non-blocking smooth move with ease-in-out.
        Cancels any ongoing move on this servo and starts fresh.
        Uses 20ms steps matching 50Hz servo refresh.
        """
        key = self._pin_to_key(pin)
        state = self._servos[key]
        self._cancel_and_wait(state)

        # Determine start position
        if state.current_deg is None:
            # Unknown position - assume at configured home
            if key == "MG995":
                start = Config.CAP_HOME_ANGLE
            else:
                start = Config.SORT_HOME_ANGLE
        else:
            start = state.current_deg

        # Calculate steps
        total_steps = max(1, round(duration / _STEP_S))
        state._cancel = threading.Event()

        logger.info("[%s] smooth %.1f° → %.1f° (%.2fs, %d steps)",
                   key, start, target, duration, total_steps)

        def _worker():
            for i in range(total_steps + 1):
                if state._cancel.is_set():
                    return
                t = i / total_steps
                deg = start + (target - start) * ease_in_out_cubic(t)
                self.move_servo(pin, deg)
                time.sleep(_STEP_S)
            logger.debug("[%s] move complete @ %.1f°", key, target)

        state._active_thread = threading.Thread(
            target=_worker, daemon=True, name=f"servo-{key}"
        )
        state._active_thread.start()

    def move_smooth_wait(
        self,
        pin: int,
        target: float,
        duration: float = 0.5,
    ) -> None:
        """Smooth move but blocking - waits until complete."""
        self.move_smooth(pin, target, duration)
        time.sleep(duration + _STEP_S * 2)  # +buffer

    def _cancel_and_wait(self, state: _ServoState) -> None:
        """Cancel ongoing move and wait for thread to finish."""
        state._cancel.set()
        if state._active_thread and state._active_thread.is_alive():
            state._active_thread.join(timeout=_STEP_S * 2)
        state._cancel = threading.Event()

    def get_position(self, pin: int) -> Optional[float]:
        """Get current servo position (degrees)."""
        key = self._pin_to_key(pin)
        return self._servos[key].current_deg

    def warmup_servo(self, pin: int) -> None:
        """Ensure PWM is active. Called after idle or before first move."""
        key = self._pin_to_key(pin)
        state = self._servos[key]
        if state.current_deg is None:
            # Unknown - move to configured home
            if key == "MG995":
                home = Config.CAP_HOME_ANGLE
            else:
                home = Config.SORT_HOME_ANGLE
            self.move_servo(pin, home)
            time.sleep(1.0)  # worst-case full sweep
        else:
            self.move_servo(pin, state.current_deg)
            time.sleep(0.05)  # 1+ PWM period

    def home_all(self) -> None:
        """Forceful return to home. Call once at startup."""
        for key, home in (("MG995", Config.CAP_HOME_ANGLE),
                          ("DS5180SSG", Config.SORT_HOME_ANGLE)):
            pin = SERVO_SPECS[key].pin
            self.warmup_servo(pin)
            self.move_servo(pin, home)
            time.sleep(Config.HOME_ALL_DELAY_S)
            logger.info("Home complete %s pin=%d angle=%.0f°", key, pin, home)

    def return_to_home_simple(self) -> None:
        """Simple sequential home return."""
        logger.info("Returning to home (simple)...")
        # MG995 first (slower)
        self.move_smooth_wait(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE, duration=0.6)
        # DS5180SSG second (faster)
        self.move_smooth_wait(Config.SERVO2_PIN, Config.SORT_HOME_ANGLE, duration=0.4)
        time.sleep(0.2)
        logger.info("Home return complete")

    def move_simple(self, pin: int, target: int, speed_ms: int = 10) -> None:
        """Legacy compatibility - uses instant move for now."""
        self.move_instant(pin, float(target))

    def idle_servos(self, force: bool = False) -> None:
        """Stop PWM to reduce heat. Only idles if at home or forced."""
        for key in ("MG995", "DS5180SSG"):
            spec = SERVO_SPECS[key]
            state = self._servos[key]
            home = Config.CAP_HOME_ANGLE if key == "MG995" else Config.SORT_HOME_ANGLE
            angle = state.current_deg if state.current_deg is not None else home
            if force or abs(angle - home) < 2.0:
                try:
                    lgpio.tx_servo(self._h, spec.pin, 0)
                except Exception:
                    pass
                state.current_deg = None

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
        self._mode     = "auto"    # "auto" | "manual" — mirrors shared_state
        self._hw:  Optional[HardwareController] = None
        self._cam: Optional[CameraStream] = None
        self._esp32_ip: str = ""
        self._cv2_ok: bool = True   # flipped False on first headless cv2.error
        self._cycle_count = 0       # For periodic auto-rehome
        self._load_config()

    def _load_config(self) -> None:
        """Load unified config from smartbin_config.json (falls back to roi_config.json)."""
        import os, json
        path = "smartbin_config.json"
        # Migration: read old roi_config.json if new file doesn't exist
        if not os.path.exists(path) and os.path.exists("roi_config.json"):
            try:
                with open("roi_config.json", "r") as f:
                    old = json.load(f)
                with open(path, "w") as f:
                    json.dump({"roi": old}, f)
                logger.info("Migrated roi_config.json -> smartbin_config.json")
            except Exception as e:
                logger.error("Migration failed: %s", e)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # ROI
            roi = data.get("roi", {})
            Config.ROI_CENTER_X = int(roi.get("cx", 320))
            Config.ROI_CENTER_Y = int(roi.get("cy", 240))
            Config.ROI_SIZE     = int(roi.get("size", 320))
            # Servo angles
            sa = data.get("servo_angles", {})
            Config.CAP_HOME_ANGLE  = int(sa.get("cap_home", Config.CAP_HOME_ANGLE))
            Config.SORT_HOME_ANGLE = int(sa.get("sort_home", Config.SORT_HOME_ANGLE))
            Config.PHOTO_ANGLE     = int(sa.get("photo", Config.PHOTO_ANGLE))
            Config.SWEEP_ANGLE     = int(sa.get("sweep", Config.SWEEP_ANGLE))
            Config.PLASTIC_ANGLE   = int(sa.get("plastic", Config.PLASTIC_ANGLE))
            Config.GLASS_ANGLE     = int(sa.get("glass", Config.GLASS_ANGLE))
            Config.METAL_ANGLE     = int(sa.get("metal", Config.METAL_ANGLE))
            Config.REJECT_SERVO_ANGLE = int(sa.get("reject", Config.REJECT_SERVO_ANGLE))
            # Detection / timing
            dt = data.get("detection", {})
            Config.DETECT_DISTANCE_CM = float(dt.get("detect_dist", Config.DETECT_DISTANCE_CM))
            Config.HAND_WITHDRAWN_DISTANCE_CM = float(dt.get("withdraw_dist", Config.HAND_WITHDRAWN_DISTANCE_CM))
            Config.COOLDOWN_SECONDS = float(dt.get("cooldown", Config.COOLDOWN_SECONDS))
            Config.MIN_CONFIDENCE_THRESHOLD = float(dt.get("min_conf", Config.MIN_CONFIDENCE_THRESHOLD))
            Config.UNCERTAINTY_REJECT_THRESHOLD = float(dt.get("uncertainty_reject", Config.UNCERTAINTY_REJECT_THRESHOLD))
            logger.info("Loaded smartbin_config.json")
        except Exception as e:
            logger.error("Failed to load config: %s", e)

    def _save_config(self) -> None:
        """Persist all tunable config to smartbin_config.json."""
        import json
        try:
            with open("smartbin_config.json", "w") as f:
                json.dump({
                    "roi": {
                        "cx": Config.ROI_CENTER_X,
                        "cy": Config.ROI_CENTER_Y,
                        "size": Config.ROI_SIZE,
                    },
                    "servo_angles": {
                        "cap_home": Config.CAP_HOME_ANGLE,
                        "sort_home": Config.SORT_HOME_ANGLE,
                        "photo": Config.PHOTO_ANGLE,
                        "sweep": Config.SWEEP_ANGLE,
                        "plastic": Config.PLASTIC_ANGLE,
                        "glass": Config.GLASS_ANGLE,
                        "metal": Config.METAL_ANGLE,
                        "reject": Config.REJECT_SERVO_ANGLE,
                    },
                    "detection": {
                        "detect_dist": Config.DETECT_DISTANCE_CM,
                        "withdraw_dist": Config.HAND_WITHDRAWN_DISTANCE_CM,
                        "cooldown": Config.COOLDOWN_SECONDS,
                        "min_conf": Config.MIN_CONFIDENCE_THRESHOLD,
                        "uncertainty_reject": Config.UNCERTAINTY_REJECT_THRESHOLD,
                    },
                }, f, indent=2)
            logger.info("Saved smartbin_config.json")
        except Exception as e:
            logger.error("Failed to save config: %s", e)

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
        self._hw.home_all()
        self._hw.idle_servos(force=True)

        # ── SIGTERM handler for systemd graceful shutdown ──────────────────
        try:
            import signal as _signal

            def _sigterm_handler(sig, frame):
                logger.info("SIGTERM received — initiating graceful shutdown")
                self._running = False

            _signal.signal(_signal.SIGTERM, _sigterm_handler)
            logger.info("SIGTERM handler registered")
        except ValueError:
            # signal.signal() only works in the main thread.
            # When running via server.py (EngineThread), shutdown is handled
            # by FastAPI lifespan cleanup instead.
            logger.info("SIGTERM handler skipped (background thread — "
                        "server lifespan handles shutdown)")

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

                if self._mode == "auto" and not is_full and cooldown_ok and state == DetectionState.IDLE:
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

                    if self._mode != "auto":
                        # Mode switched to manual while hand was present
                        state = DetectionState.IDLE
                        logger.info("Mode → manual — aborting detection")
                    elif dist_main > Config.HAND_WITHDRAWN_DISTANCE_CM:
                        state = DetectionState.PROCESSING
                        logger.info("Hand withdrawn — starting pipeline")
                        self._process_detection()
                        self._cycle_count += 1
                        
                        # Auto-rehome every 3 cycles to prevent drift accumulation
                        if self._cycle_count % 3 == 0 and self._hw:
                            logger.info("Periodic re-home (cycle %d)", self._cycle_count)
                            self._hw.return_to_home_simple()
                        
                        last_cooldown_end = time.monotonic()
                        state = DetectionState.COOLDOWN
                        if system_state:
                            system_state.update(detection_state="cooldown")
                        # ── Interruptible cooldown (responds to SIGTERM/stop ≤100ms) ──
                        cooldown_deadline = time.monotonic() + Config.COOLDOWN_SECONDS
                        while self._running and time.monotonic() < cooldown_deadline:
                            time.sleep(0.1)
                        state = DetectionState.IDLE

                # ── Yield CPU to prevent 100% burn in idle (20ms ≈ 50 Hz) ──
                time.sleep(0.02)

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
                hw.smooth_move(Config.SERVO1_PIN, angle, speed=30.0)
                time.sleep(Config.SERVO_HOLD_TIME_S)
                hw.idle_servos(force=True)
            elif name == "sort":
                hw.smooth_move(Config.SERVO2_PIN, angle, speed=30.0)
                time.sleep(Config.SERVO_HOLD_TIME_S)
                hw.idle_servos(force=True)
            elif name == "all":
                # Manual 'all' command (e.g. from Home button)
                hw.smooth_move(Config.SERVO1_PIN, angle, speed=30.0)
                hw.smooth_move(Config.SERVO2_PIN, angle, speed=30.0)
                time.sleep(Config.SERVO_HOLD_TIME_S)
                hw.idle_servos(force=True)
        elif action == "flash":
            self._send_flash(on=bool(cmd.get("on", False)))
        elif action == "mode":
            new_mode = cmd.get("mode", "auto")
            self._mode = new_mode
            if system_state:
                system_state.update(mode=new_mode)
            logger.info("Mode changed to: %s", new_mode)
        elif action == "set_roi":
            Config.ROI_CENTER_X = int(cmd.get("cx", 320))
            Config.ROI_CENTER_Y = int(cmd.get("cy", 240))
            Config.ROI_SIZE     = int(cmd.get("size", 320))
            self._save_config()
            logger.info("ROI updated & saved: cx=%d cy=%d size=%d", Config.ROI_CENTER_X, Config.ROI_CENTER_Y, Config.ROI_SIZE)
        elif action == "update_config":
            if "detect_dist"        in cmd: Config.DETECT_DISTANCE_CM              = float(cmd["detect_dist"])
            if "withdraw_dist"      in cmd: Config.HAND_WITHDRAWN_DISTANCE_CM      = float(cmd["withdraw_dist"])
            if "cooldown"           in cmd: Config.COOLDOWN_SECONDS                 = float(cmd["cooldown"])
            if "min_conf"           in cmd: Config.MIN_CONFIDENCE_THRESHOLD         = float(cmd["min_conf"])
            if "uncertainty_reject" in cmd: Config.UNCERTAINTY_REJECT_THRESHOLD     = float(cmd["uncertainty_reject"])
            if "photo_settle"       in cmd: Config.PHOTO_SETTLE_DELAY               = float(cmd["photo_settle"])
            if "flash_settle"       in cmd: Config.FLASH_SETTLE_DELAY               = float(cmd["flash_settle"])
            if "servo_settle"       in cmd: Config.SERVO_SETTLE_TIME_S            = float(cmd["servo_settle"])
            if "servo_hold"         in cmd: Config.SERVO_HOLD_TIME_S                = float(cmd["servo_hold"])
            if "home_verify"        in cmd: Config.HOME_VERIFY_DELAY_S               = float(cmd["home_verify"])
            # Servo angles
            if "cap_home"  in cmd: Config.CAP_HOME_ANGLE  = int(cmd["cap_home"])
            if "sort_home" in cmd: Config.SORT_HOME_ANGLE = int(cmd["sort_home"])
            if "photo"     in cmd: Config.PHOTO_ANGLE     = int(cmd["photo"])
            if "sweep"     in cmd: Config.SWEEP_ANGLE     = int(cmd["sweep"])
            if "plastic"   in cmd: Config.PLASTIC_ANGLE   = int(cmd["plastic"])
            if "glass"     in cmd: Config.GLASS_ANGLE     = int(cmd["glass"])
            if "metal"     in cmd: Config.METAL_ANGLE     = int(cmd["metal"])
            if "reject"    in cmd: Config.REJECT_SERVO_ANGLE = int(cmd["reject"])
            self._save_config()
            logger.info("Config updated: %s", {k: cmd[k] for k in cmd if k != "action"})
        elif action == "manual_sort":
            cls = cmd.get("class", "reject")
            logger.info("Manual sort triggered: %s", cls)
            target = self._label_to_angle(cls)
            try:
                # Servo 2 (Sort): Simple move to target bin
                hw.move_simple(Config.SERVO2_PIN, target, speed_ms=10)
                time.sleep(0.2)
                # Servo 1 (Capture): Drop position
                hw.move_simple(Config.SERVO1_PIN, Config.SWEEP_ANGLE, speed_ms=15)
                time.sleep(0.4)
            except Exception as exc:
                logger.error("Manual sort servo error: %s", exc)
            finally:
                # Return home simply
                hw.return_to_home_simple()
                time.sleep(0.3)
                hw.idle_servos(force=False)
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

        # Stop any active smart locks from previous cycle
        if hw._smart_lock.is_locked():
            logger.debug("Stopping previous smart lock")
            hw._smart_lock.stop_lock()

        try:
            # Stage 0 — Move capture arm to photo position and wait for stable
            logger.info("[t=%.3f] === STAGE 0: Move to Photo Position (%d°) ===", 
                       time.monotonic() - t0, Config.PHOTO_ANGLE)
            hw.move_simple(Config.SERVO1_PIN, Config.PHOTO_ANGLE, speed_ms=15)
            
            # Wait for servo to fully settle at photo position
            logger.info("[t=%.3f] Waiting for servo to settle...", time.monotonic() - t0)
            time.sleep(Config.PHOTO_SETTLE_DELAY)
            
            # Turn ON flash and keep it on for 1 second before capture
            # This ensures stable lighting and servo position
            logger.info("[t=%.3f] Flash ON - waiting 1 second before capture", time.monotonic() - t0)
            self._send_flash(on=True)
            time.sleep(1.0)  # USER REQUEST: keep flash on for ~1 second
            
            # Now capture the frame
            logger.info("[t=%.3f] Capturing frame...", time.monotonic() - t0)
            frame = self._capture_http_frame()
            
            # Turn OFF flash after capture
            self._send_flash(on=False)
            logger.info("[t=%.3f] Flash OFF", time.monotonic() - t0)

            # Log commanded angle at capture moment (open-loop — no feedback)
            servo1_cmd = hw.get_position(Config.SERVO1_PIN) or Config.PHOTO_ANGLE
            logger.info("[t=%.3f] Capture frame at servo1_cmd=%.0f°", time.monotonic() - t0, servo1_cmd)

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
            logger.info("[t=%.3f] === STAGE 1: Reset Arm to Neutral (home=%d°) ===", 
                       time.monotonic() - t0, Config.CAP_HOME_ANGLE)
            hw.move_simple(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE, speed_ms=15)

            # Stage 2: Rotate Bin (Servo 2)
            target_angle = self._label_to_angle(label)
            
            # Get actual position, don't assume home if unknown
            current_servo2_pos = hw.get_position(Config.SERVO2_PIN)
            if current_servo2_pos is None:
                logger.warning("[t=%.3f] Servo2 position unknown! Assuming at home %d°", 
                              time.monotonic() - t0, Config.SORT_HOME_ANGLE)
                current_servo2_pos = Config.SORT_HOME_ANGLE
            
            logger.info("[t=%.3f] === STAGE 2: Rotate Bin ===", time.monotonic() - t0)
            logger.info("[t=%.3f] Label='%s' → target_angle=%d°", time.monotonic() - t0, label, target_angle)
            logger.info("[t=%.3f] Servo2 current=%.1f°, target=%d°, distance=%.1f°",
                       time.monotonic() - t0, current_servo2_pos, target_angle,
                       abs(target_angle - current_servo2_pos))
            
            distance_servo2 = abs(target_angle - current_servo2_pos)
            if distance_servo2 < 1.0:
                logger.warning("[t=%.3f] Servo2 target ≈ current (dist=%.1f°). Still forcing move to ensure position.", 
                              time.monotonic() - t0, distance_servo2)
                # Force move
                hw.move_simple(Config.SERVO2_PIN, target_angle, speed_ms=10)
            else:
                hw.move_simple(Config.SERVO2_PIN, target_angle, speed_ms=10)
                logger.info("[t=%.3f] Servo2 movement complete", time.monotonic() - t0)

            # Stage 3: Drop (Servo 1)
            time.sleep(0.2)
            hw.move_simple(Config.SERVO1_PIN, Config.SWEEP_ANGLE, speed_ms=15)
            # USER REQUEST: Wait 2 seconds after drop before returning home
            time.sleep(2.0)

        except Exception as exc:
            logger.error("Detection pipeline crashed: %s", exc)
        finally:
            # Simple home return and lock
            hw.return_to_home_simple()
            
            # USER REQUEST: Wait 2 seconds at home before idling
            logger.info("[t=%.3f] At home, waiting 2 seconds...", time.monotonic() - t0)
            time.sleep(2.0)
            
            # Lock both servos at home
            logger.info("[t=%.3f] Locking servos at home", time.monotonic() - t0)
            hw._smart_lock.start_lock(Config.SERVO1_PIN, Config.CAP_HOME_ANGLE)
            hw._smart_lock.start_lock(Config.SERVO2_PIN, Config.SORT_HOME_ANGLE)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _label_to_angle(label: str) -> int:
        """Map predicted class name → Servo2 angle."""
        ll = label.lower()
        if "reject" in ll:
            return Config.REJECT_SERVO_ANGLE   # dedicated reject bin (defaults to home)
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
            system_state.add_sort_event(
                label, conf, inference_ms,
                image_b64=g_state.last_ai_image_b64
            )
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
        """Release all resources gracefully. Safe to call multiple times."""
        self._running = False
        if system_state:
            system_state.update(engine_running=False, detection_state="idle")
        if self._cam:
            self._cam.stop()
            self._cam = None
        if self._hw:
            self._hw.cleanup()
            self._hw = None
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