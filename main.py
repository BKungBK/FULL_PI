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

    # ── YOLO NCNN Stage-1 ─────────────────────────────────────────────
    YOLO_MODEL_DIR: str = "best_ncnn_model"
    YOLO_INPUT_SIZE: int = 320
    YOLO_CONF_THRESHOLD: float = 0.40
    YOLO_IOU_THRESHOLD: float = 0.45

    # ── TFLite Stage-2 Classifier ──────────────────────────────────────
    CLASSIFIER_MODEL_PATH: str = "stage2_efficientnet_int8.tflite"
    TFLITE_THREADS: int = 4
    VOTE_FRAMES: int = 5
    # EfficientNet output indices: 0=metal, 1=plastic, 2=glass, 3=reject
    # (YOLO class names AluCan/Glass/HDPEM/PET are for detection only — not used for sorting)
    CLASS_NAMES: Tuple[str, ...] = ("metal", "plastic", "glass", "reject")

    # ── ROI window (over 640×480 ESP32 stream) ─────────────────────────
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
    CENTER_ANGLE: int = 92
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
    yolo_detector: Optional["NcnnYoloDetector"] = None
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

@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int
    orig_h: int
    orig_w: int


class NcnnYoloDetector:
    """NCNN-backed YOLO detector that returns a single best bbox per frame."""

    INPUT_NAME = "in0"
    OUTPUT_NAME = "out0"

    def __init__(self, model_dir: str, input_size: int, conf_threshold: float, num_threads: int):
        model_path = Path(model_dir)
        param_path = model_path / "model.ncnn.param"
        bin_path = model_path / "model.ncnn.bin"

        if not param_path.exists() or not bin_path.exists():
            raise FileNotFoundError(
                f"NCNN model files not found in {model_dir} "
                f"(expected {param_path.name} and {bin_path.name})"
            )

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.net = ncnn.Net()

        if hasattr(self.net, "opt"):
            if hasattr(self.net.opt, "use_vulkan_compute"):
                self.net.opt.use_vulkan_compute = False
            if hasattr(self.net.opt, "num_threads"):
                self.net.opt.num_threads = num_threads

        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Run YOLO NCNN inference and return the best bbox in frame coordinates."""
        input_tensor, letterbox = preprocess_yolo(frame)

        # NOTE: ncnn.Extractor may not support context manager in all versions.
        # Use explicit create + call pattern for maximum compatibility.
        extractor = self.net.create_extractor()
        extractor.input(self.INPUT_NAME, ncnn.Mat(input_tensor).clone())
        ret, output = extractor.extract(self.OUTPUT_NAME)

        if ret != 0:
            raise RuntimeError(f"NCNN extractor failed with code {ret}")

        return postprocess_yolo(
            np.array(output, dtype=np.float32),
            self.conf_threshold,
            Config.YOLO_IOU_THRESHOLD,
            letterbox,
        )


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
    """Load YOLO NCNN detector and TFLite classifier.

    CLASS_NAMES = ("metal", "plastic", "glass", "reject") is hardcoded in Config.
    No external labels file is required — EfficientNet output indices map directly:
        0 → metal  |  1 → plastic  |  2 → glass  |  3 → reject
    """
    logger.info("Class names: %s", Config.CLASS_NAMES)

    logger.info("Loading YOLO NCNN detector from %s...", Config.YOLO_MODEL_DIR)
    g_state.yolo_detector = NcnnYoloDetector(
        model_dir=Config.YOLO_MODEL_DIR,
        input_size=Config.YOLO_INPUT_SIZE,
        conf_threshold=Config.YOLO_CONF_THRESHOLD,
        num_threads=Config.TFLITE_THREADS,
    )
    logger.info(
        "YOLO NCNN detector loaded. Input size: %dx%d",
        Config.YOLO_INPUT_SIZE, Config.YOLO_INPUT_SIZE,
    )

    logger.info("Loading classifier from %s...", Config.CLASSIFIER_MODEL_PATH)
    g_state.classifier_interpreter = load_tflite_model(Config.CLASSIFIER_MODEL_PATH)
    input_detail  = g_state.classifier_interpreter.get_input_details()[0]
    output_detail = g_state.classifier_interpreter.get_output_details()[0]
    input_h, input_w = get_image_size_from_shape(input_detail["shape"])
    scale, zp = _extract_quant_params(input_detail)
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


def normalize_detector_output(output: np.ndarray) -> np.ndarray:
    """
    Normalize detector output to [num_predictions, num_channels].

    Supports both legacy TFLite-style channel-first output and NCNN output,
    which may be either [num_predictions, channels] or [channels, num_predictions].
    """
    predictions = np.asarray(output)
    predictions = np.squeeze(predictions)

    if predictions.ndim != 2:
        raise ValueError(f"Unexpected detector output shape: {predictions.shape}")

    if predictions.shape[1] <= 32:
        normalized = predictions
    elif predictions.shape[0] <= 32:
        normalized = predictions.T
    else:
        normalized = predictions if predictions.shape[1] < predictions.shape[0] else predictions.T

    if normalized.shape[1] < 5:
        raise ValueError(f"Detector output has too few channels: {normalized.shape}")

    return normalized.astype(np.float32, copy=False)


def postprocess_yolo(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    letterbox: LetterboxInfo,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse YOLO output and return best detection bbox (x1, y1, x2, y2).
    Simplified for single-object waste detection.
    """
    del iou_thresh  # Reserved for future NMS improvements; best-score pick is enough here.

    predictions = normalize_detector_output(output)
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = np.max(class_scores, axis=1)

    keep = scores > conf_thresh
    if not np.any(keep):
        return None

    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    best_box = filtered_boxes[int(np.argmax(filtered_scores))]

    cx, cy, bw, bh = [float(v) for v in best_box]
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    x1 = (x1 - letterbox.pad_left) / letterbox.scale
    y1 = (y1 - letterbox.pad_top) / letterbox.scale
    x2 = (x2 - letterbox.pad_left) / letterbox.scale
    y2 = (y2 - letterbox.pad_top) / letterbox.scale

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(letterbox.orig_w), x2), min(float(letterbox.orig_h), y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (int(x1), int(y1), int(x2), int(y2))


def run_yolo_detection(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO detection on frame with ROI crop, return bbox or None.

    The bbox is returned in original frame coordinates (not ROI coordinates).
    """
    # First, crop ROI from frame
    roi_frame, roi_coords = crop_roi(frame)
    g_state.last_roi_coords = roi_coords

    detector = g_state.yolo_detector
    if detector is None:
        raise RuntimeError("YOLO detector is not initialized")

    roi_bbox = detector.detect(roi_frame)

    if roi_bbox is None:
        return None

    # Convert ROI-relative bbox to original frame coordinates
    rx1, ry1, rx2, ry2 = roi_bbox
    ox1, oy1, _, _ = roi_coords

    frame_bbox = (
        int(rx1 + ox1),
        int(ry1 + oy1),
        int(rx2 + ox1),
        int(ry2 + oy1),
    )

    return frame_bbox


def get_image_size_from_shape(shape: np.ndarray) -> Tuple[int, int]:
    """Extract HxW from a TFLite input shape."""
    dims = tuple(int(v) for v in shape)

    if len(dims) == 4:
        if dims[-1] == 3:
            return dims[1], dims[2]
        if dims[1] == 3:
            return dims[2], dims[3]
    elif len(dims) == 3:
        if dims[-1] == 3:
            return dims[0], dims[1]
        if dims[0] == 3:
            return dims[1], dims[2]

    raise ValueError(f"Unsupported classifier input shape: {dims}")


def _extract_quant_params(detail: dict) -> Tuple[float, int]:
    """Extract quantization scale and zero_point from TFLite detail dict.

    Supports both formats:
    - Legacy: detail["quantization"] = (scale, zero_point)  # tuple
    - Modern: detail["quantization_parameters"] = {"scales": [...], "zero_points": [...]}  # dict
    """
    # Try modern format first (tflite-runtime >= 2.5)
    qparams = detail.get("quantization_parameters", {})
    if isinstance(qparams, dict):
        scales = qparams.get("scales", [])
        zero_points = qparams.get("zero_points", [])
        if len(scales) > 0 and float(scales[0]) != 0.0:
            return float(scales[0]), int(zero_points[0]) if len(zero_points) > 0 else 0

    # Fallback to legacy format
    legacy = detail.get("quantization", (0.0, 0))
    if isinstance(legacy, (list, tuple)) and len(legacy) >= 2:
        return float(legacy[0]), int(legacy[1])

    return 0.0, 0


def quantize_input(input_data: np.ndarray, input_detail: dict) -> np.ndarray:
    """Convert float32 [0,1] image data into the classifier's expected dtype."""
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        return input_data.astype(np.float32, copy=False)

    if np.issubdtype(dtype, np.integer):
        scale, zero_point = _extract_quant_params(input_detail)
        if not scale:
            raise ValueError("Quantized classifier input is missing quantization scale")
        quantized = np.round(input_data / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)

    return input_data.astype(dtype, copy=False)


def dequantize_output(output_data: np.ndarray, output_detail: dict) -> np.ndarray:
    """Convert quantized classifier outputs back to float for scoring/logging."""
    if np.issubdtype(output_data.dtype, np.integer):
        scale, zero_point = _extract_quant_params(output_detail)
        if scale:
            return (output_data.astype(np.float32) - zero_point) * scale
    return output_data.astype(np.float32, copy=False)


def preprocess_classifier(crop: np.ndarray) -> np.ndarray:
    """Preprocess cropped image for the classifier based on model metadata."""
    interpreter = g_state.classifier_interpreter
    if interpreter is None:
        raise RuntimeError("Classifier is not initialized")

    input_detail = interpreter.get_input_details()[0]
    input_h, input_w = get_image_size_from_shape(input_detail["shape"])

    resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    return quantize_input(input_data, input_detail)


def run_classifier(crop: np.ndarray) -> Tuple[str, float]:
    """Run classifier on a cropped waste image."""
    interpreter = g_state.classifier_interpreter
    if interpreter is None:
        raise RuntimeError("Classifier is not initialized")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    input_data = preprocess_classifier(crop)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Get predictions
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    output_data = dequantize_output(output_data, output_details[0])

    # Get top class
    pred_idx = int(np.argmax(output_data))
    confidence = float(output_data[pred_idx])
    class_name = (
        Config.CLASS_NAMES[pred_idx]
        if pred_idx < len(Config.CLASS_NAMES)
        else "reject"
    )

    return class_name, confidence


def classify_single_frame(frame: np.ndarray) -> Tuple[str, float]:
    """Full pipeline: detect + crop + classify one frame."""
    # 1. Detect object
    bbox = run_yolo_detection(frame)
    if bbox is None:
        return "reject", 0.0  # No object detected

    x1, y1, x2, y2 = bbox

    # 2. Crop
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "reject", 0.0

    # 3. Classify
    return run_classifier(crop)


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
            Config.SERVO1_PIN: Config.CENTER_ANGLE,
            Config.SERVO2_PIN: Config.CENTER_ANGLE,
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
        steps: int = 20,
        delay: float = 0.02,
    ) -> None:
        """Linearly interpolate servo from current → target."""
        start = self._last_angle.get(pin, Config.CENTER_ANGLE)
        for i in range(steps + 1):
            current = start + (target - start) * (i / steps)
            self.move_servo(pin, current)
            time.sleep(delay)

    def idle_servos(self) -> None:
        """Stop PWM signal to reduce servo heat and prevent jitter.

        Uses tx_pwm(pin, 0, 0) to cancel the PWM waveform entirely.
        tx_servo(pin, 0) is documented to stop servos but some lgpio
        versions reject pulseWidth=0 as 'bad PWM micros'.
        """
        for pin in (Config.SERVO1_PIN, Config.SERVO2_PIN):
            try:
                lgpio.tx_pwm(self._h, pin, 0, 0)   # freq=0, duty=0 → stop
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
        self._hw.move_servo(Config.SERVO1_PIN, Config.CENTER_ANGLE)
        self._hw.move_servo(Config.SERVO2_PIN, Config.CENTER_ANGLE)
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

        try:
            while self._running:
                frame = cam.get_frame()

                # ── distance readings ────────────────────────────────
                dist_main = hw.read_distance(Config.TRIG_PIN, Config.ECHO_PIN)
                full_bin  = hw.check_bins_full()

                # ── bin distances (all 4) ─────────────────────────────
                bin_dists = [
                    hw.read_distance(Config.BIN_TRIG_PINS[i], Config.BIN_ECHO_PINS[i])
                    for i in range(len(Config.BIN_TRIG_PINS))
                ]

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
                        state = DetectionState.HAND_PRESENT
                        logger.info("Hand detected (dist=%.1f cm)", dist_main)

                elif state == DetectionState.HAND_PRESENT:
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
            angle = int(cmd.get("angle", Config.CENTER_ANGLE))
            if name == "capture":
                hw.move_servo(Config.SERVO1_PIN, angle)
            elif name == "sort":
                hw.move_servo(Config.SERVO2_PIN, angle)
            elif name == "all":
                hw.move_servo(Config.SERVO1_PIN, angle)
                hw.move_servo(Config.SERVO2_PIN, angle)
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
        elif action == "update_config":
            if "detect_dist"    in cmd: Config.DETECT_DISTANCE_CM          = float(cmd["detect_dist"])
            if "withdraw_dist"  in cmd: Config.HAND_WITHDRAWN_DISTANCE_CM  = float(cmd["withdraw_dist"])
            if "cooldown"       in cmd: Config.COOLDOWN_SECONDS             = float(cmd["cooldown"])
            if "yolo_conf"      in cmd: Config.YOLO_CONF_THRESHOLD          = float(cmd["yolo_conf"])
        elif action == "manual_sort":
            cls = cmd.get("class", "reject")
            logger.info("Manual sort triggered: %s", cls)
            target = self._label_to_angle(cls)
            hw.smooth_move(Config.SERVO2_PIN, target, steps=10, delay=0.01)
            time.sleep(0.5)
            hw.smooth_move(Config.SERVO1_PIN, Config.SWEEP_ANGLE, steps=10, delay=0.01)
            time.sleep(0.5)
            hw.move_servo(Config.SERVO1_PIN, Config.CENTER_ANGLE)
            hw.move_servo(Config.SERVO2_PIN, Config.CENTER_ANGLE)
            time.sleep(0.2)
            hw.idle_servos()
            if system_state:
                system_state.add_sort_event(cls, 1.0, 0.0)

    def _process_detection(self) -> None:
        """Execute the full detection pipeline according to timing spec.

        Called from the main loop when hand withdrawal is detected.
        Blocks the sensor loop for ~1.5 s (by design).
        """
        t0 = time.monotonic()
        hw = self._hw
        assert hw

        # t=0.00 — Pre-emptive: tilt capture arm to photo position
        hw.move_servo(Config.SERVO1_PIN, Config.PHOTO_ANGLE)
        logger.info("[t=%.3f] Pre-emptive Servo1 → %d°",
                    time.monotonic() - t0, Config.PHOTO_ANGLE)

        # Flash ON
        self._send_flash(on=True)

        # t=0.50 — Grab 5 latest frames from rolling buffer
        time.sleep(max(0.0, 0.50 - (time.monotonic() - t0)))
        frames = self._cam.get_recent_frames(Config.VOTE_FRAMES)  # type: ignore[union-attr]
        logger.info("[t=%.3f] Grabbed %d frames", time.monotonic() - t0, len(frames))

        if not frames:
            logger.warning("No frames in buffer — aborting")
            hw.move_servo(Config.SERVO1_PIN, Config.CENTER_ANGLE)
            hw.idle_servos()
            self._send_flash(on=False)
            return

        # t~0.50–0.55 — Inference (sequential 5-frame vote)
        label, conf, elapsed_ms = classify_with_voting(frames)
        logger.info(
            "[t=%.3f] Result: %s (conf=%.2f, %.1f ms)",
            time.monotonic() - t0, label, conf, elapsed_ms,
        )

        self._send_flash(on=False)
        self._record_sort(label, conf, elapsed_ms)

        # Return Servo1 to center before bin rotation
        hw.smooth_move(Config.SERVO1_PIN, Config.CENTER_ANGLE, steps=10, delay=0.01)

        # t~0.65 — Sort command: Servo2 → target bin angle
        target_angle = self._label_to_angle(label)
        hw.smooth_move(Config.SERVO2_PIN, target_angle, steps=10, delay=0.01)
        logger.info(
            "[t=%.3f] Sort: Servo2 → %d° (%s)",
            time.monotonic() - t0, target_angle, label,
        )

        # t=1.00 — Drop: Servo1 tips waste
        time.sleep(max(0.0, 1.00 - (time.monotonic() - t0)))
        hw.smooth_move(Config.SERVO1_PIN, Config.SWEEP_ANGLE, steps=10, delay=0.01)
        logger.info(
            "[t=%.3f] Drop: Servo1 → %d°",
            time.monotonic() - t0, Config.SWEEP_ANGLE,
        )

        # t=1.50 — Reset: all servos home
        time.sleep(max(0.0, 1.50 - (time.monotonic() - t0)))
        hw.move_servo(Config.SERVO1_PIN, Config.CENTER_ANGLE)
        hw.move_servo(Config.SERVO2_PIN, Config.CENTER_ANGLE)
        time.sleep(0.2)
        hw.idle_servos()
        logger.info("[t=%.3f] Reset complete", time.monotonic() - t0)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _label_to_angle(label: str) -> int:
        """Map NCNN class name → Servo2 angle.

        Labels (from labels.txt): AluCan, Glass, HDPEM, PET
        """
        ll = label.lower()
        if "reject" in ll:
            return Config.CENTER_ANGLE   # 92° — stay home, drop in default bin
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