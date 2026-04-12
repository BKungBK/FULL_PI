"""
shared_state.py — SmartBin V2 Thread-Safe Shared State
=======================================================
Single module-level singleton `system_state` that all threads read/write.

Engine thread  → writes detection results, distances, sort counts
UI thread      → reads snapshots for display (no locks needed on read)
Server thread  → reads snapshots for WebSocket broadcast
Control cmds   → UI/Server → cmd_queue → Engine polls & executes
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Waste classes (fixed — matches EfficientNet output indices) ────────
CLASSES: Tuple[str, ...] = ("metal", "plastic", "glass", "reject")

# ── Sort event dict shape ──────────────────────────────────────────────
# {"time": "HH:MM:SS", "label": str, "confidence": float,
#  "correct": Optional[bool], "actual": Optional[str]}


@dataclass
class _SystemState:
    """Internal mutable state — do NOT access directly; use SharedState methods."""

    # ── Detection ──────────────────────────────────────────────────────
    detection_state: str = "idle"      # idle | hand_present | processing | cooldown
    last_label: str = ""
    last_confidence: float = 0.0
    last_inference_ms: float = 0.0
    last_sort_time: str = ""
    last_predicted: str = ""           # used by confirm-sort UI

    # ── Live sensor ────────────────────────────────────────────────────
    main_distance_cm: float = 999.0
    bin_distances_cm: List[float] = field(default_factory=lambda: [999.0] * 4)
    esp32_ip: str = ""
    camera_connected: bool = False

    # ── Sort statistics ────────────────────────────────────────────────
    sort_counts: Dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in CLASSES}
    )
    total_sorted: int = 0
    sort_history: List[dict] = field(default_factory=list)   # capped at 100

    # ── Performance evaluation ─────────────────────────────────────────
    # confusion_matrix[actual_class][predicted_class] = count
    confusion_matrix: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {c: {p: 0 for p in CLASSES} for c in CLASSES}
    )
    sort_times_ms: List[float] = field(default_factory=list)       # last 100
    inference_times_ms: List[float] = field(default_factory=list)  # last 100

    # ── System ─────────────────────────────────────────────────────────
    uptime_start: float = field(default_factory=time.time)
    engine_running: bool = False
    mode: str = "auto"   # "auto" | "manual"


class SharedState:
    """
    Thread-safe facade over _SystemState.

    Reading  → call snapshot() which returns a plain dict copy; no lock
               held during your read code.
    Writing  → use update(), add_sort_event(), confirm_sort().
    Commands → UI/Server push via send_command(); Engine polls get_command().
    """

    MAX_HISTORY = 100
    MAX_TIMES   = 100

    def __init__(self) -> None:
        self._lock   = threading.RLock()
        self._state  = _SystemState()
        self._cmd_q: "queue.Queue[dict]" = __import__("queue").Queue()
        self._sort_cbs: List[Callable] = []   # callbacks for WS broadcast

    # ──────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a plain-dict copy of state (safe to read without lock)."""
        with self._lock:
            s = self._state
            return {
                "detection_state":    s.detection_state,
                "last_label":         s.last_label,
                "last_confidence":    s.last_confidence,
                "last_inference_ms":  s.last_inference_ms,
                "last_sort_time":     s.last_sort_time,
                "last_predicted":     s.last_predicted,
                "main_distance_cm":   s.main_distance_cm,
                "bin_distances_cm":   list(s.bin_distances_cm),
                "esp32_ip":           s.esp32_ip,
                "camera_connected":   s.camera_connected,
                "sort_counts":        dict(s.sort_counts),
                "total_sorted":       s.total_sorted,
                "sort_history":       list(s.sort_history[-10:]),
                "confusion_matrix":   {k: dict(v) for k, v in s.confusion_matrix.items()},
                "sort_times_ms":      list(s.sort_times_ms[-50:]),
                "inference_times_ms": list(s.inference_times_ms[-50:]),
                "uptime_start":       s.uptime_start,
                "engine_running":     s.engine_running,
                "mode":               s.mode,
            }

    def get_stats_payload(self) -> dict:
        """Compact payload for WebSocket 'stats' message."""
        with self._lock:
            s = self._state
            return {
                "total_sorted": s.total_sorted,
                "by_class":     dict(s.sort_counts),
                "mode":         s.mode,
            }

    # ──────────────────────────────────────────────────────────────────
    # WRITE — generic
    # ──────────────────────────────────────────────────────────────────

    def update(self, **kwargs) -> None:
        """Update one or more fields by name."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)

    def reset_stats(self) -> None:
        """Clear sort counts and history (keep confusion matrix)."""
        with self._lock:
            s = self._state
            s.sort_counts = {c: 0 for c in CLASSES}
            s.total_sorted = 0
            s.sort_history.clear()

    def reset_performance(self) -> None:
        """Clear confusion matrix and timing history."""
        with self._lock:
            s = self._state
            s.confusion_matrix = {c: {p: 0 for p in CLASSES} for c in CLASSES}
            s.sort_times_ms.clear()
            s.inference_times_ms.clear()

    # ──────────────────────────────────────────────────────────────────
    # WRITE — sort events
    # ──────────────────────────────────────────────────────────────────

    def add_sort_event(
        self,
        label: str,
        confidence: float,
        inference_ms: float,
    ) -> None:
        """Called by the engine when a sort completes."""
        with self._lock:
            s = self._state
            s.last_label       = label
            s.last_confidence  = confidence
            s.last_inference_ms = inference_ms
            s.last_predicted   = label
            s.last_sort_time   = time.strftime("%H:%M:%S")

            s.sort_counts[label] = s.sort_counts.get(label, 0) + 1
            s.total_sorted      += 1

            for lst in (s.sort_times_ms, s.inference_times_ms):
                lst.append(inference_ms)
                if len(lst) > self.MAX_TIMES:
                    lst.pop(0)

            event: dict = {
                "time":       s.last_sort_time,
                "label":      label,
                "confidence": round(confidence, 3),
                "correct":    None,
                "actual":     None,
            }
            s.sort_history.append(event)
            if len(s.sort_history) > self.MAX_HISTORY:
                s.sort_history.pop(0)

            snap = self.snapshot()   # copy *before* releasing lock

        # Fire callbacks outside the lock to avoid re-entrancy deadlocks
        for cb in self._sort_cbs:
            try:
                cb(snap)
            except Exception:
                pass

    def confirm_sort(self, predicted: str, actual: str) -> None:
        """
        Record user confirmation of the last sort result.
        Updates confusion_matrix[actual][predicted].
        """
        with self._lock:
            s = self._state
            cm = s.confusion_matrix
            if actual in cm and predicted in cm[actual]:
                cm[actual][predicted] += 1

            # Annotate the most recent un-confirmed history entry
            for event in reversed(s.sort_history):
                if event.get("correct") is None:
                    event["correct"] = (predicted == actual)
                    event["actual"]  = actual
                    break

    # ──────────────────────────────────────────────────────────────────
    # COMMAND QUEUE (UI/Server → Engine)
    # ──────────────────────────────────────────────────────────────────

    def send_command(self, cmd: dict) -> None:
        """Push a control command; engine polls with get_command()."""
        self._cmd_q.put(cmd)

    def get_command(self) -> Optional[dict]:
        """Non-blocking poll; returns None if queue empty."""
        try:
            return self._cmd_q.get_nowait()
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # CALLBACKS (WebSocket broadcast)
    # ──────────────────────────────────────────────────────────────────

    def register_sort_callback(self, cb: Callable) -> None:
        """Register a function called with snapshot dict on every sort."""
        self._sort_cbs.append(cb)

    # ──────────────────────────────────────────────────────────────────
    # CSV EXPORT
    # ──────────────────────────────────────────────────────────────────

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export sort_history to CSV. Returns the file path written."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "sort_history.csv")
        with self._lock:
            history = list(self._state.sort_history)

        fieldnames = ["time", "label", "confidence", "correct", "actual"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
        return path


# ── Module-level singleton ─────────────────────────────────────────────
system_state = SharedState()
