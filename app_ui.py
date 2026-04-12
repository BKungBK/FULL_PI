"""
app_ui.py — SmartBin V2 Desktop UI (CustomTkinter)
===================================================
Dark emerald dashboard — 4 tabs:
  1. Dashboard  — live camera + state badge + sort stats + bin levels + activity log
  2. Control    — engine start/stop, mode toggle, servo control, manual sort
  3. ประเมินผล — KPIs, per-class table, 4-chart matplotlib panel, confirm strip
  4. Settings   — IP, ROI, thresholds, WebSocket server toggle, data management

Threading model (zero-lag):
  Main thread     → CustomTkinter event loop, .after() only
  CameraThread    → decode MJPEG → queue.Queue(maxsize=2) → main reads every 33ms
  EngineThread    → SmartBinEngine.run() (GPIO / servo — blocks OK, separate thread)
  ServerThread    → uvicorn FastAPI (started from Settings tab)

Usage:
    python app_ui.py
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from shared_state import CLASSES, system_state

# =====================================================================
# 1. DESIGN SYSTEM
# =====================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

BG_0    = "#0b0f0e"
BG_1    = "#111916"
BG_2    = "#161f1b"
BG_3    = "#1c2823"
EMERALD = "#34d399"
AMBER   = "#fbbf24"
ROSE    = "#fb7185"
VIOLET  = "#a78bfa"
SKY     = "#38bdf8"
TEXT_0  = "#f0fdf4"
TEXT_1  = "#bbf7d0"
TEXT_DIM = "#3f6f5c"

CLASS_COLORS = {
    "metal":   "#60a5fa",
    "plastic": "#fbbf24",
    "glass":   "#a78bfa",
    "reject":  "#fb7185",
}
# Dark muted backgrounds per class (Tkinter-safe 6-digit hex — no alpha)
CLASS_BG = {
    "metal":   "#0f2035",   # dark navy blue
    "plastic": "#2a1f06",   # dark amber
    "glass":   "#1a1530",   # dark violet
    "reject":  "#2d1520",   # dark rose
}
STATE_COLORS = {
    "idle":         EMERALD,
    "hand_present": SKY,
    "processing":   AMBER,
    "cooldown":     "#6b7280",
}

FONT_BODY  = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 11)
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_NUM   = ("Consolas", 28, "bold")

# matplotlib dark palette
MPL_BG     = "#111916"
MPL_CARD   = "#161f1b"
MPL_GRID   = "#1c2823"
MPL_TICK   = "#6ee7b7"


# =====================================================================
# 2. BACKGROUND CAMERA READER
# =====================================================================
class CameraThread(threading.Thread):
    """Decode MJPEG from ESP32-CAM; drop old frames to stay current."""

    def __init__(self, url: str, frame_q: "queue.Queue[Image.Image]") -> None:
        super().__init__(daemon=True, name="CameraThread")
        self._url     = url
        self._q       = frame_q
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            cap: Optional[cv2.VideoCapture] = None
            try:
                cap = cv2.VideoCapture(self._url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                system_state.update(camera_connected=True)

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = self._overlay(frame)
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil   = Image.fromarray(rgb)
                    # Non-blocking put — drop oldest if full
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self._q.put_nowait(pil)
                    except queue.Full:
                        pass
            except Exception:
                pass
            finally:
                if cap:
                    cap.release()
                system_state.update(camera_connected=False)
            if self._running:
                time.sleep(2)

    @staticmethod
    def _overlay(frame: np.ndarray) -> np.ndarray:
        snap  = system_state.snapshot()
        state = snap.get("detection_state", "idle")
        dist  = snap.get("main_distance_cm", 999)
        label = snap.get("last_label", "")

        cv_colors = {
            "idle":         (52,  211, 153),
            "hand_present": (56,  189, 248),
            "processing":   (251, 191,  36),
            "cooldown":     (100, 100, 100),
        }
        color = cv_colors.get(state, (200, 200, 200))

        # ROI box (default center 320×240, size 320)
        cx, cy, size = 320, 240, 320
        half = size // 2
        h, w = frame.shape[:2]
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

        txt = f"{state.upper().replace('_',' ')}  {dist:.1f}cm"
        cv2.putText(frame, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label and state == "processing":
            cv2.putText(frame, f">>> {label.upper()} <<<", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 211, 153), 2)
        return frame


# =====================================================================
# 3. MAIN APPLICATION
# =====================================================================
class SmartBinApp(ctk.CTk):
    WIN_W = 1300
    WIN_H = 840
    SIDEBAR_W   = 210
    STATE_MS    = 120   # state poll interval
    CAM_MS      = 33    # camera refresh  (~30 fps)
    GRAPH_MS    = 3000  # graph refresh

    def __init__(self) -> None:
        super().__init__()
        self.title("SmartBin V2 — Pi 5 Dashboard")
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.minsize(960, 640)
        self.configure(fg_color=BG_0)

        # Internal state
        self._active_tab   = "dashboard"
        self._cam_thread:   Optional[CameraThread] = None
        self._frame_q:      queue.Queue = queue.Queue(maxsize=2)
        self._cam_img_ref   = None          # prevent GC
        self._engine_thread: Optional[threading.Thread] = None
        self._server_running = False
        self._last_snap:    dict = {}
        self._prev_hist_len = 0

        # Build
        self._build_sidebar()
        self._build_content()
        self._build_dashboard()
        self._build_control()
        self._build_evaluation()
        self._build_settings()

        self._show_tab("dashboard")
        self._start_camera()

        self.after(self.STATE_MS,  self._loop_state)
        self.after(self.CAM_MS,    self._loop_camera)
        self.after(self.GRAPH_MS,  self._loop_graphs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────
    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=self.SIDEBAR_W, fg_color=BG_1, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Brand
        b = ctk.CTkFrame(sb, fg_color="transparent")
        b.pack(fill="x", padx=16, pady=(24, 4))
        ctk.CTkLabel(b, text="SB", font=("Consolas", 13, "bold"),
                     fg_color=EMERALD, text_color=BG_0,
                     width=36, height=36, corner_radius=8).pack(side="left")
        m = ctk.CTkFrame(b, fg_color="transparent")
        m.pack(side="left", padx=10)
        ctk.CTkLabel(m, text="SmartBin", font=("Segoe UI", 14, "bold"),
                     text_color=TEXT_0).pack(anchor="w")
        ctk.CTkLabel(m, text="V2 · Pi 5", font=("Consolas", 9),
                     text_color=TEXT_DIM).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=BG_3).pack(fill="x", padx=16, pady=14)

        self._nav_btns: dict[str, ctk.CTkButton] = {}
        nav = [("dashboard", "📊", "Dashboard"),
               ("control",   "🎛", "Control Panel"),
               ("evaluation","📈", "ประเมินผล"),
               ("settings",  "⚙",  "Settings")]
        for tid, icon, label in nav:
            btn = ctk.CTkButton(sb, text=f"  {icon}  {label}", anchor="w",
                                font=("Segoe UI", 12), fg_color="transparent",
                                text_color=TEXT_1, hover_color=BG_3,
                                corner_radius=8, height=40,
                                command=lambda t=tid: self._show_tab(t))
            btn.pack(fill="x", padx=8, pady=2)
            self._nav_btns[tid] = btn

        footer = ctk.CTkFrame(sb, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=16, pady=16)
        ctk.CTkFrame(footer, height=1, fg_color=BG_3).pack(fill="x", pady=(0, 10))
        self._status_dot  = ctk.CTkLabel(footer, text="●  System Online",
                                          font=("Consolas", 10), text_color=EMERALD)
        self._status_dot.pack(anchor="w")
        self._uptime_lbl  = ctk.CTkLabel(footer, text="00:00:00 up",
                                          font=("Consolas", 9), text_color=TEXT_DIM)
        self._uptime_lbl.pack(anchor="w")

    def _show_tab(self, tid: str) -> None:
        self._active_tab = tid
        for t, btn in self._nav_btns.items():
            if t == tid:
                btn.configure(fg_color=BG_3, text_color=EMERALD,
                              font=("Segoe UI", 12, "bold"))
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_1,
                              font=("Segoe UI", 12))
        for t, fr in self._tab_frames.items():
            if t == tid:
                fr.pack(fill="both", expand=True)
            else:
                fr.pack_forget()
        if tid == "evaluation":
            self._refresh_graphs()

    # ─────────────────────────────────────────────────────────────────
    # CONTENT AREA
    # ─────────────────────────────────────────────────────────────────
    def _build_content(self) -> None:
        self._content     = ctk.CTkFrame(self, fg_color=BG_0, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True)
        self._tab_frames: dict[str, ctk.CTkScrollableFrame] = {}

    def _tab(self, tid: str) -> ctk.CTkScrollableFrame:
        fr = ctk.CTkScrollableFrame(self._content, fg_color=BG_0,
                                    scrollbar_button_color=BG_3)
        self._tab_frames[tid] = fr
        return fr

    # ─────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _header(self, parent, title: str, sub: str) -> None:
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=24, pady=(24, 16))
        ctk.CTkLabel(f, text=title, font=FONT_TITLE, text_color=TEXT_0).pack(anchor="w")
        ctk.CTkLabel(f, text=sub, font=FONT_BODY, text_color=TEXT_DIM).pack(anchor="w")

    def _card(self, parent, title: str = "", badge: str = "",
              full: bool = True, col: int = 0) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(parent, fg_color=BG_1, corner_radius=12,
                             border_width=1, border_color=BG_3)
        if full:
            outer.pack(fill="x", padx=24, pady=6)
        else:
            outer.grid(row=0, column=col, sticky="nsew", padx=6)

        if title:
            h = ctk.CTkFrame(outer, fg_color="transparent")
            h.pack(fill="x", padx=16, pady=(12, 6))
            ctk.CTkLabel(h, text=title, font=("Segoe UI", 12, "bold"),
                         text_color=TEXT_1).pack(side="left")
            if badge:
                ctk.CTkLabel(h, text=badge, font=("Consolas", 9),
                             fg_color=BG_3, text_color=TEXT_DIM,
                             corner_radius=4, padx=8, pady=2).pack(side="right")

        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        return inner

    # ─────────────────────────────────────────────────────────────────
    # TAB 1 — DASHBOARD
    # ─────────────────────────────────────────────────────────────────
    def _build_dashboard(self) -> None:
        f = self._tab("dashboard")
        self._header(f, "Dashboard", "Real-time waste classification monitor")

        # Top grid
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.pack(fill="x", padx=24, pady=(0, 8))
        grid.columnconfigure(0, weight=3)
        grid.columnconfigure(1, weight=2)

        # Camera card
        cam_outer = ctk.CTkFrame(grid, fg_color=BG_1, corner_radius=12,
                                 border_width=1, border_color=BG_3)
        cam_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        cam_inner = ctk.CTkFrame(cam_outer, fg_color="transparent")
        cam_inner.pack(fill="both", expand=True, padx=16, pady=14)

        h_row = ctk.CTkFrame(cam_inner, fg_color="transparent")
        h_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(h_row, text="Live Stream", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_1).pack(side="left")
        ctk.CTkLabel(h_row, text="ESP32-CAM", font=("Consolas", 9),
                     fg_color=BG_3, text_color=TEXT_DIM,
                     corner_radius=4, padx=8, pady=2).pack(side="right")

        self._state_badge = ctk.CTkLabel(cam_inner,
                                          text="⬤  IDLE  |  --- cm",
                                          font=("Consolas", 11, "bold"),
                                          text_color=EMERALD, fg_color="transparent")
        self._state_badge.pack(anchor="w", pady=(0, 6))

        cam_box = ctk.CTkFrame(cam_inner, fg_color=BG_0, corner_radius=8)
        cam_box.pack(fill="x")
        self._cam_label = ctk.CTkLabel(cam_box, text="กำลังเชื่อมต่อกล้อง…",
                                        text_color=TEXT_DIM, font=("Segoe UI", 13),
                                        fg_color=BG_0, width=480, height=360)
        self._cam_label.pack(padx=2, pady=2)

        # Right column: stats + bins
        right = ctk.CTkFrame(grid, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        # Classification counts
        sc = self._card(right, "Classification Stats", full=False, col=0)
        right.columnconfigure(0, weight=1)
        self._count_lbls: dict[str, ctk.CTkLabel] = {}
        self._bar_fills:  dict[str, ctk.CTkFrame]  = {}
        self._bar_bgs:    dict[str, ctk.CTkFrame]  = {}
        for cls in CLASSES:
            color = CLASS_COLORS[cls]
            tile = ctk.CTkFrame(sc, fg_color=BG_2, corner_radius=8)
            tile.pack(fill="x", pady=3)
            inner = ctk.CTkFrame(tile, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(inner, text=cls.upper(), font=("Consolas", 9, "bold"),
                         text_color=TEXT_DIM).pack(side="left")
            lbl = ctk.CTkLabel(inner, text="0", font=("Consolas", 20, "bold"),
                               text_color=color)
            lbl.pack(side="right")
            self._count_lbls[cls] = lbl
            bar_bg = ctk.CTkFrame(tile, fg_color=BG_0, height=3, corner_radius=2)
            bar_bg.pack(fill="x", padx=12, pady=(0, 8))
            bar_bg.pack_propagate(False)
            bar_fill = ctk.CTkFrame(bar_bg, fg_color=color, height=3,
                                    corner_radius=2, width=0)
            bar_fill.pack(side="left", fill="y")
            self._bar_fills[cls] = bar_fill
            self._bar_bgs[cls]   = bar_bg

        # Bin levels
        bn = self._card(right, "Bin Level Monitor", full=False, col=0)
        bin_g = ctk.CTkFrame(bn, fg_color="transparent")
        bin_g.pack(fill="x")
        self._bin_lbls: dict[str, ctk.CTkLabel] = {}
        for i, cls in enumerate(CLASSES):
            bin_g.columnconfigure(i, weight=1)
            cell = ctk.CTkFrame(bin_g, fg_color=BG_0, corner_radius=8,
                               border_width=1, border_color=BG_3)
            cell.grid(row=0, column=i, padx=3, pady=3, sticky="ew")
            ctk.CTkLabel(cell, text=cls.upper(), font=("Consolas", 8, "bold"),
                         text_color=TEXT_DIM).pack(pady=(8, 2))
            lbl = ctk.CTkLabel(cell, text="0%", font=("Consolas", 16, "bold"),
                               text_color=EMERALD)
            lbl.pack(pady=(0, 8))
            self._bin_lbls[cls] = lbl

        # Activity log
        log_card = self._card(f, "Activity Log", badge="Last 10")

        # Confirm strip (hidden)
        self._confirm_strip = ctk.CTkFrame(log_card, fg_color=BG_3, corner_radius=8)
        self._confirm_lbl   = ctk.CTkLabel(self._confirm_strip,
                                            text="ยืนยันผล: --",
                                            font=("Segoe UI", 12), text_color=TEXT_1)
        self._confirm_lbl.pack(side="left", padx=12, pady=8)
        cb_row = ctk.CTkFrame(self._confirm_strip, fg_color="transparent")
        cb_row.pack(side="right", padx=8)
        for cls in CLASSES:
            ctk.CTkButton(cb_row, text=cls.capitalize(), width=72, height=28,
                          font=("Consolas", 10), fg_color="transparent",
                          border_color=CLASS_COLORS[cls], border_width=1,
                          text_color=CLASS_COLORS[cls], hover_color=BG_2,
                          command=lambda c=cls: self._do_confirm(c)).pack(side="left", padx=2)

        self._log_scroll = ctk.CTkScrollableFrame(log_card, fg_color=BG_0,
                                                   height=200, corner_radius=8)
        self._log_scroll.pack(fill="x", pady=(8, 0))
        self._log_rows: list[ctk.CTkFrame] = []
        self._log_empty = ctk.CTkLabel(self._log_scroll,
                                        text="ยังไม่มีกิจกรรม — รอการคัดแยก",
                                        text_color=TEXT_DIM, font=("Segoe UI", 12))
        self._log_empty.pack(pady=32)

    def _add_log_row(self, event: dict) -> None:
        cls   = event.get("label", "reject")
        color = CLASS_COLORS.get(cls, TEXT_DIM)
        row   = ctk.CTkFrame(self._log_scroll, fg_color="transparent")
        row.pack(fill="x", pady=1)
        ctk.CTkLabel(row, text=event.get("time", ""), font=FONT_MONO,
                     text_color=TEXT_DIM, width=65).pack(side="left")
        ctk.CTkLabel(row, text=f"  {cls.upper()}  ", font=("Consolas", 10, "bold"),
                     fg_color=CLASS_BG.get(cls, BG_2), text_color=color,
                     corner_radius=4).pack(side="left", padx=6)
        ctk.CTkLabel(row, text=f"{event.get('confidence', 0):.2f}",
                     font=FONT_MONO, text_color=TEXT_DIM).pack(side="left")
        correct = event.get("correct")
        if correct is not None:
            mark  = "✓" if correct else "✗"
            mc    = EMERALD if correct else ROSE
            ctk.CTkLabel(row, text=mark, text_color=mc,
                         font=("Segoe UI", 12, "bold")).pack(side="right", padx=8)
        ctk.CTkFrame(row, height=1, fg_color=BG_3).pack(side="bottom", fill="x")
        self._log_rows.append(row)
        if len(self._log_rows) > 10:
            self._log_rows.pop(0).destroy()

    def _do_confirm(self, actual: str) -> None:
        predicted = self._last_snap.get("last_predicted", "")
        if not predicted:
            return
        system_state.confirm_sort(predicted, actual)
        self._confirm_strip.pack_forget()
        if self._active_tab == "evaluation":
            self._refresh_graphs()

    # ─────────────────────────────────────────────────────────────────
    # TAB 2 — CONTROL
    # ─────────────────────────────────────────────────────────────────
    def _build_control(self) -> None:
        f = self._tab("control")
        self._header(f, "Control Panel", "Manual servo & engine controls")

        # Engine
        ec = self._card(f, "Engine Control")
        er = ctk.CTkFrame(ec, fg_color="transparent")
        er.pack(fill="x")
        self._eng_lbl = ctk.CTkLabel(er, text="⬤  Stopped",
                                      font=("Consolas", 12, "bold"), text_color=ROSE)
        self._eng_lbl.pack(side="left", padx=(0, 16))
        ctk.CTkButton(er, text="▶  Start Engine", font=("Segoe UI", 12, "bold"),
                      fg_color=EMERALD, text_color=BG_0, hover_color="#4ade80",
                      width=160, height=38, command=self._start_engine).pack(side="left", padx=4)
        ctk.CTkButton(er, text="■  Stop", font=("Segoe UI", 12),
                      fg_color="transparent", text_color=ROSE,
                      border_color=ROSE, border_width=1, hover_color="#2d1a1a",
                      width=90, height=38, command=self._stop_engine).pack(side="left", padx=4)

        # Mode
        mc = self._card(f, "System Mode")
        mr = ctk.CTkFrame(mc, fg_color="transparent")
        mr.pack(fill="x")
        self._mode_seg = ctk.CTkSegmentedButton(mr, values=["AUTO", "MANUAL"],
                                                 command=self._on_mode,
                                                 selected_color=EMERALD,
                                                 selected_hover_color="#4ade80",
                                                 unselected_color=BG_3,
                                                 font=("Consolas", 11, "bold"),
                                                 text_color=TEXT_0)
        self._mode_seg.set("AUTO")
        self._mode_seg.pack(side="left")
        self._mode_badge = ctk.CTkLabel(mr, text="AUTO MODE",
                                         font=("Consolas", 11, "bold"), text_color=EMERALD)
        self._mode_badge.pack(side="right")

        # Servo grid
        sg = ctk.CTkFrame(f, fg_color="transparent")
        sg.pack(fill="x", padx=24, pady=(0, 8))
        sg.columnconfigure(0, weight=1)
        sg.columnconfigure(1, weight=1)

        # Servo Capture
        cap_c = self._card(sg, "Servo Capture (แขนรับ)", badge="GPIO 18", full=False, col=0)
        sr = ctk.CTkFrame(cap_c, fg_color="transparent")
        sr.pack(fill="x", pady=4)
        ctk.CTkLabel(sr, text="Angle", width=50, font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(side="left")
        self._cap_slider = ctk.CTkSlider(sr, from_=0, to=180,
                                          button_color=EMERALD, progress_color=EMERALD,
                                          command=self._on_cap_slide)
        self._cap_slider.set(92)
        self._cap_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._cap_val = ctk.CTkLabel(sr, text="92°", width=40,
                                      font=FONT_MONO, text_color=EMERALD)
        self._cap_val.pack(side="left")
        pr = ctk.CTkFrame(cap_c, fg_color="transparent")
        pr.pack(fill="x", pady=4)
        for lbl, val in [("Home 92°", 92), ("Ready 120°", 120), ("Drop 45°", 45)]:
            ctk.CTkButton(pr, text=lbl, width=88, height=30, font=("Segoe UI", 10),
                          fg_color="transparent", text_color=TEXT_1,
                          border_color=BG_3, border_width=1, hover_color=BG_3,
                          command=lambda v=val: self._set_cap(v)).pack(side="left", padx=3)
        ctk.CTkButton(cap_c, text="Apply", fg_color=EMERALD, text_color=BG_0,
                      hover_color="#4ade80", height=32,
                      command=lambda: self._cmd("servo", name="capture",
                                                angle=int(self._cap_slider.get()))
                      ).pack(fill="x", pady=(6, 0))

        # Servo Sort
        sort_c = self._card(sg, "Servo Sort (เลือกถัง)", badge="GPIO 19", full=False, col=1)
        sort_r = ctk.CTkFrame(sort_c, fg_color="transparent")
        sort_r.pack(fill="x", pady=4)
        for cls, ang in [("metal", 67), ("plastic", 112), ("glass", 157), ("reject", 92)]:
            col = CLASS_COLORS.get(cls, TEXT_DIM)
            ctk.CTkButton(sort_r, text=f"{cls.capitalize()}\n{ang}°",
                          width=80, height=50, font=("Segoe UI", 11),
                          fg_color="transparent", border_color=col,
                          border_width=1, text_color=col, hover_color=BG_2,
                          command=lambda a=ang: self._cmd("servo", name="sort", angle=a)
                          ).pack(side="left", padx=3)
        ctk.CTkButton(sort_c, text="Reset All → Home",
                      fg_color="transparent", text_color=ROSE,
                      border_color=ROSE, border_width=1,
                      hover_color="#2d1a1a", height=32,
                      command=lambda: self._cmd("servo", name="all", angle=92)
                      ).pack(fill="x", pady=(6, 0))

        # Manual sort
        man_c = self._card(f, "Manual Sort Trigger")
        mr2   = ctk.CTkFrame(man_c, fg_color="transparent")
        mr2.pack(fill="x")
        self._man_cls = ctk.CTkOptionMenu(mr2, values=list(CLASSES),
                                           font=("Segoe UI", 12),
                                           fg_color=BG_2, button_color=BG_3,
                                           width=160, height=38)
        self._man_cls.pack(side="left", padx=(0, 12))
        ctk.CTkButton(mr2, text="Sort Now", fg_color=EMERALD, text_color=BG_0,
                      hover_color="#4ade80", width=140, height=38,
                      font=("Segoe UI", 12, "bold"),
                      command=lambda: self._cmd("manual_sort",
                                                **{"class": self._man_cls.get()})
                      ).pack(side="left")

        # Flash
        fl_c = self._card(f, "Flash Control")
        fl_r = ctk.CTkFrame(fl_c, fg_color="transparent")
        fl_r.pack()
        ctk.CTkButton(fl_r, text="Flash ON", fg_color=AMBER, text_color=BG_0,
                      width=120, height=36,
                      command=lambda: self._cmd("flash", on=True)).pack(side="left", padx=6)
        ctk.CTkButton(fl_r, text="Flash OFF", fg_color="transparent",
                      text_color=AMBER, border_color=AMBER, border_width=1,
                      hover_color=BG_2, width=120, height=36,
                      command=lambda: self._cmd("flash", on=False)).pack(side="left", padx=6)

    # ─────────────────────────────────────────────────────────────────
    # TAB 3 — EVALUATION (with matplotlib graphs)
    # ─────────────────────────────────────────────────────────────────
    def _build_evaluation(self) -> None:
        f = self._tab("evaluation")

        # Header row w/ buttons
        hr = ctk.CTkFrame(f, fg_color="transparent")
        hr.pack(fill="x", padx=24, pady=(24, 0))
        ctk.CTkLabel(hr, text="ผลประเมินประสิทธิภาพ",
                     font=FONT_TITLE, text_color=TEXT_0).pack(side="left")
        for label, cmd, color, hover in [
            ("↻  Refresh",    self._refresh_graphs,  "transparent", BG_3),
            ("Export CSV",    self._export_csv,       "transparent", BG_3),
            ("Reset Data",    self._reset_perf,       "transparent", "#2d1a1a"),
        ]:
            tc = EMERALD if "Export" in label else ROSE if "Reset" in label else TEXT_1
            bc = ROSE if "Reset" in label else BG_3
            ctk.CTkButton(hr, text=label, fg_color=color, text_color=tc,
                          border_color=bc, border_width=1, hover_color=hover,
                          width=110, height=32, command=cmd).pack(side="right", padx=3)

        # KPI row
        kpi_r = ctk.CTkFrame(f, fg_color="transparent")
        kpi_r.pack(fill="x", padx=24, pady=16)
        self._kpi: dict[str, ctk.CTkLabel] = {}
        for i, (key, unit) in enumerate([("Overall Accuracy", "%"),
                                          ("Total Sorted",     ""),
                                          ("Avg Sort Time",    "ms"),
                                          ("Inference Time",   "ms")]):
            kc = ctk.CTkFrame(kpi_r, fg_color=BG_1, corner_radius=10,
                              border_width=1, border_color=BG_3)
            kc.pack(side="left", expand=True, fill="both", padx=6)
            ctk.CTkLabel(kc, text=key.upper(), font=("Consolas", 9, "bold"),
                         text_color=TEXT_DIM).pack(pady=(14, 4))
            lbl = ctk.CTkLabel(kc, text="-" + unit, font=FONT_NUM, text_color=TEXT_0)
            lbl.pack(pady=(0, 14))
            self._kpi[key] = lbl

        # Metrics table
        tc = self._card(f, "รายการประเมิน")
        cols = ["Class", "Accuracy", "Error", "Precision", "Recall", "F1", "Samples", "Result"]
        hdr = ctk.CTkFrame(tc, fg_color=BG_3, corner_radius=6)
        hdr.pack(fill="x", pady=(0, 4))
        for h in cols:
            ctk.CTkLabel(hdr, text=h, font=("Consolas", 9, "bold"),
                         text_color=TEXT_DIM).pack(side="left", expand=True, pady=6)
        self._tbl_rows: dict[str, dict[str, ctk.CTkLabel]] = {}
        for cls in CLASSES:
            row_f = ctk.CTkFrame(tc, fg_color="transparent")
            row_f.pack(fill="x", pady=2)
            ctk.CTkLabel(row_f, text="●", text_color=CLASS_COLORS[cls],
                         font=("Segoe UI", 12)).pack(side="left", padx=(4, 2))
            ctk.CTkLabel(row_f, text=cls.capitalize(), font=("Segoe UI", 12),
                         text_color=TEXT_1, width=68).pack(side="left")
            rl: dict[str, ctk.CTkLabel] = {}
            for col in ["acc", "err", "prec", "rec", "f1", "samples", "result"]:
                lbl = ctk.CTkLabel(row_f, text="-", font=FONT_MONO, text_color=TEXT_1)
                lbl.pack(side="left", expand=True)
                rl[col] = lbl
            ctk.CTkFrame(tc, height=1, fg_color=BG_3).pack(fill="x")
            self._tbl_rows[cls] = rl

        # ── Matplotlib graphs ─────────────────────────────────────────
        graph_outer = ctk.CTkFrame(f, fg_color=BG_1, corner_radius=12,
                                   border_width=1, border_color=BG_3)
        graph_outer.pack(fill="x", padx=24, pady=6)
        gh = ctk.CTkFrame(graph_outer, fg_color="transparent")
        gh.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(gh, text="Visual Analytics", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_1).pack(side="left")

        fig = Figure(figsize=(12, 7), facecolor=MPL_BG)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.1,
                            wspace=0.35, hspace=0.45)
        self._ax_bar   = fig.add_subplot(2, 2, 1, facecolor=MPL_CARD)
        self._ax_cm    = fig.add_subplot(2, 2, 2, facecolor=MPL_CARD)
        self._ax_time  = fig.add_subplot(2, 2, 3, facecolor=MPL_CARD)
        self._ax_pie   = fig.add_subplot(2, 2, 4, facecolor=MPL_CARD)
        for ax in (self._ax_bar, self._ax_cm, self._ax_time, self._ax_pie):
            ax.tick_params(colors=MPL_TICK, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(MPL_GRID)

        canvas = FigureCanvasTkAgg(fig, master=graph_outer)
        cw = canvas.get_tk_widget()
        cw.configure(bg=MPL_BG)
        cw.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._fig    = fig
        self._canvas = canvas
        self._draw_placeholder_graphs()

        # Confirm strip
        cf = ctk.CTkFrame(f, fg_color=BG_1, corner_radius=10,
                          border_width=1, border_color=BG_3)
        cf.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(cf, text="ยืนยันผลการคัดแยกล่าสุด:",
                     font=("Segoe UI", 12), text_color=TEXT_1).pack(side="left", padx=12, pady=10)
        for cls in CLASSES:
            ctk.CTkButton(cf, text=cls.capitalize(), width=90, height=32,
                          font=("Segoe UI", 11),
                          fg_color=CLASS_BG[cls],
                          text_color=CLASS_COLORS[cls],
                          border_color=CLASS_COLORS[cls], border_width=1,
                          hover_color=BG_2,
                          command=lambda c=cls: self._do_confirm(c)
                          ).pack(side="left", padx=4, pady=8)

    # ── Graph drawing helpers ─────────────────────────────────────────
    def _style_ax(self, ax) -> None:
        ax.set_facecolor(MPL_CARD)
        ax.tick_params(colors=MPL_TICK, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(MPL_GRID)

    def _draw_placeholder_graphs(self) -> None:
        for ax in (self._ax_bar, self._ax_cm, self._ax_time, self._ax_pie):
            ax.clear()
        self._ax_bar.set_title("Sort Distribution", color=TEXT_1, fontsize=10, pad=8)
        self._ax_bar.bar([c.capitalize() for c in CLASSES], [0]*4,
                         color=[CLASS_COLORS[c] for c in CLASSES], width=0.6)
        self._style_ax(self._ax_bar)
        self._ax_bar.set_ylabel("Count", color=MPL_TICK, fontsize=8)
        self._ax_bar.set_ylim(0, 5)

        self._draw_cm([[0]*4]*4)

        self._ax_time.set_title("Inference Time (ms)", color=TEXT_1, fontsize=10, pad=8)
        self._style_ax(self._ax_time)
        self._ax_time.set_xlabel("Event #", color=MPL_TICK, fontsize=8)
        self._ax_time.set_ylabel("ms", color=MPL_TICK, fontsize=8)

        self._ax_pie.set_title("Class Breakdown", color=TEXT_1, fontsize=10, pad=8)
        self._ax_pie.pie([1]*4, labels=[c.capitalize() for c in CLASSES],
                         colors=[CLASS_COLORS[c] for c in CLASSES],
                         textprops={"color": MPL_TICK, "fontsize": 7},
                         wedgeprops={"linewidth": 0.5, "edgecolor": MPL_BG},
                         startangle=90)
        self._style_ax(self._ax_pie)
        self._canvas.draw_idle()

    def _draw_cm(self, matrix: list[list[int]]) -> None:
        ax   = self._ax_cm
        data = np.array(matrix, dtype=float)
        sums = data.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        norm = data / sums
        ax.clear()
        ax.set_facecolor(MPL_CARD)
        ax.set_title("Confusion Matrix", color=TEXT_1, fontsize=10, pad=8)
        ax.imshow(norm, cmap="Greens", aspect="auto", vmin=0, vmax=1)
        labels = [c[:3].capitalize() for c in CLASSES]
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, color=MPL_TICK, fontsize=8)
        ax.set_yticklabels(labels, color=MPL_TICK, fontsize=8)
        ax.set_xlabel("Predicted", color=MPL_TICK, fontsize=8)
        ax.set_ylabel("Actual",    color=MPL_TICK, fontsize=8)
        for i in range(4):
            for j in range(4):
                tc = "black" if norm[i, j] > 0.5 else MPL_TICK
                ax.text(j, i, str(int(data[i, j])), ha="center", va="center",
                        color=tc, fontsize=8, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(MPL_GRID)

    def _refresh_graphs(self) -> None:
        snap   = self._last_snap or system_state.snapshot()
        counts = [snap["sort_counts"].get(c, 0) for c in CLASSES]
        total  = sum(counts)
        times  = snap.get("inference_times_ms", [])
        cm     = snap.get("confusion_matrix", {})

        # 1. Bar chart
        ax = self._ax_bar
        ax.clear()
        self._style_ax(ax)
        ax.set_title("Sort Distribution", color=TEXT_1, fontsize=10, pad=8)
        bars = ax.bar([c.capitalize() for c in CLASSES], counts,
                      color=[CLASS_COLORS[c] for c in CLASSES], width=0.6,
                      edgecolor=MPL_BG, linewidth=0.5)
        for bar, n in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05, str(n),
                    ha="center", va="bottom", color=MPL_TICK, fontsize=8)
        ax.set_ylim(0, max(max(counts, default=0) * 1.3, 5))
        ax.set_ylabel("Count", color=MPL_TICK, fontsize=8)

        # 2. Confusion matrix
        matrix = [[cm.get(r, {}).get(c, 0) for c in CLASSES] for r in CLASSES]
        self._draw_cm(matrix)

        # 3. Inference timeline
        ax = self._ax_time
        ax.clear()
        self._style_ax(ax)
        ax.set_title("Inference Time (ms)", color=TEXT_1, fontsize=10, pad=8)
        if times:
            ax.plot(times, color=EMERALD, linewidth=1.5, marker=".", markersize=3)
            avg = sum(times) / len(times)
            ax.axhline(y=avg, color=AMBER, linewidth=1, linestyle="--", alpha=0.8,
                       label=f"Avg {avg:.0f} ms")
            ax.legend(fontsize=7, labelcolor=MPL_TICK,
                      facecolor=MPL_GRID, edgecolor=MPL_GRID)
        ax.set_xlabel("Event #", color=MPL_TICK, fontsize=8)
        ax.set_ylabel("ms", color=MPL_TICK, fontsize=8)
        ax.fill_between(range(len(times)), times, alpha=0.15, color=EMERALD)

        # 4. Pie
        ax = self._ax_pie
        ax.clear()
        self._style_ax(ax)
        ax.set_title("Class Breakdown", color=TEXT_1, fontsize=10, pad=8)
        if total > 0:
            _, texts, autotexts = ax.pie(
                counts,
                labels=[f"{c.capitalize()} ({n})" for c, n in zip(CLASSES, counts)],
                colors=[CLASS_COLORS[c] for c in CLASSES],
                autopct="%1.0f%%", pctdistance=0.75,
                textprops={"color": MPL_TICK, "fontsize": 7},
                wedgeprops={"linewidth": 0.5, "edgecolor": MPL_BG}, startangle=90)
            for at in autotexts:
                at.set_color(BG_0); at.set_fontsize(7)
        else:
            ax.pie([1]*4, labels=[c.capitalize() for c in CLASSES],
                   colors=[CLASS_COLORS[c] for c in CLASSES],
                   textprops={"color": MPL_TICK, "fontsize": 7},
                   wedgeprops={"linewidth": 0.5, "edgecolor": MPL_BG}, startangle=90)

        self._canvas.draw_idle()
        self._update_metrics_table(snap)

    def _update_metrics_table(self, snap: dict) -> None:
        cm   = snap.get("confusion_matrix", {})
        totA = sum(sum(cm.get(r, {}).values()) for r in CLASSES)
        corr = sum(cm.get(c, {}).get(c, 0) for c in CLASSES)
        oa   = corr / totA if totA else None
        times = snap.get("inference_times_ms", [])
        avg_t = sum(times) / len(times) if times else 0

        def pct(v):
            return f"{v*100:.1f}%" if v is not None else "-"

        for cls in CLASSES:
            tp  = cm.get(cls, {}).get(cls, 0)
            tot_act  = sum(cm.get(cls, {}).values())
            tot_pred = sum(cm.get(r, {}).get(cls, 0) for r in CLASSES)
            fp  = tot_pred - tp
            fn  = tot_act  - tp
            tn  = totA - tp - fp - fn

            rec  = tp / tot_act  if tot_act   else None
            acc  = (tp + tn) / max(totA, 1)  if totA else None
            err  = 1 - acc if acc is not None else None
            prec = tp / tot_pred if tot_pred  else None
            f1   = (2 * prec * rec / (prec + rec)
                    if prec is not None and rec is not None and (prec + rec) > 0
                    else None)

            rl = self._tbl_rows.get(cls, {})
            if "acc"     in rl: rl["acc"].configure(text=pct(acc))
            if "err"     in rl:
                rl["err"].configure(text=pct(err),
                                    text_color=ROSE if (err or 0) > 0.2 else TEXT_1)
            if "prec"    in rl: rl["prec"].configure(text=pct(prec))
            if "rec"     in rl: rl["rec"].configure(text=pct(rec))
            if "f1"      in rl: rl["f1"].configure(text=pct(f1))
            if "samples" in rl: rl["samples"].configure(text=str(tot_act))
            if "result"  in rl:
                if tot_act == 0:
                    rl["result"].configure(text="ยังไม่มีข้อมูล", text_color=TEXT_DIM)
                elif f1 is not None and f1 >= 0.9:
                    rl["result"].configure(text="✓ ผ่าน", text_color=EMERALD)
                elif f1 is not None and f1 >= 0.75:
                    rl["result"].configure(text="⚠ ปานกลาง", text_color=AMBER)
                else:
                    rl["result"].configure(text="✗ ต้องปรับปรุง", text_color=ROSE)

        # KPIs
        total = snap.get("total_sorted", 0)
        last_t = times[-1] if times else None
        if "Overall Accuracy" in self._kpi:
            self._kpi["Overall Accuracy"].configure(
                text=(f"{oa*100:.1f}%" if oa is not None else "-"))
        if "Total Sorted"     in self._kpi:
            self._kpi["Total Sorted"].configure(text=str(total))
        if "Avg Sort Time"    in self._kpi:
            self._kpi["Avg Sort Time"].configure(
                text=(f"{avg_t:.0f} ms" if avg_t else "-"))
        if "Inference Time"   in self._kpi:
            self._kpi["Inference Time"].configure(
                text=(f"{last_t:.0f} ms" if last_t else "-"))

    # ─────────────────────────────────────────────────────────────────
    # TAB 4 — SETTINGS
    # ─────────────────────────────────────────────────────────────────
    def _build_settings(self) -> None:
        f = self._tab("settings")
        self._header(f, "Settings", "Connection & device configuration")

        # Connection status
        cst = self._card(f, "Connection Status")
        self._conn_lbl = ctk.CTkLabel(cst, text="⬤  Connecting…",
                                       font=("Consolas", 12), text_color=AMBER)
        self._conn_lbl.pack(anchor="w")

        # ESP32 IP
        ec = self._card(f, "ESP32-CAM")
        ctk.CTkLabel(ec, text="IP Address (blank = auto-scan)",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        ir = ctk.CTkFrame(ec, fg_color="transparent")
        ir.pack(fill="x")
        self._ip_entry = ctk.CTkEntry(ir, width=200, height=36,
                                       placeholder_text="10.42.0.177 or auto",
                                       font=FONT_MONO, fg_color=BG_2,
                                       border_color=BG_3, text_color=TEXT_0)
        self._ip_entry.insert(0, "10.42.0.177")
        self._ip_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(ir, text="Reconnect Camera",
                      fg_color=EMERALD, text_color=BG_0,
                      hover_color="#4ade80", height=36, width=160,
                      command=self._reconnect_cam).pack(side="left")

        # ROI
        roi_c = self._card(f, "ROI Configuration")
        self._roi_sl: dict[str, ctk.CTkSlider] = {}
        for lbl, key, mn, mx, dft in [("Center X", "cx", 0, 640, 320),
                                       ("Center Y", "cy", 0, 480, 240),
                                       ("Size",     "sz", 100, 480, 320)]:
            rr = ctk.CTkFrame(roi_c, fg_color="transparent")
            rr.pack(fill="x", pady=4)
            ctk.CTkLabel(rr, text=lbl, width=80, font=FONT_SMALL,
                         text_color=TEXT_DIM).pack(side="left")
            sl = ctk.CTkSlider(rr, from_=mn, to=mx,
                               button_color=EMERALD, progress_color=EMERALD)
            sl.set(dft)
            sl.pack(side="left", fill="x", expand=True, padx=8)
            vl = ctk.CTkLabel(rr, text=str(dft), font=FONT_MONO,
                              text_color=EMERALD, width=40)
            vl.pack(side="left")
            sl.configure(command=lambda v, l=vl: l.configure(text=str(int(v))))
            self._roi_sl[key] = sl
        ctk.CTkButton(roi_c, text="Apply ROI",
                      fg_color=EMERALD, text_color=BG_0,
                      hover_color="#4ade80", height=32, width=120,
                      command=self._apply_roi).pack(anchor="w", pady=(8, 0))

        # Thresholds
        thr_c = self._card(f, "Detection Thresholds")
        self._thresh: dict[str, ctk.CTkEntry] = {}
        for lbl, key, dft in [("Detect Distance (cm)",  "detect_dist",    "15.0"),
                               ("Withdraw Distance (cm)","withdraw_dist",   "20.0"),
                               ("Cooldown (s)",          "cooldown",        "5.0"),
                               ("YOLO Confidence",       "yolo_conf",       "0.40")]:
            tr = ctk.CTkFrame(thr_c, fg_color="transparent")
            tr.pack(fill="x", pady=4)
            ctk.CTkLabel(tr, text=lbl, width=200, font=FONT_SMALL,
                         text_color=TEXT_DIM).pack(side="left")
            e = ctk.CTkEntry(tr, width=100, height=32, font=FONT_MONO,
                             fg_color=BG_2, border_color=BG_3, text_color=TEXT_0)
            e.insert(0, dft)
            e.pack(side="left", padx=8)
            self._thresh[key] = e
        ctk.CTkButton(thr_c, text="Save Thresholds",
                      fg_color=EMERALD, text_color=BG_0,
                      hover_color="#4ade80", height=34, width=160,
                      command=self._save_thresh).pack(anchor="w", pady=(10, 0))

        # Server
        srv_c = self._card(f, "WebSocket Server (for ui.html)")
        srv_r = ctk.CTkFrame(srv_c, fg_color="transparent")
        srv_r.pack(fill="x")
        self._srv_lbl = ctk.CTkLabel(srv_r, text="⬤  Stopped",
                                      font=("Consolas", 12), text_color=ROSE)
        self._srv_lbl.pack(side="left", padx=(0, 16))
        self._srv_btn = ctk.CTkButton(srv_r, text="Start Server (port 8000)",
                                      fg_color=EMERALD, text_color=BG_0,
                                      hover_color="#4ade80", height=34, width=200,
                                      command=self._toggle_server)
        self._srv_btn.pack(side="left")
        ctk.CTkLabel(srv_c, text="เปิดแล้ว → ui.html เชื่อมต่อผ่าน ws://PI_IP:8000/ws",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w", pady=(8, 0))

        # Data
        dc = self._card(f, "Data Management")
        dr = ctk.CTkFrame(dc, fg_color="transparent")
        dr.pack(fill="x")
        for txt, cmd in [("Clear Statistics",       self._clear_stats),
                          ("Reset Performance Data", self._reset_perf)]:
            ctk.CTkButton(dr, text=txt, fg_color="transparent",
                          text_color=ROSE, border_color=ROSE, border_width=1,
                          hover_color="#2d1a1a", height=32, width=180,
                          command=cmd).pack(side="left", padx=4)

    # ─────────────────────────────────────────────────────────────────
    # PERIODIC LOOPS
    # ─────────────────────────────────────────────────────────────────
    def _loop_state(self) -> None:
        try:
            snap = system_state.snapshot()
            self._last_snap = snap
            self._update_dashboard(snap)
            self._update_uptime(snap)
            if self._active_tab == "evaluation":
                self._update_metrics_table(snap)
        except Exception:
            pass
        finally:
            self.after(self.STATE_MS, self._loop_state)

    def _loop_camera(self) -> None:
        try:
            pil = self._frame_q.get_nowait()
            pil = pil.resize((480, 360), Image.LANCZOS)
            img = ctk.CTkImage(pil, size=(480, 360))
            self._cam_label.configure(image=img, text="")
            self._cam_img_ref = img
        except queue.Empty:
            pass
        except Exception:
            pass
        finally:
            self.after(self.CAM_MS, self._loop_camera)

    def _loop_graphs(self) -> None:
        try:
            if self._active_tab == "evaluation":
                self._refresh_graphs()
        except Exception:
            pass
        finally:
            self.after(self.GRAPH_MS, self._loop_graphs)

    # ─────────────────────────────────────────────────────────────────
    # STATE → UI
    # ─────────────────────────────────────────────────────────────────
    def _update_dashboard(self, snap: dict) -> None:
        # State badge
        state = snap.get("detection_state", "idle")
        dist  = snap.get("main_distance_cm", 999)
        color = STATE_COLORS.get(state, TEXT_DIM)
        self._state_badge.configure(
            text=f"⬤  {state.upper().replace('_', ' ')}  |  {dist:.1f} cm",
            text_color=color)

        # Sort counts + bars
        counts = snap.get("sort_counts", {})
        total  = max(snap.get("total_sorted", 0), 1)
        for cls in CLASSES:
            n = counts.get(cls, 0)
            self._count_lbls[cls].configure(text=str(n))
            bg = self._bar_bgs[cls]
            w  = bg.winfo_width()
            if w > 10:
                self._bar_fills[cls].configure(width=int(w * n / total))

        # Bin levels
        dists = snap.get("bin_distances_cm", [999]*4)
        for i, cls in enumerate(CLASSES):
            d = dists[i] if i < len(dists) else 999
            if d >= 999:
                self._bin_lbls[cls].configure(text="N/A", text_color=TEXT_DIM)
            else:
                pct = max(0, min(100, int((50 - d) / 45 * 100)))
                self._bin_lbls[cls].configure(
                    text=f"{pct}%",
                    text_color=ROSE if pct > 80 else AMBER if pct > 50 else EMERALD)

        # Log
        hist = snap.get("sort_history", [])
        if len(hist) != self._prev_hist_len:
            self._prev_hist_len = len(hist)
            for r in self._log_rows:
                r.destroy()
            self._log_rows.clear()
            self._log_empty.pack_forget()
            for ev in hist[-10:]:
                self._add_log_row(ev)

        # Confirm strip
        predicted = snap.get("last_predicted", "")
        if predicted and hist and hist[-1].get("correct") is None:
            self._confirm_lbl.configure(text=f"ยืนยันผล: {predicted.upper()}")
            self._confirm_strip.pack(fill="x", pady=(0, 6))
        else:
            self._confirm_strip.pack_forget()

        # Engine status
        running = snap.get("engine_running", False)
        self._eng_lbl.configure(
            text="⬤  Running" if running else "⬤  Stopped",
            text_color=EMERALD if running else ROSE)

        # Connection
        cam_ok = snap.get("camera_connected", False)
        self._conn_lbl.configure(
            text=f"⬤  ESP32-CAM  {snap.get('esp32_ip', '---')}",
            text_color=EMERALD if cam_ok else AMBER)

    def _update_uptime(self, snap: dict) -> None:
        elapsed = int(time.time() - snap.get("uptime_start", time.time()))
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        self._uptime_lbl.configure(text=f"{h:02d}:{m:02d}:{s:02d} up")

    # ─────────────────────────────────────────────────────────────────
    # COMMANDS → ENGINE
    # ─────────────────────────────────────────────────────────────────
    def _cmd(self, action: str, **kwargs) -> None:
        system_state.send_command({"action": action, **kwargs})

    def _on_cap_slide(self, val: float) -> None:
        self._cap_val.configure(text=f"{int(val)}°")

    def _set_cap(self, val: int) -> None:
        self._cap_slider.set(val)
        self._cap_val.configure(text=f"{val}°")
        self._cmd("servo", name="capture", angle=val)

    def _on_mode(self, mode: str) -> None:
        system_state.update(mode=mode.lower())
        self._cmd("mode", mode=mode.lower())
        self._mode_badge.configure(
            text=f"{mode} MODE",
            text_color=EMERALD if mode == "AUTO" else AMBER)

    def _start_engine(self) -> None:
        if self._last_snap.get("engine_running"):
            return

        def _run():
            try:
                import main as eng
                import importlib
                importlib.reload(eng)
                system_state.update(engine_running=True)
                eng.main()
            except Exception as exc:
                print(f"[Engine] Error: {exc}")
            finally:
                system_state.update(engine_running=False)

        self._engine_thread = threading.Thread(target=_run, daemon=True,
                                               name="EngineThread")
        self._engine_thread.start()

    def _stop_engine(self) -> None:
        self._cmd("stop")
        system_state.update(engine_running=False)

    def _start_camera(self) -> None:
        snap = system_state.snapshot()
        ip   = snap.get("esp32_ip") or "10.42.0.177"
        url  = f"http://{ip}:81/stream"
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_thread.stop()
        self._cam_thread = CameraThread(url, self._frame_q)
        self._cam_thread.start()

    def _reconnect_cam(self) -> None:
        ip = self._ip_entry.get().strip()
        if ip:
            system_state.update(esp32_ip=ip)
        self._start_camera()

    def _apply_roi(self) -> None:
        self._cmd("set_roi",
                  cx=int(self._roi_sl["cx"].get()),
                  cy=int(self._roi_sl["cy"].get()),
                  size=int(self._roi_sl["sz"].get()))

    def _save_thresh(self) -> None:
        try:
            vals = {k: float(e.get()) for k, e in self._thresh.items()}
            self._cmd("update_config", **vals)
        except ValueError:
            pass

    def _toggle_server(self) -> None:
        try:
            import server as srv
            if not self._server_running:
                srv.start()
                self._server_running = True
                self._srv_lbl.configure(text="⬤  Running on :8000", text_color=EMERALD)
                self._srv_btn.configure(text="Stop Server")
            else:
                srv.stop()
                self._server_running = False
                self._srv_lbl.configure(text="⬤  Stopped", text_color=ROSE)
                self._srv_btn.configure(text="Start Server (port 8000)")
        except ImportError:
            self._srv_lbl.configure(text="server.py not found", text_color=AMBER)

    def _clear_stats(self) -> None:
        system_state.reset_stats()

    def _reset_perf(self) -> None:
        system_state.reset_performance()
        self._refresh_graphs()

    def _export_csv(self) -> None:
        path = system_state.export_csv()
        print(f"[UI] Exported → {path}")

    # ─────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        if self._cam_thread:
            self._cam_thread.stop()
        self.destroy()


# =====================================================================
# ENTRY POINT
# =====================================================================
def main() -> None:
    app = SmartBinApp()
    app.mainloop()


if __name__ == "__main__":
    main()
