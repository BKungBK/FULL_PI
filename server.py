"""
server.py — SmartBin V2  FastAPI + WebSocket Server
=====================================================
Entry point for the full system:
    python server.py        ← starts API + serves ui.html on :8000

Engine lifecycle managed via:
    POST /engine/start      → spawns SmartBinEngine in a background thread
    POST /engine/stop       → calls engine.shutdown()

Browser:
    http://PI_IP:8000       → ui.html (single-page dashboard)

Endpoints:
    GET  /                         → ui.html
    GET  /api/stream               → MJPEG proxy from ESP32-CAM
    GET  /api/system-status        → JSON snapshot
    GET  /api/full-snapshot        → extended snapshot (includes sort_history)
    GET  /api/bin-levels           → bin distances
    GET  /api/stats                → sort counts + totals
    POST /engine/start             → start SmartBinEngine thread
    POST /engine/stop              → stop SmartBinEngine thread
    POST /control/mode             → {"mode": "auto"|"manual"}
    POST /control/servo            → {"name": "capture"|"sort"|"all", "angle": int}
    POST /control/sort             → {"class": "metal"|...}
    POST /api/performance/confirm  → {"actual_class": "..."}
    POST /api/stats/reset
    POST /api/performance/reset
    WS   /ws                       → real-time push every 200 ms
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from shared_state import CLASSES, system_state

logger = logging.getLogger("SmartBin.Server")

# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="SmartBin V2 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_HTML = Path(__file__).parent / "ui.html"

# ── Active WebSocket clients ──────────────────────────────────────────
_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()

# =====================================================================
# Engine lifecycle  (thread-based — shares shared_state in-process)
# =====================================================================
_engine_ref   = None          # SmartBinEngine instance
_engine_thread: Optional[threading.Thread] = None
_engine_lock  = threading.Lock()


def _engine_running() -> bool:
    return _engine_thread is not None and _engine_thread.is_alive()


def _start_engine_thread() -> tuple[bool, str]:
    """Start SmartBinEngine in a background thread. Returns (ok, message)."""
    global _engine_ref, _engine_thread

    with _engine_lock:
        if _engine_running():
            return False, "Engine already running"

        try:
            import main as eng  # import once — no reload
        except ImportError as exc:
            return False, f"Cannot import main: {exc}"

        def _run() -> None:
            global _engine_ref
            try:
                engine = eng.SmartBinEngine()
                _engine_ref = engine
                engine.setup()
                engine.run()
            except Exception as exc:
                import traceback
                logger.error("Engine error:\n%s", traceback.format_exc())
            finally:
                _engine_ref = None
                system_state.update(engine_running=False)

        _engine_thread = threading.Thread(
            target=_run, daemon=True, name="EngineThread")
        _engine_thread.start()
        return True, "Engine starting…"


def _stop_engine_thread() -> tuple[bool, str]:
    """Stop the running engine gracefully."""
    global _engine_ref
    with _engine_lock:
        if not _engine_running():
            return False, "Engine not running"
        system_state.send_command({"action": "stop"})
        if _engine_ref is not None:
            try:
                _engine_ref.shutdown()
            except Exception as exc:
                logger.warning("Engine shutdown error: %s", exc)
            _engine_ref = None
        return True, "Engine stopping…"


# =====================================================================
# WebSocket broadcast helpers
# =====================================================================

async def _broadcast(payload: dict) -> None:
    """Send JSON to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    with _ws_lock:
        clients = list(_ws_clients)
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    with _ws_lock:
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


def _on_sort_event(snap: dict) -> None:
    """Callback registered with shared_state — fires on every sort."""
    payload = {
        "type": "sort_result",
        "label":      snap["last_label"],
        "confidence": snap["last_confidence"],
        "inference_ms": snap["last_inference_ms"],
        "sort_counts":  snap["sort_counts"],
        "total_sorted": snap["total_sorted"],
        "sort_history": snap["sort_history"],
    }
    if _server_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _server_loop)


system_state.register_sort_callback(_on_sort_event)

# =====================================================================
# REST endpoints — static
# =====================================================================


@app.get("/")
async def serve_ui() -> FileResponse:
    if _UI_HTML.exists():
        return FileResponse(str(_UI_HTML), media_type="text/html")
    return JSONResponse({"detail": "ui.html not found"}, status_code=404)


@app.get("/api/system-status")
async def get_system_status() -> JSONResponse:
    snap = system_state.snapshot()
    return JSONResponse({
        "success":         True,
        "mode":            snap["mode"],
        "engine_running":  snap["engine_running"],
        "esp32_ip":        snap["esp32_ip"],
        "detection_state": snap["detection_state"],
        "last_label":      snap["last_label"],
        "last_confidence": snap["last_confidence"],
        "last_inference_ms": snap["last_inference_ms"],
        "main_distance_cm":  snap["main_distance_cm"],
        "bin_distances_cm":  snap["bin_distances_cm"],
        "total_sorted":    snap["total_sorted"],
        "sort_counts":     snap["sort_counts"],
    })


@app.get("/api/full-snapshot")
async def get_full_snapshot() -> JSONResponse:
    """Extended snapshot used on initial page load."""
    return JSONResponse({"success": True, **system_state.snapshot()})


@app.get("/api/bin-levels")
async def get_bin_levels() -> JSONResponse:
    snap  = system_state.snapshot()
    dists = snap["bin_distances_cm"]
    return JSONResponse({
        "success": True,
        "levels": {
            "bin1": dists[0] if len(dists) > 0 else 999,
            "bin2": dists[1] if len(dists) > 1 else 999,
            "bin3": dists[2] if len(dists) > 2 else 999,
            "bin4": dists[3] if len(dists) > 3 else 999,
        },
    })


@app.get("/api/stats")
async def get_stats() -> JSONResponse:
    return JSONResponse({"success": True, **system_state.get_stats_payload()})


# =====================================================================
# REST endpoints — engine lifecycle
# =====================================================================


@app.post("/engine/start")
async def engine_start() -> JSONResponse:
    ok, msg = _start_engine_thread()
    return JSONResponse({"success": ok, "message": msg},
                        status_code=200 if ok else 409)


@app.post("/engine/stop")
async def engine_stop() -> JSONResponse:
    ok, msg = _stop_engine_thread()
    return JSONResponse({"success": ok, "message": msg},
                        status_code=200 if ok else 409)


@app.get("/engine/status")
async def engine_status() -> JSONResponse:
    return JSONResponse({
        "running": _engine_running(),
        "thread":  _engine_thread.name if _engine_thread else None,
    })


# =====================================================================
# REST endpoints — camera stream
# =====================================================================


@app.get("/api/stream")
async def mjpeg_stream() -> StreamingResponse:
    """Proxy the ESP32-CAM MJPEG stream to browser clients."""
    snap    = system_state.snapshot()
    esp_ip  = snap.get("esp32_ip") or "10.42.0.177"
    src_url = f"http://{esp_ip}:81/stream"

    async def _gen():
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                async with client.stream("GET", src_url) as resp:
                    async for chunk in resp.aiter_bytes(4096):
                        yield chunk
            except Exception as exc:
                logger.warning("MJPEG proxy error: %s", exc)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# =====================================================================
# REST endpoints — control
# =====================================================================


class ModeBody(BaseModel):
    mode: str


class ServoBody(BaseModel):
    name: str
    angle: object  # int or "home"


class ConfirmBody(BaseModel):
    actual_class: str


@app.post("/control/mode")
async def set_mode(body: ModeBody) -> JSONResponse:
    if body.mode not in ("auto", "manual"):
        return JSONResponse({"success": False, "detail": "Invalid mode"}, status_code=400)
    system_state.update(mode=body.mode)
    system_state.send_command({"action": "mode", "mode": body.mode})
    return JSONResponse({"success": True})


@app.post("/control/servo")
async def set_servo(body: ServoBody) -> JSONResponse:
    angle = 92 if body.angle == "home" else int(body.angle)
    system_state.send_command({"action": "servo", "name": body.name, "angle": angle})
    return JSONResponse({"success": True})


@app.post("/control/sort")
async def manual_sort(body: dict) -> JSONResponse:
    cls = body.get("class", "")
    if cls not in CLASSES:
        return JSONResponse({"success": False, "detail": "Unknown class"}, status_code=400)
    system_state.send_command({"action": "manual_sort", "class": cls})
    return JSONResponse({"success": True})


@app.post("/control/flash")
async def set_flash(body: dict) -> JSONResponse:
    on = bool(body.get("on", False))
    system_state.send_command({"action": "flash", "on": on})
    return JSONResponse({"success": True})


@app.post("/control/roi")
async def set_roi(body: dict) -> JSONResponse:
    system_state.send_command({
        "action": "set_roi",
        "cx":   int(body.get("cx", 320)),
        "cy":   int(body.get("cy", 240)),
        "size": int(body.get("size", 320)),
    })
    return JSONResponse({"success": True})


@app.post("/control/config")
async def update_config(body: dict) -> JSONResponse:
    cmd = {"action": "update_config"}
    for k in ("detect_dist", "withdraw_dist", "cooldown", "yolo_conf"):
        if k in body:
            cmd[k] = body[k]
    system_state.send_command(cmd)
    return JSONResponse({"success": True})


@app.post("/api/performance/confirm")
async def confirm_sort(body: ConfirmBody) -> JSONResponse:
    predicted = system_state.snapshot().get("last_predicted", "")
    if not predicted:
        return JSONResponse({"success": False, "detail": "No pending prediction"})
    system_state.confirm_sort(predicted, body.actual_class)
    return JSONResponse({"success": True})


@app.post("/api/stats/reset")
async def reset_stats() -> JSONResponse:
    system_state.reset_stats()
    return JSONResponse({"success": True})


@app.post("/api/performance/reset")
async def reset_performance() -> JSONResponse:
    system_state.reset_performance()
    return JSONResponse({"success": True})


@app.get("/api/export/csv")
async def export_csv() -> Response:
    path = system_state.export_csv()
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sort_history.csv"},
        )
    except FileNotFoundError:
        return JSONResponse({"success": False, "detail": "No data"}, status_code=404)


# =====================================================================
# WebSocket endpoint
# =====================================================================


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)

    # Initial full load
    snap = system_state.snapshot()
    await ws.send_json({
        "type":    "init",
        "snap":    snap,
        "engine":  _engine_running(),
    })

    try:
        while True:
            await asyncio.sleep(0.2)  # 5 fps push rate
            snap = system_state.snapshot()
            await ws.send_json({
                "type":            "tick",
                "state":           snap["detection_state"],
                "mode":            snap["mode"],
                "main_dist":       snap["main_distance_cm"],
                "bin_dists":       snap["bin_distances_cm"],
                "engine_running":  snap["engine_running"] or _engine_running(),
                "sort_counts":     snap["sort_counts"],
                "total_sorted":    snap["total_sorted"],
                "last_label":      snap["last_label"],
                "last_confidence": snap["last_confidence"],
                "sort_history":    snap["sort_history"],
                "uptime_start":    snap["uptime_start"],
            })

            # Non-blocking receive from client
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                action = msg.get("action")
                if action == "ping":
                    await ws.send_json({"type": "pong"})
                elif action == "get_snapshot":
                    snap = system_state.snapshot()
                    await ws.send_json({"type": "snapshot", "snap": snap,
                                        "engine": _engine_running()})
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# =====================================================================
# Background push — bin levels every 2 s
# =====================================================================


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.create_task(_bin_push_loop())


async def _bin_push_loop() -> None:
    while True:
        await asyncio.sleep(2)
        snap   = system_state.snapshot()
        dists  = snap["bin_distances_cm"]
        levels = {f"bin{i+1}": dists[i] if i < len(dists) else 999 for i in range(4)}
        await _broadcast({"type": "bin_levels", "levels": levels})


# =====================================================================
# Start / Stop helpers (for embedding in other scripts)
# =====================================================================

_server_thread:  Optional[threading.Thread] = None
_server_loop:    Optional[asyncio.AbstractEventLoop] = None
_uvicorn_server: Optional[uvicorn.Server] = None


def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start uvicorn in a background daemon thread (call once)."""
    global _server_thread, _server_loop, _uvicorn_server

    config = uvicorn.Config(app, host=host, port=port,
                            log_level="warning", loop="asyncio")
    _uvicorn_server = uvicorn.Server(config)

    def _run() -> None:
        global _server_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _server_loop = loop
        loop.run_until_complete(_uvicorn_server.serve())

    _server_thread = threading.Thread(target=_run, daemon=True, name="UvicornServer")
    _server_thread.start()
    logger.info("FastAPI server started on http://%s:%d", host, port)


def stop() -> None:
    global _uvicorn_server
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Update _server_loop so sort callbacks can schedule broadcasts
    async def _set_loop():
        global _server_loop
        _server_loop = asyncio.get_running_loop()

    @app.on_event("startup")
    async def _capture_loop():
        global _server_loop
        _server_loop = asyncio.get_running_loop()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
