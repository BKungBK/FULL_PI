"""
server.py — SmartBin V2  FastAPI + WebSocket Server
=====================================================
Runs alongside the desktop app to serve ui.html browser clients.

Endpoints:
    GET  /                         → redirect to ui.html (if exists)
    GET  /api/stream               → MJPEG proxy from ESP32-CAM
    GET  /api/system-status        → JSON snapshot
    GET  /api/bin-levels           → bin distances
    POST /control/mode             → {"mode": "auto"|"manual"}
    POST /control/servo            → {"name": "capture"|"sort"|"all", "angle": int}
    POST /control/sort             → {"class": "metal"|...}
    POST /api/performance/confirm  → {"actual_class": "..."}
    WS   /ws                       → real-time push every 100 ms

Usage (standalone):
    python server.py

Usage (from app_ui.py):
    import server; server.start()   # runs uvicorn in background thread
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
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

# ── Active WebSocket clients ──────────────────────────────────────────
_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()


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
        "type":  "sort_result",
        "data":  {
            "class":      snap["last_label"],
            "confidence": snap["last_confidence"],
        },
        "stats": {
            "total_sorted": snap["total_sorted"],
            "by_class":     snap["sort_counts"],
        },
    }
    # Schedule in the server event loop (safe from any thread)
    if _server_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _server_loop)


system_state.register_sort_callback(_on_sort_event)

# =====================================================================
# REST endpoints
# =====================================================================


@app.get("/api/system-status")
async def get_system_status() -> JSONResponse:
    snap = system_state.snapshot()
    return JSONResponse({
        "success":        True,
        "mode":           snap["mode"],
        "engine_running": snap["engine_running"],
        "esp32_ip":       snap["esp32_ip"],
        "detection_state": snap["detection_state"],
        "last_label":     snap["last_label"],
        "last_confidence": snap["last_confidence"],
    })


@app.get("/api/bin-levels")
async def get_bin_levels() -> JSONResponse:
    snap = system_state.snapshot()
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


# ── Control endpoints ────────────────────────────────────────────────

class ModeBody(BaseModel):
    mode: str


class ServoBody(BaseModel):
    name: str
    angle: object  # int or "home"


class SortBody(BaseModel):
    class_: str = ""

    class Config:
        populate_by_name = True

    @classmethod
    def from_cls(cls, data: dict) -> "SortBody":
        return cls(class_=data.get("class", ""))


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


# =====================================================================
# WebSocket endpoint
# =====================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)

    # Send initial state
    snap = system_state.snapshot()
    await ws.send_json({
        "type":  "connected",
        "stats": system_state.get_stats_payload(),
        "state": snap["detection_state"],
    })

    try:
        while True:
            # Push state every 200 ms
            await asyncio.sleep(0.2)
            snap = system_state.snapshot()
            await ws.send_json({
                "type":          "system_status",
                "state":         snap["detection_state"],
                "mode":          snap["mode"],
                "main_dist":     snap["main_distance_cm"],
                "engine_running": snap["engine_running"],
            })

            # Check for incoming client messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                action = msg.get("action")
                if action == "get_stats":
                    await ws.send_json({"type": "stats", "data": system_state.get_stats_payload()})
                elif action == "ping":
                    await ws.send_json({"type": "pong"})
                elif action == "mode":
                    system_state.update(mode=msg.get("mode", "auto"))
                    system_state.send_command(msg)
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# =====================================================================
# Background loop — push bin levels every 2 s
# =====================================================================

@app.on_event("startup")
async def _start_bin_push() -> None:
    asyncio.create_task(_bin_push_loop())


async def _bin_push_loop() -> None:
    while True:
        await asyncio.sleep(2)
        snap   = system_state.snapshot()
        dists  = snap["bin_distances_cm"]
        levels = {f"bin{i+1}": dists[i] if i < len(dists) else 999 for i in range(4)}
        await _broadcast({"type": "bin_levels", "levels": levels})


# =====================================================================
# Start / Stop helpers (called from app_ui.py)
# =====================================================================

_server_thread: Optional[threading.Thread] = None
_server_loop:   Optional[asyncio.AbstractEventLoop] = None
_uvicorn_server: Optional[uvicorn.Server] = None


def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start uvicorn in a background daemon thread (call once)."""
    global _server_thread, _server_loop, _uvicorn_server

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
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
    """Gracefully shut down the uvicorn server."""
    global _uvicorn_server
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
        logger.info("FastAPI server stopping …")


# =====================================================================
# CLI entry
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
