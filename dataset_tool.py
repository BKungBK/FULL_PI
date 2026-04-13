"""
dataset_tool.py — SmartBin Dataset Collection Tool
====================================================
Captures images from ESP32-CAM, crops, resizes to 96×96,
applies CLAHE post-processing, and generates 12 augmented
variants per capture for robust ML training.

Pipeline per capture:
    ESP32 VGA frame → crop ROI → resize 96×96 (INTER_AREA)
    → CLAHE enhancement → save original + 12 augmentations

Run:    python dataset_tool.py
Open:   http://localhost:8001

Output:
    dataset/
    ├── plastic/   (*.jpg, 96×96)
    ├── metal/     (*.jpg, 96×96)
    └── glass/     (*.jpg, 96×96)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────
ESP32_IP    = os.environ.get("ESP32_IP", "10.42.0.177")
DATASET_DIR = Path(__file__).parent / "dataset"
LABELS      = ("plastic", "metal", "glass")
OUTPUT_SIZE = 96
PORT        = 8001
AUGMENT_ENABLED = True   # False = save originals only

# ── Setup ─────────────────────────────────────────────────────────────
for _label in LABELS:
    (DATASET_DIR / _label).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetTool")


# ══════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def postprocess(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE adaptive contrast enhancement.

    Improves detail visibility in uneven lighting conditions.
    Matches the CLAHE applied in main.py preprocess_classifier()
    so training data and inference use the same pipeline.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def augment(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generate 12 augmented versions of a 96×96 image.

    Covers geometric, photometric, noise, and colour transforms
    to build a robust and diverse training set.

    Returns:
        list of (suffix_tag, augmented_image) tuples
    """
    results: List[Tuple[str, np.ndarray]] = []
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # ── 1. Geometric ─────────────────────────────────────────────

    # Horizontal flip
    results.append(("hflip", cv2.flip(img, 1)))

    # Small rotations  (±15°)
    for deg in [-15, 15]:
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        rot = cv2.warpAffine(
            img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
        )
        suffix = f"r{'+' if deg > 0 else ''}{deg}"
        results.append((suffix, rot))

    # Flip + rotation combo
    flipped = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D(center, 12, 1.0)
    results.append((
        "hfr12",
        cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT_101),
    ))

    # ── 2. Photometric ───────────────────────────────────────────

    # Brightness shifts
    for beta in [-30, 30]:
        tag = f"b{'+' if beta > 0 else ''}{beta}"
        results.append((tag, cv2.convertScaleAbs(img, alpha=1.0, beta=beta)))

    # Contrast (preserve mean brightness)
    mean_val = float(np.mean(img))
    for alpha, tag in [(0.75, "c075"), (1.35, "c135")]:
        adj = cv2.convertScaleAbs(
            img, alpha=alpha, beta=mean_val * (1.0 - alpha)
        )
        results.append((tag, adj))

    # ── 3. Noise & Blur ─────────────────────────────────────────

    # Gaussian noise (σ=10)
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = np.clip(
        img.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    results.append(("gn", noisy))

    # Slight Gaussian blur (simulates defocus)
    results.append(("gb", cv2.GaussianBlur(img, (3, 3), 0.7)))

    # ── 4. Colour ────────────────────────────────────────────────

    # Hue shifts (simulate daylight vs warm light)
    for shift, tag in [(10, "h+10"), (-10, "h-10")]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        shifted = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )
        results.append((tag, shifted))

    return results  # 12 variants


# ══════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(title="SmartBin Dataset Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(
        Path(__file__).parent / "dataset_ui.html",
        media_type="text/html",
    )


@app.get("/stream")
async def stream_proxy():
    """Proxy ESP32-CAM MJPEG stream."""
    src = f"http://{ESP32_IP}:81/stream"
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)

    async def _gen():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", src) as resp:
                    if resp.status_code != 200:
                        return
                    async for chunk in resp.aiter_bytes(8192):
                        yield chunk
        except Exception as exc:
            logger.warning("Stream proxy error: %s", exc)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace;boundary=123456789000000000000987654321",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
    )


class CaptureBody(BaseModel):
    label: str
    cx: int = 320
    cy: int = 240
    size: int = 200
    flash: int = 100  # Default 100 instead of 200 for enclosed bins


@app.post("/capture")
async def capture(body: CaptureBody):
    """Grab frame → crop → resize 96×96 → CLAHE → save + augment."""
    if body.label not in LABELS:
        return JSONResponse({"error": f"Invalid label: {body.label}"}, 400)

    # ── 1. Turn ON Flashlight ─────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"http://{ESP32_IP}/control?var=led_intensity&val={body.flash}")
            await asyncio.sleep(0.8) # Wait 0.8s for camera auto-exposure to clamp down
    except Exception as exc:
        logger.warning(f"Failed to turn ON flash: {exc}")

    # ── 2. Grab single JPEG from ESP32 ────────────────────────
    img_bytes = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{ESP32_IP}/capture")
            if resp.status_code == 200:
                img_bytes = resp.content
            else:
                resp_error = resp.status_code
    except Exception as exc:
        logger.error(f"Cannot reach ESP32 capture: {exc}")

    # ── 3. Turn OFF Flashlight ────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"http://{ESP32_IP}/control?var=led_intensity&val=0")
    except Exception as exc:
        logger.warning(f"Failed to turn OFF flash: {exc}")

    if not img_bytes:
        return JSONResponse({"error": "Failed to get image from ESP32"}, 500)

    # ── 4. Decode ─────────────────────────────────────────────
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Image decode failed"}, 500)

    h, w = img.shape[:2]

    # ── 5. Crop square region at (cx, cy) ─────────────────────
    half = body.size // 2
    x1 = max(0, body.cx - half)
    y1 = max(0, body.cy - half)
    x2 = min(w, body.cx + half)
    y2 = min(h, body.cy + half)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return JSONResponse({"error": "Empty crop region"}, 500)

    # ── 6. Resize to 96×96 ────────────────────────────────────
    resized = cv2.resize(
        crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA
    )

    # ── 7. Post-process: CLAHE ────────────────────────────────
    processed = postprocess(resized)

    # ── 8. Save original ──────────────────────────────────────
    ts = int(time.time() * 1000)
    save_dir = DATASET_DIR / body.label

    orig_name = f"{ts}_orig.jpg"
    cv2.imwrite(
        str(save_dir / orig_name), processed,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    saved_count = 1

    # ── 9. Data Augmentation ──────────────────────────────────
    if AUGMENT_ENABLED:
        aug_variants = augment(processed)
        for suffix, aug_img in aug_variants:
            aug_name = f"{ts}_{suffix}.jpg"
            cv2.imwrite(
                str(save_dir / aug_name), aug_img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            saved_count += 1

    logger.info(
        "Saved %s: %d images (1 orig + %d aug) crop(%d,%d %dx%d)",
        body.label, saved_count, saved_count - 1,
        x1, y1, x2 - x1, y2 - y1,
    )

    # ── Return thumbnail ──────────────────────────────────────
    _, buf = cv2.imencode(".jpg", processed)
    b64 = base64.b64encode(buf).decode()

    return JSONResponse({
        "success":     True,
        "label":       body.label,
        "filename":    orig_name,
        "thumbnail":   f"data:image/jpeg;base64,{b64}",
        "saved_count": saved_count,
        "counts":      _get_counts(),
    })


@app.get("/stats")
async def stats():
    return JSONResponse({
        "counts":   _get_counts(),
        "augment":  AUGMENT_ENABLED,
        "aug_mult": 13 if AUGMENT_ENABLED else 1,
    })


@app.get("/gallery/{label}")
async def gallery(label: str, limit: int = 30):
    """Return recent original captures (not augmented variants)."""
    if label not in LABELS:
        return JSONResponse({"error": "Invalid label"}, 400)

    folder = DATASET_DIR / label
    # Gallery shows only '_orig' images for clarity
    orig_files = sorted(
        folder.glob("*_orig.jpg"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:limit]

    items = []
    for f in orig_files:
        data = f.read_bytes()
        items.append({
            "name": f.name,
            "src":  f"data:image/jpeg;base64,{base64.b64encode(data).decode()}",
        })

    total = len(list(folder.glob("*.jpg")))
    return JSONResponse({"items": items, "total": total})


@app.delete("/image/{label}/{filename}")
async def delete_image(label: str, filename: str):
    """Delete an image and all its augmented variants."""
    if label not in LABELS:
        return JSONResponse({"error": "Invalid label"}, 400)

    safe = Path(filename).name
    folder = DATASET_DIR / label

    # If _orig file → delete all variants with same timestamp
    if safe.endswith("_orig.jpg"):
        prefix = safe.replace("_orig.jpg", "")
        for variant in folder.glob(f"{prefix}_*.jpg"):
            variant.unlink()
            logger.info("Deleted %s/%s", label, variant.name)
    else:
        path = folder / safe
        if path.exists():
            path.unlink()
            logger.info("Deleted %s/%s", label, safe)

    return JSONResponse({"success": True, "counts": _get_counts()})


# ── Helpers ───────────────────────────────────────────────────────────

def _get_counts() -> dict:
    return {
        label: len(list((DATASET_DIR / label).glob("*.jpg")))
        for label in LABELS
    }


# ── CLI entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    aug_label = "ON (x13 per capture)" if AUGMENT_ENABLED else "OFF"
    print(f"\n{'=' * 52}")
    print(f"  SmartBin Dataset Collection Tool")
    print(f"{'=' * 52}")
    print(f"  ESP32-CAM:     {ESP32_IP}")
    print(f"  Dataset:       {DATASET_DIR}")
    print(f"  Output:        {OUTPUT_SIZE}x{OUTPUT_SIZE} JPEG")
    print(f"  Labels:        {', '.join(LABELS)}")
    print(f"  CLAHE:         ON (clipLimit=2.0)")
    print(f"  Augmentation:  {aug_label}")
    print(f"  Open:          http://localhost:{PORT}")
    print(f"{'=' * 52}\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
