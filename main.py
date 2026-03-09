from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

app = FastAPI(
    title="avart-engine",
    version="0.6.0",
    description="Alpha-based silhouette engine for Avart",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_upload_to_rgba(upload: UploadFile) -> np.ndarray:
    """
    Read uploaded file as RGBA image.
    Expects PNG with transparency for best result.
    Returns numpy array shape (H, W, 4)
    """
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    pil = Image.open(io.BytesIO(data)).convert("RGBA")
    rgba = np.array(pil)

    if rgba is None or rgba.shape[2] != 4:
        raise ValueError("Could not decode RGBA image")
    return rgba
    
def alpha_to_mask(
    rgba: np.ndarray,
    alpha_threshold: int = 1,
    smooth: bool = True,
) -> np.ndarray:

    alpha = rgba[:, :, 3]

    # brug alpha direkte
    mask = np.where(alpha > alpha_threshold, 255, 0).astype(np.uint8)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask
    """
    Convert alpha channel to binary mask:
    subject = 255
    background = 0
    """

    alpha = rgba[:, :, 3]

    # brug alpha direkte
    mask = np.where(alpha > alpha_threshold, 255, 0).astype(np.uint8)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return keep_largest_component(mask)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only largest connected white component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(mask)
    clean[labels == largest_label] = 255
    return clean


def smooth_contour(contour: np.ndarray, window: int = 17) -> np.ndarray:
    """
    Moving average smoothing over contour points.
    """
    pts = contour[:, 0, :].astype(np.float32)
    n = len(pts)

    if n < window or n < 20:
        return contour

    if window % 2 == 0:
        window += 1

    pad = window // 2
    pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])

    smoothed = []
    for i in range(n):
        seg = pts_pad[i:i + window]
        smoothed.append(seg.mean(axis=0))

    smoothed = np.array(smoothed, dtype=np.int32).reshape(-1, 1, 2)
    return smoothed


def get_smoothed_outer_contour(
    mask: np.ndarray,
    epsilon_ratio: float = 0.001,
    smooth_window: int = 15,
) -> np.ndarray:
    """
    Find outer contour, simplify slightly, then smooth.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")

    largest = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    eps = max(1.0, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(largest, eps, True)

    smoothed = smooth_contour(approx, window=smooth_window)
    return smoothed


def render_preview_png(
    contour: np.ndarray,
    width: int,
    height: int,
    thickness: int = 2,
    upscale: int = 4,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> bytes:
    """
    Draw contour as black stroke on white background.
    Optional crop_to_subject for testing.
    """
    if crop_to_subject:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)

        contour = contour.copy()
        contour[:, 0, 0] -= x1
        contour[:, 0, 1] -= y1

        width = x2 - x1
        height = y2 - y1

    W = width * upscale
    H = height * upscale

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    c = contour.copy().astype(np.int32)
    c[:, 0, 0] *= upscale
    c[:, 0, 1] *= upscale

    cv2.drawContours(
        canvas,
        [c],
        -1,
        (0, 0, 0),
        thickness=max(1, thickness * upscale),
        lineType=cv2.LINE_AA,
    )

    canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Could not encode PNG")
    return png.tobytes()


def contour_to_svg(
    contour: np.ndarray,
    width: int,
    height: int,
    stroke_width: float = 2.0,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> str:
    """
    Convert contour to SVG path.
    STEP 1 version = polyline path, not bezier yet.
    """
    if crop_to_subject:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)

        contour = contour.copy()
        contour[:, 0, 0] -= x1
        contour[:, 0, 1] -= y1

        width = x2 - x1
        height = y2 - y1

    pts = contour[:, 0, :]
    if len(pts) < 3:
        raise ValueError("Contour too small for SVG")

    d = f"M {pts[0,0]} {pts[0,1]} "
    for p in pts[1:]:
        d += f"L {p[0]} {p[1]} "
    d += "Z"

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <path d="{d}" fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>
</svg>'''
    return svg


# --------------------------------------------------
# API
# --------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(10, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    smooth_window: int = Query(17, ge=5, le=51),
    thickness: int = Query(2, ge=1, le=8),
    upscale: int = Query(4, ge=1, le=8),
    crop_to_subject: bool = Query(False),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = read_upload_to_rgba(file)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(
            rgba,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
        )

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        png = render_preview_png(
            contour,
            width=w,
            height=h,
            thickness=thickness,
            upscale=upscale,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return Response(content=png, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/alpha/svg")
async def alpha_svg(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(10, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    smooth_window: int = Query(17, ge=5, le=51),
    stroke_width: float = Query(2.0, ge=0.5, le=10.0),
    crop_to_subject: bool = Query(False),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = read_upload_to_rgba(file)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(
            rgba,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
        )

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        svg = contour_to_svg(
            contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return Response(content=svg, media_type="image/svg+xml")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
