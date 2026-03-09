from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

app = FastAPI(
    title="avart-engine",
    version="0.8.0",
    description="Alpha-based silhouette engine with selective smoothing",
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
    Read uploaded PNG with transparency correctly.
    """
    data = upload.file.read()

    if not data:
        raise ValueError("Empty file")

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image")

    if len(img.shape) != 3:
        raise ValueError("Image must have color channels")

    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)

    if img.shape[2] != 4:
        raise ValueError("Image must have 4 channels")

    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return rgba


def alpha_to_mask(
    rgba: np.ndarray,
    alpha_threshold: int = 1,
    smooth: bool = True,
) -> np.ndarray:
    """
    Convert alpha channel to binary mask:
    subject = 255
    background = 0
    """
    alpha = rgba[:, :, 3]
    mask = np.where(alpha > alpha_threshold, 255, 0).astype(np.uint8)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def moving_average_closed(points: np.ndarray, window: int) -> np.ndarray:
    """
    Circular moving average for closed contour points.
    points shape: (N, 2)
    """
    n = len(points)
    if n < window or n < 10:
        return points.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    pts_pad = np.vstack([points[-pad:], points, points[:pad]])

    kernel = np.ones(window, dtype=np.float32) / window
    xs = np.convolve(pts_pad[:, 0], kernel, mode="valid")
    ys = np.convolve(pts_pad[:, 1], kernel, mode="valid")

    smoothed = np.stack([xs, ys], axis=1)
    return smoothed


def find_face_zone_indices(points: np.ndarray, face_ratio: float = 0.22) -> tuple[int, int]:
    """
    Find the contour section that is most likely the face profile.
    Assumes face points are among the rightmost contour points.
    Returns (start_idx, end_idx) on closed contour.
    """
    xs = points[:, 0]
    x_max = xs.max()
    x_min = xs.min()
    span = max(1.0, x_max - x_min)

    # Rightmost area = likely face zone
    threshold = x_max - span * face_ratio
    idx = np.where(xs >= threshold)[0]

    if len(idx) == 0:
        return 0, min(len(points) - 1, len(points) // 4)

    # take longest consecutive run
    runs = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))

    longest = max(runs, key=lambda r: r[1] - r[0])
    return longest


def selective_smooth_contour(
    contour: np.ndarray,
    face_window: int = 7,
    rest_window: int = 25,
    face_ratio: float = 0.22,
) -> np.ndarray:
    """
    Smooth face zone lightly and the rest more aggressively.
    """
    pts = contour[:, 0, :].astype(np.float32)
    n = len(pts)

    if n < 20:
        return contour

    start, end = find_face_zone_indices(pts, face_ratio=face_ratio)

    smoothed_all = moving_average_closed(pts, rest_window)
    smoothed_face = moving_average_closed(pts, face_window)

    final = smoothed_all.copy()

    if start <= end:
        final[start:end + 1] = smoothed_face[start:end + 1]
    else:
        # wrap-around case
        final[start:] = smoothed_face[start:]
        final[:end + 1] = smoothed_face[:end + 1]

    final = np.round(final).astype(np.int32).reshape(-1, 1, 2)
    return final


def get_smoothed_outer_contour(
    mask: np.ndarray,
    epsilon_ratio: float = 0.001,
    face_window: int = 7,
    rest_window: int = 25,
    face_ratio: float = 0.22,
) -> np.ndarray:
    """
    Find outer contour and smooth it selectively.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contour found")

    largest = max(contours, key=cv2.contourArea)

    # Very light simplification only
    peri = cv2.arcLength(largest, True)
    eps = max(0.5, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(largest, eps, True)

    smoothed = selective_smooth_contour(
        approx,
        face_window=face_window,
        rest_window=rest_window,
        face_ratio=face_ratio,
    )

    return smoothed


def crop_contour_to_subject(
    contour: np.ndarray,
    width: int,
    height: int,
    pad: int = 30,
):
    x, y, w, h = cv2.boundingRect(contour)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)

    cropped = contour.copy()
    cropped[:, 0, 0] -= x1
    cropped[:, 0, 1] -= y1

    return cropped, (x2 - x1), (y2 - y1)


def render_preview_png(
    contour: np.ndarray,
    width: int,
    height: int,
    thickness: int = 2,
    upscale: int = 4,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> bytes:
    if crop_to_subject:
        contour, width, height = crop_contour_to_subject(contour, width, height, pad=pad)

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


def render_debug_png(
    rgba: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    thickness: int = 2,
    upscale: int = 4,
) -> bytes:
    h, w = rgba.shape[:2]

    checker = np.zeros((h, w, 3), dtype=np.uint8)
    tile = 20
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            v = 220 if ((x // tile) + (y // tile)) % 2 == 0 else 245
            checker[y:y+tile, x:x+tile] = (v, v, v)

    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    checker_f = checker.astype(np.float32)
    panel1 = (rgb * alpha + checker_f * (1 - alpha)).astype(np.uint8)

    panel2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    panel3 = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.drawContours(panel3, [contour], -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    final_png = render_preview_png(
        contour=contour,
        width=w,
        height=h,
        thickness=thickness,
        upscale=upscale,
        crop_to_subject=False,
        pad=30,
    )
    final_arr = cv2.imdecode(np.frombuffer(final_png, np.uint8), cv2.IMREAD_COLOR)

    def label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
        return out

    panel1 = label(panel1, "1. original")
    panel2 = label(panel2, "2. mask")
    panel3 = label(panel3, "3. contour")
    final_arr = label(final_arr, "4. final")

    top = np.hstack([panel1, panel2])
    bottom = np.hstack([panel3, final_arr])
    debug = np.vstack([top, bottom])

    ok, png = cv2.imencode(".png", debug)
    if not ok:
        raise ValueError("Could not encode debug PNG")

    return png.tobytes()


def contour_to_svg(
    contour: np.ndarray,
    width: int,
    height: int,
    stroke_width: float = 2.0,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> str:
    if crop_to_subject:
        contour, width, height = crop_contour_to_subject(contour, width, height, pad=pad)

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
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.001, ge=0.0001, le=0.02),
    face_window: int = Query(7, ge=3, le=31),
    rest_window: int = Query(25, ge=5, le=61),
    face_ratio: float = Query(0.22, ge=0.05, le=0.5),
    thickness: int = Query(2, ge=1, le=8),
    upscale: int = Query(4, ge=1, le=8),
    crop_to_subject: bool = Query(True),
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
            face_window=face_window,
            rest_window=rest_window,
            face_ratio=face_ratio,
        )

        png = render_preview_png(
            contour=contour,
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


@app.post("/alpha/debug")
async def alpha_debug(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.001, ge=0.0001, le=0.02),
    face_window: int = Query(7, ge=3, le=31),
    rest_window: int = Query(25, ge=5, le=61),
    face_ratio: float = Query(0.22, ge=0.05, le=0.5),
    thickness: int = Query(2, ge=1, le=8),
    upscale: int = Query(4, ge=1, le=8),
):
    try:
        rgba = read_upload_to_rgba(file)

        mask = alpha_to_mask(
            rgba,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
        )

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            face_window=face_window,
            rest_window=rest_window,
            face_ratio=face_ratio,
        )

        png = render_debug_png(
            rgba=rgba,
            mask=mask,
            contour=contour,
            thickness=thickness,
            upscale=upscale,
        )

        return Response(content=png, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/alpha/svg")
async def alpha_svg(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.001, ge=0.0001, le=0.02),
    face_window: int = Query(7, ge=3, le=31),
    rest_window: int = Query(25, ge=5, le=61),
    face_ratio: float = Query(0.22, ge=0.05, le=0.5),
    stroke_width: float = Query(2.0, ge=0.5, le=10.0),
    crop_to_subject: bool = Query(True),
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
            face_window=face_window,
            rest_window=rest_window,
            face_ratio=face_ratio,
        )

        svg = contour_to_svg(
            contour=contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return Response(content=svg, media_type="image/svg+xml")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
