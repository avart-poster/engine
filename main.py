from __future__ import annotations

import io
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from rembg import remove, new_session

from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPDF

from svglib.svglib import svg2rlg


# --------------------------------
# AVART DESIGN SETTINGS
# --------------------------------

BG_COLOR = (0.95, 0.93, 0.90)

PAGE_W_MM = 500
PAGE_H_MM = 700

TOP_BAND_MM = 115

TITLE_FONT_SIZE = 35

LOGO_WIDTH_MM = 50
LOGO_BOTTOM_MM = 50

DEFAULT_STROKE_WIDTH = 3.5

MAX_DIMENSION = 1600
REMBG_MODEL = "u2net"


# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="avart-engine",
    version="1.0.0",
    description="Avart silhouette engine with SVG + PDF poster output",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Optional custom font
# --------------------------------------------------

TITLE_FONT = "Helvetica-Bold"

try:
    pdfmetrics.registerFont(TTFont("TheSeasonsBold", "fonts/TheSeasons-Bold.otf"))
    TITLE_FONT = "TheSeasonsBold"
except Exception:
    TITLE_FONT = "Helvetica-Bold"


# --------------------------------------------------
# rembg session
# --------------------------------------------------

_rembg_session = None


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(REMBG_MODEL)
    return _rembg_session


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def resize_if_needed_rgba(rgba: np.ndarray, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    h, w = rgba.shape[:2]
    longest = max(h, w)

    if longest <= max_dimension:
        return rgba

    scale = max_dimension / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def remove_background_if_needed(upload: UploadFile, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image")

    # Hvis upload allerede har ægte transparency
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < 250):
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            rgba = resize_if_needed_rgba(rgba, max_dimension=max_dimension)
            rgba = cv2.copyMakeBorder(
                rgba, 0, 180, 0, 0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0),
            )
            return rgba

    # Resize før rembg for stabilitet
    max_input_size = 1600
    h, w = img.shape[:2]
    scale = min(1.0, max_input_size / max(h, w))

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("Could not encode resized image")

        data = buffer.tobytes()

    output = remove(data, session=get_rembg_session())

    arr_out = np.frombuffer(output, np.uint8)
    img_out = cv2.imdecode(arr_out, cv2.IMREAD_UNCHANGED)

    if img_out is None:
        raise ValueError("Background removal failed")

    if len(img_out.shape) == 3 and img_out.shape[2] == 3:
        alpha = np.full((img_out.shape[0], img_out.shape[1], 1), 255, dtype=np.uint8)
        img_out = np.concatenate([img_out, alpha], axis=2)

    if len(img_out.shape) != 3 or img_out.shape[2] != 4:
        raise ValueError("Background removal did not return RGBA")

    # ekstra transparent bund, så contour kan gå helt ned
    img_out = cv2.copyMakeBorder(
        img_out, 0, 180, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )

    rgba = cv2.cvtColor(img_out, cv2.COLOR_BGRA2RGBA)
    return resize_if_needed_rgba(rgba, max_dimension=max_dimension)


def alpha_to_mask(
    rgba: np.ndarray,
    alpha_threshold: int = 1,
    smooth: bool = True,
) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = np.where(alpha > alpha_threshold, 255, 0).astype(np.uint8)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def smooth_contour_points(points: np.ndarray, smooth_window: int = 9) -> np.ndarray:
    n = len(points)
    if n < smooth_window or n < 10:
        return points.copy()

    if smooth_window % 2 == 0:
        smooth_window += 1

    pad = smooth_window // 2
    pts_pad = np.vstack([points[-pad:], points, points[:pad]])

    smoothed = []
    for i in range(n):
        segment = pts_pad[i:i + smooth_window]
        smoothed.append(segment.mean(axis=0))

    return np.array(smoothed, dtype=np.float32)


def get_smoothed_outer_contour(
    mask: np.ndarray,
    epsilon_ratio: float = 0.00020,
    smooth_window: int = 13,
) -> np.ndarray:
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_blur = cv2.GaussianBlur(mask, (13, 13), 0)

    contours, _ = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")

    largest = max(contours, key=cv2.contourArea)
    points = largest[:, 0, :].astype(np.float32)

    smoothed = smooth_contour_points(points, smooth_window=smooth_window)
    smoothed_contour = np.round(smoothed).astype(np.int32).reshape(-1, 1, 2)

    peri = cv2.arcLength(smoothed_contour, True)
    eps = max(0.5, peri * epsilon_ratio)

    simplified = cv2.approxPolyDP(smoothed_contour, eps, True)
    return simplified


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


def anchor_contour_to_bottom(contour: np.ndarray, height: int) -> np.ndarray:
    pts = contour[:, 0, :]
    lowest_y = pts[:, 1].max()
    shift = (height - 1) - lowest_y
    pts[:, 1] = pts[:, 1] + shift
    return contour


def open_contour_at_bottom(contour: np.ndarray, height: int, bleed: int = 0) -> np.ndarray:
    pts = contour[:, 0, :].astype(np.int32)

    ys = pts[:, 1]
    max_y = ys.max()

    band = np.where(ys >= max_y - 15)[0]
    if len(band) < 2:
        band = np.where(ys >= max_y - 5)[0]

    if len(band) < 2:
        idx_sorted = np.argsort(ys)[::-1]
        i_left, i_right = idx_sorted[0], idx_sorted[1]
    else:
        xs = pts[band, 0]
        i_left = band[np.argmin(xs)]
        i_right = band[np.argmax(xs)]

    a, b = sorted([i_left, i_right])
    open_pts = np.vstack([pts[b:], pts[:a + 1]])

    bottom_y = height - 1 + bleed
    open_pts[0, 1] = bottom_y
    open_pts[-1, 1] = bottom_y

    return open_pts.reshape(-1, 1, 2)


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

    contour = anchor_contour_to_bottom(contour, height)
    contour = open_contour_at_bottom(contour, height=height, bleed=0)

    W = width * upscale
    H = height * upscale

    canvas_img = np.full((H, W, 3), 255, dtype=np.uint8)

    pts = contour.copy().astype(np.int32)
    pts[:, 0, 0] *= upscale
    pts[:, 0, 1] *= upscale

    cv2.polylines(
        canvas_img,
        [pts],
        isClosed=False,
        color=(0, 0, 0),
        thickness=max(1, int(round(thickness * upscale))),
        lineType=cv2.LINE_AA,
    )

    canvas_img = cv2.resize(canvas_img, (width, height), interpolation=cv2.INTER_AREA)

    ok, png = cv2.imencode(".png", canvas_img)
    if not ok:
        raise ValueError("Could not encode PNG")

    return png.tobytes()


def set_stroke_width_recursive(node, stroke_width: float):
    if hasattr(node, "strokeWidth"):
        node.strokeWidth = stroke_width

    if hasattr(node, "contents"):
        for child in node.contents:
            set_stroke_width_recursive(child, stroke_width)


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
            checker[y:y + tile, x:x + tile] = (v, v, v)

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
    stroke_width: float = 3.5,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> str:
    if crop_to_subject:
        contour, width, height = crop_contour_to_subject(contour, width, height, pad=pad)

    contour = anchor_contour_to_bottom(contour, height)
    contour = open_contour_at_bottom(contour, height=height, bleed=0)
    pts = contour[:, 0, :]

    if len(pts) < 2:
        raise ValueError("Contour too small")

    d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for p in pts[1:]:
        d.append(f"L {p[0]:.2f} {p[1]:.2f}")

    path = " ".join(d)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">
  <path
    d="{path}"
    fill="none"
    stroke="black"
    stroke-width="{stroke_width}"
    stroke-linecap="round"
    stroke-linejoin="round"/>
</svg>
'''
    return svg


def estimate_head_width(contour: np.ndarray) -> float:
    pts = contour[:, 0, :].astype(np.float32)

    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    total_h = max_y - min_y

    # brug øverste del af silhouetten som "hoved"
    cutoff_y = min_y + total_h * 0.55
    head_pts = pts[pts[:, 1] <= cutoff_y]

    if len(head_pts) < 2:
        return float(pts[:, 0].max() - pts[:, 0].min())

    head_w = head_pts[:, 0].max() - head_pts[:, 0].min()
    return float(head_w)


def generate_poster_pdf(
    svg_string: str,
    name: str,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    head_width: float | None = None,
    scale_adjust: float = 0.0,  #
) -> bytes:
    width = PAGE_W_MM * mm
    height = PAGE_H_MM * mm

    top_band_h = TOP_BAND_MM * mm
    logo_width = LOGO_WIDTH_MM * mm
    logo_bottom = LOGO_BOTTOM_MM * mm

    tmp_svg = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
    tmp_svg.write(svg_string.encode("utf-8"))
    tmp_svg.close()

    drawing = svg2rlg(tmp_svg.name)
    if drawing is None:
        raise ValueError("Could not convert silhouette SVG to drawing")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(width, height))

    # background
    c.setFillColorRGB(*BG_COLOR)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # title
    c.setFillColorRGB(0, 0, 0)
    c.setFont(TITLE_FONT, TITLE_FONT_SIZE)
    c.drawCentredString(width / 2, height - (top_band_h / 2), name)

    # silhouette area
    silhouette_top_y = height - top_band_h
    silhouette_bottom_y = 0
    silhouette_height = silhouette_top_y - silhouette_bottom_y

    # original bounds
    min_x, min_y, max_x, max_y = drawing.getBounds()
    raw_w = max_x - min_x
    raw_h = max_y - min_y

    TARGET_HEAD_RATIO = 0.60
    MAX_HEIGHT_RATIO = 1.50

    if head_width is None or head_width <= 0:
        head_width = raw_w * 0.7

    silhouette_scale = (width * TARGET_HEAD_RATIO) / head_width
    silhouette_scale *= (1 + scale_adjust)
    drawing.scale(silhouette_scale, silhouette_scale)
    set_stroke_width_recursive(drawing, stroke_width / silhouette_scale)

    # recalc bounds after scaling
    min_x, min_y, max_x, max_y = drawing.getBounds()
    draw_w = max_x - min_x

    # center horizontally, anchor to bottom
    x = (width - draw_w) / 2 - min_x
    y = -min_y

    c.saveState()
    c.translate(x, y)
    renderPDF.draw(drawing, c, 0, 0)
    c.restoreState()

    # logo
    if os.path.exists("assets/avart-logo.svg"):
        logo = svg2rlg("assets/avart-logo.svg")
        if logo is not None:
            logo_scale = logo_width / logo.width
            logo.scale(logo_scale, logo_scale)

            l_min_x, l_min_y, l_max_x, l_max_y = logo.getBounds()
            logo_w = l_max_x - l_min_x

            logo_x = (width - logo_w) / 2 - l_min_x
            logo_y = logo_bottom - l_min_y

            renderPDF.draw(logo, c, logo_x, logo_y)

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()

    try:
        os.unlink(tmp_svg.name)
    except Exception:
        pass

    return pdf_bytes

# --------------------------------------------------
# API
# --------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file: UploadFile = File(...),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00020, ge=0.00005, le=0.02),
    smooth_window: int = Query(13, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=max_dimension)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(rgba, alpha_threshold=alpha_threshold, smooth=smooth)

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
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
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00020, ge=0.00005, le=0.02),
    smooth_window: int = Query(13, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=max_dimension)

        mask = alpha_to_mask(rgba, alpha_threshold=alpha_threshold, smooth=smooth)

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
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
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00020, ge=0.00005, le=0.02),
    smooth_window: int = Query(13, ge=3, le=51),
    stroke_width: float = Query(3.5, ge=0.5, le=12.0),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=max_dimension)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(rgba, alpha_threshold=alpha_threshold, smooth=smooth)

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        svg = contour_to_svg(
            contour=contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Content-Disposition": 'attachment; filename="silhouette.svg"'},
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post(
    "/poster/pdf",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "application/pdf": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "description": "PDF file",
        }
    },
)

@app.post("/poster/pdf")
async def poster_pdf(
    file: UploadFile = File(...),
    name: str = Query("Clara & Ellinor"),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00020, ge=0.00005, le=0.02),
    smooth_window: int = Query(13, ge=3, le=51),
    stroke_width: float = Query(3.5, ge=0.5, le=12.0),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
    scale_adjust: float = Query(0.0, ge=-0.2, le=0.2),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=max_dimension)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(rgba, alpha_threshold=alpha_threshold, smooth=smooth)

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        head_width = estimate_head_width(contour)

        svg = contour_to_svg(
            contour=contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        pdf_bytes = generate_poster_pdf(
            svg,
            name,
            stroke_width=stroke_width,
            head_width=head_width,
            scale_adjust=scale_adjust,
        )
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
