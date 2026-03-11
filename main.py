from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

pdfmetrics.registerFont(TTFont("TheSeasonsBold", "fonts/TheSeasons-Bold.otf"))

assets/avart-logo.svg

app = FastAPI(
    title="avart-engine",
    version="0.9.0",
    description="Alpha-based silhouette engine with auto resize and simplify-after-smoothing",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DIMENSION = 1600


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

    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
    

def draw_svg_on_pdf(c, svg_path: str, x: float, y: float, max_width: float, max_height: float):
    drawing = svg2rlg(svg_path)

    if drawing is None:
        raise ValueError(f"Could not load SVG: {svg_path}")

    scale = min(
        max_width / drawing.width,
        max_height / drawing.height
    )

    c.saveState()
    c.translate(x, y)
    c.scale(scale, scale)
    renderPDF.draw(drawing, c, 0, 0)
    c.restoreState()


def open_contour_at_bottom(contour: np.ndarray, height: int, bleed: int = 0) -> np.ndarray:
    """
    Open contour at bottom by cutting between left/right bottom points
    and placing both endpoints exactly on the bottom line.
    """
    pts = contour[:, 0, :].astype(np.int32)

    ys = pts[:, 1]
    max_y = ys.max()

    # points close to the bottom (band)
    band = np.where(ys >= max_y - 15)[0]
    if len(band) < 2:
        band = np.where(ys >= max_y - 5)[0]

    # choose leftmost and rightmost in bottom band
    xs = pts[band, 0]
    i_left = band[np.argmin(xs)]
    i_right = band[np.argmax(xs)]

    a, b = sorted([i_left, i_right])

    # open contour between them
    open_pts = np.vstack([pts[b:], pts[:a + 1]])

    bottom_y = height - 1 + bleed

    open_pts[0, 1] = bottom_y
    open_pts[-1, 1] = bottom_y

    return open_pts.reshape(-1, 1, 2)


def anchor_contour_to_bottom(contour: np.ndarray, height: int) -> np.ndarray:
    """
    Move contour so its lowest point sits exactly on the canvas bottom.
    """
    pts = contour[:, 0, :]

    lowest_y = pts[:,1].max()
    shift = (height - 1) - lowest_y

    pts[:,1] = pts[:,1] + shift

    return contour


def read_upload_to_rgba(
    upload: UploadFile,
    max_dimension: int = MAX_DIMENSION,
) -> np.ndarray:
    """
    Read uploaded image as RGBA using OpenCV, then auto-resize if needed.
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
    rgba = resize_if_needed_rgba(rgba, max_dimension=max_dimension)

    return rgba


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
    """
    Closed moving-average smoothing on contour points.
    points shape: (N, 2)
    """
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
    epsilon_ratio: float = 0.00045,
    smooth_window: int = 9,
) -> np.ndarray:
    """
    1) Find contour
    2) Smooth contour
    3) Simplify contour AFTER smoothing
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contour found")

    largest = max(contours, key=cv2.contourArea)
    points = largest[:, 0, :].astype(np.float32)

    # Smooth first
    smoothed = smooth_contour_points(points, smooth_window=smooth_window)

    smoothed_contour = np.round(smoothed).astype(np.int32).reshape(-1, 1, 2)

    # Simplify after smoothing
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

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    pts = contour.copy().astype(np.int32)
    pts[:, 0, 0] *= upscale
    pts[:, 0, 1] *= upscale

    cv2.polylines(
        canvas,
        [pts],
        isClosed=False,
        color=(0, 0, 0),
        thickness=max(1, int(round(thickness * upscale))),
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
    stroke_width: float = 2.0,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> str:
    if crop_to_subject:
        contour, width, height = crop_contour_to_subject(contour, width, height, pad=pad)

    contour = anchor_contour_to_bottom(contour, height)
    contour = open_contour_at_bottom(contour, height=height, bleed=0)

    pts = contour[:, 0, :]

    if len(pts) < 3:
        raise ValueError("Contour too small")

    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    d = []

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        mx, my = midpoint(p0, p1)

        if i == 0:
            d.append(f"M {p0[0]:.2f} {p0[1]:.2f}")
            d.append(f"Q {p0[0]:.2f} {p0[1]:.2f} {mx:.2f} {my:.2f}")
        else:
            d.append(f"Q {p0[0]:.2f} {p0[1]:.2f} {mx:.2f} {my:.2f}")

    path = " ".join(d)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
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


# --------------------------------------------------
# API
# --------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file: UploadFile = File(...),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00045, ge=0.00005, le=0.02),
    smooth_window: int = Query(9, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = read_upload_to_rgba(file, max_dimension=max_dimension)
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
    epsilon_ratio: float = Query(0.00045, ge=0.00005, le=0.02),
    smooth_window: int = Query(9, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
):
    try:
        rgba = read_upload_to_rgba(file, max_dimension=max_dimension)

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
    epsilon_ratio: float = Query(0.00045, ge=0.00005, le=0.02),
    smooth_window: int = Query(9, ge=3, le=51),
    stroke_width: float = Query(3.5, ge=0.5, le=12.0),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = read_upload_to_rgba(file, max_dimension=max_dimension)
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


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
from reportlab.lib.units import mm
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import tempfile


def generate_poster_pdf(svg_string: str, name: str) -> bytes:

    width, height = A3

    tmp_svg = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
    tmp_svg.write(svg_string.encode("utf-8"))
    tmp_svg.close()

    drawing = svg2rlg(tmp_svg.name)

    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(buffer.name, pagesize=A3)

    # baggrund
    c.setFillColorRGB(0.95, 0.93, 0.90)
    c.rect(0, 0, width, height, fill=1)

    # navn
    c.setFillColorRGB(0,0,0)
    c.setFont("TheSeasonsBold", 35)
    c.drawCentredString(width/2, height-60, name)

        # placer silhouette - intelligent scale
    scale = min(
        (width * 0.72) / drawing.width,
        (height * 0.62) / drawing.height
    )

    drawing.width *= scale
    drawing.height *= scale

    x = (width - drawing.width) / 2
    y = (height - drawing.height) / 2 - 30

    c.saveState()
    c.translate(x, y)
    renderPDF.draw(drawing, c, 0, 0)
    c.restoreState()

    # logo svg
    logo_width = 35 * mm
    logo_height = 12 * mm
    logo_x = (width - logo_width) / 2
    logo_y = 18 * mm

    draw_svg_on_pdf(
        c,
        "assets/avart-logo.svg",
        logo_x,
        logo_y,
        logo_width,
        logo_height,
    )

    c.showPage()
    c.save()

    with open(buffer.name, "rb") as f:
        pdf_bytes = f.read()

    return pdf_bytes


@app.post("/poster/pdf")
async def poster_pdf(
    file: UploadFile = File(...),
    name: str = Query("Clara & Ellinor"),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00045, ge=0.00005, le=0.02),
    smooth_window: int = Query(9, ge=3, le=51),
    stroke_width: float = Query(3.5, ge=0.5, le=12.0),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    try:
        rgba = read_upload_to_rgba(file, max_dimension=max_dimension)
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
            contour=contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        pdf_bytes = generate_poster_pdf(svg, name)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="poster.pdf"'},
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
