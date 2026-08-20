import io
import os
import tempfile
from typing import Annotated
import pymupdf

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

pdfmetrics.registerFont(
    TTFont("The Seasons Bold", "assets/The Seasons Bold.ttf")
)

# ----------------------------------
# AVART DESIGN SETTINGS
# ----------------------------------

BG_COLOR = (0.95, 0.93, 0.90)

PAGE_W_MM = 500
PAGE_H_MM = 700

TOP_BAND_MM = 115

TITLE_FONT_SIZE = 28

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


# ------------------------------------------------
# Optional custom font
# ------------------------------------------------

TITLE_FONT = "Helvetica-Bold"

try:
    pdfmetrics.registerFont(
        TTFont("The Seasons Bold", "assets/The Seasons Bold.ttf")
    )
    TITLE_FONT = "The Seasons Bold"

except Exception as e:
    print(f"Could not load custom font: {e}")


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
    scale_level: int = 0,
) -> bytes:
    width = PAGE_W_MM * mm
    height = PAGE_H_MM * mm

    # ------------------------------------------------
    # Format-specifikke værdier
    # ------------------------------------------------

    if PAGE_W_MM >= 590:       # A1: 594 × 841 mm
        title_font_size = 55
        logo_width_mm = 60

    elif PAGE_W_MM >= 500:     # 500 × 700 mm
        title_font_size = 45
        logo_width_mm = 50

    elif PAGE_W_MM >= 420:     # A2: 420 × 594 mm
        title_font_size = 35
        logo_width_mm = 40

    else:                      # A3: 297 × 420 mm
        title_font_size = 25
        logo_width_mm = 30

    top_band_h = TOP_BAND_MM * mm
    logo_width = logo_width_mm * mm
    logo_bottom = LOGO_BOTTOM_MM * mm

    # ------------------------------------------------
    # Midlertidig SVG-fil
    # ------------------------------------------------

    tmp_svg = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".svg",
    )

    try:
        tmp_svg.write(svg_string.encode("utf-8"))
        tmp_svg.close()

        drawing = svg2rlg(tmp_svg.name)

        if drawing is None:
            raise ValueError(
                "Could not convert silhouette SVG to drawing"
            )

        # ------------------------------------------------
        # Opret PDF
        # ------------------------------------------------

        buffer = io.BytesIO()

        c = canvas.Canvas(
            buffer,
            pagesize=(width, height),
        )

        # Baggrund
        c.setFillColorRGB(*BG_COLOR)
        c.rect(
            0,
            0,
            width,
            height,
            fill=1,
            stroke=0,
        )

        # ------------------------------------------------
        # Titel
        # ------------------------------------------------

        c.setFillColorRGB(0, 0, 0)
        c.setFont(
            TITLE_FONT,
            title_font_size,
        )

        c.drawCentredString(
            width / 2,
            height - (top_band_h / 2),
            name,
        )

        # ------------------------------------------------
        # Silhuettens oprindelige mål
        # ------------------------------------------------

        min_x, min_y, max_x, max_y = drawing.getBounds()

        raw_w = max_x - min_x
        raw_h = max_y - min_y

        if raw_w <= 0 or raw_h <= 0:
            raise ValueError(
                "Silhouette has invalid dimensions"
            )

        # ------------------------------------------------
        # Størrelsesniveauer
        # ------------------------------------------------
        # 0 = standard
        # +2 = maksimalt op til 110 mm fra toppen
        # -2 = mindre / længere nede
        #
        # Personen forbliver ALTID forankret i bunden.

        if orientation == "landscape":
            # 700 × 500 mm
            top_positions_mm = {
                -2: 180,
                -1: 150,
                 0: 120,
                 1: 105,
                 2: 90,
            }

        else:
            # 500 × 700 mm
            top_positions_mm = {
                -2: 210,
                -1: 185,
                 0: 160,
                 1: 135,
                 2: 110,
            }

        if scale_level not in top_positions_mm:
            raise ValueError(
                "scale_level must be between -2 and 2"
            )

        target_top_mm = top_positions_mm[scale_level]

        # Hvor høj skal silhuetten være fra bund til ønsket top?
        target_height = height - (target_top_mm * mm)

        # Beregn skalering ud fra hele silhuettens højde
        silhouette_scale = target_height / raw_h

        if silhouette_scale <= 0:
            raise ValueError(
                "Invalid silhouette scale"
            )

        drawing.scale(
            silhouette_scale,
            silhouette_scale,
        )

        # Bevar korrekt stroke efter skalering
        set_stroke_width_recursive(
            drawing,
            stroke_width / silhouette_scale,
        )

        # ------------------------------------------------
        # Nye mål efter skalering
        # ------------------------------------------------

        min_x, min_y, max_x, max_y = drawing.getBounds()

        draw_w = max_x - min_x

        # Centrer vandret
        x = (width - draw_w) / 2 - min_x

        # ALTID fast i bunden
        y = -min_y

        # ------------------------------------------------
        # Tegn silhuetten
        # ------------------------------------------------

        c.saveState()
        c.translate(x, y)

        renderPDF.draw(
            drawing,
            c,
            0,
            0,
        )

        c.restoreState()

        # ------------------------------------------------
        # Logo
        # ------------------------------------------------

        logo_path = "assets/avart-logo.svg"

        if os.path.exists(logo_path):
            logo = svg2rlg(logo_path)

            if logo is not None and logo.width > 0:
                logo_scale = logo_width / logo.width

                logo.scale(
                    logo_scale,
                    logo_scale,
                )

                (
                    logo_min_x,
                    logo_min_y,
                    logo_max_x,
                    logo_max_y,
                ) = logo.getBounds()

                rendered_logo_width = (
                    logo_max_x - logo_min_x
                )

                logo_x = (
                    width - rendered_logo_width
                ) / 2 - logo_min_x

                logo_y = (
                    logo_bottom - logo_min_y
                )

                renderPDF.draw(
                    logo,
                    c,
                    logo_x,
                    logo_y,
                )

        c.showPage()
        c.save()

        return buffer.getvalue()

    finally:
        try:
            tmp_svg.close()
        except Exception:
            pass

        try:
            os.unlink(tmp_svg.name)
        except Exception:
            pass


def generate_multi_poster_pdf(
    persons: list,
    name: str,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    orientation: str = "portrait",
    style: str = "beige_stroke",
) -> bytes:

    # ------------------------------------------------
    # SIDEFORMAT / ORIENTATION
    # ------------------------------------------------

    if orientation == "landscape":
        page_w_mm = max(PAGE_W_MM, PAGE_H_MM)
        page_h_mm = min(PAGE_W_MM, PAGE_H_MM)
    else:
        page_w_mm = min(PAGE_W_MM, PAGE_H_MM)
        page_h_mm = max(PAGE_W_MM, PAGE_H_MM)

    width = page_w_mm * mm
    height = page_h_mm * mm

    # ------------------------------------------------
    # FORMAT
    # ------------------------------------------------

    if PAGE_W_MM >= 590:       # A1
        title_font_size = 55
        logo_width_mm = 60

    elif PAGE_W_MM >= 500:     # 500 × 700
        title_font_size = 45
        logo_width_mm = 50

    elif PAGE_W_MM >= 420:     # A2
        title_font_size = 35
        logo_width_mm = 40

    else:                      # A3
        title_font_size = 25
        logo_width_mm = 30

    top_band_h = TOP_BAND_MM * mm
    logo_width = logo_width_mm * mm
    logo_bottom = LOGO_BOTTOM_MM * mm



    # ------------------------------------------------
    # HØJDENIVEAUER
    # ------------------------------------------------
    # Portrait 500 × 700:
    #   niveau 0 = 160 mm fra toppen
    #   niveau +2 = 110 mm fra toppen
    #
    # Landscape 700 × 500:
    #   niveau 0 = 120 mm fra toppen
    #   niveau +2 = 90 mm fra toppen
    #
    # Silhuetten er ALTID forankret i bunden.

    if orientation == "landscape":
        top_positions_mm = {
        -2: 180,
        -1: 150,
         0: 120,
         1: 105,
         2: 90,
        }
    else:
        top_positions_mm = {
        -2: 210,
        -1: 185,
         0: 160,
         1: 135,
         2: 110,
        }


    # ------------------------------------------------
    # ANTAL PERSONER
    # ------------------------------------------------

       # ------------------------------------------------
    # ANTAL PERSONER / VANDRET GRID
    # ------------------------------------------------

    count = len(persons)

    if count < 1 or count > 6:
        raise ValueError(
            "This version supports 1 to 6 persons"
        )

    # ------------------------------------------------
    # PORTRAIT: 1-2 personer
    # ------------------------------------------------

    if orientation == "portrait":

        if count == 1:
            center_positions = [
                0.50,
            ]

        elif count == 2:
            center_positions = [
                0.375,
                0.625,
            ]

        else:
            raise ValueError(
                "Portrait supports max 2 persons"
            )

    # ------------------------------------------------
    # LANDSCAPE: 3-6 personer
    # ------------------------------------------------

    else:

        side_margins_mm = {
            3: 125,
            4: 90,
            5: 50,
            6: 50,
        }

        if count not in side_margins_mm:
            raise ValueError(
                "Landscape supports 3 to 6 persons"
            )

        side_margin = (
            side_margins_mm[count] * mm
        )

        # Fordel personernes centre jævnt mellem
        # venstre og højre designmargin.
        #
        # Vi bruger marginen som gruppens ydre ramme.
        # Den individuelle side-sikkerhed længere nede
        # beskytter stadig mod at hår osv. bliver skåret af.

        usable_width = (
            width - (2 * side_margin)
        )

        if count == 1:
            center_positions = [
                0.50,
            ]

        else:
            center_positions = []

            for i in range(count):

                position_x = (
                    side_margin
                    + (
                        usable_width
                        * i
                        / (count - 1)
                    )
                )

                center_positions.append(
                    position_x / width
                )

    # ------------------------------------------------
    # OPRET PDF
    # ------------------------------------------------

    buffer = io.BytesIO()

    c = canvas.Canvas(
        buffer,
        pagesize=(width, height),
    )

    # Baggrund
    c.setFillColorRGB(*BG_COLOR)
    c.rect(
        0,
        0,
        width,
        height,
        fill=1,
        stroke=0,
    )

    # ------------------------------------------------
    # TITEL
    # ------------------------------------------------

    c.setFillColorRGB(0, 0, 0)

    c.setFont(
        TITLE_FONT,
        title_font_size,
    )

    c.drawCentredString(
        width / 2,
        height - (top_band_h / 2),
        name,
    )

    temporary_files = []

    try:

        # ------------------------------------------------
        # TEGN PERSONER
        # ------------------------------------------------

        for index, person in enumerate(persons):

            svg_string = person["svg"]
            scale_level = person.get("scale_level", 0)

            if scale_level not in top_positions_mm:
                raise ValueError(
                    "scale_level must be between -2 and 2"
                )

            # Midlertidig SVG
            tmp_svg = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".svg",
            )

            tmp_svg.write(
                svg_string.encode("utf-8")
            )

            tmp_svg.close()

            temporary_files.append(
                tmp_svg.name
            )

            drawing = svg2rlg(
                tmp_svg.name
            )

            if drawing is None:
                raise ValueError(
                    f"Could not convert person {index + 1} SVG"
                )

            # --------------------------------------------
            # Oprindelige mål
            # --------------------------------------------

            min_x, min_y, max_x, max_y = drawing.getBounds()

            raw_w = max_x - min_x
            raw_h = max_y - min_y

            if raw_w <= 0 or raw_h <= 0:
                raise ValueError(
                    f"Person {index + 1} has invalid dimensions"
                )

            # --------------------------------------------
            # HØJDE
            # --------------------------------------------

            target_top_mm = top_positions_mm[
                scale_level
            ]

            target_height = (
                height - (target_top_mm * mm)
            )

            silhouette_scale = (
                target_height / raw_h
            )

            drawing.scale(
                silhouette_scale,
                silhouette_scale,
            )

            # Bevar stregtykkelsen
            set_stroke_width_recursive(
                drawing,
                stroke_width / silhouette_scale,
            )

            # --------------------------------------------
            # Nye bounds
            # --------------------------------------------

            min_x, min_y, max_x, max_y = drawing.getBounds()

            draw_w = max_x - min_x


            # --------------------------------------------
            # VANDRET PLACERING
            # --------------------------------------------

            if orientation == "landscape":

                # ----------------------------------------
                # 3 PERSONER
                # ----------------------------------------
                # Yderpersonernes yderkant følger
                # designmarginen. Midterpersonen centreres.

                if count == 3:

                    if index == 0:
                        x = side_margin - min_x

                    elif index == 1:
                        target_center_x = width / 2

                        x = (
                            target_center_x
                            - (draw_w / 2)
                            - min_x
                        )

                    else:
                        x = (
                            width
                            - side_margin
                            - max_x
                        )

                # ----------------------------------------
                # 4–6 PERSONER
                # ----------------------------------------

                else:

                    usable_width = (
                        width - (2 * side_margin)
                    )

                    step = (
                        usable_width / (count - 1)
                    )

                    target_center_x = (
                        side_margin
                        + (step * index)
                    )

                    x = (
                        target_center_x
                        - (draw_w / 2)
                        - min_x
                    )

            else:

                # ----------------------------------------
                # PORTRAIT: 1–2 PERSONER
                # ----------------------------------------

                target_center_x = (
                    width
                    * center_positions[index]
                )

                x = (
                    target_center_x
                    - (draw_w / 2)
                    - min_x
                )
      

            # --------------------------------------------
            # SIKKERHED MOD BESKÅRING I SIDERNE
            # --------------------------------------------

            side_margin_safety = 10 * mm

            left_edge = x + min_x
            right_edge = x + max_x

            if left_edge < side_margin_safety:
                x += side_margin_safety - left_edge

            right_edge = x + max_x

            if right_edge > width - side_margin_safety:
                x -= right_edge - (width - side_margin_safety)

            # --------------------------------------------
            # ALTID FAST I BUNDEN
            # --------------------------------------------

            y = -min_y
            


            # --------------------------------------------
            # TEGN
            # --------------------------------------------

            c.saveState()

            c.translate(
                x,
                y,
            )

            renderPDF.draw(
                drawing,
                c,
                0,
                0,
            )

            c.restoreState()

        # ------------------------------------------------
        # LOGO
        # ------------------------------------------------

        logo_path = "assets/avart-logo.svg"

        if os.path.exists(logo_path):

            logo = svg2rlg(
                logo_path
            )

            if logo is not None and logo.width > 0:

                # ----------------------------------------
                # LOGOFARVE EFTER STYLE
                # ----------------------------------------

                if style in (
                    "taupe_stroke",
                    "burn_stroke",
                    "dark_stroke",
                ):
                    logo_color = colors.white
                else:
                    logo_color = colors.black

                def recolor_logo(node):

                    if (
                        hasattr(node, "fillColor")
                        and node.fillColor is not None
                    ):
                        node.fillColor = logo_color

                    if (
                        hasattr(node, "strokeColor")
                        and node.strokeColor is not None
                    ):
                        node.strokeColor = logo_color

                    if hasattr(node, "contents"):
                        for child in node.contents:
                            recolor_logo(child)

                recolor_logo(logo)

                # ----------------------------------------
                # SKALÉR LOGO
                # ----------------------------------------

                logo_scale = (
                    logo_width / logo.width
                )

                logo.scale(
                    logo_scale,
                    logo_scale,
                )

                (
                    logo_min_x,
                    logo_min_y,
                    logo_max_x,
                    logo_max_y,
                ) = logo.getBounds()

                rendered_logo_width = (
                    logo_max_x - logo_min_x
                )

                logo_x = (
                    width - rendered_logo_width
                ) / 2 - logo_min_x

                logo_y = (
                    logo_bottom - logo_min_y
                )

                # ----------------------------------------
                # TEGN LOGO
                # ----------------------------------------

                renderPDF.draw(
                    logo,
                    c,
                    logo_x,
                    logo_y,
                )

        c.showPage()
        c.save()

        return buffer.getvalue()

    finally:

        for temp_path in temporary_files:

            try:
                os.unlink(temp_path)

            except Exception:
                pass


    
# --------------------------------------------------
# API
# --------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file1: UploadFile = File(...),
    file2: UploadFile | None = File(None),
    file3: UploadFile | None = File(None),
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
        rgba = remove_background_if_needed(file1, max_dimension=max_dimension)
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
    file1: UploadFile = File(...),
    file2: UploadFile | None = File(None),
    file3: UploadFile | None = File(None),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(1, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.00020, ge=0.00005, le=0.02),
    smooth_window: int = Query(13, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
):
    try:
        rgba = remove_background_if_needed(file1, max_dimension=max_dimension)

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
        rgba = remove_background_if_needed(file1, max_dimension=max_dimension)
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



# ------------------------------------------------
# FÆLLES POSTER-BYGGER
# ------------------------------------------------

def build_poster_pdf(
    files,
    name,
    max_dimension,
    alpha_threshold,
    smooth,
    epsilon_ratio,
    smooth_window,
    stroke_width,
    crop_to_subject,
    pad,
):

    def process_person(file):

        rgba = remove_background_if_needed(
            file,
            max_dimension=max_dimension,
        )

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

        head_width = estimate_head_width(contour)

        svg = contour_to_svg(
            contour=contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return {
            "svg": svg,
            "head_width": head_width,
        }

    # ------------------------------------------------
    # 1–6 PERSONER
    # ------------------------------------------------

    if len(files) < 1 or len(files) > 6:
        raise ValueError(
            "Upload between 1 and 6 images"
        )

    persons = []

    for file in files:

        person = process_person(file)

        persons.append(
            {
                "svg": person["svg"],
                "head_width": person["head_width"],
                "scale_level": 0,
            }
        )

    # ------------------------------------------------
    # AUTOMATISK FORMAT
    # ------------------------------------------------

    person_count = len(persons)

    if person_count <= 2:
        orientation = "portrait"
    else:
        orientation = "landscape"

    # ------------------------------------------------
    # GENERER PDF
    # ------------------------------------------------

    pdf_bytes = generate_multi_poster_pdf(
    persons=persons,
    name=name,
    stroke_width=DEFAULT_STROKE_WIDTH,
    orientation=orientation,
    style=style,
    )

    return pdf_bytes


# ====================================================
# PROCESS – BILLEDER → SVG/PERSON-DATA
# ====================================================

@app.post("/poster/process")
async def poster_process(

    files: Annotated[
        list[UploadFile],
        File(description="Upload 1 to 6 billeder")
    ],

    max_dimension: int = Query(
        1200,
        ge=600,
        le=2000,
    ),

    alpha_threshold: int = Query(
        1,
        ge=0,
        le=255,
    ),

    smooth: bool = Query(True),

    epsilon_ratio: float = Query(
        0.00020,
        ge=0.00005,
        le=0.02,
    ),

    smooth_window: int = Query(
        13,
        ge=3,
        le=51,
    ),

    stroke_width: float = Query(
        3.5,
        ge=0.5,
        le=12.0,
    ),

    crop_to_subject: bool = Query(True),

    pad: int = Query(
        30,
        ge=0,
        le=300,
    ),

):
    try:

        if len(files) < 1 or len(files) > 6:
            raise ValueError(
                "Upload between 1 and 6 images"
            )

        persons = []

        for index, file in enumerate(files):

            rgba = remove_background_if_needed(
                file,
                max_dimension=max_dimension,
            )

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

            head_width = estimate_head_width(
                contour
            )

            svg = contour_to_svg(
                contour=contour,
                width=w,
                height=h,
                stroke_width=stroke_width,
                crop_to_subject=crop_to_subject,
                pad=pad,
            )

            persons.append(
                {
                    "index": index,
                    "svg": svg,
                    "head_width": head_width,
                    "scale_level": 0,
                    "flip": False,
                }
            )

        return JSONResponse(
            {
                "count": len(persons),
                "persons": persons,
            }
        )

    except Exception as e:

        return JSONResponse(
            {"error": str(e)},
            status_code=400,
        )


@app.post("/poster/render")
async def poster_render(
    data: dict,
):
    try:

        # ---------------------------------------------
        # DATA FRA /poster/process
        # ---------------------------------------------

        persons_data = data.get("persons", [])

        if len(persons_data) < 1 or len(persons_data) > 6:
            raise ValueError(
                "Render requires between 1 and 6 persons"
            )

        name = data.get(
            "name",
            "",
        )

        # ---------------------------------------------
        # STYLE
        # ---------------------------------------------

        style = data.get(
            "style",
            "beige_stroke",
        )

        allowed_styles = {
            "taupe_stroke",
            "beige_stroke",
            "burn_stroke",
            "dark_stroke",
            "beige_block",
            "grey_block",
        }

        if style not in allowed_styles:
            raise ValueError(
                f"Unknown poster style: {style}"
            )

        # ---------------------------------------------
        # BYG PERSON-LISTE
        # ---------------------------------------------

        persons = []

        for person in persons_data:

            svg = person.get("svg")

            if not svg:
                raise ValueError(
                    "Person is missing SVG data"
                )

            persons.append(
                {
                    "svg": svg,
                    "head_width": person.get(
                        "head_width",
                        0,
                    ),
                    "scale_level": person.get(
                        "scale_level",
                        0,
                    ),
                }
            )

        
        # ---------------------------------------------
        # FORMAT
        # ---------------------------------------------

        if len(persons) <= 2:
            orientation = "portrait"
        else:
            orientation = "landscape"

        # ---------------------------------------------
        # GENERER POSTER FRA DE ALLEREDE
        # BEHANDLEDE SVG'ER
        # ---------------------------------------------

        pdf_bytes = generate_multi_poster_pdf(
            persons=persons,
            name=name,
            stroke_width=DEFAULT_STROKE_WIDTH,
            orientation=orientation,
        )

        # ---------------------------------------------
        # PDF → PNG PREVIEW
        # ---------------------------------------------

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        page = document[0]

        pixmap = page.get_pixmap(
            dpi=150,
            alpha=False,
        )

        png_bytes = pixmap.tobytes("png")

        document.close()

        # ---------------------------------------------
        # SEND PNG TIL WEBSITE
        # ---------------------------------------------

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except Exception as e:

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=400,
        )


# ====================================================
# PDF – ENDELIG POSTER
# ====================================================

@app.post("/poster/pdf")
async def poster_pdf(

    files: Annotated[
        list[UploadFile],
        File(description="Upload 1 to 6 billeder")
    ],

    name: str = Query("Mine dejlige børnebørn"),

    max_dimension: int = Query(
        MAX_DIMENSION,
        ge=600,
        le=3000,
    ),

    alpha_threshold: int = Query(
        1,
        ge=0,
        le=255,
    ),

    smooth: bool = Query(True),

    epsilon_ratio: float = Query(
        0.00020,
        ge=0.00005,
        le=0.02,
    ),

    smooth_window: int = Query(
        13,
        ge=3,
        le=51,
    ),

    stroke_width: float = Query(
        3.5,
        ge=0.5,
        le=12.0,
    ),

    crop_to_subject: bool = Query(True),

    pad: int = Query(
        30,
        ge=0,
        le=300,
    ),

):
    try:

        pdf_bytes = build_poster_pdf(
            files=files,
            name=name,
            max_dimension=max_dimension,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                f'attachment; filename="{name}.pdf"'
            },
        )

    except Exception as e:

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=400,
        )


# ====================================================
# PNG – PREVIEW TIL HJEMMESIDEN
# ====================================================

@app.post("/poster/preview")
async def poster_preview(

    files: Annotated[
        list[UploadFile],
        File(description="Upload 1 to 6 billeder")
    ],

    name: str = Query("Mine dejlige børnebørn"),

    max_dimension: int = Query(
        MAX_DIMENSION,
        ge=600,
        le=3000,
    ),

    alpha_threshold: int = Query(
        1,
        ge=0,
        le=255,
    ),

    smooth: bool = Query(True),

    epsilon_ratio: float = Query(
        0.00020,
        ge=0.00005,
        le=0.02,
    ),

    smooth_window: int = Query(
        13,
        ge=3,
        le=51,
    ),

    stroke_width: float = Query(
        3.5,
        ge=0.5,
        le=12.0,
    ),

    crop_to_subject: bool = Query(True),

    pad: int = Query(
        30,
        ge=0,
        le=300,
    ),

):
    try:

        # Lav præcis samme poster som PDF-versionen
        pdf_bytes = build_poster_pdf(
            files=files,
            name=name,
            max_dimension=max_dimension,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
            stroke_width=stroke_width,
            crop_to_subject=crop_to_subject,
            pad=pad,
        )

        # ------------------------------------------------
        # PDF → PNG PREVIEW
        # ------------------------------------------------

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        page = document[0]

        pixmap = page.get_pixmap(
            dpi=150,
            alpha=False,
        )

        png_bytes = pixmap.tobytes("png")

        document.close()

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except Exception as e:

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=400,
        )
