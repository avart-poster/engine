import io
import os
import tempfile
import stripe
import pymysql
import json
import secrets
import subprocess
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape as html_escape
from urllib.request import urlopen
from urllib.parse import quote

from typing import Annotated
import pymupdf
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, Query, Request
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


# ==========================================
# AVART – CENTRALE FARVER
# ==========================================

AVART_COLORS = {
    "moss": {
        "CMYK": (25, 0, 50, 75),
        "RGB": (68, 72, 50),
    },
    "wine": {
        "CMYK": (50, 75, 75, 50),
        "RGB": (93, 53, 42),
    },
    "sea": {
        "CMYK": (75, 25, 25, 65),
        "RGB": (26, 64, 78),
    },
    "camel": {
        "CMYK": (0, 25, 50, 50),
        "RGB": (152, 125, 88),
    },
    "dark": {
        "CMYK": (20, 20, 20, 100),
        "RGB": (0, 0, 0),
    },
    "light": {
        "CMYK": (0, 0, 0, 0),
        "RGB": (251, 250, 247),
    },
}


def avart_cmyk(style):
    from reportlab.lib import colors

    c, m, y, k = AVART_COLORS[style]["CMYK"]

    return colors.CMYKColor(
        c / 100,
        m / 100,
        y / 100,
        k / 100,
    )

# --------------------------------------------------
# AVART DATABASE CONNECTION
# --------------------------------------------------

def get_db_connection():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        autocommit=False,
    )



register_heif_opener()

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

DEFAULT_STROKE_WIDTH = 5

MAX_DIMENSION = 1400
ALPHA_THRESHOLD = 128
SMOOTH_WINDOW = 19
EPSILON_RATIO = 0.00020

REMBG_MODEL = "isnet-general-use"


# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="avart-engine",
    version="1.0.0",
    description="Avart silhouette engine with SVG + PDF poster output",
)

# Kun vores egne domæner må kalde motoren fra en browser.
# Kan ændres på Render med ALLOWED_ORIGINS (kommasepareret).
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://avart.dk,https://www.avart.dk",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# SIKKERHED: TEST-ENDPOINTS OG SVG-SIGNATUR
# --------------------------------------------------

import hmac
import hashlib
from fastapi import HTTPException

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Test-/debug-endpoints er slået fra, medmindre ENABLE_TEST_ENDPOINTS=1 på Render.
TEST_ENDPOINTS_ENABLED = os.environ.get("ENABLE_TEST_ENDPOINTS") == "1"


def require_test_endpoints():
    if not TEST_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")


# Hemmelig nøgle, som kun serveren kender (sæt SVG_SIGNING_SECRET på Render).
SVG_SIGNING_SECRET = os.environ.get("SVG_SIGNING_SECRET", "")

if not SVG_SIGNING_SECRET:
    print("ADVARSEL: SVG_SIGNING_SECRET mangler – silhuetter bliver ikke kontrolleret.", flush=True)


def sign_svg(svg: str) -> str:
    if not SVG_SIGNING_SECRET:
        return ""
    return hmac.new(
        SVG_SIGNING_SECRET.encode("utf-8"),
        svg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_svg(svg, signature) -> bool:
    if not SVG_SIGNING_SECRET:
        return True
    if not isinstance(svg, str) or not isinstance(signature, str):
        return False
    return hmac.compare_digest(sign_svg(svg), signature)


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
        sess_opts = ort.SessionOptions()

        # Reducer ONNX Runtime's RAM-forbrug
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False

        # Vi behandler billeder sekventielt
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Begræns CPU-tråde og deres ekstra hukommelse
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        _rembg_session = new_session(
            REMBG_MODEL,
            sess_opts=sess_opts,
        )

    return _rembg_session



# --------------------------------------------------
# detect face
# --------------------------------------------------


def detect_face_region(original_rgb: np.ndarray):
    """
    Finder ansigt med OpenCV YuNet.
    Returnerer (x, y, w, h) eller None.
    """

    h, w = original_rgb.shape[:2]

    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "face_detection_yunet_2023mar.onnx",
    )

    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

    detector.setInputSize((w, h))

    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda f: f[2] * f[3],
    )

    x, y, fw, fh = face[:4]

    return (
        int(x),
        int(y),
        int(fw),
        int(fh),
    )

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}

@app.get("/health/database")
def database_health():
    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()

        connection.close()

        return {
            "ok": True,
            "database": "connected"
        }

    except Exception as e:
        print(
            f"DATABASE CONNECTION ERROR: {type(e).__name__}: {e}",
            flush=True
        )
        return JSONResponse(
            {
                "ok": False,
                "database": "connection_failed"
            },
            status_code=503,
        )


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


def remove_background_if_needed(
    upload: UploadFile,
    max_dimension: int = MAX_DIMENSION,
    return_original: bool = False,
    noir_test: bool = False,
):
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise ValueError("Empty file")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Billedet er for stort. Maks {MAX_UPLOAD_MB} MB pr. billede."
        )

    # --------------------------------------------------
    # ÅBN BILLEDE MED KORREKT ORIENTERING
    # JPEG / PNG / WEBP / HEIC / HEIF
    # Bevar eventuel transparens
    # --------------------------------------------------
    
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image.load()
    
        # Anvend EXIF-orientering, så billedet
        # ikke drejes forkert under behandlingen.
        pil_image = ImageOps.exif_transpose(pil_image)
    
        if "A" in pil_image.getbands() or "transparency" in pil_image.info:
    
            pil_image = pil_image.convert("RGBA")
    
            img = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGBA2BGRA,
            )
    
        else:
    
            pil_image = pil_image.convert("RGB")
    
            img = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR,
            )
    
    except Exception as e:
    
        raise ValueError(
            f"Could not decode image: {e}"
        )

    # --------------------------------------------------
    # GEM ORIGINALEN FØR REMBG
    # --------------------------------------------------

    if len(img.shape) == 3 and img.shape[2] == 4:
        original_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2RGB,
        )
    else:
        original_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

    # --------------------------------------------------
    # HVIS UPLOAD ALLEREDE HAR ÆGTE TRANSPARENCY
    # --------------------------------------------------

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]

        if np.any(alpha < 250):
            rgba = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2RGBA,
            )

            rgba = resize_if_needed_rgba(
                rgba,
                max_dimension=max_dimension,
            )

            original_rgb = cv2.resize(
                original_rgb,
                (rgba.shape[1], rgba.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

            rgba = cv2.copyMakeBorder(
                rgba,
                0,
                180,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0),
            )

            original_rgb = cv2.copyMakeBorder(
                original_rgb,
                0,
                180,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            if return_original:
                return rgba, original_rgb

            return rgba

    # --------------------------------------------------
    # RESIZE FØR REMBG
    # --------------------------------------------------

    max_input_size = 1600

    h, w = img.shape[:2]

    scale = min(
        1.0,
        max_input_size / max(h, w),
    )

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        original_rgb = cv2.resize(
            original_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

    # --------------------------------------------------
    # REMBG – ORIGINALBILLEDET
    # --------------------------------------------------

    # TEST: Noir-inspireret behandling før REMBG
    # Aktiveres kun ved noir_test=True
    
    if noir_test:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        gray = cv2.convertScaleAbs(
            gray,
            alpha=1.4,
            beta=-20,
        )
    
        img = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        )

    # --------------------------------------------------
    # AUTOMATISK KONTRASTFORBEDRING FØR REMBG
    # --------------------------------------------------
    
    ok, buffer = cv2.imencode(
        ".png",
        img,
    )

    if not ok:
        raise ValueError(
            "Could not encode image for background removal"
        )

    output = remove(
        buffer.tobytes(),
        session=get_rembg_session(),
    )

    arr_out = np.frombuffer(
        output,
        np.uint8,
    )

    img_out = cv2.imdecode(
        arr_out,
        cv2.IMREAD_UNCHANGED,
    )

 

    if img_out is None:
        raise ValueError(
            "Background removal failed"
        )

    if len(img_out.shape) == 3 and img_out.shape[2] == 3:
        alpha = np.full(
            (
                img_out.shape[0],
                img_out.shape[1],
                1,
            ),
            255,
            dtype=np.uint8,
        )

        img_out = np.concatenate(
            [img_out, alpha],
            axis=2,
        )

    if len(img_out.shape) != 3 or img_out.shape[2] != 4:
        raise ValueError(
            "Background removal did not return RGBA"
        )

    rgba = cv2.cvtColor(
        img_out,
        cv2.COLOR_BGRA2RGBA,
    )

    # --------------------------------------------------
    # SØRG FOR SAMME STØRRELSE
    # --------------------------------------------------

    original_rgb = cv2.resize(
        original_rgb,
        (rgba.shape[1], rgba.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------
    # TRANSPARENT BUND
    # --------------------------------------------------

    rgba = cv2.copyMakeBorder(
        rgba,
        0,
        180,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )

    original_rgb = cv2.copyMakeBorder(
        original_rgb,
        0,
        180,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    # --------------------------------------------------
    # SLUT-RESIZE
    # --------------------------------------------------

    rgba = resize_if_needed_rgba(
        rgba,
        max_dimension=max_dimension,
    )

    original_rgb = cv2.resize(
        original_rgb,
        (rgba.shape[1], rgba.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    if return_original:
        return rgba, original_rgb

    return rgba

def alpha_to_mask(
    rgba: np.ndarray,
    alpha_threshold: int = 128,
    smooth: bool = True,
) -> np.ndarray:

    alpha = rgba[:, :, 3]

    mask = np.where(
        alpha > alpha_threshold,
        255,
        0
    ).astype(np.uint8)

    if smooth:

        # Luk små huller og smalle åbninger.
        # En lille kernel begrænser påvirkningen
        # af ansigt og hår.

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        # Udfyld indvendige huller.

        contours, hierarchy = cv2.findContours(
            mask.copy(),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is not None:

            filled_mask = mask.copy()

            for i, contour in enumerate(contours):

                if hierarchy[0][i][3] != -1:

                    cv2.drawContours(
                        filled_mask,
                        contours,
                        i,
                        255,
                        thickness=cv2.FILLED,
                    )

            mask = filled_mask

    return mask


def get_smoothed_outer_contour(
    mask: np.ndarray,
    epsilon_ratio: float = 0.00020,
    smooth_window: int = 7,
    face_region=None,
) -> np.ndarray:

    # --------------------------------------------------
    # FIND PERSONENS YDERKONTUR
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("No contour found")

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    # Fjern små løkker og udstikkere med en forsigtig
    # morfologisk åbning, men beskyt ansigtet.
    
    if face_region is not None:
        fx, fy, fw, fh = face_region
    
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
    
        # Luk små indhak langs yderkonturen.
        cleaned_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )
        
        # Fjern derefter små udstikkere.
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
    
        # Beskyt ansigtsprofilen uden at beskytte
        # unødvendigt meget af hårets øverste del.
        x1 = max(0, int(fx - fw * 0.2))
        y1 = max(0, int(fy))
        x2 = min(mask.shape[1], int(fx + fw * 1.2))
        y2 = min(mask.shape[0], int(fy + fh * 1.1))
    
        cleaned_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    
        cleaned_contours, _ = cv2.findContours(
            cleaned_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
    
        if cleaned_contours:
            largest = max(
                cleaned_contours,
                key=cv2.contourArea,
            )
    

    
    # --------------------------------------------------
    # BEVAR ALLE KONTURPUNKTER
    # --------------------------------------------------
    # Vi simplificerer IKKE konturen med approxPolyDP.
    # Det er vigtigt for at bevare detaljer i fx
    # næse, mund, læber, hage og hår.
    # --------------------------------------------------

    points = largest[:, 0, :].astype(np.float32)

    # --------------------------------------------------
    # MEGET MILD SMOOTHING
    # --------------------------------------------------
    # smooth_window styrer hvor meget små pixelhak
    # udjævnes.
    #
    # 3 = næsten original
    # 5 = meget mild
    # 7 = mild
    # 9+ = mere udglatning
    #
    # Vi starter forsigtigt for at bevare ansigtet.
    # --------------------------------------------------

    if smooth_window < 3:
        smooth_window = 3

    if smooth_window % 2 == 0:
        smooth_window += 1

    n = len(points)

    if n >= smooth_window:
    
        # Meget mild smoothing omkring ansigtet.
        # Bevarer flere detaljer i næse, læber og hage.
        
        face_window = 5
    
        # Resten af silhuetten glattes kraftigere.
        body_window = 31
    
        if body_window % 2 == 0:
            body_window += 1
    
        max_window = max(face_window, body_window)
        pad = max_window // 2
    
        padded = np.vstack([
            points[-pad:],
            points,
            points[:pad],
        ])
    
        smoothed = np.empty_like(points)
    
        for i in range(n):
            px, py = points[i]
    
            use_face_smoothing = False
    
            if face_region is not None:
                fx, fy, fw, fh = face_region
    
                # Gør beskyttelsesområdet lidt større end
                # selve ansigtsboksen.
                margin_x = fw * 0.15
                margin_y = fh * 0.10
    
                if (
                    fx - margin_x <= px <= fx + fw + margin_x
                    and fy - margin_y <= py <= fy + fh + margin_y
                ):
                    use_face_smoothing = True
    
            window = (
                face_window
                if use_face_smoothing
                else body_window
            )
    
            half = window // 2
            center = i + pad
    
            segment = padded[
                center - half:
                center + half + 1
            ]
    
            smoothed[i] = segment.mean(axis=0)
    
        points = smoothed

    # --------------------------------------------------
    # TILBAGE TIL OPENCV-KONTURFORMAT
    # --------------------------------------------------

    contour = np.round(
        points
    ).astype(np.int32)

    contour = contour.reshape(
        -1,
        1,
        2,
    )

    return contour

def get_secondary_outer_contours(
    mask: np.ndarray,
    main_contour: np.ndarray,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.05,
) -> list[np.ndarray]:

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return []

    main_area = cv2.contourArea(main_contour)

    if main_area <= 0:
        return []

    secondary = []

    # Find hovedkonturen fra den oprindelige maske.
    # Den udglattede hovedkontur har andre koordinater
    # og kan derfor ikke sammenlignes med np.array_equal.

    main_index = max(
        range(len(contours)),
        key=lambda i: cv2.contourArea(contours[i]),
    )

    for index, contour in enumerate(contours):

        # Spring hovedkonturen over
        if index == main_index:
            continue

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        ratio = area / main_area

        # Fjern mikroskopisk støj
        if ratio < min_area_ratio:
            continue

        # En sekundær kontur må heller ikke være
        # en stor separat genstand
        if ratio > max_area_ratio:
            continue

        secondary.append(contour)

    return secondary


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

def get_significant_inner_contours(
    mask: np.ndarray,
    min_area_ratio: float = 0.002,
) -> list[np.ndarray]:
    """
    Find betydningsfulde indvendige huller i personens maske.

    Små huller/støj ignoreres.
    Større negative rum, fx mellem hestehale og nakke,
    bevares som indvendige konturer.
    """

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    if hierarchy is None or not contours:
        return []

    hierarchy = hierarchy[0]

    outer_index = max(
        range(len(contours)),
        key=lambda i: cv2.contourArea(contours[i])
        if hierarchy[i][3] == -1
        else -1,
    )

    outer_area = cv2.contourArea(contours[outer_index])

    if outer_area <= 0:
        return []

    inner_contours = []

    for i, contour in enumerate(contours):
        parent = hierarchy[i][3]

        # Kun direkte huller inde i personens hovedkontur
        if parent != outer_index:
            continue

        area = cv2.contourArea(contour)

        # Ignorer små maskerester/støj
        if area / outer_area < min_area_ratio:
            continue

        inner_contours.append(contour)

    return inner_contours

def contour_to_svg(
    contour: np.ndarray,
    width: int,
    height: int,
    mask: np.ndarray | None = None,
    stroke_width: float = 3.5,
    crop_to_subject: bool = False,
    pad: int = 30,
) -> str:

    # --------------------------------------------------
    # GEM ORIGINAL KOORDINAT-FORSKYDNING
    # --------------------------------------------------

    crop_x1 = 0
    crop_y1 = 0

    # --------------------------------------------------
    # CROP TIL PERSON
    # --------------------------------------------------

    if crop_to_subject:

        x, y, w, h = cv2.boundingRect(contour)

        crop_x1 = max(0, x - pad)
        crop_y1 = max(0, y - pad)

        x2 = min(
            width,
            x + w + pad,
        )

        y2 = min(
            height,
            y + h + pad,
        )

        contour = contour.copy()

        contour[:, 0, 0] -= crop_x1
        contour[:, 0, 1] -= crop_y1

        if mask is not None:
            mask = mask[
                crop_y1:y2,
                crop_x1:x2
            ].copy()

        width = x2 - crop_x1
        height = y2 - crop_y1

    else:
        contour = contour.copy()

    # --------------------------------------------------
    # FORANKR HOVEDKONTUREN I BUNDEN
    # --------------------------------------------------

    lowest_y = contour[:, 0, 1].max()

    anchor_shift = (
        (height - 1)
        - lowest_y
    )

    contour[:, 0, 1] += anchor_shift

    # --------------------------------------------------
    # ÅBN HOVEDKONTUREN I BUNDEN
    # --------------------------------------------------

    contour = open_contour_at_bottom(
        contour,
        height=height,
        bleed=0,
    )

    pts = contour[:, 0, :]

    if len(pts) < 2:
        raise ValueError(
            "Contour too small"
        )

    # --------------------------------------------------
    # HOVEDKONTUR → SVG PATH. Brug bløde kurver i stedet for lige linjestykker.
    # --------------------------------------------------
   
    pts = contour[:, 0, :].astype(np.float64)
    
    if len(pts) < 3:
        raise ValueError("Contour too small")
    
    d = [
        f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    ]
    mid_x = (pts[0][0] + pts[1][0]) / 2
    mid_y = (pts[0][1] + pts[1][1]) / 2
    
    d.append(
        f"L {mid_x:.2f} {mid_y:.2f}"
    )
    for i in range(1, len(pts) - 1):
    
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
    
        # Midtpunkt mellem dette og næste konturpunkt
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
    
        d.append(
            f"Q {p1[0]:.2f} {p1[1]:.2f} "
            f"{mid_x:.2f} {mid_y:.2f}"
        )
    
    d.append(
        f"L {pts[-1][0]:.2f} {pts[-1][1]:.2f}"
    )
    
    main_path = " ".join(d)

    # --------------------------------------------------
    # EKSTRA SVG PATHS
    # --------------------------------------------------

    extra_paths = []

    # Kun hovedkonturen skal tegnes
    mask = None

    if mask is not None:

        # --------------------------------------------------
        # INDVENDIGE NEGATIVE RUM
        # Midlertidigt slået fra:
        # Avart tegner kun den ydre silhuet.
        # --------------------------------------------------
        
        inner_contours = []

        # ----------------------------------------------
        # SEKUNDÆRE YDRE KONTURER
        # fx løse krøller / hår
        # ----------------------------------------------

        main_contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if main_contours:

            mask_main_contour = max(
                main_contours,
                key=cv2.contourArea,
            )

            secondary_contours = (
                get_secondary_outer_contours(
                    mask,
                    mask_main_contour,
                )
            )

            for secondary in secondary_contours:

                secondary = secondary.copy()

                secondary[:, 0, 1] += (
                    anchor_shift
                )

                secondary_pts = (
                    secondary[:, 0, :]
                )

                if len(secondary_pts) < 2:
                    continue

                secondary_d = [
                    (
                        f"M "
                        f"{secondary_pts[0][0]:.2f} "
                        f"{secondary_pts[0][1]:.2f}"
                    )
                ]

                for p in secondary_pts[1:]:

                    secondary_d.append(
                        (
                            f"L "
                            f"{p[0]:.2f} "
                            f"{p[1]:.2f}"
                        )
                    )

                # Sekundære masker er lukkede øer
                secondary_d.append("Z")

                extra_paths.append(
                    " ".join(
                        secondary_d
                    )
                )

    # --------------------------------------------------
    # BYG SVG FOR EKSTRA PATHS
    # --------------------------------------------------

    extra_svg = "\n".join(
        (
            f'<path '
            f'd="{path}" '
            f'fill="none" '
            f'stroke="black" '
            f'stroke-width="{stroke_width}" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
        for path in extra_paths
    )

    # --------------------------------------------------
    # FÆRDIG SVG
    # --------------------------------------------------

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>

<svg xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">

  <path
    d="{main_path}"
    fill="none"
    stroke="black"
    stroke-width="{stroke_width}"
    stroke-linecap="round"
    stroke-linejoin="round"/>

{extra_svg}

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


def poster_title_baseline(page_height, margin_mm, font_name, font_size):
    """Place the capital-letter top at the green design margin."""
    face = pdfmetrics.getFont(font_name).face
    cap_height = getattr(face, "capHeight", None)
    if not cap_height or cap_height <= 0:
        cap_height = pdfmetrics.getAscent(font_name)
    return page_height - margin_mm * mm - cap_height * font_size / 1000


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

    # Text and logo follow the green margin, not the red silhouette safe area.
    green_margin_mm = {60: 65, 50: 50, 40: 45, 30: 30}[logo_width_mm]
    title_y = poster_title_baseline(height, green_margin_mm, TITLE_FONT, title_font_size)
    logo_width = logo_width_mm * mm
    logo_bottom = green_margin_mm * mm

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
            title_y,
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
            reference_height_mm = 500
        
            reference_top_positions_mm = {
                -4: 260,
                -3: 220,
                -2: 180,
                -1: 150,
                0: 120,
                1: 105,
                2: 90,
            }
        
        else:
            reference_height_mm = 700
        
            reference_top_positions_mm = {
                -4: 290,
                -3: 250,
                -2: 210,
                -1: 185,
                0: 160,
                1: 135,
                2: 110,
            }
        
        height_ratio = page_h_mm / reference_height_mm
        
        top_positions_mm = {
            level: position * height_ratio
            for level, position in reference_top_positions_mm.items()
        }
        
        if scale_level not in top_positions_mm:
            raise ValueError(
                "scale_level must be between -4 and 2"
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

        # Logo følger posterens style
        if logo is not None:
            logo_color = colors.HexColor(
                "#EEE7DC" if style == "dark" else "#2B2B2B"
            )
        
            def recolor_logo(node):
                if hasattr(node, "fillColor") and node.fillColor is not None:
                    node.fillColor = logo_color
        
                if hasattr(node, "strokeColor") and node.strokeColor is not None:
                    node.strokeColor = logo_color
        
                if hasattr(node, "contents"):
                    for child in node.contents:
                        recolor_logo(child)
        
            recolor_logo(logo)
            
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


def poster_group_layout(person_widths, available_width):
    """Use one shared width X, step X/2, and scale only the whole group.

    X is the widest height-adjusted contour's placement width. Narrower
    contours are centered in that same slot, never stretched independently.
    """
    if not person_widths or min(person_widths) <= 0 or available_width <= 0:
        raise ValueError("Person widths and available width must be positive")
    slot_width = max(person_widths)
    group_width = slot_width * (1 + (len(person_widths) - 1) / 2)
    group_scale = min(1.0, available_width / group_width)
    slot_width *= group_scale
    group_width *= group_scale
    centers = [slot_width / 2 + index * slot_width / 2
               for index in range(len(person_widths))]
    return group_scale, group_width, centers


def generate_multi_poster_pdf(
    persons: list,
    name: str,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    orientation: str = "portrait",
    style: str = "light",
    size: str = "50x70",
) -> bytes:

    # ------------------------------------------------
    # STYLE / FARVER – CMYK TIL PRINT
    # ------------------------------------------------

    style_colors = {
        "light": avart_cmyk("light"),
        "dark": avart_cmyk("dark"),
        "moss": avart_cmyk("moss"),
        "wine": avart_cmyk("wine"),
        "camel": avart_cmyk("camel"),
        "sea": avart_cmyk("sea"),
    }

    if style not in style_colors:
        raise ValueError(f"Unknown poster style: {style}")

    bg_color = style_colors[style]

    # CMYK til tekst, silhuet og logo
    if style == "light":
        fg_color = colors.CMYKColor(0, 0, 0, 1)
    else:
        fg_color = colors.CMYKColor(0, 0, 0, 0)
    

    # ------------------------------------------------
    # SIZE / FYSISK FORMAT
    # ------------------------------------------------

    size_dimensions = {
        "A3": (297, 420),
        "A2": (420, 594),
        "50x70": (500, 700),
        "A1": (594, 841),
    }

    # ------------------------------------------------
    # AVART DESIGNMARGINER
    # ------------------------------------------------
    
    size_side_margins_mm = {
        "A3": 30,
        "A2": 45,
        "50x70": 50,
        "A1": 65,
    }

    if size not in size_dimensions:
        raise ValueError(
            f"Unknown poster size: {size}"
        )

    base_w_mm, base_h_mm = size_dimensions[size]
    
    # ------------------------------------------------
    # SIDEFORMAT / ORIENTATION
    # ------------------------------------------------

    if orientation == "landscape":
        page_w_mm = max(base_w_mm, base_h_mm)
        page_h_mm = min(base_w_mm, base_h_mm)
    else:
        page_w_mm = min(base_w_mm, base_h_mm)
        page_h_mm = max(base_w_mm, base_h_mm)

    width = page_w_mm * mm
    height = page_h_mm * mm

    # ------------------------------------------------
    # FORMAT
    # ------------------------------------------------

    if max(page_w_mm, page_h_mm) >= 841:      # A1
        title_font_size = 55
        logo_width_mm = 60
    
    elif max(page_w_mm, page_h_mm) >= 700:   # 500 × 700
        title_font_size = 45
        logo_width_mm = 50
    
    elif max(page_w_mm, page_h_mm) >= 594:   # A2
        title_font_size = 35
        logo_width_mm = 40
    
    else:                      # A3
        title_font_size = 25
        logo_width_mm = 30

    # Text and logo follow the green margin. Keep the red silhouette safe area below unchanged.
    green_margin_mm = size_side_margins_mm[size]
    title_y = poster_title_baseline(height, green_margin_mm, TITLE_FONT, title_font_size)
    logo_width = logo_width_mm * mm
    logo_bottom = green_margin_mm * mm



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
        reference_height_mm = 500
    
        reference_top_positions_mm = {
            -4: 260,
            -3: 220,
            -2: 180,
            -1: 150,
            0: 120,
            1: 105,
            2: 90,
        }
    
    else:
        reference_height_mm = 700
    
        reference_top_positions_mm = {
            -4: 290,
            -3: 250,
            -2: 210,
            -1: 185,
            0: 160,
            1: 135,
            2: 110,
        }
    
    height_ratio = page_h_mm / reference_height_mm
    
    top_positions_mm = {
        level: position * height_ratio
        for level, position in reference_top_positions_mm.items()
    }

    # ------------------------------------------------
    # MAKSIMAL HØJDE – AVART DESIGNGUIDE
    # ------------------------------------------------
    
    size_top_limits_mm = {
        "A3": 65,
        "A2": 100,
        "50x70": 115,
        "A1": 140,
    }
    
    minimum_top_mm = size_top_limits_mm[size]
    
    top_positions_mm = {
        level: max(position, minimum_top_mm)
        for level, position in top_positions_mm.items()
    }


    # ------------------------------------------------
    # ANTAL PERSONER / FÆLLES VANDRET GRID
    # ------------------------------------------------

    count = len(persons)
    if count < 1 or count > 6:
        raise ValueError("This version supports 1 to 6 persons")
    if orientation == "portrait" and count > 2:
        raise ValueError("Portrait supports max 2 persons")
    if orientation != "portrait" and count < 3:
        raise ValueError("Landscape supports 3 to 6 persons")

    # ------------------------------------------------
    # OPRET PDF
    # ------------------------------------------------

    buffer = io.BytesIO()

    c = canvas.Canvas(
        buffer,
        pagesize=(width, height),
    )

    # Baggrund
    # DARK får fysisk baggrund i PDF'en.
    # LIGHT forbliver transparent.
    
    if bg_color is not None:
        c.setFillColor(bg_color)
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

    c.setFillColor(fg_color)

    c.setFont(
        TITLE_FONT,
        title_font_size,
    )

    c.drawCentredString(
        width / 2,
        title_y,
        name,
    )

    temporary_files = []
    prepared_persons = []
    person_widths = []

    try:

        # ------------------------------------------------
        # TEGN PERSONER
        # ------------------------------------------------

        for index, person in enumerate(persons):
        
            svg_string = person["svg"]
            scale_level = person.get("scale_level", 0)
            flipped = person.get("flipped", False)

            if scale_level not in top_positions_mm:
                raise ValueError(
                    "scale_level must be between -4 and 2"
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


            # ----------------------------------------
            # STROKE EFTER STYLE
            # ----------------------------------------

            def recolor_person(node):

                if (
                    hasattr(node, "strokeColor")
                    and node.strokeColor is not None
                ):
                    node.strokeColor = fg_color

                if hasattr(node, "contents"):
                    for child in node.contents:
                        recolor_person(child)

            recolor_person(drawing)

            
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

            # Prepare every height-adjusted contour before placing the group.
            prepared_persons.append((drawing, silhouette_scale, flipped))
            person_widths.append(draw_w)

        minimum_side_margin = size_side_margins_mm[size] * mm
        maximum_group_width = width - 2 * minimum_side_margin
        group_scale, group_width, group_centers = poster_group_layout(
            person_widths, maximum_group_width,
        )
        group_left = (width - group_width) / 2

        for index, (drawing, silhouette_scale, flipped) in enumerate(prepared_persons):
            if group_scale < 1:
                drawing.scale(group_scale, group_scale)
                set_stroke_width_recursive(
                    drawing, stroke_width / (silhouette_scale * group_scale),
                )

            min_x, min_y, max_x, max_y = drawing.getBounds()
            draw_w = max_x - min_x
            target_center_x = group_left + group_centers[index]
            x = target_center_x - draw_w / 2 - min_x

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
            
            if flipped:
                c.transform(
                    -1,
                    0,
                    0,
                    1,
                    min_x + max_x,
                    0,
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

                logo_color = fg_color

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
# AVART – PRINT PDF MED 3 MM BLEED OG SKÆREMÆRKER
# --------------------------------------------------
def make_print_ready_pdf(
    pdf_bytes: bytes,
    bleed_mm: float = 3,
    style: str = "light",
) -> bytes:
    """
    Tilføjer 3 mm bleed, skæremærker og PDF-beskæringsbokse.
    Bruger AVARTs centrale CMYK-farver.
    Bevarer den eksisterende PDF som vektorgrafik.
    """
    import io
    import fitz
    from reportlab.pdfgen import canvas

    if style not in AVART_COLORS:
        raise ValueError(f"Unknown poster style: {style}")

    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    output = fitz.open()

    try:
        bleed = bleed_mm * 72 / 25.4
        mark_length = 5 * 72 / 25.4
        mark_gap = 2 * 72 / 25.4

        for source_page in source:
            w = source_page.rect.width
            h = source_page.rect.height

            # Plads til bleed og skæremærker
            margin = bleed + mark_gap + mark_length

            page = output.new_page(
                width=w + 2 * margin,
                height=h + 2 * margin,
            )

            # Plakatens færdige beskæringsformat
            trim = fitz.Rect(
                margin,
                margin,
                margin + w,
                margin + h,
            )

            # Bleed-området: 3 mm på alle sider
            bleed_rect = fitz.Rect(
                trim.x0 - bleed,
                trim.y0 - bleed,
                trim.x1 + bleed,
                trim.y1 + bleed,
            )

            # ------------------------------------------
            # BAGGRUND + BLEED I KORREKT CMYK
            # ------------------------------------------

            bleed_buffer = io.BytesIO()

            bleed_canvas = canvas.Canvas(
                bleed_buffer,
                pagesize=(
                    page.rect.width,
                    page.rect.height,
                ),
                pageCompression=1,
            )

            # Hent farven fra AVART_COLORS
            bleed_canvas.setFillColor(
                avart_cmyk(style)
            )

            # Tegn baggrunden inklusive 3 mm bleed
            bleed_canvas.rect(
                bleed_rect.x0,
                page.rect.height - bleed_rect.y1,
                bleed_rect.width,
                bleed_rect.height,
                fill=1,
                stroke=0,
            )

            bleed_canvas.save()

            # Indsæt CMYK-baggrunden i PDF'en
            bleed_pdf = fitz.open(
                stream=bleed_buffer.getvalue(),
                filetype="pdf",
            )

            try:
                page.show_pdf_page(
                    page.rect,
                    bleed_pdf,
                    0,
                    overlay=False,
                )
            finally:
                bleed_pdf.close()

            # ------------------------------------------
            # ORIGINAL PLAKAT SOM VEKTORGRAFIK
            # ------------------------------------------

            page.show_pdf_page(
                trim,
                source,
                source_page.number,
                overlay=True,
            )

            # ------------------------------------------
            # SKÆREMÆRKER
            # ------------------------------------------

            x0, y0, x1, y1 = trim
            gap = bleed + mark_gap
            length = mark_length

            marks = [
                # Øverst venstre
                ((x0, y0 - gap), (x0, y0 - gap - length)),
                ((x0 - gap, y0), (x0 - gap - length, y0)),

                # Øverst højre
                ((x1, y0 - gap), (x1, y0 - gap - length)),
                ((x1 + gap, y0), (x1 + gap + length, y0)),

                # Nederst venstre
                ((x0, y1 + gap), (x0, y1 + gap + length)),
                ((x0 - gap, y1), (x0 - gap - length, y1)),

                # Nederst højre
                ((x1, y1 + gap), (x1, y1 + gap + length)),
                ((x1 + gap, y1), (x1 + gap + length, y1)),
            ]

            for start, end in marks:
                page.draw_line(
                    fitz.Point(*start),
                    fitz.Point(*end),
                    color=(0, 0, 0),
                    width=0.3,
                )

            # ------------------------------------------
            # PDF-BOKSE
            # ------------------------------------------

            page.set_trimbox(trim)
            page.set_bleedbox(bleed_rect)

        return output.tobytes(
            garbage=4,
            deflate=True,
        )

    finally:
        output.close()
        source.close()



@app.get("/test/print-pdf")
def test_print_pdf():
    require_test_endpoints()
    from fastapi.responses import Response
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    import io

    buffer = io.BytesIO()

    # A3-testplakat
    c = canvas.Canvas(
        buffer,
        pagesize=(297 * mm, 420 * mm),
    )

    # Baggrund: Camel fra centralt farvekartotek
    c.setFillColor(avart_cmyk("camel"))
    c.rect(
        0, 0,
        297 * mm,
        420 * mm,
        fill=1,
        stroke=0,
    )

    # Hvid tekst i CMYK
    c.setFillColor(avart_cmyk("light"))
    c.setFont("Helvetica", 24)
    c.drawCentredString(
        148.5 * mm,
        210 * mm,
        "AVART PRINT TEST",
    )

    c.save()

    # Tilføj 3 mm bleed og skæremærker
    pdf_bytes = make_print_ready_pdf(
        buffer.getvalue(),
        style="camel",
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="AVART-print-test.pdf"'
        },
    )


# --------------------------------------------------
# API
# --------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file1: UploadFile = File(...),
    file2: UploadFile | None = File(None),
    file3: UploadFile | None = File(None),
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(ALPHA_THRESHOLD, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(EPSILON_RATIO, ge=0.00005, le=0.02),
    smooth_window: int = Query(SMOOTH_WINDOW, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    require_test_endpoints()
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
    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),
    alpha_threshold: int = Query(ALPHA_THRESHOLD, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(EPSILON_RATIO, ge=0.00005, le=0.02),
    smooth_window: int = Query(SMOOTH_WINDOW, ge=3, le=51),
    thickness: int = Query(2, ge=1, le=12),
    upscale: int = Query(4, ge=1, le=8),
):
    require_test_endpoints()
    try:
        rgba, original_rgb = remove_background_if_needed(
            file1,
            max_dimension=max_dimension,
            return_original=True,
            noir_test=True,
        )
        
        face_region = detect_face_region(
            original_rgb
        )
        
        print(
            "FACE REGION:",
            face_region,
            flush=True,
        )
        
        mask = alpha_to_mask(
            rgba,
            alpha_threshold=alpha_threshold,
            smooth=smooth,
        )
        
          
        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
            face_region=face_region,
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
    alpha_threshold: int = Query(ALPHA_THRESHOLD, ge=0, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(EPSILON_RATIO, ge=0.00005, le=0.02),
    smooth_window: int = Query(SMOOTH_WINDOW, ge=3, le=51),
    stroke_width: float = Query(3.5, ge=0.5, le=12.0),
    crop_to_subject: bool = Query(True),
    pad: int = Query(30, ge=0, le=300),
):
    require_test_endpoints()
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
            mask=mask,
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
    size="50x70",
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

        # rgba skal ikke bruges mere
        del rgba

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
            mask=mask,
            stroke_width=stroke_width,
            crop_to_subject=True,
            pad=30,
        )

        # De store arrays skal ikke med videre til næste billede
        del mask
        del contour

        import gc
        gc.collect()

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
    style="light",
    size=size,
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

    max_dimension: int = Query(MAX_DIMENSION, ge=600, le=3000),

    alpha_threshold: int = Query(ALPHA_THRESHOLD, ge=0, le=255),
    
    smooth: bool = Query(True),
    
    epsilon_ratio: float = Query(EPSILON_RATIO, ge=0.00005, le=0.02),
    
    smooth_window: int = Query(SMOOTH_WINDOW, ge=3, le=51),
    
    stroke_width: float = Query(
        3.5,
        ge=0.5,
        le=12.0,
    ),

):
    try:

        if len(files) < 1 or len(files) > 6:
            raise ValueError(
                "Upload between 1 and 6 images"
            )

        persons = []

        for index, file in enumerate(files):

            rgba, original_rgb = remove_background_if_needed(
                file,
                max_dimension=max_dimension,
                return_original=True,
            )
            
            face_region = detect_face_region(
                original_rgb
            )
            
            print(
                "POSTER FACE REGION:",
                face_region,
                flush=True,
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
                face_region=face_region,
            )

            head_width = estimate_head_width(
                contour
            )

            svg = contour_to_svg(
                contour=contour,
                width=w,
                height=h,
                mask=mask,
                stroke_width=stroke_width,
                crop_to_subject=True,
                pad=30,
            )

            persons.append(
                {
                    "index": index,
                    "svg": svg,
                    "svg_sig": sign_svg(svg),
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
            "light",
        )
        
        allowed_styles = {
            "light",
            "dark",
            "moss",
            "wine",
            "camel",
            "sea",
        }

        if style not in allowed_styles:
            raise ValueError(
                f"Unknown poster style: {style}"
            )

        # ---------------------------------------------
        # SIZE
        # ---------------------------------------------

        size = data.get(
            "size",
            "50x70",
        )

        allowed_sizes = {
            "A3",
            "A2",
            "50x70",
            "A1",
        }

        if size not in allowed_sizes:
            raise ValueError(
                f"Unknown poster size: {size}"
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

            if not verify_svg(svg, person.get("svg_sig")):
                raise ValueError(
                    "Silhuetten kunne ikke godkendes. Upload billedet igen."
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
                    "flipped": person.get(
                        "flipped",
                        False,
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
            style=style,
            size=size,
        )

        # ---------------------------------------------
        # PDF → PNG PREVIEW
        # ---------------------------------------------

        # HEX-farver til skærmvisning
        preview_colors = {
            "light": "#FBFAF7",
            "dark": "#000000",
            "moss": "#444832",
            "wine": "#5D352A",
            "camel": "#987D58",
            "sea": "#1A404E",
        }

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
        
        page = document[0]
        
        pixmap = page.get_pixmap(
            dpi=150,
            alpha=True,
        )
        
        # Transparent PDF → PNG
        preview_rgba = Image.open(
            io.BytesIO(pixmap.tobytes("png"))
        ).convert("RGBA")

        # Erstat CMYK-baggrunden med den præcise HEX-farve
        if style != "light":
            preview_rgb = preview_rgba.convert("RGB")
            pixels = np.array(preview_rgb)
        
            # Baggrundsfarven aflæses i øverste venstre hjørne
            background_rgb = pixels[0, 0].copy()
        
            # Find pixels med samme farve som baggrunden
            difference = np.max(
                np.abs(pixels.astype(np.int16) - background_rgb.astype(np.int16)),
                axis=2,
            )
        
            background_mask = difference <= 2
        
            # Udskift kun baggrundens pixels
            target_rgb = tuple(
                bytes.fromhex(preview_colors[style].lstrip("#"))
            )
        
            pixels[background_mask] = target_rgb
        
            preview_rgba = Image.fromarray(pixels, "RGB").convert("RGBA")

        # Fjern hvid baggrund fra LIGHT-preview,
        # så den korrekte LIGHT-farve kan vises.
        if style == "light":
            pixels = np.array(preview_rgba)
        
            # Find hvide og næsten hvide baggrundspixels
            rgb = pixels[:, :, :3]
        
            white_mask = np.all(rgb >= 250, axis=2)
        
            # Gør baggrunden gennemsigtig
            pixels[white_mask, 3] = 0
        
            preview_rgba = Image.fromarray(pixels, "RGBA")
            
        # LIGHT baggrund kun til preview på hjemmesiden
        preview_background = Image.new(
            "RGBA",
            preview_rgba.size,
            preview_colors[style],
        )
        
        preview_image = Image.alpha_composite(
            preview_background,
            preview_rgba,
        ).convert("RGB")
        
        preview_buffer = io.BytesIO()
        
        preview_image.save(
            preview_buffer,
            format="PNG",
        )
        
        png_bytes = preview_buffer.getvalue()

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
        128,
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
    
    size: str = Query("50x70"),
    
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
            size=size,
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
        128,
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
    
    size: str = Query("50x70"),
    
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
            size=size,
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
            alpha=True,
        )
        
        preview_rgba = Image.open(
            io.BytesIO(pixmap.tobytes("png"))
        ).convert("RGBA")
        
        preview_background = Image.new(
            "RGBA",
            preview_rgba.size,
            "#FBFAF7",
        )
        
        preview_image = Image.alpha_composite(
            preview_background,
            preview_rgba,
        ).convert("RGB")
        
        preview_buffer = io.BytesIO()
        
        preview_image.save(
            preview_buffer,
            format="PNG",
        )
        
        png_bytes = preview_buffer.getvalue()

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

# =============================================
# STRIPE CHECKOUT
# =============================================

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")


@app.post("/create-checkout-session")
async def create_checkout_session(data: dict):

    prices = {
        "A3": 299,
        "A2": 399,
        "50x70": 499,
        "A1": 599,
    }

    allowed_styles = {
        "light", "dark", "moss",
        "wine", "camel", "sea",
    }

    items = data.get("items", [])
    gift_code = data.get("gift_code") or ""
    gift_ore = 0

    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        return JSONResponse(
            {"error": "Kurven er tom eller ugyldig."},
            status_code=400,
        )

    line_items = []
    validated_items = []
    gift_items = []
    subtotal = 0

    try:
        for index, item in enumerate(items, start=1):

            # Gavekort i samme kurv som plakater
            if item.get("type") == "gift":
                amount_ore, gift_size = gift_amount_from_request(
                    {"size": item.get("size"), "amount": item.get("price")}
                )
                subtotal += amount_ore // 100
                gift_items.append({
                    "item_number": index,
                    "amount_ore": amount_ore,
                    "size": gift_size,
                })
                gift_name = (
                    f"AVART gavekort – {GIFT_SIZE_LABELS[gift_size]} plakat"
                    if gift_size else "AVART gavekort"
                )
                line_items.append({
                    "price_data": {
                        "currency": "dkk",
                        "product_data": {"name": gift_name},
                        "unit_amount": amount_ore,
                    },
                    "quantity": 1,
                })
                continue

            size = item["size"].strip().upper().replace("X", "x") if item["size"].strip().upper() == "50X70" else item["size"].strip().upper()
            style = item["style"]
            name = item.get("name", "")
            persons = item["persons"]

            if size not in prices:
                raise ValueError("Ugyldig plakatstørrelse.")

            if style not in allowed_styles:
                raise ValueError("Ugyldig plakatfarve.")

            if not isinstance(name, str) or len(name) > 255:
                raise ValueError("Ugyldig plakattekst.")

            if not isinstance(persons, list) or not 1 <= len(persons) <= 6:
                raise ValueError("Ugyldigt antal personer.")

            for person in persons:
                if not isinstance(person, dict):
                    raise ValueError("Ugyldige persondata.")

                if not isinstance(person.get("svg"), str) or not person["svg"]:
                    raise ValueError("En person mangler SVG.")

                if not verify_svg(person["svg"], person.get("svg_sig")):
                    raise ValueError("Silhuetten kunne ikke godkendes. Upload billedet igen.")

                if person.get("scale_level", 0) not in (-4, -3, -2, -1, 0, 1, 2):
                    raise ValueError("Ugyldig skalering.")

                if not isinstance(person.get("flipped", False), bool):
                    raise ValueError("Ugyldig spejlvending.")

            price = prices[size]
            subtotal += price

            validated_items.append({
                "item_number": index,
                "size": size,
                "style": style,
                "name": name,
                "price": price,
                "persons": persons,
            })

            line_items.append({
                "price_data": {
                    "currency": "dkk",
                    "product_data": {
                        "name": f"AVART plakat – {size.upper() if size != '50x70' else '50×70'} – {style}"
                    },
                    "unit_amount": price * 100,
                },
                "quantity": 1,
            })

    except (KeyError, TypeError, ValueError) as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=400,
        )

    # Unikt ordrenummer, fx AVA-A81C92F04D
    order_number = "AVA-" + secrets.token_hex(5).upper()

    connection = None
    order_id = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO orders (
                    order_number,
                    payment_status,
                    order_status,
                    subtotal_dkk,
                    shipping_dkk,
                    total_dkk
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    order_number,
                    "pending",
                    "pending",
                    subtotal,
                    49,
                    subtotal + 49,
                ),
            )

            order_id = cursor.lastrowid

            for item in validated_items:

                design_json = json.dumps(
                    {
                        "persons": item["persons"],
                        "name": item["name"],
                        "style": item["style"],
                        "size": item["size"],
                    },
                    ensure_ascii=False,
                )

                cursor.execute(
                    """
                    INSERT INTO order_items (
                        order_id,
                        item_number,
                        size,
                        style,
                        poster_name,
                        price_dkk,
                        design_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        order_id,
                        item["item_number"],
                        item["size"],
                        item["style"],
                        item["name"],
                        item["price"],
                        design_json,
                    ),
                )

            # Gavekort købt i denne ordre (blandet kurv) – hvert gavekort
            # får sit eget kort og sin egen kode, når betalingen er gennemført
            for gift in gift_items:
                reference = "GAVE-" + secrets.token_hex(5).upper()
                cursor.execute(
                    "INSERT INTO avart_gift_purchases (reference, amount_ore, poster_size) VALUES (%s, %s, %s)",
                    (reference, gift["amount_ore"], gift["size"]),
                )
                gift["reference"] = reference

            # Gavekort: reserver saldo (kun plakater – fragt betales altid)
            if gift_normalize_code(gift_code):
                gift_ore = reserve_gift_card(cursor, gift_code, order_id, subtotal * 100)
                cursor.execute(
                    "UPDATE orders SET gift_card_ore = %s WHERE id = %s",
                    (gift_ore, order_id),
                )

        connection.commit()

    except ValueError as e:
        if connection:
            connection.rollback()
        return JSONResponse({"error": str(e)}, status_code=400)

    except Exception:
        if connection:
            connection.rollback()

        return JSONResponse(
            {"error": "Bestillingen kunne ikke gemmes."},
            status_code=500,
        )

    finally:
        if connection:
            connection.close()

    # Træk gavekortbeløbet fra plakatlinjerne (fragten røres ikke)
    remaining_gift = gift_ore
    for line in line_items:
        if remaining_gift <= 0:
            break
        cut = min(remaining_gift, line["price_data"]["unit_amount"])
        line["price_data"]["unit_amount"] -= cut
        line["price_data"]["product_data"]["name"] += " (gavekort fratrukket)"
        remaining_gift -= cut

    # Opret Stripe Checkout efter at ordren er gemt
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            allow_promotion_codes=False,

            shipping_options=[
                {
                    "shipping_rate_data": {
                        "type": "fixed_amount",
                        "fixed_amount": {
                            "amount": 4900,
                            "currency": "dkk",
                        },
                        "display_name": "GLS-fragt",
                    }
                }
            ],

            client_reference_id=order_number,

            metadata={
                "kind": "poster_order",
                "order_id": str(order_id),
                "order_number": order_number,
            },

            billing_address_collection="required",
            
            shipping_address_collection={
                "allowed_countries": ["DK"]
            },
            
            phone_number_collection={
                "enabled": True
            },

            success_url=(
                "https://avart.dk/?payment=success"
                "&session_id={CHECKOUT_SESSION_ID}"
            ),

            cancel_url="https://avart.dk/?payment=cancel",

            expires_at=int(__import__("time").time()) + 1800,
        )

        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE orders
                    SET stripe_session_id = %s
                    WHERE id = %s
                    """,
                    (session.id, order_id),
                )

                for gift in gift_items:
                    cursor.execute(
                        "UPDATE avart_gift_purchases SET stripe_session_id = %s WHERE reference = %s",
                        (session.id, gift["reference"]),
                    )

            connection.commit()

        finally:
            connection.close()

        return {
            "url": session.url,
            "order_number": order_number,
        }

    except Exception:
        if gift_ore:
            try:
                connection = get_db_connection()
                with connection.cursor() as cursor:
                    release_gift_reservation(cursor, order_id)
                connection.commit()
                connection.close()
            except Exception:
                pass
        return JSONResponse(
            {
                "error": "Betalingen kunne ikke oprettes. Prøv igen."
            },
            status_code=500,
        )

def send_order_email(to_email, subject, body, attachments=None):
    """Sender en e-mail fra AVARTs ordremail."""

    message = EmailMessage()
    message["From"] = os.environ["SMTP_USER"]
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    for filename, pdf_bytes in (attachments or []):
        message.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=filename
        )

    with smtplib.SMTP_SSL(
        os.environ["SMTP_HOST"],
        int(os.environ["SMTP_PORT"]),
        timeout=30
    ) as smtp:
        smtp.login(
            os.environ["SMTP_USER"],
            os.environ["SMTP_PASSWORD"]
        )
        smtp.send_message(message)

# --------------------------------------------------
# AVART – KUNDEORDREBEKRÆFTELSE (HTML + VILKÅR)
# --------------------------------------------------

AVART_TERMS_PDF_URL = "https://avart.dk/docs/AVART-Handelsbetingelser.pdf"
AVART_TIMEZONE = ZoneInfo("Europe/Copenhagen")


def avart_money(amount):
    return f"{float(amount):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def avart_schedule(paid_at):
    """Ordrer senest mandag medtages i den kommende tirsdagsproduktion."""
    if paid_at is None:
        paid_at = datetime.now(AVART_TIMEZONE)
    elif paid_at.tzinfo is None:
        # Forudsætter at DB-sessionens tidszone er Europe/Copenhagen.
        paid_at = paid_at.replace(tzinfo=AVART_TIMEZONE)
    else:
        paid_at = paid_at.astimezone(AVART_TIMEZONE)
    days_until_tuesday = (1 - paid_at.weekday()) % 7
    if days_until_tuesday == 0:
        days_until_tuesday = 7
    production = (paid_at + timedelta(days=days_until_tuesday)).date()
    shipping = production + timedelta(days=1)
    months = ["", "januar", "februar", "marts", "april", "maj", "juni", "juli", "august", "september", "oktober", "november", "december"]
    return (f"Tirsdag den {production.day}. {months[production.month]}",
            f"Onsdag den {shipping.day}. {months[shipping.month]}")


def avart_download_terms_pdf():
    with urlopen(AVART_TERMS_PDF_URL, timeout=20) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000 or not data.startswith(b"%PDF-"):
        raise ValueError("Handelsbetingelserne er ikke en gyldig PDF under 5 MB")
    return data


def avart_poster_preview(items):
    """Generér den faktiske produktions-PDF direkte til preview."""
    first = items[0]
    design = json.loads(first["design_json"])
    persons = design["persons"]
    orientation = "portrait" if len(persons) <= 2 else "landscape"
    pdf_bytes = generate_multi_poster_pdf(
        persons=persons,
        name=design["name"],
        stroke_width=DEFAULT_STROKE_WIDTH,
        orientation=orientation,
        style=design["style"],
        size=design["size"],
    )
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(0.6, 0.6), alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()


def avart_customer_email_html(order, items, gift_rows=None):
    template_path = os.path.join(os.path.dirname(__file__), "email_templates", "ordrebekraeftelse.html")
    with open(template_path, encoding="utf-8") as file:
        template = file.read()
    if not items:
        start = template.index("<!--PRODUCTION_START-->")
        end = template.index("<!--PRODUCTION_END-->") + len("<!--PRODUCTION_END-->")
        template = template[:start] + template[end:]
    paid_at = order.get("paid_at")
    prod_date, ship_date = avart_schedule(paid_at)
    if isinstance(paid_at, datetime):
        date_value = paid_at.astimezone(AVART_TIMEZONE) if paid_at.tzinfo else paid_at
        months = ["", "januar", "februar", "marts", "april", "maj", "juni", "juli", "august", "september", "oktober", "november", "december"]
        order_date = f"{date_value.day}. {months[date_value.month]} {date_value.year} kl. {date_value:%H.%M}"
    else:
        order_date = ""
    rows = []
    for item in items:
        size = {"50x70": "50×70"}.get(item["size"], item["size"])
        rows.append('<tr><td style="padding:21px 0 18px">{}</td><td style="padding:21px 0 18px">{}</td><td style="padding:21px 0 18px">1 stk.</td><td align="right" style="padding:21px 0 18px">{}</td></tr>'.format(
            html_escape(str(item["style"]).upper()), html_escape(size), avart_money(item["price_dkk"])))
    rows.extend(gift_rows or [])
    address = ", ".join(filter(None, [order.get("shipping_address"),
                 " ".join(filter(None, [str(order.get("shipping_postal_code") or ""), order.get("shipping_city") or ""]))]))
    total = (order["amount_paid_ore"] / 100) if order.get("amount_paid_ore") is not None else float(order["total_dkk"])
    gift_deduction = int(order.get("gift_card_ore") or 0) / 100
    deduction_row = (
        '<tr><td colspan="3" style="padding:16px 0">Gavekort</td>'
        '<td align="right" style="padding:16px 0">−{}</td></tr>'.format(avart_money(gift_deduction))
        if gift_deduction else ""
    )
    intro = "Vi har modtaget din ordre og lægger din plakat i kø! Snart er dit billede på vej fra idé til virkelighed."
    if not items:
        intro = ("Hvert gavekort er vedhæftet som PDF, så du kan printe dem eller sende dem videre. "
                 "Koden indtastes i kurven på avart.dk. Et eventuelt restbeløb gemmes til næste køb. "
                 f"Gavekortene er gyldige i {GIFT_VALID_YEARS} år.")
    substitutions = {
        "INTRO_TEXT": intro, "PAYMENT_STATUS": "Betalt med Stripe",
        "GIFT_DEDUCTION_ROW": deduction_row,
        "PRODUCTION_DATE": prod_date, "SHIPPING_DATE": ship_date,
        "ORDER_NUMBER": order["order_number"], "ORDER_DATE": order_date,
        "ITEM_ROWS": "".join(rows), "SHIPPING_PRICE": avart_money(order["shipping_dkk"]),
        "TOTAL_PRICE": avart_money(total), "VAT_AMOUNT": avart_money(total / 5),
        "SHIPPING_NAME": order.get("shipping_name") or order.get("customer_name") or "",
        "SHIPPING_ADDRESS": address,
    }
    for key, value in substitutions.items():
        template = template.replace("{{" + key + "}}", value if key in ("ITEM_ROWS", "GIFT_DEDUCTION_ROW") else html_escape(str(value)))
    # Encode the subject in the mailto URL, including the current order number.
    template = template.replace("subject=Sp%C3%B8rgsm%C3%A5l%20til%20ordre%20" + html_escape(order["order_number"]),
        "subject=" + quote("Spørgsmål til ordre " + order["order_number"]))
    return template


def avart_gift_item_rows(gifts):
    rows = []
    for gift in gifts:
        size = gift.get("poster_size") or ""
        amount = int(gift.get("amount_ore", gift.get("original_ore", 0))) / 100
        rows.append('<tr><td style="padding:21px 0 18px">GAVEKORT</td>'
                    '<td style="padding:21px 0 18px">{}</td>'
                    '<td style="padding:21px 0 18px">1 stk.</td>'
                    '<td align="right" style="padding:21px 0 18px">{}</td></tr>'.format(
                        html_escape(str(size)), avart_money(amount)))
    return rows


def send_avart_confirmation_message(order, items, gift_rows=None, attachments=None):
    """Samme HTML og vedhæftninger for plakater, gavekort og blandede ordrer."""
    html = avart_customer_email_html(order, items, gift_rows)
    terms = avart_download_terms_pdf()
    logo_path = os.path.join(os.path.dirname(__file__), "email_templates", "avart-logo.png")
    with open(logo_path, "rb") as file:
        logo = file.read()
    message = EmailMessage()
    message["From"] = os.environ["SMTP_USER"]
    message["To"] = order["customer_email"]
    message["Subject"] = f"AVART – Ordrebekræftelse {order['order_number']}"
    message.set_content(f"Tak for din bestilling hos AVART. Ordrenummer: {order['order_number']}. "
                        "Handelsbetingelser og eventuelle gavekort er vedhæftet.")
    message.add_alternative(html, subtype="html")
    html_part = message.get_payload()[-1]
    html_part.add_related(logo, maintype="image", subtype="png", cid="<avart_logo>", filename="avart-logo.png")
    if items:
        html_part.add_related(avart_poster_preview(items), maintype="image", subtype="png",
                              cid="<avart_preview>", filename="plakat-preview.png")
    message.add_attachment(terms, maintype="application", subtype="pdf", filename="AVART-Handelsbetingelser.pdf")
    for filename, data in (attachments or []):
        message.add_attachment(data, maintype="application", subtype="pdf", filename=filename)
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]), timeout=30) as smtp:
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(message)


def send_customer_confirmation(order_id, gift_attachments=None):
    """Send kundemail med DB-lås, så samtidige webhook-kald ikke sender dubletter.

    SMTP og database er ikke én transaktion. Ved en usikker SMTP/DB-fejl
    kræves manuel kontrol af afsendelsen før et nyt forsøg.
    """
    connection = get_db_connection()
    lock_name = f"avart_customer_email_{int(order_id)}"
    lock_acquired = False
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 10) AS acquired", (lock_name,))
            lock_acquired = cursor.fetchone()["acquired"] == 1
            if not lock_acquired:
                raise RuntimeError("Kundemail behandles allerede af et andet kald")

            cursor.execute("SELECT * FROM orders WHERE id=%s", (order_id,))
            order = cursor.fetchone()
            if not order or order["payment_status"] != "paid":
                raise ValueError("Kun betalte ordrer kan bekræftes")
            if order.get("customer_email_sent_at"):
                # Ældre blandede ordrer kan allerede have fået kvitteringen,
                # men stadig mangle gavekort-PDF'erne. De må ikke gå tabt.
                if gift_attachments:
                    cursor.execute(
                        "SELECT poster_size, amount_ore FROM avart_gift_purchases "
                        "WHERE stripe_session_id = %s ORDER BY id",
                        (order["stripe_session_id"],),
                    )
                    send_avart_confirmation_message(order, [], avart_gift_item_rows(cursor.fetchall()), gift_attachments)
                return
            if not order.get("customer_email"):
                raise ValueError("Kundens e-mail mangler")
            cursor.execute(
                "SELECT * FROM order_items WHERE order_id=%s ORDER BY item_number",
                (order_id,),
            )
            items = cursor.fetchall()

            # Gavekort købt i samme ordre (blandet kurv)
            gift_rows = []
            if order.get("stripe_session_id"):
                cursor.execute(
                    "SELECT poster_size, amount_ore FROM avart_gift_purchases "
                    "WHERE stripe_session_id = %s ORDER BY id",
                    (order["stripe_session_id"],),
                )
                gift_rows = avart_gift_item_rows(cursor.fetchall())

        if not items and not gift_rows:
            raise ValueError("Ordren har ingen produkter")

        send_avart_confirmation_message(order, items, gift_rows, gift_attachments)

        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE orders SET customer_email_sent_at=NOW() "
                "WHERE id=%s AND customer_email_sent_at IS NULL",
                (order_id,),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if lock_acquired:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
            except Exception as error:
                print(f"AVART EMAIL LOCK RELEASE ERROR: {error}", flush=True)
        connection.close()



def send_production_email(order_id):
    """Sender produktionsfiler og leveringsoplysninger til trykkeriet."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM orders
                WHERE id = %s
                """,
                (order_id,),
            )
            order = cursor.fetchone()

            if not order:
                raise ValueError("Ordren findes ikke.")

            if order["payment_status"] != "paid":
                raise ValueError("Ordren er ikke betalt.")

            if order["production_email_sent_at"]:
                return

            cursor.execute(
                """
                SELECT *
                FROM order_items
                WHERE order_id = %s
                ORDER BY id
                """,
                (order_id,),
            )
            items = cursor.fetchall()

        if not items:
            raise ValueError("Ordren indeholder ingen plakater.")

        attachments = []
        poster_lines = []
        item_files = []

        for index, item in enumerate(items, start=1):
            design = json.loads(item["design_json"])

            persons = design["persons"]
            name = design["name"]
            style = design["style"]
            size = design["size"]

            orientation = (
                "portrait"
                if len(persons) <= 2
                else "landscape"
            )

            pdf_bytes = generate_multi_poster_pdf(
                persons=persons,
                name=name,
                stroke_width=DEFAULT_STROKE_WIDTH,
                orientation=orientation,
                style=style,
                size=size,
            )

            print_pdf = make_print_ready_pdf(
                pdf_bytes,
                style=style,
            )

            filename = (
                f"{order['order_number']}_"
                f"{item['item_number']}.pdf"
            )

            attachments.append((filename, print_pdf))
            item_files.append((item["id"], filename))

            poster_lines.append(
                f"Plakat {index}: "
                f"{design.get('size') or 'Ukendt størrelse'} / "
                f"{design.get('style') or 'Ukendt stil'}"
            )

        body = f"""NY PRODUKTIONSORDRE – AVART

Ordrenummer: {order['order_number']}

LEVERINGSOPLYSNINGER
Navn: {order.get('shipping_name') or order.get('customer_name') or ''}
Adresse: {order.get('shipping_address') or ''}
Postnummer: {order.get('shipping_postal_code') or ''}
By: {order.get('shipping_city') or ''}
Land: {order.get('shipping_country') or 'DK'}

KONTAKT
E-mail: {order.get('customer_email') or ''}
Telefon: {order.get('customer_phone') or ''}

PLAKATER
{chr(10).join(poster_lines)}

De trykklare PDF-filer er vedhæftet.
"""

        send_order_email(
            os.environ["PRODUCTION_EMAIL"],
            f"AVART – Produktionsordre {order['order_number']}",
            body,
            attachments,
        )

        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE orders
                SET production_email_sent_at = NOW()
                WHERE id = %s
                  AND production_email_sent_at IS NULL
                """,
                (order_id,),
            )

            for item_id, filename in item_files:
                cursor.execute(
                    """
                    UPDATE order_items
                    SET production_pdf_path = %s
                    WHERE id = %s
                    """,
                    (filename, item_id),
                )

        connection.commit()

    finally:
        connection.close()





# --------------------------------------------------
# STRIPE WEBHOOK – BETALINGSBEKRÆFTELSE
# --------------------------------------------------

# ==================================================
# AVART – GAVEKORT
# ==================================================
import hashlib

GIFT_SIZE_PRICES = {"A3": 299, "A2": 399, "50x70": 499, "A1": 599}
GIFT_SIZE_LABELS = {"A3": "A3 · 297×420 mm", "A2": "A2 · 420×594 mm",
                    "50x70": "50×70 cm", "A1": "A1 · 594×841 mm"}
GIFT_MIN_DKK = 100
GIFT_MAX_DKK = 5000
GIFT_VALID_YEARS = 3  # Dansk standard forældelse for gavekort


def gift_normalize_code(code):
    if not isinstance(code, str):
        return ""
    return "".join(ch for ch in code.lower() if ch.isalnum())


def gift_code_hash(code):
    return hashlib.sha256(gift_normalize_code(code).encode()).hexdigest()


def gift_new_code():
    # Små bogstaver som i Louises eksempel (akls486e3).
    # Tegn der let forveksles (i, l, o, 0, 1) er udeladt.
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(9))


def gift_amount_from_request(data):
    """Returnerer (beløb i øre, størrelse eller None). Prisen sættes altid på serveren."""
    size = data.get("size")
    if size:
        size = "50x70" if str(size).strip().upper() == "50X70" else str(size).strip().upper()
        if size not in GIFT_SIZE_PRICES:
            raise ValueError("Ugyldig plakatstørrelse.")
        return GIFT_SIZE_PRICES[size] * 100, size
    amount = data.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, str)):
        raise ValueError("Ugyldigt beløb.")
    amount = str(amount).strip()
    if not amount.isdigit():
        raise ValueError("Beløbet skal være et helt antal kroner.")
    amount = int(amount)
    if not GIFT_MIN_DKK <= amount <= GIFT_MAX_DKK:
        raise ValueError(f"Beløbet skal være mellem {GIFT_MIN_DKK} og {GIFT_MAX_DKK} kr.")
    return amount * 100, None


def gift_available_ore(cursor, card_id, balance_ore):
    cursor.execute(
        """
        SELECT COALESCE(SUM(amount_ore), 0) AS reserved
        FROM avart_gift_reservations
        WHERE card_id = %s AND status = 'active' AND expires_at > NOW()
        """,
        (card_id,),
    )
    return max(0, int(balance_ore) - int(cursor.fetchone()["reserved"]))


def gift_find_card(cursor, code, lock=False):
    cursor.execute(
        "SELECT id, balance_ore, expires_at FROM avart_gift_cards "
        "WHERE code_hash = %s AND expires_at >= CURDATE()" + (" FOR UPDATE" if lock else ""),
        (gift_code_hash(code),),
    )
    return cursor.fetchone()


def reserve_gift_card(cursor, code, order_id, max_ore):
    """Låser gavekortet og reserverer op til max_ore. Returnerer reserveret beløb i øre."""
    card = gift_find_card(cursor, code, lock=True)
    if not card:
        raise ValueError("Gavekortet findes ikke eller er udløbet.")
    available = gift_available_ore(cursor, card["id"], card["balance_ore"])
    if available <= 0:
        raise ValueError("Gavekortet er brugt op.")
    amount = min(available, max_ore)
    cursor.execute(
        """
        INSERT INTO avart_gift_reservations (card_id, order_id, amount_ore, status, expires_at)
        VALUES (%s, %s, %s, 'active', NOW() + INTERVAL 35 MINUTE)
        """,
        (card["id"], order_id, amount),
    )
    return amount


def settle_gift_reservation(cursor, order_id):
    """Trækker reservationen fra saldoen. Sikker at kalde flere gange."""
    cursor.execute(
        "SELECT id, card_id, amount_ore, status FROM avart_gift_reservations "
        "WHERE order_id = %s FOR UPDATE",
        (order_id,),
    )
    reservation = cursor.fetchone()
    if not reservation or reservation["status"] == "spent":
        return
    cursor.execute(
        "SELECT balance_ore FROM avart_gift_cards WHERE id = %s FOR UPDATE",
        (reservation["card_id"],),
    )
    card = cursor.fetchone()
    new_balance = max(0, int(card["balance_ore"]) - int(reservation["amount_ore"]))
    if int(card["balance_ore"]) < int(reservation["amount_ore"]):
        print(f"AVART GIFT NEEDS REVIEW: order_id={order_id} balance too low", flush=True)
    cursor.execute("UPDATE avart_gift_cards SET balance_ore = %s WHERE id = %s",
                   (new_balance, reservation["card_id"]))
    cursor.execute("UPDATE avart_gift_reservations SET status = 'spent' WHERE id = %s",
                   (reservation["id"],))


def release_gift_reservation(cursor, order_id):
    cursor.execute(
        "UPDATE avart_gift_reservations SET status = 'released' "
        "WHERE order_id = %s AND status = 'active'",
        (order_id,),
    )


@app.post("/gift-card/balance")
async def gift_card_balance(data: dict):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            card = gift_find_card(cursor, data.get("code"))
            if not card:
                return JSONResponse({"error": "Gavekortet findes ikke eller er udløbet."}, status_code=404)
            available = gift_available_ore(cursor, card["id"], card["balance_ore"])
        connection.rollback()
        return {"balance_dkk": available / 100}
    finally:
        connection.close()


@app.post("/gift-card/checkout")
async def gift_card_checkout(data: dict):

    # Ét eller flere gavekort ad gangen:
    # {"gifts": [{"size": "A2"}, {"amount": "500"}]}
    gifts = data.get("gifts")
    if not isinstance(gifts, list) or not 1 <= len(gifts) <= 10:
        gifts = [data]

    parsed = []
    try:
        for gift in gifts:
            amount_ore, size = gift_amount_from_request(gift)
            parsed.append((amount_ore, size))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    references = []
    connection = get_db_connection()
    try:
        try:
            with connection.cursor() as cursor:
                for amount_ore, size in parsed:
                    reference = "GAVE-" + secrets.token_hex(5).upper()
                    cursor.execute(
                        "INSERT INTO avart_gift_purchases (reference, amount_ore, poster_size) VALUES (%s, %s, %s)",
                        (reference, amount_ore, size),
                    )
                    references.append(reference)
            connection.commit()
        except Exception:
            connection.rollback()
            return JSONResponse({"error": "Gavekortet kunne ikke oprettes."}, status_code=500)
    finally:
        connection.close()

    line_items = []
    for amount_ore, size in parsed:
        name = f"AVART gavekort – {GIFT_SIZE_LABELS[size]} plakat" if size else "AVART gavekort"
        line_items.append({
            "price_data": {
                "currency": "dkk",
                "product_data": {"name": name},
                "unit_amount": amount_ore,
            },
            "quantity": 1,
        })

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            metadata={"kind": "gift_card"},
            customer_creation="if_required",
            billing_address_collection="auto",
            success_url="https://avart.dk/?payment=success&gift=1&session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://avart.dk/?payment=cancel#gavekort",
            expires_at=int(__import__("time").time()) + 1800,
        )
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                for reference in references:
                    cursor.execute(
                        "UPDATE avart_gift_purchases SET stripe_session_id = %s WHERE reference = %s",
                        (session.id, reference),
                    )
            connection.commit()
        finally:
            connection.close()
        return {"url": session.url}
    except Exception:
        return JSONResponse({"error": "Betalingen kunne ikke oprettes. Prøv igen."}, status_code=500)


# --------------------------------------------------
# GAVEKORT-PDF – Louises foldbare A4-skabelon.
# Kun de lyserøde felter udfyldes; resten af designet røres ikke.
# Skabelon og skrifttyper ligger i mappen "giftcard" ved siden af main.py.
# --------------------------------------------------
GIFTCARD_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "giftcard", "giftcard_template.pdf")
GIFTCARD_FONT_SEASONS = os.path.join(os.path.dirname(__file__), "giftcard", "The-Seasons-Bold.ttf")
GIFTCARD_FONT_SS3_SEMI = os.path.join(os.path.dirname(__file__), "giftcard", "SourceSans3-Semibold.ttf")
GIFTCARD_FONT_SS3_REG = os.path.join(os.path.dirname(__file__), "giftcard", "SourceSans3-Regular.ttf")

GIFTCARD_MONTHS = ["", "januar", "februar", "marts", "april", "maj", "juni",
                   "juli", "august", "september", "oktober", "november", "december"]


def _gift_span_is_pink(span):
    color = span["color"]
    red = (color >> 16) & 255
    green = (color >> 8) & 255
    blue = color & 255
    return red >= 200 and green <= 70 and blue >= 100


def gift_card_pdf(code, amount_ore, size, expires_at):
    """Udfylder skabelonen: overskrift, værdi, kode og udløbsdato i sort."""
    document = pymupdf.open(GIFTCARD_TEMPLATE_PATH)
    page = document[0]

    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    spans.append(span)

    pink_spans = [span for span in spans if _gift_span_is_pink(span)]

    # Find linjen med værdien (teksten "værdi") og ryd hele linjen
    value_y = None
    for span in spans:
        if "værdi" in span["text"].lower():
            value_y = span["bbox"][1]
    if value_y is not None:
        for span in spans:
            if abs(span["bbox"][1] - value_y) < 3:
                page.add_redact_annot(pymupdf.Rect(span["bbox"]))
    for span in pink_spans:
        page.add_redact_annot(pymupdf.Rect(span["bbox"]))
    page.apply_redactions(images=0, graphics=0)

    font_seasons = pymupdf.Font(fontfile=GIFTCARD_FONT_SEASONS)
    font_ss3_semi = pymupdf.Font(fontfile=GIFTCARD_FONT_SS3_SEMI)
    font_ss3_reg = pymupdf.Font(fontfile=GIFTCARD_FONT_SS3_REG)

    black = (0x23 / 255, 0x1F / 255, 0x20 / 255)
    writer = pymupdf.TextWriter(page_rect=page.rect)

    def put(x, y, font_size, text, font):
        writer.append((x, y), text, font=font, fontsize=font_size)

    # Overskrift: "Gavekort, A2" → "Gavekort, 500 kr."
    heading_text = size if size else f"{int(amount_ore / 100)} kr."
    heading = [s for s in pink_spans
               if s["font"].endswith("TheSeasonsBold") and s["bbox"][1] < 200]
    if heading:
        span = heading[0]
        put(span["origin"][0], span["origin"][1], span["size"], heading_text, font_seasons)

    # Koden centreret i den hvide boks
    box = pymupdf.Rect(72.1, 258.1, 345.4, 293.9)
    code_spans = [s for s in pink_spans
                  if s["font"].endswith("TheSeasonsBold") and s["bbox"][1] >= 200]
    if code_spans:
        span = code_spans[0]
        code_width = font_seasons.text_length(code, span["size"])
        put((box.x0 + box.x1) / 2 - code_width / 2, span["origin"][1],
            span["size"], code, font_seasons)

    # Værdilinjen skrives igen i sort
    value_base = None
    for span in spans:
        if "værdi" in span["text"].lower():
            value_base = span["origin"][1]
    if value_base is not None:
        x = 72.1
        for text, font in (
            ("en ", font_ss3_reg),
            ("værdi på ", font_ss3_semi),
            (f"{int(amount_ore / 100)}", font_ss3_semi),
            (" kr. ", font_ss3_semi),
            ("og kan bruges til alle plakatstørrelser.", font_ss3_reg),
        ):
            put(x, value_base, 10, text, font)
            x += font.text_length(text, 10)

    # Udløbsdato
    date_spans = [s for s in pink_spans
                  if s["font"].startswith("SourceSans3-Semibold") and s["bbox"][1] > 380]
    if date_spans:
        span = date_spans[0]
        date_text = f"{expires_at.day}. {GIFTCARD_MONTHS[expires_at.month]} {expires_at.year}."
        put(span["origin"][0], span["origin"][1], span["size"], date_text, font_ss3_semi)

    writer.write_text(page, color=black)

    buffer = io.BytesIO()
    document.save(buffer, garbage=3, deflate=True)
    document.close()
    return buffer.getvalue()


def fulfil_gift_purchases(session, order_id=None):
    """Udsteder ét gavekort pr. købte gavekort og mailer alle PDF'er til køberen."""
    session_id = session.get("id")
    email = (session.get("customer_details") or {}).get("email")
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM avart_gift_purchases WHERE stripe_session_id = %s ORDER BY id FOR UPDATE",
                (session_id,),
            )
            purchases = cursor.fetchall()
            if not purchases:
                connection.rollback()
                return {"received": True, "gift_cards": "not_found"}
            if session.get("currency") != "dkk" or (order_id is None and session.get("amount_total") != sum(
                    int(purchase["amount_ore"]) for purchase in purchases)):
                connection.rollback()
                return {"received": True, "gift_cards": "amount_mismatch"}

            expires = datetime.now(AVART_TIMEZONE).date() + timedelta(days=365 * GIFT_VALID_YEARS + 1)
            for purchase in purchases:
                if purchase["status"] != "pending":
                    continue
                code = gift_new_code()
                cursor.execute(
                    """
                    INSERT INTO avart_gift_cards
                      (purchase_id, code_hash, delivery_code, original_ore, balance_ore, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (purchase["id"], gift_code_hash(code), code, purchase["amount_ore"],
                     purchase["amount_ore"], expires),
                )
                cursor.execute(
                    "UPDATE avart_gift_purchases SET status = 'paid', buyer_email = %s, paid_at = NOW() WHERE id = %s",
                    (email, purchase["id"]),
                )

            cursor.execute(
                "SELECT c.*, p.poster_size, p.reference, p.paid_at FROM avart_gift_cards c "
                "JOIN avart_gift_purchases p ON p.id = c.purchase_id "
                "WHERE p.stripe_session_id = %s ORDER BY c.id FOR UPDATE",
                (session_id,),
            )
            cards = cursor.fetchall()
            if not cards:
                connection.commit()
                return {"received": True, "gift_cards": "not_found"}
            cards_to_send = [card for card in cards
                             if card["delivery_status"] in ("pending", "sending")]
            if not cards_to_send:
                connection.commit()
                return {"received": True, "gift_cards": "sent"}
            # Markér før afsendelse, så Stripe-genforsøg ikke sender dubletter.
            for card in cards_to_send:
                cursor.execute("UPDATE avart_gift_cards SET delivery_status = 'sending' WHERE id = %s",
                               (card["id"],))
        connection.commit()
    except Exception:
        connection.rollback()
        return {"received": True, "gift_cards": "error"}
    finally:
        connection.close()

    status = "sent"
    try:
        attachments = []
        used_filenames = set()
        for card in cards_to_send:
            pdf_bytes = gift_card_pdf(card["delivery_code"], card["original_ore"],
                                      card["poster_size"], card["expires_at"])
            base = card["poster_size"] or f"{int(card['original_ore']) // 100}kr"
            filename = f"AVART-gavekort-{base}.pdf"
            counter = 2
            while filename in used_filenames:
                filename = f"AVART-gavekort-{base}-{counter}.pdf"
                counter += 1
            used_filenames.add(filename)
            attachments.append((filename, pdf_bytes))

        if order_id is not None:
            send_customer_confirmation(order_id, gift_attachments=attachments)
        else:
            customer = session.get("customer_details") or {}
            address = customer.get("address") or {}
            order = {
                "order_number": cards[0]["reference"], "paid_at": cards[0].get("paid_at"),
                "customer_email": email, "customer_name": customer.get("name") or "",
                "shipping_address": ", ".join(filter(None, [address.get("line1"), address.get("line2")])),
                "shipping_postal_code": address.get("postal_code"), "shipping_city": address.get("city"),
                "shipping_dkk": 0, "total_dkk": session["amount_total"] / 100,
                "amount_paid_ore": session["amount_total"], "gift_card_ore": 0,
            }
            send_avart_confirmation_message(order, [], avart_gift_item_rows(cards), attachments)

    except Exception as e:
        status = "needs_review"
        print(f"AVART GIFT EMAIL NEEDS REVIEW: session_id={session_id} {type(e).__name__}: {e}", flush=True)

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            for card in cards_to_send:
                cursor.execute(
                    "UPDATE avart_gift_cards SET delivery_status = %s, "
                    "delivery_code = IF(%s = 'sent', NULL, delivery_code) WHERE id = %s",
                    (status, status, card["id"]),
                )
        connection.commit()
    finally:
        connection.close()
    return {"received": True, "gift_cards": status}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):

    payload = await request.body()
    signature = request.headers.get("stripe-signature")

    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not signature or not webhook_secret:
        return JSONResponse(
            {"error": "Webhook-konfiguration mangler."},
            status_code=400,
        )

    # Kontrollér, at beskeden faktisk kommer fra Stripe
    try:
        event = stripe.Webhook.construct_event(
            payload,
            signature,
            webhook_secret,
        )

    except (ValueError, stripe.error.SignatureVerificationError):
        return JSONResponse(
            {"error": "Ugyldig Stripe-signatur."},
            status_code=400,
        )

    # Udløbet betaling: frigiv et eventuelt reserveret gavekortbeløb
    if event["type"] == "checkout.session.expired":
        expired = event["data"]["object"].to_dict()
        order_ref = (expired.get("metadata") or {}).get("order_id")
        if order_ref and str(order_ref).isdigit():
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    release_gift_reservation(cursor, int(order_ref))
                connection.commit()
            finally:
                connection.close()
        return {"received": True}

    # Vi behandler kun gennemførte Checkout-sessioner
    if event["type"] != "checkout.session.completed":
        return {"received": True}

    session = event["data"]["object"]
    session = session.to_dict()

    # En gennemført Checkout-session er ikke altid betalt endnu
    if session.get("payment_status") != "paid":
        return {"received": True, "payment": "pending"}

    if (session.get("metadata") or {}).get("kind") == "gift_card":
        return fulfil_gift_purchases(session)

    order_number = session.get("client_reference_id")
    session_id = session.get("id")

    if not order_number or not session_id:
        return JSONResponse(
            {"error": "Ordreoplysninger mangler."},
            status_code=400,
        )

    connection = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:

            # Find ordren og lås den under opdateringen
            cursor.execute(
                """
                SELECT id, payment_status, subtotal_dkk, shipping_dkk, gift_card_ore
                FROM orders
                WHERE order_number = %s
                  AND stripe_session_id = %s
                FOR UPDATE
                """,
                (order_number, session_id),
            )

            order = cursor.fetchone()

            if not order:
                connection.rollback()
                return JSONResponse(
                    {"error": "Ordren blev ikke fundet."},
                    status_code=404,
                )

            # Kontrollér, at Stripe har modtaget det rigtige beløb
            amount_total = session.get("amount_total")
            currency = session.get("currency")

            # Rabatkoder gør totalen mindre, så vi tjekker delbeløbene
            details = session.get("total_details") or {}
            amount_subtotal = session.get("amount_subtotal")
            discount_ore = int(details.get("amount_discount") or 0)
            shipping_ore = int(details.get("amount_shipping") or 0)
            expected_subtotal = order["subtotal_dkk"] * 100 - int(order["gift_card_ore"] or 0)

            if (
                amount_subtotal != expected_subtotal
                or shipping_ore != order["shipping_dkk"] * 100
                or amount_total != amount_subtotal + shipping_ore - discount_ore
                or currency != "dkk"
            ):
                connection.rollback()
                return JSONResponse(
                    {"error": "Betalingsbeløbet stemmer ikke."},
                    status_code=400,
                )

            # Bevar idempotent betalingsopdatering, men tillad genforsøg på kundemail.
            already_paid = order["payment_status"] == "paid"

            customer = session.get("customer_details") or {}
            shipping = session.get("shipping_details") or {}
            address = shipping.get("address") or {}

            # Nyere Stripe-versioner kan have leveringsoplysninger
            # under collected_information
            collected = session.get("collected_information") or {}

            if not shipping:
                shipping = collected.get("shipping_details") or {}
                address = shipping.get("address") or {}

            if not already_paid:
                cursor.execute(
                    """
                UPDATE orders
                SET
                    payment_status = %s,
                    order_status = %s,
                    stripe_payment_intent_id = %s,
                    customer_name = %s,
                    customer_email = %s,
                    customer_phone = %s,
                    shipping_name = %s,
                    shipping_address = %s,
                    shipping_postal_code = %s,
                    shipping_city = %s,
                    shipping_country = %s,
                    discount_ore = %s,
                    amount_paid_ore = %s,
                    paid_at = NOW()
                WHERE id = %s
                """,
                (
                    "paid",
                    "paid",
                    session.get("payment_intent"),
                    customer.get("name"),
                    customer.get("email"),
                    customer.get("phone"),
                    shipping.get("name"),
                    address.get("line1", "") + (
                        ", " + address["line2"]
                        if address.get("line2") else ""
                    ),
                    address.get("postal_code"),
                    address.get("city"),
                    address.get("country", "DK"),
                    discount_ore,
                    amount_total,
                    order["id"],
                ),
            )
                settle_gift_reservation(cursor, order["id"])

        connection.commit()

    except Exception:
        if connection:
            connection.rollback()

        # Stripe forsøger igen ved serverfejl
        return JSONResponse(
            {"error": "Ordren kunne ikke opdateres."},
            status_code=500,
        )

    finally:
        if connection:
            connection.close()
            
    try:
        send_production_email(int(order["id"]))
        print(
            f"AVART PRODUCTION EMAIL SENT: order_id={order['id']}",
            flush=True,
        )
    except Exception as e:
        print(
            f"AVART PRODUCTION EMAIL ERROR: {type(e).__name__}: {e}",
            flush=True,
        )   

    
    # Gavekort-PDF'er og ordrebekræftelse sendes samlet ved blandede køb.
    try:
        gift_result = fulfil_gift_purchases(session, order_id=int(order["id"]))
        if gift_result.get("gift_cards") == "not_found":
            send_customer_confirmation(int(order["id"]))
        elif gift_result.get("gift_cards") != "sent":
            return {"received": True, "payment": "paid", "customer_email": "needs_review"}
    except Exception as e:
        print(f"AVART CUSTOMER EMAIL NEEDS REVIEW: order_id={order['id']} {type(e).__name__}: {e}", flush=True)
        return {"received": True, "payment": "paid", "customer_email": "needs_review"}

    return {"received": True, "payment": "paid"}


@app.get("/checkout-session")
async def get_checkout_session(session_id: str = Query(...)):
    try:
        session = stripe.checkout.Session.retrieve(
            session_id
        )

        return {
            "id": session.id,
            "payment_status": session.payment_status,
            "customer_details": session.customer_details,
            "shipping_details": session.to_dict().get("shipping_details"),
            "amount_total": session.amount_total,
            "currency": session.currency,
        }

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=400
        )


