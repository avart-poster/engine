import io
import os
import tempfile
import stripe
import pymysql
import json
import secrets
import subprocess

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

# ---------------------------------------------
# DATABASE CONNECTION – AVART ORDERS
# ---------------------------------------------

def get_db_connection():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )



@app.get("/health/database")
def database_health():
    try:
        connection = pymysql.connect(
            host=os.environ["DB_HOST"],
            port=int(os.environ.get("DB_PORT", "3306")),
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            database=os.environ["DB_NAME"],
            connect_timeout=10,
            read_timeout=10,
        )

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
):
    data = upload.file.read()

    if not data:
        raise ValueError("Empty file")

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

    # Bevar den oprindelige kontur.
    # Ingen blur eller morphology.
    mask = np.where(
        alpha > alpha_threshold,
        255,
        0
    ).astype(np.uint8)

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
        body_window = min(31, smooth_window + 12)
    
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
                margin_x = fw * 0.35
                margin_y = fh * 0.20
    
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

    for contour in contours:

        # Spring selve hovedkonturen over
        if np.array_equal(contour, main_contour):
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
    # HOVEDKONTUR → SVG PATH
    # --------------------------------------------------

    d = [
        f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    ]

    for p in pts[1:]:

        d.append(
            f"L {p[0]:.2f} {p[1]:.2f}"
        )

    main_path = " ".join(d)

    # --------------------------------------------------
    # EKSTRA SVG PATHS
    # --------------------------------------------------

    extra_paths = []

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
            reference_height_mm = 500
        
            reference_top_positions_mm = {
                -2: 180,
                -1: 150,
                0: 120,
                1: 105,
                2: 90,
            }
        
        else:
            reference_height_mm = 700
        
            reference_top_positions_mm = {
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
        "light": None,
        "dark": colors.CMYKColor(0, 0, 0, 100),
        "moss": colors.CMYKColor(100, 50, 100, 50),
        "wine": colors.CMYKColor(50, 75, 75, 50),
        "camel": colors.CMYKColor(0, 25, 50, 50),
        "sea": colors.CMYKColor(100, 50, 25, 50),
    }

    if style not in style_colors:
        raise ValueError(f"Unknown poster style: {style}")

    bg_color = style_colors[style]

    # CMYK til tekst, silhuet og logo
    if style == "light":
        fg_color = colors.CMYKColor(0, 0, 0, 100)
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
        reference_height_mm = 500
    
        reference_top_positions_mm = {
            -2: 180,
            -1: 150,
            0: 120,
            1: 105,
            2: 90,
        }
    
    else:
        reference_height_mm = 700
    
        reference_top_positions_mm = {
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
            flipped = person.get("flipped", False)

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
    
                    # Første person:
                    # venstre yderkant = designmargin
                    if index == 0:
    
                        x = (
                            side_margin
                            - min_x
                        )
    
                    # Sidste person:
                    # højre yderkant = designmargin
                    elif index == count - 1:
    
                        x = (
                            width
                            - side_margin
                            - max_x
                        )
    
                    # Personerne imellem fordeles jævnt
                    # mellem designmarginerne
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
# AVART – OVERFØR PRODUKTIONSFIL TIL CHOSTING
# --------------------------------------------------

def upload_production_pdf(pdf_bytes: bytes, filename: str):

    import re

    if not re.fullmatch(r"AVA-[A-F0-9]+_[0-9]+\.pdf", filename):
        raise ValueError("Ugyldigt produktionsfilnavn.")

    remote_host = "178.20.216.107"
    remote_user = "avartdk"
    remote_folder = "/home/avartdk/avart-orders"

    ssh_key = "/etc/secrets/avart_render_new"

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:

        subprocess.run(
            [
                "scp",
                "-4",
                "-P", "222",
                "-i", ssh_key,
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                temp_path,
                f"{remote_user}@{remote_host}:{remote_folder}/{filename}",
            ],
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )

        return f"{remote_folder}/{filename}"

    finally:

        if os.path.exists(temp_path):
            os.unlink(temp_path)


# --------------------------------------------------
# AVART – PRINT PDF MED 3 MM BLEED OG SKÆREMÆRKER
# --------------------------------------------------

def make_print_ready_pdf(pdf_bytes: bytes, bleed_mm: float = 3) -> bytes:
    """
    Tilføjer 3 mm bleed, skæremærker og PDF-beskæringsbokse.
    Bevarer den eksisterende PDF som vektorgrafik.
    """
    import fitz

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

            trim = fitz.Rect(
                margin,
                margin,
                margin + w,
                margin + h,
            )

            # Indsæt original PDF som vektorgrafik
            page.show_pdf_page(trim, source, source_page.number)

            # Udvid kantfarven til bleed-området.
            # Dette udfylder bleed for plakater med farvet baggrund.
            if source_page.get_drawings():
                drawings = source_page.get_drawings()
                background = next(
                    (
                        d.get("fill")
                        for d in drawings
                        if d.get("fill") is not None
                        and d["rect"].width >= w - 1
                        and d["rect"].height >= h - 1
                    ),
                    None,
                )

                if background is not None:
                    bleed_rect = fitz.Rect(
                        trim.x0 - bleed,
                        trim.y0 - bleed,
                        trim.x1 + bleed,
                        trim.y1 + bleed,
                    )
                    page.draw_rect(
                        bleed_rect,
                        color=None,
                        fill=background,
                        overlay=False,
                    )

            # Skæremærker – placeret uden for bleed
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

            # PDF-bokse
            page.set_trimbox(trim)
            page.set_bleedbox(
                fitz.Rect(
                    trim.x0 - bleed,
                    trim.y0 - bleed,
                    trim.x1 + bleed,
                    trim.y1 + bleed,
                )
            )

        return output.tobytes(garbage=4, deflate=True)

    finally:
        output.close()
        source.close()

@app.get("/test/print-pdf")
def test_print_pdf():
    from fastapi.responses import Response

    # Enkel A3-test med farvet baggrund
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(297 * mm, 420 * mm))

    c.setFillColor(colors.CMYKColor(0, 25, 50, 50))
    c.rect(0, 0, 297 * mm, 420 * mm, fill=1, stroke=0)

    c.setFillColor(colors.CMYKColor(0, 0, 0, 100))
    c.setFont("Helvetica", 24)
    c.drawCentredString(148.5 * mm, 210 * mm, "AVART PRINT TEST")

    c.save()

    pdf_bytes = make_print_ready_pdf(buffer.getvalue())

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
    try:
        rgba, original_rgb = remove_background_if_needed(
            file1,
            max_dimension=max_dimension,
            return_original=True,
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

    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        return JSONResponse(
            {"error": "Kurven er tom eller ugyldig."},
            status_code=400,
        )

    line_items = []
    validated_items = []
    subtotal = 0

    try:
        for index, item in enumerate(items, start=1):

            size = item["size"]
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

                if person.get("scale_level", 0) not in (-2, -1, 0, 1, 2):
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

        connection.commit()

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

    # Opret Stripe Checkout efter at ordren er gemt
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,

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
                "order_id": str(order_id),
                "order_number": order_number,
            },

            billing_address_collection="required",

            shipping_address_collection={
                "allowed_countries": ["DK"]
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

            connection.commit()

        finally:
            connection.close()

        return {
            "url": session.url,
            "order_number": order_number,
        }

    except Exception:
        return JSONResponse(
            {
                "error": "Betalingen kunne ikke oprettes. Prøv igen."
            },
            status_code=500,
        )


# --------------------------------------------------
# AVART – PRODUKTIONSFILER TIL BETALTE ORDRER
# --------------------------------------------------

def produce_order_pdfs(order_id: int):
    """
    Genererer produktions-PDF'er for en betalt ordre
    og overfører dem til cHosting.
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT order_number, payment_status
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

            cursor.execute(
                """
                SELECT id, item_number, design_json
                FROM order_items
                WHERE order_id = %s
                ORDER BY item_number
                """,
                (order_id,),
            )

            items = cursor.fetchall()

        for item in items:

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

            print_pdf = make_print_ready_pdf(pdf_bytes)

            filename = (
                f"{order['order_number']}_"
                f"{item['item_number']}.pdf"
            )

            remote_path = upload_production_pdf(
                print_pdf,
                filename,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE order_items
                    SET production_pdf_path = %s
                    WHERE id = %s
                      AND order_id = %s
                    """,
                    (
                        remote_path,
                        item["id"],
                        order_id,
                    ),
                )
            
            connection.commit()

            print(
                f"AVART PRODUCTION FILE: {remote_path}",
                flush=True,
            )

    finally:
        connection.close()



# --------------------------------------------------
# STRIPE WEBHOOK – BETALINGSBEKRÆFTELSE
# --------------------------------------------------

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

    # Vi behandler kun gennemførte Checkout-sessioner
    if event["type"] != "checkout.session.completed":
        return {"received": True}

    session = event["data"]["object"]
    session = session.to_dict()

    # En gennemført Checkout-session er ikke altid betalt endnu
    if session.get("payment_status") != "paid":
        return {"received": True, "payment": "pending"}

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
                SELECT id, payment_status, total_dkk
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

            if (
                amount_total != order["total_dkk"] * 100
                or currency != "dkk"
            ):
                connection.rollback()
                return JSONResponse(
                    {"error": "Betalingsbeløbet stemmer ikke."},
                    status_code=400,
                )

            # Undgå dobbeltbehandling af samme betaling
            if order["payment_status"] == "paid":
                connection.commit()
                return {"received": True, "already_paid": True}

            customer = session.get("customer_details") or {}
            shipping = session.get("shipping_details") or {}
            address = shipping.get("address") or {}

            # Nyere Stripe-versioner kan have leveringsoplysninger
            # under collected_information
            collected = session.get("collected_information") or {}

            if not shipping:
                shipping = collected.get("shipping_details") or {}
                address = shipping.get("address") or {}

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
                    order["id"],
                ),
            )

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


