from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io

import numpy as np
import cv2
from rembg import remove, new_session

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

import cairosvg


app = FastAPI(title="avart-engine")

MAX_DIMENSION = 1600
REMBG_MODEL = "u2net"
_rembg_session = None


# ---------------------------------
# Helpers
# ---------------------------------

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(REMBG_MODEL)
    return _rembg_session


def resize_if_needed_rgba(img: np.ndarray, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_dimension:
        return img

    scale = max_dimension / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def remove_background_if_needed(upload: UploadFile, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image")

    # Hvis billedet allerede har alpha og faktisk transparency, så brug det direkte
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < 250):
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return resize_if_needed_rgba(rgba, max_dimension=max_dimension)

    # Resize før rembg
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

    # Fjern baggrund
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

    # Ekstra transparent bund
    bottom_pad = 180
    img_out = cv2.copyMakeBorder(
        img_out,
        0,
        bottom_pad,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )

    rgba = cv2.cvtColor(img_out, cv2.COLOR_BGRA2RGBA)
    return resize_if_needed_rgba(rgba, max_dimension=max_dimension)


def alpha_to_mask(rgba: np.ndarray, alpha_threshold: int = 1) -> np.ndarray:
    alpha = rgba[:, :, 3]
    return (alpha > alpha_threshold).astype(np.uint8) * 255


def get_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found")
    return max(contours, key=cv2.contourArea)


def smooth_contour(contour: np.ndarray, epsilon_ratio: float = 0.002):
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * peri
    return cv2.approxPolyDP(contour, epsilon, True)


def contour_to_svg(contour: np.ndarray, width: int, height: int, stroke_width: float = 3.5) -> str:
    pts = contour[:, 0, :]
    path = "M " + " L ".join([f"{x},{y}" for x, y in pts])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{path}" fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />
</svg>"""
    return svg


def generate_poster(
    svg_string: str,
    name: str,
    bg_color: str = "#e9e3db",
    margin_mm: int = 20,
    logo_path: str | None = None,
) -> bytes:
    # SVG -> PNG bytes
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"))

    # PNG -> PDF
    buffer = io.BytesIO()
    page_w, page_h = A4
    c = canvas.Canvas(buffer, pagesize=A4)

    # Baggrund
    c.setFillColor(colors.HexColor(bg_color))
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    margin = margin_mm * mm

    # Titel
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, page_h - margin - 8, name)

    # Motiv
    image = ImageReader(io.BytesIO(png_bytes))
    img_w, img_h = image.getSize()

    max_draw_w = page_w - (2 * margin)
    max_draw_h = page_h - (2 * margin) - 90

    scale = min(max_draw_w / img_w, max_draw_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    x = (page_w - draw_w) / 2
    y = margin + 25

    c.drawImage(image, x, y, width=draw_w, height=draw_h, mask="auto")

    # Logo / fallback tekst
    if logo_path:
        try:
            logo = ImageReader(logo_path)
            logo_w, logo_h = logo.getSize()
            target_w = 40 * mm
            target_h = target_w * (logo_h / logo_w)
            logo_x = (page_w - target_w) / 2
            logo_y = margin - 2
            c.drawImage(logo, logo_x, logo_y, width=target_w, height=target_h, mask="auto")
        except Exception:
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(page_w / 2, margin - 2 + 8, "avart")
    else:
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(page_w / 2, margin - 2 + 8, "avart")

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------------------------
# API
# ---------------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


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
async def poster_pdf(
    file: UploadFile = File(...),
    name: str = Query("Test"),
    stroke_width: float = Query(3.5),
    bg_color: str = Query("#e9e3db"),
    margin_mm: int = Query(20, ge=0, le=60),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=MAX_DIMENSION)

        mask = alpha_to_mask(rgba)
        contour = get_contour(mask)
        contour = smooth_contour(contour)

        h, w = rgba.shape[:2]
        svg = contour_to_svg(contour, w, h, stroke_width)

        pdf_bytes = generate_poster(
            svg,
            name=name,
            bg_color=bg_color,
            margin_mm=margin_mm,
            logo_path=None,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{name}.pdf"'
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
