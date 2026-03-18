from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
import numpy as np
import cv2
from rembg import remove, new_session

app = FastAPI()

MAX_DIMENSION = 1600
REMBG_MODEL = "u2net"

_rembg_session = None


# ------------------------
# Helpers
# ------------------------

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(REMBG_MODEL)
    return _rembg_session


def resize_if_needed_rgba(img: np.ndarray, max_dimension: int):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img

    scale = max_dimension / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def remove_background_if_needed(upload: UploadFile, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image")

    # Hvis allerede transparent
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

    # Ekstra bund
    bottom_pad = 180
    img_out = cv2.copyMakeBorder(
        img_out,
        0,
        bottom_pad,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0)
    )

    rgba = cv2.cvtColor(img_out, cv2.COLOR_BGRA2RGBA)
    return resize_if_needed_rgba(rgba, max_dimension=max_dimension)


def alpha_to_mask(rgba: np.ndarray, alpha_threshold: int = 1):
    alpha = rgba[:, :, 3]
    return (alpha > alpha_threshold).astype(np.uint8) * 255


def get_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found")
    return max(contours, key=cv2.contourArea)


def smooth_contour(contour, epsilon_ratio=0.002):
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * peri
    return cv2.approxPolyDP(contour, epsilon, True)


def contour_to_svg(contour, width, height, stroke_width=3):
    pts = contour[:, 0, :]
    path = "M " + " L ".join([f"{x},{y}" for x, y in pts]) + " Z"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<path d="{path}" fill="none" stroke="black" stroke-width="{stroke_width}" />
</svg>"""
    return svg


def generate_poster(svg: str, name: str):
    return f"""<html>
<body style="background:#e9e3db; display:flex; justify-content:center;">
<div style="width:600px; text-align:center; font-family:sans-serif;">
<h2>{name}</h2>
{svg}
<p style="margin-top:40px;">avart</p>
</div>
</body>
</html>""".encode("utf-8")


# ------------------------
# API
# ------------------------

@app.post("/poster/pdf")
async def poster_pdf(
    file: UploadFile = File(...),
    name: str = "Test",
    stroke_width: float = Query(3.5),
):
    try:
        rgba = remove_background_if_needed(file)

        mask = alpha_to_mask(rgba)
        contour = get_contour(mask)
        contour = smooth_contour(contour)

        h, w = rgba.shape[:2]

        svg = contour_to_svg(contour, w, h, stroke_width)

        # 🔥 generate PDF (skal returnere bytes!)
        pdf_bytes = generate_poster(svg, name)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{name}.pdf"'
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
