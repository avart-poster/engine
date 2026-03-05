from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import numpy as np
import cv2

app = FastAPI(title="Avart Engine")

@app.get("/")
def root():
    return {"ok": True, "service": "avart-engine"}

@app.get("/health")
def health():
    return {"ok": True}

def _read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload JPG/PNG.")
    return img

@app.post("/stroke/preview", response_class=Response)
async def stroke_preview(
    file: UploadFile = File(...),
    stroke_px: int = 3,
):
    """
    Upload a portrait (JPG/PNG) and get a simple contour stroke preview (PNG).
    This is a *starter* endpoint — we will improve quality next.
    """
    data = await file.read()
    img = _read_upload_to_bgr(data)

    # --- Simple contour extraction (starter) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 40, 120)

    # Make lines thicker
    k = max(1, int(stroke_px))
    kernel = np.ones((k, k), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert to black lines on transparent background (RGBA)
    h, w = edges.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 0  # fully transparent
    rgba[edges > 0] = (0, 0, 0, 255)  # black opaque where edges are

    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        return Response(status_code=500, content=b"Could not encode PNG")

    return Response(content=png.tobytes(), media_type="image/png")
