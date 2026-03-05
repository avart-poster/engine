from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2

app = FastAPI(title="avart-engine")

# ✅ CORS så avart.dk kan kalde din Render-service fra browseren
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://avart.dk",
        "https://www.avart.dk",
        "http://localhost:5500",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Try JPG/PNG.")
    return img


def _stroke_contour(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple + robust first version:
    - assumes light background
    - finds main silhouette/subject edge
    - returns a clean black stroke on white background
    """
    h, w = img_bgr.shape[:2]

    # grayscale + denoise
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive threshold (works ok even if light isn't perfect)
    # subject becomes white (foreground) on black
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )

    # cleanup small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    # find contours
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # return blank
        return np.full((h, w, 3), 255, dtype=np.uint8)

    # pick largest contour (main subject)
    c = max(contours, key=cv2.contourArea)

    # simplify slightly (smoother line)
    eps = 0.0025 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)

    # render stroke
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.drawContours(out, [c], -1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    return out


@app.post("/stroke/preview")
async def stroke_preview(file: UploadFile = File(...)):
    """
    Upload image -> returns PNG preview (stroke on white)
    """
    try:
        data = await file.read()
        img = _read_image_from_upload(data)
        out = _stroke_contour(img)

        ok, png = cv2.imencode(".png", out)
        if not ok:
            return JSONResponse({"ok": False, "error": "PNG encode failed"}, status_code=500)

        return Response(content=png.tobytes(), media_type="image/png")

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
