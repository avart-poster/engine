from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response, JSONResponse
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def read_upload_to_rgba(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    pil = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(pil)


def alpha_to_mask(rgba, alpha_threshold=10, smooth=True):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    # stærkere threshold (renere silhuet)
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 🔥 NYT: fjern støj
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 🔥 NYT: fjern små prikker
    mask = cv2.medianBlur(mask, 7)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def get_outer_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # største contour
    contour = max(contours, key=cv2.contourArea)
    return contour


def smooth_contour(contour, epsilon_ratio=0.001):
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def draw_contour(contour, width, height, thickness=2):
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 0, 0), thickness)

    return canvas


def to_png_bytes(img):
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------
# DEBUG (4 views)
# ---------------------------------------------------

@app.post("/alpha/debug")
async def alpha_debug(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(10),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.001),
    thickness: int = Query(2),
):
    try:
        rgba = read_upload_to_rgba(file)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(rgba, alpha_threshold, smooth)

        contour = get_outer_contour(mask)
        contour = smooth_contour(contour, epsilon_ratio)

        final = draw_contour(contour, w, h, thickness)

        # lav debug grid
        original = rgba[:, :, :3]

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        contour_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        if contour is not None:
            cv2.drawContours(contour_img, [contour], -1, (255, 0, 0), 2)

        top = np.hstack([original, mask_rgb])
        bottom = np.hstack([contour_img, final])
        grid = np.vstack([top, bottom])

        png = to_png_bytes(grid)

        return Response(content=png, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------
# PREVIEW (ren silhouette)
# ---------------------------------------------------

@app.post("/alpha/preview")
async def alpha_preview(
    file: UploadFile = File(...),
    alpha_threshold: int = Query(10),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.001),
    thickness: int = Query(2),
):
    try:
        rgba = read_upload_to_rgba(file)
        h, w = rgba.shape[:2]

        mask = alpha_to_mask(rgba, alpha_threshold, smooth)

        contour = get_outer_contour(mask)
        contour = smooth_contour(contour, epsilon_ratio)

        final = draw_contour(contour, w, h, thickness)

        png = to_png_bytes(final)

        return Response(content=png, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
