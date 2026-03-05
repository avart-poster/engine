from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2


app = FastAPI(title="Avart Engine")

# allow calls from avart website later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"hello": "world"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/stroke/preview")
async def stroke_preview(file: UploadFile = File(...)):

    contents = await file.read()

    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 140)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    canvas = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

    if contours:
        biggest = max(contours, key=cv2.contourArea)

        cv2.drawContours(
            canvas,
            [biggest],
            -1,
            (0, 0, 0, 255),
            4
        )

    _, png = cv2.imencode(".png", canvas)

    return Response(
        content=png.tobytes(),
        media_type="image/png"
    )
