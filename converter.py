# converter.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np
import converter_core as core

app = FastAPI()

# ──────────────────────────────  CORS  ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────  /convert  ────────────────────────────
@app.post("/convert")
async def convert(
    img: UploadFile = File(...),
    thr: int        = Form(60),
    minlen: int     = Form(40),
   
    canny_low: int       = Form(50),
    canny_high: int      = Form(150),
    hough_thresh: int    = Form(50),
    maxgap: int          = Form(10),
    minpix: int          = Form(30),
):
    data  = await img.read()
    gray  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

    out = core.image_to_lines(
        gray, thr, minlen,
        canny_low, canny_high,
        hough_thresh, maxgap, minpix
    )
    return JSONResponse(out)
                    
