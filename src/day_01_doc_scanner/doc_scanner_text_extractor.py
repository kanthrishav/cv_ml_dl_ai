#!/usr/bin/env python3
import cv2
import pytesseract
import numpy as np
from picamera2 import Picamera2
from numpy import array, diff, argmin, argmax, int32

# --- Configurable parameters ---
BLUR_KSIZE       = (5,5)
CANNY_LOW, CANNY_HIGH = 50, 150
CONTOUR_APPROX_EPS= 0.02        # % of perimeter
WIDTH, HEIGHT    = 3040, 4056   # scan resolution
# -------------------------------

def order_quad(pts):
    pts = pts.reshape(4,2).astype("float32")
    s   = pts.sum(axis=1)
    d   = diff(pts, axis=1).reshape(4)
    tl  = pts[argmin(s)]
    br  = pts[argmax(s)]
    tr  = pts[argmin(d)]
    bl  = pts[argmax(d)]
    return array([tl, tr, br, bl], dtype="float32")

# initialize camera
picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
picam.configure(config)
picam.start()

try:
    while True:
        frame = picam.capture_array()

        # 1) Preprocess for edge detection
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
        edges   = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH,
                            apertureSize=3, L2gradient=True)

        # 2) Find the largest 4-point contour
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea,
                         reverse=True)[:5]
        screenCnt = None
        for c in cnts:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,
                        CONTOUR_APPROX_EPS * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        # 3) Warp the document
        if screenCnt is not None:
            pts    = screenCnt.reshape(4,2).astype("float32")
            quad   = order_quad(pts)
            dst    = array([[0,0],
                            [WIDTH-1,0],
                            [WIDTH-1,HEIGHT-1],
                            [0,HEIGHT-1]], dtype="float32")
            M      = cv2.getPerspectiveTransform(quad, dst)
            warp   = cv2.warpPerspective(frame, M,
                                         (WIDTH, HEIGHT))
        else:
            # if no quad found, skip OCR this frame
            cv2.imshow("Original", cv2.resize(frame, (720,480)))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            else:
                continue

        # 4) OCR with Tesseract
        # Convert to RGB for pytesseract
        rgb_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb_warp, output_type=pytesseract.Output.DICT
        )

        # 5) Overlay OCR text on a copy of the warp
        annotated = warp.copy()
        n_boxes = len(data["level"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = (data["left"][i],
                          data["top"][i],
                          data["width"][i],
                          data["height"][i])
            # draw bounding box (optional)
            cv2.rectangle(annotated,
                          (x, y), (x+w, y+h),
                          (0,255,0), 2)
            # overlay text just above the box
            cv2.putText(annotated, text,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2,
                        lineType=cv2.LINE_AA)

        # 6) Display side by side
        small_orig = cv2.resize(frame,   (480,720))
        small_warp = cv2.resize(warp,    (480,720))
        small_ocr  = cv2.resize(annotated,
                                (480,720))
        top = np.hstack([small_orig, small_warp])
        bot = np.hstack([np.zeros_like(small_orig), small_ocr])
        out = np.vstack([top, bot])

        cv2.imshow("Orig | Warp        Blank | OCR Annotated", out)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    picam.stop()
    cv2.destroyAllWindows()
