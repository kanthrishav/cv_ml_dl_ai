#!/usr/bin/env python3
"""
Two‐Stage Document Scanner with Simultaneous Preview + High‐Res Scan Windows

- Preview: 640×480 continuous video for edge/quad detection
- Capture: on ‘c’, switch to 4056×3040 still, warp, enhance, show in its own window
- Preview remains running underneath
- Press ‘q’ in preview to exit
"""

import cv2
import math
import numpy as np
from picamera2 import Picamera2

# — Parameters —
PREV_W, PREV_H       = 1280, 720
FULL_W, FULL_H       = 4056, 3040
BLUR_KSIZE           = (5, 5)
CANNY_LOW, CANNY_HIGH= 50, 150
EPSILON              = 0.1       # approxPolyDP epsilon  
MIN_AREA             = 10000     # min area in preview coords
TOP_K                = 5         # how many contours to test
# ——————————

def order_quad(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).reshape(4)
    tl  = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def internal_angles(q):
    def ang(a,b,c):
        ba=a-b; bc=c-b
        cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
        return math.degrees(math.acos(max(-1,min(1,cos))))
    return [
        ang(q[3],q[0],q[1]),
        ang(q[0],q[1],q[2]),
        ang(q[1],q[2],q[3]),
        ang(q[2],q[3],q[0])
    ]

# Initialize camera
picam = Picamera2()
preview_cfg = picam.create_preview_configuration(
    main={"size": (PREV_W, PREV_H), "format": "RGB888"}
)
still_cfg   = picam.create_still_configuration(
    main={"size": (FULL_W, FULL_H), "format": "RGB888"}
)

# Start in preview mode
picam.configure(preview_cfg)
picam.start()

cv2.namedWindow("Original Preview", cv2.WINDOW_NORMAL)
cv2.namedWindow("High-Res Scan",     cv2.WINDOW_NORMAL)

try:
    while True:
        # -- 1) Grab preview frame and detect quad --
        frame = picam.capture_array()
        # cv2.imshow("High-Res Scan", frame)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH,
                         apertureSize=3, L2gradient=True)

        # find contours & approximate 4-point candidates
        cnts,_   = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts     = sorted(cnts, key=cv2.contourArea, reverse=True)[:TOP_K*2]
        quads    = []
        for c in cnts:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, EPSILON*peri, True)
            if len(approx)==4 and cv2.contourArea(approx)>MIN_AREA:
                quad = order_quad(approx)
                if min(internal_angles(quad)) > 60:
                    quads.append((quad, cv2.contourArea(approx)))
        if not quads:
            # fallback to any quadrilateral
            for c in cnts:
                approx = cv2.approxPolyDP(c, EPSILON*cv2.arcLength(c,True), True)
                if len(approx)==4:
                    quads.append((order_quad(approx), cv2.contourArea(approx)))
                    break
        # pick the largest‐area quad
        quad = quads[0][0] if quads else None

        # draw the detected quad on preview
        vis = frame.copy()
        if quad is not None:
            pts = quad.reshape(-1,1,2).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0,255,0), 2)
        cv2.imshow("Original Preview", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and quad is not None:
            # -- 2) Capture full-res still and warp --
            picam.stop()
            picam.configure(still_cfg)
            picam.start()
            full = picam.capture_array()
            picam.stop()
            picam.configure(preview_cfg)
            picam.start()

            # scale quad to full resolution
            sx, sy = FULL_W / PREV_W, FULL_H / PREV_H
            fq = quad * np.array([sx, sy], dtype="float32")

            dst = np.array([
                [0,         0],
                [FULL_W-1,  0],
                [FULL_W-1, FULL_H-1],
                [0,       FULL_H-1]
            ], dtype="float32")
            M    = cv2.getPerspectiveTransform(fq, dst)
            warp = cv2.warpPerspective(full, M, (FULL_W, FULL_H))

            # enhance: CLAHE + adaptive threshold + cleanup
            warpg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl1   = clahe.apply(warpg)
            th    = cv2.adaptiveThreshold(
                cl1, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25, C=10
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            clean  = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

            # save and preview
            cv2.imwrite("scan_fullres.png", clean)
            # downscale for display, preserving aspect ratio
            w = 800
            h = int(800 * FULL_H / FULL_W)
            preview = cv2.resize(clean, (w, h))
            cv2.imshow("High-Res Scan", preview)
            cv2.waitKey(0)
        elif key == ord('q'):
            break

finally:
    picam.stop()
    cv2.destroyAllWindows()
