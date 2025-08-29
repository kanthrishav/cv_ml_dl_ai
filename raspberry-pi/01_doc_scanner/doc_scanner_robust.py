#!/usr/bin/env python3
"""
====================================================================
Document Scanner (RPi + Picamera2 + OpenCV) — Robust Quad Detection
====================================================================

Hardware: Raspberry Pi 5 (8 GB) + IMX500 (Picamera2 RGB888 stream)
Goal    : Detect document at arbitrary rotation (0–360°), warp to
          a top-down scan, and (optionally) run OCR/OSD for orientation.

Pipeline (high level)
---------------------
1) Capture -> grayscale -> CLAHE -> blur
2) Auto-Canny + morphology to stabilize edges across rotations
3) External contours only; shortlist with rotation-invariant gates
4) Force a 4-corner page polygon (hull -> approxPolyDP) and score
5) Warp via 4-point perspective transform
6) (Optional) Orientation with Tesseract OSD (debounced majority)
7) (Optional) OCR overlay side-by-side

This file intentionally uses:
- **snake_case** for function names (PEP 8),
- **camelCase** for variable names (project preference),
- **no magic numbers** in the body — all constants are defined below.

References / API notes:
- Contour hierarchy + retrieval modes (e.g., RETR_EXTERNAL)  [OpenCV].  :contentReference[oaicite:1]{index=1}
- Contour approximation via Douglas–Peucker (`approxPolyDP`)  [OpenCV].   :contentReference[oaicite:2]{index=2}
- 4-point perspective warp (`getPerspectiveTransform`)        [OpenCV].   :contentReference[oaicite:3]{index=3}
- Tesseract OSD for orientation/confidence                     [tessdoc].  :contentReference[oaicite:4]{index=4}

Author : Rishav Kanth
"""

# -------------------------------#
#            Imports             #
# -------------------------------#
import cv2
import pytesseract
from pytesseract import Output  # structured OSD / OCR outputs
import math
import numpy as np
from picamera2 import Picamera2
from numpy import array, diff, argmin, argmax, int32

# -------------------------------#
#           Constants            #
# -------------------------------#

# Camera / image canvas
WIDTH, HEIGHT                 = 4056, 3040            # scan resolution canvas
# WIDTH, HEIGHT               = 1280, 1920           # alt canvas (commented)
PREVIEW_SCALE                 = 0.20                  # preview window scale

# Preprocess (CLAHE + blur)
CLAHE_CLIP_LIMIT              = 2.0
CLAHE_TILE_GRID_SIZE          = (8, 8)
BLUR_KSIZE                    = (5, 5)

# Canny
CANNY_APERTURE_SIZE           = 3
CANNY_L2GRADIENT              = True

# Morphology to seal edges
MORPH_KSIZE                   = (5, 5)
MORPH_CLOSE_ITER              = 1
DILATE_ITER                   = 1

# Contours (retrieval/chain)
RETR_MODE                     = cv2.RETR_EXTERNAL     # external only (ignore text holes)
CHAIN_MODE                    = cv2.CHAIN_APPROX_SIMPLE

# Shortlisting / geometry gates
TOP_K                         = 10                    # shortlist size multiplier
MIN_AREA                      = 10000                 # absolute contour area gate
MIN_SIDE_FRAC                 = 1.0 / 7.0             # doc side >= 1/7 * min(W,H)
MAX_SIDE_FRAC                 = 0.50                  # doc side <= 1/2 * min(W,H)
ASPECT_MIN                    = 0.8                   # allowed aspect range
ASPECT_MAX                    = 1.5
EXTENT_MIN                    = 0.65                  # area / (bbox area)
SOLIDITY_MIN                  = 0.90                  # area / hull area
ANGLE_MIN_DEG                 = 50.0                  # min internal angle (quad plausibility)
ANGLE_MAX_DEG                 = 130.0                 # max internal angle

# Approximation parameters
EPSILON                       = 0.02                  # baseline eps (% of perimeter)
EPS_ADAPT_START               = 0.015                 # starting eps for hull-approx (absolute)
EPS_ADAPT_MULTS               = (1.0, 1.5, 2.0, 2.5, 3.5)  # adaptive multipliers
MERGE_EPS_MULTS               = (1.0, 1.5, 2.0, 2.5, 3.0)  # for merged hull
MERGE_TOP_REL_SCORE           = 0.70                  # include candidates within 70% of best

# Numerics / small guards
EPS_DENOM                     = 1e-6                  # division guard
MIN_SIZE_PX                   = 1                     # minimal positive size
MIN_WARP_DIM_PX               = 4                     # minimal warp canvas side
DEG90                         = 90.0                  # for rectangularity score calc

# Drawing / UI
DRAW_COLOR_BGR                = (255, 0, 0)
DRAW_THICKNESS                = 5
ANNOT_TEXT_COLOR              = (0, 0, 255)
ANNOT_TEXT_SCALE              = 0.5
ANNOT_TEXT_THICKNESS          = 1
WHITE_VALUE                   = 255
WINDOW_ALL_CONTOURS           = "All Contour Feed"
WINDOW_TOPK_CONTOURS          = "topk Contour Feed"
WINDOW_FEW_CONTOURS           = "few Contour Feed"
WINDOW_SELECTED_CONTOUR       = "selected Contour Feed"
WINDOW_SCAN                   = "scan"
WINDOW_OCR_ANNOT              = "OCR Annotated"

# OCR / OSD
ACTIVATE_OCR                  = True
OSD_HISTORY_LEN               = 5                      # majority window for OSD debouncing
OSD_CONF_MIN                  = 5.0                    # minimal confidence to trust OSD

# -------------------------------#
#        Helper Functions        #
# -------------------------------#

def order_quad(pts):
    """
    Order 4 points (x, y) into canonical TL, TR, BR, BL.
    This ensures a consistent mapping for perspective warp.
    """
    pts = pts.reshape(4, 2).astype("float32")
    s   = pts.sum(axis=1)                        # x + y
    d   = diff(pts, axis=1).reshape(4)           # x - y
    tl  = pts[argmin(s)]
    br  = pts[argmax(s)]
    tr  = pts[argmin(d)]
    bl  = pts[argmax(d)]
    return array([tl, tr, br, bl], dtype="float32")

def internal_angles(q):
    """
    Compute the 4 internal angles of a quadrilateral `q` (TL, TR, BR, BL).
    Used to suppress highly skewed/non-rectangular quads.
    """
    def ang(a, b, c):
        ba = a - b
        bc = c - b
        cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
    return [
        ang(q[3], q[0], q[1]),
        ang(q[0], q[1], q[2]),
        ang(q[1], q[2], q[3]),
        ang(q[2], q[3], q[0])
    ]

def auto_canny_thresholds(imgGray):
    """
    Auto-select Canny thresholds from the median intensity (robust to lighting).
    Returns (low, high).
    """
    v = np.median(imgGray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    return lower, upper

# -------------------------------#
#           Main Loop            #
# -------------------------------#

# OSD debounce state (rotation majority vote)
osdHist = []           # last N rotations
lastRot = None         # last applied rotation

# Initialize Picamera2 for RGB888 frames (works well with OpenCV)
piCam = Picamera2()
camConfig = piCam.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
piCam.configure(camConfig)
piCam.start()

smallPreview = (int(WIDTH * PREVIEW_SCALE), int(HEIGHT * PREVIEW_SCALE))

try:
    while True:
        # -------- 1) Capture + preprocessing  --------
        frameRgb   = piCam.capture_array()                           # RGB888 frame
        gray8      = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2GRAY)      # grayscale

        # Contrast Limited Adaptive Histogram Equalization (local contrast)
        claheObj   = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        grayClahe  = claheObj.apply(gray8)

        # Slight Gaussian blur to suppress noise before edges
        blurImg    = cv2.GaussianBlur(grayClahe, BLUR_KSIZE, 0)

        # -------- 2) Edges + morphology (rotation-robust)  --------
        cannyLo, cannyHi = auto_canny_thresholds(blurImg)
        edgesMap   = cv2.Canny(blurImg, cannyLo, cannyHi,
                               apertureSize=CANNY_APERTURE_SIZE,
                               L2gradient=CANNY_L2GRADIENT)

        # Close small gaps in page border; then dilate to link corners
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
        edgesMap    = cv2.morphologyEx(edgesMap, cv2.MORPH_CLOSE, morphKernel, iterations=MORPH_CLOSE_ITER)
        edgesMap    = cv2.dilate(edgesMap, morphKernel, iterations=DILATE_ITER)

        # -------- 3) Contours: external only (ignore text holes)  --------
        cntsList, hierarchyArr = cv2.findContours(edgesMap, RETR_MODE, CHAIN_MODE)

        previewAllContours = cv2.drawContours(frameRgb.copy(), cntsList, -1, DRAW_COLOR_BGR, DRAW_THICKNESS)
        previewAllContours = cv2.resize(previewAllContours, smallPreview)
        cv2.imshow(WINDOW_ALL_CONTOURS, previewAllContours)

        # Keep top candidates by area
        cntsList = sorted(cntsList, key=cv2.contourArea, reverse=True)[:TOP_K * 2]
        if hierarchyArr is None:
            hrList = np.zeros((1, len(cntsList), 4), dtype=np.int32)  # preserved structure
        else:
            hrList = np.zeros((1, len(cntsList), 4), dtype=np.int32)

        previewTopKContours = cv2.drawContours(frameRgb.copy(), cntsList, -1, DRAW_COLOR_BGR, DRAW_THICKNESS)
        previewTopKContours = cv2.resize(previewTopKContours, smallPreview)
        cv2.imshow(WINDOW_TOPK_CONTOURS, previewTopKContours)

        # -------- 4) Shortlist quads (rotation-invariant gates) --------
        quadsList = []    # (quadPoints, rectArea, score, contour)
        cntList2  = []    # for preview window
        newHrList = []

        minDim   = float(min(WIDTH, HEIGHT))
        minSide  = MIN_SIDE_FRAC * minDim
        maxSide  = MAX_SIDE_FRAC * minDim
        aspectMin, aspectMax = ASPECT_MIN, ASPECT_MAX

        for cntItem in cntsList:
            if cv2.contourArea(cntItem) < MIN_AREA:
                continue

            # Size/aspect gating via minAreaRect (rotation-invariant)
            minRect = cv2.minAreaRect(cntItem)                 # ((cx,cy),(w,h),angle)
            (cx, cy), (rw, rh), ang = minRect
            if rw < rh:
                wLen, hLen = rh, rw
            else:
                wLen, hLen = rw, rh
            if wLen < MIN_SIZE_PX or hLen < MIN_SIZE_PX:
                continue

            rectArea = wLen * hLen
            aspect   = wLen / max(hLen, EPS_DENOM)

            # Filledness/convexity to reject text strokes or grout lines
            areaContour = cv2.contourArea(cntItem)
            hullContour = cv2.convexHull(cntItem)
            solidity    = areaContour / (cv2.contourArea(hullContour) + EPS_DENOM)

            bx, by, bw, bh = cv2.boundingRect(cntItem)
            extent      = areaContour / float(bw * bh)

            # Gate: size band + aspect band + extent + solidity
            if not (minSide <= wLen <= maxSide and
                    minSide <= hLen <= maxSide and
                    aspectMin <= aspect <= aspectMax and
                    extent > EXTENT_MIN and
                    solidity > SOLIDITY_MIN):
                continue

            # Stabilize the true page polygon:
            #   approximate the convex hull, adapting epsilon until we get 4 vertices.
            periHull = cv2.arcLength(hullContour, True)
            epsStart = max(EPSILON, EPS_ADAPT_START)
            approxPoly = None
            for mul in EPS_ADAPT_MULTS:
                a = cv2.approxPolyDP(hullContour, (epsStart * mul) * periHull, True)
                if len(a) == 4:
                    approxPoly = a
                    break
            if approxPoly is None:
                # Fallback: use oriented box (OK for scoring/preview; warp still works)
                approxPoly = cv2.boxPoints(minRect).astype(np.float32).reshape(-1, 1, 2)

            quadPoints  = order_quad(approxPoly.reshape(4, 2).astype(np.float32))
            anglesList  = internal_angles(quadPoints)

            # Suppress non-rectangular quads (heavy skew/noise)
            if min(anglesList) < ANGLE_MIN_DEG or max(anglesList) > ANGLE_MAX_DEG:
                continue

            # Composite score: larger + more filled + more rectangular
            rectangularityScore = (min(anglesList) / DEG90) * (DEG90 / max(anglesList))
            scoreVal = (rectArea / (WIDTH * HEIGHT)) * extent * solidity * rectangularityScore

            quadsList.append((quadPoints, rectArea, scoreVal, cntItem))
            cntList2.append(cntItem)

        previewFewContours = cv2.drawContours(frameRgb.copy(), cntList2, -1, DRAW_COLOR_BGR, DRAW_THICKNESS)
        previewFewContours = cv2.resize(previewFewContours, smallPreview)
        cv2.imshow(WINDOW_FEW_CONTOURS, previewFewContours)

        # -------- 5) Pick best; if multiple close, merge & re-approx --------
        screenCnt = None
        selectedContour = None

        if len(quadsList) == 1:
            screenCnt = quadsList[0][0].reshape(-1, 1, 2).astype(int32)
            selectedContour = quadsList[0][3]
        elif len(quadsList) > 1:
            quadsList.sort(key=lambda t: t[2], reverse=True)
            topGroup = [q for q in quadsList if q[2] >= MERGE_TOP_REL_SCORE * quadsList[0][2]]

            mergedContour = np.vstack([q[3] for q in topGroup])
            hullMerged    = cv2.convexHull(mergedContour)
            periMerged    = cv2.arcLength(hullMerged, True)

            approxMerged = None
            for mul in MERGE_EPS_MULTS:
                am = cv2.approxPolyDP(hullMerged, (EPSILON * mul) * periMerged, True)
                if len(am) == 4:
                    approxMerged = am
                    break
            if approxMerged is None:
                approxMerged = cv2.boxPoints(cv2.minAreaRect(mergedContour)).astype(np.float32).reshape(-1, 1, 2)

            screenCnt = order_quad(approxMerged.reshape(4, 2).astype(np.float32)).reshape(-1, 1, 2).astype(int32)
            selectedContour = mergedContour
        else:
            # Fallback: try a strict 4-pt approx on any Top-K contour
            for cnt2 in cntsList:
                peri = cv2.arcLength(cnt2, True)
                a = cv2.approxPolyDP(cnt2, EPSILON * peri, True)
                if len(a) == 4:
                    screenCnt = order_quad(a.reshape(4, 2).astype(np.float32)).reshape(-1, 1, 2).astype(int32)
                    selectedContour = cnt2
                    break

        try:
            previewSelectedContour = cv2.drawContours(frameRgb.copy(), selectedContour, -1, DRAW_COLOR_BGR, DRAW_THICKNESS)
            previewSelectedContour = cv2.resize(previewSelectedContour, smallPreview)
            cv2.imshow(WINDOW_SELECTED_CONTOUR, previewSelectedContour)
        except Exception:
            pass

        # -------- 6) Perspective warp (exact 4-point mapping) --------
        if screenCnt is not None:
            ptsFloat = screenCnt.reshape(4, 2).astype("float32")
            quadPoints = order_quad(ptsFloat)

            # Average opposite sides for width/height (robust, avoids elongation)
            wA = np.linalg.norm(quadPoints[1] - quadPoints[0])
            wB = np.linalg.norm(quadPoints[2] - quadPoints[3])
            hA = np.linalg.norm(quadPoints[3] - quadPoints[0])
            hB = np.linalg.norm(quadPoints[2] - quadPoints[1])
            wEst = max(int(round((wA + wB) * 0.5)), MIN_WARP_DIM_PX)
            hEst = max(int(round((hA + hB) * 0.5)), MIN_WARP_DIM_PX)

            # Preserve detected aspect within WIDTH x HEIGHT canvas
            scale  = min(WIDTH / float(wEst), HEIGHT / float(hEst), 1.0)
            dstW   = max(int(wEst * scale), MIN_WARP_DIM_PX)
            dstH   = max(int(hEst * scale), MIN_WARP_DIM_PX)

            dstPts = array([[0, 0],
                            [dstW - 1, 0],
                            [dstW - 1, dstH - 1],
                            [0, dstH - 1]], dtype="float32")
            mPerspective = cv2.getPerspectiveTransform(quadPoints, dstPts)
            warpImage    = cv2.warpPerspective(frameRgb, mPerspective, (dstW, dstH), flags=cv2.INTER_LINEAR)
        else:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            else:
                continue

        # -------- 7) Orientation correction (OSD, debounced) --------
        if ACTIVATE_OCR:
            try:
                osdInfo = pytesseract.image_to_osd(warpImage, output_type=Output.DICT)
                rotDeg  = int(osdInfo.get("rotate", 0))                    # {0,90,180,270}
                orientConf = float(osdInfo.get("orientation_conf", 0))     # confidence scalar

                if orientConf >= OSD_CONF_MIN:  # trust only if confident
                    osdHist.append(rotDeg)
                    if len(osdHist) > OSD_HISTORY_LEN:
                        osdHist.pop(0)

                    # Majority vote across recent frames to prevent flip-flop
                    votesRot = max(set(osdHist), key=osdHist.count)
                    if votesRot != lastRot:
                        lastRot = votesRot
                        if votesRot == 90:
                            warpImage = cv2.rotate(warpImage, cv2.ROTATE_90_CLOCKWISE)
                        elif votesRot == 180:
                            warpImage = cv2.rotate(warpImage, cv2.ROTATE_180)
                        elif votesRot == 270:
                            warpImage = cv2.rotate(warpImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            except Exception:
                pass

            # -------- 8) OCR overlay (side-by-side preview) --------
            # Convert to RGB for Tesseract input
            rgbWarp = cv2.cvtColor(warpImage, cv2.COLOR_BGR2RGB)
            ocrData = pytesseract.image_to_data(rgbWarp, output_type=pytesseract.Output.DICT)

            # Build a white canvas and draw recognized words for quick visual QA
            annotatedImg = WHITE_VALUE * np.ones(warpImage.shape, dtype=warpImage.dtype)
            nBoxes = len(ocrData["level"])
            for i in range(nBoxes):
                text = ocrData["text"][i].strip()
                if not text:
                    continue
                x, y, wBox, hBox = (ocrData["left"][i],
                                    ocrData["top"][i],
                                    ocrData["width"][i],
                                    ocrData["height"][i])
                cv2.putText(annotatedImg, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            ANNOT_TEXT_SCALE, ANNOT_TEXT_COLOR, ANNOT_TEXT_THICKNESS,
                            lineType=cv2.LINE_AA)

            print(warpImage.shape)
            print(annotatedImg.shape)

            # Make sure concat preconditions hold (same dtype/rows/channels)
            if annotatedImg.ndim == 2:
                annotatedImg = cv2.cvtColor(annotatedImg, cv2.COLOR_GRAY2BGR)
            if warpImage.ndim == 2:
                warpImage = cv2.cvtColor(warpImage, cv2.COLOR_GRAY2BGR)
            if annotatedImg.shape[-1] == 4 and warpImage.shape[-1] == 3:
                annotatedImg = cv2.cvtColor(annotatedImg, cv2.COLOR_BGRA2BGR)
            elif warpImage.shape[-1] == 4 and annotatedImg.shape[-1] == 3:
                warpImage = cv2.cvtColor(warpImage, cv2.COLOR_BGRA2BGR)
            if annotatedImg.dtype != warpImage.dtype:
                annotatedImg = annotatedImg.astype(warpImage.dtype)
            if annotatedImg.shape[0] != warpImage.shape[0]:
                annotatedImg = cv2.resize(annotatedImg, (annotatedImg.shape[1], warpImage.shape[0]))

            warpImage    = np.ascontiguousarray(warpImage)
            annotatedImg = np.ascontiguousarray(annotatedImg)

            ocrPreview = cv2.hconcat([warpImage, annotatedImg])
            cv2.imshow(WINDOW_OCR_ANNOT, ocrPreview)

        # Optional on-screen size readout for tuning (commented in original)
        # cv2.putText(warpImage, f"scan: {warpImage.shape[1]}x{warpImage.shape[0]}",
        #             (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(WINDOW_SCAN, warpImage)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    piCam.stop()
    cv2.destroyAllWindows()
