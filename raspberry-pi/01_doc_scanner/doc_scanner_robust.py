#!/usr/bin/env python3
import cv2
import pytesseract
from pytesseract import Output  # FIX: structured OSD output
import math
import numpy as np
from picamera2 import Picamera2
from numpy import array, diff, argmin, argmax, int32

# --- Configurable parameters (kept) ---
BLUR_KSIZE       = (5,5)
CANNY_LOW, CANNY_HIGH = 50, 150
CONTOUR_APPROX_EPS= 0.02
WIDTH, HEIGHT    = 4056, 3040
# WIDTH, HEIGHT  = 1280, 1920
Top_K = 10
MIN_AREA = 10000
EPSILON = 0.02
activate_OCR = True
# --------------------------------------

def order_quad(pts):
    pts = pts.reshape(4,2).astype("float32")
    s   = pts.sum(axis=1)
    d   = diff(pts, axis=1).reshape(4)
    tl  = pts[argmin(s)]
    br  = pts[argmax(s)]
    tr  = pts[argmin(d)]
    bl  = pts[argmax(d)]
    return array([tl, tr, br, bl], dtype="float32")

def internal_angles(q):
    def ang(a,b,c):
        ba=a-b; bc=c-b
        cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
        return math.degrees(math.acos(max(-1,min(1,cos))))
    return [ ang(q[3],q[0],q[1]),
             ang(q[0],q[1],q[2]),
             ang(q[1],q[2],q[3]),
             ang(q[2],q[3],q[0]) ]

def auto_canny_thresholds(img_gray):
    v = np.median(img_gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    return lower, upper

# FIX: small OSD debounce state
osd_hist = []        # last N rotations
OSD_N = 5
last_rot = None

# initialize camera
picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
picam.configure(config)
picam.start()
smallPreview = ((int)(WIDTH*0.2), (int)(HEIGHT*0.2))

try:
    while True:
        frame = picam.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)      # Picamera2 manual recommends RGB888 for OpenCV

        # --- Contrast + noise handling ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        grayc = clahe.apply(gray)
        blur  = cv2.GaussianBlur(grayc, BLUR_KSIZE, 0)

        # --- Auto-Canny (robust to illumination) ---
        lo, hi = auto_canny_thresholds(blur)
        edges  = cv2.Canny(blur, lo, hi, apertureSize=3, L2gradient=True)

        # --- Close gaps at steep rotations; join corners ---
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
        edges = cv2.dilate(edges, k, iterations=1)

        # Contours: external only (ignore text holes)
        cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours1 = cv2.drawContours(frame.copy(), cnts, -1, (255, 0, 0), 5)
        contours1 = cv2.resize(contours1, smallPreview)
        cv2.imshow("All Contour Feed", contours1)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:Top_K*2]
        if hierarchy is None:
            hr = np.zeros((1, len(cnts), 4), dtype=np.int32)
        else:
            hr = np.zeros((1, len(cnts), 4), dtype=np.int32)  # keep structure

        contours2 = cv2.drawContours(frame.copy(), cnts, -1, (255, 0, 0), 5)
        contours2 = cv2.resize(contours2, smallPreview)
        cv2.imshow("topk Contour Feed", contours2)

        quads = []
        cnt = []
        newhr = []

        min_dim   = float(min(WIDTH, HEIGHT))
        min_side  = (1.0/7.0) * min_dim   # your constraint
        max_side  = 0.50 * min_dim        # your constraint
        min_aspr, max_aspr = 0.8, 1.5

        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue

            # rotation-invariant size filter via minAreaRect (gating only)
            rect = cv2.minAreaRect(c)                       # ((cx,cy),(w,h),angle)
            (cx, cy), (rw, rh), ang = rect
            if rw < rh: w, h = rh, rw
            else:       w, h = rw, rh
            if w < 1 or h < 1: 
                continue
            rect_area = w*h
            aspr      = w / max(h, 1e-6)

            # contour solidity/extent to reject text & thin lines
            area_     = cv2.contourArea(c)
            hull      = cv2.convexHull(c)
            solidity  = area_ / (cv2.contourArea(hull)+1e-6)

            x,y,ww,hh = cv2.boundingRect(c)
            extent    = area_ / float(ww*hh)

            if not (min_side <= w <= max_side and min_side <= h <= max_side and
                    min_aspr <= aspr <= max_aspr and extent > 0.65 and solidity > 0.90):
                continue

            # --------- FIX: get the TRUE page quad (not a bounding box) ----------
            # 1) stabilize shape by approximating the convex hull, not raw contour
            peri_h = cv2.arcLength(hull, True)
            # adapt epsilon until we get 4 points (cap it to avoid over-smoothing)
            eps = max(EPSILON, 0.015)
            approx = None
            for k in [eps, eps*1.5, eps*2.0, eps*2.5, eps*3.5]:
                a = cv2.approxPolyDP(hull, k*peri_h, True)
                if len(a) == 4:
                    approx = a
                    break
            if approx is None:
                # fall back: use boxPoints only for scoring, not for warp
                approx = cv2.boxPoints(rect).astype(np.float32).reshape(-1,1,2)

            quad = order_quad(approx.reshape(4,2).astype(np.float32))
            angles = internal_angles(quad)

            if min(angles) < 50 or max(angles) > 130:
                # too distorted to be a page (perspective + noise), reject
                continue

            # prefer larger, more filled, more rectangular candidates
            rectangularity = min(angles)/90.0 * (90.0/max(angles))
            score = (rect_area / (WIDTH*HEIGHT)) * extent * solidity * rectangularity
            quads.append((quad, rect_area, score, c))
            cnt.append(c)

        contours3 = cv2.drawContours(frame.copy(), cnt, -1, (255, 0, 0), 5)
        contours3 = cv2.resize(contours3, smallPreview)
        cv2.imshow("few Contour Feed", contours3)

        # choose best quad; if multiple, merge and re-approx hull->4 points
        screenCnt = None
        c = None
        if len(quads) == 1:
            screenCnt = quads[0][0].reshape(-1,1,2).astype(int32)
            c = quads[0][3]
        elif len(quads) > 1:
            quads.sort(key=lambda t: t[2], reverse=True)
            top = [q for q in quads if q[2] >= 0.7*quads[0][2]]
            merged = np.vstack([q[3] for q in top])
            hull_m = cv2.convexHull(merged)
            peri_m = cv2.arcLength(hull_m, True)
            approx_m = None
            for k in [EPSILON, EPSILON*1.5, EPSILON*2.0, EPSILON*2.5, EPSILON*3.0]:
                am = cv2.approxPolyDP(hull_m, k*peri_m, True)
                if len(am) == 4:
                    approx_m = am
                    break
            if approx_m is None:
                approx_m = cv2.boxPoints(cv2.minAreaRect(merged)).astype(np.float32).reshape(-1,1,2)
            screenCnt = order_quad(approx_m.reshape(4,2).astype(np.float32)).reshape(-1,1,2).astype(int32)
            c = merged
        else:
            # last resort: try a strict 4-pt approx on any of the Top_K
            for c2 in cnts:
                peri   = cv2.arcLength(c2, True)
                a = cv2.approxPolyDP(c2, CONTOUR_APPROX_EPS * peri, True)
                if len(a) == 4:
                    screenCnt = order_quad(a.reshape(4,2).astype(np.float32)).reshape(-1,1,2).astype(int32)
                    c = c2
                    break

        try:
            contours4 = cv2.drawContours(frame.copy(), c, -1, (255, 0, 0), 5)
            contours4 = cv2.resize(contours4, smallPreview)
            cv2.imshow("selected Contour Feed", contours4)
        except:
            pass

        # -------------------------- Warp (exact) ---------------------------
        if screenCnt is not None:
            pts  = screenCnt.reshape(4,2).astype("float32")
            quad = order_quad(pts)

            # robust size from averaged opposite sides (no elongation/shortening)
            wA = np.linalg.norm(quad[1] - quad[0])
            wB = np.linalg.norm(quad[2] - quad[3])
            hA = np.linalg.norm(quad[3] - quad[0])
            hB = np.linalg.norm(quad[2] - quad[1])
            w_est = max(int(round((wA + wB)*0.5)), 4)
            h_est = max(int(round((hA + hB)*0.5)), 4)

            # constrain to canvas but preserve detected aspect
            scale = min(WIDTH/float(w_est), HEIGHT/float(h_est), 1.0)
            dst_w = max(int(w_est*scale), 4)
            dst_h = max(int(h_est*scale), 4)

            dst  = array([[0,0],
                          [dst_w-1,0],
                          [dst_w-1,dst_h-1],
                          [0,dst_h-1]], dtype="float32")
            M    = cv2.getPerspectiveTransform(quad, dst)   # true perspective warp
            warp = cv2.warpPerspective(frame, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
        else:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            else:
                continue

        # ----------------- Orientation correction (debounced) --------------
        if activate_OCR:
            try:
                osd = pytesseract.image_to_osd(warp, output_type=Output.DICT)
                rot = int(osd.get("rotate", 0))
                conf = float(osd.get("orientation_conf", 0))
                if conf >= 5:  # small threshold to avoid flip-flop
                    osd_hist.append(rot)
                    if len(osd_hist) > OSD_N:
                        osd_hist.pop(0)
                    # majority vote across last N
                    votes = max(set(osd_hist), key=osd_hist.count)
                    if votes != last_rot:
                        last_rot = votes
                        if votes == 90:
                            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
                        elif votes == 180:
                            warp = cv2.rotate(warp, cv2.ROTATE_180)
                        elif votes == 270:
                            warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            except Exception:
                pass

            # Convert to RGB for pytesseract
            rgb_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(
                rgb_warp, output_type=pytesseract.Output.DICT
            )

            # 5) Overlay OCR text on a copy of the warp
            # annotated = warp.copy()
            annotated = 255*np.ones(warp.shape)
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
                # cv2.rectangle(annotated,
                #               (x, y), (x+w, y+h),
                #               (255,0,0), 1)
                # overlay text just above the box
                cv2.putText(annotated, text,
                            # (x, y-10),
                            (x, y),
                            # 0.5,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 1,
                            lineType=cv2.LINE_AA)
            print(warp.shape)
            print(annotated.shape)
            if annotated.ndim == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

            if warp.ndim == 2:
                warp = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR)

            # If one accidentally became 4-channel (BGRA), drop alpha
            if annotated.shape[-1] == 4 and warp.shape[-1] == 3:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)
            elif warp.shape[-1] == 4 and annotated.shape[-1] == 3:
                warp = cv2.cvtColor(warp, cv2.COLOR_BGRA2BGR)

            # Enforce same dtype
            if annotated.dtype != warp.dtype:
                annotated = annotated.astype(warp.dtype)

            # Enforce same number of rows
            if annotated.shape[0] != warp.shape[0]:
                annotated = cv2.resize(annotated, (annotated.shape[1], warp.shape[0]))

            # Ensure contiguous
            warp = np.ascontiguousarray(warp)
            annotated = np.ascontiguousarray(annotated)

            ocrPreview = cv2.hconcat([warp, annotated])
            cv2.imshow("OCR Annotated", ocrPreview)

        # # optional overlay for tuning
        # cv2.putText(warp, f"scan: {warp.shape[1]}x{warp.shape[0]}", (10, 24),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("scan", warp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    picam.stop()
    cv2.destroyAllWindows()