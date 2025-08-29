#!/usr/bin/env python3
"""
===============================================================================
Brand Localizer (100 cm range) — Picamera2 + SIFT/FLANN + Homography
-------------------------------------------------------------------------------
Purpose:
    - Detect multiple brand logos in a live camera feed using template images.
    - Count each brand instance and draw stable, axis-aligned rectangles.
    - Keep the code behavior IDENTICAL to the original (no functional changes).

What this script does:
    1) Continuously captures frames from a Raspberry Pi camera via Picamera2.
    2) Loads templates (one image per brand) and computes SIFT descriptors.
    3) For each frame:
        - Extracts SIFT features.
        - Performs mutual + ratio-tested matches to each template.
        - Spatially clusters matches and estimates homography per cluster.
        - Validates with quality gates; draws rectangles over valid hits.
        - Applies NMS + center-merge de-duplication and light tracking.
        - Overlays per-brand counts and FPS.

Notes:
    - All constants are defined below in CAPITAL_CASE.
    - All function names are snake_case.
    - All variables (non-constants) use camelCase.
    - No functional changes were made; this is a commented, cleaned version.

Author: Rishav Kanth
===============================================================================
"""

import os
import cv2
import time
import threading
import numpy as np
from picamera2 import Picamera2

# =============================================================================
# CONFIGURATION CONSTANTS (ALL CAPS)
# =============================================================================

# Paths / window titles
TEMPLATE_DIR = "templates"
WINDOW_NAME_MAIN = "Localization"
WINDOW_NAME_TEMPLATE_PREFIX = "Template: "

# Camera configuration
DETECT_W, DETECT_H = 1280, 720
FPS_TARGET = 30
CAMERA_WARMUP_SEC = 1.0
NO_FRAME_SLEEP_SEC = 0.01

# Visualization settings
DRAW_COLOR_BOX = (0, 255, 0)          # rectangle color (green)
DRAW_THICKNESS = 3                    # rectangle thickness
TEXT_COLOR_COUNT = (255, 255, 0)      # cyan-yellow-ish for counts
TEXT_COLOR_FPS = (0, 255, 255)        # yellow for FPS
TEXT_SCALE_COUNT = 1.0
TEXT_SCALE_FPS = 0.8
TEXT_THICKNESS = 2
QUIT_KEY = ord('q')

# SIFT / FLANN feature extraction & matching
SIFT_FEATURES = 900
RATIO_TEST = 0.80                      # typical 0.7–0.8
FLANN_TREES = 5
FLANN_CHECKS = 100                     # more exhaustive search
DRAW_KP_FLAGS = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

# Match clustering
MIN_CLUSTER_MATCH = 6                  # base floor; adapted per-template
CLUSTER_RADIUS = 50
USE_CLOSING = True
CLOSE_KERNEL = 11

# Homography sanity checks
RANSAC_THRESH = 5.0                    # px reprojection threshold
MIN_INLIERS_ABS = 6
INLIER_RATIO_MIN = 0.30
MAX_REPROJ_ERR = 4.0                   # px average inlier error

# Box de-duplication & geometry guards
IOU_NMS_THRESH = 0.30
CENTER_MERGE_FRAC = 0.45               # center-distance merge
ASPECT_TOL = 0.55                      # aspect ratio guard vs template

# Lightweight tracking for rectangle stability
SMOOTH_ALPHA = 0.6
IOU_ASSOC_THRESH = 0.45
MISS_TTL_FRAMES = 6
APPEAR_MIN_HITS = 1
SIZE_JUMP_MAX = 1.8

# =============================================================================
# SHARED STATE (GLOBALS)
# =============================================================================
latestFrame, stopCapture = None, False
frameLock = threading.Lock()


# =============================================================================
# CAMERA THREAD
# =============================================================================
def camera_thread():
    """
    Capture frames continuously from the Pi camera and update a shared buffer.

    The camera is configured to produce 1280x720 RGB frames. Frames are stored
    into a global variable (latestFrame) protected by a lock for thread safety.
    """
    global latestFrame, stopCapture

    piCam = Picamera2()
    cfg = piCam.create_video_configuration(
        main={"size": (DETECT_W, DETECT_H), "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET},
    )
    piCam.configure(cfg)
    piCam.start()

    # Allow sensor/ISP to settle
    time.sleep(CAMERA_WARMUP_SEC)

    while not stopCapture:
        frame = piCam.capture_array()  # RGB for display
        with frameLock:
            latestFrame = frame

    piCam.stop()


# =============================================================================
# IMAGE PREP
# =============================================================================
def prep_gray_bgr(imgBgr):
    """
    Convert a BGR image to grayscale and apply CLAHE to improve keypoint
    detection on low-contrast regions.
    """
    gray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# =============================================================================
# TEMPLATE LOADING
# =============================================================================
def load_templates():
    """
    Load all templates from TEMPLATE_DIR, scale them relative to detection
    resolution, compute SIFT keypoints/descriptors, and create a FLANN matcher
    for each template.

    Returns:
        List[dict]: Each dict contains:
            name, img, kp, des, w, h, matcher, ar
    """
    templates = []
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)

    for fileName in sorted(os.listdir(TEMPLATE_DIR)):
        if not fileName.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(TEMPLATE_DIR, fileName)
        imgHigh = cv2.imread(path)  # BGR
        if imgHigh is None:
            continue

        hHigh, wHigh = imgHigh.shape[:2]
        scale = min(DETECT_W / wHigh, DETECT_H / hHigh)
        wScaled, hScaled = int(wHigh * scale), int(hHigh * scale)
        imgScaled = cv2.resize(imgHigh, (wScaled, hScaled), interpolation=cv2.INTER_AREA)
        gray = prep_gray_bgr(imgScaled)

        kp, des = sift.detectAndCompute(gray, None)

        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=FLANN_TREES),
            dict(checks=FLANN_CHECKS),
        )

        templates.append(
            {
                "name": os.path.splitext(fileName)[0],
                "img": imgScaled,
                "kp": kp,
                "des": des,
                "w": wScaled,
                "h": hScaled,
                "matcher": matcher,
                "ar": wScaled / float(hScaled),
            }
        )

        # (One-time) visualize template keypoints
        kpVis = cv2.drawKeypoints(
            imgScaled,
            kp,
            None,
            color=(0, 255, 0),
            flags=DRAW_KP_FLAGS,
        )
        cv2.imshow(f"{WINDOW_NAME_TEMPLATE_PREFIX}{fileName}", kpVis)
        cv2.waitKey(1)

    return templates


# =============================================================================
# HELPERS
# =============================================================================
def clamp_int(v, lo, hi):
    """Clamp v to [lo, hi], returning an int."""
    return max(lo, min(hi, int(v)))


def reproj_error(H, src, dst, mask):
    """
    Compute the average Euclidean reprojection error for inlier correspondences.

    Args:
        H (np.ndarray): Homography matrix.
        src (np.ndarray): Source points (Nx1x2).
        dst (np.ndarray): Destination points (Nx1x2).
        mask (np.ndarray): Inlier mask from RANSAC.

    Returns:
        float: Mean reprojection error (pixels). Large values indicate poor fit.
    """
    if H is None or mask is None:
        return 1e9
    src2 = cv2.perspectiveTransform(src, H)
    inliers = mask.ravel().astype(bool)
    if not np.any(inliers):
        return 1e9
    diff = src2[inliers] - dst[inliers]
    return float(np.mean(np.linalg.norm(diff, axis=2)))


def quad_ok(quad, imgW, imgH, tplW, tplH):
    """
    Basic quad sanity checks:
      - Within a small boundary outside the image.
      - Area not absurdly small or covering most of the frame.

    Args:
        quad (np.ndarray): 4x1x2 float32 quad points in the scene.
        imgW, imgH (int): Scene width and height.
        tplW, tplH (int): Template width/height (scaled).

    Returns:
        bool: True if the quad passes sanity checks.
    """
    pts = quad.reshape(4, 2).astype(np.float32)

    # Allow small overshoot to cope with rounding
    if (pts[:, 0] < -0.08 * imgW).any() or (pts[:, 0] > 1.08 * imgW).any():
        return False
    if (pts[:, 1] < -0.08 * imgH).any() or (pts[:, 1] > 1.08 * imgH).any():
        return False

    area = abs(cv2.contourArea(pts.astype(np.int32)))
    if area < 0.002 * (tplW * tplH) or area > 0.80 * (imgW * imgH):
        return False

    return True


def iou(boxA, boxB):
    """Intersection-over-Union for two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, thr=IOU_NMS_THRESH):
    """Greedy NMS that keeps boxes with IoU less than a threshold."""
    kept = []
    for box in boxes:
        if all(iou(box, k) < thr for k in kept):
            kept.append(box)
    return kept


def center_merge_boxes(boxes, frac=CENTER_MERGE_FRAC):
    """
    Merge boxes whose centers are within a fraction of the min side of either box.

    Ordered by area (largest first) to keep the most stable box.
    """
    if not boxes:
        return boxes

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3], reverse=True)
    keep = []

    for i in order:
        xi, yi, wi, hi = boxes[i]
        cxi, cyi = xi + wi / 2.0, yi + hi / 2.0
        ri = frac * min(wi, hi)

        shouldKeep = True
        for xj, yj, wj, hj in keep:
            cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
            rj = frac * min(wj, hj)
            if abs(cxi - cxj) <= max(ri, rj) and abs(cyi - cyj) <= max(ri, rj):
                shouldKeep = False
                break

        if shouldKeep:
            keep.append(boxes[i])

    return keep


def ema_box(prevBox, newBox, alpha=SMOOTH_ALPHA):
    """Exponential moving average for a [x, y, w, h] box."""
    return tuple(alpha * np.array(newBox) + (1 - alpha) * np.array(prevBox))


def mutual_ratio_matches(desT, desS, matcher):
    """
    Mutual (reciprocal) nearest-neighbor + Lowe ratio test.

    Args:
        desT, desS: Template and scene descriptors.
        matcher:    FLANN matcher (pre-constructed).

    Returns:
        List[cv2.DMatch]: "Good" matches that pass mutual & ratio tests.
    """
    if desT is None or desS is None or len(desT) == 0 or len(desS) == 0:
        return []

    rawTs = matcher.knnMatch(desT, desS, k=2)
    rawSt = matcher.knnMatch(desS, desT, k=1)

    # scene->template best mapping
    bestTForScene = {}
    for lst in rawSt:
        if len(lst) > 0:
            m = lst[0]
            bestTForScene[m.queryIdx] = m.trainIdx

    good = []
    for pair in rawTs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            if bestTForScene.get(m.trainIdx, -1) == m.queryIdx:
                good.append(m)

    return good


# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    """
    Entry point: start camera thread, load templates, and process frames.
    """
    global stopCapture, latestFrame

    templates = load_templates()
    if not templates:
        print("No templates found in", TEMPLATE_DIR)
        return

    # Track dictionary per brand name: list of {bbox, hits, miss, area_ema}
    tracks = {tpl["name"]: [] for tpl in templates}

    # Start camera capture thread
    threading.Thread(target=camera_thread, daemon=True).start()

    # SIFT for scene features (same settings as templates)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)

    cv2.namedWindow(WINDOW_NAME_MAIN, cv2.WINDOW_NORMAL)
    prevTime = time.time()
    fps = 0.0

    try:
        while True:
            # Acquire the latest frame snapshot
            with frameLock:
                frame = None if latestFrame is None else latestFrame.copy()
            if frame is None:
                time.sleep(NO_FRAME_SLEEP_SEC)
                continue

            # Display uses RGB; processing uses BGR/GRAY
            frameBgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            sceneGray = prep_gray_bgr(frameBgr)
            kpScene, desScene = sift.detectAndCompute(sceneGray, None)

            vis = frame.copy()
            yText = 30
            imgH, imgW = sceneGray.shape  # (H, W)

            for tpl in templates:
                name = tpl["name"]
                kpTpl, desTpl, matcher = tpl["kp"], tpl["des"], tpl["matcher"]
                wTpl, hTpl, arTpl = tpl["w"], tpl["h"], tpl["ar"]

                # Adaptive minimum match count per template
                minCluster = max(MIN_CLUSTER_MATCH, int(0.015 * max(1, len(kpTpl))))

                # 1) Robust mutual + ratio matches
                goodMatches = mutual_ratio_matches(desTpl, desScene, matcher)

                detBoxes = []
                if len(goodMatches) >= minCluster:
                    # 2) Cluster matches via dilation bubbles and connected components
                    mask = np.zeros((DETECT_H, DETECT_W), dtype=np.uint8)
                    for m in goodMatches:
                        x, y = kpScene[m.trainIdx].pt
                        cx = clamp_int(round(x), 0, DETECT_W - 1)
                        cy = clamp_int(round(y), 0, DETECT_H - 1)
                        cv2.circle(mask, (cx, cy), CLUSTER_RADIUS, 255, -1)

                    if USE_CLOSING and CLOSE_KERNEL >= 3:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                    nLabels, labels = cv2.connectedComponents(mask)

                    # 3) Homography -> rectangle with quality gates
                    for lbl in range(1, nLabels):
                        cluster = []
                        for m in goodMatches:
                            x, y = kpScene[m.trainIdx].pt
                            xi = clamp_int(round(x), 0, DETECT_W - 1)
                            yi = clamp_int(round(y), 0, DETECT_H - 1)
                            if labels[yi, xi] == lbl:
                                cluster.append(m)
                        if len(cluster) < minCluster:
                            continue

                        src = np.float32([kpTpl[m.queryIdx].pt for m in cluster]).reshape(-1, 1, 2)
                        dst = np.float32([kpScene[m.trainIdx].pt for m in cluster]).reshape(-1, 1, 2)

                        H, maskH = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                        if H is None or maskH is None:
                            continue

                        inliers = int(maskH.ravel().sum())
                        if inliers < MIN_INLIERS_ABS or (inliers / len(cluster)) < INLIER_RATIO_MIN:
                            continue

                        if reproj_error(H, src, dst, maskH) > MAX_REPROJ_ERR:
                            continue

                        # Warp template corners and build axis-aligned rectangle
                        corners = np.float32([[0, 0], [wTpl, 0], [wTpl, hTpl], [0, hTpl]]).reshape(-1, 1, 2)
                        quad = cv2.perspectiveTransform(corners, H)
                        if not quad_ok(quad, imgW, imgH, wTpl, hTpl):
                            continue

                        x, y, w, h = cv2.boundingRect(np.int32(quad))
                        ar = w / float(h) if h > 0 else 1.0
                        if not (arTpl * ASPECT_TOL <= ar <= arTpl / ASPECT_TOL):
                            continue

                        detBoxes.append((x, y, w, h))

                # 4) Per-frame NMS + center-merge de-duplication
                boxes = center_merge_boxes(nms_boxes(detBoxes, IOU_NMS_THRESH), CENTER_MERGE_FRAC)

                # 5) Lightweight tracking with size-jump guard
                curTracks = tracks[name]
                for t in curTracks:
                    t["miss"] += 1

                used = set()
                for keepBox in boxes:
                    bestIoU, bestIdx = 0.0, -1
                    for i, t in enumerate(curTracks):
                        if i in used:
                            continue
                        j = iou(keepBox, t["bbox"])
                        if j > bestIoU:
                            bestIoU, bestIdx = j, i

                    areaNow = float(keepBox[2] * keepBox[3])

                    if bestIoU >= IOU_ASSOC_THRESH and bestIdx >= 0:
                        t = curTracks[bestIdx]
                        if "area_ema" in t:
                            if not (t["area_ema"] / SIZE_JUMP_MAX <= areaNow <= t["area_ema"] * SIZE_JUMP_MAX):
                                t["miss"] = 0
                                continue
                            t["area_ema"] = 0.6 * areaNow + 0.4 * t["area_ema"]
                        else:
                            t["area_ema"] = areaNow

                        t["bbox"] = ema_box(t["bbox"], keepBox, SMOOTH_ALPHA)
                        t["miss"] = 0
                        t["hits"] += 1
                        used.add(bestIdx)
                    else:
                        curTracks.append(
                            {"bbox": tuple(map(float, keepBox)), "hits": 1, "miss": 0, "area_ema": areaNow}
                        )

                # Prune stale tracks
                tracks[name] = [t for t in curTracks if t["miss"] <= MISS_TTL_FRAMES]

                # 6) Draw & count (rectangles only)
                shown = 0
                for t in tracks[name]:
                    if t["hits"] >= APPEAR_MIN_HITS:
                        x, y, w, h = map(int, t["bbox"])
                        cv2.rectangle(vis, (x, y), (x + w, y + h), DRAW_COLOR_BOX, DRAW_THICKNESS)
                        shown += 1

                cv2.putText(
                    vis,
                    f"{name}: {shown}",
                    (10, yText),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SCALE_COUNT,
                    TEXT_COLOR_COUNT,
                    TEXT_THICKNESS,
                )
                yText += 40

            # FPS overlay (EMA)
            now = time.time()
            inst = 1.0 / (now - prevTime) if now > prevTime else 0.0
            prevTime = now
            fps = inst if fps == 0 else (0.8 * fps + 0.2 * inst)

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, yText),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE_FPS,
                TEXT_COLOR_FPS,
                TEXT_THICKNESS,
            )

            cv2.imshow(WINDOW_NAME_MAIN, vis)
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break

    finally:
        stopCapture = True
        cv2.destroyAllWindows()


# =============================================================================
# BOOTSTRAP
# =============================================================================
if __name__ == "__main__":
    main()
