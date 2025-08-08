#!/usr/bin/env python3
import os
import cv2
import time
import threading
import numpy as np
from picamera2 import Picamera2

# ——— CONFIGURATION —————————————————————————————————————
TEMPLATE_DIR      = "templates"       # folder with template PNG/JPG files
DETECT_W, DETECT_H= 1280, 720         # detection resolution
FPS_TARGET        = 30                # camera FPS
MIN_CLUSTER_MATCH = 4                 # min matches to consider a valid detection
RATIO_TEST        = 0.75              # Lowe’s ratio threshold
RANSAC_THRESH     = 5.0               # homography RANSAC reprojection threshold
SIFT_FEATURES     = 500               # number of SIFT features to detect
CLUSTER_RADIUS    = 50                # pixels: radius around each match for clustering
# ——————————————————————————————————————————————————————————————

# Shared frame buffer & control
latest_frame = None
frame_lock   = threading.Lock()
stop_capture = False

def camera_thread():
    """Continuously capture frames from the Pi AI camera at 1280×720 @30 FPS."""
    global latest_frame, stop_capture
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (DETECT_W, DETECT_H), "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET}
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)  # let the sensor/ISP settle

    while not stop_capture:
        frame = picam.capture_array()
        with frame_lock:
            latest_frame = frame

    picam.stop()

def prep_gray(img):
    """Convert to grayscale and apply CLAHE for better keypoint detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def load_templates():
    """
    Load all images from TEMPLATE_DIR, downscale them to fit
    DETECT_W×DETECT_H (preserving aspect), compute SIFT keypoints/descriptors.
    Returns a list of dicts with: name, image, gray, kp, des, w, h, matcher.
    """
    templates = []
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    # FLANN matcher template-specific for performance
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(TEMPLATE_DIR, fname)
        img_hr = cv2.imread(path)
        if img_hr is None:
            continue
        h_hr, w_hr = img_hr.shape[:2]
        # downscale to detection size ratio
        scale = min(DETECT_W / w_hr, DETECT_H / h_hr)
        w_s, h_s = int(w_hr * scale), int(h_hr * scale)
        img_s = cv2.resize(img_hr, (w_s, h_s), interpolation=cv2.INTER_AREA)
        gray = prep_gray(img_s)
        kp, des = sift.detectAndCompute(gray, None)
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )
        templates.append({
            "name": os.path.splitext(fname)[0],
            "img": img_s,
            "gray": gray,
            "kp": kp,
            "des": des,
            "w": w_s,
            "h": h_s,
            "matcher": matcher
        })
        # Show keypoints once
        kp_vis = cv2.drawKeypoints(
            img_s, kp, None, color=(0,255,0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow(f"Template: {fname}", kp_vis)
    return templates

def main():
    global stop_capture, latest_frame
    # 1. Load templates
    templates = load_templates()
    if not templates:
        print("No templates found in", TEMPLATE_DIR)
        return

    # 2. Start camera thread
    cap_thread = threading.Thread(target=camera_thread, daemon=True)
    cap_thread.start()

    # 3. SIFT detector for scene
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)

    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # 4. Grab latest frame
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            scene_gray = prep_gray(frame)
            kp_s, des_s = sift.detectAndCompute(scene_gray, None)

            # Prepare a visualization copy
            vis = frame.copy()
            y_text = 30

            # 5. For each template, match → cluster → homography
            for tpl in templates:
                name, kp_t, des_t, matcher = tpl["name"], tpl["kp"], tpl["des"], tpl["matcher"]
                w_t, h_t = tpl["w"], tpl["h"]

                good = []
                if des_s is not None and len(kp_s)>0:
                    raw = matcher.knnMatch(des_t, des_s, k=2)
                    for m,n in raw:
                        if m.distance < RATIO_TEST * n.distance:
                            good.append(m)

                count = 0
                if len(good) >= MIN_CLUSTER_MATCH:
                    # 6. Build mask of matches for clustering
                    mask = np.zeros((DETECT_H, DETECT_W), dtype=np.uint8)
                    for m in good:
                        x,y = kp_s[m.trainIdx].pt
                        cv2.circle(mask, (int(x),int(y)), CLUSTER_RADIUS, 255, -1)
                    # connected components
                    n_lbl, labels = cv2.connectedComponents(mask)

                    for lbl in range(1, n_lbl):
                        # collect matches in this cluster
                        cluster = []
                        for m in good:
                            x,y = kp_s[m.trainIdx].pt
                            if labels[int(y),int(x)] == lbl:
                                cluster.append(m)
                        if len(cluster) < MIN_CLUSTER_MATCH:
                            continue
                        # homography
                        src = np.float32([kp_t[m.queryIdx].pt for m in cluster]).reshape(-1,1,2)
                        dst = np.float32([kp_s[m.trainIdx].pt for m in cluster]).reshape(-1,1,2)
                        H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                        if H is not None and mask_h.sum() >= MIN_CLUSTER_MATCH:
                            # draw warped rectangle
                            corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
                            scene_c = cv2.perspectiveTransform(corners, H)
                            vis = cv2.polylines(vis, [np.int32(scene_c)], True, (0,255,0), 3)
                            count += 1

                # overlay count for this template
                cv2.putText(vis, f"{name}: {count}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                y_text += 40

            # 7. FPS overlay
            now = time.time()
            inst = 1.0/(now - prev_time) if now>prev_time else 0.0
            prev_time = now
            fps = inst if fps==0 else (0.8*fps + 0.2*inst)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Localization", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_capture = True
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
