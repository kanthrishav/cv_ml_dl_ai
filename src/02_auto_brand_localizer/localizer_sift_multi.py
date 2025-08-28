#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2

# ——— CONFIG ———————————————————————————————————————
TEMPLATE_PATH     = "template_alloutUltra.png"
DETECT_W, DETECT_H = 1280, 720     # detection resolution
FPS_TARGET        = 30
MIN_CLUSTER_MATCH = 4             # at least this many matches per detected instance
RATIO_TEST        = 0.75
RANSAC_THRESH     = 5.0
SIFT_FEATURES     = 500
# clustering mask dilation radius
CLUSTER_RADIUS    = 50
# ——————————————————————————————————————————————————————

# Shared frame storage
latest_frame = None
stop_thread  = False
frame_lock   = threading.Lock()

def camera_capture():
    global latest_frame, stop_thread
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size":(DETECT_W,DETECT_H),"format":"RGB888"},
        controls={"FrameRate":FPS_TARGET}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)
    while not stop_thread:
        frame = picam.capture_array()
        with frame_lock:
            latest_frame = frame
    picam.stop()

def prep_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe= cv2.createCLAHE(2.0,(8,8))
    return clahe.apply(gray)

def main():
    global stop_thread
    # 1. Load & downscale template (preserve aspect ratio)
    tpl_hr = cv2.imread(TEMPLATE_PATH)
    if tpl_hr is None:
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    h_hr, w_hr = tpl_hr.shape[:2]
    scale = min(DETECT_W/w_hr, DETECT_H/h_hr)
    tpl = cv2.resize(tpl_hr, (int(w_hr*scale), int(h_hr*scale)), interpolation=cv2.INTER_AREA)
    tpl_gray = prep_gray(tpl)
    h_t, w_t = tpl_gray.shape

    # 2. Init SIFT & FLANN
    sift    = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    kp_t, des_t = sift.detectAndCompute(tpl_gray, None)

    # 3. Show template keypoints once
    tpl_kpv = cv2.drawKeypoints(tpl, kp_t, None, color=(0,255,0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Template Keypoints", tpl_kpv)

    # 4. Start camera thread
    cap_thread = threading.Thread(target=camera_capture, daemon=True)
    cap_thread.start()

    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)
    prev = time.time()
    fps  = 0.0

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            scene_gray = prep_gray(frame)
            kp_s, des_s = sift.detectAndCompute(scene_gray, None)

            # 5. Match descriptors
            good = []
            if des_s is not None and len(kp_s)>0:
                raw = matcher.knnMatch(des_t, des_s, k=2)
                for m,n in raw:
                    if m.distance < RATIO_TEST * n.distance:
                        good.append(m)

            vis = frame.copy()

            if len(good) >= MIN_CLUSTER_MATCH:
                # 6. Cluster matches by proximity
                mask = np.zeros((DETECT_H, DETECT_W), dtype=np.uint8)
                for m in good:
                    x,y = kp_s[m.trainIdx].pt
                    cv2.circle(mask, (int(x),int(y)), CLUSTER_RADIUS, 255, -1)
                # connected components
                num_labels, labels = cv2.connectedComponents(mask)
                count = 0

                for lbl in range(1, num_labels):
                    pts = [good[i] for i,m in enumerate(good) if labels[
                        int(kp_s[m.trainIdx].pt[1]), int(kp_s[m.trainIdx].pt[0])
                    ] == lbl]
                    if len(pts) < MIN_CLUSTER_MATCH:
                        continue
                    # homography per cluster
                    src = np.float32([kp_t[m.queryIdx].pt for m in pts]).reshape(-1,1,2)
                    dst = np.float32([kp_s[m.trainIdx].pt for m in pts]).reshape(-1,1,2)
                    H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                    if H is not None and mask_h.sum() >= MIN_CLUSTER_MATCH:
                        # draw quadrilateral
                        corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
                        scene_c = cv2.perspectiveTransform(corners, H)
                        vis = cv2.polylines(vis, [np.int32(scene_c)],
                                           True, (0,255,0), 3)
                        count += 1

                cv2.putText(vis, f"Count: {count}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            else:
                cv2.putText(vis, "No sufficient matches", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # 7. FPS overlay
            now = time.time()
            inst=1.0/(now-prev) if now>prev else 0.0
            prev = now
            fps  = inst if fps==0 else (0.8*fps+0.2*inst)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow("Localization", vis)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

    finally:
        stop_thread = True
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
