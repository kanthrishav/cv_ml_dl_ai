#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2

# ——— CONFIG ———————————————————————————————————————
TEMPLATE_PATH   = "template_anchor.png"
DETECT_W, DETECT_H = 1280, 720      # detection resolution
TARGET_FPS      = 30
MIN_MATCHES     = 8
RATIO_TEST      = 0.75
RANSAC_THRESH   = 5.0
SIFT_FEATURES   = 500               # fewer for speed
# ——————————————————————————————————————————————————————

# Shared frame buffer
latest_frame = None
frame_lock   = threading.Lock()
stop_capture = False

def camera_thread():
    global latest_frame, stop_capture
    picam = Picamera2()
    video_config = picam.create_video_configuration(
        main={"size": (DETECT_W, DETECT_H), "format": "RGB888"},
        controls={"FrameRate": TARGET_FPS}
    )
    picam.configure(video_config)
    picam.start()
    time.sleep(1)  # let ISP settle

    while not stop_capture:
        frame = picam.capture_array()
        with frame_lock:
            latest_frame = frame

    picam.stop()

def prep_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def main():
    global stop_capture
    # 1. Load high-res template and downscale to match detection scale (preserve aspect)
    tpl_hr = cv2.imread(TEMPLATE_PATH)
    if tpl_hr is None:
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    th, tw = tpl_hr.shape[:2]
    scale = min(DETECT_W / tw, DETECT_H / th)
    tpl_small = cv2.resize(tpl_hr, (int(tw*scale), int(th*scale)),
                            interpolation=cv2.INTER_AREA)
    tpl_gray  = prep_gray(tpl_small)

    # 2. Init SIFT + FLANN
    sift    = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    kp_tpl, des_tpl = sift.detectAndCompute(tpl_gray, None)

    # Show template keypoints once
    tpl_kp_vis = cv2.drawKeypoints(
        tpl_small, kp_tpl, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0,255,0)
    )
    cv2.imshow("Template Keypoints", tpl_kp_vis)

    # 3. Start camera thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps_smooth = 0.0
    frame_count = 0

    try:
        while True:
            # 4. Grab the latest frame
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            # Optionally skip every other frame for speed:
            # if frame_count % 2 == 1:
            #     cv2.imshow("Localization", frame)
            #     if cv2.waitKey(1)&0xFF==ord('q'):
            #         break
            #     continue

            scene_gray = prep_gray(frame)

            # 5. SIFT detection + description
            kp_scn, des_scn = sift.detectAndCompute(scene_gray, None)
            good = []
            if des_scn is not None and len(des_scn)>0:
                raw = matcher.knnMatch(des_tpl, des_scn, k=2)
                for m,n in raw:
                    if m.distance < RATIO_TEST * n.distance:
                        good.append(m)

            vis = frame.copy()
            # 6. Homography if enough matches
            if len(good) >= MIN_MATCHES:
                src_pts = np.float32([ kp_tpl[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_scn[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
                if H is not None and mask.sum() >= MIN_MATCHES:
                    # warp rectangle
                    h2, w2 = tpl_gray.shape
                    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
                    scene_c = cv2.perspectiveTransform(corners, H)
                    vis = cv2.polylines(vis, [np.int32(scene_c)], True, (0,255,0), 3)
                    x0,y0 = np.int32(scene_c[0][0])
                    cv2.putText(vis, "Detected", (x0,y0-10),
                                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
                else:
                    cv2.putText(vis, "Homography fail", (20,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            else:
                cv2.putText(vis, f"Matches: {len(good)} < {MIN_MATCHES}", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

            # 7. FPS
            now = time.time()
            inst = 1.0/(now - prev_time) if now>prev_time else 0
            prev_time = now
            fps_smooth = inst if fps_smooth==0 else (0.8*fps_smooth + 0.2*inst)
            cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

            cv2.imshow("Localization", vis)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

    finally:
        stop_capture = True
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
