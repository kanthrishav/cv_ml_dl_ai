#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ——— CONFIG ——————————————————————————————————————————
TEMPLATE_PATH   = sys.argv[1]
CAM_W, CAM_H    = 640, 480       # downscale for speed
MIN_MATCHES     = 8              # at least this many good matches
RATIO_TEST      = 0.75           # Lowe's ratio
RANSAC_THRESH   = 5.0            # reprojection threshold for homography
# ————————————————————————————————————————————————————————

def prep_image(img, do_clahe=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    return gray

def init_detector():
    # SIFT is in contrib; available in recent OpenCV
    return cv2.SIFT_create()

def init_matcher():
    # FLANN parameters for SIFT
    index_params = dict(algorithm=1, trees=5)   # KDTree
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def draw_keypoints_window(name, img, kps):
    vis = cv2.drawKeypoints(img, kps, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(0,255,0))
    cv2.imshow(name, vis)

def draw_matches_window(name, tmpl, kpt1, scene, kpt2, matches):
    vis = cv2.drawMatches(tmpl, kpt1, scene, kpt2,
                          matches, None,
                          matchColor=(0,255,0),
                          singlePointColor=(255,0,0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(name, vis)

def main():
    # 1. Load & prep template
    tpl_color = cv2.imread(TEMPLATE_PATH)
    if tpl_color is None:
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    tpl_color = cv2.resize(tpl_color, (CAM_W//2, CAM_H//2), interpolation=cv2.INTER_AREA)
    tpl_gray  = prep_image(tpl_color)

    # 2. Init SIFT + matcher, compute template features
    sift    = init_detector()
    matcher = init_matcher()
    kp_tpl, des_tpl = sift.detectAndCompute(tpl_gray, None)

    # Show template keypoints
    draw_keypoints_window("Template KPs", tpl_color, kp_tpl)

    # 3. Start Pi camera at 640×480
    picam = Picamera2()
    conf = picam.create_preview_configuration({"size":(CAM_W,CAM_H),"format":"RGB888"})
    picam.configure(conf)
    picam.start()
    time.sleep(2)

    cv2.namedWindow("Scene KPs", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Raw Matches", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Inliers Matches", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)

    prev = time.time()
    fps = 0.0

    try:
        while True:
            frame = picam.capture_array()
            frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
            scene_gray = prep_image(frame)

            # 4. Scene keypoints
            kp_scn, des_scn = sift.detectAndCompute(scene_gray, None)
            draw_keypoints_window("Scene KPs", frame, kp_scn)

            good = []
            if des_scn is not None and len(kp_scn)>0:
                # 5. KNN match + ratio
                raw = matcher.knnMatch(des_tpl, des_scn, k=2)
                for m,n in raw:
                    if m.distance < RATIO_TEST * n.distance:
                        good.append(m)

            # Show raw matches (up to 20)
            draw_matches_window("Raw Matches", tpl_color, kp_tpl,
                                frame, kp_scn, good[:20])

            H, mask = None, None
            inliers = []
            if len(good) >= MIN_MATCHES:
                # 6. Compute homography
                src = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp_scn[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                if mask is not None:
                    # extract inlier matches
                    for i,m in enumerate(good):
                        if mask[i]:
                            inliers.append(m)

            # Show inliers matches
            draw_matches_window("Inliers Matches", tpl_color, kp_tpl,
                                frame, kp_scn, inliers[:20])

            vis = frame.copy()
            # 7. Draw localization via warped corners if homography valid
            if H is not None and mask.sum() >= MIN_MATCHES:
                h_t, w_t = tpl_gray.shape
                corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
                scene_c = cv2.perspectiveTransform(corners, H)
                vis = cv2.polylines(vis, [np.int32(scene_c)], True, (0,255,0), 3)
                x0,y0 = np.int32(scene_c[0][0])
                cv2.putText(vis, "Detected", (x0,y0-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
            else:
                cv2.putText(vis, "No reliable detection", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

            # 8. FPS overlay
            now = time.time()
            inst = 1.0/(now-prev) if now>prev else 0.0
            prev = now
            fps = inst if fps==0 else (0.8*fps + 0.2*inst)
            cv2.putText(vis, f"FPS: {fps:.1f}", (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

            cv2.imshow("Localization", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
