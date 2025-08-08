#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
from picamera2 import Picamera2
from PIL import Image

# ==== CONFIG ====
CAPTURE_COUNT   = 7            # number of images to capture
IMG_WIDTH       = 640
IMG_HEIGHT      = 480
IMG_PATH        = "./captures"  # where to save captured frames
TEMPLATE_FILE   = "template.jpg"
PANORAMA_FILE   = "panorama.jpg"
# =================

def capture_images():
    """Capture CAPTURE_COUNT images with PiCamera2 on keypress."""
    os.makedirs(IMG_PATH, exist_ok=True)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (IMG_WIDTH, IMG_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # sensor warm-up

    print(f"[INFO] Press 'c' to capture each of {CAPTURE_COUNT} frames.")
    captured = []
    idx = 0
    while idx < CAPTURE_COUNT:
        frame = picam2.capture_array()
        disp = frame.copy()
        cv2.putText(disp, f"Press 'c' to capture ({idx}/{CAPTURE_COUNT})",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capture Mode", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            filename = os.path.join(IMG_PATH, f"img{idx}.jpg")
            cv2.imwrite(filename, frame)
            captured.append(filename)
            print(f"[INFO] Captured {filename}")
            idx += 1
            time.sleep(0.5)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    return captured

def choose_template(image_files):
    """Display thumbnails and let user pick one as template."""
    thumbs = []
    for fname in image_files:
        img = cv2.imread(fname)
        thumb = cv2.resize(img, (IMG_WIDTH//4, IMG_HEIGHT//4))
        thumbs.append(thumb)

    cols = min(len(thumbs), 4)
    rows = (len(thumbs) + cols - 1)//cols
    canvas = np.zeros((rows*(IMG_HEIGHT//4), cols*(IMG_WIDTH//4), 3), dtype=np.uint8)
    for i, th in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y, x = r*(IMG_HEIGHT//4), c*(IMG_WIDTH//4)
        canvas[y:y+th.shape[0], x:x+th.shape[1]] = th
        cv2.putText(canvas, str(i), (x+5,y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    print("[INFO] Press the number key of the template image to select it.")
    while True:
        cv2.imshow("Select Template", canvas)
        key = cv2.waitKey(0) & 0xFF
        idx = key - ord('0')
        if 0 <= idx < len(image_files):
            cv2.destroyAllWindows()
            sel = image_files[idx]
            img = cv2.imread(sel)
            cv2.imwrite(TEMPLATE_FILE, img)
            print(f"[INFO] Selected template: {sel}")
            return sel

def stitch_pair(img1, img2):
    """Stitch img2 onto img1 (the template/panorama) and return combined panorama."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    if len(good) < 10:
        raise RuntimeError("Not enough good matches to compute homography")

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp img2 into img1’s plane
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans = [-xmin, -ymin]
    H_trans = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])

    panorama = cv2.warpPerspective(img2, H_trans.dot(H), (xmax-xmin, ymax-ymin))
    panorama[trans[1]:h1+trans[1], trans[0]:w1+trans[0]] = img1

    return panorama

def create_panorama(template_file, image_files):
    """Iteratively stitch all images onto the template."""
    pano = cv2.imread(template_file)
    others = [f for f in image_files if f != template_file]
    for fname in others:
        img = cv2.imread(fname)
        print(f"[INFO] Stitching {fname} onto panorama...")
        pano = stitch_pair(pano, img)
    cv2.imwrite(PANORAMA_FILE, pano)
    print(f"[INFO] Panorama saved as {PANORAMA_FILE}")
    cv2.imshow("Final Panorama", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 1. Capture
    files = capture_images()

    # 2. Choose template
    template = choose_template(files)

    # 3. Stitch panorama
    create_panorama(template, files)

if __name__ == "__main__":
    main()
