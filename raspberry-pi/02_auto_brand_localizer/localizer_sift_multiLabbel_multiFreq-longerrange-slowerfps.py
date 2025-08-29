#!/usr/bin/env python3
import os, cv2, time, threading, numpy as np
from picamera2 import Picamera2

# ——— CONFIG (generic, not scene-specific) ——————————————————————————
TEMPLATE_DIR       = "templates"
DETECT_W, DETECT_H = 1280, 720
FPS_TARGET         = 30

# SIFT/FLANN
SIFT_FEATURES      = 900                 # raise a bit for distant logos
RATIO_TEST         = 0.80                # Lowe ratio (typical 0.7–0.8)
FLANN_TREES        = 5
FLANN_CHECKS       = 150                 # more exhaustive search → better recall

# Match clustering
MIN_CLUSTER_MATCH  = 6                   # base floor; we adapt per template below

# Homography sanity
RANSAC_THRESH      = 5.0                 # px reprojection threshold (RANSAC)
MIN_INLIERS_ABS    = 6                   # base floor; adapted per template
INLIER_RATIO_MIN   = 0.30
MAX_REPROJ_ERR     = 4.0                 # avg inlier reprojection error (px)

# Boxes & de-dup
IOU_NMS_THRESH     = 0.30
CENTER_MERGE_FRAC  = 0.55                # collapse near-duplicate rectangles
ASPECT_TOL         = 0.55                # accept if w/h within [ar*tol, ar/tol]

# Lightweight tracking (rectangles only) for stability
SMOOTH_ALPHA       = 0.6
IOU_ASSOC_THRESH   = 0.45
MISS_TTL_FRAMES    = 6
APPEAR_MIN_HITS    = 1
SIZE_JUMP_MAX      = 1.8                 # reject >1.8× area jumps
# ————————————————————————————————————————————————————————————————

latest_frame, stop_capture = None, False
frame_lock = threading.Lock()

def camera_thread():
    """RGB frames for display; we convert to BGR only for processing."""
    global latest_frame, stop_capture
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (DETECT_W, DETECT_H), "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET}
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)
    while not stop_capture:
        frame = picam.capture_array()  # RGB
        with frame_lock:
            latest_frame = frame
    picam.stop()

def make_sift():
    """SIFT tuned to keep more small/low-contrast features (generic)."""
    return cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        nOctaveLayers=5,          # more scales → better at distance
        contrastThreshold=0.02,   # retain more low-contrast points
        edgeThreshold=12,         # filter out fewer edge-like points (retain more)
        sigma=1.2
    )

def prep_gray_bgr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def load_templates():
    templates = []
    sift = make_sift()
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        path = os.path.join(TEMPLATE_DIR, fname)
        img_hr = cv2.imread(path)  # BGR
        if img_hr is None:
            continue
        h_hr, w_hr = img_hr.shape[:2]
        scale = min(DETECT_W/w_hr, DETECT_H/h_hr)
        w_s, h_s = int(w_hr*scale), int(h_hr*scale)
        img_s = cv2.resize(img_hr, (w_s, h_s), interpolation=cv2.INTER_AREA)
        gray  = prep_gray_bgr(img_s)
        kp, des = sift.detectAndCompute(gray, None)
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=FLANN_TREES),
            dict(checks=FLANN_CHECKS)
        )
        templates.append({
            "name": os.path.splitext(fname)[0],
            "img": img_s, "kp": kp, "des": des, "w": w_s, "h": h_s,
            "matcher": matcher, "ar": w_s/float(h_s)
        })
        # optional once-only preview like your original
        kp_vis = cv2.drawKeypoints(img_s, kp, None, color=(0,255,0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(f"Template: {fname}", kp_vis); cv2.waitKey(1)
    return templates

# ——— helpers ————————————————————————————————————————————————
def clamp_int(v, lo, hi): return max(lo, min(hi, int(v)))

def reproj_error(H, src, dst, mask):
    if H is None or mask is None: return 1e9
    src2 = cv2.perspectiveTransform(src, H)
    inl = mask.ravel().astype(bool)
    if not np.any(inl): return 1e9
    diff = src2[inl] - dst[inl]
    return float(np.mean(np.linalg.norm(diff, axis=2)))

def quad_ok(quad, W, H, tw, th):
    pts = quad.reshape(4,2).astype(np.float32)
    if (pts[:,0] < -0.08*W).any() or (pts[:,0] > 1.08*W).any(): return False
    if (pts[:,1] < -0.08*H).any() or (pts[:,1] > 1.08*H).any(): return False
    area = abs(cv2.contourArea(pts.astype(np.int32)))
    if area < 0.002*(tw*th) or area > 0.80*(W*H): return False
    return True

def iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    xa,ya=max(x1,x2),max(y1,y2); xb,yb=min(x1+w1,x2+w2),min(y1+h1,y2+h2)
    inter=max(0,xb-xa)*max(0,yb-ya); union=w1*h1+w2*h2-inter
    return inter/union if union>0 else 0.0

def nms_boxes(boxes, thr=IOU_NMS_THRESH):
    kept=[]
    for b in boxes:
        if all(iou(b,k) < thr for k in kept): kept.append(b)
    return kept

def center_merge_boxes(boxes, frac=CENTER_MERGE_FRAC):
    if not boxes: return boxes
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][2]*boxes[i][3], reverse=True)
    keep=[]
    for i in order:
        xi,yi,wi,hi=boxes[i]; cxi,cyi=xi+wi/2.0, yi+hi/2.0; ri=frac*min(wi,hi)
        if all(abs(cxi-(xj+wj/2.0))>max(ri,frac*min(wj,hj)) or
               abs(cyi-(yj+hj/2.0))>max(ri,frac*min(wj,hj)) for xj,yj,wj,hj in keep):
            keep.append(boxes[i])
    return keep

def ema_box(prev, new, a=SMOOTH_ALPHA):
    return tuple(a*np.array(new)+(1-a)*np.array(prev))

def mutual_ratio_matches(des_t, des_s, matcher):
    """FLANN + Lowe + reciprocal best (symmetric) to stabilize matches."""
    if des_t is None or des_s is None or len(des_t)==0 or len(des_s)==0:
        return []
    raw_ts = matcher.knnMatch(des_t, des_s, k=2)
    raw_st = matcher.knnMatch(des_s, des_t, k=1)
    best_t_for_scene = {}
    for lst in raw_st:
        if len(lst)>0:
            m = lst[0]
            best_t_for_scene[m.queryIdx] = m.trainIdx
    good=[]
    for pair in raw_ts:
        if len(pair)<2: continue
        m,n = pair
        if m.distance < RATIO_TEST * n.distance:
            if best_t_for_scene.get(m.trainIdx, -1) == m.queryIdx:
                good.append(m)
    return good

# ——— main —————————————————————————————————————————————————————
def main():
    global stop_capture, latest_frame
    templates = load_templates()
    if not templates:
        print("No templates found in", TEMPLATE_DIR); return

    # tracks per brand for smoothing + size clamp
    tracks = {tpl["name"]: [] for tpl in templates}  # each: {bbox(float), hits, miss, area_ema}

    threading.Thread(target=camera_thread, daemon=True).start()
    sift_scene = make_sift()

    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)
    prev_time=time.time(); fps=0.0

    # scale-aware clustering radius & closing kernel (computed once for this resolution)
    min_side = min(DETECT_W, DETECT_H)
    radius   = max(18, int(0.028 * min_side))              # ~36 px @ 1280×720
    k_side   = max(3, (radius // 2) * 2 + 1)               # odd
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_side, k_side))

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01); continue

            # process in BGR; display RGB
            frame_bgr  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            scene_gray = prep_gray_bgr(frame_bgr)
            kp_s, des_s = sift_scene.detectAndCompute(scene_gray, None)

            vis = frame.copy()
            y_text=30
            H_img, W_img = scene_gray.shape

            for tpl in templates:
                name, kp_t, des_t, matcher = tpl["name"], tpl["kp"], tpl["des"], tpl["matcher"]
                w_t, h_t, ar_t = tpl["w"], tpl["h"], tpl["ar"]

                # adaptive minima per template (generic)
                min_cluster = max(MIN_CLUSTER_MATCH, int(0.010 * max(1, len(kp_t))))
                min_inliers_abs = max(MIN_INLIERS_ABS, int(0.40 * min_cluster))

                # robust matches
                good = mutual_ratio_matches(des_t, des_s, matcher)
                det_boxes = []

                if len(good) >= min_cluster:
                    # cluster match locations
                    mask = np.zeros((DETECT_H, DETECT_W), dtype=np.uint8)
                    for m in good:
                        x,y = kp_s[m.trainIdx].pt
                        cx = clamp_int(round(x),0,DETECT_W-1)
                        cy = clamp_int(round(y),0,DETECT_H-1)
                        cv2.circle(mask,(cx,cy),radius,255,-1)

                    if k_side >= 3:
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                    n_lbl, labels = cv2.connectedComponents(mask)

                    # homography → rectangle with quality gates
                    for lbl in range(1, n_lbl):
                        cluster=[]
                        for m in good:
                            x,y = kp_s[m.trainIdx].pt
                            xi = clamp_int(round(x),0,DETECT_W-1)
                            yi = clamp_int(round(y),0,DETECT_H-1)
                            if labels[yi,xi]==lbl:
                                cluster.append(m)
                        if len(cluster) < min_cluster:
                            continue

                        src = np.float32([kp_t[m.queryIdx].pt for m in cluster]).reshape(-1,1,2)
                        dst = np.float32([kp_s[m.trainIdx].pt for m in cluster]).reshape(-1,1,2)
                        H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                        if H is None or mask_h is None:
                            continue

                        inliers = int(mask_h.ravel().sum())
                        if inliers < min_inliers_abs or (inliers/len(cluster)) < INLIER_RATIO_MIN:
                            continue
                        if reproj_error(H, src, dst, mask_h) > MAX_REPROJ_ERR:
                            continue

                        corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
                        quad = cv2.perspectiveTransform(corners, H)
                        if not quad_ok(quad, W_img, H_img, w_t, h_t):
                            continue

                        x,y,w,h = cv2.boundingRect(np.int32(quad))
                        ar = w/float(h) if h>0 else 1.0
                        if not (ar_t*ASPECT_TOL <= ar <= ar_t/ASPECT_TOL):
                            continue
                        det_boxes.append((x,y,w,h))

                # per-frame NMS + center-merge de-dup
                boxes = center_merge_boxes(nms_boxes(det_boxes, IOU_NMS_THRESH), CENTER_MERGE_FRAC)

                # small tracks with size-jump guard
                cur = tracks[name]
                for t in cur: t["miss"] += 1
                used=set()
                for kb in boxes:
                    best, idx = 0.0, -1
                    for i,t in enumerate(cur):
                        if i in used: continue
                        j = iou(kb, t["bbox"])
                        if j>best: best, idx = j, i
                    area = float(kb[2]*kb[3])
                    if best >= IOU_ASSOC_THRESH and idx>=0:
                        t = cur[idx]
                        if "area_ema" in t:
                            if not (t["area_ema"]/SIZE_JUMP_MAX <= area <= t["area_ema"]*SIZE_JUMP_MAX):
                                t["miss"]=0; continue
                            t["area_ema"] = 0.6*area + 0.4*t["area_ema"]
                        else:
                            t["area_ema"] = area
                        t["bbox"] = ema_box(t["bbox"], kb, SMOOTH_ALPHA)
                        t["miss"]=0; t["hits"]+=1; used.add(idx)
                    else:
                        cur.append({"bbox": tuple(map(float,kb)), "hits":1, "miss":0, "area_ema": area})

                tracks[name] = [t for t in cur if t["miss"] <= MISS_TTL_FRAMES]

                # draw & count
                shown=0
                for t in tracks[name]:
                    if t["hits"] >= APPEAR_MIN_HITS:
                        x,y,w,h = map(int, t["bbox"])
                        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),3)
                        shown += 1
                cv2.putText(vis, f"{name}: {shown}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                y_text += 40

            # FPS overlay
            now=time.time(); inst=1.0/(now-prev_time) if now>prev_time else 0.0
            prev_time=now; fps=inst if fps==0 else (0.8*fps+0.2*inst)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Localization", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        stop_capture=True
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
