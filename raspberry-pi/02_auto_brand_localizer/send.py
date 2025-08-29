#!/usr/bin/env python3
# Real-time IMX500 ROI proposals + AKAZE template verification (≥15 FPS target)
# - Uses on-sensor detector (SSD by default) for fast ROIs
# - Verifies brands via template matching inside ROIs with AKAZE (binary + Hamming)
# - Stable rectangular boxes, NMS, de-duplication, light tracking
# - Auto-installs/downloads RPK model if missing

import os, sys, subprocess, shutil, time, threading, urllib.request, cv2, numpy as np
from pathlib import Path
from picamera2 import Picamera2

# ---------- Config ----------
TEMPLATE_DIR        = "templates"           # same as your current flow (file names = brand names)
MAIN_W, MAIN_H      = 1280, 720             # display / final draw
LORES_W, LORES_H    = 640, 360              # where on-sensor bboxes map easily & AKAZE runs
FPS_TARGET          = 30

# IMX500 model (change to YOLO or NanoDet if you prefer)
# All of these exist in the official Raspberry Pi IMX500 Model Zoo:
#   /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
#   /usr/share/imx500-models/imx500_network_yolo11n_pp.rpk
#   /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk
#   /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk
MODEL_FILENAME      = "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
APT_PACKAGE         = "imx500-models"
APT_MODEL_PATH      = f"/usr/share/{APT_PACKAGE}/{MODEL_FILENAME}"
RAW_FALLBACK_URL    = f"https://raw.githubusercontent.com/raspberrypi/imx500-models/main/{MODEL_FILENAME}"

# ROI / post-processing
MAX_ROIS_PER_FRAME  = 12     # cap total verification work
CONF_THRESH         = 0.20    # sensor confidences are cheap; verifier refines
NMS_IOU_ROI         = 0.45

# AKAZE (fast, robust enough at your working distances)
AKAZE_THRESH        = 0.001
AKAZE_FEATURES_CAP  = 700     # hard cap per ROI for speed
RATIO_TEST          = 0.80    # Lowe-like ratio for binary knn
MIN_MATCHES         = 12      # before homography
RANSAC_THRESH       = 4.0     # px (lores space)
MIN_INLIERS         = 8
INLIER_RATIO_MIN    = 0.30
MAX_REPROJ_ERR      = 4.0

# Box de-dup & tracking
DRAW_COLOR          = (0,255,0)
IOU_NMS_FINAL       = 0.30
CENTER_MERGE_FRAC   = 0.45
ASPECT_TOL          = 0.55    # AR guard vs template
SMOOTH_ALPHA        = 0.65
IOU_ASSOC_THRESH    = 0.45
MISS_TTL_FRAMES     = 6
APPEAR_MIN_HITS     = 1
SIZE_JUMP_MAX       = 1.8

# ---------- IMX500 setup ----------
IMX_OK = True
try:
    from picamera2.devices import IMX500
except Exception:
    IMX_OK = False

def ensure_model_path():
    """
    Ensure we have a usable RPK on the system.
    1) If /usr/share/imx500-models/<file> exists -> use it.
    2) Try apt install imx500-models.
    3) Download from GitHub raw into ./models and use that.
    """
    # Option 1: already in /usr/share
    if os.path.exists(APT_MODEL_PATH):
        return APT_MODEL_PATH

    # Option 2: apt install (best effort)
    try:
        if shutil.which("apt"):
            subprocess.run(["sudo","apt","update"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["sudo","apt","install","-y",APT_PACKAGE], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(APT_MODEL_PATH):
                return APT_MODEL_PATH
    except Exception:
        pass

    # Option 3: download raw to ./models/
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    local_path = models_dir / MODEL_FILENAME
    if not local_path.exists():
        try:
            print(f"Downloading model to {local_path} ...")
            urllib.request.urlretrieve(RAW_FALLBACK_URL, str(local_path))
        except Exception as e:
            print("Model download failed:", e)
    if local_path.exists():
        return str(local_path)

    raise FileNotFoundError("Could not obtain IMX500 model RPK.")

# ---------- small utils ----------
def clamp_int(v, lo, hi): return max(lo, min(hi, int(v)))

def iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    xa,ya=max(x1,x2),max(y1,y2); xb,yb=min(x1+w1,x2+w2),min(y1+h1,y2+h1)
    inter=max(0,xb-xa)*max(0,yb-ya); union=w1*h1+w2*h2-inter
    return inter/union if union>0 else 0.0

def nms_boxes(boxes, thr):
    kept=[]
    for b in boxes:
        if all(iou(b,k) < thr for k in kept): kept.append(b)
    return kept

def center_merge_boxes(boxes, frac):
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

def reproj_error(H, src, dst, mask):
    if H is None or mask is None: return 1e9
    src2 = cv2.perspectiveTransform(src, H)
    inl = mask.ravel().astype(bool)
    if not np.any(inl): return 1e9
    diff = src2[inl] - dst[inl]
    return float(np.mean(np.linalg.norm(diff, axis=2)))

# ---------- Templates (AKAZE) ----------
def load_templates():
    akaze = cv2.AKAZE_create(threshold=AKAZE_THRESH)
    tmpls=[]
    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if not fname.lower().endswith((".png",".jpg",".jpeg")): continue
        path = os.path.join(TEMPLATE_DIR, fname)
        img = cv2.imread(path)
        if img is None: continue
        # Scale each template roughly to lores scale for stable geometry
        h, w = img.shape[:2]
        scale = min(LORES_W/w, LORES_H/h)
        ws, hs = max(40, int(w*scale)), max(40, int(h*scale))
        img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = akaze.detectAndCompute(gray, None)
        if des is None or len(kp)==0: continue
        name = os.path.splitext(fname)[0]
        tmpls.append({
            "name": name, "w": ws, "h": hs, "ar": ws/float(hs),
            "kp": kp, "des": des
        })
    if not tmpls:
        print("No templates found in", TEMPLATE_DIR)
    return tmpls

# ---------- IMX500 ROI parsing ----------
def parse_imx500_rois(imx500, metadata):
    """
    Returns lores-space ROIs [(x,y,w,h), ...]
    Supports common 'pp' detectors:
      - SSD MobileNetV2 FPN Lite (likely Nx7: [img_id,label,score,xmin,ymin,xmax,ymax])
      - YOLO11n/YOLOv8n (pp): often Nx6: [x,y,w,h,score,class] normalized
    """
    rois=[]
    try:
        outs = imx500.get_outputs(metadata)
    except Exception:
        return rois

    if not outs: return rois
    W, H = LORES_W, LORES_H

    for out in outs:
        arr = np.array(out)
        if arr.ndim != 2 or arr.shape[1] < 6:
            continue

        # Heuristics: SSD-style 7-tuple
        if arr.shape[1] >= 7:
            for det in arr:
                # SSD: [image_id, label, score, xmin, ymin, xmax, ymax] normalized 0..1
                score = float(det[2])
                if score < CONF_THRESH: continue
                x1 = clamp_int(int(det[3]*W), 0, W-1)
                y1 = clamp_int(int(det[4]*H), 0, H-1)
                x2 = clamp_int(int(det[5]*W), 0, W-1)
                y2 = clamp_int(int(det[6]*H), 0, H-1)
                if x2<=x1 or y2<=y1: continue
                rois.append((x1,y1,x2-x1,y2-y1))
        else:
            # YOLO-like 6-tuple: [x,y,w,h,score,class] normalized
            for det in arr:
                score = float(det[4])
                if score < CONF_THRESH: continue
                cx = float(det[0])*W; cy = float(det[1])*H
                ww = float(det[2])*W; hh = float(det[3])*H
                x1 = clamp_int(int(cx - ww/2), 0, W-1)
                y1 = clamp_int(int(cy - hh/2), 0, H-1)
                x2 = clamp_int(int(cx + ww/2), 0, W-1)
                y2 = clamp_int(int(cy + hh/2), 0, H-1)
                if x2<=x1 or y2<=y1: continue
                rois.append((x1,y1,x2-x1,y2-y1))

    # Deduplicate / limit
    rois = center_merge_boxes(nms_boxes(rois, NMS_IOU_ROI), 0.5)
    if len(rois) > MAX_ROIS_PER_FRAME:
        # keep biggest first (most likely)
        rois = sorted(rois, key=lambda b: b[2]*b[3], reverse=True)[:MAX_ROIS_PER_FRAME]
    return rois

# ---------- Matcher + verify inside ROIs ----------
def verify_templates_in_roi(scene_gray, roi, templates, bf):
    rx,ry,rw,rh = roi
    patch = scene_gray[ry:ry+rh, rx:rx+rw]
    # Limit features per ROI for speed
    akaze = cv2.AKAZE_create(threshold=AKAZE_THRESH)
    kps, dess = akaze.detectAndCompute(patch, None)
    if dess is None or len(kps)==0:
        return {}

    # Cap descriptors to speed up matching (keep strongest by response)
    if len(kps) > AKAZE_FEATURES_CAP:
        idx = np.argsort([-kp.response for kp in kps])[:AKAZE_FEATURES_CAP]
        kps = [kps[i] for i in idx]
        dess = dess[idx]

    found = {}  # name -> list of boxes (lores space)
    for tpl in templates:
        des_t = tpl["des"]
        if des_t is None or len(des_t)==0: continue

        # KNN + ratio test (binary descriptors, Hamming)
        matches = bf.knnMatch(des_t, dess, k=2)
        good=[]
        for m,n in matches:
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)
        if len(good) < max(MIN_MATCHES, int(0.015*len(des_t))):
            continue

        # homography on ROI coords
        src = np.float32([tpl["kp"][m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kps[m.trainIdx].pt        for m in good]).reshape(-1,1,2)
        H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is None or mask_h is None: 
            continue

        inl = int(mask_h.ravel().sum())
        if inl < MIN_INLIERS or (inl/len(good)) < INLIER_RATIO_MIN:
            continue
        if reproj_error(H, src, dst, mask_h) > MAX_REPROJ_ERR:
            continue

        # warp template quad -> ROI space -> rectangle -> full lores space
        w_t, h_t = tpl["w"], tpl["h"]
        corners = np.float32([[0,0],[w_t,0],[w_t,h_t],[0,h_t]]).reshape(-1,1,2)
        quad = cv2.perspectiveTransform(corners, H)
        x,y,w,h = cv2.boundingRect(np.int32(quad))
        # AR guard vs template
        ar_t = tpl["ar"]; ar = w/float(h) if h>0 else 1.0
        if not (ar_t*ASPECT_TOL <= ar <= ar_t/ASPECT_TOL):
            continue
        # shift by ROI offset
        box = (rx+x, ry+y, w, h)
        found.setdefault(tpl["name"], []).append(box)

    return found

# ---------- Main ----------
def main():
    if not IMX_OK:
        print("Picamera2 IMX500 module not available. Update picamera2 / OS.")
        sys.exit(1)

    # Ensure we have a model
    model_path = ensure_model_path()

    # Load templates
    templates = load_templates()
    if not templates:
        sys.exit(1)

    # Init camera (main + lores)
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main = {"size": (MAIN_W, MAIN_H), "format": "RGB888"},
        lores= {"size": (LORES_W, LORES_H), "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET},
        buffer_count=6
    )
    picam2.configure(cfg)

    # Attach IMX500 model
    imx = IMX500(model_path)                 # loads RPK onto the sensor
    try:
        # (Optional progress) imx.show_network_fw_progress_bar()
        pass
    except Exception:
        pass

    tracks = {tpl["name"]: [] for tpl in templates}
    # Single BF matcher (Hamming) reused across calls
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    picam2.start()
    time.sleep(0.4)
    prev_t = time.time(); fps = 0.0

    cv2.namedWindow("Localization", cv2.WINDOW_NORMAL)

    try:
        while True:
            req = picam2.capture_request()
            frame_main = req.make_array("main")   # RGB
            lores      = req.make_array("lores")  # RGB
            meta       = req.get_metadata()
            req.release()

            # Prepare images (OpenCV expects BGR for drawing)
            vis_bgr  = cv2.cvtColor(frame_main, cv2.COLOR_RGB2BGR)
            lo_bgr   = cv2.cvtColor(lores,      cv2.COLOR_RGB2BGR)
            scene_gray = cv2.cvtColor(lo_bgr, cv2.COLOR_BGR2GRAY)

            # --- 1) IMX500 proposals (lores space) ---
            rois = parse_imx500_rois(imx, meta)
            if not rois:
                rois = [(0,0,LORES_W,LORES_H)]  # fallback if model outputs nothing

            # --- 2) Verify templates only inside those ROIs ---
            per_brand_boxes_lo = {tpl["name"]: [] for tpl in templates}
            for roi in rois:
                found = verify_templates_in_roi(scene_gray, roi, templates, bf)
                for name, lst in found.items():
                    per_brand_boxes_lo[name].extend(lst)

            # --- 3) Draw with de-dup + tracking ---
            sx, sy = MAIN_W/float(LORES_W), MAIN_H/float(LORES_H)
            y_text = 30

            for tpl in templates:
                name = tpl["name"]
                # dedupe in lores, then scale to main
                boxes_lo = center_merge_boxes(nms_boxes(per_brand_boxes_lo[name], IOU_NMS_FINAL),
                                              CENTER_MERGE_FRAC)
                boxes = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x,y,w,h) in boxes_lo]

                # update tracks
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

                shown=0
                for t in tracks[name]:
                    if t["hits"] >= APPEAR_MIN_HITS:
                        x,y,w,h = map(int, t["bbox"])
                        cv2.rectangle(vis_bgr,(x,y),(x+w,y+h), DRAW_COLOR, 3)
                        shown += 1
                cv2.putText(vis_bgr, f"{name}: {shown}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                y_text += 40

            # --- 4) FPS overlay ---
            now=time.time()
            inst = 1.0/(now - prev_t) if now>prev_t else 0.0
            prev_t = now
            fps = inst if fps==0 else (0.8*fps + 0.2*inst)
            cv2.putText(vis_bgr, f"FPS: {fps:.1f}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Localization", vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()
