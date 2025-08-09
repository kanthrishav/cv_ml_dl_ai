#!/usr/bin/env python3
# async_depth_detect_shared_camera.py

import threading, time, collections
import cv2, numpy as np, torch
from picamera2 import Picamera2
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# --- Settings ---
DEVICE        = torch.device("cpu")
MIDAS_TYPE    = "MiDaS_small"
TS_MODEL_PATH = "yolov5n.torchscript.pt"
CONF_THRESH   = 0.25

CAM_SIZE      = (640, 480)
DISPLAY_FPS   = 30
DEPTH_FPS     = 5
MAX_DEPTH_VAL = 10.0

BINS = {
    "0-1m":  (0.0, 1/MAX_DEPTH_VAL),
    "1-2m":  (1/MAX_DEPTH_VAL, 2/MAX_DEPTH_VAL),
    "2-5m":  (2/MAX_DEPTH_VAL, 5/MAX_DEPTH_VAL),
    "5m+":   (5/MAX_DEPTH_VAL, 1.0),
}
# -------------------

# Shared state
depth_overlay = None
depth_norm    = None
detections    = None

depth_lock = threading.Lock()
det_lock   = threading.Lock()

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_TYPE).to(DEVICE).eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform if MIDAS_TYPE=="MiDaS_small" else transforms.default_transform

# Load TorchScript YOLOv5n
det_model = torch.jit.load(TS_MODEL_PATH).to(DEVICE).eval()

def depth_detect_loop(picam):
    """Thread: capture frame, run depth+YOLO detection at DEPTH_FPS."""
    global depth_overlay, depth_norm, detections

    interval = 1.0 / DEPTH_FPS
    while True:
        t0 = time.time()
        frame = picam.capture_array()

        # Depth
        inp = transform(frame).to(DEVICE)
        with torch.no_grad():
            d = midas(inp)
            d = F.interpolate(
                d.unsqueeze(1),
                size=CAM_SIZE[::-1],
                mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
        dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d8 = np.uint8(dn*255)
        cmap = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)

        # Detection
        img = cv2.resize(frame, CAM_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = ToTensor()(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = det_model(tensor)[0].cpu().numpy()
        preds = preds[preds[:,4] >= CONF_THRESH]

        with depth_lock:
            depth_overlay = cmap
            depth_norm    = dn
        with det_lock:
            detections = preds

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

def main_display(picam):
    """Main loop: display at 30 FPS, overlay latest results."""
    global depth_overlay, depth_norm, detections

    prev_time = time.time()
    fps = None

    while True:
        frame = picam.capture_array()
        h, w = CAM_SIZE[1], CAM_SIZE[0]

        # Fetch latest
        with depth_lock:
            dov = depth_overlay.copy() if depth_overlay is not None else np.zeros((h,w,3),np.uint8)
            dn  = depth_norm.copy()    if depth_norm    is not None else np.zeros((h,w),np.float32)
        with det_lock:
            dets = detections.copy() if detections is not None else np.empty((0,6))

        vis = frame.copy()
        counts = collections.Counter()

        # Overlay detections & bins
        for x1,y1,x2,y2,conf,cls in dets:
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cx, cy = (x1+x2)//2, (y1+y2)//2
            depth_val = dn[cy, cx]
            for name,(lo,hi) in BINS.items():
                if lo <= depth_val < hi:
                    counts[name] += 1
                    break
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{int(cls)} {conf:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        y0 = 30
        for name in BINS:
            cv2.putText(vis, f"{name}: {counts.get(name,0)}",
                        (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            y0 += 25

        now = time.time()
        inst = 1.0/(now-prev_time) if now!=prev_time else 0
        prev_time = now
        fps = inst if fps is None else (0.8*fps + 0.2*inst)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        out = np.hstack((vis, dov))
        cv2.imshow("Live | Depth+Detect+Binning", out)
        if cv2.waitKey(int(1000/DISPLAY_FPS)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    # 1) Initialize Picamera2 once
    picam = Picamera2()
    cfg = picam.create_preview_configuration(main={"format":"RGB888","size":CAM_SIZE})
    picam.configure(cfg)
    picam.start()
    time.sleep(1)

    # 2) Start depth+detect thread using the same picam
    t = threading.Thread(target=depth_detect_loop, args=(picam,), daemon=True)
    t.start()

    # 3) Run display loop (reuses picam)
    main_display(picam)
