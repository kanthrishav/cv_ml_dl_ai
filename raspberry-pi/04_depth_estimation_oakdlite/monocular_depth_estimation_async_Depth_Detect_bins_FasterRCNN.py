#!/usr/bin/env python3
# async_depth_detect_torchvision.py

import threading, time, collections
import cv2, numpy as np, torch
from picamera2 import Picamera2
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

# --- Settings ---
DEVICE        = torch.device("cpu")

# Depth model
MIDAS_TYPE    = "MiDaS_small"
# Detection model
CONF_THRESH   = 0.5

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
tf = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = tf.small_transform if MIDAS_TYPE=="MiDaS_small" else tf.default_transform

# Load TorchVision detector
det_model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).to(DEVICE).eval()

COCO_LABELS = [
    '__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A','handbag',
    'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','N/A','book','clock',
    'vase','scissors','teddy bear','hair drier','toothbrush'
]

def depth_detect_loop(picam):
    global depth_overlay, depth_norm, detections

    interval = 1.0 / DEPTH_FPS
    while True:
        t0 = time.time()
        frame = picam.capture_array()  # HxWx3 uint8

        # Depth
        inp = transform(frame).to(DEVICE)
        with torch.no_grad():
            d = midas(inp)
            d = F.interpolate(
                d.unsqueeze(1), size=CAM_SIZE[::-1],
                mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
        dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d8 = np.uint8(dn * 255)
        cmap = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)

        # Detection
        img = cv2.resize(frame, CAM_SIZE)
        tensor = ToTensor()(img).to(DEVICE)
        with torch.no_grad():
            out = det_model([tensor])[0]
        # Filter boxes by confidence
        boxes = out['boxes'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        mask = scores >= CONF_THRESH

        preds = np.hstack((
            boxes[mask],
            scores[mask, None],
            labels[mask, None]
        ))  # shape [M,6]: x1,y1,x2,y2,score,label

        with depth_lock:
            depth_overlay = cmap
            depth_norm    = dn
        with det_lock:
            detections = preds

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

def main_display(picam):
    global depth_overlay, depth_norm, detections

    prev_time = time.time()
    fps = None

    while True:
        frame = picam.capture_array()
        h, w = CAM_SIZE[1], CAM_SIZE[0]

        with depth_lock:
            dov = depth_overlay.copy() if depth_overlay is not None else np.zeros((h,w,3),np.uint8)
            dn  = depth_norm.copy()    if depth_norm    is not None else np.zeros((h,w),np.float32)
        with det_lock:
            dets = detections.copy() if detections is not None else np.empty((0,6))

        vis = frame.copy()
        counts = collections.Counter()

        for x1,y1,x2,y2,score,label in dets:
            x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
            cx, cy = (x1+x2)//2, (y1+y2)//2
            val = dn[cy, cx]
            for name,(lo,hi) in BINS.items():
                if lo <= val < hi:
                    counts[name]+=1
                    break
            cls_name = COCO_LABELS[int(label)]
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,f"{cls_name}:{score:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

        y0 = 30
        for name in BINS:
            cv2.putText(vis,f"{name}:{counts.get(name,0)}",(10,y0),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            y0+=25

        now = time.time()
        inst = 1.0/(now-prev_time) if now!=prev_time else 0
        prev_time = now
        fps = inst if fps is None else (0.8*fps + 0.2*inst)
        cv2.putText(vis,f"FPS:{fps:.1f}",(10,y0),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        out = np.hstack((vis, dov))
        cv2.imshow("Live | Depth+Detect+Binning", out)
        if cv2.waitKey(int(1000/DISPLAY_FPS)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__=="__main__":
    # Single Picamera2 instance
    picam = Picamera2()
    cfg = picam.create_preview_configuration(
        main={"format":"RGB888","size":CAM_SIZE}
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)

    # Start detection thread
    t = threading.Thread(target=depth_detect_loop, args=(picam,), daemon=True)
    t.start()

    # Run display
    main_display(picam)
