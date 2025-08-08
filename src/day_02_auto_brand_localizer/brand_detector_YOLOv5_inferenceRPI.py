#!/usr/bin/env python3
"""
detect_logos_pi.py

1. Downloads the TorchScript model brand_detector_ts.pt if missing.
2. Captures 1280×720 @30FPS live video via Picamera2.
3. Runs the logo detector at ~10–15FPS on Pi5 CPU.
4. Draws boxes, class names, and per-class counts on the streaming window.
"""

import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from picamera2 import Picamera2
import threading
import urllib.request

MODEL_URL  = "https://<your_host>/brand_detector_ts.pt"  # Host your TS file from laptop
MODEL_FILE = "brand_detector_ts.pt"

DETECT_W, DETECT_H = 1280, 720
FPS_TARGET        = 30
CONF_THRESH       = 0.25
IOU_THRESH        = 0.45

# Shared state
latest_frame = None
stop_flag    = False
lock         = threading.Lock()

def camera_thread():
    global latest_frame, stop_flag
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (DETECT_W, DETECT_H), "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET}
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)
    while not stop_flag:
        frame = picam.capture_array()
        with lock:
            latest_frame = frame
    picam.stop()

def download_model():
    if not Path(MODEL_FILE).exists():
        print("[INFO] Downloading TorchScript model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[INFO] Model downloaded.")

def preprocess(img, target_size=640):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dy, dx = (target_size - nh) // 2, (target_size - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = img_resized
    img_t = torch.from_numpy(canvas).permute(2,0,1).float() / 255.0
    return img_t.unsqueeze(0), scale, dx, dy

def non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45):
    # pred: [N,6] => x1,y1,x2,y2,conf,class
    boxes=[]
    if pred is None or len(pred)==0: return boxes
    mask = pred[:,4] >= conf_thresh
    dets = pred[mask]
    if not dets.size(0): return boxes
    xyxy = dets[:,:4].cpu().numpy()
    scores= dets[:,4].cpu().numpy()
    classes=dets[:,5].cpu().numpy().astype(int)
    idxs = cv2.dnn.NMSBoxes(
        bboxes=xyxy.tolist(),
        scores=scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh
    )
    for i in idxs:
        i=i[0]
        x1,y1,x2,y2=xyxy[i].astype(int)
        conf=scores[i]
        cls=classes[i]
        boxes.append((x1,y1,x2,y2,conf,cls))
    return boxes

def main():
    global stop_flag, latest_frame
    download_model()
    model = torch.jit.load(MODEL_FILE).eval()

    # start camera thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    cv2.namedWindow("Logo Detection", cv2.WINDOW_NORMAL)
    prev = time.time()
    fps=0.0

    try:
        while True:
            with lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            orig = frame.copy()
            img_t, scale, dx, dy = preprocess(frame, 640)
            with torch.no_grad():
                pred = model(img_t)[0]

            # map boxes back
            dets = []
            for *xyxy, conf, cls in pred.cpu().numpy():
                x1,y1,x2,y2=xyxy
                x1 = max(int((x1-dx)/scale),0)
                y1 = max(int((y1-dy)/scale),0)
                x2 = min(int((x2-dx)/scale),orig.shape[1])
                y2 = min(int((y2-dy)/scale),orig.shape[0])
                dets.append([x1,y1,x2,y2,conf,cls])
            dets = non_max_suppression(torch.Tensor(dets), CONF_THRESH, IOU_THRESH)

            vis = orig.copy()
            counts = {}
            for x1,y1,x2,y2,conf,cls in dets:
                label = f"{cls}:{conf*100:.1f}%"
                counts[cls]=counts.get(cls,0)+1
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis,label,(x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            y=30
            for cls,count in counts.items():
                cv2.putText(vis,f"{cls}: {count}",(10,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                y+=30

            now=time.time()
            inst=1.0/(now-prev) if now>prev else 0
            prev=now
            fps=inst if fps==0 else (0.8*fps+0.2*inst)
            cv2.putText(vis,f"FPS:{fps:.1f}",(10,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

            cv2.imshow("Logo Detection",vis)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
    finally:
        stop_flag=True
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
