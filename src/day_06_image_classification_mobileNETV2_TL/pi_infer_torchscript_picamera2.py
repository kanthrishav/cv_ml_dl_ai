#!/usr/bin/env python3
# pi_infer_torchscript_picamera2.py
# Live classification on Raspberry Pi 5 using PiCamera2 + TorchScript model.

import os, time
from datetime import datetime
import numpy as np
import cv2
import torch
from torchvision import transforms
from picamera2 import Picamera2

# ========================== CONFIG (EDIT HERE) ==========================
MODEL_PATH   = "./model_ts.pt"     # copy from export_torch/model_ts.pt
LABELS_PATH  = "./labels.txt"      # copy from export_torch/labels.txt
IMG_SIZE     = 224
TOPK         = 3
# ======================================================================

def load_labels(p):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def choose_next_label(candidates, current):
    if not candidates: return None
    if current not in candidates: return candidates[0]
    i = candidates.index(current)
    return candidates[(i + 1) % len(candidates)]

def save_template(frame_bgr, label):
    out_dir = os.path.join("templates", label)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"template_{label}_{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    print(f"[INFO] Template saved: {path}")

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def main():
    # Load model + labels
    model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
    labels = load_labels(LABELS_PATH)

    # Preproc
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # PiCamera2 init
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.1)

    template_label = None
    last_candidates = []
    fps_sm = None
    t_prev = time.time()

    try:
        while True:
            frame_rgb = picam2.capture_array()   # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # Resize to IMG_SIZE for model
            img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            inp = tfm(img).unsqueeze(0)  # [1,3,H,W]

            with torch.no_grad():
                out = model(inp)         # [1, C]
                probs = softmax_np(out[0].numpy())

            topk_idx = probs.argsort()[-TOPK:][::-1]
            topk_lbl = [labels[i] for i in topk_idx]
            topk_pr  = [float(probs[i]) for i in topk_idx]
            last_candidates = topk_lbl

            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_prev)); t_prev = now
            fps_sm = fps if fps_sm is None else (0.8 * fps_sm + 0.2 * fps)

            # Overlay
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (5, 5), (635, 5 + 30*(TOPK + 3)), (0,0,0), -1)
            disp = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)

            y = 30
            cv2.putText(disp, f"FPS: {fps_sm:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y += 30
            for i, (lbl, pr) in enumerate(zip(topk_lbl, topk_pr)):
                color = (0,255,0) if (template_label == lbl) else (255,255,255)
                cv2.putText(disp, f"{i+1}. {lbl}: {pr*100:.1f}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 26
            templ_txt = f"Template: {template_label if template_label else '(none)'}"
            cv2.putText(disp, templ_txt, (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            cv2.imshow("Pi TorchScript Classifier (PiCamera2)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('t'):
                template_label = topk_lbl[0] if topk_lbl else None
                if template_label: save_template(frame_bgr, template_label)
            elif k == ord('n'):
                if last_candidates:
                    template_label = choose_next_label(last_candidates, template_label)
            elif k == ord('u'):
                template_label = None
            elif k == ord('s'):
                os.makedirs("snapshots", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join("snapshots", f"snap_{ts}.jpg")
                cv2.imwrite(path, frame_bgr)
                print(f"[INFO] Snapshot saved: {path}")

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
