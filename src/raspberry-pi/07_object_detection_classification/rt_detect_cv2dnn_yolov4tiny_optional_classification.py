# rt_detect_cv2dnn_yolov4tiny.py
# PiCamera2 → OpenCV DNN YOLOv4-tiny (COCO) → boxes + labels + probs + FPS
# Optional: household MobileNetV2 (TorchScript FP32) ROI refinement.
# No ultralytics, no SciPy; works with numpy==1.24.2.

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from picamera2 import Picamera2

# YOLOv4-tiny files (place in same directory)
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"

CONF_THRESH = 0.25
NMS_THRESH = 0.45
INP_SIZE = 416  # try 320 or 416 for speed/accuracy trade-off

# =========================
# Household classifier constants (your TorchScript MobileNetV2)
HOUSEHOLD_ENABLE = True  # set False to disable refinement
HOUSEHOLD_DIR = Path(__file__).parent / "export_torch"
HOUSEHOLD_MODEL_FILE = HOUSEHOLD_DIR / "model_ts.pt"   # TorchScript FP32 file
HOUSEHOLD_LABELS_FILE = HOUSEHOLD_DIR / "labels.txt"   # one label per line
HOUSEHOLD_INPUT_SIZE = 224
# =========================


def load_classes(names_path):
    with open(names_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def draw_box(img, box, label, conf, hh_text="", color=(0, 200, 0)):
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    base = f"{label} {conf*100:.1f}%"
    txt = base + (f" | HH:{hh_text}" if hh_text else "")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 2, y_top + th + 4), color, -1)
    cv2.putText(img, txt, (x1 + 1, y_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


class HouseholdClassifier:
    def __init__(self):
        self.enabled = False
        self.labels = []
        self.model = None

        if not HOUSEHOLD_ENABLE:
            return

        try:
            if HOUSEHOLD_MODEL_FILE.exists() and HOUSEHOLD_LABELS_FILE.exists():
                with open(HOUSEHOLD_LABELS_FILE, "r") as f:
                    self.labels = [ln.strip() for ln in f if ln.strip()]
                self.model = torch.jit.load(str(HOUSEHOLD_MODEL_FILE), map_location="cpu").eval()
                self.enabled = True
            else:
                print(f"[HH] Skipping household classifier; files not found in {HOUSEHOLD_DIR}")
        except Exception as e:
            print(f"[HH] Failed to initialize household classifier: {e}")
            self.enabled = False

        self.tf = T.Compose([
            T.Resize((HOUSEHOLD_INPUT_SIZE, HOUSEHOLD_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
        ])

    def classify(self, roi_bgr: np.ndarray):
        if not self.enabled or roi_bgr is None or roi_bgr.size == 0:
            return ""
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil = torchvision.transforms.functional.to_pil_image(rgb)
        x = self.tf(pil).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            lbl = self.labels[idx.item()] if idx.item() < len(self.labels) else str(idx.item())
            return f"{lbl} {float(conf.item()*100.0):.1f}%"


def main():
    # Ensure required YOLO files exist
    for p in (CFG, WEIGHTS, NAMES):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    classes = load_classes(NAMES)

    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln_all = net.getLayerNames()
    out_layers = [ln_all[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # PiCamera2
    W, H = 1280, 720
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    # Household classifier
    hh = HouseholdClassifier()

    fps_s, prev = None, time.time()

    try:
        while True:
            frame_rgb = picam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            (Hf, Wf) = frame.shape[:2]

            # Build blob (BGR expected; we already have BGR)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INP_SIZE, INP_SIZE), swapRB=False, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(out_layers)

            boxes, confidences, class_ids = [], [], []
            for output in layer_outputs:
                for det in output:
                    scores = det[5:]
                    class_id = int(np.argmax(scores))
                    conf = float(scores[class_id])
                    if conf >= CONF_THRESH:
                        bx, by, bw, bh = det[0:4]
                        cx, cy = int(bx * Wf), int(by * Hf)
                        w, h = int(bw * Wf), int(bh * Hf)
                        x = int(cx - w / 2)
                        y = int(cy - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(conf)
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y, w, h = boxes[i]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(Wf, x + w), min(Hf, y + h)
                    label = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])

                    hh_text = ""
                    if hh.enabled and x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        hh_text = hh.classify(roi)

                    draw_box(frame, boxes[i], label, confidences[i], hh_text=hh_text)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            fps_s = fps if fps_s is None else (0.8 * fps_s + 0.2 * fps)

            cv2.putText(frame, f"FPS: {fps_s:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("YOLOv4-tiny (OpenCV DNN) + Household", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
