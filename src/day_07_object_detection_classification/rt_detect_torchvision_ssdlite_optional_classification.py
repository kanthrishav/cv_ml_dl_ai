# rt_detect_torchvision_ssdlite.py
# PiCamera2 → torchvision SSDLite320_MobileNetV3 (COCO) → boxes + labels + probs + FPS
# Optional: household MobileNetV2 (TorchScript FP32) ROI refinement.
# No ultralytics, no SciPy; works with numpy==1.24.2.

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from picamera2 import Picamera2

import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision import transforms as T

# =========================
# Household classifier constants (your TorchScript MobileNetV2)
HOUSEHOLD_ENABLE = True  # set False to disable refinement
HOUSEHOLD_DIR = Path(__file__).parent / "export_torch"
HOUSEHOLD_MODEL_FILE = HOUSEHOLD_DIR / "model_ts.pt"   # TorchScript FP32 file
HOUSEHOLD_LABELS_FILE = HOUSEHOLD_DIR / "labels.txt"   # one label per line
HOUSEHOLD_INPUT_SIZE = 224
# =========================

# Fixed ImageNet normalization to avoid weights.meta access differences across versions
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def draw_box(img, box, label, score, hh_text="", color=(0, 200, 0)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    base = f"{label} {score*100:.1f}%"
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
                # Load labels
                with open(HOUSEHOLD_LABELS_FILE, "r") as f:
                    self.labels = [ln.strip() for ln in f if ln.strip()]
                # Load TorchScript model
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
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def classify(self, roi_bgr: np.ndarray):
        if not self.enabled or roi_bgr is None or roi_bgr.size == 0:
            return ""
        # BGR → RGB → PIL
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil = torchvision.transforms.functional.to_pil_image(rgb)
        x = self.tf(pil).unsqueeze(0)  # [1,3,H,W]
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            lbl = self.labels[idx.item()] if idx.item() < len(self.labels) else str(idx.item())
            return f"{lbl} {float(conf.item()*100.0):.1f}%"


def main():
    # Keep CPU threads modest on Pi 5
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    # Load detector (pretrained on COCO)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    categories = weights.meta.get("categories", [str(i) for i in range(91)])  # fallback

    model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to("cpu")

    # Explicit, version-robust 320x320 preprocessing
    det_tf = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # PiCamera2 setup
    W, H = 1280, 720
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    # Household classifier (optional)
    hh = HouseholdClassifier()

    fps_s, prev = None, time.time()

    try:
        while True:
            # Capture RGB888 and keep a BGR copy for drawing/cropping
            frame_rgb = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            Hf, Wf = frame_bgr.shape[:2]

            # Prepare 320x320 input
            pil320 = torchvision.transforms.functional.to_pil_image(
                cv2.resize(frame_rgb, (320, 320), interpolation=cv2.INTER_LINEAR)
            )
            inp = det_tf(pil320).unsqueeze(0)  # [1,3,320,320]

            with torch.no_grad():
                output = model(inp)[0]  # dict with 'boxes','labels','scores'

            # Boxes are in 320x320 coords → scale back to original (Wf,Hf)
            boxes = output["boxes"].cpu().numpy() if "boxes" in output else np.empty((0, 4))
            labels = output["labels"].cpu().numpy() if "labels" in output else np.empty((0,), dtype=int)
            scores = output["scores"].cpu().numpy() if "scores" in output else np.empty((0,))

            keep = scores > 0.25
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            sx = Wf / 320.0
            sy = Hf / 320.0
            if boxes.size > 0:
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * sx, 0, Wf - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * sy, 0, Hf - 1)

            for box, lab, sc in zip(boxes, labels, scores):
                x1, y1, x2, y2 = [int(v) for v in box]
                name = categories[int(lab)] if int(lab) < len(categories) else str(int(lab))

                hh_text = ""
                if hh.enabled and x2 > x1 and y2 > y1:
                    roi = frame_bgr[y1:y2, x1:x2]
                    hh_text = hh.classify(roi)

                draw_box(frame_bgr, (x1, y1, x2, y2), name, float(sc), hh_text=hh_text)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            fps_s = fps if fps_s is None else (0.8 * fps_s + 0.2 * fps)

            cv2.putText(frame_bgr, f"FPS: {fps_s:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("SSDLite320_MobileNetV3 + Household (fixed)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
