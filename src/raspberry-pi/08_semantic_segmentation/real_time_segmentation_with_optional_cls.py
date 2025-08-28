#!/usr/bin/env python3
"""
Real-Time Semantic Segmentation (PiCamera2 + PyTorch) with Optional Per-Object Classification

- Segmentation: LRASPP MobileNetV3 Large (torchvision, pretrained on VOC-21)
- Camera: PiCamera2 (use your Raspberry Pi HQ or 5MP camera)
- Classification (optional):
    * If --use_custom_cls true: load TorchScript MobileNetV2 from export_torch/model_ts.pt
      and labels from export_torch/labels.txt (index -> name, one per line).
    * Else: use torchvision MobileNetV2 pretrained (ImageNet) + ImageNet class labels.

All paths are hard-coded in this file (no path args). The ONLY arg is --use_custom_cls.

Quit: press 'q'
"""

import os
import time
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from picamera2 import Picamera2
from typing import List, Tuple, Optional

from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
from torchvision import models as tv_models
import torch.nn.functional as F


# ----------------------------
# Configuration
# ----------------------------
# Camera (display) resolution
CAM_W, CAM_H = 640, 360        # 16:9 is fast for preview; adjust if you prefer 640x480

# Segmentation model input (smaller -> faster)
IN_W, IN_H = 320, 320

# Overlay alpha for segmentation mask
ALPHA = 0.45

# How many largest components to classify per frame
MAX_COMPONENTS_TOTAL = 5

# Ignore very small components (pixels) to avoid noise
MIN_PIXELS_COMPONENT = 700

# Show top-K semantic classes by area for legend
SHOW_TOP_K = 5

# FPS smoothing
FPS_SMOOTHING = 0.8

# Torch CPU threading
USE_THREADS = max(1, (os.cpu_count() or 4))

# Classification input size and normalization (ImageNet)
CLS_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Fixed VOC-21 labels and palette
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]
VOC_PALETTE = np.array([
    [  0,   0,   0],  # background
    [128,   0,   0],  # aeroplane
    [  0, 128,   0],  # bicycle
    [128, 128,   0],  # bird
    [  0,   0, 128],  # boat
    [128,   0, 128],  # bottle
    [  0, 128, 128],  # bus
    [128, 128, 128],  # car
    [ 64,   0,   0],  # cat
    [192,   0,   0],  # chair
    [ 64, 128,   0],  # cow
    [192, 128,   0],  # diningtable
    [ 64,   0, 128],  # dog
    [192,   0, 128],  # horse
    [ 64, 128, 128],  # motorbike
    [192, 128, 128],  # person
    [  0,  64,   0],  # potted plant
    [128,  64,   0],  # sheep
    [  0, 192,   0],  # sofa
    [128, 192,   0],  # train
    [  0,  64, 128],  # tv/monitor
], dtype=np.uint8)

# Paths for optional custom classifier (hard-coded)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "export_torch")
TS_MODEL_PATH = os.path.join(EXPORT_DIR, "model_ts.pt")
TS_LABELS_PATH = os.path.join(EXPORT_DIR, "labels.txt")


# ----------------------------
# Utilities
# ----------------------------
def build_seg_model():
    """Load LRASPP MobileNetV3 Large with pretrained weights; prepare preprocess."""
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()
    try:
        torch.set_num_threads(USE_THREADS)
    except Exception:
        pass

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def preprocess(rgb_np: np.ndarray) -> torch.Tensor:
        # Resize to model input
        resized = cv2.resize(rgb_np, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # CHW
        t = normalize(t)
        return t.unsqueeze(0)  # NCHW

    return model, preprocess


def colorize_mask(mask_hw: np.ndarray, palette: np.ndarray) -> np.ndarray:
    mask_hw = np.clip(mask_hw, 0, len(palette) - 1)
    return palette[mask_hw]


def topk_classes(mask_hw: np.ndarray, k: int, classes=VOC_CLASSES,
                 background_index=0, min_pixels=200):
    vals, counts = np.unique(mask_hw, return_counts=True)
    items = [(int(v), int(c)) for v, c in zip(vals, counts)
             if v != background_index and c >= min_pixels and v < len(classes)]
    items.sort(key=lambda x: x[1], reverse=True)
    return [(classes[v], c) for v, c in items[:k]]


def find_components_by_class(mask_hw: np.ndarray,
                             min_pixels: int,
                             max_total: int) -> List[Tuple[int, Tuple[int, int, int, int], int]]:
    """
    For each class (excluding background), run connected components on mask==class,
    collect (area, bbox, class_id). Return top 'max_total' by area across all classes.

    Returns list of tuples: (area, (x, y, w, h), class_id)
    """
    H, W = mask_hw.shape
    candidates = []
    for cls_id in range(1, len(VOC_CLASSES)):  # skip background=0
        binary = (mask_hw == cls_id).astype(np.uint8)
        if binary.sum() < min_pixels:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        # stats: [label, x, y, w, h, area] for label >=1 are components
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            w = int(stats[lab, cv2.CC_STAT_WIDTH])
            h = int(stats[lab, cv2.CC_STAT_HEIGHT])
            # clamp bbox to image
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            candidates.append((area, (x, y, w, h), cls_id))

    # largest first
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[:max_total]


# ----------------------------
# Classification (optional)
# ----------------------------
class Classifier:
    def __init__(self, use_custom: bool):
        self.use_custom = use_custom
        self.device = torch.device("cpu")
        self.model = None
        self.labels = None
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if self.use_custom:
            # Load TorchScript MobileNetV2 and labels from export_torch
            if not os.path.isfile(TS_MODEL_PATH):
                raise FileNotFoundError(f"Custom classifier model not found: {TS_MODEL_PATH}")
            if not os.path.isfile(TS_LABELS_PATH):
                raise FileNotFoundError(f"Custom labels file not found: {TS_LABELS_PATH}")
            self.model = torch.jit.load(TS_MODEL_PATH, map_location=self.device).eval()
            with open(TS_LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [ln.strip() for ln in f if ln.strip()]
        else:
            # Fallback: torchvision MobileNetV2 pretrained
            self.model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
            # Load ImageNet class names (hard-coded minimal fallback)
            # If you prefer, replace with a local 'imagenet_classes.txt'.
            self.labels = None  # will create "class_<idx>" if None

    def classify_crop(self, bgr_crop: np.ndarray) -> Tuple[str, float]:
        """
        bgr_crop: HxWx3 uint8
        Returns: (label, probability)
        """
        # Resize to 224x224 and convert to RGB
        resized = cv2.resize(bgr_crop, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = self.preprocess(rgb).unsqueeze(0)  # 1x3x224x224
        with torch.no_grad():
            logits = self.model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            p = float(probs[idx])
        label = self._label_for_idx(idx)
        return label, p

    def _label_for_idx(self, idx: int) -> str:
        if self.labels and 0 <= idx < len(self.labels):
            return self.labels[idx]
        return f"class_{idx}"


# ----------------------------
# Main
# ----------------------------
def main(use_custom_cls: bool):
    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Segmentation
    seg_model, seg_preprocess = build_seg_model()

    # Optional classifier
    classifier: Optional[Classifier] = None
    try:
        classifier = Classifier(use_custom=use_custom_cls)
        cls_enabled = True
        cls_source = "custom TorchScript" if use_custom_cls else "torchvision MobileNetV2"
        print(f"[INFO] Classification enabled ({cls_source}).")
    except Exception as e:
        cls_enabled = False
        print(f"[WARN] Classification disabled: {e}")

    fps_smooth = None
    t_prev = time.time()

    try:
        with torch.no_grad():
            while True:
                # Capture frame
                frame_rgb = picam2.capture_array()  # RGB
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Segmentation inference
                inp = seg_preprocess(frame_rgb)
                out = seg_model(inp)["out"]           # 1xCxHxW
                pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                mask_up = cv2.resize(pred, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)

                # Build overlay
                color_mask = VOC_PALETTE[mask_up]
                overlay = cv2.addWeighted(frame_bgr, 1.0 - ALPHA, color_mask, ALPHA, 0.0)

                # Legend of top semantic classes
                legend = topk_classes(mask_up, SHOW_TOP_K, VOC_CLASSES)
                y = 24
                cv2.putText(overlay, "Top semantic classes:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y += 22
                total_pixels = mask_up.size
                for name, count in legend:
                    pct = 100.0 * count / total_pixels
                    text = f"- {name}: {pct:.1f}%"
                    cv2.putText(overlay, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    y += 20

                # Extract components and (optionally) classify each crop
                components = find_components_by_class(mask_up, MIN_PIXELS_COMPONENT, MAX_COMPONENTS_TOTAL)
                for area, (x, y, w, h), cls_id in components:
                    # Expand bbox slightly for classification context
                    pad = 4
                    x0 = max(0, x - pad); y0 = max(0, y - pad)
                    x1 = min(CAM_W, x + w + pad); y1 = min(CAM_H, y + h + pad)
                    crop = frame_bgr[y0:y1, x0:x1]

                    # Draw bbox with semantic class color
                    color = tuple(int(c) for c in VOC_PALETTE[cls_id].tolist())
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

                    # Label text: semantic cls [+ classifier result if enabled]
                    label_txt = VOC_CLASSES[cls_id]
                    if cls_enabled and crop.size > 0:
                        try:
                            c_label, c_prob = classifier.classify_crop(crop)
                            label_txt = f"{label_txt} | {c_label} {c_prob*100:.1f}%"
                        except Exception as e:
                            # Keep going even if one crop fails
                            pass

                    # Text background and text
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ty = max(0, y0 - th - 4)
                    cv2.rectangle(overlay, (x0, ty), (x0 + tw + 2, ty + th + 4), color, -1)
                    cv2.putText(overlay, label_txt, (x0 + 1, ty + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # FPS
                t_now = time.time()
                inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
                t_prev = t_now
                fps_smooth = inst_fps if fps_smooth is None else (
                    FPS_SMOOTHING * fps_smooth + (1.0 - FPS_SMOOTHING) * inst_fps
                )
                cv2.putText(overlay, f"FPS: {fps_smooth:.1f}", (overlay.shape[1]-140, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Show
                cv2.imshow("RPi5 Segmentation + Optional Classification", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPi5 Semantic Segmentation with Optional Classification")
    parser.add_argument("--use_custom_cls", type=str, default="false",
                        help="Set to 'true' to use export_torch/model_ts.pt and labels.txt for classification")
    args = parser.parse_args()
    use_custom = args.use_custom_cls.strip().lower() in ("1", "true", "yes", "y")
    main(use_custom_cls=use_custom)
