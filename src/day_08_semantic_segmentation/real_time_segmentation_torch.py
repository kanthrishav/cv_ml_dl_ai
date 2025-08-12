#!/usr/bin/env python3
"""
Real-Time Semantic Segmentation on Raspberry Pi 5 (CPU) using PiCamera2 + PyTorch
Model: LRASPP MobileNetV3 Large (torchvision, pretrained on VOC-21)
Camera: Raspberry Pi HQ camera (or 5MP Pi camera), via PiCamera2

Quit: press 'q' in the display window
"""

import os
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from picamera2 import Picamera2
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)

# ----------------------------
# Configuration
# ----------------------------
# Camera preview size (display)
CAM_W, CAM_H = 640, 360    # 16:9; adjust if you prefer 4:3 e.g., 640x480
# Model input size (inference); smaller => faster
IN_W, IN_H = 320, 320

ALPHA = 0.45               # overlay strength
SHOW_TOP_K = 5             # top classes by area (excluding background)
FPS_SMOOTHING = 0.8        # higher -> smoother FPS
USE_THREADS = max(1, (os.cpu_count() or 4))  # let PyTorch use CPU cores

# Fixed ImageNet normalization (robust across torchvision versions)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# VOC-21 class labels (LRASPP MobileNetV3 Large pretrained)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

# Distinct BGR colors for the 21 classes (OpenCV uses BGR)
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


def build_model():
    """Load LRASPP MobileNetV3 Large with pretrained weights."""
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()

    # Torch threading for CPU
    try:
        torch.set_num_threads(USE_THREADS)
    except Exception:
        pass

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def preprocess(rgb_np: np.ndarray) -> torch.Tensor:
        """
        rgb_np: HxWx3 uint8 (RGB)
        returns: 1x3xIN_HxIN_W float tensor
        """
        resized = cv2.resize(rgb_np, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # CHW
        t = normalize(t)
        return t.unsqueeze(0)  # NCHW

    # Stick with VOC classes (21)
    classes = VOC_CLASSES
    return model, preprocess, classes


def colorize_mask(mask_hw: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map each class index to a BGR color image."""
    mask_hw = np.clip(mask_hw, 0, len(palette) - 1)
    return palette[mask_hw]


def topk_classes(mask_hw: np.ndarray, k: int, classes, background_index=0, min_pixels=200):
    """Return top-k class names by area (#pixels), excluding background and tiny speckles."""
    vals, counts = np.unique(mask_hw, return_counts=True)
    items = [(int(v), int(c)) for v, c in zip(vals, counts)
             if v != background_index and c >= min_pixels and v < len(classes)]
    items.sort(key=lambda x: x[1], reverse=True)
    return [(classes[v], c) for v, c in items[:k]]


def main():
    # 1) Camera setup (PiCamera2)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # 2) Model
    model, preprocess, classes = build_model()

    # 3) Main loop
    fps_smooth = None
    t_prev = time.time()

    try:
        with torch.no_grad():
            while True:
                # Capture RGB frame from PiCamera2
                frame_rgb = picam2.capture_array()  # HxWx3 RGB
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Prepare input
                inp = preprocess(frame_rgb)

                # Inference
                out = model(inp)["out"]  # 1xCxHxW
                pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Upscale mask to display size
                mask_up = cv2.resize(pred, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)

                # Colorize and overlay
                color_mask = colorize_mask(mask_up, VOC_PALETTE)
                overlay = cv2.addWeighted(frame_bgr, 1.0 - ALPHA, color_mask, ALPHA, 0.0)

                # Legend: top-K classes by area
                legend = topk_classes(mask_up, SHOW_TOP_K, classes)
                y = 24
                cv2.putText(overlay, "Top classes:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y += 22
                total_pixels = mask_up.size
                for name, count in legend:
                    pct = 100.0 * count / total_pixels
                    text = f"- {name}: {pct:.1f}%"
                    cv2.putText(overlay, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    y += 20

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
                cv2.imshow("RPi5 Semantic Segmentation (LRASPP-MobileNetV3)", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cv2.destroyAllWindows()
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
