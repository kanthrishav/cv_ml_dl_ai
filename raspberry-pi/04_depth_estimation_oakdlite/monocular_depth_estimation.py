#!/usr/bin/env python3
# monodepth_imx500.py

import time
import cv2
import torch
import numpy as np
from picamera2 import Picamera2, Preview
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 1) Select device (Pi CPU; if you have TorchScript or GPU on Jetson, tweak here)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load MiDaS small model from torch.hub
model_type = "MiDaS_small"  # small = faster, less accurate
midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()

# 3) MiDaS preprocessing & postprocessing transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "MiDaS_small":
    transform = midas_transforms.small_transform
else:
    transform = midas_transforms.default_transform

# 4) Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format":"RGB888", "size":(640,480)})
picam2.configure(config)
picam2.start()
time.sleep(2)  # warm up

print("Starting monocular depth on IMX500. Press 'q' to quit.")

prev_time = time.time()
fps_smooth = None

try:
    while True:
        # 5) Capture frame
        frame = picam2.capture_array()  # HxWx3 uint8 RGB

        # 6) Preprocess for MiDaS
        input_batch = transform(frame).to(device)

        # 7) Inference
        with torch.no_grad():
            prediction = midas(input_batch)

            # 8) Resize & normalize depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = np.nanmin(depth_map), np.nanmax(depth_map)
        depth_vis = (depth_map - depth_min) / (depth_max - depth_min)  # [0,1]
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # 9) Compute FPS
        now = time.time()
        inst_fps = 1.0 / (now - prev_time)
        prev_time = now
        fps_smooth = inst_fps if fps_smooth is None else (0.8*fps_smooth + 0.2*inst_fps)

        # 10) Overlay FPS on color frame
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # 11) Stack and show
        vis = np.hstack((frame, depth_vis))
        cv2.imshow("IMX500 Monocular Depth", vis)

        # 12) Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
