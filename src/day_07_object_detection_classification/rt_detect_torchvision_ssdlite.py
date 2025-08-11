# rt_detect_torchvision_ssdlite.py
# PiCamera2 → torchvision SSDLite320_MobileNetV3 (COCO) → boxes + labels + probs + FPS
# No ultralytics, no SciPy. Stays friendly with numpy==1.24.2.

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2

# torchvision detection (ensure torchvision is installed with your torch)
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights


def draw_box(img, box, label, score, color=(0, 200, 0)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {score*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 2, y_top + th + 4), color, -1)
    cv2.putText(img, txt, (x1 + 1, y_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # Pin torch threads for Pi 5 (tune if you like)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    # Load pretrained weights + model
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    categories = weights.meta["categories"]
    preprocess = weights.transforms()  # handles resize to 320 and normalization

    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    model.to("cpu")

    # Camera config (use HQ cam or 5MP; PiCamera2 will grab the active one)
    W, H = 1280, 720   # capture size; inference still goes to 320 via transforms
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    fps_smooth = None
    prev = time.time()

    try:
        while True:
            # PiCamera2 gives RGB888
            frame_rgb = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # To PIL for torchvision transforms
            pil = to_pil_image(frame_rgb)
            inp = preprocess(pil).unsqueeze(0)  # [1,3,320,320]

            with torch.no_grad():
                out = model(inp)[0]  # dict: boxes [N,4], labels [N], scores [N]

            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            # Filter low scores
            keep = scores > 0.25
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            for box, lab, sc in zip(boxes, labels, scores):
                name = categories[int(lab)] if int(lab) < len(categories) else str(int(lab))
                draw_box(frame_bgr, box, name, float(sc))

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            fps_smooth = fps if fps_smooth is None else 0.8 * fps_smooth + 0.2 * fps

            cv2.putText(frame_bgr, f"FPS: {fps_smooth:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("SSDLite320_MobileNetV3 (COCO)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
