# rt_object_detection_pi.py
# Real-time multi-object detection on Raspberry Pi 5 with PiCamera2 + PyTorch (Ultralytics)
# Optional second-stage classification with your MobileNetV2 household model.

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2
from ultralytics import YOLO  # supports yolov5n/s and yolov8n/s weights via model names

# Optional refinement
try:
    from household_classifier import HouseholdClassifier
except Exception:
    HouseholdClassifier = None


def smooth_fps(prev_fps, inst, alpha=0.8):
    return inst if prev_fps is None else (alpha * prev_fps + (1 - alpha) * inst)


def put_label(img, x1, y1, text, color=(0, 200, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1_txt = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, y1_txt), (x1 + tw + 2, y1_txt + th + 4), color, -1)
    cv2.putText(img, text, (x1 + 1, y1_txt + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time object detection on Raspberry Pi 5 (PiCamera2 + PyTorch)")
    parser.add_argument("--model", type=str, default="yolov8n",
                        choices=["yolov8n", "yolov5n", "yolov5s"],
                        help="Pretrained detector (COCO).")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference input size (short side). Try 320 or 416 for CPU.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections to draw.")
    parser.add_argument("--household-ckpt", type=str, default="",
                        help="Path to your MobileNetV2 household .pt checkpoint (optional).")
    parser.add_argument("--household-topk", type=int, default=1,
                        help="Top-k to display from household classifier.")
    parser.add_argument("--camera-size", type=str, default="1280x720",
                        help="PiCamera2 main stream size, e.g., 1280x720 or 640x480.")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS.")
    args = parser.parse_args()

    # Load detector
    model_name = args.model
    if model_name == "yolov8n":
        det = YOLO("yolov8n.pt")
    elif model_name == "yolov5n":
        det = YOLO("yolov5n.pt")
    else:
        det = YOLO("yolov5s.pt")

    # Hints to keep it lean on CPU
    det.to("cpu")
    # Ultralytics auto-handles preprocessing/letterbox; pass imgsz via predict call.

    # Optional second-stage classifier
    hh = None
    if args.household_ckpt and HouseholdClassifier is not None:
        ckpt_path = Path(args.household_ckpt)
        if ckpt_path.exists():
            try:
                hh = HouseholdClassifier(ckpt_path)
                print(f"[INFO] Household classifier loaded: {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Failed to load household classifier: {e}")
        else:
            print(f"[WARN] Household checkpoint not found: {ckpt_path}")

    # Camera config (PiCamera2)
    W, H = map(int, args.camera_size.lower().split("x"))
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()

    prev_t = time.time()
    fps_s = None

    try:
        while True:
            frame_rgb = picam.capture_array()  # RGB888
            # Convert to BGR for OpenCV drawing and detector convenience
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            t0 = time.time()
            # Run detector
            results_list = det.predict(
                source=frame,  # numpy array
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device="cpu",
                verbose=False,
                max_det=args.max_det
            )
            res = results_list[0]
            names = res.names  # COCO names dict

            # Draw detections
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    label = names.get(cls_id, str(cls_id))
                    color = (0, 200, 0)

                    # Optional second-stage classification on the ROI
                    hh_text = ""
                    if hh is not None:
                        # defensive clipping
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2c > x1c and y2c > y1c:
                            roi_bgr = frame[y1c:y2c, x1c:x2c]
                            hh_label, hh_prob = hh.classify(roi_bgr, topk=args.household_topk)
                            if hh_label:
                                hh_text = f" | HH:{hh_label} {hh_prob:.1f}%"

                    # Box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf*100:.1f}%{hh_text}"
                    put_label(frame, x1, y1, text, color)

            # FPS
            t1 = time.time()
            inst_fps = 1.0 / max(1e-6, (t1 - prev_t))
            prev_t = t1
            fps_s = smooth_fps(fps_s, inst_fps)
            if args.show_fps:
                cv2.putText(frame, f"FPS: {fps_s:.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(f"RT Detection [{model_name}] {args.imgsz}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Make Ctrl+C snappy on the Pi terminal
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
