# rt_detect_cv2dnn_yolov4tiny.py
# PiCamera2 → OpenCV DNN YOLOv4-tiny → boxes + labels + probs + FPS
# No ultralytics, no SciPy. Works with numpy==1.24.2.

import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"
CONF_THRESH = 0.25
NMS_THRESH = 0.45
INP_SIZE = 416  # 320 or 416 recommended on Pi 5 CPU


def load_classes(names_path):
    with open(names_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def draw_box(img, box, label, conf, color=(0, 200, 0)):
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 2, y_top + th + 4), color, -1)
    cv2.putText(img, txt, (x1 + 1, y_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # Sanity check files
    for p in (CFG, WEIGHTS, NAMES):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    classes = load_classes(NAMES)

    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln_all = net.getLayerNames()
    out_layers = [ln_all[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Camera
    W, H = 1280, 720
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    fps_smooth, prev = None, time.time()

    try:
        while True:
            frame_rgb = picam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            (Hf, Wf) = frame.shape[:2]

            # blob (YOLO expects 1/255 scaled, BGR)
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
                        # YOLO outputs center x,y and width/height as ratios
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
                    label = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])
                    draw_box(frame, boxes[i], label, confidences[i])

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            fps_smooth = fps if fps_smooth is None else (0.8 * fps_smooth + 0.2 * fps)

            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("YOLOv4-tiny (OpenCV DNN)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
