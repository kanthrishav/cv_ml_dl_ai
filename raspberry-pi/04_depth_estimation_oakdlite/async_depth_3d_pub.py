#!/usr/bin/env python3
# async_depth_3d_pub.py
# RGB + MiDaS depth (640x480), strict-sync bundle, robust exit, and ZMQ publisher
# Publishes: normalized depth (float16) + JPEG colormap + intrinsics + A,B (if available)

import threading, time, os, json, math, signal
import cv2, numpy as np, torch
from picamera2 import Picamera2
import torch.nn.functional as F
import zmq

# ---------------- Settings ----------------
DEVICE        = torch.device("cpu")
torch.set_num_threads(os.cpu_count())
cv2.setUseOptimized(True)

MIDAS_TYPE    = "MiDaS_small"
CAM_SIZE      = (640, 480)  # (W,H)
W, H          = CAM_SIZE
DISPLAY_FPS   = 30
TARGET_FPS    = 20.0        # HUD only; no throttling

# IMX500 intrinsics (scaled to 640x480) from product brief FoV (≈66.3° × 52.3°)
FOVX_DEG, FOVY_DEG = 66.3, 52.3
def _f_from_fov(pix, deg): return pix / (2.0 * math.tan(math.radians(deg)*0.5))
fx = float(_f_from_fov(W, FOVX_DEG))   # ~489 px
fy = float(_f_from_fov(H, FOVY_DEG))   # ~460 px
cx, cy = W/2.0, H/2.0

# Metric mapping Z_m ≈ A*(1 - dn) + B (loaded if available)
CALIB_FILE = "imx500_metric.json"
METRIC_A, METRIC_B = 2.0, 0.0
CALIB_AVAILABLE    = False
def _load_calib():
    global METRIC_A, METRIC_B, CALIB_AVAILABLE
    try:
        with open(CALIB_FILE, "r") as f:
            d = json.load(f)
            METRIC_A = float(d["A"]); METRIC_B = float(d["B"])
            CALIB_AVAILABLE = True
    except Exception:
        CALIB_AVAILABLE = False
_load_calib()

# ---------------- Robust kill ----------------
RUNNING = True
def _stop(*_):
    global RUNNING; RUNNING = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------------- Shared bundle for strict sync ----------------
bundle_lock = threading.Lock()
latest_bundle = None  # {"id":int,"rgb":..., "dn":..., "cmap":...}

# ---------------- Load MiDaS ----------------
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_TYPE).to(DEVICE).eval()
tf = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = tf.small_transform if MIDAS_TYPE=="MiDaS_small" else tf.default_transform

# ---------------- ZMQ Publisher ----------------
context = zmq.Context.instance()
pub = context.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 1)             # drop old if slow consumer
pub.bind("tcp://*:5556")                  # publishes on localhost:5556

def publish_frame(frame_id, dn, cmap, A, B, fx, fy, cx, cy):
    # dn -> float16 raw; cmap -> JPEG bytes
    try:
        dn16 = dn.astype(np.float16).tobytes()
        ok, jpg = cv2.imencode(".jpg", cmap, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        header = {
            "id": int(frame_id), "w": int(dn.shape[1]), "h": int(dn.shape[0]),
            "a": float(A), "b": float(B),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "metric": bool(CALIB_AVAILABLE)
        }
        pub.send_multipart([b"pc", json.dumps(header).encode("utf-8"), dn16, jpg.tobytes()])
    except Exception:
        pass  # never block or crash publisher

# ---------------- Worker: capture -> depth -> publish bundle ----------------
def worker_pipeline(picam):
    """Strict order pipeline; no sleep; pushes fully-processed bundles and publishes ZMQ."""
    frame_id = 0
    while RUNNING:
        rgb = picam.capture_array()  # (H,W,3) RGB888

        with torch.inference_mode():
            inp = transform(rgb).to(DEVICE)
            d = midas(inp)
            d = F.interpolate(d.unsqueeze(1), size=CAM_SIZE[::-1],
                              mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
        cmap = cv2.applyColorMap(np.uint8(dn*255), cv2.COLORMAP_INFERNO)

        with bundle_lock:
            frame_id += 1
            global latest_bundle
            latest_bundle = {"id":frame_id, "rgb":rgb, "dn":dn, "cmap":cmap}

        publish_frame(frame_id, dn, cmap, METRIC_A, METRIC_B, fx, fy, cx, cy)

# ---------------- Main (display RGB + Depth) ----------------
def main():
    global RUNNING

    picam = Picamera2()
    cfg = picam.create_preview_configuration(main={"format":"RGB888","size":CAM_SIZE})
    picam.configure(cfg); picam.start(); time.sleep(0.2)

    t = threading.Thread(target=worker_pipeline, args=(picam,), daemon=True)
    t.start()

    # Wait for first processed bundle (so window opens with data)
    while True:
        with bundle_lock:
            ready = latest_bundle is not None
        if ready: break
        time.sleep(0.001)

    cv2.namedWindow("Live | RGB + Depth (Publisher)", cv2.WINDOW_NORMAL)

    last_id = -1
    prev = time.time()
    fps = None

    try:
        while RUNNING:
            # Wait for next bundle (strict sync: one update = one processed frame)
            b = None
            while RUNNING:
                with bundle_lock:
                    if latest_bundle is not None and latest_bundle["id"] != last_id:
                        b = latest_bundle; last_id = b["id"]
                if b is not None: break
                time.sleep(0.0005)
            if not RUNNING or b is None:
                break

            rgb = b["rgb"].copy()
            dn  = b["dn"].copy()
            dov = b["cmap"].copy()

            # FPS HUD
            now = time.time(); inst = 1.0/max(1e-6, now-prev); prev = now
            fps = inst if fps is None else (0.8*fps + 0.2*inst)
            hud = rgb.copy()
            cv2.putText(hud, f"FPS(sync):{fps:.1f} (target {TARGET_FPS:.0f})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if CALIB_AVAILABLE:
                cv2.putText(hud, f"Metric A={METRIC_A:.3f}  B={METRIC_B:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

            # Show window (RGB+Depth)
            cv2.imshow("Live | RGB + Depth (Publisher)", np.hstack((hud, dov)))
            k = cv2.waitKey(max(1, int(1000/DISPLAY_FPS))) & 0xFF
            if k in (27, ord('q'), ord('Q')):   # Esc / q / Q
                break

    except KeyboardInterrupt:
        pass
    finally:
        RUNNING = False
        try: picam.stop()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        os._exit(0)  # last-resort hard exit

if __name__ == "__main__":
    main()
