#!/usr/bin/env python3
"""

RGB + MiDaS monocular depth (640x480) on Raspberry Pi + IMX500, with:
  • Strictly synchronized RGB/Depth display (single producer bundle)
  • Robust exit/cleanup (handles Ctrl+C and TERM)
  • ZeroMQ publisher for a separate live 3D point-cloud viewer
    - Publishes normalized depth (float16), JPEG color map, intrinsics, and A/B metric mapping (if available)

USAGE
  $ python imx500_mono_depth_fasterRCNN_pub.py
  (Start your viewer in a separate process; this script publishes to tcp://*:5556)

Author : Rishav Kanth
"""

# ────────────────────────────────────────────────────────────────────────────────
# Standard library
import threading
import time
import os
import json
import math
import signal

# Third-party
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from picamera2 import Picamera2
import zmq

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (no magic numbers below this line)

# Torch / OpenCV
DEVICE_TYPE        = "cpu"          # Torch device string ("cpu" on Pi)
CV_COLORMAP        = cv2.COLORMAP_INFERNO
TORCH_NUM_THREADS  = os.cpu_count() # Allow Torch to use all cores
JPEG_QUALITY       = 80             # Publisher JPEG quality for the colormap

# MiDaS
MIDAS_TYPE         = "MiDaS_small"  # Depth model ID for torch.hub

# Camera / display geometry
CAM_WIDTH          = 640            # Pixels (W)
CAM_HEIGHT         = 480            # Pixels (H)
CAM_SIZE           = (CAM_WIDTH, CAM_HEIGHT)
DISPLAY_FPS        = 30             # Window refresh hint (ms timing)
TARGET_FPS_HUD     = 20.0           # HUD text only (no throttling)

# FOV (approx from IMX500 product brief; used to derive intrinsics for 640x480)
FOVX_DEG           = 66.3
FOVY_DEG           = 52.3

# Calibration / metric mapping: Z_m ≈ A * (1 - dn) + B
CALIB_FILE_PATH    = "imx500_metric.json"
METRIC_A_DEFAULT   = 2.0
METRIC_B_DEFAULT   = 0.0

# ZMQ
ZMQ_ENDPOINT       = "tcp://*:5556" # Publisher bind address
ZMQ_TOPIC_BYTES    = b"pc"          # Topic for multipart messages
ZMQ_SND_HWM        = 1              # Drop old frames if the viewer is slow

# Window / text
WINDOW_TITLE       = "Live | RGB + Depth (Publisher)"
HUD_POS_FPS        = (10, 25)       # (x,y)
HUD_POS_METRIC     = (10, 50)       # (x,y)
HUD_FONT           = cv2.FONT_HERSHEY_SIMPLEX
HUD_FPS_COLOR      = (255, 0, 0)    # BGR
HUD_METRIC_COLOR   = (50, 220, 50)  # BGR
HUD_FPS_SCALE      = 0.7
HUD_METRIC_SCALE   = 0.6
HUD_THICKNESS      = 2

# Timing
READY_POLL_SLEEP_S = 0.001          # Sleep while waiting for first bundle
BUNDLE_POLL_SLEEP_S= 0.0005         # Sleep while polling for next bundle
STARTUP_DELAY_S    = 0.2            # Picamera2 warmup

# ────────────────────────────────────────────────────────────────────────────────
# Derived configuration (computed once)

def _f_from_fov(pixels: int, degrees: float) -> float:
    """Compute focal length in pixels given image size and horizontal/vertical FOV."""
    return pixels / (2.0 * math.tan(math.radians(degrees) * 0.5))

# Derived intrinsics for 640x480
camFx = float(_f_from_fov(CAM_WIDTH,  FOVX_DEG))  # ≈ 489 px
camFy = float(_f_from_fov(CAM_HEIGHT, FOVY_DEG))  # ≈ 460 px
camCx = CAM_WIDTH  / 2.0
camCy = CAM_HEIGHT / 2.0

# Torch / OpenCV baseline setup
device = torch.device(DEVICE_TYPE)
torch.set_num_threads(TORCH_NUM_THREADS)
cv2.setUseOptimized(True)

# Metric calibration A/B (optionally loaded from JSON)
metricA = METRIC_A_DEFAULT
metricB = METRIC_B_DEFAULT
calibAvailable = False  # Whether a JSON metric mapping was found

# Runtime state flags
runningFlag = True      # Global run/stop latch for threads & main loop

# Shared bundle for strict sync display & publish
bundleLock = threading.Lock()
latestBundle = None     # Dict: {"id": int, "rgb": np.ndarray(H,W,3), "dn": np.ndarray(H,W), "cmap": np.ndarray(H,W,3)}

# ────────────────────────────────────────────────────────────────────────────────
# Setup: load MiDaS model and transforms

midas = torch.hub.load("intel-isl/MiDaS", MIDAS_TYPE).to(device).eval()
tf = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = tf.small_transform if MIDAS_TYPE == "MiDaS_small" else tf.default_transform

# ────────────────────────────────────────────────────────────────────────────────
# Setup: calibration JSON (metric A/B)

def _load_calib() -> None:
    """Load metric mapping (A,B) from CALIB_FILE_PATH if present."""
    global metricA, metricB, calibAvailable
    try:
        with open(CALIB_FILE_PATH, "r") as f:
            payload = json.load(f)
        metricA = float(payload["A"])
        metricB = float(payload["B"])
        calibAvailable = True
    except Exception:
        calibAvailable = False

_load_calib()

# ────────────────────────────────────────────────────────────────────────────────
# Robust kill handlers

def _stop(*_args) -> None:
    """Signal handler to request a clean shutdown."""
    global runningFlag
    runningFlag = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ────────────────────────────────────────────────────────────────────────────────
# ZMQ publisher

zmqContext = zmq.Context.instance()
zmqPub = zmqContext.socket(zmq.PUB)
zmqPub.setsockopt(zmq.SNDHWM, ZMQ_SND_HWM)   # Drop old frames when subscriber lags
zmqPub.bind(ZMQ_ENDPOINT)

def publish_frame(frameId: int,
                  depthNorm: np.ndarray,
                  depthCmap: np.ndarray,
                  metricAVal: float,
                  metricBVal: float,
                  camFxVal: float,
                  camFyVal: float,
                  camCxVal: float,
                  camCyVal: float) -> None:
    """
    Publish a single frame over ZMQ as a multipart message:
      [topic][JSON header][float16 normalized depth][JPEG colormap]
    """
    try:
        depthBytes = depthNorm.astype(np.float16).tobytes()
        ok, jpeg = cv2.imencode(".jpg", depthCmap, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return

        header = {
            "id": int(frameId),
            "w":  int(depthNorm.shape[1]),
            "h":  int(depthNorm.shape[0]),
            "a":  float(metricAVal),
            "b":  float(metricBVal),
            "fx": float(camFxVal),
            "fy": float(camFyVal),
            "cx": float(camCxVal),
            "cy": float(camCyVal),
            "metric": bool(calibAvailable)
        }
        zmqPub.send_multipart([ZMQ_TOPIC_BYTES,
                               json.dumps(header).encode("utf-8"),
                               depthBytes,
                               jpeg.tobytes()])
    except Exception:
        # Never block or crash the publisher on serialization errors
        pass

# ────────────────────────────────────────────────────────────────────────────────
# Worker thread: capture → depth → normalized map → bundle + publish

def worker_pipeline(piCam: Picamera2) -> None:
    """
    Strict, ordered pipeline:
      1) Capture RGB frame
      2) Run MiDaS depth
      3) Normalize depth to 0..1
      4) Build colorized map
      5) Publish bundle + ZMQ
    """
    frameId = 0

    while runningFlag:
        # 1) Capture RGB
        rgb = piCam.capture_array()  # (H,W,3) RGB888

        # 2) Depth inference (no gradients)
        with torch.inference_mode():
            inpTensor = transform(rgb).to(device)
            depthRaw = midas(inpTensor)
            depthRaw = F.interpolate(
                depthRaw.unsqueeze(1),
                size=CAM_SIZE[::-1],           # (H,W)
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

        # 3) Normalize to 0..1
        depthNorm = (depthRaw - depthRaw.min()) / (depthRaw.max() - depthRaw.min() + 1e-6)

        # 4) Pseudocolor for on-screen view / preview in viewer
        depthCmap = cv2.applyColorMap(np.uint8(depthNorm * 255), CV_COLORMAP)

        # 5) Publish bundle (for display sync) and ZMQ
        with bundleLock:
            frameId += 1
            global latestBundle
            latestBundle = {"id": frameId, "rgb": rgb, "dn": depthNorm, "cmap": depthCmap}

        publish_frame(frameId, depthNorm, depthCmap, metricA, metricB, camFx, camFy, camCx, camCy)

# ────────────────────────────────────────────────────────────────────────────────
# Main loop: synchronized display of RGB+Depth

def main() -> None:
    """Initialize camera, launch worker, and render synchronized RGB + depth."""
    global runningFlag

    # Camera init
    piCam = Picamera2()
    cfg = piCam.create_preview_configuration(main={"format": "RGB888", "size": CAM_SIZE})
    piCam.configure(cfg)
    piCam.start()
    time.sleep(STARTUP_DELAY_S)  # brief warmup

    # Start worker thread
    workerThread = threading.Thread(target=worker_pipeline, args=(piCam,), daemon=True)
    workerThread.start()

    # Wait for first processed bundle so the window opens with content
    while True:
        with bundleLock:
            isReady = latestBundle is not None
        if isReady:
            break
        time.sleep(READY_POLL_SLEEP_S)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    lastId = -1
    prevTime = time.time()
    fpsSmooth = None

    try:
        while runningFlag:
            # Poll until we see a new bundle (strict sync: 1 update == 1 processed frame)
            bundle = None
            while runningFlag:
                with bundleLock:
                    if latestBundle is not None and latestBundle["id"] != lastId:
                        bundle = latestBundle
                        lastId = bundle["id"]
                if bundle is not None:
                    break
                time.sleep(BUNDLE_POLL_SLEEP_S)

            if not runningFlag or bundle is None:
                break

            rgb = bundle["rgb"].copy()
            depthNorm = bundle["dn"].copy()
            depthVis = bundle["cmap"].copy()

            # FPS HUD (EMA smoothing)
            nowTime = time.time()
            instFps = 1.0 / max(1e-6, (nowTime - prevTime))
            prevTime = nowTime
            fpsSmooth = instFps if fpsSmooth is None else (0.8 * fpsSmooth + 0.2 * instFps)

            hudImg = rgb.copy()
            cv2.putText(
                hudImg,
                f"FPS(sync):{fpsSmooth:.1f} (target {TARGET_FPS_HUD:.0f})",
                HUD_POS_FPS, HUD_FONT, HUD_FPS_SCALE, HUD_FPS_COLOR, HUD_THICKNESS
            )
            if calibAvailable:
                cv2.putText(
                    hudImg,
                    f"Metric A={metricA:.3f}  B={metricB:.3f}",
                    HUD_POS_METRIC, HUD_FONT, HUD_METRIC_SCALE, HUD_METRIC_COLOR, HUD_THICKNESS
                )

            # Side-by-side: RGB HUD | Depth colormap
            cv2.imshow(WINDOW_TITLE, np.hstack((hudImg, depthVis)))

            key = cv2.waitKey(max(1, int(1000 / DISPLAY_FPS))) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # Esc / q / Q
                break

    except KeyboardInterrupt:
        pass
    finally:
        runningFlag = False
        try:
            piCam.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Last-resort hard exit to ensure threads don't hang the process
        os._exit(0)

# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
