#!/usr/bin/env python3
"""
imx500_calibrate.py

Purpose
-------
Interactive helper to derive a simple metric mapping for monocular depth produced
elsewhere (e.g., MiDaS). You draw one or more rectangular ROIs over the live RGB
feed, then enter the known real-world distance (in meters) for each ROI. The
script fits the linear model:

    Z_m ≈ A * (1 - dn) + B

where:
    dn  ∈ [0, 1] is a *normalized* proxy value captured from the current frame
         (this tool uses a lightweight grayscale normalization as a stand-in).
    A,B are saved to `imx500_metric.json` for the main app to consume.

Controls
--------
• Mouse: click-drag-release to add an ROI
• D : enter known distance (meters) for the *last* ROI
• S : solve and save A,B to JSON
• C : clear all ROIs
• Q / Esc : quit

Author : Rishav Kanth
"""

# ────────────────────────────────────────────────────────────────────────────────
# Imports

import cv2
import numpy as np
import json
import os
import signal
import sys
import time
from picamera2 import Picamera2

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (no magic numbers below this line)

CALIB_FILE_PATH        = "imx500_metric.json"     # Output JSON with keys {"A": ..., "B": ...}
CAM_SIZE               = (640, 480)               # (W, H) for the preview stream
WINDOW_TITLE           = "Calibrate(IMX500)"      # OpenCV window name
STARTUP_DELAY_S        = 0.2                      # Camera warm-up delay

# ROI drawing styles
ROI_COLOR              = (0, 255, 255)            # BGR for finalized ROI boxes
ROI_ACTIVE_COLOR       = (0, 255, 0)              # BGR for active drag box
ROI_RECT_THICKNESS     = 2                        # px for finalized ROI box
ROI_ACTIVE_THICKNESS   = 1                        # px for active ROI box
TEXT_COLOR             = (0, 200, 255)            # BGR for distance text
TEXT_FONT              = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE             = 0.5
TEXT_THICKNESS         = 2
TEXT_OFFSET_Y          = 6                        # px upward offset so text sits above box

# Keyboard controls
KEY_CLEAR_LOWER        = ord('c')
KEY_CLEAR_UPPER        = ord('C')
KEY_DIST_LOWER         = ord('d')
KEY_DIST_UPPER         = ord('D')
KEY_SAVE_LOWER         = ord('s')
KEY_SAVE_UPPER         = ord('S')
KEY_QUIT_LOWER         = ord('q')
KEY_QUIT_UPPER         = ord('Q')
WAITKEY_DELAY_MS       = 1                        # cv2.waitKey delay

# Math / numeric stability
EPS                    = 1e-6

# ────────────────────────────────────────────────────────────────────────────────
# Global run-state (robust kill)

runningFlag = True

def _stop(*_):
    """Signal handler to request a clean shutdown."""
    global runningFlag
    runningFlag = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ────────────────────────────────────────────────────────────────────────────────
# ROI state (list of entries: [ (x1,y1,x2,y2), zMeters or None ])

rois = []            # Accumulated ROIs with optional distance assignments
currRect = None      # Active drag rectangle [x1, y1, x2, y2]
dnCache = None       # Cached normalized proxy image (computed on-demand)

# ────────────────────────────────────────────────────────────────────────────────
# Mouse interaction

def on_mouse(event, x, y, _flags, _param):
    """
    OpenCV mouse callback to collect rectangular ROIs:
      • Press & hold left button to start a rectangle.
      • Drag to size.
      • Release to finalize and append to the ROI list.
    """
    global currRect, rois
    if event == cv2.EVENT_LBUTTONDOWN:
        currRect = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and currRect is not None:
        currRect[2], currRect[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP and currRect is not None:
        x1, y1, x2, y2 = currRect
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        rois.append([(x1, y1, x2, y2), None])  # distance to be filled later
        currRect = None

# ────────────────────────────────────────────────────────────────────────────────
# Fitting the metric mapping

def solve_a_b(samples):
    """
    Fit A,B in Z = A*(1 - dn) + B using least squares.
    samples: list of tuples (dn_mean, Z_meters)
    Returns: (A, B)
    """
    dnVals = np.array([s[0] for s in samples], dtype=np.float64)
    zVals  = np.array([s[1] for s in samples], dtype=np.float64)
    X = np.stack([1.0 - dnVals, np.ones_like(dnVals)], axis=1)  # columns: (1 - dn), 1
    ab, *_ = np.linalg.lstsq(X, zVals, rcond=None)
    aVal, bVal = float(ab[0]), float(ab[1])
    return aVal, bVal

# ────────────────────────────────────────────────────────────────────────────────
# Main application loop

def main():
    global runningFlag, dnCache

    # Initialize camera (Picamera2 preview stream @ CAM_SIZE)
    picam = Picamera2()
    cfg = picam.create_preview_configuration(main={"format": "RGB888", "size": CAM_SIZE})
    picam.configure(cfg)
    picam.start()
    time.sleep(STARTUP_DELAY_S)

    # Create window and attach mouse callback
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    print("[Calibrate] Draw ROI(s) with the mouse.")
    print("[Calibrate] Press D to enter distance (m) for last ROI; S=save; C=clear; Q/Esc=quit.")

    try:
        while runningFlag:
            # Acquire current preview frame
            frame = picam.capture_array()
            vis = frame.copy()

            # Draw finalized ROIs (and label with known Z if available)
            for (x1, y1, x2, y2), zMeters in rois:
                cv2.rectangle(vis, (x1, y1), (x2, y2), ROI_COLOR, ROI_RECT_THICKNESS)
                if zMeters is not None:
                    cv2.putText(
                        vis,
                        f"{zMeters:.2f} m",
                        (x1, max(0, y1 - TEXT_OFFSET_Y)),
                        TEXT_FONT,
                        TEXT_SCALE,
                        TEXT_COLOR,
                        TEXT_THICKNESS
                    )

            # Draw active rectangle while dragging
            if currRect is not None:
                x1, y1, x2, y2 = currRect
                cv2.rectangle(vis, (x1, y1), (x2, y2), ROI_ACTIVE_COLOR, ROI_ACTIVE_THICKNESS)

            cv2.imshow(WINDOW_TITLE, vis)

            # Hotkeys
            key = cv2.waitKey(WAITKEY_DELAY_MS) & 0xFF
            if key in (27, KEY_QUIT_LOWER, KEY_QUIT_UPPER):  # Esc / q / Q
                break

            elif key in (KEY_CLEAR_LOWER, KEY_CLEAR_UPPER):
                rois.clear()
                dnCache = None
                print("[Calibrate] Cleared ROIs.")

            elif key in (KEY_DIST_LOWER, KEY_DIST_UPPER):
                if not rois:
                    print("[Calibrate] Draw an ROI first.")
                    continue
                # Ask for distance in meters for the last ROI
                try:
                    zMeters = float(input("Enter known distance (meters): ").strip())
                except Exception:
                    print("[Calibrate] Invalid input.")
                    continue

                # Compute normalized proxy (dn) if not already cached
                if dnCache is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    dnCache = (gray - gray.min()) / (gray.max() - gray.min() + EPS)

                # Record the (dn_mean, Z_m) sample for the last ROI
                (x1, y1, x2, y2), _ = rois[-1]
                roiDn = float(np.mean(dnCache[y1:y2, x1:x2]))
                rois[-1][1] = zMeters
                print(f"[Calibrate] ROI {len(rois)} set to {zMeters:.3f} m (dn_mean={roiDn:.4f})")

            elif key in (KEY_SAVE_LOWER, KEY_SAVE_UPPER):
                # Require at least one ROI with an assigned distance
                filled = [r for r in rois if r[1] is not None]
                if not filled:
                    print("[Calibrate] Add at least one distance (D) before saving.")
                    continue

                # Ensure dn cache exists (same proxy as when 'D' was pressed)
                if dnCache is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    dnCache = (gray - gray.min()) / (gray.max() - gray.min() + EPS)

                # Build samples
                samples = []
                for (x1, y1, x2, y2), zMeters in filled:
                    roiDn = float(np.mean(dnCache[y1:y2, x1:x2]))
                    samples.append((roiDn, float(zMeters)))

                # Solve A,B
                if len(samples) == 1:
                    # 1-point fit: fix B=0 (keeps behavior identical to original)
                    dnMean, z = samples[0]
                    aVal = z / max(EPS, (1.0 - dnMean))
                    bVal = 0.0
                else:
                    aVal, bVal = solve_a_b(samples)

                # Persist to JSON
                with open(CALIB_FILE_PATH, "w") as f:
                    json.dump({"A": float(aVal), "B": float(bVal)}, f)
                print(f"[Calibrate] Saved {CALIB_FILE_PATH}: A={aVal:.6f}, B={bVal:.6f}")
                # Keep running so the user can continue refining; quit when done.

    finally:
        # Robust cleanup (mirror original behavior)
        try:
            picam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
