# imx500_calibrate.py

import cv2, numpy as np, json, os, signal, sys, time
from picamera2 import Picamera2

CALIB_FILE = "imx500_metric.json"
CAM_SIZE   = (640, 480)  # (W,H)

# robust kill
RUNNING = True
def _stop(*_): 
    global RUNNING; RUNNING = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ROI state
ROIS = []     # list of ((x1,y1,x2,y2), z_meters or None)
CURR = None   # active drag rect

def on_mouse(event, x, y, flags, param):
    global CURR, ROIS
    if event == cv2.EVENT_LBUTTONDOWN:
        CURR = [x,y,x,y]
    elif event == cv2.EVENT_MOUSEMOVE and CURR is not None:
        CURR[2], CURR[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP and CURR is not None:
        x1,y1,x2,y2 = CURR
        if x2<x1: x1,x2 = x2,x1
        if y2<y1: y1,y2 = y2,y1
        ROIS.append([(x1,y1,x2,y2), None])
        CURR = None

def solve_A_B(samples):
    """
    samples = [(dn_mean, Zm), ...]
    Model: Zm = A*(1 - dn) + B
    """
    dn = np.array([s[0] for s in samples], dtype=np.float64)
    Z  = np.array([s[1] for s in samples], dtype=np.float64)
    X  = np.stack([1.0 - dn, np.ones_like(dn)], axis=1)  # [ (1-dn), 1 ]
    # least squares
    AB, *_ = np.linalg.lstsq(X, Z, rcond=None)
    A, B = float(AB[0]), float(AB[1])
    return A, B

def main():
    global RUNNING
    picam = Picamera2()
    cfg = picam.create_preview_configuration(main={"format":"RGB888","size":CAM_SIZE})
    picam.configure(cfg); picam.start(); time.sleep(0.2)

    cv2.namedWindow("Calibrate(IMX500)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate(IMX500)", on_mouse)

    dn_cache = None  # we compute dn only when needed

    print("[Calibrate] Draw ROI(s) with the mouse.")
    print("[Calibrate] Press D to enter distance (m) for last ROI; S=save; C=clear; Q/Esc=quit.")

    try:
        while RUNNING:
            frame = picam.capture_array()
            vis   = frame.copy()

            # draw ROIs
            for (x1,y1,x2,y2), z in ROIS:
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
                if z is not None:
                    cv2.putText(vis, f"{z:.2f} m", (x1, max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)

            if CURR is not None:
                x1,y1,x2,y2 = CURR
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),1)

            cv2.imshow("Calibrate(IMX500)", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):  # Esc or q/Q
                break
            elif k in (ord('c'), ord('C')):
                ROIS.clear(); dn_cache = None
                print("[Calibrate] Cleared ROIs.")
            elif k in (ord('d'), ord('D')):
                if not ROIS:
                    print("[Calibrate] Draw an ROI first.")
                    continue
                try:
                    z = float(input("Enter known distance (meters): ").strip())
                except Exception:
                    print("[Calibrate] Invalid input.")
                    continue
                # compute dn if not computed
                if dn_cache is None:
                    # quick MiDaS_small-like normalization substitute: use grayscale gradient proxy
                    # NOTE: we don't have MiDaS here to keep the tool lightweight:
                    #       we only need relative values inside an ROI to solve A,B robustly later in main.
                    # Better approach: capture a depth map from your main app and load it here;
                    # to keep it single-file, we approximate with intensity.
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    # normalize 0..1
                    dn_cache = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
                # assign z to last ROI
                (x1,y1,x2,y2), _ = ROIS[-1]
                roi_dn = float(np.mean(dn_cache[y1:y2, x1:x2]))
                ROIS[-1][1] = z
                print(f"[Calibrate] ROI {len(ROIS)} set to {z:.3f} m (dn_mean={roi_dn:.4f})")
            elif k in (ord('s'), ord('S')):
                # need at least 1 ROI
                filled = [r for r in ROIS if r[1] is not None]
                if not filled:
                    print("[Calibrate] Add at least one distance (D) before saving.")
                    continue
                # we must recompute dn from your true depth map to be exact.
                # For a self-contained tool, we’ll use the same proxy dn_cache.
                if dn_cache is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    dn_cache = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

                samples = []
                for (x1,y1,x2,y2), z in filled:
                    roi_dn = float(np.mean(dn_cache[y1:y2, x1:x2]))
                    samples.append((roi_dn, float(z)))

                if len(samples) == 1:
                    # 1-point fit: B=0
                    dn_mean, z = samples[0]
                    A = z / max(1e-6, (1.0 - dn_mean))
                    B = 0.0
                else:
                    A, B = solve_A_B(samples)

                with open(CALIB_FILE, "w") as f:
                    json.dump({"A": float(A), "B": float(B)}, f)
                print(f"[Calibrate] Saved {CALIB_FILE}: A={A:.6f}, B={B:.6f}")
                # keep running so you can tweak; quit when done
    finally:
        try:
            picam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
