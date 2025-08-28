#!/usr/bin/env python3
# OAK-D Lite + Raspberry Pi 5
# Fast, stable RGB + Depth (no detection) with tuning knobs and no-crash alignment.

import time
import numpy as np
import cv2
import depthai as dai

# ===================== TUNING CONSTANTS =====================
# Color preview geometry (depth aligns to this if ALIGN_DEPTH_TO_RGB=True).
RGB_PREVIEW_W, RGB_PREVIEW_H = 640, 360     # lower = faster (e.g., 640x360, 960x540)

# Purely UI scaling on the Pi (no device cost). 1.0 is fastest.
WINDOW_SCALE = 1.0

# Target FPS for camera streams and XLink.
FPS_TARGET = 30

# Stereo mono input resolution: "400p" (fast) or "720p" (slower, finer).
MONO_RES = "400p"

# Align depth to RGB FOV/pixels. Requires Left-Right Check to be ON (device rule).
ALIGN_DEPTH_TO_RGB = True   # True = same FOV as RGB but a bit slower; False = max FPS (left-mono FOV)

# Stereo quality/perf. If ALIGN_DEPTH_TO_RGB=True, LEFT_RIGHT_CHECK is forced True.
LEFT_RIGHT_CHECK = False    # used only when ALIGN_DEPTH_TO_RGB is False
SUBPIXEL = False            # smoother depth; costs FPS
MEDIAN_FILTER = "OFF"       # OFF | K3 | K5 | K7  (median kernel size)

# Host queue sizing — small prevents backpressure/hangs.
XLINK_MAX_QUEUE = 3

# Depth legend settings (meters).
AUTO_RANGE_DEFAULT = True
LEGEND_FIXED_MIN_M = 0.4
LEGEND_FIXED_MAX_M = 5.0
LEGEND_W = 70
INVALID_TO_BLACK = True
# ============================================================

def mono_res_from_text(s):
    s = s.upper()
    if "720" in s: return dai.MonoCameraProperties.SensorResolution.THE_720_P
    return dai.MonoCameraProperties.SensorResolution.THE_400_P

def median_from_text(s):
    s = s.upper()
    if s == "K3": return dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
    if s == "K5": return dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
    if s == "K7": return dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

def make_pipeline():
    p = dai.Pipeline()

    # ----- Color camera (preview supports keep-aspect) -----
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(FPS_TARGET)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewSize(RGB_PREVIEW_W, RGB_PREVIEW_H)
    cam.setPreviewKeepAspectRatio(True)   # full FOV (letterbox if aspect differs)

    # ----- Mono cameras for stereo -----
    monoL = p.create(dai.node.MonoCamera)
    monoR = p.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoL.setResolution(mono_res_from_text(MONO_RES))
    monoR.setResolution(mono_res_from_text(MONO_RES))
    monoL.setFps(FPS_TARGET)
    monoR.setFps(FPS_TARGET)

    # ----- Stereo depth -----
    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(median_from_text(MEDIAN_FILTER))
    # Enforce device requirement: alignment to RGB needs LRC ON.
    if ALIGN_DEPTH_TO_RGB:
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    else:
        stereo.setLeftRightCheck(LEFT_RIGHT_CHECK)
        # no alignment => depth is in rectified-left geometry (faster)

    stereo.setSubpixel(SUBPIXEL)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # ----- Host outputs (rate-limited & small queues) -----
    xout_rgb   = p.create(dai.node.XLinkOut); xout_rgb.setStreamName("rgb");   xout_rgb.setFpsLimit(FPS_TARGET)
    xout_depth = p.create(dai.node.XLinkOut); xout_depth.setStreamName("depth"); xout_depth.setFpsLimit(FPS_TARGET)
    cam.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return p

def robust_percentile_range_m(depth_mm):
    d = depth_mm.astype(np.float32)
    valid = d > 0
    if not np.any(valid):
        return LEGEND_FIXED_MIN_M, LEGEND_FIXED_MAX_M
    vals = d[valid]
    lo = np.percentile(vals, 5)
    hi = np.percentile(vals, 95)
    lo_m = max(0.2, lo / 1000.0)
    hi_m = max(lo_m + 0.1, min(10.0, hi / 1000.0))
    return lo_m, hi_m

def colorize_depth_with_legend(depth_mm, out_w, out_h, lo_m, hi_m):
    d = cv2.resize(depth_mm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    invalid = (d == 0)
    lo_mm = int(max(0.0, lo_m) * 1000)
    hi_mm = int(max(lo_mm + 1, hi_m * 1000))
    d_clamped = np.clip(d, lo_mm, hi_mm).astype(np.float32)
    # 0=near,1=far; invert so near=red, far=blue
    u8 = ((1.0 - (d_clamped - lo_mm) / float(hi_mm - lo_mm)) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    if INVALID_TO_BLACK:
        color[invalid] = (0, 0, 0)

    # Legend (top=near/red, bottom=far/blue)
    grad = np.linspace(255, 0, out_h, dtype=np.uint8).reshape(out_h, 1)
    legend = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_ticks = max(2, min(10, int(np.ceil(hi_m - lo_m))))
    for i in range(n_ticks + 1):
        val_m = lo_m + (hi_m - lo_m) * (i / n_ticks)
        y = int(out_h * (i / n_ticks))
        y = np.clip(y, 0, out_h - 1)
        cv2.line(legend, (0, y), (10, y), (255, 255, 255), 1)
        cv2.putText(legend, f"{val_m:.1f} m", (12, min(out_h - 6, y + 14)),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "NEAR", (5, 18), font, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, "FAR",  (5, out_h - 8), font, 0.5, (255, 255, 255), 1)
    legend = cv2.resize(legend, (LEGEND_W, out_h), interpolation=cv2.INTER_NEAREST)
    return np.hstack([color, legend])

def get_latest(q):
    """Drain non-blocking queue and return the newest packet; handle device hiccups."""
    try:
        pkt = q.tryGet()
    except RuntimeError:
        return None
    last = None
    while pkt is not None:
        last = pkt
        try:
            pkt = q.tryGet()
        except RuntimeError:
            break
    return last

def main():
    pipeline = make_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb   = dev.getOutputQueue("rgb",   maxSize=XLINK_MAX_QUEUE, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=XLINK_MAX_QUEUE, blocking=False)

        cv2.namedWindow("RGB",   cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        auto_range = AUTO_RANGE_DEFAULT
        fps_smooth, t_prev = None, time.time()

        print("Hotkeys:  a=auto-range   f=fullscreen   q=quit")

        latest_rgb = None
        latest_mm  = None

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('a'):
                auto_range = not auto_range
            elif k == ord('f'):
                for win in ("RGB", "Depth"):
                    state = int(cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN))
                    new_state = cv2.WINDOW_NORMAL if state == cv2.WINDOW_FULLSCREEN \
                                else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, new_state)

            pkt_rgb = get_latest(q_rgb)
            if pkt_rgb is not None:
                latest_rgb = pkt_rgb.getCvFrame()
            pkt_dep = get_latest(q_depth)
            if pkt_dep is not None:
                latest_mm = pkt_dep.getFrame()

            if latest_rgb is None or latest_mm is None:
                continue

            if auto_range:
                lo_m, hi_m = robust_percentile_range_m(latest_mm)
            else:
                lo_m, hi_m = LEGEND_FIXED_MIN_M, LEGEND_FIXED_MAX_M

            depth_vis = colorize_depth_with_legend(
                latest_mm, latest_rgb.shape[1], latest_rgb.shape[0], lo_m, hi_m
            )

            rgb_show, depth_show = latest_rgb, depth_vis
            if WINDOW_SCALE != 1.0:
                rgb_show   = cv2.resize(rgb_show, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                                        interpolation=cv2.INTER_NEAREST)
                depth_show = cv2.resize(depth_show, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                                        interpolation=cv2.INTER_NEAREST)

            now = time.time()
            inst = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            fps_smooth = inst if fps_smooth is None else (0.8*fps_smooth + 0.2*inst)

            cv2.putText(rgb_show,   f"FPS {fps_smooth:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(depth_show, f"{'AUTO' if auto_range else 'FIXED'} [{lo_m:.2f}-{hi_m:.2f} m]",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("RGB",   rgb_show)
            cv2.imshow("Depth", depth_show)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
