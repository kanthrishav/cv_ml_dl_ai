#!/usr/bin/env python3
"""
oakd_stereo_depth_pub.py

OAK-D Lite + Raspberry Pi 5
Fast, stable RGB + Depth (no detection) with tuning knobs and robust alignment.
Also publishes a ZMQ stream for a live 3D point-cloud viewer (metric depth + intrinsics).

What it does
------------
1) Builds a DepthAI pipeline:
   • Color preview for the RGB window (letterboxed if aspect differs).
   • StereoDepth for metric depth (in millimeters), optionally aligned to RGB FOV.
2) Shows two synchronized OpenCV windows: "RGB" and "Depth" (with legend).
3) Publishes each depth frame (at its native/aligned size) over ZMQ:
   topic=b"pc", payload=(uint16 depth-mm, JPEG colormap, intrinsics for THAT size).

Controls
--------
a : toggle auto-range for the depth color legend
f : toggle fullscreen for both windows
q : quit

Author : Rishav Kanth
"""

# ────────────────────────────────────────────────────────────────────────────────
# Standard / third-party imports

import time
import os
import json
import numpy as np
import cv2
import depthai as dai
import zmq

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (define *everything* configurable here; no magic numbers in code)

# RGB preview (depth legend is drawn to match this view; device cost is low)
RGB_PREVIEW_W = 640
RGB_PREVIEW_H = 360

# UI scaling for the Pi display (applied on host only)
WINDOW_SCALE = 1.0

# Target FPS for camera streams and XLink
FPS_TARGET = 30

# Mono resolution for stereo depth ("400p" is faster; "720p" is finer)
MONO_RES = "400p"  # "400p" or "720p"

# Align depth output to RGB FOV/pixels. Requires L-R check ON (device rule).
ALIGN_DEPTH_TO_RGB = True

# When not aligning to RGB, you may still turn on L-R check (used only then)
LEFT_RIGHT_CHECK = False

# Subpixel disparity (smoother depth; costs FPS)
SUBPIXEL = False

# Median filter for depth cleanup: "OFF" | "K3" | "K5" | "K7"
MEDIAN_FILTER = "OFF"

# Host queue sizing for XLink (small to avoid backpressure/hangs)
XLINK_MAX_QUEUE = 3

# Depth legend settings (meters)
AUTO_RANGE_DEFAULT = True
LEGEND_FIXED_MIN_M = 0.4
LEGEND_FIXED_MAX_M = 5.0
LEGEND_W = 70
INVALID_TO_BLACK = True

# ZMQ publisher
ZMQ_BIND_ENDPOINT = "tcp://*:5556"  # same port the viewer subscribes to
ZMQ_SND_HWM = 1                     # drop frames if the viewer is slow
ZMQ_TOPIC = b"pc"                   # multipart topic prefix for the viewer

# JPEG compression for the colormap sent to the viewer
JPEG_QUALITY = 80

# Window titles
WINDOW_RGB_TITLE = "RGB"
WINDOW_DEPTH_TITLE = "Depth"

# ────────────────────────────────────────────────────────────────────────────────
# ZMQ publisher (non-blocking)

zmqCtx = zmq.Context.instance()
pubSock = zmqCtx.socket(zmq.PUB)
pubSock.setsockopt(zmq.SNDHWM, ZMQ_SND_HWM)  # prefer newest frame
pubSock.bind(ZMQ_BIND_ENDPOINT)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers: string → DepthAI enums

def mono_res_from_text(text):
    """Map '400p'/'720p' to DepthAI MonoCamera resolution enum."""
    s = text.upper()
    if "720" in s:
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    return dai.MonoCameraProperties.SensorResolution.THE_400_P

def median_from_text(text):
    """Map 'OFF'/'K3'/'K5'/'K7' to StereoDepth median filter enum."""
    s = text.upper()
    if s == "K3":
        return dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
    if s == "K5":
        return dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
    if s == "K7":
        return dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

# ────────────────────────────────────────────────────────────────────────────────
# Pipeline creation

def make_pipeline():
    """Create a DepthAI pipeline with color preview and stereo depth."""
    pipeline = dai.Pipeline()

    # ----- Color camera (preview supports keep-aspect) -----
    camNode = pipeline.create(dai.node.ColorCamera)
    camNode.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camNode.setFps(FPS_TARGET)
    camNode.setInterleaved(False)
    camNode.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camNode.setPreviewSize(RGB_PREVIEW_W, RGB_PREVIEW_H)
    # Keep full FOV; preview letterboxes if aspect ratio differs
    camNode.setPreviewKeepAspectRatio(True)

    # ----- Mono cameras for stereo -----
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoLeft.setResolution(mono_res_from_text(MONO_RES))
    monoRight.setResolution(mono_res_from_text(MONO_RES))
    monoLeft.setFps(FPS_TARGET)
    monoRight.setFps(FPS_TARGET)

    # ----- Stereo depth -----
    stereoNode = pipeline.create(dai.node.StereoDepth)
    stereoNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereoNode.initialConfig.setConfidenceThreshold(200)
    stereoNode.initialConfig.setMedianFilter(median_from_text(MEDIAN_FILTER))
    if ALIGN_DEPTH_TO_RGB:
        # Device requirement: alignment to RGB needs L-R check on
        stereoNode.setLeftRightCheck(True)
        stereoNode.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB FOV
    else:
        stereoNode.setLeftRightCheck(LEFT_RIGHT_CHECK)
    stereoNode.setSubpixel(SUBPIXEL)

    monoLeft.out.link(stereoNode.left)
    monoRight.out.link(stereoNode.right)

    # ----- Host outputs -----
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.setFpsLimit(FPS_TARGET)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    xoutDepth.setFpsLimit(FPS_TARGET)

    camNode.preview.link(xoutRgb.input)
    stereoNode.depth.link(xoutDepth.input)

    return pipeline

# ────────────────────────────────────────────────────────────────────────────────
# Depth colorization (with legend) for the on-screen "Depth" window

def robust_percentile_range_m(depthMm):
    """
    Compute a visually stable near/far range in meters using robust percentiles.
    Ignores invalid (zero) depths; clamps to [0.2, 10] meters for readability.
    """
    d = depthMm.astype(np.float32)
    valid = d > 0
    if not np.any(valid):
        return LEGEND_FIXED_MIN_M, LEGEND_FIXED_MAX_M
    vals = d[valid]
    lo = np.percentile(vals, 5)
    hi = np.percentile(vals, 95)
    loM = max(0.2, lo / 1000.0)
    hiM = max(loM + 0.1, min(10.0, hi / 1000.0))
    return loM, hiM

def colorize_depth_with_legend(depthMm, outW, outH, loM, hiM):
    """
    Colorize depth (mm) into a JET map sized to (outW,outH) plus a side legend.
    Returns: (depth_with_legend, plain_colormap_without_legend)
    """
    d = cv2.resize(depthMm, (outW, outH), interpolation=cv2.INTER_NEAREST)
    invalid = (d == 0)

    loMm = int(max(0.0, loM) * 1000)
    hiMm = int(max(loMm + 1, hiM * 1000))
    dClamped = np.clip(d, loMm, hiMm).astype(np.float32)

    # 0=near,1=far; invert so near=red, far=blue
    u8 = ((1.0 - (dClamped - loMm) / float(hiMm - loMm)) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    if INVALID_TO_BLACK:
        color[invalid] = (0, 0, 0)

    # Build a vertical legend (top=near/red, bottom=far/blue)
    grad = np.linspace(255, 0, outH, dtype=np.uint8).reshape(outH, 1)
    legend = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    font = cv2.FONT_HERSHEY_SIMPLEX
    nTicks = max(2, min(10, int(np.ceil(hiM - loM))))
    for i in range(nTicks + 1):
        valM = loM + (hiM - loM) * (i / nTicks)
        y = int(outH * (i / nTicks))
        y = np.clip(y, 0, outH - 1)
        cv2.line(legend, (0, y), (10, y), (255, 255, 255), 1)
        cv2.putText(legend, f"{valM:.1f} m",
                    (12, min(outH - 6, y + 14)),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "NEAR", (5, 18), font, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, "FAR",  (5, outH - 8), font, 0.5, (255, 255, 255), 1)
    legend = cv2.resize(legend, (LEGEND_W, outH), interpolation=cv2.INTER_NEAREST)

    return np.hstack([color, legend]), color

# ────────────────────────────────────────────────────────────────────────────────
# Queue helper: drain to newest packet

def get_latest(queue):
    """Drain a non-blocking DepthAI output queue and return the newest packet."""
    try:
        pkt = queue.tryGet()
    except RuntimeError:
        return None
    lastPkt = None
    while pkt is not None:
        lastPkt = pkt
        try:
            pkt = queue.tryGet()
        except RuntimeError:
            break
    return lastPkt

# ────────────────────────────────────────────────────────────────────────────────
# ZMQ publisher for viewer

def publish_oak_frame(depthMm, cmapRgb, fx, fy, cx, cy):
    """
    Publish raw uint16 depth (millimeters) + JPEG colormap at the SAME (w,h),
    along with intrinsics that match that size.
    """
    try:
        h, w = depthMm.shape
        ok, jpg = cv2.imencode(".jpg", cmapRgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        header = {
            "mode": "mm",      # viewer expects metric depth in mm
            "w": int(w), "h": int(h),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "metric": True
        }
        pubSock.send_multipart([ZMQ_TOPIC,
                                json.dumps(header).encode("utf-8"),
                                depthMm.tobytes(),
                                jpg.tobytes()])
    except Exception:
        # Never allow publisher to crash the app
        pass

# ────────────────────────────────────────────────────────────────────────────────
# Main loop

def main():
    """Initialize device, render RGB/Depth, and publish metric depth for the viewer."""
    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb",   maxSize=XLINK_MAX_QUEUE, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=XLINK_MAX_QUEUE, blocking=False)

        # Read per-device calibration (intrinsics/extrinsics)
        calib = device.readCalibration()  # authoritative EEPROM data

        cv2.namedWindow(WINDOW_RGB_TITLE,   cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_DEPTH_TITLE, cv2.WINDOW_NORMAL)

        autoRange = AUTO_RANGE_DEFAULT
        fpsSmooth, tPrev = None, time.time()
        print("Hotkeys:  a=auto-range   f=fullscreen   q=quit")

        latestRgb = None
        latestMm = None

        # Cache intrinsics for the *published* depth frames (size-specific)
        fx = fy = cx = cy = None
        depthSizeUsed = None  # (w, h) for which we computed intrinsics

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                autoRange = not autoRange
            elif key == ord('f'):
                for win in (WINDOW_RGB_TITLE, WINDOW_DEPTH_TITLE):
                    state = int(cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN))
                    newState = cv2.WINDOW_NORMAL if state == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, newState)

            pktRgb = get_latest(qRgb)
            if pktRgb is not None:
                latestRgb = pktRgb.getCvFrame()

            pktDepth = get_latest(qDepth)
            if pktDepth is not None:
                latestMm = pktDepth.getFrame()  # uint16 depth (mm)

            if latestRgb is None or latestMm is None:
                continue

            # --- For publishing: compute intrinsics for the *exact* depth map size ---
            hD, wD = latestMm.shape[:2]
            if depthSizeUsed != (wD, hD):
                if ALIGN_DEPTH_TO_RGB:
                    # Depth aligned to RGB FOV; ask color intrinsics at THIS size
                    K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, wD, hD))
                else:
                    # Not aligned: depth is in rectified-left geometry; use left intrinsics at THIS size
                    K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, wD, hD))
                fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
                depthSizeUsed = (wD, hD)

            # --- Display: RGB + legend depth, sized to RGB preview for UI only ---
            if autoRange:
                loM, hiM = robust_percentile_range_m(latestMm)
            else:
                loM, hiM = LEGEND_FIXED_MIN_M, LEGEND_FIXED_MAX_M

            depthVis, _ = colorize_depth_with_legend(
                latestMm, latestRgb.shape[1], latestRgb.shape[0], loM, hiM
            )
            rgbShow, depthShow = latestRgb, depthVis

            if WINDOW_SCALE != 1.0:
                rgbShow   = cv2.resize(rgbShow, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                                       interpolation=cv2.INTER_NEAREST)
                depthShow = cv2.resize(depthShow, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                                       interpolation=cv2.INTER_NEAREST)

            now = time.time()
            inst = 1.0 / (now - tPrev) if now > tPrev else 0.0
            tPrev = now
            fpsSmooth = inst if fpsSmooth is None else (0.8 * fpsSmooth + 0.2 * inst)

            cv2.putText(rgbShow, f"FPS {fpsSmooth:.1f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(depthShow, f"{'AUTO' if autoRange else 'FIXED'} [{loM:.2f}-{hiM:.2f} m]",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW_RGB_TITLE,   rgbShow)
            cv2.imshow(WINDOW_DEPTH_TITLE, depthShow)

            # --- Publish depth AT NATIVE/ALIGNED SIZE + plain JET colormap of SAME SIZE ---
            loMPub, hiMPub = robust_percentile_range_m(latestMm)
            loMmPub, hiMmPub = int(loMPub * 1000), int(hiMPub * 1000)
            dClip = np.clip(latestMm, loMmPub, hiMmPub).astype(np.float32)
            u8 = ((1.0 - (dClip - loMmPub) / float(max(1, hiMmPub - loMmPub))) * 255.0).astype(np.uint8)
            cmapForView = cv2.applyColorMap(u8, cv2.COLORMAP_JET)

            publish_oak_frame(latestMm, cmapForView, fx, fy, cx, cy)

        cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
