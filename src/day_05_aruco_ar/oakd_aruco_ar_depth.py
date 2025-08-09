# oakd_aruco_ar_depth.py
# OAK-D Lite: RGB + metric stereo depth aligned to RGB, ArUco detection (OpenCV 4.6 legacy API),
# pose (rvec/tvec), AR overlay (axes + cube), side-by-side with colorized depth.
# 't' => template selector to choose/save a marker patch; 'q' => quit.

import os
import time
import math
import cv2
import numpy as np
import depthai as dai

# ---------------------- User Settings ----------------------
# ArUco dictionary and printed marker size (millimetres)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_MM = 40.0  # print a 50.0 mm square (side length)

# Color preview size; keep to a common 16:9 to match align-to-RGB
RGB_SIZE = (1280, 720)

# Stereo depth tuning (metric depth in mm)
USE_SUBPIXEL = True
CONF_THRESH = 200  # 0..255; higher = stricter
MEDIAN_KERNEL = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Template saving
TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)
# -----------------------------------------------------------


def build_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    mono_l = pipeline.createMonoCamera()
    mono_r = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    xout_rgb = pipeline.createXLinkOut()
    xout_depth = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    # --- Color camera ---
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # updated naming
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(RGB_SIZE[0], RGB_SIZE[1])
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.preview.link(xout_rgb.input)

    # --- Mono cams for stereo ---
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # 640x400
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # --- Stereo depth ---
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # modern default
    stereo.initialConfig.setMedianFilter(MEDIAN_KERNEL)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(USE_SUBPIXEL)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to color
    stereo.initialConfig.setConfidenceThreshold(CONF_THRESH)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)

    return pipeline


def get_rgb_calibration(device, width, height):
    """
    Read RGB intrinsics/distortion for the given output size from device EEPROM.
    """
    calib = device.readCalibration()
    K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height), dtype=np.float32)
    dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32).ravel()
    if dist.size > 8:
        dist = dist[:8]
    return K, dist


# ----------- ArUco (OpenCV 4.6 legacy-compatible) -----------
def make_aruco_detector_legacy():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    # 4.6 uses DetectorParameters_create()
    params = cv2.aruco.DetectorParameters_create()
    return aruco_dict, params

def detect_markers_legacy(gray, aruco_dict, params):
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected
# ------------------------------------------------------------


def draw_axes(img, rvec, tvec, K, dist, length=MARKER_SIZE_MM * 0.5):
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)

def draw_cube(img, rvec, tvec, K, dist, size=MARKER_SIZE_MM):
    s = size
    pts_3d = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
    ], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    # base
    for i in range(4):
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[(i + 1) % 4]), (0, 255, 0), 2)
    # pillars
    for i in range(4):
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[i + 4]), (0, 255, 0), 2)
    # top
    for i in range(4, 8):
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[4 + ((i + 1 - 4) % 4)]), (0, 255, 0), 2)

def warp_marker_to_square(frame_bgr, corners, out_size=256):
    src = np.array(corners, dtype=np.float32).reshape(4, 2)  # tl,tr,br,bl (OpenCV returns this order)
    dst = np.array([[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, H, (out_size, out_size))

def select_template_label(frame_bgr, ids, corners, current_id=None):
    """
    UI to pick one ArUco ID as the 'template' and optionally save its rectified patch.
    Controls:
      LEFT/RIGHT or A/D: cycle IDs
      's': save rectified 256x256 PNG (templates/marker_<ID>.png)
      ENTER: accept and return chosen ID
      'q': cancel (keep current)
    """
    if ids is None or len(ids) == 0:
        print("[INFO] No markers visible to select.")
        return current_id

    ids_list = [int(x) for x in ids.flatten()]
    idx = 0
    if current_id in ids_list:
        idx = ids_list.index(current_id)

    while True:
        disp = frame_bgr.copy()
        chosen = ids_list[idx]
        for i, mid in enumerate(ids_list):
            c = corners[i][0].astype(int)
            color = (0, 255, 0) if mid == chosen else (255, 0, 0)
            cv2.polylines(disp, [c], True, color, 2)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(disp, f"ID {mid}", (cx - 25, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(disp, "Template: LEFT/RIGHT (A/D), 's' save, ENTER accept, 'q' cancel",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Template Select", disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (81, ord('a')):  # left
            idx = (idx - 1) % len(ids_list)
        elif key in (83, ord('d')):  # right
            idx = (idx + 1) % len(ids_list)
        elif key == ord('s'):
            i = ids_list.index(chosen)
            roi = warp_marker_to_square(frame_bgr, corners[i], 256)
            out_path = os.path.join(TEMPLATE_DIR, f"marker_{chosen}.png")
            cv2.imwrite(out_path, roi)
            print(f"[SAVED] {out_path}")
        elif key in (13, 10):  # Enter
            cv2.destroyWindow("Template Select")
            return chosen
        elif key == ord('q'):
            cv2.destroyWindow("Template Select")
            return current_id

def colorize_depth_mm(depth_frame_mm, max_display_mm=4000):
    """
    Convert a 16-bit depth frame (mm) to a color map for visualization.
    Clips to [0, max_display_mm]. Near = bright (inverted).
    """
    depth = np.clip(depth_frame_mm, 0, max_display_mm).astype(np.uint16)
    norm = (255.0 * (1.0 - (depth.astype(np.float32) / (max_display_mm + 1e-6)))).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

def main():
    pipeline = build_pipeline()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Intrinsics/distortion for the chosen RGB output size
        K, dist = get_rgb_calibration(device, RGB_SIZE[0], RGB_SIZE[1])

        # ArUco (OpenCV 4.6 legacy)
        aruco_dict, aruco_params = make_aruco_detector_legacy()

        template_id = None
        fps_smooth = None
        t_prev = time.time()

        print("[INFO] Running. Press 't' to choose/save a template marker; 'q' to quit.")

        while True:
            in_rgb = q_rgb.get()
            frame_bgr = in_rgb.getCvFrame()                    # BGR, (w,h) = RGB_SIZE
            h, w = frame_bgr.shape[:2]

            # Depth aligned to RGB (uint16 mm)
            in_depth = q_depth.tryGet()
            if in_depth is not None:
                depth_mm = in_depth.getFrame()
                depth_color = colorize_depth_mm(depth_mm, max_display_mm=5000)
                if depth_color.shape[:2] != (h, w):
                    depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                depth_color = np.zeros_like(frame_bgr)

            # ArUco detection (legacy API)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers_legacy(gray, aruco_dict, aruco_params)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
                # Pose in the same units as MARKER_SIZE_MM (mm)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIZE_MM, K, dist
                )
                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    draw_axes(frame_bgr, rvec, tvec, K, dist, length=MARKER_SIZE_MM * 0.5)
                    draw_cube(frame_bgr, rvec, tvec, K, dist, size=MARKER_SIZE_MM)

                    # Show distance (||tvec||) in cm near marker centroid
                    dist_cm = float(np.linalg.norm(tvec)) / 10.0
                    c = corners[i][0].astype(int)
                    cx, cy = c.mean(axis=0).astype(int)
                    cv2.putText(frame_bgr, f"{dist_cm:.1f} cm",
                                (cx - 30, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FPS
            now = time.time()
            inst = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            fps_smooth = inst if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst)

            # Compose view
            stacked = np.hstack([frame_bgr, depth_color])
            cv2.putText(stacked, f"OAK-D Lite AR (left) | Depth (right)   FPS: {fps_smooth:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if ids is not None and len(ids) > 0:
                cv2.putText(stacked, "Press 't' to choose/save a template marker",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)

            cv2.imshow("OAK-D Lite: ArUco AR + Metric Depth", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                if ids is None or len(ids) == 0:
                    print("[INFO] No markers visible.")
                else:
                    template_id = select_template_label(frame_bgr, ids, corners, current_id=template_id)
                    if template_id is not None:
                        print(f"[INFO] Template marker set to ID {template_id}")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
