# oakd_aruco_ar_depth_shapes.py
# OAK-D Lite: RGB + aligned metric depth. ArUco-based AR with two shapes
# (cuboid & cylinder) anchored in the marker coordinate frame.
# OpenCV 4.6 ArUco legacy API (DetectorParameters_create + detectMarkers).
# Keys: 'r' re-randomize shapes, 't' template select/save, 'q' quit.

import os
import time
import math
import random
import cv2
import numpy as np
import depthai as dai

# ======================= USER CONSTANTS =======================

# --- Marker dictionary and *printed* marker size (mm) ---
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_MM = 50.0  # make sure the printed square is exactly 50.0 mm

# --- RGB output for rendering ---
RGB_SIZE = (1280, 720)  # width, height

# --- Stereo depth params (aligned depth in mm) ---
USE_SUBPIXEL = True
CONF_THRESH = 200
MEDIAN_KERNEL = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# --- Which marker to anchor to (None = pick the largest visible) ---
TARGET_MARKER_ID = None

# --- Shape placement control ---
# Coordinates are in the *marker frame* (mm). Reference is the MARKER CENTER.
# z=0 sits on the marker plane; positive height is drawn toward the camera as -z internally.

RANDOMIZE_ON_START = True      # randomize positions at startup
RANDOM_RANGE_XY_MM = int(MARKER_SIZE_MM * 0.45)  # +/- range for x,y around center

# Shape list (two shapes by default). You can edit or add more.
SHAPES = [
    dict(  # CUBOID
        type="cuboid",
        # size: (width_x, depth_y, height) in mm
        size_mm=(30.0, 24.0, 18.0),
        # position of base center relative to marker center (x,y,z) in mm
        pos_mm=(+10.0, -10.0, 0.0),
        # orientation about local XYZ (degrees)
        rot_deg=(0.0, 0.0, 25.0),
        color=(0, 255, 0),
        thickness=2,
    ),
    dict(  # CYLINDER
        type="cylinder",
        # cylinder params (radius, height) in mm, and number of segments for projection
        radius_mm=12.0,
        height_mm=22.0,
        segments=36,
        pos_mm=(-12.0, +12.0, 0.0),
        rot_deg=(0.0, 0.0, 0.0),
        color=(0, 200, 255),
        thickness=2,
    ),
]

# Template save folder
TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# ======================= PIPELINE BUILD =======================

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

    # Color camera (CAM_A)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(RGB_SIZE[0], RGB_SIZE[1])
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.preview.link(xout_rgb.input)

    # Mono cams for stereo
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # 640x400
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # Stereo depth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(MEDIAN_KERNEL)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(USE_SUBPIXEL)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.initialConfig.setConfidenceThreshold(CONF_THRESH)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)

    return pipeline

def get_rgb_calibration(device, width, height):
    calib = device.readCalibration()
    K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height), dtype=np.float32)
    dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32).ravel()
    if dist.size > 8:
        dist = dist[:8]
    return K, dist

# ======================= ARUCO (OpenCV 4.6) =======================

def make_aruco_detector_legacy():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters_create()
    return aruco_dict, params

def detect_markers_legacy(gray, aruco_dict, params):
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected

# ======================= GEOMETRY HELPERS =======================

def euler_deg_to_R(rx, ry, rz, order="xyz"):
    """Euler angles in degrees -> 3x3 rotation matrix. order in {'xyz','zyx',...}."""
    ax, ay, az = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0, math.cos(ax), -math.sin(ax)],[0, math.sin(ax), math.cos(ax)]], dtype=np.float32)
    Ry = np.array([[ math.cos(ay),0, math.sin(ay)],[0,1,0],[-math.sin(ay),0, math.cos(ay)]], dtype=np.float32)
    Rz = np.array([[ math.cos(az),-math.sin(az),0],[ math.sin(az), math.cos(az),0],[0,0,1]], dtype=np.float32)
    # Compose
    R = np.eye(3, dtype=np.float32)
    for c in order:
        R = R @ {'x':Rx,'y':Ry,'z':Rz}[c]
    return R

def project_points(obj_pts_mkr, rvec, tvec, K, dist):
    """obj_pts_mkr: (N,3) in marker frame -> project to image"""
    img_pts, _ = cv2.projectPoints(obj_pts_mkr.astype(np.float32), rvec, tvec, K, dist)
    return img_pts.reshape(-1,2).astype(int)

def marker_center_offset():
    """Offset (in mm) from marker origin (top-left) to its center."""
    c = MARKER_SIZE_MM * 0.5
    return np.array([c, c, 0.0], dtype=np.float32)

def randomize_shapes(shapes):
    for s in shapes:
        dx = random.uniform(-RANDOM_RANGE_XY_MM, RANDOM_RANGE_XY_MM)
        dy = random.uniform(-RANDOM_RANGE_XY_MM, RANDOM_RANGE_XY_MM)
        # keep on plane (z=0 base), you can tweak if you want floating shapes
        s['pos_mm'] = (dx, dy, 0.0)

# ======================= SHAPE MODELS =======================

def build_cuboid_points(size_mm, pos_mm, rot_deg):
    """Return a set of 3D edges (pairs of points) of a cuboid in marker frame.
       Cuboid is defined by width_x, depth_y, height; base center at pos_mm on z=0 plane."""
    wx, dy, h = size_mm
    x2, y2 = wx/2.0, dy/2.0
    # Base center in marker coords
    base_center = marker_center_offset() + np.array([pos_mm[0], pos_mm[1], 0.0], dtype=np.float32)
    # Local (centered) box vertices, base at z=0, top at z=-h
    # We rotate around the base center.
    base = np.array([[-x2,-y2, 0], [ x2,-y2, 0], [ x2, y2, 0], [-x2, y2, 0]], dtype=np.float32)
    top  = np.array([[-x2,-y2,-h], [ x2,-y2,-h], [ x2, y2,-h], [-x2, y2,-h]], dtype=np.float32)

    R = euler_deg_to_R(*rot_deg, order="xyz")
    base_rot = (base @ R.T) + base_center
    top_rot  = (top  @ R.T) + base_center

    # Return edges as list of 3D line segments (pairs)
    edges = []
    for i in range(4):
        edges.append((base_rot[i], base_rot[(i+1)%4]))  # base square
        edges.append((top_rot[i],  top_rot[(i+1)%4]))   # top square
        edges.append((base_rot[i], top_rot[i]))         # pillars
    return edges

def build_cylinder_points(radius_mm, height_mm, pos_mm, rot_deg, segments=36):
    """Return polylines approximating a cylinder (top/bottom circles + verticals)."""
    base_center = marker_center_offset() + np.array([pos_mm[0], pos_mm[1], 0.0], dtype=np.float32)
    R = euler_deg_to_R(*rot_deg, order="xyz")

    # Circle in local XY; base at z=0, top at z=-height
    angles = np.linspace(0, 2*np.pi, num=segments, endpoint=False)
    circ = np.stack([radius_mm*np.cos(angles), radius_mm*np.sin(angles), np.zeros_like(angles)], axis=1).astype(np.float32)
    circ_top = circ.copy(); circ_top[:,2] = -height_mm

    # rotate & translate
    circ_b = (circ @ R.T) + base_center
    circ_t = (circ_top @ R.T) + base_center

    # Build edges (circle polylines + a few vertical lines)
    edges = []
    for i in range(segments):
        edges.append((circ_b[i], circ_b[(i+1)%segments]))
        edges.append((circ_t[i], circ_t[(i+1)%segments]))
    # verticals at 0, 90, 180, 270 deg for visual clarity
    for i in [0, segments//4, segments//2, 3*segments//4]:
        edges.append((circ_b[i], circ_t[i]))
    return edges

# ======================= RENDER HELPERS =======================

def draw_edges(img, edges_3d, rvec, tvec, K, dist, color=(0,255,0), thickness=2):
    if not edges_3d:
        return
    # Flatten all points once for projection
    pts = np.vstack([np.stack([a,b], axis=0) for (a,b) in edges_3d])  # (2M,3)
    img_pts = project_points(pts, rvec, tvec, K, dist)                # (2M,2)
    # Draw segment by segment
    for i in range(0, len(img_pts), 2):
        p1 = tuple(img_pts[i])
        p2 = tuple(img_pts[i+1])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def colorize_depth_mm(depth_frame_mm, max_display_mm=4000):
    depth = np.clip(depth_frame_mm, 0, max_display_mm).astype(np.uint16)
    norm = (255.0 * (1.0 - (depth.astype(np.float32) / (max_display_mm + 1e-6)))).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

def pick_anchor_index(ids, corners):
    """Choose which marker to anchor to. If TARGET_MARKER_ID is None, pick the largest by perimeter."""
    if ids is None or len(ids) == 0:
        return None
    if TARGET_MARKER_ID is not None:
        for i, mid in enumerate(ids.flatten()):
            if int(mid) == int(TARGET_MARKER_ID):
                return i
        return None
    # largest perimeter
    perims = [np.linalg.norm(c[0][1]-c[0][0]) + np.linalg.norm(c[0][2]-c[0][1]) +
              np.linalg.norm(c[0][3]-c[0][2]) + np.linalg.norm(c[0][0]-c[0][3]) for c in corners]
    return int(np.argmax(perims))

# ======================= TEMPLATE UTILS =======================

def warp_marker_to_square(frame_bgr, corners, out_size=256):
    src = np.array(corners, dtype=np.float32).reshape(4, 2)  # tl,tr,br,bl
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, H, (out_size, out_size))

def select_template_label(frame_bgr, ids, corners, current_id=None):
    if ids is None or len(ids) == 0: return current_id
    ids_list = [int(x) for x in ids.flatten()]
    idx = ids_list.index(current_id) if (current_id in ids_list) else 0
    while True:
        disp = frame_bgr.copy()
        chosen = ids_list[idx]
        for i, mid in enumerate(ids_list):
            c = corners[i][0].astype(int)
            color = (0,255,0) if mid == chosen else (255,0,0)
            cv2.polylines(disp, [c], True, color, 2)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(disp, f"ID {mid}", (cx-25, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(disp, "Template: LEFT/RIGHT (A/D), 's' save, ENTER accept, 'q' cancel",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Template Select", disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (81, ord('a')):
            idx = (idx - 1) % len(ids_list)
        elif key in (83, ord('d')):
            idx = (idx + 1) % len(ids_list)
        elif key == ord('s'):
            i = ids_list.index(chosen)
            roi = warp_marker_to_square(frame_bgr, corners[i], 256)
            out_path = os.path.join(TEMPLATE_DIR, f"marker_{chosen}.png")
            cv2.imwrite(out_path, roi); print(f"[SAVED] {out_path}")
        elif key in (13, 10):
            cv2.destroyWindow("Template Select"); return chosen
        elif key == ord('q'):
            cv2.destroyWindow("Template Select"); return current_id

# ======================= MAIN =======================

def main():
    # optional randomization at start
    if RANDOMIZE_ON_START:
        randomize_shapes(SHAPES)

    pipeline = build_pipeline()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Intrinsics/distortion for this preview size
        K, dist = get_rgb_calibration(device, RGB_SIZE[0], RGB_SIZE[1])

        # ArUco (legacy API)
        aruco_dict, aruco_params = make_aruco_detector_legacy()

        template_id = None
        fps_smooth, t_prev = None, time.time()

        print("[INFO] Running. Keys: r=randomize shapes, t=template picker, q=quit.")
        while True:
            in_rgb = q_rgb.get()
            frame_bgr = in_rgb.getCvFrame()
            h, w = frame_bgr.shape[:2]

            # Depth (aligned, mm)
            in_depth = q_depth.tryGet()
            if in_depth is not None:
                depth_mm = in_depth.getFrame()
                depth_color = colorize_depth_mm(depth_mm, max_display_mm=5000)
                if depth_color.shape[:2] != (h, w):
                    depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                depth_color = np.zeros_like(frame_bgr)

            # Detect markers
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers_legacy(gray, aruco_dict, aruco_params)

            anchor_idx = pick_anchor_index(ids, corners)
            if (ids is not None) and (anchor_idx is not None):
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

                # Pose (marker->camera), in mm
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_MM, K, dist)
                rvec, tvec = rvecs[anchor_idx], tvecs[anchor_idx]

                # Draw axes on anchor marker
                cv2.drawFrameAxes(frame_bgr, K, dist, rvec, tvec, MARKER_SIZE_MM * 0.5)

                # Render shapes
                for s in SHAPES:
                    if s["type"] == "cuboid":
                        edges = build_cuboid_points(s["size_mm"], s["pos_mm"], s["rot_deg"])
                        draw_edges(frame_bgr, edges, rvec, tvec, K, dist, color=s.get("color",(0,255,0)), thickness=s.get("thickness",2))
                    elif s["type"] == "cylinder":
                        edges = build_cylinder_points(s["radius_mm"], s["height_mm"], s["pos_mm"], s["rot_deg"], segments=s.get("segments",36))
                        draw_edges(frame_bgr, edges, rvec, tvec, K, dist, color=s.get("color",(0,200,255)), thickness=s.get("thickness",2))

                # Show anchor distance
                m_id = int(ids[anchor_idx])
                dist_cm = float(np.linalg.norm(tvec)) / 10.0
                c = corners[anchor_idx][0].astype(int)
                cx, cy = c.mean(axis=0).astype(int)
                cv2.putText(frame_bgr, f"Anchor ID {m_id}  {dist_cm:.1f} cm",
                            (cx-60, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # FPS
            now = time.time()
            inst = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            fps_smooth = inst if fps_smooth is None else (0.9*fps_smooth + 0.1*inst)

            stacked = np.hstack([frame_bgr, depth_color])
            cv2.putText(stacked, f"OAK-D Lite AR (left) | Depth (right)   FPS: {fps_smooth:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(stacked, "Keys: r=randomize shapes, t=template picker, q=quit",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,0), 2)

            cv2.imshow("OAK-D Lite: Marker-anchored AR (shapes) + Depth", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                if ids is not None and len(ids) > 0:
                    template_id = select_template_label(frame_bgr, ids, corners, current_id=template_id)
                    if template_id is not None:
                        print(f"[INFO] Template marker set to ID {template_id}")
                else:
                    print("[INFO] No markers visible.")
            elif key == ord('r'):
                randomize_shapes(SHAPES)
                print("[INFO] Shapes re-randomized around marker center.")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
