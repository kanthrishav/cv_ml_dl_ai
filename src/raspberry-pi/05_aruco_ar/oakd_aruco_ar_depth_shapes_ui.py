# oakd_aruco_ar_depth_shapes_ui.py
# OAK-D Lite: RGB + aligned metric depth, ArUco-anchored AR with 3 shapes
# (cuboid, cylinder, sphere) and on-window low-transparency UI sliders to
# control X/Y/Z and Scale of the *active* shape, plus a Fill toggle.
#
# OpenCV 4.6 ArUco legacy API (DetectorParameters_create + detectMarkers).
# Keys: 't' template picker, 'r' re-randomize positions, 'q' quit.

import os
import time
import math
import random
import cv2
import numpy as np
import depthai as dai

# ======================= USER CONSTANTS =======================

# ArUco dictionary and printed marker size (mm)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_MM = 50.0  # print this exact size (side length)

# RGB output resolution for rendering
RGB_SIZE = (1280, 720)  # (width, height)

# Stereo depth (aligned to RGB)
USE_SUBPIXEL = True
CONF_THRESH = 200
MEDIAN_KERNEL = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Anchor: which marker to use (None => pick largest visible)
TARGET_MARKER_ID = None

# Initial placement tuning
RANDOMIZE_ON_START = True
RANDOM_RANGE_XY_MM = int(MARKER_SIZE_MM * 0.45)  # +/- around marker center
Z_RANGE_MM = 60  # slider range for z (base height) in mm
SCALE_MIN, SCALE_MAX = 0.35, 2.5

# Template save folder
TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

WINDOW_NAME = "OAK-D Lite: Marker-anchored AR (UI) + Depth"

# ======================= SHAPES MODEL =======================

def default_shapes():
    return [
        dict(  # CUBOID
            name="CUBOID",
            type="cuboid",
            size_mm=(30.0, 24.0, 18.0),   # (width_x, depth_y, height)
            base_size_mm=(30.0, 24.0, 18.0),
            pos_mm=(+10.0, -10.0, 0.0),   # base center relative to marker center
            rot_deg=(0.0, 0.0, 25.0),
            scale=1.0,
            filled=False,
            color=(0, 255, 0),
            thickness=2,
        ),
        dict(  # CYLINDER
            name="CYLINDER",
            type="cylinder",
            radius_mm=12.0,
            height_mm=22.0,
            base_radius_mm=12.0,
            base_height_mm=22.0,
            segments=36,
            pos_mm=(-12.0, +12.0, 0.0),
            rot_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            filled=False,
            color=(0, 200, 255),
            thickness=2,
        ),
        dict(  # SPHERE
            name="SPHERE",
            type="sphere",
            radius_mm=12.0,
            base_radius_mm=12.0,
            pos_mm=(0.0, 0.0, 0.0),
            rot_deg=(0.0, 0.0, 0.0),  # not used for sphere but kept for consistency
            scale=1.0,
            filled=False,
            color=(255, 180, 0),
            thickness=2,
        ),
    ]

SHAPES = default_shapes()

def randomize_positions(shapes):
    for s in shapes:
        dx = random.uniform(-RANDOM_RANGE_XY_MM, RANDOM_RANGE_XY_MM)
        dy = random.uniform(-RANDOM_RANGE_XY_MM, RANDOM_RANGE_XY_MM)
        s["pos_mm"] = (dx, dy, 0.0)

# ======================= DEPTHAI PIPELINE =======================

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
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
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

# ======================= ARUCO (OpenCV 4.6 legacy) =======================

def make_aruco_detector_legacy():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters_create()
    return aruco_dict, params

def detect_markers_legacy(gray, aruco_dict, params):
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected

# ======================= GEOMETRY HELPERS =======================

def euler_deg_to_R(rx, ry, rz, order="xyz"):
    ax, ay, az = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0, math.cos(ax), -math.sin(ax)],[0, math.sin(ax), math.cos(ax)]], dtype=np.float32)
    Ry = np.array([[ math.cos(ay),0, math.sin(ay)],[0,1,0],[-math.sin(ay),0, math.cos(ay)]], dtype=np.float32)
    Rz = np.array([[ math.cos(az),-math.sin(az),0],[ math.sin(az), math.cos(az),0],[0,0,1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    for c in order:
        R = R @ {'x':Rx,'y':Ry,'z':Rz}[c]
    return R

def project_points(obj_pts_mkr, rvec, tvec, K, dist):
    img_pts, _ = cv2.projectPoints(obj_pts_mkr.astype(np.float32), rvec, tvec, K, dist)
    return img_pts.reshape(-1,2).astype(int)

def marker_center_offset():
    c = MARKER_SIZE_MM * 0.5
    return np.array([c, c, 0.0], dtype=np.float32)

def pick_anchor_index(ids, corners):
    if ids is None or len(ids) == 0:
        return None
    if TARGET_MARKER_ID is not None:
        for i, mid in enumerate(ids.flatten()):
            if int(mid) == int(TARGET_MARKER_ID):
                return i
        return None
    perims = [np.linalg.norm(c[0][1]-c[0][0]) + np.linalg.norm(c[0][2]-c[0][1]) +
              np.linalg.norm(c[0][3]-c[0][2]) + np.linalg.norm(c[0][0]-c[0][3]) for c in corners]
    return int(np.argmax(perims))

# ======================= SHAPE BUILDERS =======================

def build_cuboid_edges_faces(size_mm, pos_mm, rot_deg):
    """Return edges (=list of segments) and faces (=list of quads) in marker coords."""
    wx, dy, h = size_mm
    x2, y2 = wx/2.0, dy/2.0
    base_center = marker_center_offset() + np.array([pos_mm[0], pos_mm[1], pos_mm[2]], dtype=np.float32)
    base = np.array([[-x2,-y2, 0], [ x2,-y2, 0], [ x2, y2, 0], [-x2, y2, 0]], dtype=np.float32)
    top  = np.array([[-x2,-y2,-h], [ x2,-y2,-h], [ x2, y2,-h], [-x2, y2,-h]], dtype=np.float32)
    R = euler_deg_to_R(*rot_deg, order="xyz")
    base_rot = (base @ R.T) + base_center
    top_rot  = (top  @ R.T) + base_center
    # edges
    edges = []
    for i in range(4):
        edges.append((base_rot[i], base_rot[(i+1)%4]))
        edges.append((top_rot[i],  top_rot[(i+1)%4]))
        edges.append((base_rot[i], top_rot[i]))
    # faces as quads (order: base, top, and 4 sides)
    faces = [
        base_rot[[0,1,2,3]],
        top_rot [[0,1,2,3]],
        np.array([base_rot[0], base_rot[1], top_rot[1], top_rot[0]]),
        np.array([base_rot[1], base_rot[2], top_rot[2], top_rot[1]]),
        np.array([base_rot[2], base_rot[3], top_rot[3], top_rot[2]]),
        np.array([base_rot[3], base_rot[0], top_rot[0], top_rot[3]]),
    ]
    return edges, faces

def build_cylinder_edges_faces(radius_mm, height_mm, pos_mm, rot_deg, segments=36):
    base_center = marker_center_offset() + np.array([pos_mm[0], pos_mm[1], pos_mm[2]], dtype=np.float32)
    R = euler_deg_to_R(*rot_deg, order="xyz")
    ang = np.linspace(0, 2*np.pi, num=segments, endpoint=False)
    circ = np.stack([radius_mm*np.cos(ang), radius_mm*np.sin(ang), np.zeros_like(ang)], axis=1).astype(np.float32)
    circ_top = circ.copy(); circ_top[:,2] = -height_mm
    circ_b = (circ @ R.T) + base_center
    circ_t = (circ_top @ R.T) + base_center
    edges = []
    for i in range(segments):
        edges.append((circ_b[i], circ_b[(i+1)%segments]))
        edges.append((circ_t[i], circ_t[(i+1)%segments]))
    for i in [0, segments//4, segments//2, 3*segments//4]:
        edges.append((circ_b[i], circ_t[i]))
    # faces: top/bottom polygons + side as strips (quads)
    faces = [circ_b, circ_t]
    for i in range(segments):
        faces.append(np.array([circ_b[i], circ_b[(i+1)%segments], circ_t[(i+1)%segments], circ_t[i]]))
    return edges, faces

def build_sphere(center_pos_mm, radius_mm):
    """Return center (3d) and radius in marker coords; rendering handles projection."""
    center = marker_center_offset() + np.array(center_pos_mm, dtype=np.float32)
    return center, radius_mm

# ======================= RENDERING =======================

def draw_edges(img, edges_3d, rvec, tvec, K, dist, color=(0,255,0), thickness=2):
    if not edges_3d:
        return
    pts = np.vstack([np.stack([a,b], axis=0) for (a,b) in edges_3d])  # (2M,3)
    img_pts = project_points(pts, rvec, tvec, K, dist)                # (2M,2)
    for i in range(0, len(img_pts), 2):
        p1 = tuple(img_pts[i]); p2 = tuple(img_pts[i+1])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def draw_faces_filled(img, faces_3d, rvec, tvec, K, dist, face_color=(60,255,60), alpha=0.25):
    """Painter's order by average depth: draw far -> near."""
    if not faces_3d:
        return
    # project and sort
    # Need per-vertex camera-space Z to sort
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1).astype(np.float32)
    proj = []
    for poly in faces_3d:
        Pc = (R @ poly.T + t).T  # (N,3)
        zmean = float(Pc[:,2].mean())
        pts2d = project_points(poly, rvec, tvec, K, dist)
        proj.append((zmean, pts2d))
    proj.sort(key=lambda x: x[0], reverse=True)  # far (large z) first
    # draw with alpha
    overlay = img.copy()
    for _, pts in proj:
        cv2.fillConvexPoly(overlay, pts.astype(np.int32), face_color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)

def draw_sphere(img, center_marker_mm, radius_mm, rvec, tvec, K, dist, color=(255,180,0), thickness=2, filled=False, alpha=0.25):
    # Project center
    center_px = project_points(center_marker_mm.reshape(1,3), rvec, tvec, K, dist)[0]
    cx, cy = int(center_px[0]), int(center_px[1])
    # Compute camera-space Z of the center
    R, _ = cv2.Rodrigues(rvec)
    Pc = (R @ center_marker_mm.reshape(3,1) + tvec.reshape(3,1)).ravel()
    Zc = float(Pc[2])
    if Zc <= 1e-3:
        return
    fx, fy = K[0,0], K[1,1]
    rx = int(abs(fx * radius_mm / Zc))
    ry = int(abs(fy * radius_mm / Zc))
    if filled:
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0,0,0), 1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, thickness, cv2.LINE_AA)

def colorize_depth_mm(depth_frame_mm, max_display_mm=5000):
    depth = np.clip(depth_frame_mm, 0, max_display_mm).astype(np.uint16)
    norm = (255.0 * (1.0 - (depth.astype(np.float32) / (max_display_mm + 1e-6)))).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

# ======================= UI (Overlay Sliders & Widgets) =======================

class Slider:
    def __init__(self, label, x, y, w, h, vmin, vmax, v0):
        self.label = label; self.rect = (x, y, w, h)
        self.vmin = vmin; self.vmax = vmax; self.value = np.clip(v0, vmin, vmax)
        self.drag = False

    def hit(self, px, py):
        x,y,w,h = self.rect
        return (x <= px <= x+w) and (y <= py <= y+h)

    def set_from_mouse(self, px):
        x,y,w,h = self.rect
        t = np.clip((px - x) / max(1, w), 0.0, 1.0)
        self.value = self.vmin + t * (self.vmax - self.vmin)

    def draw(self, img):
        x,y,w,h = self.rect
        # bg (transparent)
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        # bar
        margin = 6
        bx, by, bw, bh = x+margin, y+int(h*0.55), w-2*margin, int(h*0.25)
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (180,180,180), 1)
        # knob position
        t = (self.value - self.vmin) / (self.vmax - self.vmin + 1e-9)
        kx = int(bx + t * bw)
        cv2.line(img, (kx, by), (kx, by+bh), (0,255,255), 3)
        # label + value
        txt = f"{self.label}: {self.value:.1f}"
        cv2.putText(img, txt, (x+8, y+int(h*0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

class CheckBox:
    def __init__(self, label, x, y, size, checked=False):
        self.label = label; self.x=x; self.y=y; self.size=size; self.checked=checked

    def hit(self, px, py):
        return (self.x <= px <= self.x+self.size) and (self.y <= py <= self.y+self.size)

    def toggle(self):
        self.checked = not self.checked

    def draw(self, img):
        overlay = img.copy()
        # translucent bg behind label area
        cv2.rectangle(overlay, (self.x-6, self.y-6), (self.x + self.size + 200, self.y + self.size + 10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        # box
        cv2.rectangle(img, (self.x, self.y), (self.x+self.size, self.y+self.size), (200,200,200), 1)
        if self.checked:
            cv2.line(img, (self.x+3,self.y+self.size//2), (self.x+self.size-3,self.y+self.size//2), (0,255,255), 2)
            cv2.line(img, (self.x+self.size//2, self.y+3), (self.x+self.size//2, self.y+self.size-3), (0,255,255), 2)
        cv2.putText(img, f"{self.label}", (self.x + self.size + 8, self.y + self.size - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

class RadioGroup:
    def __init__(self, labels, x, y, w_each=120, h=28, active_idx=0):
        self.labels = labels; self.x=x; self.y=y; self.w_each=w_each; self.h=h; self.active=active_idx

    def hit(self, px, py):
        # return index if any
        for i,_ in enumerate(self.labels):
            rx = self.x + i*self.w_each
            if (rx <= px <= rx+self.w_each) and (self.y <= py <= self.y+self.h):
                return i
        return None

    def draw(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (self.x-6, self.y-6), (self.x + self.w_each*len(self.labels) + 6, self.y + self.h + 6), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        for i,lab in enumerate(self.labels):
            rx = self.x + i*self.w_each
            color_bg = (40,140,255) if i==self.active else (120,120,120)
            cv2.rectangle(img, (rx, self.y), (rx+self.w_each, self.y+self.h), color_bg, 1)
            cv2.putText(img, lab, (rx+8, self.y+self.h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

# Global UI state (set in main after window exists)
UI = {
    "active_shape": 0,  # 0=cuboid,1=cylinder,2=sphere
    "sliders": {},      # created in main (x,y,z,scale)
    "fill_cb": None,
    "radio": None,
    "dragging": None,   # which slider label is being dragged
}

def ui_build_controls(img_w, img_h):
    # Place UI on the left image (which is w=RGB_SIZE[0])
    # Margins
    left = 18
    top = 18
    radio = RadioGroup(["CUBOID","CYLINDER","SPHERE"], left, top, w_each=130, h=28, active_idx=0)
    y = top + 40
    sliders = {
        "X (mm)": Slider("X (mm)", left, y, 360, 32, vmin=-RANDOM_RANGE_XY_MM*1.5, vmax=+RANDOM_RANGE_XY_MM*1.5, v0=0.0),
        "Y (mm)": Slider("Y (mm)", left, y+40, 360, 32, vmin=-RANDOM_RANGE_XY_MM*1.5, vmax=+RANDOM_RANGE_XY_MM*1.5, v0=0.0),
        "Z (mm)": Slider("Z (mm)", left, y+80, 360, 32, vmin=0.0, vmax=Z_RANGE_MM, v0=0.0),
        "Scale":  Slider("Scale",  left, y+120, 360, 32, vmin=SCALE_MIN, vmax=SCALE_MAX, v0=1.0),
    }
    cb = CheckBox("Fill (semi-transparent)", left, y+170, 22, checked=False)
    UI["radio"] = radio
    UI["sliders"] = sliders
    UI["fill_cb"] = cb
    UI["dragging"] = None

def ui_sync_from_shape(shape):
    # update sliders + checkbox to reflect current shape's state
    UI["sliders"]["X (mm)"].value = shape["pos_mm"][0]
    UI["sliders"]["Y (mm)"].value = shape["pos_mm"][1]
    UI["sliders"]["Z (mm)"].value = shape["pos_mm"][2]
    UI["sliders"]["Scale"].value  = shape.get("scale", 1.0)
    UI["fill_cb"].checked         = shape.get("filled", False)

def ui_apply_to_shape(shape):
    # write current UI slider values back into the shape
    x = UI["sliders"]["X (mm)"].value
    y = UI["sliders"]["Y (mm)"].value
    z = UI["sliders"]["Z (mm)"].value
    sc = UI["sliders"]["Scale"].value
    shape["pos_mm"] = (x, y, z)
    shape["scale"]  = sc
    shape["filled"] = UI["fill_cb"].checked
    return shape

def ui_draw(img):
    # draw radio + sliders + checkbox on img (left half)
    UI["radio"].draw(img)
    for s in UI["sliders"].values():
        s.draw(img)
    UI["fill_cb"].draw(img)

def ui_mouse(event, x, y, flags, param):
    # handle clicks only on the left image (where UI is drawn)
    if event == cv2.EVENT_LBUTTONDOWN:
        # radio?
        idx = UI["radio"].hit(x, y)
        if idx is not None:
            UI["active_shape"] = idx
            UI["radio"].active = idx
            ui_sync_from_shape(SHAPES[idx])
            return
        # checkbox?
        if UI["fill_cb"].hit(x, y):
            UI["fill_cb"].toggle()
            return
        # sliders?
        for name, s in UI["sliders"].items():
            if s.hit(x, y):
                UI["dragging"] = name
                s.set_from_mouse(x)
                return
    elif event == cv2.EVENT_MOUSEMOVE:
        if UI["dragging"] is not None:
            s = UI["sliders"][UI["dragging"]]
            s.set_from_mouse(x)
    elif event == cv2.EVENT_LBUTTONUP:
        UI["dragging"] = None

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
    global SHAPES
    if RANDOMIZE_ON_START:
        randomize_positions(SHAPES)

    pipeline = build_pipeline()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        K, dist = get_rgb_calibration(device, RGB_SIZE[0], RGB_SIZE[1])
        aruco_dict, aruco_params = make_aruco_detector_legacy()

        # Build UI and mouse callback
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        ui_build_controls(RGB_SIZE[0], RGB_SIZE[1])
        ui_sync_from_shape(SHAPES[UI["active_shape"]])
        cv2.setMouseCallback(WINDOW_NAME, ui_mouse)

        template_id = None
        fps_smooth, t_prev = None, time.time()

        print("[INFO] Running. Mouse: radio/slider/checkbox on left pane. Keys: t=template, r=randomize, q=quit")
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
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_MM, K, dist)
                rvec, tvec = rvecs[anchor_idx], tvecs[anchor_idx]

                # Sync UI -> active shape parameters
                SHAPES[UI["active_shape"]] = ui_apply_to_shape(SHAPES[UI["active_shape"]])

                # Draw axes on anchor
                cv2.drawFrameAxes(frame_bgr, K, dist, rvec, tvec, MARKER_SIZE_MM * 0.5)

                # Render each shape
                for s in SHAPES:
                    if s["type"] == "cuboid":
                        wx,dy,hh = s["base_size_mm"]
                        sc = s.get("scale",1.0)
                        size = (wx*sc, dy*sc, hh*sc)
                        edges, faces = build_cuboid_edges_faces(size, s["pos_mm"], s["rot_deg"])
                        if s.get("filled", False):
                            draw_faces_filled(frame_bgr, faces, rvec, tvec, K, dist, face_color=(80,210,80), alpha=0.25)
                        draw_edges(frame_bgr, edges, rvec, tvec, K, dist, color=s.get("color",(0,255,0)), thickness=s.get("thickness",2))

                    elif s["type"] == "cylinder":
                        rad = s["base_radius_mm"] * s.get("scale",1.0)
                        hgt = s["base_height_mm"] * s.get("scale",1.0)
                        edges, faces = build_cylinder_edges_faces(rad, hgt, s["pos_mm"], s["rot_deg"], segments=s.get("segments",36))
                        if s.get("filled", False):
                            draw_faces_filled(frame_bgr, faces, rvec, tvec, K, dist, face_color=(80,200,240), alpha=0.25)
                        draw_edges(frame_bgr, edges, rvec, tvec, K, dist, color=s.get("color",(0,200,255)), thickness=s.get("thickness",2))

                    elif s["type"] == "sphere":
                        rad = s["base_radius_mm"] * s.get("scale",1.0)
                        center3d = marker_center_offset() + np.array(s["pos_mm"], dtype=np.float32)
                        draw_sphere(frame_bgr, center3d, rad, rvec, tvec, K, dist,
                                    color=s.get("color",(255,180,0)),
                                    thickness=s.get("thickness",2),
                                    filled=s.get("filled", False), alpha=0.25)

                # Anchor info
                m_id = int(ids[anchor_idx])
                dist_cm = float(np.linalg.norm(tvec)) / 10.0
                c = corners[anchor_idx][0].astype(int)
                cx, cy = c.mean(axis=0).astype(int)
                cv2.putText(frame_bgr, f"Anchor ID {m_id}  {dist_cm:.1f} cm",
                            (cx-60, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # Compose and UI overlay
            stacked = np.hstack([frame_bgr, depth_color])
            # FPS
            now = time.time()
            inst = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            fps_smooth = inst if fps_smooth is None else (0.9*fps_smooth + 0.1*inst)

            cv2.putText(stacked, f"AR (left) | Depth (right)   FPS: {fps_smooth:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(stacked, "Mouse: pick shape (radio), drag sliders, click Fill | Keys: t=template, r=randomize, q=quit",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,0), 2)

            # Draw UI on the left pane only
            ui_draw(stacked)

            cv2.imshow(WINDOW_NAME, stacked)
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
                randomize_positions(SHAPES)
                ui_sync_from_shape(SHAPES[UI["active_shape"]])
                print("[INFO] Shapes re-randomized around marker center.")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
