#!/usr/bin/env python3
# OAK-D Lite dual-axis turntable scan -> TSDF -> STL/PLY/OBJ
# Fix: use foreground 3D ICP (object motion) instead of RGB-D odometry (camera motion),
# so rotating the OBJECT works with a stationary camera.

import os, sys, time, math, subprocess, shutil
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-input", "--no-cache-dir"] + pkgs)

# DepthAI
try:
    import depthai as dai
except Exception:
    _pip_install(["depthai"])
    import depthai as dai

# Open3D core (avoid dash/flask imports)
try:
    import importlib
    o3d = importlib.import_module("open3d.cpu.pybind")
except Exception:
    try:
        _pip_install(["open3d>=0.18.0"])
        import importlib
        o3d = importlib.import_module("open3d.cpu.pybind")
    except Exception:
        import open3d as o3d

import numpy as np
import cv2
from collections import deque

# ---------------- TUNING ----------------
MODE = os.environ.get("SCAN_MODE", "BALANCED").upper()  # FAST | BALANCED | QUALITY
RGB_W, RGB_H = 640, 360
FPS_TARGET   = 30
MONO_RES     = "400p"             # 400p or 720p

# per-pass limits
MAX_SCAN_SECONDS_PASS = 90
MAX_FRAMES_PASS = 1600

# TSDF params (meters)
if MODE == "FAST":
    VOXEL_LEN = 0.010; SDF_TRUNC = 0.040; MAX_DEPTH = 3.5
elif MODE == "QUALITY":
    VOXEL_LEN = 0.004; SDF_TRUNC = 0.020; MAX_DEPTH = 3.0
else:  # BALANCED
    VOXEL_LEN = 0.006; SDF_TRUNC = 0.030; MAX_DEPTH = 3.2

MIN_DEPTH = 0.25

# progress/coverage control
ACCUM_DEG_GOAL = 340.0
PROGRESS_PRINT_EVERY_S = 1.0

# ICP/segmentation
ICP_MAX_CORR = 0.03   # 3cm correspondence radius
ICP_VOX = max(0.008, VOXEL_LEN*2.0)
FITNESS_MIN = 0.15    # if ICP fitness lower than this, skip frame
DEPTH_BAND_NEAR = 250.0  # mm below center median kept
DEPTH_BAND_FAR  = 350.0  # mm above center median kept
MIN_OBJ_PTS = 2000       # require some points after masking

# wait behavior
WAIT_STILL_SECONDS = 0.8
WAIT_STILL_TIMEOUT = 4.0

SAVE_POINTCLOUDS = True
BEEP = True
# ---------------------------------------

def _beep(times=1):
    if not BEEP: return
    for _ in range(times):
        for player in ("paplay", "aplay"):
            if shutil.which(player):
                try:
                    if player == "paplay":
                        os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga >/dev/null 2>&1")
                    else:
                        os.system("printf '\\a';")
                except Exception:
                    pass
                break
        else:
            try: print("\a", end="", flush=True)
            except Exception: pass
        time.sleep(0.1)

def mono_res_from_text(s):
    s = s.upper()
    return dai.MonoCameraProperties.SensorResolution.THE_720_P if "720" in s \
           else dai.MonoCameraProperties.SensorResolution.THE_400_P

def make_pipeline():
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(FPS_TARGET)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewSize(RGB_W, RGB_H)
    cam.setPreviewKeepAspectRatio(True)

    monoL = p.create(dai.node.MonoCamera)
    monoR = p.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoL.setResolution(mono_res_from_text(MONO_RES))
    monoR.setResolution(mono_res_from_text(MONO_RES))
    monoL.setFps(FPS_TARGET); monoR.setFps(FPS_TARGET)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)  # deprecation warning OK
    # Align depth to RGB with identical size (per Luxonis docs)
    stereo.setLeftRightCheck(True)                          # required/robust for aligned depth
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)       # align to RGB FOV
    stereo.setOutputSize(RGB_W, RGB_H)                      # force RGB==Depth size
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

    monoL.out.link(stereo.left); monoR.out.link(stereo.right)
    x_rgb = p.create(dai.node.XLinkOut); x_rgb.setStreamName("rgb"); x_rgb.setFpsLimit(FPS_TARGET)
    x_dep = p.create(dai.node.XLinkOut); x_dep.setStreamName("depth"); x_dep.setFpsLimit(FPS_TARGET)
    cam.preview.link(x_rgb.input); stereo.depth.link(x_dep.input)
    return p

def get_latest(q):
    try: pkt = q.tryGet()
    except RuntimeError: return None
    last = None
    while pkt is not None:
        last = pkt
        try: pkt = q.tryGet()
        except RuntimeError: break
    return last

def mask_object(depth_mm):
    """Auto-foreground mask: assume object is the closest large blob near the center."""
    z = depth_mm.astype(np.float32)
    valid = (z > 0)
    if np.count_nonzero(valid) < 1000:
        return np.zeros_like(depth_mm, np.uint8)

    h, w = z.shape
    x0, y0 = int(0.3*w), int(0.3*h)
    x1, y1 = int(0.7*w), int(0.7*h)
    center = z[y0:y1, x0:x1]
    center_v = center[center > 0]
    if center_v.size >= 500:
        d_med = float(np.median(center_v))
    else:
        vals = z[valid]
        d_med = float(np.percentile(vals, 20.0))

    lo = max(0.0, d_med - DEPTH_BAND_NEAR)
    hi = d_med + DEPTH_BAND_FAR
    m = (z >= lo) & (z <= hi)

    m = (m.astype(np.uint8) * 255)
    m = cv2.medianBlur(m, 5)
    # keep largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(labels == largest, 255, 0).astype(np.uint8)
    return m

def rgbd_masked(rgb_bgr, depth_mm, mask_u8):
    """Zero out background in depth so Open3D ignores it."""
    dep = depth_mm.copy()
    dep[mask_u8 == 0] = 0
    z = dep.astype(np.float32) * 0.001
    z[(z < MIN_DEPTH) | (z > MAX_DEPTH)] = 0.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(z),
        convert_rgb_to_intensity=False
    )
    return rgbd

def pcd_from_rgbd(rgbd, pin):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pin)
    return pcd

def yaw_deg_from_R(R):
    # Open3D image/camera convention: +Z forward; yaw around +Y.
    return abs(math.degrees(math.atan2(R[0,2], R[2,2])))

def icp_object_delta(prev_pcd, curr_pcd):
    """Estimate object motion between frames via point-to-plane ICP on masked clouds."""
    p1 = prev_pcd.voxel_down_sample(ICP_VOX)
    p2 = curr_pcd.voxel_down_sample(ICP_VOX)
    if len(p1.points) < MIN_OBJ_PTS or len(p2.points) < MIN_OBJ_PTS:
        return None, 0.0
    p1.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    p2.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    result = o3d.pipelines.registration.registration_icp(
        p2, p1, max_correspondence_distance=ICP_MAX_CORR,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result.transformation, result.fitness

def scan_pass(dev, pin, pass_name, instructions, seconds_cap=MAX_SCAN_SECONDS_PASS):
    q_rgb = dev.getOutputQueue("rgb",   maxSize=4, blocking=False)
    q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)

    print(f"\n[PASS {pass_name}] {instructions}")
    _beep()

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LEN, sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Camera pose in "world" (static object) frame; we synthesize motion as inverse of object motion.
    T_cam = np.eye(4)

    # ---- WAIT FOR STILLNESS (with timeout) ----
    print(f"[PASS {pass_name}] Waiting for scene to be still (~{WAIT_STILL_SECONDS}s)...")
    stable_start = None
    last_gray = None
    t_wait0 = time.time()
    while True:
        pkt_rgb = get_latest(q_rgb); pkt_dep = get_latest(q_dep)
        if pkt_rgb is None or pkt_dep is None:
            if time.time() - t_wait0 > WAIT_STILL_TIMEOUT: break
            continue
        rgb = pkt_rgb.getCvFrame(); dep = pkt_dep.getFrame()
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if last_gray is None:
            last_gray = g.copy(); continue
        score = float(np.mean(cv2.absdiff(g, last_gray)))
        last_gray = g
        if score < 1.0:
            if stable_start is None: stable_start = time.time()
            if time.time() - stable_start >= WAIT_STILL_SECONDS: break
        else:
            stable_start = None
        if time.time() - t_wait0 > WAIT_STILL_TIMEOUT:
            break

    print(f"[PASS {pass_name}] SCANNING STARTED – rotate smoothly now.")
    _beep(2)

    accum_yaw = 0.0
    frames_used = 0
    last_print = time.time()

    # initialize with first valid foreground
    prev_pcd = None
    t0 = time.time()

    while True:
        if (time.time() - t0) > seconds_cap or frames_used >= MAX_FRAMES_PASS:
            print(f"[PASS {pass_name}] Auto-stop (time/frames cap)."); break

        pkt_rgb = get_latest(q_rgb); pkt_dep = get_latest(q_dep)
        if pkt_rgb is None or pkt_dep is None:
            time.sleep(0.001); continue

        rgb = pkt_rgb.getCvFrame(); dep = pkt_dep.getFrame()
        mask = mask_object(dep)
        if np.count_nonzero(mask) < MIN_OBJ_PTS:
            continue

        rgbd = rgbd_masked(rgb, dep, mask)
        pcd  = pcd_from_rgbd(rgbd, pin)

        if len(pcd.points) < MIN_OBJ_PTS:
            continue

        if prev_pcd is None:
            # first integration
            volume.integrate(rgbd, pin, np.linalg.inv(T_cam))
            prev_pcd = pcd
            frames_used += 1
            continue

        T_obj, fitness = icp_object_delta(prev_pcd, pcd)
        if T_obj is None or fitness < FITNESS_MIN:
            # low confidence; skip (no accumulation)
            continue

        # Object moved by T_obj in camera coords -> equivalent camera delta is inverse
        T_cam_delta = np.linalg.inv(T_obj)
        T_cam = T_cam @ T_cam_delta

        # integrate this frame
        volume.integrate(rgbd, pin, np.linalg.inv(T_cam))
        prev_pcd = pcd
        frames_used += 1

        # coverage from camera yaw
        R = T_cam_delta[:3, :3]
        accum_yaw += yaw_deg_from_R(R)

        now = time.time()
        if now - last_print >= PROGRESS_PRINT_EVERY_S:
            last_print = now
            print(f"[PASS {pass_name}] coverage: {accum_yaw:.0f}/{ACCUM_DEG_GOAL:.0f} deg, frames: {frames_used}")

        if accum_yaw >= ACCUM_DEG_GOAL:
            print(f"[PASS {pass_name}] PASS COMPLETE (coverage {accum_yaw:.1f} deg).")
            _beep(2)
            break

    return frames_used, volume

def downsample_and_features(pcd, voxel=0.01):
    p_down = pcd.voxel_down_sample(voxel)
    if len(p_down.points) == 0: return p_down, None
    p_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        p_down, o3d.geometry.KDTreeSearchParamKNN(knn=100)
    )
    return p_down, fpfh

def register_fragments(pcd1, pcd2):
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return np.eye(4)
    v = 0.02
    s1, f1 = downsample_and_features(pcd1, v)
    s2, f2 = downsample_and_features(pcd2, v)
    if f1 is None or f2 is None:
        return np.eye(4)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s2, s1, f2, f1, True,
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )
    T_init = result_ransac.transformation
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, max_correspondence_distance=0.02,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp.transformation

def main():
    print(f"[INFO] Dual-axis scan (turntable ICP) | Mode={MODE}  (voxel={VOXEL_LEN}m, trunc={SDF_TRUNC}m)")
    p = make_pipeline()
    with dai.Device(p) as dev:
        # Intrinsics from device EEPROM (exact RGB_WxRGB_H)
        calib = dev.readCalibration()
        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, RGB_W, RGB_H), dtype=np.float64)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        pin = o3d.camera.PinholeCameraIntrinsic(RGB_W, RGB_H, fx, fy, cx, cy)
        print(f"[INFO] Intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        print("\n=== PHASE 1 (Z-axis) ===")
        print("Place object upright, center at camera height. Rotate 360° about its vertical (Z).")
        f1, vol1 = scan_pass(dev, pin, "Z", "Rotate object around Z (upright)...")
        if f1 < 3:
            print("[ERROR] Pass Z collected too few frames. Check depth visibility/lighting and try again.")
            return

        print("\n=== PHASE 2 (Y-axis) ===")
        print("Lay the object on its SIDE (original Y becomes vertical). Rotate 360° about that new vertical.")
        _beep(); time.sleep(1.5)
        f2, vol2 = scan_pass(dev, pin, "Y", "Lay object on side, then rotate around new vertical (original Y)...")
        if f2 < 3:
            print("[WARN] Pass Y had few frames; continuing with Pass Z only.")

        print("[INFO] Extracting fragment clouds for registration…")
        p1 = vol1.extract_point_cloud(); p1.estimate_normals()
        if f2 >= 3:
            p2 = vol2.extract_point_cloud(); p2.estimate_normals()
        else:
            p2 = o3d.geometry.PointCloud()

        if SAVE_POINTCLOUDS:
            if len(p1.points): o3d.io.write_point_cloud("fragment1_points.ply", p1)
            if len(p2.points): o3d.io.write_point_cloud("fragment2_points.ply", p2)

        T_21 = np.eye(4)
        if len(p1.points) and len(p2.points):
            print("[INFO] Registering fragments (global + ICP)…")
            T_21 = register_fragments(p1, p2)  # maps frag2 → frag1

        print("[INFO] Re-integrating fragments into a single TSDF…")
        final_vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_LEN, sdf_trunc=SDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # For re-integration we need stored keyframes; we approximate by sampling fused clouds
        # Simpler: directly merge meshes from both volumes (fast path)
        mesh1 = vol1.extract_triangle_mesh()
        mesh1.compute_vertex_normals()
        if f2 >= 3 and len(p2.points):
            mesh2 = vol2.extract_triangle_mesh(); mesh2.compute_vertex_normals()
            mesh2.transform(T_21)
            mesh = mesh1 + mesh2
        else:
            mesh = mesh1

        mesh = mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh("scan_mesh.stl", mesh, print_progress=True)
        o3d.io.write_triangle_mesh("scan_mesh.ply", mesh, print_progress=False)
        o3d.io.write_triangle_mesh("scan_mesh.obj", mesh, print_progress=False)
        if SAVE_POINTCLOUDS:
            pcd = mesh.sample_points_poisson_disk(200000) if len(mesh.triangles) else p1
            if len(pcd.points): o3d.io.write_point_cloud("scan_points.ply", pcd, print_progress=False)
        print("[OK] Saved: scan_mesh.stl / scan_mesh.ply / scan_mesh.obj")

if __name__ == "__main__":
    main()
