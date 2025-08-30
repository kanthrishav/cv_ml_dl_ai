#!/usr/bin/env python3
"""
pc_live_view_qt.py

Live 3D point-cloud viewer (PyQtGraph OpenGL) that subscribes to a ZMQ stream on
tcp://127.0.0.1:5556.

The publisher is expected to send multipart frames with the topic "pc":
  [b"pc"] [header-json] [depth-payload] [jpeg-colormap]
  - header fields (subset): w,h, fx,fy,cx,cy, a,b, metric (bool)
  - if metric==True and payload is float16 'dn': Z ≈ a*(1-dn)+b (meters)
  - some publishers may send metric depth directly (uint16 millimeters)

Controls (focused on the viewer window):
  O         : toggle perspective / ortho-like (tiny FOV)
  + / -     : point size up/down
  ] / [     : density scale 0.5×, 1×, 2×
  R / F     : yaw rotate ±5°
  M / N     : near-cut distance +/− (meters)
  S         : save PNG of current view
  Q / Esc   : quit

Author : Rishav KAnth
"""

# ────────────────────────────────────────────────────────────────────────────────
# Standard / third-party imports
import sys, json, time, math, os, signal
import numpy as np
import zmq, cv2
from PySide6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (no magic numbers below)

# ZMQ
ZMQ_CONNECT_ADDR      = "tcp://127.0.0.1:5556"
ZMQ_TOPIC             = "pc"
ZMQ_RCV_HWM           = 1  # drop backlog; prefer most recent frame

# Window / camera defaults
WINDOW_TITLE          = "PointCloud3D (Live)"
INIT_WINDOW_W         = 1200
INIT_WINDOW_H         = 700
INIT_CAM_DISTANCE     = 6
INIT_CAM_ELEVATION    = 10
INIT_CAM_AZIMUTH      = 45
PERSPECTIVE_FOV_DEG   = 60        # default perspective FOV
ORTHO_LIKE_FOV_DEG    = 1         # tiny FOV approximates orthographic

# Axes colors (RGB 0..255)
AXIS_COLOR_X          = (255,   0,   0)  # X right (red)
AXIS_COLOR_Y          = (  0, 255,   0)  # Y forward (green)
AXIS_COLOR_Z          = (  0,   0, 255)  # Z up (blue)

# Point cloud rendering
POINT_SIZE_INIT       = 1.5
POINT_SIZE_MIN        = 0.5
POINT_SIZE_MAX        = 12.0
DENSITY_SCALE_INIT    = 1.0       # 0.5, 1.0, 2.0
DENSITY_SCALE_STEP    = 0.5
DENSITY_SCALE_MIN     = 0.5
DENSITY_SCALE_MAX     = 2.0

# Interaction steps
YAW_STEP_DEG          = 5.0
NEAR_CUT_INIT_M       = 0.75      # drop anything nearer than this distance
NEAR_CUT_MIN_M        = 0.05
NEAR_CUT_MAX_M        = 5.0
NEAR_CUT_STEP_M       = 0.05

# Polling / UI
POLL_TIMER_MS         = 1         # poll ZMQ as fast as possible
HUD_QSS               = "QLabel { color: white; background-color: rgba(0,0,0,120); padding: 4px; }"

# ────────────────────────────────────────────────────────────────────────────────
# ZMQ subscriber (module-level so the class can use it directly)
zmqCtx = zmq.Context.instance()
zmqSub = zmqCtx.socket(zmq.SUB)
zmqSub.setsockopt(zmq.RCVHWM, ZMQ_RCV_HWM)         # prefer newest frame
zmqSub.connect(ZMQ_CONNECT_ADDR)
zmqSub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_TOPIC)

# ────────────────────────────────────────────────────────────────────────────────
class LivePC(QtWidgets.QWidget):
    """
    Main widget that renders the 3D scatter and HUD. Uses GLViewWidget from
    PyQtGraph for fast OpenGL-based point rendering. The scene coordinate frame:

        GL axes: X → right (red), Y → forward (green), Z → up (blue)

    Incoming camera frame is converted from camera coordinates
    (X right, Y down, Z forward) to GL coordinates as:
        (Xg, Yg, Zg) = (Xr, Zr, -Yr)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(INIT_WINDOW_W, INIT_WINDOW_H)

        # ---- GL view & camera setup ----
        self.glView = gl.GLViewWidget()
        self.glView.setCameraPosition(distance=INIT_CAM_DISTANCE,
                                      elevation=INIT_CAM_ELEVATION,
                                      azimuth=INIT_CAM_AZIMUTH)
        self.glView.setCameraParams(fov=PERSPECTIVE_FOV_DEG)  # perspective by default
        self.glView.opts['center'] = pg.Vector(0, 0, 0)
        self.orthoMode = False

        # ---- World axes (for orientation) ----
        self.axesItems = []
        self._add_axis((0, 0, 0), (1.0, 0,   0),   AXIS_COLOR_X)  # X
        self._add_axis((0, 0, 0), (0,   1.0, 0),   AXIS_COLOR_Y)  # Y
        self._add_axis((0, 0, 0), (0,   0,   1.0), AXIS_COLOR_Z)  # Z

        # ---- Point cloud scatter (opaque for speed/visibility) ----
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            size=POINT_SIZE_INIT,
            pxMode=True
        )
        self.scatter.setGLOptions('opaque')
        self.glView.addItem(self.scatter)
        self.pointSize = POINT_SIZE_INIT

        # ---- HUD label ----
        self.hudLabel = QtWidgets.QLabel("", self)
        self.hudLabel.setStyleSheet(HUD_QSS)
        self.hudLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # ---- Layout ----
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.glView)
        layout.addWidget(self.hudLabel)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        # ---- ZMQ poll timer ----
        self.pollTimer = QtCore.QTimer(self)
        self.pollTimer.timeout.connect(self.update_from_stream)
        self.pollTimer.start(POLL_TIMER_MS)

        # ---- State (updated by incoming headers) ----
        self.fx = 489.0; self.fy = 460.0
        self.cx = 320.0; self.cy = 240.0
        self.metricA = 2.0; self.metricB = 0.0
        self.isMetric = False
        self.rotYawDeg = 0.0           # extra yaw rotation around camera Y (forward)
        self.densityScale = DENSITY_SCALE_INIT
        self.nearCutM = NEAR_CUT_INIT_M

        # Center cloud when first points arrive
        self.centerSet = False

        # Robust close control
        self.isRunning = True

    # ───────────── helpers ─────────────
    def _add_axis(self, p0, p1, colorRgb):
        """Add a single axis line segment to the GL scene."""
        pts = np.array([p0, p1], dtype=np.float32)
        col = np.array([[colorRgb[0]/255.0, colorRgb[1]/255.0, colorRgb[2]/255.0, 1.0]]*2, dtype=np.float32)
        item = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
        item.setGLOptions('opaque')
        self.glView.addItem(item)
        self.axesItems.append(item)

    # ───────────── UI / hotkeys (Qt override; name must remain) ─────────────
    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        """Handle viewer hotkeys (see header)."""
        k = ev.key()
        if k in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.isRunning = False
            self.close()
        elif k == QtCore.Qt.Key_O:
            self.orthoMode = not self.orthoMode
            self.glView.setCameraParams(fov=(ORTHO_LIKE_FOV_DEG if self.orthoMode else PERSPECTIVE_FOV_DEG))
        elif k == QtCore.Qt.Key_Plus or k == QtCore.Qt.Key_Equal:
            self.pointSize = min(POINT_SIZE_MAX, self.pointSize + 0.5)
            self.scatter.setData(size=self.pointSize)
        elif k == QtCore.Qt.Key_Minus or k == QtCore.Qt.Key_Underscore:
            self.pointSize = max(POINT_SIZE_MIN, self.pointSize - 0.5)
            self.scatter.setData(size=self.pointSize)
        elif k == QtCore.Qt.Key_BracketRight:
            self.densityScale = min(DENSITY_SCALE_MAX, self.densityScale + DENSITY_SCALE_STEP)
        elif k == QtCore.Qt.Key_BracketLeft:
            self.densityScale = max(DENSITY_SCALE_MIN, self.densityScale - DENSITY_SCALE_STEP)
        elif k == QtCore.Qt.Key_R:
            self.rotYawDeg += YAW_STEP_DEG
        elif k == QtCore.Qt.Key_F:
            self.rotYawDeg -= YAW_STEP_DEG
        elif k == QtCore.Qt.Key_M:
            self.nearCutM = min(NEAR_CUT_MAX_M, self.nearCutM + NEAR_CUT_STEP_M)
        elif k == QtCore.Qt.Key_N:
            self.nearCutM = max(NEAR_CUT_MIN_M, self.nearCutM - NEAR_CUT_STEP_M)
        elif k == QtCore.Qt.Key_S:
            fn = f"pc_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            try:
                img = self.glView.readQImage()
                img.save(fn)
            except Exception:
                try:
                    img = self.glView.grabFramebuffer()
                    img.save(fn)
                except Exception as e:
                    print("[viewer] screenshot failed:", e); return
            print(f"[viewer] Saved {fn}")

    # Qt override; must remain camelCase
    def closeEvent(self, ev):
        """Ensure background polling stops when the window closes."""
        self.isRunning = False
        ev.accept()

    # ───────────── ZMQ → 3D (main update loop) ─────────────
    def update_from_stream(self):
        """Poll ZMQ for the newest frame, reconstruct the 3D cloud, and update GL."""
        if not self.isRunning:
            return
        try:
            # Drain to the newest packet to avoid lag
            got = False
            while True:
                try:
                    topic, headerRaw, dnRaw, jpgRaw = zmqSub.recv_multipart(flags=zmq.NOBLOCK)
                    got = True
                except zmq.Again:
                    break
            if not got:
                return

            header = json.loads(headerRaw.decode("utf-8"))
            w, h = int(header["w"]), int(header["h"])
            self.metricA, self.metricB = float(header["a"]), float(header["b"])
            self.fx, self.fy = float(header["fx"]), float(header["fy"])
            self.cx, self.cy = float(header["cx"]), float(header["cy"])
            self.isMetric = bool(header.get("metric", False))

            dn = np.frombuffer(dnRaw, dtype=np.float16).astype(np.float32).reshape(h, w)
            cmap = cv2.imdecode(np.frombuffer(jpgRaw, dtype=np.uint8), cv2.IMREAD_COLOR)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)

            # Optional density rescale (for interactivity only)
            if self.densityScale != 1.0:
                interp = cv2.INTER_LINEAR if self.densityScale > 1.0 else cv2.INTER_AREA
                dn   = cv2.resize(dn,   None, fx=self.densityScale, fy=self.densityScale, interpolation=interp)
                cmap = cv2.resize(cmap, None, fx=self.densityScale, fy=self.densityScale, interpolation=cv2.INTER_LINEAR)
                h, w = dn.shape

            # ---- Near-cut mask & metric Z ----
            if self.isMetric:
                # Metric Z (meters) from normalized depth using the provided A/B
                Zc = np.clip(self.metricA * (1.0 - dn) + self.metricB, 1e-3, 100.0)
                keep = Zc >= self.nearCutM
            else:
                # Approximate near-cut using the mapping if A is available; otherwise fall back
                if abs(self.metricA) > 1e-6:
                    dnCut = 1.0 - (self.nearCutM - self.metricB) / self.metricA
                else:
                    dnCut = 0.8
                dnCut = float(np.clip(dnCut, 0.0, 1.0))
                keep = dn <= dnCut      # near = high dn → drop
                Zc = 1.0 - dn           # relative (no units)

            # ---- Back-project pixels to 3D in camera frame (X right, Y down, Z forward) ----
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            if self.isMetric:
                Xc = (uu - self.cx) * Zc / self.fx
                Yc = (vv - self.cy) * Zc / self.fy
            else:
                # Relative XY if intrinsics are not meaningful
                Xc = (uu - w * 0.5) / float(w)
                Yc = (vv - h * 0.5) / float(h)

            # ---- Apply viewer yaw around camera Y (forward) ----
            yaw = math.radians(self.rotYawDeg)
            cyaw, syaw = math.cos(yaw), math.sin(yaw)
            Xr = Xc * cyaw + Zc * syaw
            Zr = -Xc * syaw + Zc * cyaw
            Yr = Yc

            # ---- Camera → GL mapping: (Xr, Zr, -Yr) ----
            Xg = Xr.astype(np.float32)
            Yg = Zr.astype(np.float32)
            Zg = (-Yr).astype(np.float32)

            # ---- Apply near-cut mask & update GL scatter ----
            keep = keep.reshape(-1)
            pos = np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3).astype(np.float32)
            col = (cmap.reshape(-1, 3).astype(np.float32) / 255.0)
            col = np.concatenate([col, np.ones((col.shape[0], 1), np.float32)], axis=1)
            pos = pos[keep]
            col = col[keep]

            # Center view on the first valid cloud
            if not self.centerSet and pos.size:
                med = np.median(pos, axis=0)
                self.glView.opts['center'] = pg.Vector(float(med[0]), float(med[1]), float(med[2]))
                self.centerSet = True

            if pos.size:
                self.scatter.setData(pos=pos, color=col, size=self.pointSize, pxMode=True)
            else:
                # if everything got filtered, draw a transparent dummy point
                self.scatter.setData(pos=np.zeros((1, 3), np.float32), color=(1, 1, 1, 0), size=self.pointSize)

            # ---- HUD text ----
            self.hudLabel.setText(
                f"Projection: {'ORTHO-like' if self.orthoMode else 'Perspective'}   "
                f"Point size: {self.pointSize:.1f}   Density: {self.densityScale:.1f}x   "
                f"Near-cut: {self.nearCutM:.2f} m   "
                f"{'Units: meters ' if self.isMetric else 'Units: relative '} "
                f"(A={self.metricA:.3f}, B={self.metricB:.3f})   "
                f"Axes(GL): X→right (red), Y→forward (green), Z→up (blue)"
            )

        except Exception as e:
            print("[viewer] error:", e)

# ────────────────────────────────────────────────────────────────────────────────
def main():
    """Qt application bootstrap and robust Ctrl+C handling."""
    app = QtWidgets.QApplication(sys.argv)
    widget = LivePC()
    widget.show()

    # Robust kill on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: QtWidgets.QApplication.quit())
    sys.exit(app.exec())

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
