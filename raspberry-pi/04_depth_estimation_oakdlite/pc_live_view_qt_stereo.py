#!/usr/bin/env python3
"""
pc_live_view_qt.py  (works for monocular 'dn' and OAK 'mm')

Live 3D point-cloud viewer using PyQtGraph OpenGL. Subscribes to a ZMQ PUB stream
on tcp://127.0.0.1:5556 with topic "pc". Frames are multipart:
    [b"pc"] [header-json] [payload] [jpeg-colormap]

Header (subset):
  - mode: "dn" (normalized depth, monocular) or "mm" (metric depth, OAK)
  - w, h: frame width/height
  - fx, fy, cx, cy: intrinsics for *this* frame size
  - a, b : optional MiDaS metric mapping (Z ≈ a*(1 - dn) + b)
  - metric: bool, if true axes units are meters

Controls (focus the viewer window):
  O         : toggle perspective / ortho-like (tiny FOV)
  + / -     : point size up/down
  ] / [     : density scale 0.5× / 2×
  R / F     : yaw rotate +/− 5°
  H         : flip X (mirror) if your visual expectation prefers it
  S         : save PNG of current view
  Q / Esc   : quit

Author : Rishav Kanth
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
# CONSTANTS  (no magic numbers below)

# ZMQ subscriber
ZMQ_CONNECT_ADDR   = "tcp://127.0.0.1:5556"
ZMQ_TOPIC          = "pc"
ZMQ_RCV_HWM        = 1

# Window & camera defaults
WINDOW_TITLE       = "PointCloud3D (Live)"
INIT_WINDOW_W      = 1200
INIT_WINDOW_H      = 700
INIT_CAM_DISTANCE  = 6
INIT_CAM_ELEVATION = 10
INIT_CAM_AZIMUTH   = 45
PERSPECTIVE_FOV    = 60       # degrees
ORTHO_LIKE_FOV     = 1        # tiny FOV ≈ orthographic

# Axes colors (RGB 0..255)
AXIS_COLOR_X       = (255,   0,   0)  # X → right (red)
AXIS_COLOR_Y       = (  0, 255,   0)  # Y → forward (green)
AXIS_COLOR_Z       = (  0,   0, 255)  # Z → up (blue)

# Point cloud rendering / interaction
POINT_SIZE_INIT    = 1.5
POINT_SIZE_MIN     = 0.5
POINT_SIZE_MAX     = 12.0
DENSITY_INIT       = 1.0      # 0.5, 1.0, 2.0
DENSITY_STEP       = 0.5
DENSITY_MIN        = 0.5
DENSITY_MAX        = 2.0
YAW_STEP_DEG       = 5.0

# Polling / HUD
POLL_TIMER_MS      = 1
HUD_QSS            = "QLabel { color: white; background-color: rgba(0,0,0,120); padding: 4px; }"

# ────────────────────────────────────────────────────────────────────────────────
# ZMQ SUB socket (module-level so the widget can read it directly)

zmqCtx = zmq.Context.instance()
zmqSub = zmqCtx.socket(zmq.SUB)
zmqSub.setsockopt(zmq.RCVHWM, ZMQ_RCV_HWM)
zmqSub.connect(ZMQ_CONNECT_ADDR)
zmqSub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_TOPIC)

# ────────────────────────────────────────────────────────────────────────────────
class LivePC(QtWidgets.QWidget):
    """Main widget that renders the 3D scatter and a small HUD."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(INIT_WINDOW_W, INIT_WINDOW_H)

        # ---- GL view camera ----
        self.glView = gl.GLViewWidget()
        self.glView.setCameraPosition(distance=INIT_CAM_DISTANCE,
                                      elevation=INIT_CAM_ELEVATION,
                                      azimuth=INIT_CAM_AZIMUTH)
        self.glView.setCameraParams(fov=PERSPECTIVE_FOV)   # perspective by default
        self.glView.opts['center'] = pg.Vector(0, 0, 0)
        self.orthoMode = False

        # ---- World axes for orientation ----
        self.axesItems = []
        self._add_axis((0, 0, 0), (1, 0, 0), AXIS_COLOR_X)   # X right (red)
        self._add_axis((0, 0, 0), (0, 1, 0), AXIS_COLOR_Y)   # Y forward (green)
        self._add_axis((0, 0, 0), (0, 0, 1), AXIS_COLOR_Z)   # Z up (blue)

        # ---- Point cloud scatter (opaque) ----
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            size=POINT_SIZE_INIT,
            pxMode=True
        )
        self.scatter.setGLOptions('opaque')
        self.glView.addItem(self.scatter)
        self.pointSize = POINT_SIZE_INIT

        # ---- HUD ----
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

        # ---- Runtime state (updated by headers/keys) ----
        self.fx = 489.0; self.fy = 460.0
        self.cx = 320.0; self.cy = 240.0
        self.metricA = 2.0; self.metricB = 0.0
        self.isMetric = False
        self.modeName = "dn"          # "dn" (monocular) or "mm" (OAK)
        self.rotYawDeg = 0.0
        self.densityScale = DENSITY_INIT
        self.centerSet = False
        self.flipX = False            # toggled by 'H'
        self.isRunning = True

    # ─────────── helpers ───────────
    def _add_axis(self, p0, p1, colorRgb):
        """Add a single axis line segment to the GL scene."""
        pts = np.array([p0, p1], dtype=np.float32)
        col = np.array([[c/255.0 for c in (*colorRgb, 255)]], dtype=np.float32)
        col = np.repeat(col, 2, axis=0)
        item = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
        item.setGLOptions('opaque')
        self.glView.addItem(item)
        self.axesItems.append(item)

    # ─────────── UI (Qt overrides; names must stay camelCase) ───────────
    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        """Hotkeys: see file header for the full list."""
        k = ev.key()
        if k in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.isRunning = False
            self.close()
        elif k == QtCore.Qt.Key_O:
            self.orthoMode = not self.orthoMode
            self.glView.setCameraParams(fov=(ORTHO_LIKE_FOV if self.orthoMode else PERSPECTIVE_FOV))
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.pointSize = min(POINT_SIZE_MAX, self.pointSize + 0.5)
            self.scatter.setData(size=self.pointSize)
        elif k in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.pointSize = max(POINT_SIZE_MIN, self.pointSize - 0.5)
            self.scatter.setData(size=self.pointSize)
        elif k == QtCore.Qt.Key_BracketRight:
            self.densityScale = min(DENSITY_MAX, self.densityScale + DENSITY_STEP)
        elif k == QtCore.Qt.Key_BracketLeft:
            self.densityScale = max(DENSITY_MIN, self.densityScale - DENSITY_STEP)
        elif k == QtCore.Qt.Key_R:
            self.rotYawDeg += YAW_STEP_DEG
        elif k == QtCore.Qt.Key_F:
            self.rotYawDeg -= YAW_STEP_DEG
        elif k == QtCore.Qt.Key_H:
            self.flipX = not self.flipX
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
                    print("[viewer] screenshot failed:", e)
                    return
            print(f"[viewer] Saved {fn}")

    def closeEvent(self, ev):
        """Stop polling when the window closes."""
        self.isRunning = False
        ev.accept()

    # ─────────── ZMQ → 3D (main update loop) ───────────
    def update_from_stream(self):
        """Drain ZMQ to newest frame, back-project to 3D, and update GL scatter."""
        if not self.isRunning:
            return
        try:
            # Drain queue to the newest message (avoid lag)
            got = False
            while True:
                try:
                    topic, headerRaw, payloadRaw, jpgRaw = zmqSub.recv_multipart(flags=zmq.NOBLOCK)
                    got = True
                except zmq.Again:
                    break
            if not got:
                return

            # ---- Deserialize header & decode colormap ----
            header = json.loads(headerRaw.decode("utf-8"))
            w, h = int(header["w"]), int(header["h"])
            self.fx, self.fy = float(header["fx"]), float(header["fy"])
            self.cx, self.cy = float(header["cx"]), float(header["cy"])
            self.isMetric = bool(header.get("metric", False))
            self.modeName = header.get("mode", "dn")

            cmap = cv2.imdecode(np.frombuffer(jpgRaw, dtype=np.uint8), cv2.IMREAD_COLOR)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)

            # ---- Rebuild Z (meters) and XY (camera frame) ----
            if self.modeName == "mm":
                zMm = np.frombuffer(payloadRaw, dtype=np.uint16).reshape(h, w)
                Z = zMm.astype(np.float32) * 0.001  # mm → m
                uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
                Xc = (uu - self.cx) * Z / self.fx
                Yc = (vv - self.cy) * Z / self.fy
            else:
                dn = np.frombuffer(payloadRaw, dtype=np.float16).astype(np.float32).reshape(h, w)
                self.metricA = float(header.get("a", 2.0))
                self.metricB = float(header.get("b", 0.0))
                Z = np.clip(self.metricA * (1.0 - dn) + self.metricB, 1e-3, 100.0)
                uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
                if self.isMetric:
                    Xc = (uu - self.cx) * Z / self.fx
                    Yc = (vv - self.cy) * Z / self.fy
                else:
                    # Relative XY if intrinsics not meaningful
                    Xc = (uu - w * 0.5) / float(w)
                    Yc = (vv - h * 0.5) / float(h)

            # ---- Optional density rescale (for interactivity only) ----
            if self.densityScale != 1.0:
                interp = cv2.INTER_LINEAR if self.densityScale > 1.0 else cv2.INTER_AREA
                Z    = cv2.resize(Z,    None, fx=self.densityScale, fy=self.densityScale, interpolation=interp)
                Xc   = cv2.resize(Xc,   None, fx=self.densityScale, fy=self.densityScale, interpolation=interp)
                Yc   = cv2.resize(Yc,   None, fx=self.densityScale, fy=self.densityScale, interpolation=interp)
                cmap = cv2.resize(cmap, None, fx=self.densityScale, fy=self.densityScale, interpolation=cv2.INTER_LINEAR)
                h, w = Z.shape

            # ---- Optional mirror (visual preference) ----
            if self.flipX:
                Xc = -Xc

            # ---- Apply viewer yaw around camera Y (forward) ----
            yaw = math.radians(self.rotYawDeg)
            cyaw, syaw = math.cos(yaw), math.sin(yaw)
            Xr = Xc * cyaw + Z * syaw
            Zr = -Xc * syaw + Z * cyaw
            Yr = Yc

            # ---- Camera → GL mapping: (Xr, Zr, -Yr) ----
            pos = np.stack(
                [Xr.astype(np.float32), Zr.astype(np.float32), (-Yr).astype(np.float32)],
                axis=-1
            ).reshape(-1, 3)

            # RGBA colors (opaque)
            col = (cmap.reshape(-1, 3).astype(np.float32) / 255.0)
            col = np.concatenate([col, np.ones((col.shape[0], 1), np.float32)], axis=1)

            # Center on first valid cloud
            if not self.centerSet and pos.size:
                med = np.median(pos, axis=0)
                self.glView.opts['center'] = pg.Vector(float(med[0]), float(med[1]), float(med[2]))
                self.centerSet = True

            # ---- Update scatter (fast path) ----
            self.scatter.setData(pos=pos, color=col, size=self.pointSize, pxMode=True)

            # ---- HUD ----
            modeTxt = "OAK metric (m)" if self.modeName == "mm" else ("Monocular meters" if self.isMetric else "Monocular relative")
            self.hudLabel.setText(
                f"Projection: {'ORTHO-like' if self.orthoMode else 'Perspective'}   "
                f"Point size: {self.pointSize:.1f}   Density: {self.densityScale:.1f}x   "
                f"{modeTxt}   FlipX:{self.flipX}   "
                f"Axes(GL): X→right (red), Y→forward (green), Z→up (blue)"
            )

        except Exception as e:
            print("[viewer] error:", e)

# ────────────────────────────────────────────────────────────────────────────────
def main():
    """Qt application bootstrap + robust Ctrl+C handling."""
    app = QtWidgets.QApplication(sys.argv)
    widget = LivePC()
    widget.show()
    signal.signal(signal.SIGINT, lambda *_: QtWidgets.QApplication.quit())
    sys.exit(app.exec())

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
