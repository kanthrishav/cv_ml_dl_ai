#!/usr/bin/env python3
# pc_live_view_qt.py
# Live 3D point-cloud viewer (PyQtGraph OpenGL) subscribing to tcp://127.0.0.1:5556

import sys, json, time, math, os, signal
import numpy as np
import zmq, cv2
from PySide6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ---------- ZMQ SUB ----------
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.RCVHWM, 1)
sub.connect("tcp://127.0.0.1:5556")
sub.setsockopt_string(zmq.SUBSCRIBE, "pc")

class LivePC(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PointCloud3D (Live)")
        self.resize(1200, 700)

        # GL view
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=6, elevation=10, azimuth=45, fov=60)  # perspective by default
        self.view.opts['center'] = pg.Vector(0,0,0)
        self.ortho_mode = False

        # Axes (camera coordinates: X right, Y down, Z forward)
        self.axes_items = []
        self._add_axis((0,0,0), (1.0,0,0), (255,  0,  0))  # X red
        self._add_axis((0,0,0), (0,1.0,0), (  0,255,  0))  # Y green
        self._add_axis((0,0,0), (0,0,1.0), (  0,  0,255))  # Z blue

        # Scatter
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(1,1,1,1), size=1.5, pxMode=True)
        self.view.addItem(self.scatter)
        self.point_size = 1.5

        # HUD label
        self.hud = QtWidgets.QLabel("", self)
        self.hud.setStyleSheet("QLabel { color: white; background-color: rgba(0,0,0,120); padding: 4px; }")
        self.hud.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addWidget(self.hud)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        # Timer to poll ZMQ
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_from_stream)
        self.timer.start(1)  # as fast as possible

        # State
        self.fx = 489.0; self.fy = 460.0
        self.cx = 320.0; self.cy = 240.0
        self.A = 2.0; self.B = 0.0
        self.metric = False
        self.rot_yaw = 0.0         # extra yaw rotation for viewer
        self.density_scale = 1.0    # 0.5, 1.0, 2.0

        # Robust close
        self._running = True

    def _add_axis(self, p0, p1, color):
        pts = np.array([p0, p1], dtype=float)
        col = np.array([[c/255.0 for c in color]]*2)
        item = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True, mode='lines')
        self.view.addItem(item)
        self.axes_items.append(item)

    # ----------- UI ----------
    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self._running = False
            self.close()
        elif k == QtCore.Qt.Key_O:
            self.ortho_mode = not self.ortho_mode
            self.view.setCameraParams(fov=(1 if self.ortho_mode else 60))
        elif k == QtCore.Qt.Key_Plus or k == QtCore.Qt.Key_Equal:
            self.point_size = min(12.0, self.point_size + 0.5)
            self.scatter.setData(size=self.point_size)
        elif k == QtCore.Qt.Key_Minus or k == QtCore.Qt.Key_Underscore:
            self.point_size = max(0.5, self.point_size - 0.5)
            self.scatter.setData(size=self.point_size)
        elif k == QtCore.Qt.Key_BracketRight:
            self.density_scale = min(2.0, self.density_scale + 0.5)
        elif k == QtCore.Qt.Key_BracketLeft:
            self.density_scale = max(0.5, self.density_scale - 0.5)
        elif k == QtCore.Qt.Key_R:
            self.rot_yaw += 5.0
        elif k == QtCore.Qt.Key_F:
            self.rot_yaw -= 5.0
        elif k == QtCore.Qt.Key_S:
            # save PNG of the GL view
            img = self.view.readQImage()
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"pc_screenshot_{ts}.png"
            img.save(fn)
            print(f"[viewer] Saved {fn}")

    def closeEvent(self, ev):
        self._running = False
        ev.accept()

    # ----------- ZMQ -> 3D -----------
    def update_from_stream(self):
        if not self._running: return
        try:
            # drain ZMQ to last (avoid lag)
            got = False
            while True:
                try:
                    topic, header_raw, dn_raw, jpg = sub.recv_multipart(flags=zmq.NOBLOCK)
                    got = True
                except zmq.Again:
                    break
            if not got:
                return

            header = json.loads(header_raw.decode("utf-8"))
            w, h = int(header["w"]), int(header["h"])
            self.A, self.B = float(header["a"]), float(header["b"])
            self.fx, self.fy = float(header["fx"]), float(header["fy"])
            self.cx, self.cy = float(header["cx"]), float(header["cy"])
            self.metric = bool(header.get("metric", False))

            dn = np.frombuffer(dn_raw, dtype=np.float16).astype(np.float32).reshape(h, w)
            cmap = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)

            # optional density up/down + simple hole-fill via bilinear
            if self.density_scale != 1.0:
                interp = cv2.INTER_LINEAR if self.density_scale > 1.0 else cv2.INTER_AREA
                dn   = cv2.resize(dn,   None, fx=self.density_scale, fy=self.density_scale, interpolation=interp)
                cmap = cv2.resize(cmap, None, fx=self.density_scale, fy=self.density_scale, interpolation=cv2.INTER_LINEAR)
                h, w = dn.shape

            # Build metric 3D (camera frame: X right, Y down, Z forward)
            Z = np.clip(self.A * (1.0 - dn) + self.B, 1e-3, 100.0) if self.metric else (1.0 - dn)
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            if self.metric:
                X = (uu - self.cx) * Z / self.fx
                Y = (vv - self.cy) * Z / self.fy
            else:
                X = (uu - w*0.5) / float(w)
                Y = (vv - h*0.5) / float(h)

            # Apply yaw rotation (viewer control) around camera Y axis
            yaw = math.radians(self.rot_yaw)
            cyaw, syaw = math.cos(yaw), math.sin(yaw)
            Xr = X*cyaw + Z*syaw
            Zr = -X*syaw + Z*cyaw
            Yr = Y

            pos = np.stack([Xr, Yr, Zr], axis=-1).reshape(-1, 3)
            col = (cmap.reshape(-1,3).astype(np.float32) / 255.0)
            col = np.concatenate([col, np.ones((col.shape[0],1), np.float32)], axis=1)  # add alpha

            # Update scatter efficiently
            self.scatter.setData(pos=pos, color=col, size=self.point_size, pxMode=True)

            self.hud.setText(
                f"Projection: {'ORTHO-like' if self.ortho_mode else 'Perspective'}   "
                f"Point size: {self.point_size:.1f}   Density: {self.density_scale:.1f}x   "
                f"{'Units: meters ' if self.metric else 'Units: relative '} "
                f"(A={self.A:.3f}, B={self.B:.3f})   Axes: X→right (red), Y→down (green), Z→forward (blue)"
            )

        except Exception as e:
            print("[viewer] error:", e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LivePC()
    w.show()
    # robust kill on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: QtWidgets.QApplication.quit())
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
