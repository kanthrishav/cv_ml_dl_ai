#!/usr/bin/env python3
# pc_live_view_qt.py  (works for monocular "dn" and OAK "mm")
# Keys: O = ortho/persp, +/- = point size, [/] = density, R/F = yaw, H = flip X, S = PNG, Q/Esc = quit

import sys, json, time, math, os, signal
import numpy as np
import zmq, cv2
from PySide6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=6, elevation=10, azimuth=45)
        self.view.setCameraParams(fov=60)
        self.view.opts['center'] = pg.Vector(0,0,0)
        self.ortho_mode = False

        self.axes_items = []
        self._add_axis((0,0,0), (1,0,0), (255,0,0))     # X right (red)
        self._add_axis((0,0,0), (0,1,0), (0,255,0))     # Y forward (green)
        self._add_axis((0,0,0), (0,0,1), (0,0,255))     # Z up (blue)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1,3), dtype=np.float32),
            color=(1.0,1.0,1.0,1.0),
            size=1.5, pxMode=True
        )
        self.scatter.setGLOptions('opaque')
        self.view.addItem(self.scatter)
        self.point_size = 1.5

        self.hud = QtWidgets.QLabel("", self)
        self.hud.setStyleSheet("QLabel { color: white; background-color: rgba(0,0,0,120); padding: 4px; }")
        self.hud.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view); layout.addWidget(self.hud)
        layout.setStretch(0, 1); layout.setStretch(1, 0)

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.update_from_stream); self.timer.start(1)

        # State
        self.fx = 489.0; self.fy = 460.0
        self.cx = 320.0; self.cy = 240.0
        self.A = 2.0; self.B = 0.0
        self.is_metric = False
        self.mode = "dn"    # "dn" (monocular) or "mm" (OAK)
        self.rot_yaw = 0.0
        self.density_scale = 1.0
        self.center_set = False
        self.flip_x = False  # <--- H toggles this

        self._running = True

    def _add_axis(self, p0, p1, color_rgb):
        pts = np.array([p0, p1], dtype=np.float32)
        col = np.array([[c/255.0 for c in (*color_rgb, 255)]], dtype=np.float32)
        col = np.repeat(col, 2, axis=0)
        item = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
        item.setGLOptions('opaque'); self.view.addItem(item); self.axes_items.append(item)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q): self._running = False; self.close()
        elif k == QtCore.Qt.Key_O: self.ortho_mode = not self.ortho_mode; self.view.setCameraParams(fov=(1 if self.ortho_mode else 60))
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal): self.point_size = min(12.0, self.point_size + 0.5); self.scatter.setData(size=self.point_size)
        elif k in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore): self.point_size = max(0.5, self.point_size - 0.5); self.scatter.setData(size=self.point_size)
        elif k == QtCore.Qt.Key_BracketRight: self.density_scale = min(2.0, self.density_scale + 0.5)
        elif k == QtCore.Qt.Key_BracketLeft:  self.density_scale = max(0.5, self.density_scale - 0.5)
        elif k == QtCore.Qt.Key_R: self.rot_yaw += 5.0
        elif k == QtCore.Qt.Key_F: self.rot_yaw -= 5.0
        elif k == QtCore.Qt.Key_H: self.flip_x = not self.flip_x   # <--- mirror X if needed
        elif k == QtCore.Qt.Key_S:
            fn = f"pc_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            try: img = self.view.readQImage(); img.save(fn)
            except Exception:
                try: img = self.view.grabFramebuffer(); img.save(fn)
                except Exception as e: print("[viewer] screenshot failed:", e); return
            print(f"[viewer] Saved {fn}")

    def closeEvent(self, ev): self._running = False; ev.accept()

    def update_from_stream(self):
        if not self._running: return
        try:
            got = False
            while True:
                try:
                    topic, header_raw, payload_raw, jpg = sub.recv_multipart(flags=zmq.NOBLOCK)
                    got = True
                except zmq.Again:
                    break
            if not got: return

            header = json.loads(header_raw.decode("utf-8"))
            w, h = int(header["w"]), int(header["h"])
            self.fx, self.fy = float(header["fx"]), float(header["fy"])
            self.cx, self.cy = float(header["cx"]), float(header["cy"])
            self.is_metric = bool(header.get("metric", False))
            self.mode = header.get("mode", "dn")

            cmap = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)

            if self.mode == "mm":
                z_mm = np.frombuffer(payload_raw, dtype=np.uint16).reshape(h, w)
                Z = z_mm.astype(np.float32) * 0.001  # meters
                uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                Xc = (uu - self.cx) * Z / self.fx
                Yc = (vv - self.cy) * Z / self.fy
            else:
                dn = np.frombuffer(payload_raw, dtype=np.float16).astype(np.float32).reshape(h, w)
                self.A = float(header.get("a", 2.0)); self.B = float(header.get("b", 0.0))
                Z = np.clip(self.A * (1.0 - dn) + self.B, 1e-3, 100.0)
                uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                if self.is_metric:
                    Xc = (uu - self.cx) * Z / self.fx
                    Yc = (vv - self.cy) * Z / self.fy
                else:
                    Xc = (uu - w*0.5) / float(w)
                    Yc = (vv - h*0.5) / float(h)

            # Optional density rescale
            if self.density_scale != 1.0:
                interp = cv2.INTER_LINEAR if self.density_scale > 1.0 else cv2.INTER_AREA
                Z   = cv2.resize(Z,   None, fx=self.density_scale, fy=self.density_scale, interpolation=interp)
                Xc  = cv2.resize(Xc,  None, fx=self.density_scale, fy=self.density_scale, interpolation=interp)
                Yc  = cv2.resize(Yc,  None, fx=self.density_scale, fy=self.density_scale, interpolation=interp)
                cmap= cv2.resize(cmap,None, fx=self.density_scale, fy=self.density_scale, interpolation=cv2.INTER_LINEAR)
                h, w = Z.shape

            # Flip X if requested (in case your visual expectation prefers it)
            if self.flip_x: Xc = -Xc

            # Viewer yaw about camera forward
            yaw = math.radians(self.rot_yaw)
            cyaw, syaw = math.cos(yaw), math.sin(yaw)
            Xr = Xc*cyaw + Z*syaw
            Zr = -Xc*syaw + Z*cyaw
            Yr = Yc

            # Camera -> GL: X(right), Y(forward), Z(up)
            pos = np.stack([Xr.astype(np.float32),
                            Zr.astype(np.float32),
                            (-Yr).astype(np.float32)], axis=-1).reshape(-1, 3)
            col = (cmap.reshape(-1,3).astype(np.float32) / 255.0)
            col = np.concatenate([col, np.ones((col.shape[0],1), np.float32)], axis=1)

            if not self.center_set and pos.size:
                med = np.median(pos, axis=0)
                self.view.opts['center'] = pg.Vector(float(med[0]), float(med[1]), float(med[2]))
                self.center_set = True

            self.scatter.setData(pos=pos, color=col, size=self.point_size, pxMode=True)

            mode_txt = "OAK metric (m)" if self.mode == "mm" else ("Monocular meters" if self.is_metric else "Monocular relative")
            self.hud.setText(
                f"Projection: {'ORTHO-like' if self.ortho_mode else 'Perspective'}   "
                f"Point size: {self.point_size:.1f}   Density: {self.density_scale:.1f}x   "
                f"{mode_txt}   FlipX:{self.flip_x}   "
                f"Axes(GL): X→right (red), Y→forward (green), Z→up (blue)"
            )

        except Exception as e:
            print("[viewer] error:", e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LivePC(); w.show()
    signal.signal(signal.SIGINT, lambda *_: QtWidgets.QApplication.quit())
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
