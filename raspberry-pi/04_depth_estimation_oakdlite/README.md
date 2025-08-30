# Depth From Mono and Stereo Cameras With Live 3D Point Clouds

<img width="1743" height="883" alt="image" src="https://github.com/user-attachments/assets/23b74743-6df1-43d0-9208-ea4c63e4a08e" />

<img width="1742" height="951" alt="image" src="https://github.com/user-attachments/assets/f34b8a18-df08-4cd1-a240-13f4e165e11e" />

<img width="1733" height="882" alt="image" src="https://github.com/user-attachments/assets/70f7e4c9-7868-4988-9491-7f58519fff9e" />

<img width="1701" height="835" alt="image" src="https://github.com/user-attachments/assets/0bcc6488-9aab-41a6-af4a-31df90680053" />

A principled, production-style mini project that demonstrates fast, synchronized RGB + depth acquisition, metric scaling, and live 3D visualization using a separate GPU-accelerated viewer.

## What This Repository Contains

 - Monocular depth pipeline using a Raspberry Pi 5, IMX500 camera, and a lightweight depth model.
 - Publishes normalized depth, a colormap preview, camera intrinsics, and an optional metric mapping.
 - Strictly synchronized RGB + depth display and an external publisher for the 3D viewer.
 - Stereo depth pipeline using an OAK-D Lite.
 - Runs hardware stereo to produce metric depth in millimeters, optionally aligned to RGB.
 - Publishes depth at its native size with intrinsics for that exact size.
 - Unified live 3D point-cloud viewer (Qt + PyQtGraph OpenGL).
 - Subscribes to either monocular (“dn”) or stereo (“mm”) publisher without code changes.
 - nteractive controls for camera, density, point size, orthographic mode, screenshot, and an optional X-flip.
 - IMX500 calibration helper for monocular metric scaling.
 - ROI-based collection of (dn_mean, Z_meters) pairs and least-squares fit for A and B in Z ≈ A(1 − dn) + B.

## Highlights
- Strict synchronization between RGB, depth, and the point cloud feed using a single producer “bundle” and frame IDs.
- Three detached windows, all in lockstep: RGB HUD, depth map, and 3D point cloud viewer.
- Metric point clouds when metric calibration or metric depth is available; otherwise, plots in relative units.
- Robustness: signal-based shutdown; non-blocking queues; low HWM publishing to avoid UI lag.
- Configurability: alignment to RGB, left-right consistency check, subpixel mode, median filters, density scaling, orthographic camera toggle, and more.

## Repository Layout
### Monocular publisher
imx500_mono_depth_FasterRCNN_pub.py
Publishes normalized depth and a preview colormap, plus intrinsics and A,B (if available). Also shows a local RGB+Depth HUD window.

### Stereo publisher
oakd_stereo_depth_pub.py
Builds a DepthAI pipeline, displays RGB + legend depth, and publishes metric depth in millimeters at its native size with matching intrinsics.

### Unified 3D viewer
pc_live_view_qt_stereo.py
Subscribes on tcp://127.0.0.1:5556 topic pc. Works with both monocular (“dn”) and stereo (“mm”). Fast OpenGL scatter rendering.

### Calibration
imx500_calibrate_verbose.py
ROI selection on live RGB, collect known distances, fit A,B, and save imx500_metric.json.

## Supported Hardware and OS

 - Raspberry Pi 5 with IMX500 camera for monocular depth.
 - OAK-D Lite for stereo depth.
 - Linux environment with V4L2/libcamera stack and a working OpenGL driver for the viewer.

## Software Requirements

 - Python 3.10 or newer.
 - Packages used across components:
numpy, opencv-python, pyzmq, PySide6, pyqtgraph, torch (monocular), depthai (stereo), picamera2 (Pi camera).
 - OpenGL-capable display for the Qt viewer.

## Quick Start
### Create an environment and install packages
 - Create and activate a Python virtual environment.
 - Install the required packages listed above for the pipelines you intend to run.
 - Ensure camera interfaces are enabled and accessible by your user.

### Run the monocular pipeline
 - Start the publisher
 - Opens a combined RGB + Depth HUD window
 - Publishes on tcp://*:5556, topic pc.
 - Start the viewer in a separate terminal
 - The point cloud window should populate within a second.
 - Use hotkeys below for interactivity.
 - (Optional) Run the calibration helper to generate imx500_metric.json
 - Draw an ROI, press D, enter the known distance in meters.
 - After one or more ROIs are set, press S to save A and B.
 
### Run the stereo pipeline (OAK-D Lite)
 - Connect the device and start the publisher
 - Opens RGB and Depth windows.
 - Publishes metric depth in millimeters plus exact intrinsics for that frame size.
 - Start the viewer

## Viewer Hotkeys

O: Toggle perspective and ortho-like camera

+ / −: Increase or decrease point size

] / [: Increase or decrease density scaling (0.5×, 1×, 2×)

R / F: Rotate yaw by ±5 degrees

H: Flip X (mirror)

S: Save a PNG of the current view

Q / Esc: Quit

## How It Works
### Data flow

#### Capture
Monocular: RGB frames from IMX500 via Picamera2.
Stereo: color preview + stereo mono streams inside the OAK device.

#### Depth estimation
Monocular: A lightweight depth model produces a normalized depth map dn ∈ [0, 1].
Stereo: DepthAI computes metric depth in millimeters; optional alignment to the color camera field of view.

#### Metric scaling for monocular
When available, we apply Z_meters ≈ A * (1 − dn) + B.
A,B are produced by the calibration helper and saved to imx500_metric.json.
If no calibration file is present, the viewer still renders a relative-units point cloud.

#### Intrinsics
Monocular: intrinsics are provided or derived for the live stream size.
Stereo: intrinsics are read directly from the OAK calibration for the exact published depth size.

#### Back-projection to 3D
Pixel coordinates (u, v) are converted to camera-frame coordinates (X, Y, Z) using fx, fy, cx, cy and Z.
The viewer applies a simple camera-to-GL mapping so axes appear as: X right, Y forward, Z up.

#### Transport and visualization
The publisher sends depth plus a JPEG colormap and a small JSON header over ZMQ.
The viewer drains to the newest message (no lag), reconstructs the cloud, and draws it with OpenGL scatter.

### Why a separate viewer process

 - Decouples rendering from capture and inference.
 - Avoids UI backpressure: publishers drop stale frames when the viewer is slow.
 - Keeps the pipelines deterministic and synchronized.

## Design Choices and Alternatives

### Depth model choice for monocular
Chosen: a lightweight model for CPU-only inference on the Pi.
Alternatives: larger models or transformer-based estimators yield higher fidelity but drastically reduce frame rate on embedded CPUs. We favored interactivity and synchronization over marginal accuracy.

### Stereo via OAK-D Lite
Chosen: hardware stereo that outputs metric depth at useful frame rates with optional alignment to RGB.
Alternatives: CPU stereo or custom CUDA code would increase complexity and reduce portability on a Pi-first setup.

### Qt + PyQtGraph OpenGL viewer
Chosen: fast scatter rendering, simple API, straightforward keyboard interaction, no heavyweight scene graph.
Alternatives: Matplotlib or Plotly are slower for dense, live point clouds; Open3D is powerful but heavier to deploy for a minimal, always-on viewer.

### ZMQ PUB/SUB
Chosen: trivial setup, message framing, low overhead, and non-blocking high-water-mark control.
Alternatives: shared memory or gRPC offer different trade-offs but add complexity for this scale.

### Alignment and filtering (stereo)
Tunable: alignment to RGB, left-right check, subpixel mode, and median filtering.
Alternatives: always-on heavy filtering harms frame rate; we expose knobs so users can tune per scene.

## Performance and Physical Parameters

The project keeps the resolution unchanged across flights and aims to maximize FPS while preserving visual fidelity and synchronization.

### Frame sizes
Monocular RGB: 640 × 480.
Stereo depth: native OAK depth resolution (e.g., 400p or 720p mono bases) and, if enabled, aligned to the RGB FOV.

### Point cloud density
Base point count equals the number of pixels in the published depth frame.
Effective points after density scaling approximate:
points ≈ width × height × (densityScale)²
for uniform scaling.

### Synchronization and latency
Single-producer “bundle” handoff guarantees that the RGB HUD, depth preview, and published depth describe the same moment.
The viewer discards stale messages and renders only the newest frame.

### CPU and memory considerations
Monocular CPU inference is compute-bound; the script caps Torch threads to all available cores.
Point clouds transfer as raw depth plus JPEG color, balancing bandwidth and CPU time.
The viewer holds at most one full frame’s worth of point data.

## Calibration Workflow for Monocular Metric Depth

 - Run imx500_calibrate_verbose.py.
 - Draw an ROI on a target with a known distance.
 - Press D and enter the distance in meters.
 - Repeat for multiple ROIs at different distances for a better fit.
 - Press S to save A and B to imx500_metric.json.
 - Re-run the monocular publisher; it will automatically pick up A and B and publish metric point clouds.
 - Do you need the scene to contain only that ROI target?
    No, but clearer ROIs with uniform surfaces yield more stable dn_mean and thus a better fit.

## Development Sequence

1. Baseline monocular depth at 640 × 480 with synchronized RGB and depth HUD.
2. 3D point cloud visualization added as a detached window, with camera axes and basic interaction.
3. Strict synchronization across RGB, depth, and 3D by introducing a single-producer bundle and frame IDs.
4. Calibration tooling introduced to derive A and B for monocular metric scaling using simple ROI sampling.
5. ZMQ publisher integrated to decouple compute from rendering and to support remote viewing.
6. Unified viewer taught to understand both “dn” (monocular) and “mm” (stereo metric) payloads, with density scaling, yaw rotation, orthographic toggle, X-flip, and one-click PNG.
7. Stereo pipeline (OAK-D Lite) implemented with alignment and quality knobs; publishers now send matching intrinsics for the exact depth size.
8. Code cleanup and hardening: constants moved to the top, naming normalized, robust exits added, queues bounded, and windows made reliably killable.

## Tuning Guide

 - Need more FPS on stereo
 - Disable alignment to RGB when not required.
 - Keep subpixel off; use a smaller median kernel.
 - Use the 400p preset.
 - Need smoother clouds at distance
 - Enable subpixel and use K5 or K7 median.
 - Use 720p mono and alignment to RGB so overlay matches the color frame.
 - Viewer is heavy
 - Reduce densityScale or point size.
 - Keep only one viewer instance.
 - Ensure your OpenGL driver is active.
 - Monocular looks too “relative”
 - Run calibration and supply A,B.
 - Use large, flat, well-lit ROIs at several depths.

## Known Limitations

 - Monocular metric accuracy depends on calibration targets and lighting; the mapping is an approximation.
 - Extremely textureless or reflective surfaces may confuse stereo or monocular depth.
 - On very low power systems, dense point clouds at 2× density scaling may affect interactivity.

## Troubleshooting

 - Blank 3D window
   - Verify PyQtGraph OpenGL and Qt are installed correctly.
   - Ensure the publisher is running and the viewer subscribes on the correct port.
 - Viewer shows axes but no points
   - Confirm the publisher is sending frames and the viewer logs do not show decoding errors.
   - Check that depth values are nonzero and intrinsics match the published depth size.
 - Cannot close windows
   - Press Q or Esc in the viewer; Ctrl+C in the terminal.
   - Scripts include signal handlers; if still stuck, kill the process from the shell.
 - Point cloud looks mirrored left-right
   - Press H in the viewer to flip the X axis for visual preference.

## Glossary

dn: Normalized depth value in [0, 1] from the monocular estimator.

A, B: Calibration coefficients for converting dn to meters via Z ≈ A(1 − dn) + B.

fx, fy, cx, cy: Camera intrinsics for the current frame size.

Alignment: Warping depth to match the RGB camera’s field of view and pixel geometry.
