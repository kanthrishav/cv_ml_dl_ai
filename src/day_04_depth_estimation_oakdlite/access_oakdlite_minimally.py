#!/usr/bin/env python3
import depthai as dai
import cv2

# 1) Build pipeline
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
# Use CAM_A (the on-board color sensor) instead of the deprecated RGB constant
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# 2) Start device
print("Looking for Oak-D Lite…")
with dai.Device(pipeline) as device:
    print("Connected to:", device.getDeviceInfo().getMxId())
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("Streaming preview. Press 'q' to quit.")

    # 3) Display loop
    while True:
        inRgb = q.tryGet()
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.imshow("Oak-D Lite Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
