# list_devices.py
import depthai as dai

devices = dai.Device.getAllAvailableDevices()
print("Detected devices:", devices)
