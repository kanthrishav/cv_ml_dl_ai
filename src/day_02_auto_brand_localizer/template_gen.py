#!/usr/bin/env python3
import cv2
import time
from picamera2 import Picamera2

def select_template(save_path="template.png"):
    """
    Capture one frame from the Pi camera, allow the user to select an ROI,
    and save it as the template image.
    """
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (4000, 3000)})
    picam.configure(config)
    picam.start()
    time.sleep(2)  # warm up camera

    frame = picam.capture_array()
    picam.stop()

    # Let user draw the ROI rectangle on the frame
    bbox = cv2.selectROI("Select Template (draw bounding box and press ENTER/SPACE)", frame, False, False)
    x, y, w, h = [int(v) for v in bbox]
    template = frame[y:y+h, x:x+w]
    name_ = input("Enter name of template : ")
    save_path = "template_" + name_ + ".png"
    cv2.imwrite(save_path, template)
    cv2.destroyAllWindows()
    print(f"[INFO] Template saved to '{save_path}'.")

if __name__ == "__main__":
    select_template()