#!/usr/bin/env python3
import os, shutil

DATA_DIR = os.path.join("..", "..", "..", "data", "openimages")   # <-- your dataset root

def count_images(d):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".ppm",".pgm"}
    return sum(1 for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)

removed = []
for split in ("train","val"):
    split_dir = os.path.join(DATA_DIR, split)
    for cls in list(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir): continue
        n = count_images(cls_dir)
        if n == 0:
            shutil.rmtree(cls_dir)
            removed.append((split, cls))

print("[INFO] Removed empty class dirs:", removed)
print("[INFO] Done.")
