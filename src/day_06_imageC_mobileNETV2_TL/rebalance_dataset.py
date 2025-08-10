#!/usr/bin/env python3
# rebalance_dataset.py
# Ensure every class has at least 1 image in train and val, or drop the class if empty.

import os, shutil, random

DATA_DIR = os.path.join("..", "..", "..", "data", "openimages")   # <-- adjust if needed
TRAIN = os.path.join(DATA_DIR, "train")
VAL   = os.path.join(DATA_DIR, "val")
VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".ppm",".pgm"}

random.seed(42)

def list_images(d):
    if not os.path.isdir(d): return []
    return [os.path.join(d,f) for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in VALID_EXTS and os.path.isfile(os.path.join(d,f))]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    train_classes = {c for c in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, c))}
    val_classes   = {c for c in os.listdir(VAL)   if os.path.isdir(os.path.join(VAL,   c))}
    all_classes = sorted(train_classes | val_classes)

    moved_v2t = []  # (cls, n)
    moved_t2v = []
    dropped   = []

    for cls in all_classes:
        tdir = os.path.join(TRAIN, cls)
        vdir = os.path.join(VAL,   cls)
        ensure_dir(tdir); ensure_dir(vdir)

        timgs = list_images(tdir)
        vimgs = list_images(vdir)
        tN, vN = len(timgs), len(vimgs)

        # Drop classes with 0 total images
        if tN == 0 and vN == 0:
            try:
                shutil.rmtree(tdir, ignore_errors=True)
                shutil.rmtree(vdir, ignore_errors=True)
            except Exception:
                pass
            dropped.append(cls)
            continue

        # If train has none but val has some, move a few from val->train
        if tN == 0 and vN > 0:
            k = max(1, min(8, vN // 10))  # move at least 1, up to ~10%
            random.shuffle(vimgs)
            for p in vimgs[:k]:
                shutil.move(p, os.path.join(tdir, os.path.basename(p)))
            moved_v2t.append((cls, k))
            tN += k; vN -= k

        # If val has none but train has some, move a few train->val
        if vN == 0 and tN > 0:
            k = max(1, min(8, tN // 10))
            random.shuffle(timgs)
            # refresh list in case we already moved some above
            timgs = list_images(tdir)
            random.shuffle(timgs)
            for p in timgs[:k]:
                shutil.move(p, os.path.join(vdir, os.path.basename(p)))
            moved_t2v.append((cls, k))

    print(f"[INFO] Rebalance complete.")
    if moved_v2t:
        print("[INFO] val → train:", sum(n for _,n in moved_v2t), "files across",
              len(moved_v2t), "classes")
    if moved_t2v:
        print("[INFO] train → val:", sum(n for _,n in moved_t2v), "files across",
              len(moved_t2v), "classes")
    if dropped:
        print("[INFO] Dropped empty classes:", dropped)

if __name__ == "__main__":
    main()

