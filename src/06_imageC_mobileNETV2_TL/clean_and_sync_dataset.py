#!/usr/bin/env python3
# clean_and_sync_dataset.py
import os, shutil, random

DATA_DIR = os.path.join("..", "..", "..", "data", "openimages")           # <-- change if needed
TRAIN = os.path.join(DATA_DIR, "train")
VAL   = os.path.join(DATA_DIR, "val")
VALID = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".ppm",".pgm"}
random.seed(42)

def files(d):
    if not os.path.isdir(d): return []
    out=[]
    for f in os.listdir(d):
        p=os.path.join(d,f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in VALID:
            out.append(p)
    return out

def ensure(d): os.makedirs(d, exist_ok=True)

def main():
    tset = {c for c in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN,c))}
    vset = {c for c in os.listdir(VAL)   if os.path.isdir(os.path.join(VAL,c))}
    classes = sorted(tset | vset)

    moved_v2t=moved_t2v=0
    dropped=[]
    for cls in classes:
        tdir = os.path.join(TRAIN, cls); ensure(tdir)
        vdir = os.path.join(VAL,   cls); ensure(vdir)
        timgs = files(tdir); vimgs = files(vdir)
        tN, vN = len(timgs), len(vimgs)
        tot = tN + vN

        # If absolutely empty -> drop class entirely
        if tot == 0:
            shutil.rmtree(tdir, ignore_errors=True)
            shutil.rmtree(vdir, ignore_errors=True)
            dropped.append(cls)
            continue

        # If train empty but val has some -> move 1..min(8, ~10%)
        if tN == 0 and vN > 0:
            k = max(1, min(8, max(1, vN//10)))
            random.shuffle(vimgs)
            for p in vimgs[:k]:
                shutil.move(p, os.path.join(tdir, os.path.basename(p)))
            moved_v2t += k

        # If val empty but train has some -> move 1..min(8, ~10%)
        timgs = files(tdir); vimgs = files(vdir)
        tN, vN = len(timgs), len(vimgs)
        if vN == 0 and tN > 0:
            k = max(1, min(8, max(1, tN//10)))
            random.shuffle(timgs)
            for p in timgs[:k]:
                shutil.move(p, os.path.join(vdir, os.path.basename(p)))
            moved_t2v += k

    # Final pass: remove any empty class folders so ImageFolder can't complain
    def prune_empty(root):
        removed=[]
        for cls in list(os.listdir(root)):
            p=os.path.join(root,cls)
            if not os.path.isdir(p): continue
            if len(files(p))==0:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(cls)
        return removed

    rem_t = prune_empty(TRAIN)
    rem_v = prune_empty(VAL)

    # Sync class sets again (if pruning removed a class from only one split)
    tset = {c for c in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN,c))}
    vset = {c for c in os.listdir(VAL)   if os.path.isdir(os.path.join(VAL,c))}
    only_t = sorted(tset - vset)
    only_v = sorted(vset - tset)
    # Remove lonely classes so both splits share identical class list
    for cls in only_t:
        shutil.rmtree(os.path.join(TRAIN,cls), ignore_errors=True)
    for cls in only_v:
        shutil.rmtree(os.path.join(VAL,cls), ignore_errors=True)

    tset2 = {c for c in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN,c))}
    vset2 = {c for c in os.listdir(VAL)   if os.path.isdir(os.path.join(VAL,  c))}

    print("[INFO] Dropped empty classes:", dropped)
    print("[INFO] Moved val→train:", moved_v2t, " | train→val:", moved_t2v)
    print("[INFO] train classes:", len(tset2), " val classes:", len(vset2))
    print("[INFO] Done. Splits now have identical non-empty class sets.")

if __name__ == "__main__":
    main()
