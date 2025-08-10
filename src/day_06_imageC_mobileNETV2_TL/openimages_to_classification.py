#!/usr/bin/env python3
# openimages_to_classification_v1_1.py
# Build a classification dataset from Open Images *validation + test* only (small CSVs).
# This avoids the gigantic train CSV that often fails or takes forever to fetch.
# No CLI args: edit CONFIG at the top.

import os, csv, io, sys, time, random, urllib.request
from collections import defaultdict
from PIL import Image

# ========================== CONFIG (EDIT HERE) ==========================
CLASS_NAMES = [
    # (same 127 classes you provided — keep/tweak as needed)
    "Coffee cup","Washing machine","Blender","Gas stove","Mechanical fan","Kettle",
    "Refrigerator","Mixer","Ceiling fan","Sink","Shower","Toilet","Door handle",
    "Coffee","Beer","Juice","Wine","Mobile phone","Ac power plugs and socket-outlet",
    "Computer mouse","Computer keyboard","Laptop","Headphones","Fork","Knife",
    "Drinking Straw","Cutting board","Spatula","Ladle","Chopsticks","Ratchet",
    "Scissors","Wrench","Screwdriver","Toothbrush","Waste container","Flowerpot",
    "Handbag","Briefcase","Suitcase","Backpack","Sandal","High heels","Jeans",
    "Swim cap","Glasses","Umbrella","Watch","Scarf","Sun hat","Earrings","Necklace",
    "Belt","Jacket","Miniskirt","Suit","Shirt","Coat","Dress","Shorts","Milk",
    "Cheese","Houseplant","Cabbage","Carrot","Salad","Broccoli","Bell pepper",
    "Cucumber","Tomato","Radish","Potato","Mushroom","Egg","Bread","Apple","Banana",
    "Pineapple","Lemon","Strawberry","Pear","Orange","Mango","Watermelon","Chair",
    "Desk","Sofa bed","Wardrobe","Nightstand","Bookcase","Coffee table","Kitchen & dining",
    "Chest of drawers","Cupboard","Bench","Stool","Shelf","Wall clock","Bathroom cabinet",
    "Drawer","Closet","Mirror","Window blind","Curtain","Pressure Cooker","Pillow",
    "Paper towel","Stapler","Eraser","Pen","Envelope","Adhesive tape","Plastic bag",
    "Toilet paper","Calculator","Box","Tap","Pencil Sharpener","Human beard","Human leg",
    "Human arm","Human foot","Human head","Human ear","Human mouth","Human face","Human eye",
]

OUT_DIR = os.path.join("..", "..", "..", "data", "openimages")   # output dataset root
PER_CLASS = 300                 # max crops per class (train+val combined)
TRAIN_RATIO = 0.85              # train/val split
MIN_BOX_FRAC = 0.07             # min bbox side vs min(image width,height)
PADDING = 0.15                  # padding around bbox (fraction of bbox size)
SEED = 42

# **Stable small files** (V5/2018_04) — validation & test only
# Official download pages reference these exact CSVs.  ✔ Boxes  ✔ Image URLs
# - V5 boxes: validation/test (small)  :contentReference[oaicite:2]{index=2}
# - 2018_04 image metadata with URLs: validation/test  :contentReference[oaicite:3]{index=3}
URLS = {
    "class_desc": [
        "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
    ],
    "val_boxes": [
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv",
    ],
    "test_boxes": [
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv",
    ],
    "val_images": [
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    ],
    "test_images": [
        "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    ],
}
# ======================================================================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def fetch_csv_any(url_list, local_path):
    if os.path.isfile(local_path): return local_path
    last_err = None
    for url in url_list:
        try:
            print(f"[INFO] Downloading: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(local_path, "wb") as f:
                f.write(r.read())
            print(f"[INFO] Saved: {local_path}")
            return local_path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download {local_path}: {last_err}")

def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]

def load_class_map(class_csv_path):
    rows = read_csv_rows(class_csv_path)
    label_to_name = {}
    for r in rows:
        if len(r) < 2: continue
        label_to_name[r[0]] = r[1]
    return label_to_name

def resolve_requested(classes_req, label_to_name, report_dir):
    resolved = {}
    unresolved = []
    ambiguous = []
    for raw in classes_req:
        q = raw.strip().casefold()
        exact = [lab for lab, disp in label_to_name.items() if disp.casefold() == q]
        if len(exact) == 1:
            disp = [v for v in label_to_name.values() if v.casefold() == q][0]
            resolved[disp] = exact[0]; continue
        matches = [(lab, disp) for lab, disp in label_to_name.items() if q in disp.casefold()]
        if len(matches) == 1:
            lab, disp = matches[0]; resolved[disp] = lab
        elif len(matches) > 1:
            matches.sort(key=lambda x: len(x[1]))
            lab, disp = matches[0]; resolved[disp] = lab
            ambiguous.append((raw, [m[1] for m in matches]))
        else:
            unresolved.append(raw)

    ensure_dir(report_dir)
    with open(os.path.join(report_dir, "resolved_map.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["Requested","ResolvedDisplay","LabelID"])
        # pick a representative "requested" string per resolved display for auditing
        for disp, lab in sorted(resolved.items(), key=lambda x: x[0].lower()):
            rq = None
            for rr in classes_req:
                if rr.strip().casefold() == disp.casefold() or rr.strip().casefold() in disp.casefold() or disp.casefold() in rr.strip().casefold():
                    rq = rr; break
            w.writerow([rq if rq else disp, disp, lab])
    with open(os.path.join(report_dir, "unresolved.txt"), "w", encoding="utf-8") as f:
        for u in unresolved: f.write(u+"\n")
    with open(os.path.join(report_dir, "ambiguous.txt"), "w", encoding="utf-8") as f:
        for raw, opts in ambiguous: f.write(raw+" -> "+"; ".join(opts)+"\n")

    print(f"[INFO] Resolved: {len(resolved)} | Unresolved: {len(unresolved)} | Ambiguous: {len(ambiguous)}")
    if unresolved:  print(f"[WARN] See {os.path.join(report_dir,'unresolved.txt')}")
    if ambiguous:   print(f"[WARN] See {os.path.join(report_dir,'ambiguous.txt')}")
    return resolved

def load_image_url_map(meta_csv_path):
    rows = read_csv_rows(meta_csv_path)
    header = rows[0]
    id_idx  = header.index("ImageID") if "ImageID" in header else 0
    url_idx = header.index("OriginalURL") if "OriginalURL" in header else 1
    m = {}
    for r in rows[1:]:
        if len(r) <= max(id_idx, url_idx): continue
        m[r[id_idx]] = r[url_idx]
    return m

def filtered_annotations(bbox_csv_path, target_labels_set):
    with open(bbox_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = row["LabelName"]
            if lab not in target_labels_set: continue
            try:
                xmin = float(row["XMin"]); xmax = float(row["XMax"])
                ymin = float(row["YMin"]); ymax = float(row["YMax"])
            except Exception:
                continue
            yield (row["ImageID"], lab, xmin, xmax, ymin, ymax)

def download_image(url, timeout=25):
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def crop_and_save(img_pil, bbox_norm, padding, min_side_frac, out_path):
    w, h = img_pil.size
    xmin, xmax, ymin, ymax = bbox_norm
    x1 = int(xmin * w); x2 = int(xmax * w)
    y1 = int(ymin * h); y2 = int(ymax * h)
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)
    min_side = min(w, h)
    if (bw / min_side) < min_side_frac and (bh / min_side) < min_side_frac:
        return False
    pad_x = int(padding * bw); pad_y = int(padding * bh)
    x1p = max(0, x1 - pad_x); y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x); y2p = min(h, y2 + pad_y)
    crop = img_pil.crop((x1p, y1p, x2p, y2p)).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    crop.save(out_path, "JPEG", quality=92)
    return True

def main():
    random.seed(SEED)
    out_root = os.path.abspath(OUT_DIR)
    tmp_dir = os.path.join(out_root, "_tmp"); ensure_dir(tmp_dir)
    work_dir = os.path.join(out_root, "_work"); ensure_dir(work_dir)

    # Download CSVs (small + stable)
    class_csv      = fetch_csv_any(URLS["class_desc"], os.path.join(tmp_dir, "class-descriptions-boxable.csv"))
    val_boxes_csv  = fetch_csv_any(URLS["val_boxes"], os.path.join(tmp_dir, "validation-annotations-bbox.csv"))
    test_boxes_csv = fetch_csv_any(URLS["test_boxes"], os.path.join(tmp_dir, "test-annotations-bbox.csv"))
    val_meta_csv   = fetch_csv_any(URLS["val_images"], os.path.join(tmp_dir, "validation-images-with-rotation.csv"))
    test_meta_csv  = fetch_csv_any(URLS["test_images"], os.path.join(tmp_dir, "test-images-with-rotation.csv"))

    # Resolve classes
    label_to_name = load_class_map(class_csv)
    report_dir = os.path.join(out_root, "_reports")
    resolved = resolve_requested(CLASS_NAMES, label_to_name, report_dir)
    if not resolved:
        print("[ERROR] None of the requested classes could be resolved. Edit CLASS_NAMES."); sys.exit(1)
    target_labels = set(resolved.values())

    # Image URL maps
    val_url_map  = load_image_url_map(val_meta_csv)
    test_url_map = load_image_url_map(test_meta_csv)

    # Per-class counters
    counters = {lab: 0 for lab in target_labels}

    def process_split(split_name, boxes_csv, url_map):
        total = 0; cache = {}
        for (img_id, lab, xmin, xmax, ymin, ymax) in filtered_annotations(boxes_csv, target_labels):
            if all(counters[l] >= PER_CLASS for l in target_labels): break
            if counters[lab] >= PER_CLASS: continue
            url = url_map.get(img_id); 
            if url is None: 
                continue
            img = cache.get(img_id)
            if img is None:
                img = download_image(url)
                if img is None: continue
                cache[img_id] = img
                # keep cache small
                if len(cache) > 12: cache.pop(next(iter(cache)))
            disp_name = [dn for dn, lid in resolved.items() if lid == lab][0]
            out_dir = os.path.join(work_dir, split_name, disp_name); ensure_dir(out_dir)
            fn = f"{img_id}_{counters[lab]:05d}.jpg"
            ok = crop_and_save(img, (xmin, xmax, ymin, ymax), PADDING, MIN_BOX_FRAC, os.path.join(out_dir, fn))
            if ok:
                counters[lab] += 1; total += 1
                if counters[lab] % 50 == 0:
                    print(f"[INFO] {split_name}:{disp_name} -> {counters[lab]} crops")
        print(f"[INFO] {split_name}: saved {total} crops")

    print("[INFO] Processing VALIDATION annotations ...")
    process_split("val", val_boxes_csv, val_url_map)
    need_more = [lab for lab in target_labels if counters[lab] < PER_CLASS]
    if need_more:
        print("[INFO] Topping up from TEST annotations ...")
        process_split("test", test_boxes_csv, test_url_map)

    # Merge work_dir into final train/val split
    final_train = os.path.join(out_root, "train"); final_val = os.path.join(out_root, "val")
    ensure_dir(final_train); ensure_dir(final_val)

    print("[INFO] Creating final train/val folders ...")
    disp_classes = sorted(resolved.keys())
    for disp_name in disp_classes:
        # collect all crops from val+test work folders
        imgs = []
        for split in ("val","test"):
            src = os.path.join(work_dir, split, disp_name)
            if os.path.isdir(src):
                imgs += [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(".jpg")]
        random.shuffle(imgs)
        if not imgs: 
            print(f"[WARN] No images collected for class '{disp_name}'")
            continue
        n_train = int(len(imgs) * TRAIN_RATIO)
        tr_list, va_list = imgs[:n_train], imgs[n_train:]
        dst_tr = os.path.join(final_train, disp_name); ensure_dir(dst_tr)
        dst_va = os.path.join(final_val,   disp_name); ensure_dir(dst_va)
        for s in tr_list: os.replace(s, os.path.join(dst_tr, os.path.basename(s)))
        for s in va_list: os.replace(s, os.path.join(dst_va, os.path.basename(s)))
        print(f"[INFO] {disp_name}: total {len(imgs)} → train {len(tr_list)}, val {len(va_list)}")

    # Write labels.txt
    with open(os.path.join(out_root, "labels.txt"), "w", encoding="utf-8") as f:
        for disp in disp_classes: f.write(disp + "\n")
    print(f"[INFO] labels.txt -> {os.path.join(out_root,'labels.txt')}")

    # Cleanup temp
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    print("[DONE] Dataset ready at:", os.path.abspath(out_root))
    print("       Reports in:", os.path.join(out_root, "_reports"))

if __name__ == "__main__":
    main()
