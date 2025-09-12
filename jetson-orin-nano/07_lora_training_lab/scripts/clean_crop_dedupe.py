#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
import cv2
import imagehash

RAW = Path("/workspace/datasets/raw")
CROP = Path("/workspace/datasets/crops")
CROP.mkdir(parents=True, exist_ok=True)

RES = 512

def find_haar_xml() -> str | None:
    candidates = []
    # If cv2 includes a data module with haarcascades dir
    if hasattr(cv2, "data") and getattr(cv2.data, "haarcascades", None):
        candidates.append(cv2.data.haarcascades)
    # Common distro paths
    candidates += [
        "/usr/share/opencv4/haarcascades/",
        "/usr/share/opencv/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
    ]
    for base in candidates:
        xml = os.path.join(base, "haarcascade_frontalface_default.xml")
        if os.path.isfile(xml):
            return xml
    return None

FACE_XML = find_haar_xml()
detector = cv2.CascadeClassifier(FACE_XML) if FACE_XML else None

def dedupe_key(p: Path):
    try:
        h = imagehash.average_hash(Image.open(p).convert("RGB"))
        return f"{h}-{p.stat().st_size}"
    except Exception:
        return None

def center_crop_to_square(img):
    h, w = img.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    return img[y0:y0+m, x0:x0+m]

for person_dir in RAW.iterdir():
    if not person_dir.is_dir():
        continue
    outdir = CROP / person_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    seen = set()

    for imgp in sorted(person_dir.glob("*")):
        img = cv2.imread(str(imgp), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Try face detection if cascade is available; else fallback
        roi = None
        if detector is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                pad = int(0.3 * max(w, h))
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
                roi = img[y0:y1, x0:x1]

        if roi is None:
            # No cascade or no face found: center-crop to square (skip tiny)
            if min(img.shape[:2]) < 200:
                continue
            roi = center_crop_to_square(img)
	
        # Resize to fit within RES while preserving aspect, then pad to RES×RES
        h0, w0 = roi.shape[:2]
        scale = RES / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (RES - new_h) // 2; bottom = RES - new_h - top
        left = (RES - new_w) // 2; right = RES - new_w - left
        canvas = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        out_file = outdir / (imgp.stem + "_"+str(RES)+".jpg")
        cv2.imwrite(str(out_file), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # De-duplicate
        key = dedupe_key(out_file)
        if key is None or key in seen:
            try:
                out_file.unlink()
            except Exception:
                pass
        else:
            seen.add(key)
