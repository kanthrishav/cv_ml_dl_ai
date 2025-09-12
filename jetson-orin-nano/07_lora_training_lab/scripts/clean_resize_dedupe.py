#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
from typing import Tuple
import cv2
from PIL import Image
import imagehash
import numpy as np

RAW = Path("/workspace/datasets/raw")
OUT = Path("/workspace/datasets/crops_full")  # new output dir 

def dedupe_key(p: Path):
    try:
        h = imagehash.average_hash(Image.open(p).convert("RGB"))
        return f"{h}-{p.stat().st_size}"
    except Exception:
        return None

def center_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    return img[y0:y0+m, x0:x0+m]

def pad_color(img: np.ndarray, mode: str = "median") -> Tuple[int, int, int]:
    """
    Determine padding color:
      - median (default): median color of image (robust to outliers)
      - black / white
      - edge: replicate edge pixels when padding (handled separately)
    """
    if mode == "black":
        return (0, 0, 0)
    if mode == "white":
        return (255, 255, 255)
    if mode == "median":
        med = np.median(img.reshape(-1, 3), axis=0)
        return tuple(int(x) for x in med[::-1])  # cv2 uses BGR
    # fallback
    return (0, 0, 0)

def letterbox(img: np.ndarray, res: int, border: str = "median") -> np.ndarray:
    """
    Scale the *longer* side to res, preserve aspect, then pad to res x res.
    Keeps entire content (no cropping).
    """
    h, w = img.shape[:2]
    scale = res / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (res - new_h) // 2
    bottom = res - new_h - top
    left = (res - new_w) // 2
    right = res - new_w - left

    if border == "edge":
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    color = pad_color(resized, border)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def fit_then_center_crop(img: np.ndarray, res: int) -> np.ndarray:
    """
    Scale so *shorter* side == res, then center-crop to res x res (may crop edges).
    """
    h, w = img.shape[:2]
    scale = res / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return center_crop_to_square(resized)[:res, :res]

def normalize(img: np.ndarray, res: int, mode: str, border: str) -> np.ndarray:
    if mode == "letterbox":
        return letterbox(img, res, border)
    elif mode == "center_crop_square":
        cropped = center_crop_to_square(img)
        return cv2.resize(cropped, (res, res), interpolation=cv2.INTER_AREA)
    elif mode == "fit_then_center_crop":
        return fit_then_center_crop(img, res)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def process(res: int, mode: str, border: str, min_side: int, quality: int):
    OUT.mkdir(parents=True, exist_ok=True)
    count_in, count_out = 0, 0

    for person_dir in sorted(RAW.iterdir()):
        if not person_dir.is_dir():
            continue
        outdir = OUT / person_dir.name
        outdir.mkdir(parents=True, exist_ok=True)
        seen = set()

        for imgp in sorted(person_dir.glob("*")):
            ext = imgp.suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
                continue

            img = cv2.imread(str(imgp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            count_in += 1

            # Skip tiny sources to avoid upscaling junk
            if min(img.shape[:2]) < min_side:
                continue

            canvas = normalize(img, res, mode, border)
            out_file = outdir / f"{imgp.stem}_{res}.jpg"
            ok = cv2.imwrite(str(out_file), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                continue

            # de-dup
            key = dedupe_key(out_file)
            if key is None or key in seen:
                try:
                    out_file.unlink()
                except Exception:
                    pass
            else:
                seen.add(key)
                count_out += 1

    print(f"[done] processed: {count_in} images → saved: {count_out} images at {res}×{res} under {OUT}")

def main():
    ap = argparse.ArgumentParser(description="Full-frame resize + pad (no face crop) with de-dup")
    ap.add_argument("--res", type=int, default=512, help="Output size (square)")
    ap.add_argument("--mode", type=str, default="letterbox",
                    choices=["letterbox", "center_crop_square", "fit_then_center_crop"],
                    help="Resize strategy (default: letterbox to keep full image)")
    ap.add_argument("--border", type=str, default="median",
                    choices=["median", "black", "white", "edge"],
                    help="Padding color strategy for letterbox (default: median)")
    ap.add_argument("--min-side", type=int, default=256,
                    help="Skip images with min(h,w) < this (default: 256)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    args = ap.parse_args()
    process(args.res, args.mode, args.border, args.min_side, args.quality)

if __name__ == "__main__":
    main()

