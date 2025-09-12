#!/usr/bin/env python3
"""
Make captions from image filenames.

Example:
  "mp_mark__standing__hitting_a_ball.jpg"
    -> "mp_mark, standing, hitting a ball"

Default input:  /workspace/datasets/crops_full/<person>/*.jpg|png|webp|...
Default output: /workspace/datasets/captions/<same-stem>.txt

Notes:
- We DO NOT add or infer anything beyond the filename.
- We skip obvious junk (signature/logo/autograph) if present in filename.
- Captions are saved using the image *stem* + ".txt" to match Kohya's --caption_extension .txt.
"""

from pathlib import Path
import argparse

SKIP_SUBSTR = ("signature", "logo", "autograph")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

def caption_from_stem(stem: str) -> str:
    # split on double underscore, then replace single underscores with spaces
    parts = stem.split("__")
    cleaned = [p.replace("_", " ").strip() for p in parts if p.strip()]
    return ", ".join(cleaned)

def main():
    ap = argparse.ArgumentParser(description="Generate captions (.txt) from image filenames")
    ap.add_argument("--images_root", default="/workspace/datasets/crops_full",
                    help="Root folder containing subfolders of images")
    ap.add_argument("--out_dir", default="/workspace/datasets/captions",
                    help="Where to write caption .txt files")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing .txt files if they exist")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print actions without writing files")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        raise SystemExit(f"[FATAL] images_root not found: {images_root}")

    created, skipped = 0, 0

    # Walk one level of person folders; also tolerate flat layouts
    candidates = []
    subdirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    if subdirs:
        for d in subdirs:
            candidates.extend(sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]))
    else:
        candidates.extend(sorted([p for p in images_root.iterdir() if p.suffix.lower() in IMG_EXTS]))

    if not candidates:
        print(f"[warn] no images under {images_root}")
        return

    for imgp in candidates:
        low = imgp.name.lower()
        if any(s in low for s in SKIP_SUBSTR):
            skipped += 1
            continue

        # caption filename: same STEM as image, with .txt (Kohya-compatible)
        txt_path = out_dir / f"{imgp.stem}.txt"
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue

        cap = caption_from_stem(imgp.stem)
        if args.dry_run:
            print(f"[dry] {imgp.name} -> {txt_path.name}: {cap}")
        else:
            txt_path.write_text(cap, encoding="utf-8")
            created += 1

    print(f"[ok] captions written: {created}, skipped: {skipped}")
    print(f"[out] {out_dir}")

if __name__ == "__main__":
    main()

