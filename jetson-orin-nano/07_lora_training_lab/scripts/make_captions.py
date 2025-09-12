#!/usr/bin/env python3
from pathlib import Path
import re

CROP = Path("/workspace/datasets/crops")
CAPS = Path("/workspace/datasets/captions"); CAPS.mkdir(parents=True, exist_ok=True)

def token_for_folder(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+","_", name.lower()).strip("_")
    return f"rk_{base}"

BASE_TAGS = "clean lines, detailed shading, high quality"
SKIP_SUBSTR = ("signature", "logo", "autograph")

l = [p for p in CROP.iterdir() if p.is_dir()]
for person_dir in sorted(l):
    token = token_for_folder(person_dir.name)
    for img in sorted(person_dir.glob("*.jpg")):
        low = img.name.lower()
        if any(s in low for s in SKIP_SUBSTR):
            continue
        cap = f"{token}, {person_dir.name.replace('_',' ')}, {BASE_TAGS}"
        (CAPS / (img.stem + ".txt")).write_text(cap)
print("[ok] captions regenerated as .txt files.")
