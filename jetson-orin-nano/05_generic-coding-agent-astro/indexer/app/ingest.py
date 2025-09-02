from __future__ import annotations
import os, tarfile, tempfile, shutil
from typing import List, Dict, Any, Iterable
from .chunker import doc_to_records

SUPPORTED_EXT = (
    ".html",".htm",".md",".markdown",".mdown",".mkd",".txt",".rst"
)

def safe_walk_files(root: str) -> Iterable[str]:
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("_static","_images","_build","node_modules",".git")]
        for f in files:
            low = f.lower()
            if low.endswith(SUPPORTED_EXT):
                yield os.path.join(base, f)
            elif low.endswith((".tgz",".tar.gz")):
                # try to expand Zeal/Dash docset archives on the fly
                full = os.path.join(base, f)
                with tempfile.TemporaryDirectory() as td:
                    try:
                        with tarfile.open(full, "r:gz") as tf:
                            tf.extractall(td)
                        for path in safe_walk_files(td):
                            yield path
                    except Exception:
                        continue

def harvest_to_records(source_id: str, path: str, max_chars: int, min_chars: int) -> List[Dict[str,Any]]:
    records: List[Dict[str,Any]] = []
    for p in safe_walk_files(path):
        records.extend(doc_to_records(source_id, path, p, max_chars, min_chars))
    return records

