from __future__ import annotations
import os, re, hashlib
from typing import Iterable, Dict, Any, List, Tuple
from bs4 import BeautifulSoup

# Simple heuristics: turn HTML/Markdown/plain into text sections of ~N chars

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]

def read_file_to_text(path: str) -> Tuple[str, str]:
    """
    Returns (mimetype, text)
    """
    low = path.lower()
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return ("error", "")

    # detect encoding
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        import chardet
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        text = raw.decode(enc, "ignore")

    if low.endswith((".html", ".htm")):
        soup = BeautifulSoup(text, "lxml")
        # remove scripts/styles/navs
        for t in soup(["script","style","nav","header","footer","noscript"]):
            t.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else os.path.basename(path))
        body = soup.get_text("\n", strip=True)
        return ("text/html", f"# {title}\n\n{body}")
    elif low.endswith((".md",".markdown",".mdown",".mkd")):
        # keep raw markdown; our chunker splits by headings
        return ("text/markdown", text)
    else:
        return ("text/plain", text)

def split_into_chunks(text: str, max_chars: int = 1500, min_chars: int = 200) -> List[str]:
    """
    Heading-aware split: break on markdown headings or HTML-like headings first,
    then enforce size windows.
    """
    # normalize line endings
    text = text.replace("\r\n","\n").replace("\r","\n")

    # pre-split on headings
    parts: List[str] = re.split(r"(?m)^(#{1,6}\s+.*)$", text)
    # re-glue: keep heading with following content
    blocks: List[str] = []
    i = 0
    while i < len(parts):
        if parts[i].startswith("#"):
            head = parts[i].strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            blocks.append(head + "\n" + body)
            i += 2
        else:
            if parts[i].strip():
                blocks.append(parts[i])
            i += 1

    # now size windows
    chunks: List[str] = []
    buf = ""
    for b in blocks:
        if len(buf) + len(b) <= max_chars:
            buf += ("\n\n" + b) if buf else b
        else:
            if buf:
                if len(buf) >= min_chars:
                    chunks.append(buf.strip())
                else:
                    # if too small, try to merge anyway
                    buf += ("\n\n" + b)
                    chunks.append(buf.strip())
                    buf = ""
            else:
                # a single huge block: hard wrap
                for i in range(0, len(b), max_chars):
                    segment = b[i:i+max_chars].strip()
                    if segment:
                        chunks.append(segment)
                buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

def doc_to_records(source_id: str, root: str, path: str, max_chars: int, min_chars: int) -> List[Dict[str,Any]]:
    mtype, text = read_file_to_text(path)
    if not text.strip():
        return []
    rel = os.path.relpath(path, root)
    chunks = split_into_chunks(text, max_chars=max_chars, min_chars=min_chars)
    records: List[Dict[str,Any]] = []
    for idx, ch in enumerate(chunks):
        rec_id = _hash(f"{source_id}/{rel}#{idx}")
        title = ch.splitlines()[0].strip() if ch.strip() else os.path.basename(path)
        records.append({
            "id": rec_id,
            "source": source_id,
            "relpath": rel,
            "mimetype": mtype,
            "chunk_index": idx,
            "title": title[:200],
            "text": ch,
        })
    return records

