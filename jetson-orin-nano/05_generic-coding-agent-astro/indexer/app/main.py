from uuid import uuid5, NAMESPACE_URL
import os
import re
import gc
import json
import time
import math
import asyncio
import hashlib
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Iterable

from fastapi import FastAPI, Query
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import httpx
from bs4 import BeautifulSoup

from qdrant_client import QdrantClient
from qdrant_client.http import models as qd
from qdrant_client.conversions.common_types import Record

# Meilisearch is optional; treat as best-effort
try:
    import meilisearch  # type: ignore
except Exception:  # pragma: no cover
    meilisearch = None

# Embeddings
from fastembed import TextEmbedding

# ---------------------------
# Environment / Config
# ---------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").strip()
QDRANT_HOST = os.getenv("QDRANT_HOST", "").strip() or "qdrant"
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks").strip()
QDRANT_RESET_ON_DIM_MISMATCH = os.getenv("QDRANT_RESET_ON_DIM_MISMATCH", "1") == "1"

MEILI_URL = os.getenv("MEILI_URL", "http://meilisearch:7700").strip()
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "").strip()
MEILI_INDEX = os.getenv("MEILI_INDEX", "kb_chunks").strip()

KB_ROOT = os.getenv("KB_ROOT", "/kb").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()

# Web settings
WEB_TIMEOUT = float(os.getenv("WEB_TIMEOUT", "10"))  # per-request hard timeout
WEB_BUDGET_SEC = float(os.getenv("WEB_BUDGET_SEC", "12"))  # overall search budget
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "12"))
WEB_FETCH_LIMIT = int(os.getenv("WEB_FETCH_LIMIT", "12"))  # cap total fetches
WEB_MAX_PER_DOMAIN = int(os.getenv("WEB_MAX_PER_DOMAIN", "4"))
WEB_USER_AGENT = os.getenv(
    "WEB_USER_AGENT",
    "Mozilla/5.0 (compatible; astro-indexer/1.0; +https://example.invalid)",
).strip()
WEB_ALLOWLIST = [
    d.strip()
    for d in os.getenv(
        "WEB_ALLOWLIST",
        "go.dev,pkg.go.dev,docs.python.org,doc.rust-lang.org,learn.microsoft.com,pytorch.org,numpy.org,pandas.pydata.org,nodejs.org"
    ).split(",")
    if d.strip()
]

SEARCH_ENGINES_RAW = os.getenv("SEARCH_ENGINES", "bing_lite,ddg_html,ddg,ddg_lite").lower().split(",")
SEARCH_ENGINES = [e.strip() for e in SEARCH_ENGINES_RAW if e.strip()]

# ---------------------------
# App
# ---------------------------

app = FastAPI(title="Astro Indexer", version="0.4.0-phase3-final")


# HELPERS

def _collection_vector_size() -> Optional[int]:
    try:
        info = _qdrant.get_collection(QDRANT_COLLECTION)
        cfg = getattr(info, "config", None)
        if not cfg or not getattr(cfg, "params", None):
            return None
        vectors = cfg.params.vectors
        if isinstance(vectors, qd.VectorParams):
            return vectors.size
        if isinstance(vectors, dict):
            # named vectors -> take first one
            for v in vectors.values():
                return v.size
    except Exception:
        return None

def _ensure_qdrant_dim_ok():
    try:
        size = _collection_vector_size()
        if size is None:
            return
        if _EMB_DIM and size != _EMB_DIM:
            if QDRANT_RESET_ON_DIM_MISMATCH:
                _qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qd.VectorParams(size=_EMB_DIM, distance=qd.Distance.COSINE),
                    optimizers_config=qd.OptimizersConfigDiff(indexing_threshold=10000),
                )
            else:
                raise RuntimeError(f"Qdrant vector size {size} != embedder dim {_EMB_DIM}")
    except Exception as e:
        # Re-raise so caller returns JSON error
        raise

def _looks_like_search_results(url: str) -> bool:
    u = url.lower()
    return ("pkg.go.dev/search" in u) or ("duckduckgo.com" in u) or ("bing.com/search" in u)


# ---------------------------
# Embeddings
# ---------------------------

_embedder = TextEmbedding(model_name=EMBED_MODEL)
try:
    _probe_vec = next(iter(_embedder.embed(["probe"])), [])
except Exception:
    _probe_vec = []
_EMB_DIM = len(_probe_vec) if isinstance(_probe_vec, (list, tuple)) else 0

# ---------------------------
# Stores (Qdrant / Meili)
# ---------------------------

_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, url=None)

def _ensure_qdrant() -> None:
    exists = False
    try:
        exists = _qdrant.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False
    if not exists:
        _qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qd.VectorParams(size=_EMB_DIM, distance=qd.Distance.COSINE),
            optimizers_config=qd.OptimizersConfigDiff(indexing_threshold=10000),
        )
    else:
        _ensure_qdrant_dim_ok()

_ensure_qdrant()

# Meilisearch best-effort
_meili = None
try:
    if meilisearch is not None:
        _meili = meilisearch.Client(MEILI_URL, MEILI_KEY or None)
        # lazily create
        try:
            _meili.get_index(MEILI_INDEX)
        except Exception:
            _meili.create_index(MEILI_INDEX, {"primaryKey": "id"})
except Exception:
    _meili = None

# ---------------------------
# Simple LRU cache for web pages
# ---------------------------

class LRUCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.d: Dict[str, Tuple[float, str]] = {}
        self.q: deque[str] = deque()

    def get(self, k: str) -> Optional[str]:
        v = self.d.get(k)
        if not v:
            return None
        # move to end (MRU)
        try:
            self.q.remove(k)
        except ValueError:
            pass
        self.q.append(k)
        return v[1]

    def set(self, k: str, v: str) -> None:
        if k in self.d:
            try:
                self.q.remove(k)
            except ValueError:
                pass
        self.d[k] = (time.time(), v)
        self.q.append(k)
        while len(self.q) > self.max_size:
            old = self.q.popleft()
            self.d.pop(old, None)

    def clear(self):
        self.d.clear()
        self.q.clear()


_WEB_CACHE = LRUCache(max_size=512)

# ---------------------------
# Metrics
# ---------------------------

METRICS = {
    "web": {
        "fetch_attempts": 0,
        "fetched_ok": 0,
        "timeouts": 0,
        "errors": 0,
        "cache_hits": 0,
        "dedup_dropped": 0,
        "rank_scored": 0,
        "engine_hits": defaultdict(int),
        "site_hits": defaultdict(int),
        "last_budget_sec": 0.0,
    },
    "ingest": {
        "files_indexed": 0,
        "chunks_indexed": 0,
        "last_duration_sec": 0.0,
    },
}

def _metrics_snapshot():
    # return JSON-serializable snapshot
    web = METRICS["web"]
    return {
        "web": {
            "fetch_attempts": web["fetch_attempts"],
            "fetched_ok": web["fetched_ok"],
            "timeouts": web["timeouts"],
            "errors": web["errors"],
            "cache_hits": web["cache_hits"],
            "dedup_dropped": web["dedup_dropped"],
            "rank_scored": web["rank_scored"],
            "engine_hits": dict(web["engine_hits"]),
            "site_hits": dict(web["site_hits"]),
            "last_budget_sec": web["last_budget_sec"],
        },
        "ingest": dict(METRICS["ingest"]),
    }

# ---------------------------
# Utilities
# ---------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _split_chunks(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        va = a[i]
        vb = b[i]
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

# ---------------------------
# Ingest (filesystem under /kb)
# ---------------------------

TEXT_EXTS = {".md", ".txt", ".rst", ".html", ".htm"}
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar"}

def _iter_source_files(root: str) -> Iterable[str]:
    for base, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            ext = os.path.splitext(p)[1].lower()
            if ext in BINARY_EXTS:
                continue
            yield p

def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:
        return ""

def _rel_path(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return os.path.basename(path)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    # fastembed returns a generator of numpy arrays (but its fine to handle as list)
    vecs = list(_embedder.embed(texts))
    # ensure list of python lists
    out: List[List[float]] = []
    for v in vecs:
        if hasattr(v, "tolist"):
            out.append(v.tolist())
        else:
            out.append(list(v))
    return out

def _qdrant_upsert(points: List[qd.PointStruct]) -> None:
    if not points:
        return
    _qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
        wait=True,
    )

# ---------------------------
# Web fetch / search
# ---------------------------

HEADERS = {
    "User-Agent": WEB_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _allowed(url: str, allowlist: List[str]) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    except Exception:
        return False
    for d in allowlist:
        d = d.lower()
        if host == d or host.endswith("." + d):
            return True
    return False

async def _http_get_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    # Cache
    cached = _WEB_CACHE.get(url)
    if cached is not None:
        METRICS["web"]["cache_hits"] += 1
        return cached
    METRICS["web"]["fetch_attempts"] += 1
    try:
        r = await client.get(url, headers=HEADERS, timeout=WEB_TIMEOUT, follow_redirects=True)
        if r.status_code >= 200 and r.status_code < 400:
            text = r.text or ""
            _WEB_CACHE.set(url, text)
            METRICS["web"]["fetched_ok"] += 1
            return text
        else:
            METRICS["web"]["errors"] += 1
            return None
    except httpx.TimeoutException:
        METRICS["web"]["timeouts"] += 1
        return None
    except Exception:
        METRICS["web"]["errors"] += 1
        return None

def _extract_text_basic(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # kill scripts/styles/navs
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalize
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _norm_ws(text)
    return text

# --- Site-specific search helpers (fast + reliable) ---

async def _search_pkg_go_dev(client: httpx.AsyncClient, q: str, limit: int) -> List[str]:
    # Example: https://pkg.go.dev/search?q=generics%20type%20parameters
    url = f"https://pkg.go.dev/search?q={httpx.QueryParams({'q': q})['q']}"
    html = await _http_get_text(client, url)
    out: List[str] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://pkg.go.dev" + href
        if href.startswith("https://pkg.go.dev/"):
            out.append(href)
        if len(out) >= limit:
            break
    return out

async def _search_docs_python(client: httpx.AsyncClient, q: str, limit: int) -> List[str]:
    # Example: https://docs.python.org/3/search.html?q=venv+activate
    url = "https://docs.python.org/3/search.html"
    params = {"q": q}
    try:
        r = await client.get(url, params=params, headers=HEADERS, timeout=WEB_TIMEOUT, follow_redirects=True)
        html = r.text if r.status_code == 200 else ""
    except Exception:
        html = ""
    out: List[str] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # absolute or relative
        if href.startswith("/3/"):
            href = "https://docs.python.org" + href
        if href.startswith("https://docs.python.org/"):
            out.append(href)
        if len(out) >= limit:
            break
    return out

# Fallback: lightweight engine scrapers (best-effort)
async def _bing_lite(client: httpx.AsyncClient, q: str, sites: List[str], limit: int) -> List[str]:
    # Use Bing with site: filters; parse result anchors
    site_filter = " OR ".join([f"site:{s}" for s in sites]) if sites else ""
    q2 = f"{q} {site_filter}".strip()
    url = "https://www.bing.com/search"
    params = {"q": q2}
    try:
        r = await client.get(url, params=params, headers=HEADERS, timeout=WEB_TIMEOUT, follow_redirects=True)
        html = r.text if r.status_code == 200 else ""
    except Exception:
        html = ""
    out: List[str] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Filter bing redirectors
        if "bing.com/ck/a" in href or "go.microsoft.com" in href:
            continue
        if href.startswith("http"):
            out.append(href)
        if len(out) >= limit:
            break
    return out

async def _ddg_html(client: httpx.AsyncClient, q: str, sites: List[str], limit: int) -> List[str]:
    site_filter = " ".join([f"site:{s}" for s in sites]) if sites else ""
    q2 = f"{q} {site_filter}".strip()
    url = "https://html.duckduckgo.com/html/"
    params = {"q": q2}
    try:
        r = await client.post(url, data=params, headers=HEADERS, timeout=WEB_TIMEOUT, follow_redirects=True)
        html = r.text if r.status_code == 200 else ""
    except Exception:
        html = ""
    out: List[str] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            out.append(href)
        if len(out) >= limit:
            break
    return out

async def _site_search(client: httpx.AsyncClient, q: str, sites: List[str], per_site: int) -> Tuple[List[str], Dict[str, int]]:
    urls: List[str] = []
    stats: Dict[str, int] = defaultdict(int)
    tasks = []

    for s in sites:
        s = s.lower()
        if s == "pkg.go.dev":
            tasks.append(_search_pkg_go_dev(client, q, per_site))
        elif s == "docs.python.org":
            tasks.append(_search_docs_python(client, q, per_site))
        else:
            # no explicit handler; skip for now (better than hanging)
            tasks.append(asyncio.sleep(0, result=[]))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for site, lst in zip(sites, results):
        if isinstance(lst, list):
            stats[site] += len(lst)
            urls.extend(lst)

    return urls, stats

async def _engine_search(client: httpx.AsyncClient, q: str, sites: List[str], limit: int) -> Tuple[List[str], Dict[str, int]]:
    engine_funcs = {
        "bing_lite": _bing_lite,
        "ddg_html": _ddg_html,
        # ddg / ddg_lite variants can be wired here if desired; keep two for reliability
    }
    engine_stats: Dict[str, int] = defaultdict(int)
    urls: List[str] = []
    tasks = []
    for e in SEARCH_ENGINES:
        fn = engine_funcs.get(e)
        if fn:
            tasks.append(fn(client, q, sites, limit))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for e, lst in zip([e for e in SEARCH_ENGINES if engine_funcs.get(e)], results):
        if isinstance(lst, list):
            engine_stats[e] += len(lst)
            urls.extend(lst)
    return urls, engine_stats

# domain semaphores for concurrency control
_domain_semaphores: Dict[str, asyncio.Semaphore] = {}

def _domain_of(url: str) -> str:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        return host
    except Exception:
        return "unknown"

def _semaphore_for(url: str) -> asyncio.Semaphore:
    host = _domain_of(url)
    if host not in _domain_semaphores:
        _domain_semaphores[host] = asyncio.Semaphore(WEB_MAX_PER_DOMAIN)
    return _domain_semaphores[host]

async def _fetch_and_extract(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
    # returns (url, extracted_text or "")
    sem = _semaphore_for(url)
    async with sem:
        html = await _http_get_text(client, url)
    if not html:
        return url, ""
    text = _extract_text_basic(html)
    return url, text

async def _web_collect(q: str, sites: List[str], limit: int) -> Dict[str, Any]:
    """
    Budgeted, concurrency-capped web gather; returns debug info and extracted docs.
    """
    started = time.time()
    timeout_budget = WEB_BUDGET_SEC

    async with httpx.AsyncClient(http2=False, timeout=WEB_TIMEOUT, headers=HEADERS) as client:
        # Site-specific search first (reliable + targeted)
        per_site = max(3, min(12, limit))
        site_urls, site_stats = await _site_search(client, q, sites, per_site)

        # Engines (best-effort)
        engine_urls, engine_stats = await _engine_search(client, q, sites, WEB_MAX_RESULTS)

        raw_urls = engine_urls
        allowed_urls = [u for u in raw_urls if _allowed(u, WEB_ALLOWLIST)]
        # prefer site_urls at the front
        candidate_urls = list(dict.fromkeys(site_urls + allowed_urls))  # dedup while preserving order

        # Hard cap fetch count
        candidate_urls = candidate_urls[:WEB_FETCH_LIMIT]

        # Fetch with overall budget (cooperative)
        docs: Dict[str, str] = {}
        fetch_tasks = []
        candidate_urls = [u for u in candidate_urls if not _looks_like_search_results(u)]
        for u in candidate_urls:
            fetch_tasks.append(_fetch_and_extract(client, u))

        done, pending = await asyncio.wait(
            fetch_tasks,
            timeout=timeout_budget,
            return_when=asyncio.ALL_COMPLETED,
        )

        # cancel leftovers
        for p in pending:
            try:
                p.cancel()
            except Exception:
                pass

        for d in done:
            try:
                url, text = d.result()
            except Exception:
                url, text = "", ""
            if url and text:
                docs[url] = text

        METRICS["web"]["engine_hits"].update(engine_stats)
        for k, v in site_stats.items():
            METRICS["web"]["site_hits"][k] += v

        elapsed = time.time() - started
        METRICS["web"]["last_budget_sec"] = round(elapsed, 3)

        return {
            "raw_urls": raw_urls,
            "allowed_urls": allowed_urls,
            "site_urls": site_urls,
            "site_stats": site_stats,
            "engine_stats": engine_stats,
            "docs": docs,
        }

def _rank_web_docs(q_vec: List[float], docs: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
    # Very light heuristic boost: prefer docs/overview/readme-ish pages for package sites
    scored: List[Tuple[float, str, str]] = []
    texts: List[str] = []
    urls: List[str] = []
    for u, t in docs.items():
        urls.append(u)
        texts.append(t)

    # Embed once; then compute cosine
    if not texts:
        return []
    # chunk each doc lightly to improve matching (first N chunks only)
    items: List[Tuple[str, int, str]] = []  # (url, chunk_idx, text)
    for u, t in zip(urls, texts):
        chunks = _split_chunks(t, max_chars=1000, overlap=120)
        # cap chunks per doc
        chunks = chunks[:4]
        for i, c in enumerate(chunks):
            items.append((u, i, c))

    emb = _embed_texts([c for (_u, _i, c) in items])
    # rank
    for (u, i, c), v in zip(items, emb):
        base = _cosine(q_vec, v)
        boost = 0.0
        # heuristic boosts
        if "pkg.go.dev" in u:
            if "tab=doc" in u or u.endswith("/doc") or "/overview" in u:
                boost += 0.05
        if "docs.python.org" in u:
            if "/tutorial/" in u or "/library/" in u:
                boost += 0.03
        score = base + boost
        scored.append((score, u, c))

    # de-dup across near-identical chunks by URL + first 120 chars hash
    seen_keys = set()
    dedup: List[Tuple[float, str, str]] = []
    for sc, u, c in sorted(scored, key=lambda x: x[0], reverse=True):
        key = (u, _sha1(c[:120]))
        if key in seen_keys:
            METRICS["web"]["dedup_dropped"] += 1
            continue
        seen_keys.add(key)
        dedup.append((sc, u, c))

    METRICS["web"]["rank_scored"] += len(dedup)
    out: List[Dict[str, Any]] = []
    for sc, u, c in dedup[:limit]:
        out.append({
            "score": float(sc),
            "source": f"web@{_domain_of(u)}",
            "rel_path": u,
            "chunk_index": 0,
            "text": c[:2000],  # avoid huge payloads
        })
    return out

# ---------------------------
# API Models
# ---------------------------

class IngestResp(BaseModel):
    ok: bool
    source: str
    files_seen: int
    files_indexed: int
    chunks_indexed: int
    duration_sec: float

class HealthResp(BaseModel):
    status: str
    index: str
    collection: str
    kb_root: str

# ---------------------------
# Endpoints
# ---------------------------

@app.exception_handler(Exception)
async def _all_exceptions(_req: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@app.get("/health", response_model=HealthResp)
async def health():
    return HealthResp(
        status="ok",
        index=QDRANT_COLLECTION,
        collection=QDRANT_COLLECTION,
        kb_root=KB_ROOT,
    )

@app.get("/counts")
async def counts():
    q_count = None
    try:
        info = _qdrant.get_collection(QDRANT_COLLECTION)
        if info and info.points_count is not None:
            q_count = info.points_count
    except Exception:
        q_count = None

    m_count = None
    if _meili:
        try:
            idx = _meili.get_index(MEILI_INDEX)
            stats = idx.get_stats()
            m_count = stats.get("numberOfDocuments")
        except Exception:
            m_count = None

    return {"qdrant": q_count, "meili": m_count}

@app.get("/sources")
async def sources():
    out = []
    try:
        for name in os.listdir(KB_ROOT):
            p = os.path.join(KB_ROOT, name)
            if os.path.isdir(p):
                out.append(name)
    except Exception:
        pass
    return {"sources": sorted(out)}

@app.post("/ingest", response_model=IngestResp)
async def ingest(
    source: str = Query(..., description="Path inside /kb, e.g. 'tldr' or 'zeal_docsets/extracted/Python'"),
    max_files: int = Query(200, ge=1, le=10000),
    max_chunks: int = Query(20, ge=1, le=200),
):
    t0 = time.time()
    try:
        _ensure_qdrant()           # ensure exists
        _ensure_qdrant_dim_ok()    # ensure dimension matches embedder

        root = os.path.join(KB_ROOT, source)
        if not os.path.exists(root):
            return JSONResponse({"ok": False, "error": f"Source not found under /kb: {source}"}, status_code=404)

        files_seen = files_indexed = chunks_indexed = 0
        points: List[qd.PointStruct] = []
        batch_size = 128

        for i, path in enumerate(_iter_source_files(root)):
            files_seen += 1
            if files_indexed >= max_files:
                break
            txt = _read_file(path)
            if not txt:
                continue
            chunks = _split_chunks(txt, max_chars=1200, overlap=150)[:max_chunks]
            if not chunks:
                continue
            vecs = _embed_texts(chunks)
            rp = _rel_path(path, KB_ROOT)
            for j, (c, v) in enumerate(zip(chunks, vecs)):
                pid = str(uuid5(NAMESPACE_URL, f"{source}::{rp}::{j}"))
                payload = {"source": source.split("/")[0], "rel_path": rp, "chunk_index": j, "text": c}
                points.append(qd.PointStruct(id=pid, vector=v, payload=payload))
                chunks_indexed += 1

            files_indexed += 1
            if len(points) >= batch_size:
                _qdrant_upsert(points)
                points = []

        if points:
            _qdrant_upsert(points)

        METRICS["ingest"]["files_indexed"] += files_indexed
        METRICS["ingest"]["chunks_indexed"] += chunks_indexed
        METRICS["ingest"]["last_duration_sec"] = round(time.time() - t0, 3)

        gc.collect()

        return IngestResp(
            ok=True,
            source=source,
            files_seen=files_seen,
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            duration_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/search")
async def search(
    q: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
    web: Optional[int] = Query(None, description="1 enables web fallback"),
    web_only: Optional[int] = Query(None, description="1 = web search only"),
    min_local_score: float = Query(0.0),
    sites: Optional[str] = Query(None, description="comma-separated allowlist for web search"),
):
    # 1) local
    hits: List[Dict[str, Any]] = []
    q_vec = _embed_texts([q])[0] if q else [0.0] * _EMB_DIM

    if not web_only:
        try:
            res = _qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q_vec,
                limit=limit * 2,  # fetch a bit more, then filter
                with_payload=True,
                score_threshold=min_local_score if min_local_score > 0 else None,
            )
            for p in res:
                pl = p.payload or {}
                hits.append({
                    "score": float(p.score or 0.0),
                    "source": str(pl.get("source", "")),
                    "rel_path": str(pl.get("rel_path", "")),
                    "chunk_index": int(pl.get("chunk_index", 0)),
                    "text": str(pl.get("text", ""))[:2000],
                })
        except Exception:
            hits = []

    # already enough?
    if hits and (len(hits) >= limit or not web and not web_only):
        return {"hits": hits[:limit]}

    # 2) web fallback
    if web_only or web:
        site_list = []
        if sites:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
        else:
            # default: use global allowlist
            site_list = WEB_ALLOWLIST[:]

        info = await _web_collect(q, site_list, limit)
        docs = info.get("docs", {})
        web_hits = _rank_web_docs(q_vec, docs, limit)
        return {"hits": (web_hits[:limit] if web_only else (hits + web_hits)[:limit])}

    return {"hits": hits[:limit]}

@app.get("/web_debug")
async def web_debug(
    q: str = Query(...),
    sites: Optional[str] = Query(None),
):
    site_list = [s.strip() for s in sites.split(",")] if sites else WEB_ALLOWLIST[:]
    info = await _web_collect(q, site_list, limit=WEB_MAX_RESULTS)
    # Strip docs body to keep response small
    d = dict(info)
    d["docs"] = {k: f"[{len(v)} chars]" for k, v in (info.get("docs") or {}).items()}
    d["allowlist"] = site_list
    return JSONResponse(d)

@app.get("/netcheck")
async def netcheck():
    out = {"outbound_ok": False, "engines": {}, "site_search": {}}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("https://example.com", timeout=5.0)
            out["outbound_ok"] = r.status_code == 200
    except Exception:
        out["outbound_ok"] = False

    # Engines quick probe
    try:
        async with httpx.AsyncClient(timeout=WEB_TIMEOUT, headers=HEADERS) as client:
            eng_urls, eng_stats = await _engine_search(client, "site:docs.python.org venv", ["docs.python.org"], 5)
            out["engines"] = {
                "ddg": {"ok": "true" if eng_stats.get("ddg", 0) > 0 else "false", "note": f"{eng_stats.get('ddg', 0)}"},
                "ddg_html": {"ok": "true" if eng_stats.get("ddg_html", 0) > 0 else "false", "note": f"{eng_stats.get('ddg_html', 0)}"},
                "ddg_lite": {"ok": "false", "note": "n/a"},
                "bing_lite": {"ok": "true" if eng_stats.get("bing_lite", 0) > 0 else "false", "note": f"{eng_stats.get('bing_lite', 0)}"},
            }
            # Site search probes
            sites = ["go.dev", "pkg.go.dev", "docs.python.org", "pytorch.org", "numpy.org", "pandas.pydata.org", "doc.rust-lang.org", "nodejs.org", "learn.microsoft.com"]
            site_urls, site_stats = await _site_search(client, "generics type parameters", sites, 5)
            out["site_search"] = {s: {"ok": "true" if site_stats.get(s, 0) > 0 else "false", "note": f"raw={site_stats.get(s, 0)} allowed={site_stats.get(s, 0)}"} for s in sites}
    except Exception:
        pass

    return JSONResponse(out)

@app.get("/metrics")
async def metrics():
    return JSONResponse(_metrics_snapshot())

@app.post("/web_cache/clear")
async def web_cache_clear():
    _WEB_CACHE.clear()
    return {"ok": True}

# ---------------------------
# Startup logs
# ---------------------------

@app.on_event("startup")
async def on_startup():
    print(f"[startup] Loading embedding model: {EMBED_MODEL}", flush=True)
    print(f"[startup] Embedding dimension = {_EMB_DIM}", flush=True)
    try:
        info = _qdrant.get_collection(QDRANT_COLLECTION)
        if info:
            print(f"[startup] Using existing Qdrant collection '{QDRANT_COLLECTION}'", flush=True)
    except Exception:
        print(f"[startup] Creating Qdrant collection '{QDRANT_COLLECTION}'", flush=True)
        _ensure_qdrant()
    if _meili:
        try:
            _meili.get_indexes()
            print("[startup] Meilisearch reachable.", flush=True)
        except Exception:
            print("[startup] Meilisearch not reachable (continuing).", flush=True)

# ---------------------------
# Root/help
# ---------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return (
        "astro-indexer online\n"
        "Endpoints: /health /counts /sources /ingest /search /web_debug /netcheck /metrics /web_cache/clear\n"
    )

