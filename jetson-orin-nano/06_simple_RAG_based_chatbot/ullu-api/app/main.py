import os, hashlib, time, json, re, math, uuid, sqlite3
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import httpx
import trafilatura
from duckduckgo_search import DDGS

DATA_DIR = os.getenv("ULLA_DATA_DIR", "/data")
SQLITE_PATH = os.getenv("SQLITE_PATH", f"{DATA_DIR}/ullu.db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://ullu-qdrant:6333")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.1:instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

MAX_LINKS_DEFAULT = int(os.getenv("MAX_LINKS_DEFAULT", "8"))
MAX_CHARS_PER_LINK = int(os.getenv("MAX_CHARS_PER_LINK", "20000"))
ITERATIONS_DEFAULT = int(os.getenv("ITERATIONS_DEFAULT", "2"))
MIN_LOCAL_SCORE = float(os.getenv("MIN_LOCAL_SCORE", "0.55"))

os.makedirs(DATA_DIR, exist_ok=True)

# --- sqlite tiny helpers (threads/memory/jobs) ---
def _db():
    con = sqlite3.connect(SQLITE_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""CREATE TABLE IF NOT EXISTS threads(
        id TEXT PRIMARY KEY, name TEXT, kb_paths TEXT, created_at REAL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS memory(
        thread_id TEXT, role TEXT, content TEXT, ts REAL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, status TEXT, params TEXT, result TEXT, started REAL, updated REAL
    )""")
    return con

# --- Qdrant ---
qc = QdrantClient(url=QDRANT_URL)

def kb_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]

def ensure_collection(col: str, dim: int = 768):
    exists = qc.collection_exists(col)
    if not exists:
        qc.create_collection(
            collection_name=col,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20000),
            quantization_config=qm.BinaryQuantization(False)
        )

def embed(texts: List[str]) -> List[List[float]]:
    # Batched embeddings via Ollama
    out = []
    B = 64
    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(texts), B):
            chunk = texts[i:i+B]
            r = client.post(f"{OLLAMA_BASE}/api/embeddings",
                            json={"model": EMBED_MODEL, "input": chunk})
            r.raise_for_status()
            out.extend([v["embedding"] for v in r.json()["data"]])
    return out

def chat_ollama(messages: List[Dict[str, str]], max_tokens=512, temperature=0.0) -> str:
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{OLLAMA_BASE}/api/chat",
                        json={
                            "model": GEN_MODEL,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            },
                        })
        r.raise_for_status()
        return r.json()["message"]["content"]

# --- File parsing (unstructured auto) ---
from unstructured.partition.auto import partition

def load_file_text(path: str) -> str:
    try:
        elements = partition(filename=path, strategy="hi_res")  # falls back if deps missing
        txt = "\n".join([e.text for e in elements if hasattr(e, "text") and e.text])
        return txt
    except Exception:
        return ""

def chunk(text: str, chunk_size=800, chunk_overlap=120):
    # simple word-chunker
    words = text.split()
    i = 0
    while i < len(words):
        yield " ".join(words[i:i+chunk_size])
        i += (chunk_size - chunk_overlap)

def index_dir(kb_path: str, col: str):
    ensure_collection(col)
    payloads, texts = [], []
    for root, _, files in os.walk(kb_path):
        for f in files:
            if not re.search(r"\.(pdf|txt|md|html?|docx?|pptx?|xlsx?)$", f, re.I):
                continue
            p = os.path.join(root, f)
            txt = load_file_text(p)
            if not txt.strip():
                continue
            for i, ch in enumerate(chunk(txt)):
                texts.append(ch)
                payloads.append({"src": p, "chunk": i})
            # batch upserts per file to keep RAM stable
            if len(texts) >= 256:
                vecs = embed(texts)
                qc.upsert(col, points=[qm.PointStruct(id=int(uuid.uuid4().int>>64),
                              vector=v, payload=pl) for v, pl in zip(vecs, payloads)])
                texts, payloads = [], []
    if texts:
        vecs = embed(texts)
        qc.upsert(col, points=[qm.PointStruct(id=int(uuid.uuid4().int>>64),
                        vector=v, payload=pl) for v, pl in zip(vecs, payloads)])

def search_local(cols: List[str], query: str, top_k=6) -> List[Dict[str, Any]]:
    vec = embed([query])[0]
    hits = []
    for c in cols:
        if not qc.collection_exists(c): 
            continue
        res = qc.search(c, query_vector=vec, limit=top_k)
        for r in res:
            hits.append({"score": r.score, **r.payload})
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]

def web_search_and_fetch(q: str, max_links=8, max_chars=20000, iterations=2):
    # DDG search + fetch with trafilatura; basic iterative refinement
    links = []
    query = q
    for it in range(iterations):
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_links):
                url = r.get("href") or r.get("url")
                if url and url not in links:
                    links.append(url)
        # simple refinement: append "pdf OR site:arxiv.org OR site:acm.org"
        query = q + " pdf OR site:arxiv.org OR site:acm.org"
    docs = []
    for url in links[:max_links]:
        try:
            html = trafilatura.fetch_url(url, no_ssl=True)
            txt = trafilatura.extract(html, include_comments=False) if html else ""
            if not txt:
                continue
            docs.append({"url": url, "text": txt[:max_chars]})
        except Exception:
            continue
    return docs

def build_citation_block(items):
    cits = []
    seen = set()
    for it in items:
        src = it.get("src") or it.get("url")
        if src and src not in seen:
            cits.append(f"- {src}")
            seen.add(src)
    return "\n".join(cits)

# --- API models ---
class IndexReq(BaseModel):
    kb_paths: List[str]

class QueryReq(BaseModel):
    prompt: str
    mode: Literal["local","web","hybrid","auto"] = "auto"
    kb_paths: List[str] = Field(default_factory=list)
    top_k: int = 6
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_local_score: float = MIN_LOCAL_SCORE
    web_max_links: int = MAX_LINKS_DEFAULT
    web_max_chars: int = MAX_CHARS_PER_LINK
    web_iterations: int = ITERATIONS_DEFAULT
    hybrid_local_weight: float = 0.7
    max_tokens: int = 512
    temperature: float = 0.0
    thread_id: Optional[str] = None
    persist_memory: bool = True
    memory_max_turns: int = 100

class ResearchReq(BaseModel):
    topic: str
    kb_paths: List[str] = Field(default_factory=list)
    time_budget_min: int = 10
    web_max_links: int = 30
    web_iterations: int = 3
    max_chars_per_link: int = 40000
    hybrid_local_weight: float = 0.5

app = FastAPI(title="ullu-api")

@app.post("/v1/index")
def do_index(req: IndexReq):
    for p in req.kb_paths:
        if not os.path.isdir(p):
            raise HTTPException(400, f"KB path not found: {p}")
    for p in req.kb_paths:
        col = f"ullu_{kb_id(p)}"
        index_dir(p, col)
    return {"status":"ok","indexed":[{"path":p,"collection":f"ullu_{kb_id(p)}"} for p in req.kb_paths]}

@app.get("/v1/kb/list")
def kb_list():
    # show mounted kb
    roots = []
    for root, dirs, files in os.walk("/mnt/kb"):
        roots.append(root)
    return {"kb_mount_root": "/mnt/kb", "paths": sorted(set([p for p in roots if p != "/mnt/kb"]))}

def save_memory(thread_id: str, role: str, content: str):
    if not thread_id: return
    con = _db()
    con.execute("INSERT INTO memory(thread_id,role,content,ts) VALUES(?,?,?,?)",
                (thread_id, role, content, time.time()))
    # trim
    cur = con.execute("SELECT COUNT(*) FROM memory WHERE thread_id=?", (thread_id,))
    n = cur.fetchone()[0]
    if n > 500:
        con.execute("""
            DELETE FROM memory WHERE rowid IN (
              SELECT rowid FROM memory WHERE thread_id=? ORDER BY ts ASC LIMIT ?
            )""",(thread_id, n-500))
    con.commit(); con.close()

def load_memory(thread_id: str, limit: int):
    if not thread_id: return []
    con = _db()
    cur = con.execute("SELECT role,content FROM memory WHERE thread_id=? ORDER BY ts ASC", (thread_id,))
    rows = cur.fetchall(); con.close()
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return [{"role":r[0], "content":r[1]} for r in rows]

@app.post("/v1/query")
def query(req: QueryReq):
    # prepare thread
    tid = req.thread_id or str(uuid.uuid4())
    # retrieve
    local_cols = [f"ullu_{kb_id(p)}" for p in req.kb_paths]
    local_hits = search_local(local_cols, req.prompt, top_k=req.top_k) if req.kb_paths else []
    best_local = (local_hits[0]["score"] if local_hits else 0.0)

    web_docs = []
    use_web = (req.mode in ["web","hybrid"]) or (req.mode=="auto" and best_local < req.min_local_score)
    if use_web:
        web_docs = web_search_and_fetch(req.prompt, req.web_max_links, req.web_max_chars, req.web_iterations)

    # assemble contexts
    local_ctx = [{"source":h["src"], "text": f"...{h['chunk']}... {h['src']}"} for h in local_hits]
    # re-pull text for selected local hits (fast minimal): (we saved only payload)
    # for accuracy, re-parse those files and slice chunks again:
    true_local_ctx=[]
    for h in local_hits:
        txt = load_file_text(h["src"])
        chunks = list(chunk(txt, req.chunk_size, req.chunk_overlap))
        if h["chunk"] < len(chunks):
            true_local_ctx.append({"source":h["src"], "text":chunks[h["chunk"]]})
    local_ctx = true_local_ctx

    # hybrid weighting (rank merge)
    combined = []
    if req.mode == "local":
        combined = local_ctx
    elif req.mode == "web":
        combined = [{"source":d["url"], "text":d["text"]} for d in web_docs]
    else:
        lw = max(0.0, min(1.0, req.hybrid_local_weight))
        L = [{"source":c["source"], "text":c["text"], "w": lw} for c in local_ctx]
        W = [{"source":d["url"], "text":d["text"], "w": 1.0-lw} for d in web_docs]
        combined = L + W

    # build prompt with strict anti-hallucination guard
    kb_names = ", ".join(req.kb_paths) if req.kb_paths else "(none)"
    citations = build_citation_block(combined)
    context_blobs = "\n\n".join([f"[{i+1}] {c['text']}" for i,c in enumerate(combined)])
    system = (
        "You are 'ullu', a precise assistant. Answer ONLY using the supplied context.\n"
        "If the answer is not fully supported by the context, say:\n"
        f"\"I couldn’t find this in the specified knowledge base(s): {kb_names}. "
        "Would you like me to search the web or another KB?\"\n"
        "Always include a Cited Sources section with the exact sources used."
    )
    user = f"Question: {req.prompt}\n\nContext:\n{context_blobs or '(no context)'}\n\n"
    msgs = [{"role":"system","content":system}]
    # include short-term memory for continuity
    if req.persist_memory:
        msgs += load_memory(tid, limit=req.memory_max_turns)
    msgs += [{"role":"user","content":user}]
    answer = chat_ollama(msgs, max_tokens=req.max_tokens, temperature=req.temperature)
    final = f"{answer}\n\n---\nCited Sources:\n{citations or '(none used)'}"

    if req.persist_memory:
        save_memory(tid, "user", req.prompt)
        save_memory(tid, "assistant", final)

    return {"thread_id": tid, "mode_used": ("local" if not use_web else ("hybrid" if local_ctx else "web")),
            "best_local_score": best_local, "answer": final}

@app.post("/v1/research")
def deep_research(req: ResearchReq, bt: BackgroundTasks):
    job_id = str(uuid.uuid4())
    con = _db()
    con.execute("INSERT INTO jobs(id,status,params,started,updated) VALUES(?,?,?,?,?)",
                (job_id,"running",json.dumps(req.model_dump()),time.time(),time.time()))
    con.commit(); con.close()

    def run():
        try:
            # naive breadth-first iterations; limit breadth by web_max_links
            docs = web_search_and_fetch(req.topic, req.web_max_links, req.max_chars_per_link, req.web_iterations)
            # optionally mix with local KB
            local_cols = [f"ullu_{kb_id(p)}" for p in req.kb_paths]
            local_hits = search_local(local_cols, req.topic, top_k=20) if req.kb_paths else []
            local_texts=[]
            for h in local_hits:
                txt = load_file_text(h["src"])
                chunks = list(chunk(txt))
                if h["chunk"] < len(chunks):
                    local_texts.append({"source":h["src"], "text":chunks[h["chunk"]]})
            lw = req.hybrid_local_weight
            combined = [{"source":d["url"], "text":d["text"]} for d in docs] + local_texts

            context = "\n\n".join([f"- {c['source']}\n{c['text'][:2000]}" for c in combined])
            prompt = (f"Topic: {req.topic}\nSynthesize a structured research brief with sections: "
                      "Key Findings, Contradictions, Open Questions, Data/Benchmarks, "
                      "Implementation Pointers, Citations (with URLs). Use only the provided notes.\n\nNotes:\n"+context)
            report = chat_ollama([
                {"role":"system","content":"You produce rigorous, source-grounded research briefs."},
                {"role":"user","content":prompt}
            ], max_tokens=1200, temperature=0.1)

            con = _db()
            con.execute("UPDATE jobs SET status=?, result=?, updated=? WHERE id=?",
                        ("done", report, time.time(), job_id))
            con.commit(); con.close()
        except Exception as e:
            con = _db()
            con.execute("UPDATE jobs SET status=?, result=?, updated=? WHERE id=?",
                        ("error", f"{type(e).__name__}: {e}", time.time(), job_id))
            con.commit(); con.close()

    bt.add_task(run)
    return {"job_id": job_id, "status": "running"}

@app.get("/v1/research/{job_id}")
def research_status(job_id: str):
    con = _db()
    cur = con.execute("SELECT status,result,params,started,updated FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    con.close()
    if not row: raise HTTPException(404, "job not found")
    status, result, params, started, updated = row
    return {"job_id":job_id, "status":status, "result":result, "params":json.loads(params or "{}"),
            "started":started, "updated":updated}

