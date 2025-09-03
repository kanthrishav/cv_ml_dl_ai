import os
import io
import json
import time
import uuid
from typing import List, Optional, Dict, Any

import numpy as np
import faiss
import httpx

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()

DATA_DIR = "/data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "meta.json")

# -------------------------- Logging
def log(*args):
    if LOG_LEVEL in ("debug", "info"):
        print(*args, flush=True)

# -------------------------- Embeddings
from sentence_transformers import SentenceTransformer
_model = None

def get_model():
    global _model
    if _model is None:
        log("[api] Loading embedding model:", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    if not isinstance(vecs, np.ndarray):
        vecs = np.array(vecs, dtype="float32")
    return vecs.astype("float32")

# -------------------------- Store
class Store:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.meta: Dict[str, Any] = {"docs": [], "vectors": []}  # vectors: [ {doc_id, chunk_id, text, source, ts} ... ]
        self.dim: Optional[int] = None
        self._load()

    def _load(self):
        if os.path.exists(INDEX_PATH):
            log("[api] Loading FAISS index")
            self.index = faiss.read_index(INDEX_PATH)
            self.dim = self.index.d
        else:
            self.index = None
            self.dim = None

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {"docs": [], "vectors": []}

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @property
    def n_vectors(self) -> int:
        return len(self.meta["vectors"])

    @property
    def n_docs(self) -> int:
        return len(self.meta["docs"])

    def reset(self):
        log("[api] Resetting store")
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        self.index = None
        self.dim = None
        self.meta = {"docs": [], "vectors": []}

    def add(self, filename: str, chunks: List[str], embeddings: np.ndarray):
        assert len(chunks) == embeddings.shape[0], "chunks/embeddings length mismatch"

        if self.index is None:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)

        start_id = self.n_vectors
        self.index.add(embeddings)

        doc_id = str(uuid.uuid4())
        now = int(time.time())
        self.meta["docs"].append({
            "doc_id": doc_id,
            "filename": filename,
            "n_chunks": len(chunks),
            "created_at": now
        })

        for i, text in enumerate(chunks):
            self.meta["vectors"].append({
                "doc_id": doc_id,
                "chunk_id": i,
                "source": filename,
                "text": text,
                "ts": now
            })

        self._save()
        log(f"[api] Added doc={filename} chunks={len(chunks)} total_vectors={self.n_vectors}")

    def search(self, query: str, top_k: int = 3):
        if self.index is None or self.n_vectors == 0:
            return []

        q_vec = embed_texts([query])  # normalized
        D, I = self.index.search(q_vec, min(top_k, self.n_vectors))
        idxs = I[0].tolist()
        scores = D[0].tolist()

        results = []
        for ridx, score in zip(idxs, scores):
            if ridx < 0 or ridx >= self.n_vectors:
                continue
            meta = self.meta["vectors"][ridx]
            results.append({
                "score": float(score),
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
            })
        return results

store = Store()

# -------------------------- Chunking & parsing
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap) if (chunk_size - overlap) > 0 else chunk_size
    return chunks

def load_file_to_text(up: UploadFile) -> str:
    name = (up.filename or "upload").lower()
    content = up.file.read()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            text.append(t)
        return "\n".join(text)
    else:
        # treat as text (txt, md, csv, etc.)
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")

# -------------------------- LLM (Ollama) or fallback
async def ollama_generate(prompt: str, model: str = "llama3", temperature: float = 0.2, max_tokens: int = 512) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
    except Exception as e:
        log("[api] ollama error:", e)
        return ""

# -------------------------- FastAPI app & schemas
app = FastAPI(title="ullu-api", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

class GenerateRequest(BaseModel):
    query: str
    rag: bool = True
    top_k: int = 3
    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: Optional[str] = None

class GenerateResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    used_rag: bool
    top_k: int

@app.get("/health")
def health():
    return {"status": "ok", "docs": store.n_docs, "chunks": store.n_vectors}

@app.get("/echo")
def echo(text: str):
    return {"ok": True, "echo": text}

@app.post("/rag/index")
async def rag_index(files: List[UploadFile] = File(...)):
    total_chunks = 0
    total_files = 0
    for f in files:
        text = load_file_to_text(f)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        vecs = embed_texts(chunks)
        store.add(filename=f.filename or "upload", chunks=chunks, embeddings=vecs)
        total_chunks += len(chunks)
        total_files += 1
    return {"status": "ok", "indexed_files": total_files, "indexed_chunks": total_chunks, "stats": {"docs": store.n_docs, "chunks": store.n_vectors}}

@app.post("/rag/reset")
async def rag_reset():
    store.reset()
    return {"status": "ok"}

@app.get("/rag/stats")
async def rag_stats():
    return {"status": "ok", "docs": store.n_docs, "chunks": store.n_vectors}

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    context = []
    citations = []
    if req.rag:
        results = store.search(req.query, top_k=req.top_k)
        for r in results:
            context.append(f"[{r['source']}|chunk {r['chunk_id']}|score {r['score']:.3f}]\n{r['text']}")
            citations.append({
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "preview": r["text"][:300]
            })

    # Build prompt
    system = req.system_prompt or "You are a helpful assistant. Use the provided context if relevant. Cite sources by filename and chunk."
    context_text = "\n\n".join(context) if context else "No external context."
    user_prompt = f"{system}\n\n# Context\n{context_text}\n\n# Task\nUser query: {req.query}\n\nAnswer:"
    answer = await ollama_generate(user_prompt, model=req.model, temperature=req.temperature, max_tokens=req.max_tokens)

    if not answer:
        # Fallback (no LLM reachable): return stitched context
        answer = "LLM unavailable. Here are the most relevant context snippets:\n\n" + "\n\n---\n\n".join(c["preview"] for c in citations) if citations else "LLM unavailable and no RAG data."

    return GenerateResponse(answer=answer, citations=citations, used_rag=req.rag, top_k=req.top_k)

