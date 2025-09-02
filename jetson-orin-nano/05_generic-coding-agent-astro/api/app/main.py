"""
astro-api (Step 3): LLM wired to Ollama + real runner.

Endpoints:
- GET  /health
- GET  /config
- GET  /ui                        (tiny HTML to exercise API)
- GET  /llm/ping
- POST /llm/complete             {prompt, model?}
- POST /run                      {problem, project_name, use_model?, max_iterations?}
- GET  /stream/{run_id}          (SSE tail of run log)
- GET  /runs/{run_id}
- POST /control/kill-run
- POST /control/offload-model
- POST /control/stop-container   (kills the api process)

This is a minimal, testable backend. Later steps will expand retries,
RAG, git, doxygen, multi-language, etc.
"""

from __future__ import annotations
import os
import json
import time
import uuid
import threading
from typing import Dict, Any, Callable, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

from . import llm_client, runner

app = FastAPI()

# ---------- config ----------
WORKSPACES_ROOT = os.environ.get("WORKSPACES_ROOT", "/ssd/agents/astro/workspaces")
RUN_LOG_DIR = os.environ.get("RUN_LOG_DIR", "/ssd/agents/astro/logs")
DEVICE_DEFAULT = os.environ.get("DEVICE_DEFAULT", "gpu")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.2:3b")
ALLOW_CONTAINER_STOP = os.environ.get("ALLOW_CONTAINER_STOP", "true").lower() == "true"
TRTLLM_ADMIN = os.environ.get("TRTLLM_ADMIN", "http://trtllm-admin:8090")

os.makedirs(WORKSPACES_ROOT, exist_ok=True)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# ---------- run state ----------
_runs_lock = threading.Lock()
_runs: Dict[str, Dict[str, Any]] = {}  # run_id -> info


def _log_writer(log_path: str) -> Callable[[str], None]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def _log(line: str) -> None:
        ts = time.strftime("[%H:%M:%S] ", time.localtime())
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts}{line}\n")
    return _log


def _tail_log_lines(log_path: str, stop_flag: Callable[[], bool]):
    # yield existing lines first, then follow
    last_pos = 0
    while not os.path.exists(log_path):
        time.sleep(0.05)
    with open(log_path, "r", encoding="utf-8") as f:
        # existing
        for line in f:
            yield f"data: {line.rstrip()}\n\n"
        last_pos = f.tell()

    # follow until run stops and file has no more data
    while True:
        if stop_flag():
            # flush whatever remains once more
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                rest = f.read()
                if rest:
                    for line in rest.splitlines():
                        yield f"data: {line}\n\n"
            break
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            chunk = f.read()
            if chunk:
                for line in chunk.splitlines():
                    yield f"data: {line}\n\n"
                last_pos = f.tell()
        time.sleep(0.15)


# ---------- routes ----------

@app.get("/health")
def health():
    return {"status": "ok", "api": "astro-api (Step 3 LLM wired)"}

@app.get("/config")
def config():
    return {
        "workspaces_root": WORKSPACES_ROOT,
        "run_log_dir": RUN_LOG_DIR,
        "device_default": DEVICE_DEFAULT,
        "allow_container_stop": ALLOW_CONTAINER_STOP,
        "default_model": DEFAULT_MODEL,
    }

@app.get("/ui", response_class=HTMLResponse)
def ui():
    # Tiny barebones page to run a job and hit controls.
    return """
<!doctype html><meta charset="utf-8">
<title>astro-api</title>
<style>
  body { font: 14px system-ui, sans-serif; margin: 24px; max-width: 880px; }
  label { display:block; margin: 8px 0 4px; }
  textarea,input { width: 100%; }
  .row { display:flex; gap:12px; margin: 10px 0; }
  .row > button { flex:1; padding:10px; }
  #out { white-space: pre; background:#111; color:#ddd; padding:10px; border-radius:6px; height: 280px; overflow:auto; }
</style>
<h1>astro-api</h1>
<section>
  <h3>Run a task</h3>
  <label>Problem</label>
  <textarea id="problem" rows="4">Create a CLI that converts Celsius to Fahrenheit with tests</textarea>
  <div class="row">
    <div style="flex:2">
      <label>Project name</label>
      <input id="project" value="temp_cli">
    </div>
    <div style="flex:1">
      <label>Model</label>
      <input id="model" value="llama3.2:3b">
    </div>
  </div>
  <button onclick="runTask()">Run</button>
  <div id="runmeta"></div>
  <div id="out"></div>
</section>
<section>
  <h3>Controls</h3>
  <div class="row">
    <button onclick="hit('/control/kill-run')">Kill current run</button>
    <button onclick="hit('/control/offload-model')">Offload model</button>
    <button onclick="hit('/control/stop-container')">Stop API container</button>
  </div>
  <div id="ctrl"></div>
</section>
<script>
async function runTask(){
  const problem = document.getElementById('problem').value;
  const project = document.getElementById('project').value;
  const model = document.getElementById('model').value;
  const r = await fetch('/run',{method:'POST',headers:{'content-type':'application/json'},
    body: JSON.stringify({problem: problem, project_name: project, use_model: model, max_iterations: 1})});
  const j = await r.json();
  document.getElementById('runmeta').textContent = JSON.stringify(j,null,2);
  if(j.run_id){
    const sse = new EventSource('/stream/'+j.run_id);
    const out = document.getElementById('out');
    sse.onmessage = (ev)=>{ out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
  }
}
async function hit(path){
  const r = await fetch(path,{method:'POST'});
  document.getElementById('ctrl').textContent = await r.text();
}
</script>
"""

@app.get("/llm/ping")
def llm_ping():
    return llm_client.ping()

class CompleteBody(BaseModel := type("CompleteBody",(object,),{})):  # small trick to avoid pydantic import here
    def __init__(self, prompt: str, model: Optional[str] = None):
        self.prompt = prompt
        self.model = model

@app.post("/llm/complete")
async def llm_complete(req: Request):
    body = await req.json()
    prompt = body.get("prompt","").strip()
    model = body.get("model") or DEFAULT_MODEL
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")
    return llm_client.complete(prompt=prompt, model=model)

@app.post("/run")
async def start_run(req: Request):
    body = await req.json()
    max_iterations = int(body.get("max_iterations", 5))
    problem = (body.get("problem") or "").strip()
    project = (body.get("project_name") or "").strip()
    model = (body.get("use_model") or DEFAULT_MODEL).strip()
    if not problem or not project:
        raise HTTPException(status_code=400, detail="Require 'problem' and 'project_name'")

    run_id = uuid.uuid4().hex[:8]
    log_path = os.path.join(RUN_LOG_DIR, f"run_{run_id}.log")
    logger = _log_writer(log_path)

    info = {
        "run_id": run_id,
        "project_name": project,
        "workspace": os.path.join(WORKSPACES_ROOT, project),
        "status": "running",
        "summary": None,
        "started_at": time.time(),
        "finished_at": None,
        "log_path": log_path,
        "_cancel": False,
    }
    with _runs_lock:
        _runs[run_id] = info

    def cancel_flag() -> bool:
        with _runs_lock:
            return _runs.get(run_id, {}).get("_cancel", False)

    def worker():
        try:
            logger(f"Workspace: {info['workspace']}")
            logger(f"Problem: {problem}")
            res = runner.run(
                problem=problem,
                project_name=project,
                model=model,
                workspaces_root=WORKSPACES_ROOT,
                log=logger,
                cancel_flag=cancel_flag,
                max_iterations=max_iterations,
            )
            with _runs_lock:
                info["status"] = res.get("status","failed")
                info["summary"] = res.get("summary")
        except Exception as e:  # pragma: no cover
            logger(f"Runner crashed: {type(e).__name__}: {e}")
            with _runs_lock:
                info["status"] = "failed"
                info["summary"] = "Runner exception."
        finally:
            with _runs_lock:
                info["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id, "workspace": info["workspace"], "status": "running"}

from fastapi import Body

@app.post("/tasks")
async def tasks_compat(body: dict = Body(...)):
    """
    Compatibility endpoint for older examples:
    Accepts: {"project": "...", "task": "...", "use_model"?, "max_iterations"?}
    Mirrors /run behavior.
    """
    problem = (body.get("task") or body.get("problem") or "").strip()
    project = (body.get("project") or body.get("project_name") or "").strip()
    model = (body.get("use_model") or DEFAULT_MODEL).strip()
    max_iterations = int(body.get("max_iterations", 5))

    if not problem or not project:
        raise HTTPException(status_code=400, detail="Require 'project'/'project_name' and 'task'/'problem'")

    run_id = uuid.uuid4().hex[:8]
    log_path = os.path.join(RUN_LOG_DIR, f"run_{run_id}.log")
    logger = _log_writer(log_path)

    info = {
        "run_id": run_id,
        "project_name": project,
        "workspace": os.path.join(WORKSPACES_ROOT, project),
        "status": "running",
        "summary": None,
        "started_at": time.time(),
        "finished_at": None,
        "log_path": log_path,
        "_cancel": False,
    }
    with _runs_lock:
        _runs[run_id] = info

    def cancel_flag() -> bool:
        with _runs_lock:
            return _runs.get(run_id, {}).get("_cancel", False)

    def worker():
        try:
            logger(f"Workspace: {info['workspace']}")
            logger(f"Problem: {problem}")
            res = runner.run(
                problem=problem,
                project_name=project,
                model=model,
                workspaces_root=WORKSPACES_ROOT,
                log=logger,
                cancel_flag=cancel_flag,
                max_iterations=max_iterations,
            )
            with _runs_lock:
                info["status"] = res.get("status","failed")
                info["summary"] = res.get("summary")
        except Exception as e:
            logger(f"Runner crashed: {type(e).__name__}: {e}")
            with _runs_lock:
                info["status"] = "failed"
                info["summary"] = "Runner exception."
        finally:
            with _runs_lock:
                info["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id, "workspace": info["workspace"], "status": "running"}


@app.get("/stream/{run_id}")
def stream(run_id: str):
    with _runs_lock:
        info = _runs.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    def stopped():
        with _runs_lock:
            s = _runs.get(run_id, {}).get("status")
        return s in ("done","failed","cancelled")

    generator = _tail_log_lines(info["log_path"], stop_flag=stopped)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/runs/{run_id}")
def run_status(run_id: str):
    with _runs_lock:
        info = _runs.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return {
        "run_id": run_id,
        "project_name": info["project_name"],
        "workspace": info["workspace"],
        "status": info["status"],
        "summary": info["summary"],
        "started_at": info["started_at"],
        "finished_at": info["finished_at"],
    }

@app.post("/control/kill-run")
def kill_run():
    n = 0
    with _runs_lock:
        for r in _runs.values():
            if r["status"] == "running":
                r["_cancel"] = True
                n += 1
    return {"ok": True, "message": f"Cancelled {n} run(s)."}

@app.post("/control/offload-model")
def offload_model():
    try:
        r = httpx.post(f"{TRTLLM_ADMIN}/offload", timeout=2.0)
        return {"ok": True, "message": "Requested model offload.", "detail": {"trtllm_admin_offload_status": r.status_code, "body": r.text}}
    except Exception as e:
        return {"ok": True, "message": "Requested model offload.", "detail": {"error": f"{type(e).__name__}: {e}"}}

@app.post("/control/stop-container")
def stop_container():
    if not ALLOW_CONTAINER_STOP:
        raise HTTPException(status_code=403, detail="Container stop disabled")
    # FastAPI/uvicorn friendly exit
    os._exit(0)

