import os, time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

app = FastAPI(title="trtllm-admin-stub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_STATE = {"loaded": False, "model_name": None, "ts": None}

@app.get("/health")
def health():
    return {"status": "ok", "stub": True, "models_dir": MODELS_DIR, "state": MODEL_STATE}

@app.post("/load")
def load(model_name: str = "dev-3b"):
    # Step 2: here we'll load a real TensorRT-LLM engine
    MODEL_STATE.update({"loaded": True, "model_name": model_name, "ts": time.time()})
    return {"ok": True, "message": f"Loaded {model_name} (stub)", "state": MODEL_STATE}

@app.post("/unload")
def unload():
    MODEL_STATE.update({"loaded": False, "model_name": None, "ts": time.time()})
    # Step 2: here well tear down the real session and free GPU memory
    return {"ok": True, "message": "Unloaded model (stub)", "state": MODEL_STATE}

