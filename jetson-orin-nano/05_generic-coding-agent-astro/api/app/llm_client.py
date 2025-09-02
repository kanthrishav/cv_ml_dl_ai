"""
Minimal Ollama client wrapper used by astro-api.

- Respects OLLAMA_HOST from environment so the container can talk
  to the host's Ollama (we inject host.docker.internal in compose).
- Provides simple .ping() and .complete() helpers.
"""

from __future__ import annotations
import os
import traceback
from typing import Dict, Any

try:
    import ollama  # pip install ollama
except Exception as e:  # pragma: no cover
    ollama = None
    _import_err = e
else:
    _import_err = None


def _host() -> str:
    # The python client uses env OLLAMA_HOST (http://host:11434 by default)
    return os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")


def ping() -> Dict[str, Any]:
    """
    Try listing models from Ollama to verify connectivity.
    """
    if ollama is None:
        return {
            "ok": False,
            "error": f"python 'ollama' import failed: {_import_err!r}",
        }
    try:
        ollama_client = ollama.Client(host=_host())
        models = ollama_client.list()
        names = []
        for m in models.get("models", []):
            # Some clients return "name", others return "model"
            names.append(m.get("name") or m.get("model") or m.get("digest"))
        return {"ok": True, "host": _host(), "model_count": len(names), "models": names}
    except Exception as e:
        return {
            "ok": False,
            "host": _host(),
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def complete(prompt: str, model: str, temperature: float = 0.2, keep_alive: str = "5m") -> Dict[str, Any]:
    """
    Synchronous completion via ollama.generate(...).
    Returns a dict with keys: ok, text, raw (ollama response) or error.
    """
    if ollama is None:
        return {"ok": False, "error": f"python 'ollama' import failed: {_import_err!r}"}

    try:
        client = ollama.Client(host=_host())
        resp = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
            keep_alive=keep_alive,
            stream=False,
        )
        text = (resp or {}).get("response", "")
        return {"ok": True, "text": text, "raw": resp}
    except Exception as e:
        return {
            "ok": False,
            "host": _host(),
            "model": model,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

