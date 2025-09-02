# /ssd/agents/astro/api/app/runner.py  (OVERWRITE WHOLE FILE)
"""
LLM-driven runner (Phase 4, monorepo + venv pool + sanitization).

Task-agnostic behavior:
- Monorepo: Git root is WORKSPACES_ROOT; tasks live as subdirs under it.
- Safe writes under src/, tests/, docs/, scripts/, and a few top-level files only.
- Sanitization: strip stray 'path=...' lines and ``` inside file bodies.
- Venv pool reuse/creation with meaningful names (e.g., py310-numpy-1.26.4__pandas-2.2.2).
- Generic auto-heal:
  * If tests import a different local package name, create an alias package.
  * If pytest reports ModuleNotFoundError for a dotted local submodule (pkg.sub),
    synthesize a stub module (and stub names if something imports from it).
  * If circular-import or packaging code creeps into __init__.py, neutralize it.
  * Third-party missing modules are solved via the shared venv pool.
- RAG: retrieve helpful snippets (if indexer is available) and inject into plan/fix prompts.
"""

from __future__ import annotations
import os
import re
import sys
import json
import shlex
import uuid
import shutil
import subprocess
from typing import Callable, Dict, List, Tuple, Optional

from . import llm_client

# ---------- constants ----------
# venv pool must live on a writable in-container path (overridable via compose env)
VENV_POOL_ROOT = os.environ.get("VENV_POOL_ROOT", "/workspaces/.venv-pool")

# Prefer explicit RAG_SEARCH_URL; else /search under KB_INDEX_URL; else default.
RAG_SEARCH_URL = (
    os.environ.get("RAG_SEARCH_URL")
    or (os.environ.get("KB_INDEX_URL", "").rstrip("/") + "/search" if os.environ.get("KB_INDEX_URL") else "http://astro-indexer:8383/search")
)

_ALLOWED_TOP_DIRS = {"src", "tests", "docs", "scripts"}
_ALLOWED_TOP_FILES = {"README.md", "requirements.txt", "Doxyfile", ".gitignore"}

# stdlib names we will NOT auto-install even if an ImportError appears
_STDLIB_DENY = {
    "sys","os","re","json","math","time","pathlib","subprocess","shutil","typing","unittest","dataclasses",
    "logging","itertools","functools","collections","statistics","abc","enum","asyncio","http","email",
    "urllib","socket","multiprocessing","threading","sqlite3","tkinter","fractions","decimal","io","hashlib",
    "gzip","bz2","lzma","zipfile","tarfile","tempfile","random","secrets","heapq","bisect","array","typing_extensions",
    "src"
}

# ```python path=...``` fence capture
FENCE_WITH_PATH = re.compile(
    r"```[^\n`]*\s*path\s*=\s*([^\n`]+)\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# ```\n# path: ...\n``` capture
FENCE_WITH_HASH_PATH = re.compile(
    r"```[^\n`]*\n#\s*path\s*:\s*([^\n`]+)\n(.*?)```",
    re.DOTALL | re.IGNORECASE
)

# dotted names like demo_hello.greet
_MNF_RE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_\.]+)'")

# detect circular-import text from pytest/importlib
_CIRCULAR_RE = re.compile(
    r"partially initialized module .* most likely due to a circular import",
    re.IGNORECASE
)

# packaging smell inside __init__.py that breaks pytest collection
_INIT_PKG_SMELL = re.compile(r"\b(setup|setuptools|distutils)\b", re.IGNORECASE)

# ---------- small utils ----------
def _has_git() -> bool:
    return shutil.which("git") is not None

def _mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _run(cmd: List[str], cwd: str, env: Dict[str, str], log: Callable[[str], None]) -> Tuple[int, str, str]:
    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as e:
        # surface immediately with a helpful message
        log(f"EXEC ERROR: {e}")
        raise
    out, err = proc.communicate()
    if out:
        for line in out.splitlines(): log(line)
    if err:
        for line in err.splitlines(): log(line)
    return proc.returncode, out, err

def _is_allowed_rel_path(rel_path: str) -> bool:
    rel = rel_path.strip().lstrip("/").replace("\\", "/")
    top = rel.split("/", 1)[0]
    return (top in _ALLOWED_TOP_DIRS) or (rel in _ALLOWED_TOP_FILES)

def _sanitize_content(rel_path: str, content: str) -> str:
    # nuke stray 'path=...' lines and accidental backticks inside code bodies
    lines = content.replace("\r\n", "\n").splitlines()
    cleaned: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("```") or re.match(r"^\s*path\s*=\s*.+$", s):
            continue
        cleaned.append(ln)
    txt = "\n".join(cleaned)
    if txt and not txt.endswith("\n"):
        txt += "\n"
    return txt

def _write_file(base: str, rel_path: str, content: str, log: Optional[Callable[[str], None]] = None) -> str:
    rel_path = rel_path.strip().lstrip("/").replace("\r\n", "\n")
    if rel_path in {"__file__", ".", ".."} or not _is_allowed_rel_path(rel_path):
        if log: log(f"SKIP write (disallowed path): {rel_path}")
        return os.path.join(base, "_skipped_", rel_path.replace("/", "_"))
    content = _sanitize_content(rel_path, content)
    abs_path = os.path.normpath(os.path.join(base, rel_path))
    if not abs_path.startswith(os.path.abspath(base) + os.sep):
        raise ValueError(f"refusing to write outside workspace: {rel_path}")
    _mkdirp(os.path.dirname(abs_path))
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path

def _parse_files_from_llm(markdown: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []

    def _clean(p: str) -> str:
        # remove surrounding quotes if the model emits path='...'/path="..."
        return p.strip().strip("'").strip('"')

    for m in FENCE_WITH_PATH.finditer(markdown):
        files.append((_clean(m.group(1)), m.group(2)))

    if not files:
        for m in FENCE_WITH_HASH_PATH.finditer(markdown):
            files.append((_clean(m.group(1)), m.group(2)))
    return files


# ---------- venv helpers ----------
def _py_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"

def _venv_bins(venv_dir: str) -> Dict[str, str]:
    return {
        "python": os.path.join(venv_dir, "bin", "python"),
        "pip":    os.path.join(venv_dir, "bin", "pip"),
        "pytest": os.path.join(venv_dir, "bin", "pytest"),
    }

def _pip_cmd(venv: Dict[str, str]) -> List[str]:
    # Always use module invocation; avoids broken shebangs after renames.
    return [venv["python"], "-m", "pip"]

def _ensure_src_pkg(ws: str, log: Callable[[str], None]) -> None:
    p = os.path.join(ws, "src", "__init__.py")
    if not os.path.exists(p):
        _write_file(ws, "src/__init__.py", '"""src package marker for tests importing src.*"""\n', log)
        log("Created src/__init__.py to support imports like `from src.pkg import ...`")

def _probe_pip(venv: Dict[str, str], log: Callable[[str], None]) -> bool:
    try:
        code, _, _ = _run(_pip_cmd(venv) + ["--version"], cwd="/", env=os.environ.copy(), log=log)
        return code == 0
    except Exception:
        return False

def _recreate_venv(venv_dir: str, log: Callable[[str], None]) -> Dict[str, str]:
    shutil.rmtree(venv_dir, ignore_errors=True)
    _mkdirp(venv_dir)
    bins = _venv_bins(venv_dir)
    log(f"Creating Python venv at {venv_dir}")
    _run(["python3", "-m", "venv", venv_dir], cwd="/", env=os.environ.copy(), log=log)
    return bins

def _ensure_python_venv(venv_dir: str, log: Callable[[str], None]) -> Dict[str, str]:
    bins = _venv_bins(venv_dir)
    py = bins["python"]; pip = bins["pip"]

    def _make():
        log(f"Creating Python venv at {venv_dir}")
        _run(["python3", "-m", "venv", venv_dir], cwd="/", env=os.environ.copy(), log=log)

    if not os.path.exists(py):
        _make()

    def _pip_ok() -> bool:
        try:
            code, _, _ = _run([pip, "--version"], cwd="/", env=os.environ.copy(), log=lambda *_: None)
            return code == 0
        except Exception:
            return False

    if not os.path.exists(pip) or not _pip_ok():
        _run([py, "-m", "ensurepip", "--upgrade"], cwd="/", env=os.environ.copy(), log=log)

    _run([pip, "install", "--upgrade", "pip"], cwd="/", env=os.environ.copy(), log=log)
    _run([pip, "install", "-q", "pytest"], cwd="/", env=os.environ.copy(), log=log)
    return bins



def _pip_check(venv: Dict[str, str], cwd: str, log: Callable[[str], None]) -> bool:
    code, _, _ = _run(_pip_cmd(venv) + ["check"], cwd=cwd, env=os.environ.copy(), log=log)
    return code == 0

def _pool_dirs() -> List[str]:
    if not os.path.isdir(VENV_POOL_ROOT): return []
    return [os.path.join(VENV_POOL_ROOT, d) for d in os.listdir(VENV_POOL_ROOT)
            if os.path.isdir(os.path.join(VENV_POOL_ROOT, d))]

def _pool_has(venv_dir: str, pkg: str) -> Optional[str]:
    bins = _venv_bins(venv_dir)
    if not os.path.exists(bins["python"]): return None
    try:
        p = subprocess.Popen([bins["python"], "-m", "pip", "show", pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = p.communicate(timeout=10)
        if p.returncode != 0: return None
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":",1)[1].strip()
    except Exception:
        return None
    return None

def _venv_pool_find(pkgs: List[str]) -> Optional[Dict[str, str]]:
    for d in _pool_dirs():
        ok = True
        for name in pkgs:
            if _pool_has(d, name) is None:
                ok = False; break
        if ok:
            return _venv_bins(d)
    return None

def _venv_pool_create_named(name: str, log: Callable[[str], None]) -> Dict[str, str]:
    path = os.path.join(VENV_POOL_ROOT, name)
    _mkdirp(VENV_POOL_ROOT)
    bins = _ensure_python_venv(path, log)
    return bins

def _venv_pool_create_for(pkgs: List[str], log: Callable[[str], None]) -> Dict[str, str]:
    tmp = os.path.join(VENV_POOL_ROOT, f"tmp-{uuid.uuid4().hex[:8]}")
    _mkdirp(VENV_POOL_ROOT)
    bins_tmp = _ensure_python_venv(tmp, log)
    _run(_pip_cmd(bins_tmp) + ["install", "-q"] + pkgs, cwd="/", env=os.environ.copy(), log=log)

    parts = []
    for n in pkgs:
        ver = _pool_has(tmp, n) or "unknown"
        parts.append(f"{n}-{ver}")
    final_name = f"{_py_tag()}-" + "__".join(parts)
    final = os.path.join(VENV_POOL_ROOT, final_name)
    try:
        os.rename(tmp, final)
    except Exception:
        final = tmp
    log(f"Created pool venv: {os.path.basename(final)}")
    return _venv_bins(final)

def _resolve_missing_via_pool(missing: List[str], current_venv: Dict[str,str],
                              ws: str, log: Callable[[str], None]) -> Dict[str,str] | None:
    # never try to pip-install the pseudo-package 'src'
    pkgs = [n for n in missing if n not in _STDLIB_DENY and n != "src" and len(n) >= 3]
    if not pkgs:
        return None
    if len(pkgs) == 1:
        found = _venv_pool_find(pkgs)
        if found:
            log(f"Using pool venv for {pkgs[0]}")
            return found
        return _venv_pool_create_for(pkgs, log)
    found = _venv_pool_find(pkgs)
    if found:
        log(f"Using pool venv containing: {', '.join(pkgs)}")
        return found
    return _venv_pool_create_for(pkgs, log)

# ---------- git helpers ----------
def _git_commit_only(repo_root: str, paths: List[str], msg: str, log: Callable[[str], None]) -> None:
    if not _has_git():
        log(f"Git not available  skipping commit: {msg}")
        return
    args = ["add"] + paths + ["--"]
    _git(repo_root, args, log)
    _git(repo_root, ["commit", "-m", msg], log)

def _git(ws: str, args: list[str], log: Callable[[str], None]) -> int:
    if not _has_git():
        log("git not found; skipping: " + " ".join(args))
        return 127
    cmd = ["git"] + args
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        p = subprocess.Popen(cmd, cwd=ws, env=os.environ.copy(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate()
    except FileNotFoundError:
        log("git missing at runtime; skipping git command.")
        return 127
    if out:
        for line in out.splitlines(): log(line)
    if err:
        for line in err.splitlines(): log(line)
    return p.returncode

def _git_sync_with_remote(repo_root: str, log: Callable[[str], None]) -> None:
    """
    Try to synchronize with origin/main if it exists.
    - If remote not set or unreachable, just return.
    - If main exists remotely and we have no commits, create an orphan main and pull.
    - Otherwise: fetch + pull --rebase --autostash (idempotent).
    All failures are logged but non-fatal.
    """
    if not _has_git():
        return
    # remote may not exist
    if _git(repo_root, ["remote", "get-url", "origin"], log) != 0:
        return
    _git(repo_root, ["fetch", "--all", "--prune"], log)

    # ensure we have a local main
    rc = _git(repo_root, ["rev-parse", "--verify", "main"], log)
    if rc != 0:
        # no local main; try to create and track remote main if present
        if _git(repo_root, ["rev-parse", "--verify", "origin/main"], log) == 0:
            _git(repo_root, ["checkout", "-b", "main", "--track", "origin/main"], log)
        else:
            # create main anyway; remote may be empty
            _git(repo_root, ["checkout", "-b", "main"], log)

    # pull with rebase to avoid merge commits; autostash to survive local skeleton files
    _git(repo_root, ["pull", "--rebase", "--autostash", "origin", "main"], log)

def _git_init(repo_root: str, log: Callable[[str], None]) -> None:
    if not _has_git():
        log("Git not available  skipping repo init.")
        return
    _mkdirp(repo_root)
    _git(repo_root, ["init", "-b", "main"], log)
    user = os.environ.get("GIT_USER_NAME", "astro-agent")
    email = os.environ.get("GIT_USER_EMAIL", "astro@example.local")
    _git(repo_root, ["config", "user.name", user], log)
    _git(repo_root, ["config", "user.email", email], log)
    remote = os.environ.get("GIT_REMOTE")
    if remote:
        rc = _git(repo_root, ["remote", "get-url", "origin"], log)
        if rc != 0:
            _git(repo_root, ["remote", "add", "origin", remote], log)
        else:
            _git(repo_root, ["remote", "set-url", "origin", remote], log)

def _git_commit(repo_root: str, msg: str, log: Callable[[str], None]) -> None:
    if not _has_git():
        log(f"Git not available  skipping commit: {msg}")
        return
    _git(repo_root, ["add", "-A"], log)
    _git(repo_root, ["commit", "-m", msg], log)

def _git_push(repo_root: str, log: Callable[[str], None]) -> None:
    if not _has_git():
        log("Git not available  skipping push.")
        return
    if os.environ.get("GIT_REMOTE"):
        _git(repo_root, ["push", "-u", "origin", "main"], log)

# ---------- RAG ----------
def _rag_snippets(query: str, k: int = 6) -> str:
    """
    Try POST first; on 404/405 fallback to GET.
    Accept {"hits":[{"content"|"text":...}]} or {"results":[...]}.
    """
    url = RAG_SEARCH_URL
    try:
        import httpx
        try:
            r = httpx.post(url, json={"q": query, "k": k}, timeout=5.0)
            if r.status_code in (404, 405):
                raise RuntimeError("fallback-to-get")
        except Exception:
            r = httpx.get(url, params={"q": query, "k": str(k)}, timeout=5.0)
        if r.status_code != 200:
            return ""
        data = r.json()
        items = data.get("hits", data.get("results", [])) or []
        texts = []
        for h in items:
            t = (h.get("content") or h.get("text") or "")[:800]
            if t:
                texts.append(t)
        return "\n".join(f"- {t}" for t in texts[:k]) if texts else ""
    except Exception:
        return ""

# ---------- planning / docs ----------
def _llm_plan(problem: str, project: str) -> dict:
    # pull some context (best-effort)
    ctx = _rag_snippets(f"Plan a Python package for: {problem}. Rules for pytest project layout and avoiding circular imports.", k=5)
    context_block = f"\nHelpful context:\n{ctx}\n" if ctx else ""

    plan_prompt = (
        "You are a software planner. Return STRICT JSON with keys:\n"
        '{ "language":"python", "files":["src/{pkg}/__init__.py","src/{pkg}/main.py","tests/test_main.py"],'
        '  "deps":[], "test_strategy":"1-2 lines", "rationale":"1-2 lines" }\n'
        "Rules: Only JSON. No code fences. No commentary."
        f"{context_block}\n\n"
        f"Project: {project}\nTask: {problem}\n"
    )
    resp = llm_client.complete(prompt=plan_prompt, model=os.environ.get("DEFAULT_MODEL","llama3.2:3b"))
    raw = resp.get("text","") if resp.get("ok") else ""
    try:
        return json.loads(raw)
    except Exception:
        pkg = project.replace("-", "_")
        return {
            "language": "python",
            "files": [f"src/{pkg}/__init__.py", f"src/{pkg}/main.py", "tests/test_main.py"],
            "deps": [],
            "test_strategy": "Unit tests via pytest.",
            "rationale": "Fallback plan.",
        }

def _write_plan_md(ws: str, plan: dict, project: str) -> None:
    lines = [
        "# PLAN",
        "",
        f"**Language:** {plan.get('language','python')}",
        "**Files:**",
    ] + [f"- {p}" for p in plan.get("files", [])] + [
        "",
        "**Dependencies:**",
    ] + [f"- {d}" for d in plan.get("deps", [])] + [
        "",
        f"**Test strategy:** {plan.get('test_strategy','')}",
        f"**Rationale:** {plan.get('rationale','')}",
        "",
    ]
    _write_file(ws, "docs/PLAN.md", "\n".join(lines))

def _doxygen_scaffold(ws: str, project: str, log: Callable[[str], None]) -> None:
    _mkdirp(os.path.join(ws, "scripts"))
    doxy = (
        f'PROJECT_NAME = "{project}"\n'
        "OUTPUT_DIRECTORY = docs\n"
        "RECURSIVE = YES\n"
        "GENERATE_HTML = YES\n"
        "INPUT = src\n"
    )
    _write_file(ws, "Doxyfile", doxy, log)
    _write_file(ws, "scripts/gen_docs.sh", "#!/usr/bin/env bash\nset -euo pipefail\ndoxygen Doxyfile\n", log)
    _run(["chmod", "+x", "scripts/gen_docs.sh"], cwd=ws, env=os.environ.copy(), log=log)

def _manifests(ws: str, plan: dict, project: str, log: Callable[[str], None]) -> None:
    lang = (plan.get("language") or "python").lower()
    deps = plan.get("deps") or []
    if lang == "python":
        _write_file(ws, "requirements.txt", ("\n".join(deps) + ("\n" if deps else "")), log)
        log("Wrote requirements.txt")

# ---------- auto-heal (generic) ----------
def _project_top_packages(ws: str) -> set:
    src = os.path.join(ws, "src")
    if not os.path.isdir(src):
        return set()
    return {d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))}

def _missing_modules_from_output(output: str) -> List[str]:
    return list(sorted(set(_MNF_RE.findall(output or ""))))

def _tests_imports_name(ws: str, name: str) -> bool:
    tests = os.path.join(ws, "tests")
    if not os.path.isdir(tests):
        return False
    pat1 = re.compile(rf"^\s*import\s+{re.escape(name)}\b")
    pat2 = re.compile(rf"^\s*from\s+{re.escape(name)}\s+import\b")
    for root, _, files in os.walk(tests):
        for fn in files:
            if not fn.endswith(".py"): continue
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if pat1.search(line) or pat2.search(line):
                            return True
            except Exception:
                pass
    return False

def _maybe_alias_package(ws: str, real_pkg: str, missing_name: str, log: Callable[[str], None]) -> bool:
    # Never alias the 'src' pseudo-package; we support it via PYTHONPATH + src/__init__.py
    if missing_name in {real_pkg, "src"}:
        return False
    if not _tests_imports_name(ws, missing_name):
        return False
    if os.path.isdir(os.path.join(ws, "src", missing_name)):
        return False
    code = (
        f"# Auto-generated alias so tests importing '{missing_name}' refer to our package '{real_pkg}'.\n"
        f"from {real_pkg} import *  # noqa: F401,F403\n"
    )
    _write_file(ws, f"src/{missing_name}/__init__.py", code, log)
    log(f"Created alias package: src/{missing_name}/__init__.py -> re-exports from {real_pkg}")
    return True

def _neutralize_pkg_init(ws: str, real_pkg: str, log: Callable[[str], None]) -> bool:
    init_p = os.path.join(ws, "src", real_pkg, "__init__.py")
    if not os.path.exists(init_p):
        return False
    try:
        content = (
            '"""Auto-minimized to avoid circular imports/packaging side effects during tests."""\n'
            "__all__ = []\n"
        )
        _write_file(ws, f"src/{real_pkg}/__init__.py", content, log)
        log(f"Neutralized src/{real_pkg}/__init__.py")
        return True
    except Exception as e:
        log(f"Failed to neutralize __init__.py: {e}")
        return False

def _init_suspicious(ws: str, real_pkg: str) -> bool:
    p = os.path.join(ws, "src", real_pkg, "__init__.py")
    try:
        with open(p, "r", encoding="utf-8") as f:
            t = f.read()
        return bool(_INIT_PKG_SMELL.search(t))
    except Exception:
        return False

def _collect_imported_names_for_submodule(ws: str, pkg: str, sub: str) -> List[str]:
    """Scan src/pkg/*.py for lines like: from .sub import A, B"""
    names: set[str] = set()
    root = os.path.join(ws, "src", pkg)
    if not os.path.isdir(root):
        return []
    pat = re.compile(rf"^\s*from\s+\.\s*{re.escape(sub)}\s+import\s+(.+)$")
    for fn in os.listdir(root):
        if not fn.endswith(".py"): continue
        try:
            with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                for line in f:
                    m = pat.match(line)
                    if not m: continue
                    raw = m.group(1)
                    if raw.strip() == "*":
                        continue
                    for part in raw.split(","):
                        nm = part.strip().split(" as ")[0].strip()
                        if nm:
                            names.add(nm)
        except Exception:
            continue
    return sorted(names)

def _synthesize_local_submodule(ws: str, pkg: str, sub: str, log: Callable[[str], None]) -> bool:
    """
    Create src/pkg/sub.py if missing. If we can detect names imported from it,
    stub them with NotImplementedError so pytest produces actionable failures.
    """
    mod_p = os.path.join(ws, "src", pkg, f"{sub}.py")
    if os.path.exists(mod_p):
        return False
    names = _collect_imported_names_for_submodule(ws, pkg, sub)
    lines = [
        f'"""Auto-generated stub for missing local submodule {pkg}.{sub}.',
        'The LLM should implement real API in subsequent patches."""',
        "",
    ]
    for nm in names:
        lines.append(f"def {nm}(*args, **kwargs):")
        lines.append(f"    raise NotImplementedError('auto-stub: implement {pkg}.{sub}.{nm}')")
        lines.append("")
    if not names:
        lines.append("pass\n")
    _write_file(ws, f"src/{pkg}/{sub}.py", "\n".join(lines), log)
    log(f"Synthesized local submodule src/{pkg}/{sub}.py with stubs: {', '.join(names) or '(none)'}")
    return True

def _auto_heal_imports(test_output: str, venv: Dict[str,str], ws: str, real_pkg: str,
                       log: Callable[[str], None]) -> Tuple[bool, Optional[Dict[str,str]]]:
    """
    Generic healing:
      1) If circular import OR suspicious packaging code in __init__.py -> neutralize it.
      2) For ModuleNotFoundError:
         - If it's a local dotted submodule (real_pkg.sub), synthesize stub file.
         - If tests import a different local top-name, create alias package.
         - Ensure 'src' pseudo-package exists (never try to pip-install it).
         - Else treat as third-party and resolve via the shared venv pool.
    Returns (did_anything, new_venv_or_None).
    """
    did_any = False
    output = (test_output or "")

    # 1) neutralize __init__ if circular import or packaging smell
    if _CIRCULAR_RE.search(output) or _init_suspicious(ws, real_pkg):
        if _neutralize_pkg_init(ws, real_pkg, log):
            did_any = True

    # collect missing imports
    missing = _missing_modules_from_output(output)

    # handle 'src' explicitly (create marker package if missing)
    if "src" in missing:
        _ensure_src_pkg(ws, log)
        did_any = True
        missing = [m for m in missing if m != "src"]

    # handle local dotted submodules first (e.g., pkg.greet)
    for full in list(missing):
        if "." in full:
            base, sub = full.split(".", 1)
            if base == real_pkg and re.fullmatch(r"[A-Za-z0-9_]+", sub):
                if _synthesize_local_submodule(ws, real_pkg, sub, log):
                    did_any = True
                missing = [m for m in missing if m != full]

    if not missing:
        return did_any, None

    leftovers: List[str] = []
    top_pkgs = _project_top_packages(ws)

    # alias wrong local import names (consider base of dotted names)
    for name in missing:
        base = name.split(".", 1)[0]
        if base in top_pkgs or base == real_pkg:
            continue
        if _maybe_alias_package(ws, real_pkg, base, log):
            did_any = True
        else:
            leftovers.append(base)

    # third-party modules -> shared venv pool (exclude 'src' already)
    leftovers = sorted(set([n for n in leftovers if n not in _STDLIB_DENY and n != "src" and len(n) >= 3]))
    if leftovers:
        new_venv = _resolve_missing_via_pool(leftovers, venv, ws, log)
        if new_venv:
            did_any = True
            return did_any, new_venv

    return did_any, None

# ---------- embedded tests relocator ----------
def _move_embedded_tests_to_top(ws: str, pkg: str, log: Callable[[str], None]) -> int:
    src_tests_dir = os.path.join(ws, "src", pkg, "tests")
    top_tests_dir = os.path.join(ws, "tests")
    moved = 0
    if os.path.isdir(src_tests_dir):
        _mkdirp(top_tests_dir)
        for root, _, files in os.walk(src_tests_dir):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                src_path = os.path.join(root, fn)
                rel_inside = os.path.relpath(src_path, src_tests_dir)
                dest_path = os.path.join(top_tests_dir, rel_inside)
                _mkdirp(os.path.dirname(dest_path))
                shutil.move(src_path, dest_path)
                log(f"Moved embedded test {src_path} -> {dest_path}")
                moved += 1
        try:
            shutil.rmtree(src_tests_dir)
        except Exception:
            pass
    return moved

# ---------- main entry ----------
def run(
    problem: str,
    project_name: str,
    model: str,
    workspaces_root: str,
    log: Callable[[str], None],
    cancel_flag: Callable[[], bool],
    max_iterations: int = 5,
) -> Dict[str, str]:
    """
    plan -> scaffold -> codegen -> test -> (fix loop) + doxygen + git.
    Python-first; Ollama-only LLM client; RAG-fed prompts when available.
    """
    repo_root = workspaces_root
    ws = os.path.join(repo_root, project_name)
    pkg = project_name.replace("-", "_")
    src_dir = os.path.join(ws, "src", pkg)
    docs_dir = os.path.join(ws, "docs")
    inputs_dir = os.path.join(ws, "inputs")
    _mkdirp(src_dir); _mkdirp(docs_dir); _mkdirp(inputs_dir)
    _readme = (
        f"# {project_name}\n\n"
        f"- **Problem:** {problem}\n"
        f"- **Structure**\n"
        f"  - `src/`  code\n"
        f"  - `tests/`  tests (top level)\n"
        f"  - `docs/`  planning/notes (e.g., PLAN.md)\n"
        f"  - `inputs/`  any input samples\n"
        f"  - `scripts/`  build/docs helpers\n"
    )
    _write_file(ws, "README.md", _readme, log)

    # .gitignore at repo root (avoid tracking venvs)
    gi_repo = os.path.join(repo_root, ".gitignore")
    if not os.path.exists(gi_repo):
        _write_file(repo_root, ".gitignore",
                    ".pytest_cache/\n__pycache__/\n*.pyc\n*/agentPyWork/\n*.egg-info/\n", log)

    log(f"Workspace: {ws}")
    log(f"Problem: {problem}")

    # Git at repo root
    _git_init(repo_root, log)
    _git_sync_with_remote(repo_root, log)
    _git_commit_only(repo_root, [project_name, ".gitignore"], "chore: scaffold workspace root", log)


    # Plan & manifests
    log("Step 1/7: Create plan")
    plan = _llm_plan(problem, project_name)
    _write_plan_md(ws, plan, project_name)
    _manifests(ws, plan, project_name, log)
    _git_commit_only(repo_root, [project_name], "docs: add PLAN.md and manifests", log)

    # Doxygen scaffold
    log("Step 2/7: Doxygen scaffolding")
    try:
        _doxygen_scaffold(ws, project_name, log)
        _git_commit_only(repo_root, [project_name], "chore: add Doxygen scaffolding", log)
    except Exception as e:
        log(f"Doxygen scaffolding skipped: {e}")

    # Workspace-local venv
    log("Step 3/7: Ensure Python venv (workspace-local)")
    # default global venv: /workspaces/.venv-pool/agentPyWork
    venv = _ensure_python_venv(os.path.join(VENV_POOL_ROOT, "agentPyWork"), log)
    if cancel_flag():
        return {"status": "cancelled", "summary": "Run cancelled."}

    # Install deps if present
    req_path = os.path.join(ws, "requirements.txt")
    if os.path.exists(req_path) and os.path.getsize(req_path) > 0:
        code, _, _ = _run(_pip_cmd(venv) + ["install", "-q", "-r", "requirements.txt"], cwd=ws, env=os.environ.copy(), log=log)
        if code != 0:
            _git_commit_only(repo_root, [project_name], "chore: record failed dependency installation", log)
            return {"status": "failed", "summary": "pip install failed."}
        _git_commit_only(repo_root, [project_name], "chore: install dependencies", log)

    if cancel_flag():
        return {"status": "cancelled", "summary": "Run cancelled."}

    # initial codegen
    log("Step 4/7: Ask LLM to emit initial file set")
    ctx = _rag_snippets("pytest project layout; avoid circular imports; writing simple Python packages with tests", k=6)
    ctx_block = f"\nHelpful context for code emission:\n{ctx}\n" if ctx else ""
    system_rules = (
        "You are a precise code emitter. Output ONLY code fences, each with an explicit path= hint.\n"
        f"Emit at least these files:\n"
        f"```python path=src/{pkg}/__init__.py\n...```\n"
        f"```python path=src/{pkg}/main.py\n...```\n"
        f"```python path=tests/test_main.py\n...```\n"
        "Rules:\n"
        "- All tests must live under top-level 'tests/' (NOT inside the package).\n"
        "- Do NOT call setup(), use setuptools, or run code in __init__.py; keep it minimal (no eager imports).\n"
        "- Avoid circular imports; keep modules focused.\n"
        "- No narration, no markdown outside fences."
        f"{ctx_block}"
    )
    user_prompt = (
        f"Project: {project_name}\nGoal: {problem}\n"
        f"- Language: Python\n"
        f"- CLI may run via `python -m {pkg}` or a minimal main\n"
        f"- Keep it simple; standard library preferred\n"
        f"- Use code fences with path= as shown\n"
        f"Now emit files only."
    )
    resp = llm_client.complete(prompt=system_rules + "\n\n" + user_prompt, model=model)
    llm_text = resp.get("text", "") if resp.get("ok") else ""
    files = _parse_files_from_llm(llm_text)

    if not files:
        log("LLM returned no parsable files; writing a minimal skeleton to drive fixes.")
        files = [
            (f"src/{pkg}/__init__.py", '"""package skeleton (no eager imports)"""\n__all__ = []\n'),
            (f"src/{pkg}/main.py",
             "def _todo():\n"
             "    raise NotImplementedError('model must implement the task')\n"
             "if __name__ == '__main__':\n"
             "    _todo()\n"),
            ("tests/test_placeholder.py",
             "def test_placeholder():\n"
             "    # placeholder failing test so the model generates real implementation\n"
             "    assert False, 'placeholder test  implementation required'\n"),
        ]

    # write files
    log("Step 5/7: Write files to workspace")
    for rel, content in files:
        abs_path = _write_file(ws, rel, content, log)
        log(f"Wrote {rel} -> {abs_path}")

    if not os.path.exists(os.path.join(ws, "tests")):
        _mkdirp(os.path.join(ws, "tests"))
    moved = _move_embedded_tests_to_top(ws, pkg, log)
    if moved > 0:
        log(f"Relocated {moved} embedded test file(s) into top-level tests/")
    _ensure_src_pkg(ws, log)

    _git_commit_only(repo_root, [project_name], "feat: initial codegen", log)

    if cancel_flag():
        return {"status": "cancelled", "summary": "Run cancelled."}

    # test/fix loop
    log("Step 6/7: Test & self-correct loop")
    env = os.environ.copy()
    # support both 'import pkg' and 'import src.pkg'
    env["PYTHONPATH"] = os.pathsep.join([ws, os.path.join(ws, "src")])

    attempt = 0
    status: Dict[str, str] = {"status": "failed", "summary": "Unknown failure."}

    while attempt <= max_iterations:
        # proactively neutralize suspicious __init__ to avoid the 'invalid command tests' meltdown
        if _init_suspicious(ws, pkg):
            _neutralize_pkg_init(ws, pkg, log)

        # Run pytest as a module to avoid broken console-script shebangs after venv renames
        code, out, err = _run([venv["python"], "-m", "pytest", "-q", "tests"], cwd=ws, env=env, log=log)
        if code == 0:
            log(f"SUCCESS: All tests passed on attempt {attempt}.")
            _git_commit_only(repo_root, [project_name], f"tests: green on attempt {attempt}", log)
            status = {"status": "done", "summary": f"All tests passed in {attempt} fix iteration(s)."}
            break

        # Auto-heal (generic)
        healed, new_venv = _auto_heal_imports(out + "\n" + err, venv, ws, pkg, log)
        if new_venv:
            venv = new_venv  # subsequent iterations use the new python
        if healed:
            _git_commit_only(repo_root, [project_name], f"chore: auto-heal imports #{attempt}", log)
            attempt += 1
            continue

        if attempt == max_iterations:
            log("FAIL: Reached max_iterations; stopping.")
            status = {"status": "failed", "summary": "Tests failed after max_iterations."}
            break

        attempt += 1
        # RAG-assisted fix prompt
        rag = _rag_snippets(
            f"Fix pytest failures for a Python package. Problem: {problem}. "
            f"Recent pytest output (truncated):\n{(out + err)[:1200]}",
            k=6
        )
        hint = (
            "Fix the code to satisfy tests. Output ONLY code fences, each with an explicit path=.\n"
            "Emit only files that need changes. No narration.\n"
            "Do NOT add packaging code (setup/setuptools) into package modules. Keep __init__.py minimal."
        )
        newline = "\n"
        prompt = (
            f"{hint}{newline}Project: {project_name}{newline}Task: {problem}{newline}Plan files: {plan.get('files',[])}{newline}"
            f"{'Helpful context:{newline}' + rag if rag else ''}{newline}"
            f"Test output:{newline}{out}{newline}{err}")[:8000]

        resp = llm_client.complete(prompt=prompt, model=os.environ.get("DEFAULT_MODEL","llama3.2:3b"))
        text = resp.get("text","") if resp.get("ok") else ""
        patches = _parse_files_from_llm(text)

        if not patches:
            log("LLM returned no patches; stopping.")
            status = {"status": "failed", "summary": "No patches produced by model."}
            break

        for rel, content in patches:
            abs_path = _write_file(ws, rel, content, log)
            log(f"Patched {rel} -> {abs_path}")
        _git_commit_only(repo_root, [project_name], f"fix: attempt {attempt}", log)

    # optional push
    try:
        _git_push(repo_root, log)
    except Exception as e:
        log(f"git push skipped: {e}")

    log("Step 7/7: Done")
    return status

