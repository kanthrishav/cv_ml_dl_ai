#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8088}"

usage() {
  cat <<'EOF'
astro-run.sh - submit a coding task to astro-api and auto-stream logs.

USAGE:
  astro-run.sh --project <name> --task "<prompt>" [--model <ollama_model>] [--max-iters N] [--no-stream]

OPTIONS:
  --project     Required. Task folder name (used under astro_repos/<project>).
  --task        Required. Natural-language problem description.
  --model       Ollama model name. Default: llama3.2:3b
  --max-iters   Max fix iterations. Default: 5
  --no-stream   Submit only; print run_id and exit.

ENV:
  API_URL       Default http://127.0.0.1:8088
EOF
}

PROJECT=""
TASK=""
MODEL="llama3.2:3b"
MAX_ITERS=5
STREAM=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    --no-stream) STREAM=0; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${PROJECT}" || -z "${TASK}" ]]; then
  echo "Missing --project or --task"; usage; exit 1
fi

# Build JSON safely (jq if present, else python3)
if command -v jq >/dev/null 2>&1; then
  JSON_PAYLOAD=$(jq -n \
    --arg problem "$TASK" \
    --arg project "$PROJECT" \
    --arg model "$MODEL" \
    --argjson max_iterations "$MAX_ITERS" \
    '{problem:$problem, project_name:$project, use_model:$model, max_iterations:$max_iterations}')
else
  JSON_PAYLOAD=$(python3 - "$TASK" "$PROJECT" "$MODEL" "$MAX_ITERS" <<'PY'
import json, sys
task, project, model, max_iters = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print(json.dumps({"problem": task, "project_name": project, "use_model": model, "max_iterations": max_iters}))
PY
)
fi

RESP=$(curl -s -X POST "${API_URL}/run" \
  -H 'content-type: application/json' \
  --data-binary "${JSON_PAYLOAD}")

# Extract run_id/workspace robustly (jq if present, else python3)
if command -v jq >/dev/null 2>&1; then
  RUN_ID=$(printf %s "${RESP}" | jq -r '.run_id // empty')
  WORKSPACE=$(printf %s "${RESP}" | jq -r '.workspace // empty')
else
  RUN_ID=$(python3 - <<'PY' <<<"${RESP}"
import json, sys
try:
    print(json.load(sys.stdin)['run_id'])
except Exception:
    pass
PY
)
  WORKSPACE=$(python3 - <<'PY' <<<"${RESP}"
import json, sys
try:
    print(json.load(sys.stdin)['workspace'])
except Exception:
    pass
PY
)
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "Failed to start run. Response:"
  echo "${RESP}"
  exit 1
fi

echo "run_id=${RUN_ID}"
echo "workspace=${WORKSPACE}"

# Auto-stream (the API sends existing lines first, so you won't miss the start)
if [[ "${STREAM}" -eq 1 ]]; then
  echo
  echo "---- streaming logs ----"
  curl -N "${API_URL}/stream/${RUN_ID}"
fi

