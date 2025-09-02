#!/usr/bin/env bash
# /ssd/agents/astro/rag_run.sh
# Robust RAG runner: survives flaky KB/search and always reaches the API.

#set -euo pipefail

KB_URL="${KB_URL:-http://127.0.0.1:8383}"
API_URL="${API_URL:-http://127.0.0.1:8088}"
TOPK="${TOPK:-8}"
MAXCHARS="${MAXCHARS:-4000}"
TIMEOUT="${TIMEOUT:-25}"   # curl max time per request (seconds)

PROBLEM="${1:?Usage: rag_run.sh '<problem>' [project_name] [model] [max_iterations]}"
PROJECT_NAME="${2:-proj_$(date +%s)}"
MODEL="${3:-llama3.2:3b}"
ITER="${4:-1}"

debug() { [[ "${DEBUG:-0}" != "0" ]] && echo "[DEBUG] $*" >&2 || true; }

# -------------------------------
# 1) Retrieve context (tolerant)
# -------------------------------
CTX=""
if [[ "${NO_RETRIEVAL:-0}" != "0" ]]; then
  debug "Skipping retrieval (NO_RETRIEVAL=1)"
else
  debug "Retrieving context from KB: $KB_URL/search  q='$PROBLEM'  limit=$TOPK"
  # NOTE: .hits? // [] prevents jq from failing if .hits is missing/null
  if ! CTX="$(
    curl -sS --get "$KB_URL/search" --max-time "$TIMEOUT" \
      --data-urlencode "q=$PROBLEM" \
      --data-urlencode "limit=$TOPK" \
    | jq -r '
        .hits? // [] | .[]
        | select(.text != null and (.text|length>0))
        | "### " + (.source // "") + " :: " + (.rel_path // "") + "\n"
          + ((.text|gsub("\r";""))|sub("^\\s+";"")|sub("\\s+$";"")) + "\n---\n"
      ' 2>/dev/null \
    | head -c "$MAXCHARS"
  )"; then
    echo "[WARN] KB retrieval failed; continuing with empty context." >&2
    CTX=""
  fi
fi

# -------------------------------
# 2) Build augmented problem
# -------------------------------
AUG_PROBLEM=$(printf '%s\n%s\n%s\n%s\n\n%s\n' \
"Use the following retrieved context snippets when relevant. If any snippet conflicts, prefer official docs. Cite the source field in inline comments." \
"=== RETRIEVED CONTEXT START ===" \
"${CTX}" \
"=== RETRIEVED CONTEXT END ===" \
"=== TASK ===
${PROBLEM}")

if [[ "${DRY_RUN:-0}" != "0" ]]; then
  # Preview the exact prompt (no API call)
  printf '%s\n' "$AUG_PROBLEM"
  exit 0
fi

# -------------------------------
# 3) Call the API
# -------------------------------
# Build JSON safely with jq (ITER must be numeric)
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] max_iterations must be an integer, got: '$ITER'" >&2
  exit 1
fi

PAYLOAD="$(jq -Rn \
  --arg problem "$AUG_PROBLEM" \
  --arg project_name "$PROJECT_NAME" \
  --arg use_model "$MODEL" \
  --argjson max_iterations "$ITER" \
  '{problem:$problem, project_name:$project_name, use_model:$use_model, max_iterations:$max_iterations}'
)"

debug "POST $API_URL/run  model=$MODEL  project=$PROJECT_NAME  iters=$ITER"
RESP="$(curl -sS -f -m "$TIMEOUT" \
  -H 'content-type: application/json' \
  -d "$PAYLOAD" \
  "$API_URL/run")" || {
    code=$?
    echo "[ERROR] POST $API_URL/run failed (exit $code)" >&2
    exit $code
  }

# Print API response (should contain run_id)
echo "$RESP"

