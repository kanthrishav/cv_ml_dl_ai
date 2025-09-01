#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------
# Utility helpers (pretty log)
# --------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
ok() { echo -e "[\e[32mOK\e[0m]    $(ts)  $*"; }
info(){ echo -e "[\e[34mINFO\e[0m]  $(ts)  $*"; }
warn(){ echo -e "[\e[33mWARN\e[0m]  $(ts)  $*"; }
err(){ echo -e "[\e[31mERROR\e[0m] $(ts)  $*" >&2; }

trap 'err "Script failed at line $LINENO"; exit 1' ERR

# --------------------------
# Step 1: Sanity checks
# --------------------------
info "Step 1/8: Sanity checks (docker, compose plugin, SSD paths, running Ollama)"
command -v docker >/dev/null || { err "Docker not found in PATH"; exit 1; }
docker compose version >/dev/null 2>&1 || { err "Docker Compose plugin missing (docker compose)"; exit 1; }

[ -d /ssd ] || { err "/ssd does not exist"; exit 1; }
[ -w /ssd ] || { err "/ssd is not writable"; exit 1; }

COMPOSE_DIR="/ssd/ollama/compose"
MAIN_COMPOSE="${COMPOSE_DIR}/docker-compose.yml"
[ -f "${MAIN_COMPOSE}" ] || { err "Expected ${MAIN_COMPOSE} to exist (your Ollama compose)."; exit 1; }

# Is Ollama service present and reachable?
if ! docker ps --format '{{.Names}}' | grep -qx 'ollama'; then
  info "Ollama container not running; attempting to start via compose..."
  (cd "${COMPOSE_DIR}" && docker compose up -d ollama)
  sleep 2
fi
docker ps --format '{{.Names}}' | grep -qx 'ollama' || { err "Ollama container still not running. Please start it, then re-run."; exit 1; }
ok "Ollama container is running."

# --------------------------
# Step 2: Prepare Open WebUI SSD folders
# --------------------------
info "Step 2/8: Creating SSD folders for Open WebUI persistent data and library"
OWUI_DATA="/ssd/openwebui/data"
OWUI_LIB="/ssd/openwebui/library"
sudo mkdir -p "${OWUI_DATA}" "${OWUI_LIB}"
sudo chown -R "$(id -u)":"$(id -g)" /ssd/openwebui || true
ok "Created: ${OWUI_DATA} (writable) and ${OWUI_LIB} (read-only mount for documents)."

# --------------------------
# Step 3: Create a compose override (SAFE) that adds Open WebUI
#         (We do NOT edit your main file; we add docker-compose.override.yml)
# --------------------------
info "Step 3/8: Writing docker-compose.override.yml for Open WebUI (no changes to your main file)"
OVERRIDE="${COMPOSE_DIR}/docker-compose.override.yml"

# Backup any existing override
if [ -f "${OVERRIDE}" ]; then
  cp -a "${OVERRIDE}" "${OVERRIDE}.bak.$(date +%s)"
  warn "Found an existing override file; backed up to ${OVERRIDE}.bak.*"
fi

cat > "${OVERRIDE}" <<'YAML'
services:
  # Open WebUI service (reuses existing 'ollama' service over the same compose network)
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped

    # Web UI on host port 3000
    ports:
      - "3000:8080"

    # Tell WebUI to talk to the 'ollama' service in this compose project (not localhost)
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      # Optional UX defaults on first boot; WebUI persists them internally
      - ENABLE_PERSISTENT_CONFIG=true
      - ENABLE_WEB_SEARCH=true
      - WEB_SEARCH_ENGINE=duckduckgo

      # Make HF downloads more reliable IF you decide to use HF in WebUI:
      - HF_HUB_DISABLE_HF_TRANSFER=1
      - HF_HOME=/app/backend/data/hf-home
      - TRANSFORMERS_CACHE=/app/backend/data/hf-cache

      # (Optional) Identify this client in outbound requests
      - USER_AGENT=OpenWebUI-on-Jetson

    volumes:
      # Persist ALL Open WebUI data on SSD (DB, uploads, vector store, settings)
      - /ssd/openwebui/data:/app/backend/data
      # Read-only shared library of documents/images available inside WebUI as /library
      - /ssd/openwebui/library:/library:ro

    # Keep DNS inside container stable (fixes earlier DNS lookup failures)
    dns:
      - 1.1.1.1
      - 8.8.8.8

    # Ensure this starts after 'ollama' so the connection is valid on first boot
    depends_on:
      - ollama
YAML

ok "Wrote ${OVERRIDE}"

# --------------------------
# Step 4: Validate compose and pull Open WebUI image (stored under /ssd/docker)
# --------------------------
info "Step 4/8: Validating compose and pulling Open WebUI image"
( cd "${COMPOSE_DIR}" && docker compose config >/dev/null )
ok "Compose syntax is valid."

( cd "${COMPOSE_DIR}" && docker compose pull open-webui )
ok "Image pulled."

# --------------------------
# Step 5: Start/Restart the stack (ollama + open-webui)
# --------------------------
info "Step 5/8: Starting (or recreating) Open WebUI"
( cd "${COMPOSE_DIR}" && docker compose up -d open-webui )
sleep 1
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | sed -n '1,3p'

# --------------------------
# Step 6: Wait for Open WebUI to become healthy
# --------------------------
info "Step 6/8: Waiting for Open WebUI health status..."
for i in $(seq 1 60); do
  status="$(docker inspect -f '{{json .State.Health.Status}}' open-webui 2>/dev/null || echo '"starting"')"
  status="${status%\"}"; status="${status#\"}"
  if [ "${status}" = "healthy" ]; then
    ok "Open WebUI is healthy."
    break
  fi
  if [ "${status}" = "unhealthy" ]; then
    warn "Open WebUI healthcheck is UNHEALTHY (attempt ${i}/60). Tailing last logs for clues..."
    docker logs --tail=50 open-webui || true
  fi
  sleep 2
  if [ "${i}" -eq 60 ]; then
    warn "Timeout waiting for health. Continuing; service may still be reachable."
  fi
done

# --------------------------
# Step 7: Smoke tests (host endpoints + container DNS)
# --------------------------
info "Step 7/8: Running smoke tests (HTTP + DNS)"
curl -fsS http://127.0.0.1:3000/ >/dev/null && ok "WebUI HTTP is responding on http://127.0.0.1:3000/"
curl -fsS http://127.0.0.1:11434/api/version && ok "Ollama API is responding on http://127.0.0.1:11434/"
docker exec -i open-webui sh -lc 'getent hosts google.com >/dev/null && getent hosts huggingface.co >/dev/null && echo OK' \
  >/dev/null && ok "Container DNS resolution works inside open-webui."

# --------------------------
# Step 8: Pre-pull a local embeddings model in OLLAMA (to avoid HF downloads in WebUI)
# --------------------------
info "Step 8/8: Pre-pulling a local embeddings model in Ollama (nomic-embed-text)"
if docker exec -i ollama ollama pull nomic-embed-text >/dev/null 2>&1; then
  ok "Pulled embeddings model 'nomic-embed-text' into Ollama (stored under /ssd/ollama/models)."
  info "In Open WebUI: Admin Panel → Settings → Documents → set Embedding Provider = 'Ollama' and Model = 'nomic-embed-text'."
else
  warn "Could not pre-pull 'nomic-embed-text'. You can run it later: docker exec -i ollama ollama pull nomic-embed-text"
fi

echo
ok "DONE. Open WebUI is ready."
info "Open your browser at:  http://<JETSON-IP>:3000"
info "First visit: create the admin account. Then in Settings → Connections ensure OLLAMA_BASE_URL = http://ollama:11434"
info "For RAG: upload documents (stored under /ssd/openwebui/data) or place read-only files under /ssd/openwebui/library."
echo
