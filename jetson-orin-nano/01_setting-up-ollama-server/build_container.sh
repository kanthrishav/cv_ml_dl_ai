# ===== save as: setup_ollama_on_ssd.sh =====
#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- Config you may change ----------
SSD_ROOT="/ssd"
APP_ROOT="$SSD_ROOT/ollama"
IMAGE_REPO="dustynv/ollama"
IMAGE_TAG="${IMAGE_TAG:-r36.4.0}"   # override by exporting IMAGE_TAG if you want a different published tag
CONTAINER_NAME="ollama"
HOST_PORT="11434"
COMPOSE_DIR="$APP_ROOT/compose"
BIN_DIR="$APP_ROOT/bin"
MODELS_DIR="$APP_ROOT/models"
DATA_DIR="$APP_ROOT/data"
LOGS_DIR="$APP_ROOT/logs"
WRAPPER="$BIN_DIR/ollama"
# ------------------------------------------

# Pretty logging
info(){ printf "\033[1;36m[INFO]\033[0m %s\n" "$*"; }
ok(){   printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err(){  printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; }

die(){ err "$1"; exit 1; }

require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "Missing '$1'. Install it and re-run."; }

# Validate YAML
info "Validating Compose file…"
docker compose -f "$COMPOSE_FILE" config >/dev/null
ok "Compose syntax OK"

# Pull image (this can be large; we print progress)
info "Pulling image ${IMAGE_REPO}:${IMAGE_TAG} …"
docker compose -f "$COMPOSE_FILE" pull
ok "Image pulled"

# Start container
info "Starting container '${CONTAINER_NAME}' …"
docker compose -f "$COMPOSE_FILE" up -d
sleep 1

# Confirm it's Up (not Exited)
STATUS="$(docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Status}}' || true)"
if [[ "$STATUS" != Up* ]]; then
  err "Container is not 'Up'. Current status: $STATUS"
  info "Last 120 logs lines:"
  docker logs --tail=120 "$CONTAINER_NAME" || true
  die "Startup failed. Check logs above."
fi
ok "Container is running: $STATUS"

# Health check (poll the API)
info "Waiting for Ollama API on http://127.0.0.1:${HOST_PORT} …"
ATTEMPTS=30
until curl -fsS "http://127.0.0.1:${HOST_PORT}/api/version" >/dev/null 2>&1; do
  ((ATTEMPTS--)) || { err "API did not become ready in time."; docker logs --tail=120 "$CONTAINER_NAME" || true; die "Exiting."; }
  sleep 1
done
ok "Ollama API is reachable."

# Wrapper CLI
info "Installing 'ollama' wrapper CLI to $WRAPPER …"
cat >"$WRAPPER" <<'WRAP'
#!/usr/bin/env bash
set -Eeuo pipefail
NAME="ollama"
COMPOSE_DIR="/ssd/ollama/compose"

# Simple helper to decide docker or sudo docker
_docker(){
  if docker ps >/dev/null 2>&1; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

# Ensure container is running
if ! _docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[ollama-wrapper] starting $NAME container…" >&2
  ( cd "$COMPOSE_DIR" && _docker compose up -d >/dev/null )
  sleep 1
fi

# If user exported OLLAMA_HOST, pass it through
if [ "${OLLAMA_HOST:-}" != "" ]; then
  exec _docker exec -e OLLAMA_HOST="$OLLAMA_HOST" -it "$NAME" ollama "$@"
else
  exec _docker exec -it "$NAME" ollama "$@"
fi
WRAP
chmod +x "$WRAPPER"

# Ensure wrapper on PATH (idempotent)
if ! grep -qF "$BIN_DIR" "$HOME/.bashrc" ; then
  info "Adding $BIN_DIR to PATH in ~/.bashrc"
  printf '\n# Ollama wrapper on SSD\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "$HOME/.bashrc"
fi
export PATH="$BIN_DIR:$PATH"
ok "Wrapper installed. Try:  ollama --help"

# Final health summary
info "Summary:"
echo "  • Compose file  : $COMPOSE_FILE"
echo "  • Container     : $CONTAINER_NAME ($(docker ps --filter name=^${CONTAINER_NAME}$ --format '{{.Status}}'))"
echo "  • API endpoint  : http://127.0.0.1:${HOST_PORT}"
echo "  • Models dir    : $MODELS_DIR"
echo "  • Data dir      : $DATA_DIR"
echo "  • Logs dir      : $LOGS_DIR"
echo "  • Wrapper CLI   : $WRAPPER"
ok "Ollama CLI on SSD is ready."

info "Quick test (optional):"
echo "  ollama --help"
echo "  # pull a small model, e.g.:"
echo "  # ollama pull llama3.2:3b   # ~3B, quick sanity model"
echo "  # ollama run llama3.2:3b \"Say hello from Jetson\""
