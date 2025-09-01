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

info "Pre-flight checks…"
require_cmd docker
if ! docker version >/dev/null 2>&1; then
  die "Docker not responding. Ensure the Docker daemon is running: sudo systemctl status docker"
fi

if ! docker compose version >/dev/null 2>&1; then
  die "Docker Compose plugin missing. Install docker compose plugin and re-run."
fi
ok "Docker & Compose available"

# Verify SSD mount to avoid accidentally using the microSD
if [ ! -d "$SSD_ROOT" ]; then
  die "SSD mount point $SSD_ROOT not found. Mount your SSD at $SSD_ROOT and re-run."
fi
if ! mountpoint -q "$SSD_ROOT"; then
  warn "$SSD_ROOT exists but is not a mountpoint. This would put data on the microSD. Aborting for safety."
  die "Please mount your SSD at $SSD_ROOT (e.g., /dev/nvme0n1p1 -> $SSD_ROOT) and re-run."
fi
ok "SSD is mounted at $SSD_ROOT"

# Create folder structure on SSD (idempotent)
info "Creating directory layout on SSD under $APP_ROOT …"
sudo mkdir -p "$MODELS_DIR" "$DATA_DIR" "$LOGS_DIR" "$BIN_DIR" "$COMPOSE_DIR"
sudo chown -R "$(id -un)":"$(id -gn)" "$APP_ROOT"
ok "Folders ready:
  $MODELS_DIR
  $DATA_DIR
  $LOGS_DIR
  $BIN_DIR
  $COMPOSE_DIR"
