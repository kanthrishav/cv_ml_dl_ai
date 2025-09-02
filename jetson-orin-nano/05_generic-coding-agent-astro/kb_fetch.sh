#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# Local KB bootstrapper for /ssd/kb
# - Fetches Dash/Zeal docsets by reading Kapeli FEEDS (XML -> .tgz URL)
# - Puts archives under: /ssd/kb/zeal_docsets/archives/<slug>/<slug>.tgz
# - Extracts under:      /ssd/kb/zeal_docsets/extracted/<slug>/
# - Adds light extras:   tldr-pages
# - Optional mirrors:    MIRROR_HEAVY=1 ./kb_fetch.sh
# - Optional SE dump:    SE_DUMP=1 ./kb_fetch.sh   (HUGE)
# No base-OS Python is used. Pure shell + curl/wget/tar/git.
# =====================================================================

KB_ROOT="/ssd/kb"
DOCSETS_DIR="$KB_ROOT/zeal_docsets"
FEEDS_DIR="$DOCSETS_DIR/feeds"
ARCHIVES_DIR="$DOCSETS_DIR/archives"
EXTRACT_DIR="$DOCSETS_DIR/extracted"

mkdir -p "$KB_ROOT" "$DOCSETS_DIR" "$FEEDS_DIR" "$ARCHIVES_DIR" "$EXTRACT_DIR"

echo "[feeds] cloning/updating Kapeli/feeds"
if [ ! -d "$FEEDS_DIR/.git" ]; then
  git clone --depth=1 https://github.com/Kapeli/feeds "$FEEDS_DIR"
else
  git -C "$FEEDS_DIR" pull --ff-only
fi

# -------- feed candidates (Kapeli filenames) -------------------------
# Key = logical topic we want. Value = space-separated feed basenames
# we'll try in order. The first one that exists wins.
# NOTE: Python & C++ are special (Python_3.xml, C++.xml)
declare -A FEED_CANDIDATES=(
  [Python]="Python_3 Python Python_2"
  [C]="C"
  [Cpp]="C++ CPlusPlus"
  [NodeJS]="NodeJS"
  [TypeScript]="TypeScript"
  [React]="React"
  [OpenCV]="OpenCV OpenCV_Python OpenCV_C++"
  [PyTorch]="PyTorch"
  [TensorFlow]="TensorFlow"
  [Bash]="Bash"
  [Docker]="Docker"
  [Git]="Git"
)

# -------- slug mapping (directory-safe names) ------------------------
slug_for() {
  local feed_id="$1"
  case "$feed_id" in
    "C++") echo "CPlusPlus" ;;
    "Python_3") echo "Python" ;;
    "Python_2") echo "Python2" ;;
    *) echo "$feed_id" ;;
  esac
}

echo "[docsets] resolving feed URLs and downloading archives"
for LOGICAL in "${!FEED_CANDIDATES[@]}"; do
  FOUND_FILE=""
  FOUND_ID=""
  for cand in ${FEED_CANDIDATES[$LOGICAL]}; do
    if [ -f "$FEEDS_DIR/$cand.xml" ]; then
      FOUND_FILE="$FEEDS_DIR/$cand.xml"
      FOUND_ID="$cand"
      break
    fi
  done

  if [ -z "${FOUND_FILE}" ]; then
    echo "  - $LOGICAL: no matching feed file found, skipped"
    continue
  fi

  # Get the first <url> (latest release)
  URL="$(grep -oP '(?<=<url>).*?(?=</url>)' "$FOUND_FILE" | head -n1 || true)"
  if [[ -z "${URL:-}" || ! "$URL" =~ ^https?:// ]]; then
    echo "  - $LOGICAL: no valid <url> in $FOUND_FILE, skipped"
    continue
  fi

  SLUG="$(slug_for "$FOUND_ID")"
  mkdir -p "$ARCHIVES_DIR/$SLUG" "$EXTRACT_DIR/$SLUG"
  TGZ="$ARCHIVES_DIR/$SLUG/${SLUG}.tgz"

  echo "  - $LOGICAL ($FOUND_ID) -> $SLUG: downloading"
  curl -L --fail --retry 3 -o "$TGZ" "$URL"

  echo "  - $LOGICAL: extracting to $EXTRACT_DIR/$SLUG"
  tar -xzf "$TGZ" -C "$EXTRACT_DIR/$SLUG"

  # Convenience symlinks for tricky names (optional)
  if [ "$FOUND_ID" = "C++" ]; then
    ln -sfn "$EXTRACT_DIR/$SLUG" "$EXTRACT_DIR/C++"
  fi
done

# --------- tldr pages (fast, high-signal) ----------------------------
TLDR_DIR="$KB_ROOT/tldr"
if [ ! -d "$TLDR_DIR/.git" ]; then
  echo "[tldr] cloning tldr-pages"
  git clone --depth=1 https://github.com/tldr-pages/tldr "$TLDR_DIR"
else
  echo "[tldr] updating tldr-pages"
  git -C "$TLDR_DIR" pull --ff-only
fi

# --------- heavy website mirrors (optional) --------------------------
if [ "${MIRROR_HEAVY:-0}" = "1" ]; then
  echo "[mirror] starting site mirrors (this can take a long time)"
  WGET_OPTS="--mirror --page-requisites --adjust-extension --convert-links --no-parent --no-verbose --retry-connrefused --waitretry=1 --tries=3"

  mkdir -p "$KB_ROOT/python"      && wget $WGET_OPTS -P "$KB_ROOT/python"      https://docs.python.org/3.10/
  mkdir -p "$KB_ROOT/pytorch"     && wget $WGET_OPTS -P "$KB_ROOT/pytorch"     https://pytorch.org/docs/stable/
  mkdir -p "$KB_ROOT/opencv"      && wget $WGET_OPTS -P "$KB_ROOT/opencv"      https://docs.opencv.org/4.x/
  mkdir -p "$KB_ROOT/huggingface" && wget $WGET_OPTS -P "$KB_ROOT/huggingface" https://huggingface.co/docs/transformers/index
  mkdir -p "$KB_ROOT/tensorflow"  && wget $WGET_OPTS -P "$KB_ROOT/tensorflow"  https://www.tensorflow.org/guide
  mkdir -p "$KB_ROOT/nvidia"      && wget $WGET_OPTS -P "$KB_ROOT/nvidia"      https://nvidia.github.io/TensorRT-LLM/
fi

# --------- official StackExchange dump (very large) ------------------
if [ "${SE_DUMP:-0}" = "1" ]; then
  echo "[stackexchange] fetching official dumps"
  SE_DIR="$KB_ROOT/stackexchange"
  mkdir -p "$SE_DIR"
  cd "$SE_DIR"
  aria2c -x8 -s8 -c \
    https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z \
    https://archive.org/download/stackexchange/stackoverflow.com-Tags.7z
  7z x -y stackoverflow.com-Posts.7z
  7z x -y stackoverflow.com-Tags.7z
fi

echo "[done] local KB prepared under $KB_ROOT"

