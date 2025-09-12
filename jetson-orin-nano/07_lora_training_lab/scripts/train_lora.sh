#!/usr/bin/env bash
set -euo pipefail

CKPT_FILE="/workspace/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"   # your mounted model file
DATA="/workspace/datasets/db"                    # DreamBooth parent with subfolders
OUT="/workspace/outputs"
LOG="/workspace/logs"
NAME="sd_512_lora"

mkdir -p "$OUT" "$LOG"

# sanity checks (follow symlinks when counting)
[ -f "$CKPT_FILE" ] || { echo "[FATAL] Missing $CKPT_FILE"; exit 2; }
IMG_COUNT="$(find -L "$DATA" -mindepth 2 -maxdepth 2 \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l || true)"
[ "$IMG_COUNT" -gt 0 ] || { echo "[FATAL] No images under $DATA/<concept>/ (followed symlinks)"; exit 2; }

echo "[ok] Found $IMG_COUNT training images under $DATA"

# --- PREVIEWS (guard against SIGPIPE due to head) ---
set +o pipefail
echo "[info] Example concept folders:"
find "$DATA" -mindepth 1 -maxdepth 1 -type d | head -n 6 || true
echo "[info] Example image links:"
find -L "$DATA" -mindepth 2 -maxdepth 2 -iname "*.jpg" | head -n 6 || true
set -o pipefail
# ----------------------------------------------------

# ensure sd-scripts exists
if [ ! -f /workspace/sd-scripts/train_network.py ]; then
  echo "[info] cloning sd-scripts..."
  git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
fi

echo "[info] starting training..."
exec python3 -u /workspace/sd-scripts/train_network.py \
  --pretrained_model_name_or_path "$CKPT_FILE" \
  --network_module "networks.lora" \
  --network_dim 8 --network_alpha 8 \
  --clip_skip 2 \
  --train_data_dir "$DATA" \
  --caption_extension .txt \
  --resolution 512,512 \
  --output_dir "$OUT" \
  --output_name "$NAME" \
  --prior_loss_weight 0.0 \
  --max_train_steps 4000 \
  --save_every_n_steps 500 \
  --save_precision fp16 \
  --sample_every_n_steps 500 \
  --sample_prompts "/workspace/prompts.txt" \
  --learning_rate 1e-4 \
  --unet_lr 1e-4 \
  --text_encoder_lr 5e-5 \
  --optimizer_type "AdamW" \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 100 \
  --train_batch_size 1 \
  --mixed_precision "fp16" \
  --sdpa \
  --gradient_checkpointing \
  --bucket_reso_steps 64 \
  --min_snr_gamma 5.0 \
  --cache_latents \
  --max_data_loader_n_workers 0 \
  --log_with "tensorboard" \
  --logging_dir "$LOG"
