#!/usr/bin/env bash
set -euo pipefail

# Run CLIP-style zero-shot baselines in order:
# 1) HuggingFace CLIP
# 2) OpenCLIP
# 3) RemoteCLIP, if REMOTECLIP_CKPT is provided
#
# Usage:
#   bash scripts/run_clip_three_baselines.sh
#
# Optional overrides:
#   DATA_ROOT=/path/to/fmow_key_subset_imagefolder \
#   BATCH_SIZE=32 \
#   REMOTECLIP_CKPT=/path/to/RemoteCLIP-ViT-B-32.pt \
#   bash scripts/run_clip_three_baselines.sh

DATA_ROOT="${DATA_ROOT:-data/fmow_key_subset_imagefolder}"
OUT_ROOT="${OUT_ROOT:-outputs/clip_baselines}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-32}"

HF_CLIP_MODEL="${HF_CLIP_MODEL:-openai/clip-vit-base-patch32}"
OPENCLIP_MODEL="${OPENCLIP_MODEL:-ViT-B-32}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-openai}"

REMOTECLIP_MODEL="${REMOTECLIP_MODEL:-ViT-B-32}"
REMOTECLIP_CKPT="${REMOTECLIP_CKPT:-}"

mkdir -p "${OUT_ROOT}"

echo "[1/3] HuggingFace CLIP zero-shot"
python3 scripts/eval_clip_zero_shot.py \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --backend transformers \
  --model-id "${HF_CLIP_MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUT_ROOT}/hf_clip_vit_b32"

echo "[2/3] OpenCLIP zero-shot"
python3 scripts/eval_clip_zero_shot.py \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --backend open_clip \
  --open-clip-model "${OPENCLIP_MODEL}" \
  --open-clip-pretrained "${OPENCLIP_PRETRAINED}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUT_ROOT}/openclip_vit_b32"

if [[ -n "${REMOTECLIP_CKPT}" ]]; then
  if [[ ! -f "${REMOTECLIP_CKPT}" ]]; then
    echo "[error] REMOTECLIP_CKPT does not exist: ${REMOTECLIP_CKPT}" >&2
    exit 1
  fi

  echo "[3/3] RemoteCLIP zero-shot"
  python3 scripts/eval_clip_zero_shot.py \
    --data-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --backend open_clip \
    --open-clip-model "${REMOTECLIP_MODEL}" \
    --open-clip-pretrained "${REMOTECLIP_CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${OUT_ROOT}/remoteclip_vit_b32"
else
  echo "[3/3] RemoteCLIP skipped."
  echo "      Set REMOTECLIP_CKPT=/path/to/RemoteCLIP-ViT-B-32.pt to enable it."
fi

echo "Done. Results:"
echo "  ${OUT_ROOT}/hf_clip_vit_b32"
echo "  ${OUT_ROOT}/openclip_vit_b32"
if [[ -n "${REMOTECLIP_CKPT}" ]]; then
  echo "  ${OUT_ROOT}/remoteclip_vit_b32"
fi

