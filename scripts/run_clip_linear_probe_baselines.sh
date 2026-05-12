#!/usr/bin/env bash
set -euo pipefail

# Run linear-probe baselines in order:
# 1) HuggingFace CLIP
# 2) RemoteCLIP, if REMOTECLIP_CKPT is provided
#
# Usage:
#   DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
#   REMOTECLIP_CKPT=/root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
#   bash scripts/run_clip_linear_probe_baselines.sh

DATA_ROOT="${DATA_ROOT:-data/fmow_key_subset_imagefolder}"
OUT_ROOT="${OUT_ROOT:-outputs/clip_linear_probe}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

HF_CLIP_MODEL="${HF_CLIP_MODEL:-openai/clip-vit-base-patch32}"

REMOTECLIP_MODEL="${REMOTECLIP_MODEL:-ViT-B-32}"
REMOTECLIP_CKPT="${REMOTECLIP_CKPT:-}"

mkdir -p "${OUT_ROOT}"

echo "[1/2] HuggingFace CLIP linear probe"
python3 scripts/train_clip_linear_probe.py \
  --data-root "${DATA_ROOT}" \
  --backend transformers \
  --model-id "${HF_CLIP_MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --output-dir "${OUT_ROOT}/hf_clip_vit_b32"

if [[ -n "${REMOTECLIP_CKPT}" ]]; then
  if [[ ! -f "${REMOTECLIP_CKPT}" ]]; then
    echo "[error] REMOTECLIP_CKPT does not exist: ${REMOTECLIP_CKPT}" >&2
    exit 1
  fi

  echo "[2/2] RemoteCLIP linear probe"
  python3 scripts/train_clip_linear_probe.py \
    --data-root "${DATA_ROOT}" \
    --backend open_clip \
    --open-clip-model "${REMOTECLIP_MODEL}" \
    --open-clip-pretrained "${REMOTECLIP_CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --probe-batch-size "${PROBE_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --output-dir "${OUT_ROOT}/remoteclip_vit_b32"
else
  echo "[2/2] RemoteCLIP skipped."
  echo "      Set REMOTECLIP_CKPT=/path/to/RemoteCLIP-ViT-B-32.pt to enable it."
fi

echo "Done. Results:"
echo "  ${OUT_ROOT}/hf_clip_vit_b32"
if [[ -n "${REMOTECLIP_CKPT}" ]]; then
  echo "  ${OUT_ROOT}/remoteclip_vit_b32"
fi
