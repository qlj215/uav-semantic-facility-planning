#!/usr/bin/env bash
set -euo pipefail

# Run three ResNet-50 baselines in order:
# 1) random initialization
# 2) ImageNet pretrained, train classifier head only
# 3) ImageNet pretrained, full fine-tuning
#
# Usage:
#   bash scripts/run_resnet50_three_baselines.sh
#
# Optional overrides:
#   DATA_ROOT=/path/to/fmow_key_subset_imagefolder \
#   EPOCHS_RANDOM=20 EPOCHS_HEAD=20 EPOCHS_FINETUNE=10 \
#   BATCH_SIZE=64 IMAGE_SIZE=224 WORKERS=8 \
#   bash scripts/run_resnet50_three_baselines.sh

DATA_ROOT="${DATA_ROOT:-data/fmow_key_subset_imagefolder}"
OUT_ROOT="${OUT_ROOT:-outputs/resnet50_baselines}"

EPOCHS_RANDOM="${EPOCHS_RANDOM:-20}"
EPOCHS_HEAD="${EPOCHS_HEAD:-20}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-10}"

BATCH_SIZE="${BATCH_SIZE:-64}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
WORKERS="${WORKERS:-8}"

LR_RANDOM="${LR_RANDOM:-1e-3}"
LR_HEAD="${LR_HEAD:-1e-3}"
LR_FINETUNE="${LR_FINETUNE:-1e-4}"

mkdir -p "${OUT_ROOT}"

echo "[1/3] ResNet-50 random initialization"
python3 scripts/train_resnet50_baseline.py \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS_RANDOM}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR_RANDOM}" \
  --output-dir "${OUT_ROOT}/random_init"

echo "[2/3] ResNet-50 ImageNet pretrained, classifier head only"
python3 scripts/train_resnet50_baseline.py \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS_HEAD}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR_HEAD}" \
  --use-torchvision \
  --pretrained \
  --freeze-backbone \
  --output-dir "${OUT_ROOT}/imagenet_head_only"

echo "[3/3] ResNet-50 ImageNet pretrained, full fine-tuning"
python3 scripts/train_resnet50_baseline.py \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS_FINETUNE}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR_FINETUNE}" \
  --use-torchvision \
  --pretrained \
  --output-dir "${OUT_ROOT}/imagenet_full_finetune"

echo "Done. Results:"
echo "  ${OUT_ROOT}/random_init"
echo "  ${OUT_ROOT}/imagenet_head_only"
echo "  ${OUT_ROOT}/imagenet_full_finetune"

