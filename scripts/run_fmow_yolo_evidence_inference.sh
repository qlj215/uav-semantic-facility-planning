#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-outputs/dior_yolo/yolov8n_evidence_min/weights/best.pt}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data/fmow_key_subset_imagefolder}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fmow_yolo_evidence}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.7}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
SAMPLE_PER_CLASS="${SAMPLE_PER_CLASS:-4}"
VIS_MAX_SIDE="${VIS_MAX_SIDE:-1600}"

if [[ ! -f "${MODEL}" ]]; then
  ALT_MODEL="runs/detect/outputs/dior_yolo/yolov8n_evidence_min/weights/best.pt"
  if [[ -f "${ALT_MODEL}" ]]; then
    MODEL="${ALT_MODEL}"
  fi
fi

python3 scripts/infer_fmow_yolo_evidence.py \
  --model "${MODEL}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --imgsz "${IMGSZ}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --sample-per-class "${SAMPLE_PER_CLASS}" \
  --vis-max-side "${VIS_MAX_SIDE}" \
  --overwrite

echo "Done."
echo "Evidence JSONL: ${OUTPUT_DIR}/evidence.jsonl"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "Visualizations: ${OUTPUT_DIR}/visualizations"
