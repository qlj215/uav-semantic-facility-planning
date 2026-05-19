#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/dior_yolo_evidence}"
MANIFEST="${MANIFEST:-data/manifests/dior_evidence/all.jsonl}"
MODEL="${MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-20}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-outputs/dior_yolo}"
NAME="${NAME:-yolov8n_evidence_min}"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"

if [[ ! -f "${DATASET_DIR}/data.yaml" ]]; then
  echo "[1/2] Prepare YOLO dataset"
  python3 scripts/prepare_dior_yolo_dataset.py \
    --input "${MANIFEST}" \
    --output "${DATASET_DIR}" \
    --overwrite
else
  echo "[1/2] Use existing YOLO dataset: ${DATASET_DIR}"
fi

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
  echo "ultralytics not found, installing..."
  python3 -m pip install -U ultralytics
fi

echo "[2/2] Train YOLO detector"
python3 - <<PY
from ultralytics import YOLO

model = YOLO("${MODEL}")
model.train(
    data="${DATASET_DIR}/data.yaml",
    epochs=int("${EPOCHS}"),
    imgsz=int("${IMGSZ}"),
    batch=int("${BATCH}"),
    workers=int("${WORKERS}"),
    device="${DEVICE}",
    project="${PROJECT}",
    name="${NAME}",
    exist_ok=True,
)
PY

echo "Done."
echo "Results: ${PROJECT}/${NAME}"
