#!/usr/bin/env bash
set -euo pipefail

EVIDENCE="${EVIDENCE:-outputs/fmow_yolo_evidence/evidence.jsonl}"
PREDICTIONS="${PREDICTIONS:-outputs/clip_linear_probe_1000epochs/hf_clip_vit_b32/predictions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evidence_fusion_analysis}"

python3 scripts/analyze_fmow_evidence_fusion.py \
  --evidence "${EVIDENCE}" \
  --predictions "${PREDICTIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite

echo "Done."
echo "Report: ${OUTPUT_DIR}/fusion_report.md"
echo "Focus pairs: ${OUTPUT_DIR}/focus_pair_summary.csv"
