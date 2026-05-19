#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the small YOLO evidence + VLM/HPE review case experiment.

Environment variables:
  DATA_ROOT        fMoW ImageFolder root
  FOCUS_CASES      evidence fusion focus_pair_cases.csv
  CASE_DIR         directory for selected case subset
  OUTPUT_DIR       directory for Qwen2.5-VL review outputs
  MODEL_SOURCE     modelscope, huggingface, or local
  MODEL_ID         model id or local model path
  MODEL_CACHE_DIR  ModelScope cache directory
  PER_PAIR         cases per focus pair, default 3
  LIMIT            optional smoke-test limit
  RESUME=1         skip completed cases

Example:
  DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
  bash scripts/run_evidence_hpe_review_cases.sh
EOF
  exit 0
fi

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data/fmow_key_subset_imagefolder}"
CASE_DIR="${CASE_DIR:-outputs/evidence_hpe_review_cases}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASE_DIR}/qwen2_5_vl_7b_evidence_review}"
FOCUS_CASES="${FOCUS_CASES:-outputs/evidence_fusion_analysis/focus_pair_cases.csv}"
PER_PAIR="${PER_PAIR:-3}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_MODEL_CACHE_DIR="/root/autodl-tmp/modelscope_cache"
else
  DEFAULT_MODEL_CACHE_DIR="${HOME}/.cache/modelscope"
fi
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"

python3 scripts/create_evidence_review_cases.py \
  --focus-cases "${FOCUS_CASES}" \
  --output-dir "${CASE_DIR}" \
  --per-pair "${PER_PAIR}" \
  --overwrite

args=(
  scripts/eval_vlm_evidence_review.py
  --data-root "${DATA_ROOT}"
  --subset "${CASE_DIR}/case_subset.jsonl"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${MODEL_ID}"
  --model-source "${MODEL_SOURCE}"
  --max-pixels "${MAX_PIXELS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${MODEL_CACHE_DIR}" ]]; then
  mkdir -p "${MODEL_CACHE_DIR}"
  args+=(--model-cache-dir "${MODEL_CACHE_DIR}")
fi

if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi

if [[ "${RESUME}" == "1" ]]; then
  args+=(--resume)
fi

python3 "${args[@]}"

echo "Done."
echo "Case subset: ${CASE_DIR}/case_subset.jsonl"
echo "Report: ${OUTPUT_DIR}/case_review_report.md"
