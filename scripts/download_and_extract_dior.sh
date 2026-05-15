#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

URL="https://zenodo.org/records/11213149/files/DIOR-VOC.zip?download=1"
MD5="fc722ffcfc4579fcb7acc111cb79b08d"
DATA_ROOT="/root/autodl-tmp/data/DIOR"
ARCHIVE_DIR="/root/autodl-tmp/data/downloads"
ARCHIVE_NAME="DIOR-VOC.zip"
KEEP_ZIP=0
PREPARE_MANIFEST=0

usage() {
  cat <<'EOF'
Download and extract DIOR VOC dataset.

Usage:
  bash scripts/download_and_extract_dior.sh [options]

Options:
  --data-root PATH          Extract DIOR here. Default: /root/autodl-tmp/data/DIOR
  --archive-dir PATH        Store zip here. Default: /root/autodl-tmp/data/downloads
  --url URL                 Override download URL.
  --no-md5                  Skip MD5 check.
  --keep-zip                Keep zip after extraction.
  --prepare-manifest        Run prepare_dior_evidence_manifest.py --scan-all after extraction.
  -h, --help                Show this help.

Example:
  bash scripts/download_and_extract_dior.sh \
    --data-root /root/autodl-tmp/data/DIOR \
    --prepare-manifest
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --archive-dir)
      ARCHIVE_DIR="$2"
      shift 2
      ;;
    --url)
      URL="$2"
      shift 2
      ;;
    --no-md5)
      MD5=""
      shift
      ;;
    --keep-zip)
      KEEP_ZIP=1
      shift
      ;;
    --prepare-manifest)
      PREPARE_MANIFEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${DATA_ROOT}" "${ARCHIVE_DIR}"
ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"

echo "[1/4] Download DIOR"
echo "url: ${URL}"
echo "zip: ${ARCHIVE_PATH}"

if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 8 -s 8 -k 1M -o "${ARCHIVE_NAME}" -d "${ARCHIVE_DIR}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "${ARCHIVE_PATH}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L -C - -o "${ARCHIVE_PATH}" "${URL}"
else
  echo "Need one downloader: aria2c, wget, or curl." >&2
  exit 1
fi

if [[ -n "${MD5}" ]]; then
  echo "[2/4] Check MD5"
  echo "${MD5}  ${ARCHIVE_PATH}" | md5sum -c -
else
  echo "[2/4] Skip MD5"
fi

echo "[3/4] Extract"
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is not installed. Install it first: apt-get update && apt-get install -y unzip" >&2
  exit 1
fi

unzip -q -o "${ARCHIVE_PATH}" -d "${DATA_ROOT}"

echo "[4/4] Normalize directory layout"
if [[ ! -d "${DATA_ROOT}/Annotations" && ! -d "${DATA_ROOT}/JPEGImages" && ! -d "${DATA_ROOT}/ImageSets" ]]; then
  mapfile -t CHILD_DIRS < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ "${#CHILD_DIRS[@]}" -eq 1 ]]; then
    CHILD="${CHILD_DIRS[0]}"
    if [[ -d "${CHILD}/Annotations" || -d "${CHILD}/JPEGImages" || -d "${CHILD}/ImageSets" ]]; then
      shopt -s dotglob nullglob
      mv "${CHILD}"/* "${DATA_ROOT}/"
      rmdir "${CHILD}" 2>/dev/null || true
      shopt -u dotglob nullglob
    fi
  fi
fi

echo "DIOR directory:"
find "${DATA_ROOT}" -maxdepth 2 -type d | sort | sed -n '1,40p'

if [[ "${KEEP_ZIP}" -eq 0 ]]; then
  rm -f "${ARCHIVE_PATH}"
  echo "Removed zip: ${ARCHIVE_PATH}"
else
  echo "Kept zip: ${ARCHIVE_PATH}"
fi

if [[ "${PREPARE_MANIFEST}" -eq 1 ]]; then
  echo "Prepare evidence manifest"
  python3 "${PROJECT_ROOT}/scripts/prepare_dior_evidence_manifest.py" \
    --dior-root "${DATA_ROOT}" \
    --scan-all \
    --output-dir "${PROJECT_ROOT}/data/manifests/dior_evidence"
fi

echo "Done."
echo "Next command:"
echo "python3 scripts/prepare_dior_evidence_manifest.py --dior-root ${DATA_ROOT} --scan-all --output-dir data/manifests/dior_evidence"
