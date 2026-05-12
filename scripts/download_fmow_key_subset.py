#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bz2
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "fmow-rgb_manifest.json.bz2"
OUT_ROOT = ROOT / "data" / "fmow_key_subset"
BASE_URL = "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb"

KEY_CLASSES = [
    "military_facility",
    "airport",
    "airport_hangar",
    "airport_terminal",
    "runway",
    "port",
    "shipyard",
    "storage_tank",
    "oil_or_gas_facility",
    "factory_or_powerplant",
    "electric_substation",
    "prison",
]


def load_manifest(path: Path) -> List[str]:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def select_rgb_images(paths: Iterable[str], split: str, class_name: str, limit: int) -> List[str]:
    prefix = f"{split}/{class_name}/"
    selected: List[str] = []
    seen_sequences = set()

    for path in paths:
        if not path.startswith(prefix):
            continue
        if not path.endswith("_rgb.jpg"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        sequence_id = parts[2]
        # Prefer one RGB frame per temporal sequence to keep the starter subset diverse.
        if sequence_id in seen_sequences:
            continue
        selected.append(path)
        seen_sequences.add(sequence_id)
        if len(selected) >= limit:
            break

    return selected


def download_file(remote_path: str, local_path: Path, retries: int = 3) -> bool:
    if local_path.exists() and local_path.stat().st_size > 0:
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{remote_path}"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                local_path.write_bytes(resp.read())
            return True
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"failed to download {url}: {last_error}")


def metadata_path_for_image(image_path: str) -> str:
    return image_path[:-8] + "_rgb.json"


def build_jsonl_record(split: str, class_name: str, image_rel: str, metadata_rel: str) -> Dict[str, str]:
    return {
        "split": split,
        "facility_label": class_name,
        "image": str(OUT_ROOT / image_rel),
        "metadata": str(OUT_ROOT / metadata_rel),
        "source": "fMoW-rgb",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small fMoW-rgb key-category subset.")
    parser.add_argument("--train-limit", type=int, default=20, help="RGB images per class from train split")
    parser.add_argument("--val-limit", type=int, default=5, help="RGB images per class from val split")
    parser.add_argument("--classes", nargs="*", default=KEY_CLASSES, help="fMoW class names to download")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST}")

    paths = load_manifest(MANIFEST)
    all_records: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, int]] = {}

    for class_name in args.classes:
        stats[class_name] = {"train": 0, "val": 0}
        for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
            images = select_rgb_images(paths, split, class_name, limit)
            stats[class_name][split] = len(images)

            for image_path in images:
                meta_path = metadata_path_for_image(image_path)
                image_rel = image_path
                metadata_rel = meta_path

                downloaded_img = download_file(image_path, OUT_ROOT / image_rel)
                downloaded_meta = download_file(meta_path, OUT_ROOT / metadata_rel)
                status = "downloaded" if downloaded_img or downloaded_meta else "exists"
                print(f"[{status}] {image_path}", flush=True)
                all_records.append(build_jsonl_record(split, class_name, image_rel, metadata_rel))

    manifest_dir = OUT_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = manifest_dir / "fmow_key_subset.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats_path = manifest_dir / "fmow_key_subset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nsummary", flush=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    print(f"manifest: {out_jsonl}", flush=True)
    print(f"stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
