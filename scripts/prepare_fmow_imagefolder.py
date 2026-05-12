#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "fmow_key_subset" / "manifests" / "fmow_key_subset.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "fmow_key_subset_imagefolder"


def clean_filename(record: dict) -> str:
    image_path = Path(record["image"])
    # Include the parent sequence folder to avoid accidental duplicate names.
    return f"{image_path.parent.name}__{image_path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the downloaded fMoW key subset manifest to ImageFolder layout."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input jsonl manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output ImageFolder root.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output directory before writing.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input}")

    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = Path(record["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        records.append(record)

    copied_records = []
    for record in records:
        split = record["split"]
        label = record["facility_label"]
        src = Path(record["image"])
        dst = args.output / split / label / clean_filename(record)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        copied_records.append(
            {
                "split": split,
                "facility_label": label,
                "image": str(dst),
                "source_image": str(src),
                "source_metadata": record.get("metadata", ""),
            }
        )

    classes = sorted({r["facility_label"] for r in copied_records})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in copied_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    (args.output / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = Counter((r["split"], r["facility_label"]) for r in copied_records)
    summary = {
        "output": str(args.output),
        "total_images": len(copied_records),
        "classes": classes,
        "counts": {
            f"{split}/{label}": count for (split, label), count in sorted(counts.items())
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"ImageFolder written to: {args.output}")
    print(f"Images: {len(copied_records)}")
    print(f"Classes: {len(classes)}")
    for key, count in summary["counts"].items():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()

