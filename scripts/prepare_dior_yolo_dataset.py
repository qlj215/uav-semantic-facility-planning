#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "manifests" / "dior_evidence" / "all.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "dior_yolo_evidence"
DEFAULT_CONFIG = ROOT / "configs" / "dior_evidence_classes.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_classes(config_path: Path, records: list[dict[str, Any]]) -> list[str]:
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        classes = config.get("yolo_classes")
        if classes:
            return [str(name) for name in classes]

    names = set()
    for record in records:
        for obj in record.get("evidence_objects", []):
            names.add(str(obj["class_name"]))
    return sorted(names)


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = [r for r in records if r.get("split") == "train"]
    val_records = [r for r in records if r.get("split") in {"val", "valid", "validation", "test"}]

    if train_records and val_records:
        return train_records, val_records

    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    val_count = max(1, int(round(len(records) * val_ratio))) if records else 0
    return records[val_count:], records[:val_count]


def safe_name(record: dict[str, Any], index: int) -> str:
    image = Path(record["image"])
    image_id = str(record.get("image_id") or image.stem)
    return f"{index:06d}_{image_id}{image.suffix.lower()}"


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def yolo_line(obj: dict[str, Any], class_to_idx: dict[str, int], width: int, height: int) -> str | None:
    class_name = str(obj.get("class_name", ""))
    if class_name not in class_to_idx:
        return None

    x1, y1, x2, y2 = [float(v) for v in obj["bbox_xyxy"]]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None

    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_to_idx[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def prepare_split(
    records: list[dict[str, Any]],
    split: str,
    output_dir: Path,
    class_to_idx: dict[str, int],
    copy_images: bool,
) -> dict[str, Any]:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    image_count = 0
    box_count = 0
    skipped_missing_image = 0
    skipped_empty_label = 0

    for index, record in enumerate(records):
        image_path = Path(record["image"])
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        lines: list[str] = []

        for obj in record.get("evidence_objects", []):
            line = yolo_line(obj, class_to_idx, width, height)
            if line is None:
                continue
            lines.append(line)
            class_counts[str(obj["class_name"])] += 1

        if not lines:
            skipped_empty_label += 1
            continue

        filename = safe_name(record, index)
        out_image = image_dir / filename
        out_label = label_dir / f"{Path(filename).stem}.txt"
        link_or_copy(image_path, out_image, copy_images)
        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

        image_count += 1
        box_count += len(lines)

    return {
        "images": image_count,
        "boxes": box_count,
        "class_counts": dict(sorted(class_counts.items())),
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty_label": skipped_empty_label,
    }


def write_data_yaml(output_dir: Path, classes: list[str]) -> Path:
    data_yaml = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(classes))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DIOR evidence manifest JSONL to YOLO detection format.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input DIOR evidence JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output YOLO dataset directory.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DIOR evidence class config.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when input has no split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output directory first.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    records = load_jsonl(input_path)
    classes = load_classes(Path(args.config).expanduser(), records)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "split": {
            "train": prepare_split(train_records, "train", output_dir, class_to_idx, args.copy_images),
            "val": prepare_split(val_records, "val", output_dir, class_to_idx, args.copy_images),
        },
    }
    summary["data_yaml"] = str(write_data_yaml(output_dir, classes))

    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
