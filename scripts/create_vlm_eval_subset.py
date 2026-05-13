#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_OUTPUT = ROOT / "data" / "fmow_vlm_eval_subset.jsonl"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def list_classes(split_root: Path, requested_classes: Sequence[str] | None) -> list[str]:
    available = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
    if requested_classes is None:
        return available

    missing = [name for name in requested_classes if name not in available]
    if missing:
        raise FileNotFoundError(f"Classes not found under {split_root}: {missing}")
    return list(requested_classes)


def list_images(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def image_value(path: Path, data_root: Path, path_mode: str) -> str:
    if path_mode == "absolute":
        return str(path.resolve())
    return path.relative_to(data_root).as_posix()


def write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed VLM evaluation subset from an ImageFolder dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="Store image paths relative to data root or as absolute paths.",
    )
    parser.add_argument("--classes", nargs="+", default=None, help="Optional class list. Defaults to all class folders.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite to replace it.")
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    rng = random.Random(args.seed)
    classes = list_classes(split_root, args.classes)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    rows: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    for class_name in classes:
        images = list_images(split_root / class_name)
        selected = images if len(images) <= args.per_class else rng.sample(images, args.per_class)
        selected = sorted(selected)
        counts[class_name] = {
            "available": len(images),
            "selected": len(selected),
        }
        for path in selected:
            rows.append(
                {
                    "image": image_value(path, args.data_root, args.path_mode),
                    "label": class_name,
                    "class_id": class_to_idx[class_name],
                    "split": args.split,
                }
            )

    summary = {
        "data_root": str(args.data_root),
        "split": args.split,
        "output": str(args.output),
        "path_mode": args.path_mode,
        "seed": args.seed,
        "per_class": args.per_class,
        "classes": classes,
        "total_images": len(rows),
        "counts": counts,
    }

    write_jsonl(args.output, rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"classes={len(classes)}")
    print(f"images={len(rows)}")
    print(f"subset={args.output}")
    print(f"summary={summary_path}")
    for class_name in classes:
        info = counts[class_name]
        print(f"{class_name}: selected {info['selected']} / available {info['available']}")


if __name__ == "__main__":
    main()
