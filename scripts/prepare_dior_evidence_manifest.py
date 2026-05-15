#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dior_evidence_classes.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "manifests" / "dior_evidence"


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def load_class_map(config_path: Path) -> dict[str, str]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_map = data.get("source_class_to_evidence", {})
    return {normalize_label(k): normalize_label(v) for k, v in raw_map.items()}


def find_split_file(dior_root: Path, split: str) -> Path:
    candidates = [
        dior_root / "ImageSets" / "Main" / f"{split}.txt",
        dior_root / "ImageSets" / f"{split}.txt",
        dior_root / f"{split}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find split file for '{split}'. Searched:\n{searched}")


def read_image_ids(split_file: Path) -> list[str]:
    image_ids: list[str] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_ids.append(Path(line.split()[0]).stem)
    return image_ids


def find_annotation(dior_root: Path, image_id: str) -> Path:
    candidates = [
        dior_root / "Annotations" / f"{image_id}.xml",
        dior_root / "Annotations" / "Horizontal Bounding Boxes" / f"{image_id}.xml",
        dior_root / "Annotations" / "HorizontalBoundingBoxes" / f"{image_id}.xml",
        dior_root / "Annotations" / "horizontal" / f"{image_id}.xml",
        dior_root / "annotations" / f"{image_id}.xml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing annotation XML for image id: {image_id}")


def find_image(dior_root: Path, image_id: str, xml_root: ET.Element) -> Path:
    filename = xml_root.findtext("filename")
    names = [filename] if filename else []
    names.extend(f"{image_id}{suffix}" for suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    folders = [
        "JPEGImages",
        "JPEGImages-trainval",
        "JPEGImages-test",
        "JPEGImages_trainval",
        "JPEGImages_test",
        "images",
        "Images",
        "JPEG",
        "",
    ]

    for folder in folders:
        for name in names:
            if not name:
                continue
            path = dior_root / folder / name
            if path.exists():
                return path

    raise FileNotFoundError(f"Missing image file for image id: {image_id}")


def parse_bbox(obj: ET.Element) -> list[int] | None:
    box = obj.find("bndbox")
    if box is None:
        return None

    values: list[int] = []
    for key in ["xmin", "ymin", "xmax", "ymax"]:
        text = box.findtext(key)
        if text is None:
            return None
        values.append(int(round(float(text))))
    return values


def parse_annotation(
    dior_root: Path,
    image_id: str,
    class_map: dict[str, str],
    keep_empty: bool,
) -> dict[str, Any] | None:
    ann_path = find_annotation(dior_root, image_id)
    xml_root = ET.parse(ann_path).getroot()
    image_path = find_image(dior_root, image_id, xml_root)

    objects: list[dict[str, Any]] = []
    for obj in xml_root.findall("object"):
        source_label = normalize_label(obj.findtext("name") or "")
        evidence_label = class_map.get(source_label)
        if evidence_label is None:
            continue

        bbox = parse_bbox(obj)
        if bbox is None:
            continue

        difficult = int(obj.findtext("difficult") or 0)
        objects.append(
            {
                "class_name": evidence_label,
                "source_label": source_label,
                "bbox_xyxy": bbox,
                "difficult": difficult,
            }
        )

    if not objects and not keep_empty:
        return None

    width = xml_root.findtext("size/width")
    height = xml_root.findtext("size/height")
    counts = Counter(obj["class_name"] for obj in objects)

    return {
        "dataset": "DIOR",
        "image_id": image_id,
        "image": str(image_path.resolve()),
        "annotation": str(ann_path.resolve()),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "evidence_objects": objects,
        "evidence_counts": dict(sorted(counts.items())),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small DIOR object-evidence manifest for the facility recognition pipeline."
    )
    parser.add_argument("--dior-root", required=True, help="DIOR dataset root directory.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON class mapping config.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output manifest directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="DIOR split names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max images per split after filtering.")
    parser.add_argument("--keep-empty", action="store_true", help="Keep images without selected evidence objects.")
    args = parser.parse_args()

    dior_root = Path(args.dior_root).expanduser().resolve()
    if not dior_root.exists():
        raise FileNotFoundError(f"DIOR root not found: {dior_root}")

    class_map = load_class_map(Path(args.config).expanduser())
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dior_root": str(dior_root),
        "selected_source_classes": sorted(class_map.keys()),
        "splits": {},
    }

    for split in args.splits:
        split_file = find_split_file(dior_root, split)
        image_ids = read_image_ids(split_file)
        records: list[dict[str, Any]] = []
        evidence_counts: Counter[str] = Counter()
        missing = 0

        for image_id in image_ids:
            try:
                record = parse_annotation(dior_root, image_id, class_map, args.keep_empty)
            except FileNotFoundError as exc:
                missing += 1
                print(f"[skip] {exc}", file=sys.stderr)
                continue

            if record is None:
                continue

            record["split"] = split
            records.append(record)
            evidence_counts.update(record["evidence_counts"])

            if args.limit is not None and len(records) >= args.limit:
                break

        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(records, out_path)

        summary["splits"][split] = {
            "split_file": str(split_file),
            "output": str(out_path),
            "records": len(records),
            "missing_files": missing,
            "evidence_counts": dict(sorted(evidence_counts.items())),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
