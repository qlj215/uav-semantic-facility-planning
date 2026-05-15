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


def infer_dataset_root(path: Path, fallback: Path) -> Path:
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() == "imagesets" and i > 0:
            return Path(*parts[:i])
    return fallback


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def load_class_map(config_path: Path) -> dict[str, str]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_map = data.get("source_class_to_evidence", {})
    return {normalize_label(k): normalize_label(v) for k, v in raw_map.items()}


def split_name_candidates(split: str) -> list[str]:
    names = [split]
    if split == "train":
        names.append("trainval")
    elif split in {"val", "valid", "validation"}:
        names.extend(["val", "test"])
    return list(dict.fromkeys(names))


def find_split_file(dior_root: Path, split: str) -> tuple[Path, Path, str]:
    candidates: list[Path] = []
    for name in split_name_candidates(split):
        candidates.extend(
            [
                dior_root / "ImageSets" / "Main" / f"{name}.txt",
                dior_root / "ImageSets" / f"{name}.txt",
                dior_root / f"{name}.txt",
            ]
        )

    for path in candidates:
        if path.exists():
            return path, infer_dataset_root(path, dior_root), path.stem

    recursive_matches: list[Path] = []
    for name in split_name_candidates(split):
        recursive_matches.extend(dior_root.rglob(f"{name}.txt"))
    recursive_matches = sorted(
        recursive_matches,
        key=lambda p: ("imagesets" not in str(p).lower(), len(p.parts), str(p)),
    )
    if recursive_matches:
        path = recursive_matches[0]
        return path, infer_dataset_root(path, dior_root), path.stem

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Cannot find split file for '{split}'. Searched:\n{searched}\n"
        "If your DIOR copy has no split txt files, rerun with --scan-all."
    )


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

    matches = sorted(dior_root.rglob(f"{image_id}.xml"))
    if matches:
        matches.sort(
            key=lambda p: (
                "annotations" not in str(p).lower(),
                "horizontal" not in str(p).lower(),
                len(p.parts),
                str(p),
            )
        )
        return matches[0]

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

    recursive_matches: list[Path] = []
    for name in names:
        if name:
            recursive_matches.extend(dior_root.rglob(name))
    if recursive_matches:
        recursive_matches.sort(key=lambda p: ("jpegimages" not in str(p).lower(), len(p.parts), str(p)))
        return recursive_matches[0]

    raise FileNotFoundError(f"Missing image file for image id: {image_id}")


def collect_annotation_image_ids(dior_root: Path) -> list[str]:
    candidates: list[Path] = []
    for folder in [
        "Annotations",
        "Annotations/Horizontal Bounding Boxes",
        "Annotations/HorizontalBoundingBoxes",
        "Annotations/horizontal",
        "annotations",
    ]:
        path = dior_root / folder
        if path.exists():
            candidates.extend(path.rglob("*.xml"))

    if not candidates:
        candidates = list(dior_root.rglob("*.xml"))

    image_ids = {path.stem for path in candidates if "annotation" in str(path).lower()}
    if not image_ids:
        image_ids = {path.stem for path in candidates}
    return sorted(image_ids)


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
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Ignore split files and scan all annotation XML files into all.jsonl.",
    )
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

    if args.scan_all:
        image_ids = collect_annotation_image_ids(dior_root)
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

            record["split"] = "all"
            records.append(record)
            evidence_counts.update(record["evidence_counts"])

            if args.limit is not None and len(records) >= args.limit:
                break

        out_path = output_dir / "all.jsonl"
        write_jsonl(records, out_path)
        summary["splits"]["all"] = {
            "split_file": None,
            "output": str(out_path),
            "records": len(records),
            "missing_files": missing,
            "evidence_counts": dict(sorted(evidence_counts.items())),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    for split in args.splits:
        split_file, split_root, actual_split = find_split_file(dior_root, split)
        if split_root != dior_root:
            print(f"[info] using nested DIOR root for split '{split}': {split_root}", file=sys.stderr)
        if actual_split != split:
            print(f"[info] using split file '{actual_split}.txt' for requested split '{split}'", file=sys.stderr)
        image_ids = read_image_ids(split_file)
        records: list[dict[str, Any]] = []
        evidence_counts: Counter[str] = Counter()
        missing = 0

        for image_id in image_ids:
            try:
                record = parse_annotation(split_root, image_id, class_map, args.keep_empty)
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
            "actual_split": actual_split,
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
