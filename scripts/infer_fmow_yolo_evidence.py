#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "fmow_yolo_evidence"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


MODEL_CANDIDATES = [
    ROOT / "outputs" / "dior_yolo" / "yolov8n_evidence_min" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "outputs" / "dior_yolo" / "yolov8n_evidence_min" / "weights" / "best.pt",
    ROOT / "yolov8n_evidence_min" / "weights" / "best.pt",
]


def resolve_model(model_arg: str | None) -> Path:
    if model_arg:
        path = Path(model_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"YOLO model not found: {path}")
        return path

    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n".join(str(p) for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "YOLO model not found. Pass --model explicitly.\n"
        f"Common paths checked:\n{candidates}"
    )


def collect_imagefolder_samples(data_root: Path, split: str, max_images: int | None = None) -> list[dict[str, str]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    samples: list[dict[str, str]] = []
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            samples.append(
                {
                    "image": str(image_path),
                    "split": split,
                    "facility_label": class_dir.name,
                }
            )
            if max_images is not None and len(samples) >= max_images:
                return samples

    if not samples:
        raise RuntimeError(f"No images found under {split_root}")
    return samples


def batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def detection_records(result: Any) -> list[dict[str, Any]]:
    names = result.names
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    for box, conf, cls_id in zip(xyxy, confs, classes):
        cls_id_int = int(cls_id)
        class_name = names.get(cls_id_int, str(cls_id_int)) if isinstance(names, dict) else str(cls_id_int)
        detections.append(
            {
                "class_name": class_name,
                "confidence": round(float(conf), 6),
                "bbox_xyxy": [round(float(v), 2) for v in box],
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def summarize_record(sample: dict[str, str], detections: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_counts = Counter(d["class_name"] for d in detections)
    max_confidence: dict[str, float] = {}
    for det in detections:
        cls = det["class_name"]
        max_confidence[cls] = max(max_confidence.get(cls, 0.0), float(det["confidence"]))

    return {
        "image": sample["image"],
        "split": sample["split"],
        "facility_label": sample["facility_label"],
        "detections": detections,
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "max_confidence_by_class": {k: round(v, 6) for k, v in sorted(max_confidence.items())},
        "num_detections": len(detections),
    }


def color_for_class(class_name: str) -> tuple[int, int, int]:
    palette = [
        (35, 94, 232),
        (229, 64, 126),
        (19, 174, 128),
        (245, 166, 35),
        (122, 84, 226),
        (214, 75, 44),
        (0, 154, 201),
        (111, 170, 33),
    ]
    return palette[sum(ord(ch) for ch in class_name) % len(palette)]


def draw_detections(image_path: Path, detections: list[dict[str, Any]], out_path: Path, max_boxes: int) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections[:max_boxes]:
        x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = color_for_class(str(det["class_name"]))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        y_text = max(0, y1 - text_h - 4)
        draw.rectangle((x1, y_text, x1 + text_w + 6, y_text + text_h + 4), fill=color)
        draw.text((x1 + 3, y_text + 2), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_facility_summary(records: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "images": 0,
            "images_with_detections": 0,
            "detections": 0,
            "evidence_counts": Counter(),
        }
    )
    for record in records:
        label = record["facility_label"]
        stats[label]["images"] += 1
        stats[label]["detections"] += record["num_detections"]
        if record["num_detections"]:
            stats[label]["images_with_detections"] += 1
        stats[label]["evidence_counts"].update(record["evidence_counts"])

    rows: list[dict[str, Any]] = []
    all_evidence = sorted({k for v in stats.values() for k in v["evidence_counts"]})
    for label in sorted(stats):
        item = stats[label]
        row: dict[str, Any] = {
            "facility_label": label,
            "images": item["images"],
            "images_with_detections": item["images_with_detections"],
            "detection_image_rate": round(item["images_with_detections"] / max(1, item["images"]), 6),
            "detections": item["detections"],
        }
        for evidence in all_evidence:
            row[f"count_{evidence}"] = item["evidence_counts"].get(evidence, 0)
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def save_visual_samples(
    records: list[dict[str, Any]],
    output_dir: Path,
    sample_per_class: int,
    max_boxes: int,
) -> list[str]:
    saved: list[str] = []
    used_by_class: Counter[str] = Counter()
    visual_dir = output_dir / "visualizations"

    candidates = sorted(
        [r for r in records if r["num_detections"] > 0],
        key=lambda r: (r["facility_label"], -r["num_detections"], r["image"]),
    )
    for record in candidates:
        label = record["facility_label"]
        if used_by_class[label] >= sample_per_class:
            continue
        used_by_class[label] += 1
        image_path = Path(record["image"])
        out_name = f"{used_by_class[label]:02d}_{image_path.stem}.jpg"
        out_path = visual_dir / label / out_name
        draw_detections(image_path, record["detections"], out_path, max_boxes=max_boxes)
        saved.append(str(out_path))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DIOR-trained YOLO evidence detector on fMoW ImageFolder data.")
    parser.add_argument("--model", default=None, help="Path to YOLO best.pt. If omitted, common paths are checked.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="fMoW ImageFolder root.")
    parser.add_argument("--split", default="val", help="ImageFolder split to run, usually val.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="YOLO inference batch size.")
    parser.add_argument("--device", default="0", help="YOLO device, e.g. 0 or cpu.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--sample-per-class", type=int, default=4, help="Visualization samples per fMoW class.")
    parser.add_argument("--max-boxes-per-image", type=int, default=80, help="Max boxes drawn on each visualization.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install it with: python3 -m pip install -U ultralytics") from exc

    model_path = resolve_model(args.model)
    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_imagefolder_samples(data_root, args.split, args.max_images)
    model = YOLO(str(model_path))
    records: list[dict[str, Any]] = []

    for batch in batched(samples, args.batch_size):
        paths = [item["image"] for item in batch]
        results = model.predict(
            source=paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        for sample, result in zip(batch, results):
            detections = detection_records(result)
            records.append(summarize_record(sample, detections))

    evidence_path = output_dir / "evidence.jsonl"
    write_jsonl(records, evidence_path)
    facility_rows = write_facility_summary(records, output_dir / "facility_evidence_summary.csv")
    saved_visuals = save_visual_samples(
        records,
        output_dir,
        sample_per_class=args.sample_per_class,
        max_boxes=args.max_boxes_per_image,
    )

    evidence_counts = Counter()
    for record in records:
        evidence_counts.update(record["evidence_counts"])

    summary = {
        "model": str(model_path),
        "data_root": str(data_root),
        "split": args.split,
        "images": len(records),
        "images_with_detections": sum(1 for r in records if r["num_detections"] > 0),
        "detections": sum(r["num_detections"] for r in records),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "evidence_jsonl": str(evidence_path),
        "facility_summary_csv": str(output_dir / "facility_evidence_summary.csv"),
        "visualizations": saved_visuals,
        "facility_rows": facility_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
