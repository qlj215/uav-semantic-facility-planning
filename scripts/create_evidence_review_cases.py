#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOCUS_CASES = ROOT / "outputs" / "evidence_fusion_analysis" / "focus_pair_cases.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evidence_hpe_review_cases"

FOCUS_PAIRS = [
    "shipyard/port",
    "runway/airport",
    "storage_tank/oil_or_gas_facility",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Focus case file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, count = item.split(":", 1)
        try:
            counts[key.strip()] = int(count.strip())
        except ValueError:
            continue
    return counts


def relative_image_path(image_key: str, image_path: str) -> str:
    if image_key.startswith("val/") or image_key.startswith("train/"):
        return image_key

    path = Path(image_path)
    parts = path.parts
    for split in ("val", "train"):
        if split in parts:
            idx = parts.index(split)
            if idx + 2 < len(parts):
                return "/".join(parts[idx : idx + 3])
    return image_key or image_path


def score_row(row: dict[str, str]) -> tuple[int, float, int]:
    reasons = row.get("review_reasons", "")
    correctable = int(
        "evidence_closer_to_true_label" in reasons
        or (
            row.get("best_evidence_label") == row.get("true_label")
            and "evidence_supports_other_label" in reasons
        )
    )
    has_detection = int(int(row.get("num_detections") or 0) > 0)
    try:
        evidence_score = float(row.get("best_evidence_score") or 0.0)
    except ValueError:
        evidence_score = 0.0
    return correctable, evidence_score, has_detection


def choose_cases(rows: list[dict[str, str]], per_pair: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pair = row.get("focus_pair", "")
        if pair in FOCUS_PAIRS:
            by_pair[pair].append(row)

    for pair in FOCUS_PAIRS:
        pair_rows = by_pair.get(pair, [])
        wrong_rows = [r for r in pair_rows if not parse_bool(r.get("classification_correct", ""))]
        candidates = wrong_rows or pair_rows

        chosen: list[dict[str, str]] = []
        pair_labels = pair.split("/")

        for label in pair_labels:
            label_candidates = sorted(
                [r for r in candidates if r.get("true_label") == label],
                key=score_row,
                reverse=True,
            )
            if label_candidates:
                chosen.append(label_candidates[0])
            if len(chosen) >= per_pair:
                break

        priority_groups = [
            lambda r: "evidence_closer_to_true_label" in r.get("review_reasons", ""),
            lambda r: r.get("best_evidence_label") == r.get("true_label")
            and "evidence_supports_other_label" in r.get("review_reasons", ""),
            lambda r: int(r.get("num_detections") or 0) > 0,
            lambda r: True,
        ]

        for keep in priority_groups:
            ranked = sorted(
                [r for r in candidates if keep(r) and r not in chosen],
                key=score_row,
                reverse=True,
            )
            for row in ranked:
                chosen.append(row)
                if len(chosen) >= per_pair:
                    break
            if len(chosen) >= per_pair:
                break

        selected.extend(chosen[:per_pair])
    return selected


def evidence_summary(row: dict[str, str]) -> str:
    counts = parse_counts(row.get("evidence_counts", ""))
    if counts:
        counts_text = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    else:
        counts_text = "no YOLO object detected"

    pred_label = row.get("pred_label", "")
    pred_support = row.get("pred_support", "")
    pred_level = row.get("pred_support_level", "")
    best_label = row.get("best_evidence_label", "") or "none"
    best_score = row.get("best_evidence_score", "0")
    reasons = row.get("review_reasons", "") or "none"
    return (
        f"YOLO detections: {counts_text}. "
        f"Baseline prediction: {pred_label} with evidence support {pred_support} ({pred_level}). "
        f"Evidence-rule best label: {best_label} with score {best_score}. "
        f"Review reasons: {reasons}."
    )


def to_case_record(index: int, row: dict[str, str]) -> dict[str, Any]:
    focus_pair = row.get("focus_pair", "")
    case_prefix = focus_pair.replace("/", "_").replace("_or_", "_")
    image = relative_image_path(row.get("image_key", ""), row.get("image", ""))
    return {
        "case_id": f"{case_prefix}_{index:02d}",
        "image": image,
        "source_image_path": row.get("image", ""),
        "label": row.get("true_label", ""),
        "focus_pair": focus_pair,
        "baseline_pred": row.get("pred_label", ""),
        "baseline_confidence": float(row.get("pred_confidence") or 0.0),
        "baseline_correct": parse_bool(row.get("classification_correct", "")),
        "num_detections": int(row.get("num_detections") or 0),
        "evidence_counts": parse_counts(row.get("evidence_counts", "")),
        "best_evidence_label": row.get("best_evidence_label", ""),
        "best_evidence_score": float(row.get("best_evidence_score") or 0.0),
        "true_support": float(row.get("true_support") or 0.0),
        "true_support_level": row.get("true_support_level", ""),
        "baseline_support": float(row.get("pred_support") or 0.0),
        "baseline_support_level": row.get("pred_support_level", ""),
        "needs_review_by_evidence": parse_bool(row.get("needs_review_by_evidence", "")),
        "review_reasons": [item for item in row.get("review_reasons", "").split(";") if item],
        "true_evidence_details": row.get("true_evidence_details", ""),
        "baseline_evidence_details": row.get("pred_evidence_details", ""),
        "evidence_summary": evidence_summary(row),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "focus_pair",
        "image",
        "label",
        "baseline_pred",
        "baseline_confidence",
        "baseline_correct",
        "num_detections",
        "evidence_counts",
        "best_evidence_label",
        "best_evidence_score",
        "needs_review_by_evidence",
        "review_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list)) else row[key] for key in fields})


def write_prompt_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Evidence-HPE case subset",
        "",
        "This file previews the small qualitative experiment cases.",
        "",
        "| case_id | focus_pair | true | baseline | YOLO evidence |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        counts = ", ".join(f"{k} x{v}" for k, v in sorted(row["evidence_counts"].items())) or "none"
        lines.append(
            f"| {row['case_id']} | {row['focus_pair']} | {row['label']} | "
            f"{row['baseline_pred']} | {counts} |"
        )
    lines.extend(
        [
            "",
            "Use `scripts/eval_vlm_evidence_review.py` to run Qwen2.5-VL on these cases.",
            "The prompt does not expose the true label to the model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small evidence-enhanced VLM/HPE review case subset.")
    parser.add_argument("--focus-cases", type=Path, default=DEFAULT_FOCUS_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-pair", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.per_pair <= 0:
        raise ValueError("--per-pair must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv(args.focus_cases)
    selected = choose_cases(raw_rows, args.per_pair)
    records = [to_case_record(idx + 1, row) for idx, row in enumerate(selected)]

    write_jsonl(args.output_dir / "case_subset.jsonl", records)
    write_csv(args.output_dir / "selected_cases.csv", records)
    write_prompt_preview(args.output_dir / "case_preview.md", records)

    summary = {
        "focus_cases": str(args.focus_cases),
        "output_dir": str(args.output_dir),
        "per_pair": args.per_pair,
        "cases": len(records),
        "focus_pairs": FOCUS_PAIRS,
        "outputs": {
            "case_subset": str(args.output_dir / "case_subset.jsonl"),
            "selected_cases": str(args.output_dir / "selected_cases.csv"),
            "case_preview": str(args.output_dir / "case_preview.md"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
