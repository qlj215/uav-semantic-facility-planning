#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "outputs" / "fmow_yolo_evidence" / "evidence.jsonl"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "clip_linear_probe_1000epochs" / "hf_clip_vit_b32" / "predictions.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "evidence_fusion_analysis"

EVIDENCE_RULES: dict[str, dict[str, float]] = {
    "airport": {"aircraft": 0.35, "airport_region": 0.35, "vehicle": 0.05},
    "airport_hangar": {"aircraft": 0.35, "airport_region": 0.15, "vehicle": 0.05},
    "airport_terminal": {"aircraft": 0.25, "airport_region": 0.25, "vehicle": 0.05},
    "runway": {"aircraft": 0.30, "airport_region": 0.30},
    "port": {"ship": 0.45, "harbor": 0.35, "storage_tank": 0.10},
    "shipyard": {"ship": 0.40, "harbor": 0.30, "storage_tank": 0.20},
    "storage_tank": {"storage_tank": 0.60},
    "oil_or_gas_facility": {"storage_tank": 0.35, "industrial_chimney": 0.25, "vehicle": 0.05},
    "factory_or_powerplant": {"industrial_chimney": 0.35, "storage_tank": 0.15, "vehicle": 0.05},
    "military_facility": {"aircraft": 0.25, "vehicle": 0.20, "storage_tank": 0.10},
    "electric_substation": {},
    "prison": {"vehicle": 0.05},
}

FOCUS_PAIRS = [
    ("shipyard", "port"),
    ("runway", "airport"),
    ("storage_tank", "oil_or_gas_facility"),
]


def image_key(path: str) -> str:
    p = Path(path)
    parts = p.parts
    if "val" in parts:
        idx = parts.index("val")
        if idx + 2 < len(parts):
            return "/".join(parts[idx : idx + 3])
    return f"{p.parent.name}/{p.name}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    predictions = {}
    for record in load_jsonl(path):
        key = image_key(record.get("image_path") or record.get("image") or "")
        predictions[key] = record
    return predictions


def support_score(label: str, detections: list[dict[str, Any]]) -> tuple[float, list[str]]:
    rules = EVIDENCE_RULES.get(label, {})
    if not rules:
        return 0.0, []

    score = 0.0
    details: list[str] = []
    for det in detections:
        cls = det.get("class_name", "")
        weight = rules.get(cls, 0.0)
        if weight <= 0:
            continue
        conf = float(det.get("confidence", 0.0))
        contribution = weight * conf
        score += contribution
        details.append(f"{cls}:{conf:.2f}*{weight:.2f}={contribution:.2f}")

    return min(score, 1.0), details


def best_evidence_label(detections: list[dict[str, Any]]) -> tuple[str, float]:
    best_label = ""
    best_score = 0.0
    for label in EVIDENCE_RULES:
        score, _ = support_score(label, detections)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def support_level(score: float, strong_threshold: float, moderate_threshold: float) -> str:
    if score >= strong_threshold:
        return "strong"
    if score >= moderate_threshold:
        return "moderate"
    if score > 0:
        return "weak"
    return "none"


def is_focus_pair(a: str, b: str) -> str:
    labels = {a, b}
    for x, y in FOCUS_PAIRS:
        if labels == {x, y}:
            return f"{x}/{y}"
    return ""


def analyze_record(
    evidence: dict[str, Any],
    prediction: dict[str, Any] | None,
    strong_threshold: float,
    moderate_threshold: float,
    review_margin: float,
) -> dict[str, Any]:
    detections = evidence.get("detections", [])
    true_label = evidence["facility_label"]
    pred_label = prediction.get("pred_label", "") if prediction else ""
    pred_conf = prediction.get("confidence", None) if prediction else None

    true_support, true_details = support_score(true_label, detections)
    pred_support, pred_details = support_score(pred_label, detections) if pred_label else (0.0, [])
    best_label, best_score = best_evidence_label(detections)

    if prediction:
        classification_correct = pred_label == true_label
        focus_pair = is_focus_pair(true_label, pred_label)
    else:
        classification_correct = None
        focus_pair = ""

    needs_review = False
    reasons: list[str] = []
    if pred_label:
        if best_label and best_label != pred_label and best_score >= moderate_threshold:
            needs_review = True
            reasons.append("evidence_supports_other_label")
        if pred_support < moderate_threshold and detections:
            needs_review = True
            reasons.append("weak_evidence_for_prediction")
        if pred_label != true_label and true_support > pred_support + review_margin:
            needs_review = True
            reasons.append("evidence_closer_to_true_label")
        if pred_label == true_label and pred_support >= moderate_threshold:
            reasons.append("evidence_supports_prediction")
    else:
        if true_support < moderate_threshold:
            needs_review = True
            reasons.append("weak_evidence_for_true_label")
        else:
            reasons.append("evidence_supports_true_label")

    if not detections:
        reasons.append("no_detection")

    return {
        "image": evidence["image"],
        "image_key": image_key(evidence["image"]),
        "true_label": true_label,
        "pred_label": pred_label,
        "pred_confidence": round(float(pred_conf), 6) if pred_conf is not None else "",
        "classification_correct": classification_correct if classification_correct is not None else "",
        "focus_pair": focus_pair,
        "num_detections": evidence.get("num_detections", len(detections)),
        "evidence_counts": evidence.get("evidence_counts", {}),
        "true_support": round(true_support, 6),
        "true_support_level": support_level(true_support, strong_threshold, moderate_threshold),
        "pred_support": round(pred_support, 6) if pred_label else "",
        "pred_support_level": support_level(pred_support, strong_threshold, moderate_threshold) if pred_label else "",
        "best_evidence_label": best_label,
        "best_evidence_score": round(best_score, 6),
        "needs_review_by_evidence": needs_review,
        "review_reasons": ";".join(dict.fromkeys(reasons)),
        "true_evidence_details": "; ".join(true_details[:12]),
        "pred_evidence_details": "; ".join(pred_details[:12]),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def flatten_counts(counts: dict[str, int]) -> str:
    return "; ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_label(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["true_label"]].append(record)

    rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items()):
        total = len(items)
        detections = sum(int(r["num_detections"]) for r in items)
        detected_images = sum(1 for r in items if int(r["num_detections"]) > 0)
        review = sum(1 for r in items if r["needs_review_by_evidence"])
        strong = sum(1 for r in items if r["true_support_level"] == "strong")
        moderate_or_strong = sum(1 for r in items if r["true_support_level"] in {"moderate", "strong"})
        avg_support = sum(float(r["true_support"]) for r in items) / max(1, total)
        counts = Counter()
        for r in items:
            counts.update(r["evidence_counts"])

        rows.append(
            {
                "facility_label": label,
                "images": total,
                "detected_images": detected_images,
                "detected_image_rate": round(detected_images / max(1, total), 6),
                "detections": detections,
                "avg_true_support": round(avg_support, 6),
                "moderate_or_strong_support_images": moderate_or_strong,
                "moderate_or_strong_support_rate": round(moderate_or_strong / max(1, total), 6),
                "strong_support_images": strong,
                "needs_review_images": review,
                "needs_review_rate": round(review / max(1, total), 6),
                "evidence_counts": flatten_counts(counts),
            }
        )
    return rows


def summarize_focus_pairs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for a, b in FOCUS_PAIRS:
        pair_name = f"{a}/{b}"
        items = [
            r
            for r in records
            if r["true_label"] in {a, b} and (not r["pred_label"] or r["pred_label"] in {a, b})
        ]
        if not items:
            continue

        wrong = [r for r in items if r["classification_correct"] is False]
        review = [r for r in items if r["needs_review_by_evidence"]]
        true_support_better = [
            r
            for r in wrong
            if r["pred_support"] != "" and float(r["true_support"]) > float(r["pred_support"])
        ]
        counts = Counter()
        for r in items:
            counts.update(r["evidence_counts"])

        pair_rows.append(
            {
                "focus_pair": pair_name,
                "images": len(items),
                "classification_errors": len(wrong),
                "review_images": len(review),
                "review_rate": round(len(review) / max(1, len(items)), 6),
                "wrong_cases_where_evidence_closer_to_true": len(true_support_better),
                "evidence_counts": flatten_counts(counts),
            }
        )

        selected = sorted(
            wrong or items,
            key=lambda r: (not r["needs_review_by_evidence"], -float(r["best_evidence_score"]), r["image_key"]),
        )[:12]
        for r in selected:
            row = dict(r)
            row["focus_pair"] = pair_name
            row["evidence_counts"] = flatten_counts(row["evidence_counts"])
            case_rows.append(row)
    return pair_rows, case_rows


def write_report(
    output_dir: Path,
    records: list[dict[str, Any]],
    facility_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    prediction_path: Path | None,
) -> None:
    total = len(records)
    detected = sum(1 for r in records if int(r["num_detections"]) > 0)
    review = sum(1 for r in records if r["needs_review_by_evidence"])
    moderate_or_strong = sum(1 for r in records if r["true_support_level"] in {"moderate", "strong"})
    has_predictions = any(r["pred_label"] for r in records)
    correct = sum(1 for r in records if r["classification_correct"] is True)
    pred_total = sum(1 for r in records if r["pred_label"])

    lines = [
        "# fMoW 设施分类 + YOLO 目标证据融合分析",
        "",
        "## 总体结果",
        "",
        f"- 图像数：{total}",
        f"- 有 YOLO 目标证据的图像数：{detected}，占比 {detected / max(1, total):.2%}",
        f"- 对真实设施标签达到中等或强证据支持的图像数：{moderate_or_strong}，占比 {moderate_or_strong / max(1, total):.2%}",
        f"- 按证据规则建议复核的图像数：{review}，占比 {review / max(1, total):.2%}",
    ]
    if has_predictions:
        lines.append(f"- 接入分类预测文件：`{prediction_path}`")
        lines.append(f"- 覆盖预测样本数：{pred_total}，分类正确数：{correct}")
    else:
        lines.append("- 未接入逐图分类预测文件，本次只做真实设施标签的证据支持度分析。")

    lines.extend(
        [
            "",
            "## 重点混淆对",
            "",
            "| 混淆对 | 图像数 | 分类错误数 | 建议复核数 | 证据更接近真实标签的错误样本数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['focus_pair']} | {row['images']} | {row['classification_errors']} | "
            f"{row['review_images']} | {row['wrong_cases_where_evidence_closer_to_true']} |"
        )

    lines.extend(
        [
            "",
            "## 类别级证据支持",
            "",
            "| 类别 | 图像数 | 有检测图像占比 | 中等/强支持占比 | 建议复核占比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in facility_rows:
        lines.append(
            f"| {row['facility_label']} | {row['images']} | {float(row['detected_image_rate']):.2%} | "
            f"{float(row['moderate_or_strong_support_rate']):.2%} | {float(row['needs_review_rate']):.2%} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "这一步不直接证明分类准确率提升，而是把 YOLO 检测结果转成可解释证据和复核信号。后续应结合具体错误样本，判断证据是否能解释或纠正 `shipyard/port`、`runway/airport`、`storage_tank/oil_or_gas_facility` 三组混淆。",
        ]
    )
    (output_dir / "fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fMoW facility labels/predictions with YOLO evidence.")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE), help="fMoW YOLO evidence JSONL.")
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
        help="Optional classification predictions JSONL. Use 'none' to disable.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output analysis directory.")
    parser.add_argument("--strong-threshold", type=float, default=0.45)
    parser.add_argument("--moderate-threshold", type=float, default=0.20)
    parser.add_argument("--review-margin", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evidence_path = Path(args.evidence).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    prediction_path = None if args.predictions.lower() == "none" else Path(args.predictions).expanduser()

    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence JSONL not found: {evidence_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_records = load_jsonl(evidence_path)
    predictions = load_predictions(prediction_path)

    fused_records = []
    missing_predictions = 0
    for evidence in evidence_records:
        pred = predictions.get(image_key(evidence["image"]))
        if predictions and pred is None:
            missing_predictions += 1
        fused_records.append(
            analyze_record(
                evidence,
                pred,
                strong_threshold=args.strong_threshold,
                moderate_threshold=args.moderate_threshold,
                review_margin=args.review_margin,
            )
        )

    facility_rows = summarize_by_label(fused_records)
    pair_rows, pair_case_rows = summarize_focus_pairs(fused_records)

    write_jsonl(fused_records, output_dir / "fused_records.jsonl")
    write_csv(facility_rows, output_dir / "facility_fusion_summary.csv")
    write_csv(pair_rows, output_dir / "focus_pair_summary.csv")
    write_csv(pair_case_rows, output_dir / "focus_pair_cases.csv")
    write_report(output_dir, fused_records, facility_rows, pair_rows, prediction_path)

    summary = {
        "evidence": str(evidence_path),
        "predictions": str(prediction_path) if prediction_path else None,
        "images": len(fused_records),
        "missing_predictions": missing_predictions,
        "strong_threshold": args.strong_threshold,
        "moderate_threshold": args.moderate_threshold,
        "review_margin": args.review_margin,
        "outputs": {
            "fused_records": str(output_dir / "fused_records.jsonl"),
            "facility_summary": str(output_dir / "facility_fusion_summary.csv"),
            "focus_pair_summary": str(output_dir / "focus_pair_summary.csv"),
            "focus_pair_cases": str(output_dir / "focus_pair_cases.csv"),
            "report": str(output_dir / "fusion_report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
