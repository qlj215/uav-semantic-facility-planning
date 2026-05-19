#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from eval_vlm_hpe import (
    ABSTAIN_LABEL,
    CLASSES,
    DEFAULT_MODEL_ID,
    PRED_CLASSES,
    Qwen25VLRunner,
    append_jsonl,
    compute_confusion,
    extract_json_object,
    metrics_from_confusion,
    write_confusion_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_SUBSET = ROOT / "outputs" / "evidence_hpe_review_cases" / "case_subset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evidence_hpe_review_cases" / "qwen2_5_vl_7b_evidence_review"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_predictions(path: Path) -> tuple[set[str], list[int], list[int], int]:
    done: set[str] = set()
    y_true: list[int] = []
    y_pred: list[int] = []
    review_count = 0
    if not path.exists():
        return done, y_true, y_pred, review_count

    for row in read_jsonl(path):
        case_id = str(row.get("case_id", ""))
        true_label = str(row.get("true_label", ""))
        pred_label = str(row.get("final_label", ""))
        if case_id:
            done.add(case_id)
        if true_label in CLASSES and pred_label in PRED_CLASSES:
            y_true.append(CLASSES.index(true_label))
            y_pred.append(PRED_CLASSES.index(pred_label))
            parsed = row.get("parsed", {})
            if isinstance(parsed, dict) and bool(parsed.get("need_review", False)):
                review_count += 1
    return done, y_true, y_pred, review_count


def resolve_image_path(data_root: Path, case: dict[str, Any]) -> Path:
    relative = Path(str(case["image"]))
    candidate = data_root / relative
    if candidate.exists():
        return candidate

    source = Path(str(case.get("source_image_path") or ""))
    if source.is_absolute() and source.exists():
        return source

    return candidate


def counts_text(counts: dict[str, Any]) -> str:
    if not counts:
        return "no YOLO object detected"
    return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))


def build_prompt(case: dict[str, Any]) -> str:
    class_text = ", ".join(CLASSES)
    evidence_counts = counts_text(case.get("evidence_counts", {}))
    review_reasons = ", ".join(case.get("review_reasons", [])) or "none"
    return f"""You are a remote sensing image analyst doing evidence-based review.

Classify the image into exactly one facility category:
[{class_text}]

You are also given auxiliary evidence from a YOLO object detector and a previous classifier.
Use the image as the primary source. Treat YOLO evidence as helpful but possibly noisy.
Do not assume the previous classifier is correct.

Previous classifier:
- predicted label: {case.get("baseline_pred", "")}
- confidence: {float(case.get("baseline_confidence", 0.0)):.3f}

YOLO object evidence:
- detected objects: {evidence_counts}
- evidence-rule best label: {case.get("best_evidence_label") or "none"}
- evidence-rule score: {float(case.get("best_evidence_score", 0.0)):.3f}
- evidence support for previous classifier: {float(case.get("baseline_support", 0.0)):.3f} ({case.get("baseline_support_level", "none")})
- review trigger: {review_reasons}

Key distinction rules:
- port is a harbor/logistics area; shipyard includes ship construction or repair structures such as docks, slips, cranes, or repair basins.
- runway means the runway itself dominates the image; airport means the broader airport region.
- storage_tank is dominated by tanks; oil_or_gas_facility needs tanks plus processing, pipeline, refinery, or industrial context.

Return JSON only:
{{
  "final_label": "...",
  "confidence": 0.0,
  "supports_previous_classifier": false,
  "uses_yolo_evidence": true,
  "need_review": false,
  "evidence_interpretation": "...",
  "reason": "..."
}}

Rules:
- final_label must be one category from the list, or "other_uncertain" if the image remains too ambiguous.
- Do not mention the ground-truth label.
- Keep evidence_interpretation and reason concise.
"""


def parse_review_output(raw_text: str) -> tuple[dict[str, Any], str]:
    obj = extract_json_object(raw_text)
    label = obj.get("final_label") or obj.get("fine_label") or obj.get("label")
    if not isinstance(label, str):
        raise ValueError("Missing final_label")
    label = label.strip()
    if label in {"", "unknown", "uncertain", "other"}:
        label = ABSTAIN_LABEL
    if label not in PRED_CLASSES:
        raise ValueError(f"Invalid final_label: {label!r}")
    obj["final_label"] = label
    if label == ABSTAIN_LABEL:
        obj["need_review"] = True
    return obj, label


def write_case_report(output_dir: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    lines = [
        "# YOLO Evidence + VLM/HPE Review Case Experiment",
        "",
        "## Summary",
        "",
        f"- Cases: {len(rows)}",
        f"- Accuracy: {metrics['final']['accuracy']:.2%}",
        f"- Macro-F1: {metrics['final']['macro_f1']:.2%}",
        f"- Baseline correct cases: {metrics['baseline_correct']}",
        f"- VLM final correct cases: {metrics['vlm_correct']}",
        f"- Baseline errors fixed by evidence review: {metrics['fixed_baseline_errors']}",
        f"- Correct baseline cases broken by evidence review: {metrics['broken_baseline_correct']}",
        f"- VLM review rate: {metrics['review_rate']:.2%}",
        "",
        "## Cases",
        "",
        "| case_id | focus_pair | true | baseline | VLM final | fixed? | review? | evidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        parsed = row.get("parsed", {})
        evidence = ""
        if isinstance(parsed, dict):
            evidence = str(parsed.get("evidence_interpretation") or parsed.get("reason") or "")
        evidence = evidence.replace("|", "/")[:120]
        lines.append(
            f"| {row['case_id']} | {row['focus_pair']} | {row['true_label']} | "
            f"{row['baseline_pred']} | {row['final_label']} | {row['fixed_baseline_error']} | "
            f"{row['need_review']} | {evidence} |"
        )
    output_dir.joinpath("case_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "case_id",
        "focus_pair",
        "image",
        "true_label",
        "baseline_pred",
        "baseline_correct",
        "final_label",
        "vlm_correct",
        "fixed_baseline_error",
        "broken_baseline_correct",
        "need_review",
        "raw_output",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL evidence-enhanced review on selected fMoW cases.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-source",
        choices=["huggingface", "modelscope", "local"],
        default="huggingface",
    )
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--flash-attn", action="store_true")
    args = parser.parse_args()

    cases = read_jsonl(args.subset)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError(f"No cases found in {args.subset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    bad_outputs_path = args.output_dir / "bad_outputs.jsonl"
    if args.resume:
        done, y_true, y_pred, review_count = load_existing_predictions(predictions_path)
    else:
        done, y_true, y_pred, review_count = set(), [], [], 0
        if predictions_path.exists():
            predictions_path.unlink()
        if bad_outputs_path.exists():
            bad_outputs_path.unlink()

    runner = Qwen25VLRunner(
        model_id=args.model_id,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_flash_attention=args.flash_attn,
    )

    all_rows = read_jsonl(predictions_path) if predictions_path.exists() else []
    start = time.time()
    bad_count = 0
    print(f"Model: {args.model_id}")
    print(f"Model source: {args.model_source}")
    print(f"Resolved model path: {runner.model_path}")
    print(f"Subset: {args.subset}")
    print(f"Cases: {len(cases)}")
    print(f"Output: {args.output_dir}")

    for idx, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if case_id in done:
            continue

        true_label = str(case["label"])
        if true_label not in CLASSES:
            raise ValueError(f"Invalid true label: {true_label}")
        image_path = resolve_image_path(args.data_root, case)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {case_id}: {image_path}")

        prompt = build_prompt(case)
        raw_text = ""
        try:
            raw_text = runner.generate(image_path, prompt)
            parsed, final_label = parse_review_output(raw_text)
            true_idx = CLASSES.index(true_label)
            pred_idx = PRED_CLASSES.index(final_label)
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            if bool(parsed.get("need_review", False)):
                review_count += 1

            baseline_pred = str(case.get("baseline_pred", ""))
            baseline_correct = baseline_pred == true_label
            vlm_correct = final_label == true_label
            row = {
                "case_id": case_id,
                "image": case["image"],
                "image_path": str(image_path),
                "focus_pair": case.get("focus_pair", ""),
                "true_label": true_label,
                "baseline_pred": baseline_pred,
                "baseline_confidence": case.get("baseline_confidence", 0.0),
                "baseline_correct": baseline_correct,
                "final_label": final_label,
                "vlm_correct": vlm_correct,
                "fixed_baseline_error": (not baseline_correct) and vlm_correct,
                "broken_baseline_correct": baseline_correct and (not vlm_correct),
                "need_review": bool(parsed.get("need_review", False)),
                "parsed": parsed,
                "prompt": prompt,
                "raw_output": raw_text,
            }
            append_jsonl(predictions_path, row)
            all_rows.append(row)
        except Exception as exc:
            bad_count += 1
            append_jsonl(
                bad_outputs_path,
                {
                    "case_id": case_id,
                    "image": case.get("image", ""),
                    "true_label": true_label,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output": raw_text,
                },
            )

        print(f"processed={idx}/{len(cases)} valid={len(y_true)} bad={bad_count}")

    cm = compute_confusion(y_true, y_pred, len(CLASSES), len(PRED_CLASSES))
    final = metrics_from_confusion(cm)
    baseline_correct = sum(1 for row in all_rows if row.get("baseline_correct") is True)
    vlm_correct = sum(1 for row in all_rows if row.get("vlm_correct") is True)
    fixed = sum(1 for row in all_rows if row.get("fixed_baseline_error") is True)
    broken = sum(1 for row in all_rows if row.get("broken_baseline_correct") is True)
    metrics = {
        "model": args.model_id,
        "model_source": args.model_source,
        "resolved_model_path": runner.model_path,
        "subset": str(args.subset),
        "data_root": str(args.data_root),
        "cases": len(cases),
        "valid_predictions": len(y_true),
        "bad_outputs_this_run": bad_count,
        "baseline_correct": baseline_correct,
        "vlm_correct": vlm_correct,
        "fixed_baseline_errors": fixed,
        "broken_baseline_correct": broken,
        "review_count": review_count,
        "review_rate": review_count / len(y_true) if y_true else 0.0,
        "elapsed_seconds": round(time.time() - start, 2),
        "final": final,
        "note": "Small qualitative case experiment. Use it to discuss evidence-based review, not as a full benchmark.",
    }
    args.output_dir.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_dir.joinpath("prompt_template.txt").write_text(build_prompt(cases[0]), encoding="utf-8")
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", cm)
    write_case_csv(args.output_dir / "case_review_summary.csv", all_rows)
    write_case_report(args.output_dir, all_rows, metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
