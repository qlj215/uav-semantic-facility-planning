#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_SUBSET = ROOT / "data" / "fmow_vlm_eval_subset.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "vlm_hpe"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

CLASSES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "electric_substation",
    "factory_or_powerplant",
    "military_facility",
    "oil_or_gas_facility",
    "port",
    "prison",
    "runway",
    "shipyard",
    "storage_tank",
]

COARSE_TO_FINE = {
    "aviation": ["airport", "airport_hangar", "airport_terminal", "runway"],
    "maritime": ["port", "shipyard"],
    "energy_industrial": [
        "storage_tank",
        "oil_or_gas_facility",
        "factory_or_powerplant",
        "electric_substation",
    ],
    "security_military": ["military_facility", "prison"],
    "other_uncertain": CLASSES,
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_images(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    done = set()
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = row.get("image")
            if isinstance(image, str):
                done.add(image)
    return done


def load_existing_predictions(predictions_path: Path) -> tuple[list[int], list[int], int]:
    if not predictions_path.exists():
        return [], [], 0
    y_true: list[int] = []
    y_pred: list[int] = []
    review_count = 0
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            true_label = row.get("true_label")
            pred_label = row.get("pred_label")
            if true_label in CLASSES and pred_label in CLASSES:
                y_true.append(CLASSES.index(str(true_label)))
                y_pred.append(CLASSES.index(str(pred_label)))
                parsed = row.get("parsed")
                if isinstance(parsed, dict) and bool(parsed.get("need_review", False)):
                    review_count += 1
    return y_true, y_pred, review_count


def resolve_image_path(data_root: Path, image_value: object) -> Path:
    if not isinstance(image_value, str):
        raise TypeError(f"Invalid image value: {image_value!r}")
    path = Path(image_value)
    if path.is_absolute():
        return path
    return data_root / path


def class_to_coarse(label: str) -> str:
    for coarse, fine_labels in COARSE_TO_FINE.items():
        if label in fine_labels and coarse != "other_uncertain":
            return coarse
    return "other_uncertain"


def build_flat_prompt() -> str:
    class_text = ", ".join(CLASSES)
    return f"""You are a remote sensing image analyst.
Given this aerial or satellite image, choose exactly one facility category from the list below:

[{class_text}]

Rules:
- Return JSON only.
- The label must be exactly one category name from the list.
- Do not add markdown, comments, or extra text.

Return this JSON schema:
{{"label": "...", "confidence": 0.0, "reason": "..."}}
"""


def build_hpe_prompt() -> str:
    coarse_lines = []
    for coarse, fine_labels in COARSE_TO_FINE.items():
        if coarse == "other_uncertain":
            continue
        coarse_lines.append(f"- {coarse}: {', '.join(fine_labels)}")
    coarse_text = "\n".join(coarse_lines)
    fine_text = ", ".join(CLASSES)
    return f"""You are a remote sensing image analyst.
Classify this aerial or satellite image using hierarchical prompt engineering.

Level 1: choose one coarse_label:
{coarse_text}
- other_uncertain: use only when the image is too ambiguous

Level 2: choose one fine_label from this exact list:
[{fine_text}]

Level 3: provide concise visual evidence and decide whether human review is needed.

Rules:
- Return JSON only.
- fine_label must be exactly one category name from the fine_label list.
- coarse_label must be consistent with fine_label unless the image is highly ambiguous.
- need_review should be true if the image may be confused with another facility class.
- Do not add markdown, comments, or extra text.

Return this JSON schema:
{{
  "coarse_label": "...",
  "fine_label": "...",
  "confidence": 0.0,
  "visual_evidence": ["..."],
  "uncertainty": "...",
  "need_review": false
}}
"""


def extract_json_object(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def parse_prediction(raw_text: str, prompt_mode: str) -> tuple[dict[str, object], str]:
    obj = extract_json_object(raw_text)
    if prompt_mode == "flat":
        label = obj.get("label")
    else:
        label = obj.get("fine_label")

    if not isinstance(label, str):
        raise ValueError("Missing predicted label")
    label = label.strip()
    if label not in CLASSES:
        raise ValueError(f"Invalid predicted label: {label!r}")

    obj["pred_label"] = label
    if prompt_mode == "hpe":
        coarse = obj.get("coarse_label")
        if not isinstance(coarse, str) or coarse not in COARSE_TO_FINE:
            obj["coarse_label"] = class_to_coarse(label)
    return obj, label


def compute_confusion(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, object]:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / total if total else 0.0
    f1_scores = []
    recalls: dict[str, float] = {}
    for idx, class_name in enumerate(CLASSES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
        recalls[class_name] = recall
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class_recall": recalls,
    }


def write_confusion_csv(path: Path, cm: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASSES)
        for idx, class_name in enumerate(CLASSES):
            writer.writerow([class_name] + cm[idx].tolist())


def build_messages(image_path: Path, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path.resolve()}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class Qwen25VLRunner:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        min_pixels: int | None,
        max_pixels: int | None,
        use_flash_attention: bool,
    ):
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL inference requires transformers and qwen-vl-utils. "
                "Install on AutoDL with: pip install -U transformers accelerate qwen-vl-utils"
            ) from exc

        self.max_new_tokens = max_new_tokens
        self.process_vision_info = process_vision_info
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    @torch.no_grad()
    def generate(self, image_path: Path, prompt: str) -> str:
        messages = build_messages(image_path, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL flat/HPE prompts on the fixed fMoW VLM subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-mode", choices=["flat", "hpe"], required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Skip images already present in predictions.jsonl.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Limit visual tokens for speed and memory. Qwen examples use values like 1280*28*28.",
    )
    parser.add_argument("--flash-attn", action="store_true", help="Use flash_attention_2 if installed.")
    args = parser.parse_args()

    samples = read_jsonl(args.subset)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(f"No samples found in {args.subset}")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"qwen2_5_vl_7b_{args.prompt_mode}"

    prompt = build_flat_prompt() if args.prompt_mode == "flat" else build_hpe_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    bad_outputs_path = args.output_dir / "bad_outputs.jsonl"
    done_images = load_done_images(predictions_path) if args.resume else set()

    runner = Qwen25VLRunner(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_flash_attention=args.flash_attn,
    )

    y_true, y_pred, existing_review_count = load_existing_predictions(predictions_path) if args.resume else ([], [], 0)
    new_seen = 0
    new_parsed_ok = 0
    bad_count = 0
    review_count = existing_review_count
    start_time = time.time()

    print(f"Model: {args.model_id}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Subset: {args.subset}")
    print(f"Samples: {len(samples)}")
    print(f"Output: {args.output_dir}")

    for idx, row in enumerate(samples, start=1):
        image_key = str(row["image"])
        if image_key in done_images:
            continue

        true_label = str(row["label"])
        if true_label not in CLASSES:
            raise ValueError(f"Invalid true label in subset: {true_label!r}")
        image_path = resolve_image_path(args.data_root, row["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        new_seen += 1
        try:
            raw_text = runner.generate(image_path, prompt)
            parsed, pred_label = parse_prediction(raw_text, args.prompt_mode)
            new_parsed_ok += 1
            if bool(parsed.get("need_review", False)):
                review_count += 1

            true_idx = CLASSES.index(true_label)
            pred_idx = CLASSES.index(pred_label)
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            append_jsonl(
                predictions_path,
                {
                    "image": image_key,
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": pred_label == true_label,
                    "prompt_mode": args.prompt_mode,
                    "parsed": parsed,
                    "raw_output": raw_text,
                },
            )
        except Exception as exc:
            bad_count += 1
            append_jsonl(
                bad_outputs_path,
                {
                    "image": image_key,
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "prompt_mode": args.prompt_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        if idx == 1 or idx % 10 == 0:
            print(f"processed={idx}/{len(samples)} new_parsed_ok={new_parsed_ok} bad={bad_count}")

    if y_true:
        cm = compute_confusion(y_true, y_pred, len(CLASSES))
        final_metrics = metrics_from_confusion(cm)
    else:
        cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        final_metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_recall": {class_name: 0.0 for class_name in CLASSES},
        }

    elapsed = time.time() - start_time
    total_expected = len(samples)
    total_valid_predictions = len(y_true)
    total_attempted_this_run = new_seen
    parse_success_rate = total_valid_predictions / total_expected if total_expected else 0.0
    review_rate = review_count / total_valid_predictions if total_valid_predictions else 0.0
    result = {
        "model": args.model_id,
        "prompt_mode": args.prompt_mode,
        "data_root": str(args.data_root),
        "subset": str(args.subset),
        "classes": CLASSES,
        "samples_requested": len(samples),
        "samples_attempted_this_run": total_attempted_this_run,
        "valid_predictions_this_run": new_parsed_ok,
        "valid_predictions_total": total_valid_predictions,
        "bad_outputs_this_run": bad_count,
        "parse_success_rate": parse_success_rate,
        "review_rate": review_rate,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "elapsed_seconds": round(elapsed, 2),
        "final": final_metrics,
        "note": "Metrics are computed for predictions produced in this run. Use a fresh output dir for final reporting.",
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", cm)

    print(f"accuracy={final_metrics['accuracy'] * 100:.2f}%")
    print(f"macro_f1={final_metrics['macro_f1'] * 100:.2f}%")
    print(f"parse_success_rate={parse_success_rate * 100:.2f}%")
    print(f"review_rate={review_rate * 100:.2f}%")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved bad outputs: {bad_outputs_path}")
    print(f"Saved confusion matrix: {args.output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
