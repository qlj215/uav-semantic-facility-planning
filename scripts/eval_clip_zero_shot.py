#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_OUTPUT = ROOT / "outputs" / "clip_zero_shot"

DEFAULT_PROMPTS = [
    "a satellite image of a {label}.",
    "an overhead remote sensing image of a {label}.",
    "an aerial image showing a {label}.",
]


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.draft("RGB", (512, 512))
    return image.convert("RGB")


def list_samples(data_root: Path, split: str) -> Tuple[List[str], List[Tuple[Path, int]]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    classes = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
    samples: List[Tuple[Path, int]] = []
    for idx, class_name in enumerate(classes):
        for image_path in sorted((split_root / class_name).glob("*.jpg")):
            samples.append((image_path, idx))

    if not samples:
        raise RuntimeError(f"No jpg images found under {split_root}")
    return classes, samples


def label_to_text(label: str) -> str:
    return label.replace("_", " ")


def build_prompts(classes: Sequence[str], prompt_templates: Sequence[str]) -> List[List[str]]:
    prompts_by_class: List[List[str]] = []
    for class_name in classes:
        label = label_to_text(class_name)
        prompts_by_class.append([template.format(label=label) for template in prompt_templates])
    return prompts_by_class


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_confusion(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / total if total else 0.0
    f1_scores = []
    for idx in range(cm.shape[0]):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


def write_confusion_csv(path: Path, cm: np.ndarray, classes: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(classes))
        for idx, class_name in enumerate(classes):
            writer.writerow([class_name] + cm[idx].tolist())


class TransformersClipBackend:
    def __init__(self, model_id: str, device: torch.device):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for --backend transformers. "
                "Install it with: pip install transformers"
            ) from exc

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, prompts_by_class: Sequence[Sequence[str]]) -> torch.Tensor:
        class_features = []
        for prompts in prompts_by_class:
            inputs = self.processor(text=list(prompts), return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = normalize(features)
            class_features.append(normalize(features.mean(dim=0, keepdim=True)))
        return torch.cat(class_features, dim=0)

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return normalize(features)


class OpenClipBackend:
    def __init__(self, model_name: str, pretrained: str, device: torch.device):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip is required for --backend open_clip. "
                "Install it with: pip install open-clip-torch"
            ) from exc

        self.device = device
        self.open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, prompts_by_class: Sequence[Sequence[str]]) -> torch.Tensor:
        class_features = []
        for prompts in prompts_by_class:
            tokens = self.tokenizer(list(prompts)).to(self.device)
            features = self.model.encode_text(tokens)
            features = normalize(features)
            class_features.append(normalize(features.mean(dim=0, keepdim=True)))
        return torch.cat(class_features, dim=0)

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        features = self.model.encode_image(batch)
        return normalize(features)


def batched(items: Sequence[Tuple[Path, int]], batch_size: int) -> Iterable[Sequence[Tuple[Path, int]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot CLIP/RemoteCLIP baseline for fMoW key subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "open_clip"])
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32", help="HF CLIP model id.")
    parser.add_argument("--open-clip-model", type=str, default="ViT-B-32", help="OpenCLIP model architecture.")
    parser.add_argument(
        "--open-clip-pretrained",
        type=str,
        default="openai",
        help="OpenCLIP pretrained tag, local checkpoint path, or hf-hub model id.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-template", action="append", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    prompt_templates = args.prompt_template or DEFAULT_PROMPTS
    classes, samples = list_samples(args.data_root, args.split)
    prompts_by_class = build_prompts(classes, prompt_templates)

    if args.backend == "transformers":
        backend = TransformersClipBackend(args.model_id, device)
        model_desc = args.model_id
    else:
        backend = OpenClipBackend(args.open_clip_model, args.open_clip_pretrained, device)
        model_desc = f"{args.open_clip_model}:{args.open_clip_pretrained}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(samples)}")
    print(f"Classes: {len(classes)}")
    print(f"Backend: {args.backend}")
    print(f"Model: {model_desc}")
    print(f"Device: {device}")

    start_time = time.time()
    text_features = backend.encode_text(prompts_by_class)
    y_true: List[int] = []
    y_pred: List[int] = []
    predictions = []

    for batch in batched(samples, args.batch_size):
        paths = [path for path, _ in batch]
        labels = [label for _, label in batch]
        images = [load_image(path) for path in paths]
        image_features = backend.encode_images(images)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(preds)
        for path, label, pred, conf in zip(paths, labels, preds, confs):
            predictions.append(
                {
                    "image": str(path),
                    "true_label": classes[label],
                    "pred_label": classes[pred],
                    "confidence": float(conf),
                }
            )

    cm = compute_confusion(y_true, y_pred, len(classes))
    final_metrics = metrics_from_confusion(cm)
    elapsed = time.time() - start_time
    result = {
        "data_root": str(args.data_root),
        "split": args.split,
        "classes": classes,
        "samples": len(samples),
        "backend": args.backend,
        "model": model_desc,
        "device": str(device),
        "batch_size": args.batch_size,
        "prompt_templates": prompt_templates,
        "elapsed_seconds": round(elapsed, 2),
        "final": final_metrics,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", cm, classes)

    print(f"accuracy={final_metrics['accuracy'] * 100:.2f}%")
    print(f"macro_f1={final_metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {args.output_dir / 'confusion_matrix.csv'}")
    print(f"Saved predictions: {args.output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()

