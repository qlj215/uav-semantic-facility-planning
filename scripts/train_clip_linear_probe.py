#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_OUTPUT = ROOT / "outputs" / "clip_linear_probe"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.draft("RGB", (512, 512))
    return image.convert("RGB")


def list_samples(
    data_root: Path,
    split: str,
    class_to_idx: dict[str, int] | None = None,
) -> Tuple[List[str], List[Tuple[Path, int]]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    if class_to_idx is None:
        classes = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
    else:
        classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    samples: List[Tuple[Path, int]] = []
    for class_name in classes:
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            samples.append((image_path, class_to_idx[class_name]))

    if not samples:
        raise RuntimeError(f"No jpg images found under {split_root}")
    return classes, samples


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def as_tensor_features(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(output)}")


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
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        features = as_tensor_features(self.model.get_image_features(**inputs))
        return normalize(features).float()


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
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        features = self.model.encode_image(batch)
        return normalize(features).float()


def batched(items: Sequence[Tuple[Path, int]], batch_size: int) -> Iterable[Sequence[Tuple[Path, int]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def extract_features(
    backend: TransformersClipBackend | OpenClipBackend,
    samples: Sequence[Tuple[Path, int]],
    batch_size: int,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    features_list: List[torch.Tensor] = []
    labels: List[int] = []
    paths: List[str] = []

    for step, batch in enumerate(batched(samples, batch_size), start=1):
        image_paths = [path for path, _ in batch]
        images = [load_image(path) for path in image_paths]
        features = backend.encode_images(images).cpu()
        features_list.append(features)
        labels.extend(label for _, label in batch)
        paths.extend(str(path) for path in image_paths)

        if step == 1 or step % 20 == 0:
            seen = min(step * batch_size, len(samples))
            print(f"Extract {split_name}: {seen}/{len(samples)} images")

    return torch.cat(features_list, dim=0), torch.tensor(labels, dtype=torch.long), paths


def train_one_epoch(
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
) -> float:
    classifier.train()
    order = torch.randperm(labels.size(0), device=labels.device)
    total_loss = 0.0
    seen = 0

    for start in range(0, labels.size(0), batch_size):
        idx = order[start : start + batch_size]
        batch_features = features[idx]
        batch_labels = labels[idx]
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_labels.size(0)
        seen += batch_labels.size(0)

    return total_loss / max(1, seen)


@torch.no_grad()
def evaluate(
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray, List[int], List[float]]:
    classifier.eval()
    all_pred: List[int] = []
    all_conf: List[float] = []

    for start in range(0, labels.size(0), batch_size):
        batch_features = features[start : start + batch_size]
        logits = classifier(batch_features)
        probs = logits.softmax(dim=-1)
        all_pred.extend(probs.argmax(dim=-1).cpu().tolist())
        all_conf.extend(probs.max(dim=-1).values.cpu().tolist())

    y_true = labels.cpu().tolist()
    cm = compute_confusion(y_true, all_pred, num_classes)
    return metrics_from_confusion(cm), cm, all_pred, all_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe baseline for frozen CLIP/RemoteCLIP image features.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "open_clip"])
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32", help="HF CLIP model id.")
    parser.add_argument("--open-clip-model", type=str, default="ViT-B-32", help="OpenCLIP model architecture.")
    parser.add_argument(
        "--open-clip-pretrained",
        type=str,
        default="openai",
        help="OpenCLIP pretrained tag or local checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Image feature extraction batch size.")
    parser.add_argument("--probe-batch-size", type=int, default=256, help="Linear classifier training batch size.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_classes, train_samples = list_samples(args.data_root, "train")
    class_to_idx = {name: idx for idx, name in enumerate(train_classes)}
    classes, val_samples = list_samples(args.data_root, "val", class_to_idx)
    num_classes = len(classes)

    if args.backend == "transformers":
        backend = TransformersClipBackend(args.model_id, device)
        model_desc = args.model_id
    else:
        backend = OpenClipBackend(args.open_clip_model, args.open_clip_pretrained, device)
        model_desc = f"{args.open_clip_model}:{args.open_clip_pretrained}"

    print(f"Data root: {args.data_root}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {len(train_samples)}")
    print(f"Val images: {len(val_samples)}")
    print(f"Backend: {args.backend}")
    print(f"Model: {model_desc}")
    print(f"Device: {device}")

    start_time = time.time()
    train_features, train_labels, _ = extract_features(backend, train_samples, args.batch_size, "train")
    val_features, val_labels, val_paths = extract_features(backend, val_samples, args.batch_size, "val")

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    classifier = nn.Linear(train_features.size(1), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            classifier,
            train_features,
            train_labels,
            optimizer,
            criterion,
            args.probe_batch_size,
        )
        val_metrics, _, _, _ = evaluate(classifier, val_features, val_labels, args.probe_batch_size, num_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}% | "
            f"val_macro_f1={val_metrics['macro_f1'] * 100:.2f}%"
        )

    final_metrics, final_cm, preds, confs = evaluate(
        classifier,
        val_features,
        val_labels,
        args.probe_batch_size,
        num_classes,
    )
    elapsed = time.time() - start_time

    predictions = []
    true_labels = val_labels.cpu().tolist()
    for path, label, pred, conf in zip(val_paths, true_labels, preds, confs):
        predictions.append(
            {
                "image": path,
                "true_label": classes[label],
                "pred_label": classes[pred],
                "confidence": float(conf),
            }
        )

    result = {
        "data_root": str(args.data_root),
        "classes": classes,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "backend": args.backend,
        "model": model_desc,
        "device": str(device),
        "batch_size": args.batch_size,
        "probe_batch_size": args.probe_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "feature_dim": int(train_features.size(1)),
        "trainable_params": sum(p.numel() for p in classifier.parameters() if p.requires_grad),
        "elapsed_seconds": round(elapsed, 2),
        "final": final_metrics,
        "history": history,
        "note": "Linear probe freezes the CLIP/RemoteCLIP image encoder and trains only one linear classifier.",
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", final_cm, classes)
    torch.save(classifier.state_dict(), args.output_dir / "linear_probe.pt")

    print(f"accuracy={final_metrics['accuracy'] * 100:.2f}%")
    print(f"macro_f1={final_metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {args.output_dir / 'confusion_matrix.csv'}")
    print(f"Saved predictions: {args.output_dir / 'predictions.jsonl'}")
    print(f"Saved classifier: {args.output_dir / 'linear_probe.pt'}")


if __name__ == "__main__":
    main()
