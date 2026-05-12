#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_OUTPUT = ROOT / "outputs" / "fmow_resnet50_baseline"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, class_to_idx: dict[str, int] | None = None):
        self.root = root
        self.split = split
        self.image_size = image_size
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        classes = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples: List[Tuple[Path, int]] = []
        for class_name in sorted(self.class_to_idx):
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No jpg images found under {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path)
        image.draft("RGB", (self.image_size, self.image_size))
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, layers: List[int], num_classes: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet50(num_classes: int) -> ResNet:
    return ResNet([3, 4, 6, 3], num_classes)


def build_model(
    num_classes: int,
    use_torchvision: bool = False,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if not use_torchvision:
        if pretrained:
            raise ValueError("--pretrained requires --use-torchvision")
        if freeze_backbone:
            raise ValueError("--freeze-backbone requires --use-torchvision")
        return resnet50(num_classes)

    try:
        from torchvision import models
        from torchvision.models import ResNet50_Weights
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for --use-torchvision. "
            "Install it first, for example: pip install torchvision"
        ) from exc

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    cm = compute_confusion(all_true, all_pred, num_classes)
    return metrics_from_confusion(cm), cm


def write_confusion_csv(path: Path, cm: np.ndarray, classes: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for idx, class_name in enumerate(classes):
            writer.writerow([class_name] + cm[idx].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ResNet-50 baseline for fMoW key subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-torchvision", action="store_true", help="Use torchvision's ResNet-50 implementation.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights. Requires --use-torchvision.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final fc layer. Requires --use-torchvision.")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = ImageFolderDataset(args.data_root, "train", args.image_size)
    val_set = ImageFolderDataset(args.data_root, "val", args.image_size, train_set.class_to_idx)
    classes = [name for name, _ in sorted(train_set.class_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        num_classes=num_classes,
        use_torchvision=args.use_torchvision,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    print(f"Data root: {args.data_root}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {len(train_set)}")
    print(f"Val images: {len(val_set)}")
    print(f"Device: {device}")
    print(f"Model source: {'torchvision' if args.use_torchvision else 'local'}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Trainable params: {count_trainable_params(model) / 1e6:.2f}M")
    if not args.pretrained:
        print("Note: this run uses random initialization. Use --use-torchvision --pretrained for a stronger baseline.")

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(1, seen)
        val_metrics, cm = evaluate(model, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}% | "
            f"val_macro_f1={val_metrics['macro_f1'] * 100:.2f}%"
        )

    elapsed = time.time() - start_time
    final_metrics, final_cm = evaluate(model, val_loader, device, num_classes)
    result = {
        "data_root": str(args.data_root),
        "classes": classes,
        "train_images": len(train_set),
        "val_images": len(val_set),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "device": str(device),
        "model_source": "torchvision" if args.use_torchvision else "local",
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "trainable_params": count_trainable_params(model),
        "elapsed_seconds": round(elapsed, 2),
        "final": final_metrics,
        "history": history,
        "note": "Use torchvision pretrained weights and larger data for meaningful accuracy.",
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "class_to_idx.json").write_text(
        json.dumps(train_set.class_to_idx, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", final_cm, classes)

    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {args.output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
