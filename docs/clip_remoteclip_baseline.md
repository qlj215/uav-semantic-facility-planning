# CLIP / RemoteCLIP Baseline

This baseline evaluates zero-shot facility classification on the fMoW key-category ImageFolder.

## Outputs

```text
outputs/<run_name>/
├── metrics.json
├── confusion_matrix.csv
└── predictions.jsonl
```

## CLIP with HuggingFace Transformers

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend transformers \
  --model-id openai/clip-vit-base-patch32 \
  --batch-size 32 \
  --output-dir outputs/clip_vit_b32_zeroshot
```

## CLIP with OpenCLIP

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained openai \
  --batch-size 32 \
  --output-dir outputs/openclip_vit_b32_zeroshot
```

## RemoteCLIP

RemoteCLIP uses the OpenCLIP model interface. Download the checkpoint from the official RemoteCLIP repository or HuggingFace mirror first, then pass the checkpoint path:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /path/to/RemoteCLIP-ViT-B-32.pt \
  --batch-size 32 \
  --output-dir outputs/remoteclip_vit_b32_zeroshot
```

If using a HuggingFace OpenCLIP-compatible model, pass the hub id accepted by `open_clip`.

## Run Three Baselines

```bash
bash scripts/run_clip_three_baselines.sh
```

If the RemoteCLIP checkpoint has already been downloaded:

```bash
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
REMOTECLIP_CKPT=/root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
BATCH_SIZE=32 \
bash scripts/run_clip_three_baselines.sh
```

If `REMOTECLIP_CKPT` is not set, the script runs HuggingFace CLIP and OpenCLIP, then skips RemoteCLIP with a clear message.

## Linear Probe

Zero-shot 只测试文本提示能否直接匹配图像。Linear probe 冻结 CLIP/RemoteCLIP 图像编码器，只训练一个线性分类头，用来判断视觉特征本身是否适合 12 类设施分类。

HuggingFace CLIP:

```bash
python3 scripts/train_clip_linear_probe.py \
  --data-root /root/autodl-tmp/data/fmow_key_subset_imagefolder \
  --backend transformers \
  --model-id openai/clip-vit-base-patch32 \
  --batch-size 64 \
  --probe-batch-size 256 \
  --epochs 50 \
  --lr 1e-3 \
  --output-dir outputs/clip_linear_probe/hf_clip_vit_b32
```

RemoteCLIP:

```bash
python3 scripts/train_clip_linear_probe.py \
  --data-root /root/autodl-tmp/data/fmow_key_subset_imagefolder \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
  --batch-size 64 \
  --probe-batch-size 256 \
  --epochs 50 \
  --lr 1e-3 \
  --output-dir outputs/clip_linear_probe/remoteclip_vit_b32
```

Run both:

```bash
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
REMOTECLIP_CKPT=/root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
BATCH_SIZE=64 \
PROBE_BATCH_SIZE=256 \
EPOCHS=50 \
bash scripts/run_clip_linear_probe_baselines.sh
```

Outputs:

```text
outputs/clip_linear_probe/<run_name>/
├── metrics.json
├── confusion_matrix.csv
├── predictions.jsonl
├── class_to_idx.json
├── linear_probe.pt
├── feature_cache_train.pt
└── feature_cache_val.pt
```

Feature extraction is cached by default. The first run still needs to encode every image; later runs with the same data, model and output directory reuse `feature_cache_train.pt` and `feature_cache_val.pt`. This is useful when only tuning the linear head settings such as `--epochs`, `--lr` or `--weight-decay`.

To force a fresh extraction:

```bash
python3 scripts/train_clip_linear_probe.py ... --no-cache-features
```

## Prompt Templates

The default script averages three templates:

```text
a satellite image of a {label}.
an overhead remote sensing image of a {label}.
an aerial image showing a {label}.
```

You can override them:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --prompt-template "a satellite image of a {label}." \
  --prompt-template "a remote sensing image of a {label} facility."
```
