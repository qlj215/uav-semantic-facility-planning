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

RemoteCLIP uses the OpenCLIP model interface. Download the checkpoint from the official RemoteCLIP repository or HuggingFace mirror, then pass the checkpoint path:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /path/to/RemoteCLIP-ViT-B-32.pt \
  --batch-size 32 \
  --output-dir outputs/remoteclip_vit_b32_zeroshot
```

If using a HuggingFace OpenCLIP-compatible model, pass the hub id accepted by `open_clip`.

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

