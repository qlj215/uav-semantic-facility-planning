# AutoDL Quickstart

## 1. Clone

```bash
git clone https://github.com/qlj215/uav-semantic-facility-planning.git
cd uav-semantic-facility-planning
```

## 2. Environment

```bash
python3 -m pip install -r requirements.txt
```

Check GPU:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 3. Download Starter fMoW Subset

The repository does not track fMoW images. Download a small starter subset:

```bash
mkdir -p data/manifests
curl -L -o data/manifests/fmow-rgb_manifest.json.bz2 \
  https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2

python3 scripts/download_fmow_key_subset.py --train-limit 5 --val-limit 2
python3 scripts/prepare_fmow_imagefolder.py --overwrite
```

## 4. Run Baseline

```bash
python3 scripts/train_resnet50_baseline.py --epochs 1 --batch-size 8 --image-size 128
```

For an ImageNet-pretrained ResNet-50 baseline:

```bash
python3 scripts/train_resnet50_baseline.py \
  --epochs 20 \
  --batch-size 64 \
  --image-size 224 \
  --workers 8 \
  --lr 1e-3 \
  --use-torchvision \
  --pretrained \
  --freeze-backbone
```

For full fine-tuning after the classifier-head run:

```bash
python3 scripts/train_resnet50_baseline.py \
  --epochs 10 \
  --batch-size 64 \
  --image-size 224 \
  --workers 8 \
  --lr 1e-4 \
  --use-torchvision \
  --pretrained
```

Run all three ResNet-50 baselines in sequence:

```bash
bash scripts/run_resnet50_three_baselines.sh
```

Override common settings:

```bash
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
BATCH_SIZE=64 IMAGE_SIZE=224 WORKERS=8 \
EPOCHS_RANDOM=20 EPOCHS_HEAD=20 EPOCHS_FINETUNE=10 \
bash scripts/run_resnet50_three_baselines.sh
```

## 5. CLIP / RemoteCLIP Zero-Shot Baseline

Install optional dependencies:

```bash
python3 -m pip install transformers open-clip-torch
```

HuggingFace CLIP:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend transformers \
  --model-id openai/clip-vit-base-patch32 \
  --batch-size 32 \
  --output-dir outputs/clip_vit_b32_zeroshot
```

OpenCLIP:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained openai \
  --batch-size 32 \
  --output-dir outputs/openclip_vit_b32_zeroshot
```

RemoteCLIP:

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /path/to/RemoteCLIP-ViT-B-32.pt \
  --batch-size 32 \
  --output-dir outputs/remoteclip_vit_b32_zeroshot
```

For a larger first GPU run:

```bash
python3 scripts/download_fmow_key_subset.py --train-limit 50 --val-limit 10
python3 scripts/prepare_fmow_imagefolder.py --overwrite
python3 scripts/train_resnet50_baseline.py --epochs 10 --batch-size 64 --image-size 224 --workers 8
```
