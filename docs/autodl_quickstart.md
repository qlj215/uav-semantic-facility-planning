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

For a larger first GPU run:

```bash
python3 scripts/download_fmow_key_subset.py --train-limit 50 --val-limit 10
python3 scripts/prepare_fmow_imagefolder.py --overwrite
python3 scripts/train_resnet50_baseline.py --epochs 10 --batch-size 16 --image-size 128
```

