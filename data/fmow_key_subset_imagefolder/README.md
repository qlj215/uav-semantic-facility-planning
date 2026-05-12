# fMoW Starter ImageFolder

这是由 `data/fmow_key_subset/manifests/fmow_key_subset.jsonl` 转换得到的 ImageFolder 版本，方便先跑 CNN/ViT 类 baseline。

## 目录结构

```text
fmow_key_subset_imagefolder/
├── train/
│   ├── airport/
│   ├── airport_hangar/
│   └── ...
├── val/
│   ├── airport/
│   ├── airport_hangar/
│   └── ...
├── class_to_idx.json
├── manifest.jsonl
└── summary.json
```

## 当前规模

- 12 类。
- train：每类 5 张，共 60 张。
- val：每类 2 张，共 24 张。

该规模只用于验证流程，不适合作为正式论文结果。

## 重新生成

```bash
python3 后续规划/方案B_D_项目框架/scripts/prepare_fmow_imagefolder.py --overwrite
```

## 跑最小 ResNet-50 baseline

```bash
python3 后续规划/方案B_D_项目框架/scripts/train_resnet50_baseline.py --epochs 1 --batch-size 4 --image-size 96 --cpu
```

使用 torchvision 预训练权重：

```bash
python3 后续规划/方案B_D_项目框架/scripts/train_resnet50_baseline.py \
  --epochs 20 \
  --batch-size 64 \
  --image-size 224 \
  --workers 8 \
  --lr 1e-3 \
  --use-torchvision \
  --pretrained \
  --freeze-backbone
```

输出目录：

```text
后续规划/方案B_D_项目框架/outputs/fmow_resnet50_baseline/
```
