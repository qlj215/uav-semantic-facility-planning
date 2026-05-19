# DIOR 目标证据层

这个分支只做一件事：把多目标检测结果作为设施识别的证据层，不替代 fMoW 的设施分类任务。

## 为什么先用 DIOR

- DIOR 是已发表的光学遥感目标检测基准，类别数量适中。
- 它包含 `airplane`、`airport`、`harbor`、`ship`、`storagetank`、`vehicle`、`chimney`、`bridge` 等类别。
- 这些类别刚好能解释当前主要混淆：`shipyard/port`、`runway/airport`、`storage_tank/oil_or_gas_facility`。

## 数据关系

```text
fMoW：一张图一个设施类别
DIOR：一张图多个目标框

fMoW 分类结果 + DIOR 目标证据
-> 证据融合
-> 是否需要人工复核
-> 后续语义代价地图
```

## 生成 DIOR 证据清单

如果 AutoDL 上还没有 DIOR，先下载并解压：

```bash
bash scripts/download_and_extract_dior.sh \
  --data-root /root/autodl-tmp/data/DIOR \
  --prepare-manifest
```

假设 DIOR 已放在 `/root/autodl-tmp/data/DIOR`：

```bash
python3 scripts/prepare_dior_evidence_manifest.py \
  --dior-root /root/autodl-tmp/data/DIOR \
  --splits train val \
  --output-dir data/manifests/dior_evidence
```

快速小测试可以加 `--limit`：

```bash
python3 scripts/prepare_dior_evidence_manifest.py \
  --dior-root /root/autodl-tmp/data/DIOR \
  --splits train val \
  --limit 20
```

如果你的 DIOR 解压包没有 `ImageSets/Main/train.txt`、`val.txt` 这类划分文件，直接扫描全部 XML：

```bash
python3 scripts/prepare_dior_evidence_manifest.py \
  --dior-root /root/autodl-tmp/data/DIOR \
  --scan-all \
  --output-dir data/manifests/dior_evidence
```

输出：

```text
data/manifests/dior_evidence/train.jsonl
data/manifests/dior_evidence/val.jsonl
data/manifests/dior_evidence/summary.json
```

使用 `--scan-all` 时输出为：

```text
data/manifests/dior_evidence/all.jsonl
data/manifests/dior_evidence/summary.json
```

每条记录包含图片路径、目标框、证据类别和数量。下一步再把这些清单接到一个简单检测 baseline，不急着做复杂模型。

脚本兼容常见 DIOR 目录名，例如 `JPEGImages`、`JPEGImages-trainval`、`JPEGImages-test` 和 `Annotations/Horizontal Bounding Boxes`。

## 转成 YOLO 格式

```bash
python3 scripts/prepare_dior_yolo_dataset.py \
  --input data/manifests/dior_evidence/all.jsonl \
  --output /root/autodl-tmp/data/dior_yolo_evidence \
  --overwrite
```

默认会按 8:2 划分 train/val，并使用软链接指向原始图片，不会复制一份图片。

输出结构：

```text
/root/autodl-tmp/data/dior_yolo_evidence/
├── data.yaml
├── classes.txt
├── summary.json
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

## 最小 YOLO 检测 baseline

```bash
DATASET_DIR=/root/autodl-tmp/data/dior_yolo_evidence \
bash scripts/run_dior_yolo_min_baseline.sh
```

默认参数：

```text
MODEL=yolov8n.pt
EPOCHS=20
IMGSZ=640
BATCH=16
WORKERS=8
DEVICE=0
```

如果显存够，可以这样加大 batch：

```bash
DATASET_DIR=/root/autodl-tmp/data/dior_yolo_evidence \
BATCH=32 EPOCHS=30 \
bash scripts/run_dior_yolo_min_baseline.sh
```

## 跑到 fMoW 验证集

训练好 DIOR YOLO 后，用它给 fMoW 验证图生成目标证据：

```bash
MODEL=outputs/dior_yolo/yolov8n_evidence_min/weights/best.pt \
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
OUTPUT_DIR=outputs/fmow_yolo_evidence \
bash scripts/run_fmow_yolo_evidence_inference.sh
```

输出：

```text
outputs/fmow_yolo_evidence/evidence.jsonl
outputs/fmow_yolo_evidence/summary.json
outputs/fmow_yolo_evidence/facility_evidence_summary.csv
outputs/fmow_yolo_evidence/visualizations/
```

这一步不计算检测 mAP，因为 fMoW 没有这些目标框真值。它的作用是检查 DIOR 训练出的证据检测器能否在 fMoW 上提供可解释的目标证据。
