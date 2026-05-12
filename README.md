# 方案B+D项目框架

本目录用于执行报告中的“方案B + 方案D”主线：

```text
方案B：军民两用设施识别 + 目标证据检测
方案D：将识别结果转化为语义代价地图，并用于无人机航路规划仿真
```

研究边界：本框架只面向侦察图像理解、证据融合、语义地图构建和航路规划辅助，不包含自动打击、目标分配或武器闭环决策。

## 目录结构

```text
方案B_D_项目框架/
├── README.md
├── configs/
│   ├── facility_taxonomy.yaml
│   ├── evidence_rules.yaml
│   └── planner_config.yaml
├── data/
│   └── README.md
├── docs/
│   ├── execution_plan.md
│   └── evaluation_protocol.md
├── scripts/
│   └── run_synthetic_demo.py
├── src/
│   ├── evidence_fusion.py
│   └── semantic_grid_planner.py
└── outputs/
```

## 最小闭环

```text
设施分类结果
  + 目标检测证据
  -> 证据融合分数
  -> 语义风险/不确定性代价
  -> A*路径规划
  -> 输出传统路径与语义路径对比
```

## 快速运行

当前 demo 不依赖深度学习库，只验证 B 到 D 的接口是否清楚：

```bash
python3 后续规划/方案B_D_项目框架/scripts/run_synthetic_demo.py
```

输出文件：

```text
后续规划/方案B_D_项目框架/outputs/synthetic_demo_result.json
```

## fMoW Starter Baseline

已下载的 fMoW 关键类别 starter 子集可以先整理成 ImageFolder，再跑最小 ResNet-50 baseline：

```bash
python3 后续规划/方案B_D_项目框架/scripts/prepare_fmow_imagefolder.py --overwrite
python3 后续规划/方案B_D_项目框架/scripts/train_resnet50_baseline.py --epochs 1 --batch-size 4 --image-size 96 --cpu
```

说明：默认 baseline 使用随机初始化 ResNet-50，只验证数据读取、训练、评估和输出链路。由于 starter 子集每类训练图只有 5 张，结果不代表正式性能。

如果已安装 `torchvision`，可以启用 ImageNet 预训练权重：

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

其中 `--freeze-backbone` 表示只训练最后的分类头。去掉该参数即可全量微调，建议把学习率降到 `1e-4`。

也可以一次性顺序运行三组 ResNet-50 baseline：

```bash
bash 后续规划/方案B_D_项目框架/scripts/run_resnet50_three_baselines.sh
```

## 后续接入真实模型

1. 用 fMoW 训练或评估设施分类模型，输出设施类别和置信度。
2. 用 DOTA / DIOR / FAIR1M / xView / RarePlanes 训练目标检测模型，输出目标框、类别和置信度。
3. 将二者转成 `src/evidence_fusion.py` 中的 `FacilityPrediction` 和 `DetectionEvidence`。
4. 将融合分数写入栅格或地图图层，交给 `src/semantic_grid_planner.py` 规划路径。

## CLIP / RemoteCLIP Zero-Shot

HuggingFace CLIP：

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend transformers \
  --model-id openai/clip-vit-base-patch32 \
  --batch-size 32 \
  --output-dir outputs/clip_vit_b32_zeroshot
```

OpenCLIP：

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained openai \
  --batch-size 32 \
  --output-dir outputs/openclip_vit_b32_zeroshot
```

RemoteCLIP 权重可通过 OpenCLIP 后端接入。下载 RemoteCLIP checkpoint 后，将 `--open-clip-pretrained` 指向本地权重路径即可：

```bash
python3 scripts/eval_clip_zero_shot.py \
  --backend open_clip \
  --open-clip-model ViT-B-32 \
  --open-clip-pretrained /path/to/RemoteCLIP-ViT-B-32.pt \
  --batch-size 32 \
  --output-dir outputs/remoteclip_vit_b32_zeroshot
```

也可以依次运行 HuggingFace CLIP、OpenCLIP 和 RemoteCLIP：

```bash
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
REMOTECLIP_CKPT=/root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
bash scripts/run_clip_three_baselines.sh
```

如果不设置 `REMOTECLIP_CKPT`，脚本会跳过 RemoteCLIP，只运行前两个 CLIP baseline。

## CLIP / RemoteCLIP Linear Probe

Linear probe 会冻结 CLIP/RemoteCLIP 图像编码器，只训练一个线性分类头。它用于判断预训练视觉特征是否适合当前 12 类设施识别，比 zero-shot 更接近可用分类模型。

HuggingFace CLIP linear probe：

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

RemoteCLIP linear probe：

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

也可以顺序运行两组 linear probe：

```bash
DATA_ROOT=/root/autodl-tmp/data/fmow_key_subset_imagefolder \
REMOTECLIP_CKPT=/root/autodl-tmp/models/RemoteCLIP-ViT-B-32.pt \
BATCH_SIZE=64 \
PROBE_BATCH_SIZE=256 \
EPOCHS=50 \
bash scripts/run_clip_linear_probe_baselines.sh
```

输出目录包含：

```text
metrics.json
confusion_matrix.csv
predictions.jsonl
linear_probe.pt
class_to_idx.json
feature_cache_train.pt
feature_cache_val.pt
```

脚本默认会缓存 CLIP/RemoteCLIP 图像特征。第一次运行需要编码所有图片，会比较慢；之后如果数据、模型和输出目录不变，只修改 `--epochs`、`--lr` 等线性分类头参数，会直接复用缓存，不再重新 Extract 五千多张图片。若确实想强制重新抽特征，可加：

```bash
--no-cache-features
```
