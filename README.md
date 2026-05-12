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

说明：当前 baseline 使用随机初始化 ResNet-50，只验证数据读取、训练、评估和输出链路。由于 starter 子集每类训练图只有 5 张，结果不代表正式性能。

## 后续接入真实模型

1. 用 fMoW 训练或评估设施分类模型，输出设施类别和置信度。
2. 用 DOTA / DIOR / FAIR1M / xView / RarePlanes 训练目标检测模型，输出目标框、类别和置信度。
3. 将二者转成 `src/evidence_fusion.py` 中的 `FacilityPrediction` 和 `DetectionEvidence`。
4. 将融合分数写入栅格或地图图层，交给 `src/semantic_grid_planner.py` 规划路径。
