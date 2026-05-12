# 评估协议

## 1. 设施识别评估

数据：fMoW-rgb 或其关键类别子集。

指标：

- Top-1 Accuracy。
- Macro-F1。
- Per-class Precision / Recall / F1。
- 关键类别漏检率：`military_facility`、`airport`、`runway`、`port`、`storage_tank`。
- 混淆矩阵，重点看军事设施、工业区、机场、港口之间的混淆。

对比：

- CNN / ViT baseline。
- CLIP / RemoteCLIP zero-shot。
- VLM flat prompt。
- VLM HPE prompt。
- VLM HPE + LoRA。

## 2. 目标证据检测评估

数据：DOTA / DIOR / FAIR1M / xView / RarePlanes。

指标：

- mAP。
- AP50 / AP75。
- 小目标 AP。
- 旋转框 mAP，若使用 DOTA / FAIR1M。
- 关键证据类别召回率。

关键证据类别：

```text
aircraft
ship
vehicle
storage_tank
bridge
harbor
runway-like region
```

## 3. 证据融合评估

目标：证明“设施分类 + 目标证据”优于单纯设施分类。

指标：

- 融合前后 Top-1 / Macro-F1。
- 关键类别 Recall 提升。
- 高风险误报率。
- 需要人工复核样本比例。
- 解释一致性：模型理由是否被检测证据支持。

建议消融：

- 仅设施分类。
- 设施分类 + 目标类别。
- 设施分类 + 目标类别 + 目标数量。
- 设施分类 + 目标类别 + 数量 + 空间关系。

## 4. 语义规划评估

仿真地图输入：

- 起点和终点。
- 普通障碍物。
- 高风险设施区域。
- 低置信度复核区域。

指标：

- 路径长度。
- 规划耗时。
- 风险区域穿越次数。
- 累计风险代价。
- 目标观测覆盖率。
- 复核航点数量。

对比：

- 普通 A*。
- 语义代价 A*。
- RRT* 或 Hybrid A*，作为后续扩展。

## 5. 推荐汇报图表

- 设施类别体系图。
- HPE prompt 流程图。
- 检测证据可视化图。
- 证据融合流程图。
- 语义代价地图。
- 传统路径 vs 语义路径对比图。

