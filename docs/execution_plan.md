# 执行计划

## 阶段0：复用已有HPE基础

目标：把已有 HPE 遥感分类经验迁移到设施识别任务。

任务：

- 整理 fMoW 目标类别：`military_facility`、`airport`、`runway`、`port`、`storage_tank`、`oil_or_gas_facility`、`factory_or_powerplant` 等。
- 设计层次化标签体系：粗类、细类、证据字段。
- 准备 baseline：CNN / ViT / CLIP / RemoteCLIP / VLM + LoRA。

产出：

- 设施类别体系。
- fMoW 子集划分。
- HPE prompt 模板。

## 阶段1：方案B - 设施识别

目标：判断一个遥感图像区域属于哪类设施。

建议数据：

- 主数据：fMoW-rgb。
- 小规模补充：AID / NWPU-RESISC45 / PatternNet，用于普通场景干扰项。

方法：

- Flat prompt baseline。
- HPE：粗类 -> 细类 -> 简短理由。
- LoRA / QLoRA 微调。

核心指标：

- Top-1 Accuracy。
- Macro-F1。
- 关键类别 Recall。
- 混淆矩阵。
- 误判案例分析。

## 阶段2：方案B - 目标证据检测

目标：检测支持设施判断的关键证据。

建议数据：

- 第一版：DIOR。只取 `airplane`、`airport`、`harbor`、`ship`、`storagetank`、`vehicle`、`chimney`、`bridge`。
- 第二版：DOTA。补充旋转框、港口吊机、舰船、车辆等更复杂证据。
- 后续扩展：FAIR1M / RarePlanes / xView，不作为当前主线起点。

第一版任务不要做大：

```text
DIOR 标注 -> 证据清单 -> 简单检测 baseline -> 证据融合
```

建议模型：

- 第一版优先用普通水平框检测，先证明证据层有效。
- 后续再考虑 YOLO-OBB / Oriented R-CNN。

核心输出：

```json
{
  "class_name": "aircraft",
  "confidence": 0.91,
  "bbox": [x1, y1, x2, y2],
  "source": "detector"
}
```

## 阶段3：证据融合

目标：把设施分类结果和目标检测结果合成可解释判断。

融合逻辑：

```text
设施置信度
+ 关键证据类别
+ 证据数量
+ 空间关系
+ 不确定性
-> 设施可信度 / 风险分数 / 复核建议
```

最低可用版本：

- 规则融合：按类别设置 evidence weight。
- 后续升级：训练一个轻量 MLP / XGBoost / logistic regression 做融合。

## 阶段4：方案D - 语义航路规划仿真

目标：把识别结果转化为路径规划代价。

输入：

- 起点、终点。
- 障碍区域。
- 设施风险区域。
- 不确定观测区域。

规划器：

- 第一版：A*。
- 第二版：RRT* 或 Hybrid A*。
- 第三版：MPC 做轨迹平滑与动态约束。

对比实验：

- 无语义代价路径。
- 有风险规避代价路径。
- 有不确定性复核观测路径。

## 阶段5：论文组织

建议论文主线：

```text
问题定义
-> 设施层次化识别
-> 目标证据检测
-> 证据融合
-> 语义代价地图
-> 航路规划辅助验证
```

建议先发小论文时保留：

- 设施识别。
- 证据融合。
- 小规模语义规划仿真。

暂不展开：

- 真实飞控闭环。
- 实机部署。
- 跨视角定位。
