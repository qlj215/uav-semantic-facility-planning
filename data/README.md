# 数据目录说明

本目录不直接存放大型数据集，只记录数据准备规则。

## 推荐数据源

设施识别：

- fMoW-rgb：优先使用，规模约 200GB。
- fMoW-full：多光谱版本，规模约 3.5TB，不建议第一阶段使用。

目标证据检测：

- DOTA：旋转框检测，适合飞机、舰船、桥梁、港口、车辆等。
- DIOR：20 类遥感目标检测。
- FAIR1M：细粒度、大规模遥感目标识别。
- xView：大规模卫星目标检测。
- RarePlanes：飞机专项检测。

低空无人机补充：

- UAVid：语义分割。
- VisDrone / UAVDT：低空目标检测与跟踪。

## 建议本地组织

```text
data/
├── fmow_rgb/
├── dota/
├── dior/
├── fair1m/
├── rareplanes/
├── manifests/
└── splits/
```

## Manifest格式建议

设施识别 jsonl：

```json
{"image": "path/to/image.jpg", "facility_label": "airport", "coarse_label": "aviation"}
```

目标检测 jsonl：

```json
{"image": "path/to/image.jpg", "detections": [{"class_name": "aircraft", "confidence": 0.91, "bbox": [10, 20, 80, 120]}]}
```

融合输入 jsonl：

```json
{
  "image": "path/to/image.jpg",
  "facility_prediction": {"class_name": "airport", "confidence": 0.82},
  "detections": [{"class_name": "aircraft", "confidence": 0.91, "bbox": [10, 20, 80, 120]}]
}
```

