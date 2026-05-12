# fMoW关键类别Starter子集

该目录保存从官方 fMoW-rgb AWS S3 数据源下载的关键类别小样本子集，用于先跑通方案B+D的流程。

## 数据来源

- 官方仓库：https://github.com/fMoW/dataset
- 官方 S3：`s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb`
- 本地 manifest：`../manifests/fmow-rgb_manifest.json.bz2`

fMoW-rgb 全量约 200GB，因此这里没有下载全量数据，只下载 starter subset。

## 当前子集规模

- 类别数：12
- 每类：5 张 train + 2 张 val
- 总图像：84 张 `_rgb.jpg`
- 总元数据：84 个 `_rgb.json`
- 当前体量：约 351MB

## 当前类别

```text
military_facility
airport
airport_hangar
airport_terminal
runway
port
shipyard
storage_tank
oil_or_gas_facility
factory_or_powerplant
electric_substation
prison
```

## 清单文件

```text
manifests/fmow_key_subset.jsonl
manifests/fmow_key_subset_stats.json
```

每条 jsonl 记录包含：

```json
{
  "split": "train",
  "facility_label": "airport",
  "image": ".../airport_0_0_rgb.jpg",
  "metadata": ".../airport_0_0_rgb.json",
  "source": "fMoW-rgb"
}
```

## 扩大下载

例如每类下载 50 张 train、10 张 val：

```bash
python3 后续规划/方案B_D_项目框架/scripts/download_fmow_key_subset.py --train-limit 50 --val-limit 10
```

