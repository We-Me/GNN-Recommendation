# LightGCN 推荐系统（PyTorch Lightning 实现）

本项目是 LightGCN（Light Graph Convolutional Network） 的一个 PyTorch Lightning 实现，用于隐式反馈推荐系统，具有清晰的模块化结构、灵活的配置方式以及完整的训练与评估流程。

论文来源：[LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)

## 项目特点

- 模块化设计：数据处理、模型、训练、评估清晰解耦
- 基于 PyTorch Lightning：支持可复现训练、分布式训练
- YAML + CLI 配置：超参数可通过配置文件或命令行灵活覆盖
- BPR Loss：适用于隐式反馈推荐
- 完整评估指标：Recall@K、NDCG@K、Hit Ratio@K

## 项目结构概览

```
GNN-Recommendation/
├── config/                         # 配置文件（训练 / 数据集）
├── src/
│   ├── data/
│   │   ├── BPRDataset.py           # BPR 采样数据集
│   │   ├── preprocess.py           # 数据预处理模块
│   │   └── RecDataModule.py        # PyTorch Lightning DataModule
│   ├── loss/
│   │   └── BPRLoss.py              # BPR 损失
│   ├── metrics/
│   │   └── metrics.py              # Recall, NDCG 等评估指标
│   ├── models/
│   │   ├── LightGCN.py             # LightGCN 模型
│   │   ├── LightGCNModule.py       # LightGCN lightning 模块
│   │   └── utils.py                # 相关函数
│   ├── scripts/
│   │   ├── preprocess_raw_data.py  # 数据预处理脚本
│   │   └── train.py                # 模型训练脚本
│   └── utils/
│       └── seed.py                 # 随机种子设置便于复现
├── requirements.txt                # 项目依赖
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## 核心功能说明

### 1. **数据预处理**
- 支持 CSV 等原始数据加载
- 用户级别的 train/val/test 划分
- 最小交互数过滤

### 2. **LightGCN 模型**
- 无特征变换、无非线性
- 多层图卷积（用户-物品二部图）
- 所有层 embedding 取均值作为最终表示
- 支持 node dropout（论文中的正则方式）

### 3. **训练与日志**
- 自动保存 best/last/periodic checkpoint
- TensorBoard 可视化

### 4. **评估指标**
- Recall@K, NDCG@K, Hit Ratio@K
- 可配置 K 值 (例如 [10, 20])

## 安装与环境

### 测试环境
- Python 3.12
- CUDA 12.8

### 安装
```bash
pip install -r requirements.txt
```

## 快速使用

### 1. 数据预处理————以[HetRec 2011 Last.FM](https://grouplens.org/datasets/hetrec-2011/)为例

配置示例config/lastfm.yaml：

```yaml
processed_dir: "./datas/lastfm-processed/"

raw:
  path: "./datas/hetrec2011-lastfm-2k/user_artists.dat"
  sep: "\t"
  has_header: true
  user_col: 0
  item_col: 1
  rating_col: 2

split:
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  min_user_interactions: 10

```

```bash
python src/scripts/preprocess_raw_data.py --config config/lastfm.yaml
```

### 2. 模型训练

使用配置文件:
```bash
python src/scripts/train.py --config config/train_config.yaml
```

也可以通过命令行参数覆盖配置文件，如:
```bash
python src/scripts/train.py \
  --config config/train_config.yaml \
  --embed-dim 128 \
  --num-layers 4 \
  --lr 0.001 \
  --batch-size 1024 \
  --max-epochs 100
```

## 实验结果

在 config/lastfm.yaml，config/train_lastfm_config.yaml 配置下，hetrec2011-lastfm-2k 数据集上测试结果如下：

| Metric             | Value               |
|--------------------|---------------------|
| test/Recall@20     | 0.2244909405708313  |
| test/NDCG@20       | 0.21430368721485138 |
| test/Precision@20  | 0.10985729098320007 |

在 config/movieLens_100k.yaml，config/train_movielens_100k_config.yaml 配置下，MovieLens 100k 数据集上测试结果如下：

| Metric             | Value               |
|--------------------|---------------------|
| test/Recall@20     | 0.1645771861076355  |
| test/NDCG@20       | 0.20150409638881683 |
| test/Precision@20  | 0.15057377517223358 |

在 config/movieLens_25m.yaml，config/train_movielens_25m_config.yaml 配置下，MovieLens 25M 数据集上测试结果如下：

| Metric             | Value               |
|--------------------|---------------------|
| test/Recall@20     | 0.15345323085784912 |
| test/NDCG@20       | 0.18875136971473694 |
| test/Precision@20  | 0.13978134095668793 |

在 config/gowalla.yaml，config/train_gowalla_config.yaml 配置下，MovieLens 25M 数据集上测试结果如下：

| Metric             | Value               |
|--------------------|---------------------|
| test/Recall@20     | 0. |
| test/NDCG@20       | 0. |
| test/Precision@20  | 0. |

## License

本项目采用 MIT License
