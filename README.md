# GNN-Recommendation-System

本项目实现了一个基于 LightGCN 的图神经网络推荐系统。
支持对评分数据进行预处理、训练，并支持配置文件灵活配置参数。

## 项目结构

```text
GNN-Recommendation-System/
├── config/               # 配置文件
├── datasets/             # 数据预处理与加载模块
├── models/               # LightGCN 模型结构
├── trainers/             # 模型训练管理器
├── train_lightgcn.py     # 主训练入口
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 数据集

测试使用[HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)的数据集，
解压至`./datas`目录下即可。
通过配置参数可以使用`RawLoader`与`RawProcessor`进行预处理。

## 参数配置

可以参考`config/train_LightGCN.yaml`进行参数配置。

## TODO

- 完善预测逻辑
