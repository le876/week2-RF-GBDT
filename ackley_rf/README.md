# Ackley函数随机森林回归模型

这个项目使用随机森林（Random Forest）算法对Ackley函数进行回归建模，目标是达到测试集上的Pearson相关系数大于0.85。

## 项目结构

```
ackley_rf/
├── data/                # 数据处理相关代码和数据
├── models/              # 模型定义和训练相关代码
├── evaluation/          # 模型评估相关代码
├── scripts/             # 训练和评估脚本
├── visualizations/      # 可视化结果
└── README.md            # 项目说明文档
```

## 功能特点

- 使用随机森林算法对Ackley函数进行回归
- 支持超参数调优（网格搜索和贝叶斯优化）
- 支持GPU加速（使用CuPy和cuML库）
- 提供详细的模型评估指标和可视化
- 自动检测过拟合并提供优化建议

## 使用方法

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 如果需要GPU加速，还需安装
pip install --extra-index-url https://pypi.nvidia.com cupy-cuda12x
```

### 训练模型

```bash
# CPU训练
bash scripts/train.sh

# GPU加速训练
bash scripts/train_gpu.sh
```

### 评估模型

```bash
# 评估模型并生成可视化
bash scripts/evaluate.sh
```

## Ackley函数

Ackley函数是一个常用的非凸测试函数，定义为：

f(x) = -20 * exp(-0.2 * sqrt(0.5 * sum(x_i^2))) - exp(0.5 * sum(cos(2 * pi * x_i))) + 20 + e

其中，x是一个n维向量。这个函数有许多局部最小值，全局最小值位于原点x = 0，函数值为0。

## 性能目标

- 测试集上的Pearson相关系数 > 0.85
- 训练集上的R²分数 > 0.95
- 测试集上的MSE < 0.2

## GPU加速

本项目支持使用GPU加速训练过程。如果检测到可用的NVIDIA GPU，训练脚本会自动使用GPU进行加速。

### GPU加速要求

- NVIDIA GPU
- CUDA 11.x 或 12.x
- CuPy 和/或 cuML 库 