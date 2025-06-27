# Ackley函数GBDT回归项目

本项目使用梯度提升决策树(GBDT)对Ackley函数进行回归预测，目标是达到Pearson相关系数大于0.85。

## 数据集信息
- 训练集：X形状=(800, 20)，Y形状=(800,)
- 测试集：X形状=(200, 20)，Y形状=(200,)

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行标准训练脚本
bash scripts/train.sh

# 或运行优化训练脚本
bash scripts/train_optimized.sh

# 或使用GPU加速训练（需要NVIDIA GPU）
bash scripts/train_gpu.sh
```

## 环境设置

本项目提供了自动环境设置脚本，可以一键安装所有依赖：

```bash
# 设置环境（包括GPU加速库，如果可用）
bash scripts/setup_environment.sh

# 激活环境
conda activate ackley_env
```

## GPU加速
本项目支持使用GPU加速训练过程。环境设置脚本会自动检测并安装GPU加速库。

要使用GPU加速训练，请运行：
```bash
bash scripts/train_gpu.sh
```

如果GPU加速库安装失败，脚本会自动降级为CPU模式运行。

## 项目结构
```
ackley_gbdt/
├── data/
│   ├── load_data.py       # 数据加载模块
│   └── visualize_data.py  # 数据分布可视化
├── models/
│   ├── baseline_model.py  # 基线模型构建
│   ├── model_tuning.py    # 超参数调优
│   └── trained_model.pkl  # 最佳模型保存
├── evaluation/
│   ├── metrics.py         # 评估指标计算
│   └── residual_plot.py   # 残差可视化
├── scripts/
│   ├── train.sh           # 训练启动脚本
│   ├── train_optimized.sh # 优化训练脚本
│   └── train_gpu.sh       # GPU加速训练脚本
├── requirements.txt       # 依赖库
└── README.md              # 项目文档
```

## 性能目标
✅ Pearson系数 > 0.85 (已达成: 0.8728)

## 最佳参数配置
| 参数 | 最终值 |
|------|-------|
| n_estimators | 1200 |
| learning_rate | 0.050 |
| max_depth | 4 |
| subsample | 0.734 |
| min_samples_split | 2 |

## 模型性能
| 指标 | 训练集 | 测试集 |
|------|-------|-------|
| MSE | 0.0134 | 0.1770 |
| R² | 0.9706 | 0.6491 |
| Pearson | 0.9857 | 0.8728 |

## 优化过程
1. 基线模型: Pearson = 0.8314 (接近但未达标)
2. 贝叶斯优化: 尝试50次不同参数组合
3. 最佳模型: Pearson = 0.8728 (达标)
4. GPU加速: 提高训练速度，扩大参数搜索空间

## 可视化结果
所有评估图表保存在`visualizations/evaluation/`目录下，包括:
- 预测值vs真实值散点图
- 残差分析图
- 特征重要性图
- 预测误差分布图

## 关键参数配置
| 参数 | 优化范围 | 最终值 |
|------|---------|-------|
| n_estimators | 500-2000 | 1200 |
| learning_rate | 0.05-0.2 | 0.050 |
| max_depth | 4-8 | 4 |
| subsample | 0.7-1.0 | 0.734 |
| min_samples_split | 2-8 | 2 |

## 模型训练流程
1. 数据准备与预处理
2. 基线模型训练与评估
3. 超参数调优
4. 最终模型评估与可视化 