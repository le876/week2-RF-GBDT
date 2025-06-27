# Rosenbrock GBDT 模型

使用梯度提升决策树（GBDT）对Rosenbrock函数进行回归预测的项目。

## 项目目标

使用GBDT模型对Rosenbrock数据集进行回归预测，目标是达到Pearson相关系数 >= 0.85。

## 目录结构

```
rosenbrock_gbdt/
├── data/                   # 数据加载和预处理模块
│   └── load_data.py        # 数据加载和预处理函数
├── models/                 # 模型相关模块
│   └── model_tuning.py     # 模型调优函数
├── visualizations/         # 可视化结果保存目录
├── train.py                # 训练主脚本
└── README.md               # 项目说明文档
```

## 环境要求

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- optuna
- joblib
- graphviz (可选，用于决策树可视化)

## 安装依赖

```bash
pip install scikit-learn numpy pandas matplotlib seaborn optuna joblib
pip install python-graphviz  # 可选，用于决策树可视化
```

或者使用conda：

```bash
conda install scikit-learn numpy pandas matplotlib seaborn optuna joblib
conda install python-graphviz  # 可选，用于决策树可视化
```

## 使用方法

### 1. 数据准备

确保Rosenbrock数据集文件（`Rosenbrock_x_train.npy`, `Rosenbrock_y_train.npy`, `Rosenbrock_x_test.npy`, `Rosenbrock_y_test.npy`）位于项目根目录。

### 2. 训练模型

```bash
python train.py --n_trials 50 --timeout 3600
```

参数说明：
- `--n_trials`: 贝叶斯优化的试验次数，默认为50
- `--timeout`: 优化超时时间（秒），默认为3600（1小时）
- `--use_gpu`: 是否使用GPU加速（注意：GBDT不支持完全GPU加速）
- `--random_state`: 随机种子，默认为42
- `--save_dir`: 结果保存目录，默认为'models/tuning_results/bayesian_opt'

### 3. 查看结果

训练完成后，结果将保存在指定的保存目录中，包括：
- 最佳模型文件（`best_model.pkl`）
- 最佳参数（`best_params.json`）
- 试验历史（`trials_history.csv`）
- 各种可视化图表，包括：
  - 特征重要性图
  - 预测值与实际值对比图
  - 误差分布图
  - 残差图
  - 决策树结构图
  - 评估指标变化趋势图

### 可视化输出说明
在`models/tuning_results/bayesian_opt/[timestamp]`目录中包含以下分析图表：

#### 优化过程分析
- `optimization_progress.png` - 优化进度曲线（包含最佳MSE跟踪）
- `parameter_distributions.png` - 超参数分布直方图
- `parameter_correlations.png` - 超参数相关性热图
- `metrics_trend.png` - 指标趋势图（含MSE和Pearson变化）

#### 模型性能分析
- `test_prediction_vs_actual.png` - 测试集预测值 vs 实际值散点图
- `test_residual_distribution.png` - 测试集残差分布直方图
- `test_residuals_vs_predicted.png` - 残差 vs 预测值关系图
- `error_distribution_detailed.png` - 预测误差分布（箱线图+小提琴图）

#### 模型结构分析
- `feature_importance.png` - 特征重要性排序图
- `tree_structures/`目录包含：
  - `tree_structures_overview.png` - 前3棵树结构预览（显示前2层）
  - `tree_{0-2}_detailed.png` - 单棵树详细结构（显示前3层）

#### 高级分析
- `visualizations/prediction_vs_actual_detailed.png` - 带统计指标的预测分析
- `visualizations/residuals_detailed.png` - 四合一残差分析图

## 超参数调优

本项目使用贝叶斯优化进行超参数调优，主要调优的超参数包括：
- `n_estimators`: 树的数量
- `learning_rate`: 学习率
- `max_depth`: 树的最大深度
- `min_samples_split`: 分裂内部节点所需的最小样本数
- `min_samples_leaf`: 叶节点所需的最小样本数
- `subsample`: 用于拟合个体基学习器的样本比例
- `max_features`: 寻找最佳分割时考虑的特征数量

## 评估指标

主要评估指标包括：
- MSE（均方误差）：用于优化模型
- Pearson相关系数：目标是达到 >= 0.85
- R²（决定系数）：评估模型解释方差的能力 