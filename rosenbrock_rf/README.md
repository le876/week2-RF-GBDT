# Rosenbrock 随机森林模型

使用随机森林（Random Forest）对Rosenbrock函数进行回归预测的项目。

## 项目目标

使用随机森林模型对Rosenbrock数据集进行回归预测，目标是达到Pearson相关系数 >= 0.85。

## 目录结构

```
rosenbrock_rf/
├── data/                   # 数据加载和预处理模块
│   └── load_data.py        # 数据加载和预处理函数
├── models/                 # 模型相关模块
│   └── model_tuning.py     # 模型调优函数（可选）
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
- joblib
- graphviz (可选，用于决策树可视化)

## 安装依赖

```bash
pip install scikit-learn numpy pandas matplotlib seaborn joblib
pip install python-graphviz  # 可选，用于决策树可视化
```

或者使用conda：

```bash
conda install scikit-learn numpy pandas matplotlib seaborn joblib
conda install python-graphviz  # 可选，用于决策树可视化
```

## 特征工程

本项目实现了针对非线性回归问题的通用特征工程方法，提升模型性能的同时保持方法的通用性：

### 1. 数据预处理
- **标准化**：对所有输入特征进行标准化处理，使其均值为0，标准差为1
- **缺失值处理**：自动检测并填充缺失值

### 2. 通用非线性特征
基于Rosenbrock函数的特性，我们构造了以下通用特征：

1. **平方项**：x_i^2（所有维度的平方项）- 捕捉二次关系
2. **相邻维度耦合项**：x_{i+1} - x_i^2（捕捉相邻特征间的非线性关系）
3. **一阶偏移项**：(1 - x_i)^2（捕捉与参考点的偏移）
4. **相邻维度交互项**：x_i * x_{i+1}（捕捉相邻特征的交互效应）

### 3. 特征工程效果
- **特征数量**：从原始的n个特征扩展到约4n个特征
- **通用性**：这些特征工程方法适用于大多数非线性回归问题
- **可解释性**：保持了特征的物理意义，便于理解模型决策

## 使用方法

### 1. 数据准备

确保Rosenbrock数据集文件（`Rosenbrock_x_train.npy`, `Rosenbrock_y_train.npy`, `Rosenbrock_x_test.npy`, `Rosenbrock_y_test.npy`）位于项目根目录。

### 2. 训练模型

```bash
python train.py --n_estimators 800 --max_depth 70
```

参数说明：
- `--n_estimators`: 树的数量，默认为500
- `--max_depth`: 树的最大深度，默认为None（无限制）
- `--min_samples_split`: 分裂内部节点所需的最小样本数，默认为2
- `--min_samples_leaf`: 叶节点所需的最小样本数，默认为1
- `--max_features`: 寻找最佳分割时考虑的特征数量，默认为'sqrt'
- `--bootstrap`: 是否使用bootstrap样本，默认为True
- `--random_state`: 随机种子，默认为42
- `--save_dir`: 结果保存目录，默认为'models/results'

### 3. 贝叶斯优化

为获得最佳性能，推荐使用贝叶斯优化自动调整超参数：

```bash
bash run_bayesian_opt.sh
```

贝叶斯优化会：
- 自动应用特征工程
- 搜索最优超参数组合
- 生成详细的性能分析报告
- 保存最佳模型和可视化结果

### 4. 简化特征工程

为了研究不同特征工程方法对模型性能的影响，我们提供了多种特征工程选项：

#### 4.1 仅使用平方项特征

只保留原始特征和平方项特征，运行：

```bash
bash run_squared_features.sh
```

这种简化的特征工程方法：
- 只包含原始特征和平方项特征（x_i和x_i^2）
- 特征数量从原始的n个特征扩展到2n个特征
- 结果保存在`models/tuning_results/squared_features`目录

#### 4.2 完整特征工程

使用所有设计的特征工程方法，运行：

```bash
bash run_bayesian_opt.sh
```

#### 4.3 简化特征工程

使用部分特征工程方法，运行：

```bash
bash run_simplified_features.sh
```

### 5. 查看结果

训练完成后，结果将保存在指定的保存目录中，包括：
- 模型文件（`best_model.pkl`）
- 参数（`best_params.json`）
- 评估指标（`metrics.csv`）
- 各种可视化图表，包括：
  - 特征重要性图（显示工程特征的影响）
  - 预测值与真实值对比图（含偏移/缩放诊断）
  - 误差分布图
  - 残差图
  - 详细的预测分析图

## 随机森林模型优势

随机森林是一种集成学习方法，它通过构建多个决策树并合并它们的预测结果来提高模型性能。其主要优势包括：

1. **高准确性**：通过集成多个决策树的结果，随机森林通常能够获得更高的预测准确性。
2. **抗过拟合**：随机森林通过随机选择特征和样本，减少了过拟合的风险。
3. **特征重要性评估**：随机森林可以评估特征的重要性，帮助理解数据。
4. **处理高维数据**：随机森林能够有效处理高维数据，无需特征选择。
5. **处理缺失值**：随机森林能够处理数据中的缺失值。

## 超参数调优

对于随机森林模型，以下超参数对模型性能影响较大：

- `n_estimators`: 树的数量，通常越多越好，但会增加计算成本。
- `max_depth`: 树的最大深度，控制模型复杂度，防止过拟合。
- `min_samples_split`: 分裂内部节点所需的最小样本数，较大的值可以防止过拟合。
- `min_samples_leaf`: 叶节点所需的最小样本数，较大的值可以防止过拟合。
- `max_features`: 寻找最佳分割时考虑的特征数量，影响树的多样性。

## 评估指标

主要评估指标包括：
- MSE（均方误差）：评估预测值与真实值的平方差的平均值。
- Pearson相关系数：评估预测值与真实值之间的线性相关性，目标是达到 >= 0.85。
- R²（决定系数）：评估模型解释方差的能力。

### 新增可视化功能
- `learning_curves.png`: 训练集与验证集的MSE和Pearson相关系数对比曲线
- `metrics_trend.csv`: 包含所有试验的详细指标数据
- `prediction_truth_diagnosis.png`: 预测值-真实值散点图，含回归线分析（检测偏移/缩放问题）
- 所有可视化图表统一使用英文字体和国际单位符号 