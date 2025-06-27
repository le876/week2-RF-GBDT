#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p models/tuning_results/squared_features
mkdir -p visualizations

# 运行贝叶斯优化脚本
echo "开始Rosenbrock随机森林模型的贝叶斯优化（仅平方项特征工程）..."

# 导入模型调优模块并运行贝叶斯优化
python - <<END
import logging
import numpy as np
import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data.load_data import load_data, preprocess_data, get_feature_names, engineer_rosenbrock_features_squared_only
from models.model_tuning import bayesian_optimization, save_tuning_results
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

logger.info("加载数据并应用仅平方项特征工程...")
data = load_data()

# 复制数据以避免修改原始数据
processed_data = {
    'X_train': data['X_train'].copy(),
    'y_train': data['y_train'].copy(),
    'X_test': data['X_test'].copy(),
    'y_test': data['y_test'].copy()
}

# 检查是否有缺失值
for key, array in processed_data.items():
    if np.isnan(array).any():
        logger.warning(f"{key}中存在缺失值，将使用0填充")
        processed_data[key] = np.nan_to_num(array, nan=0.0)

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(processed_data['X_train'])
X_test_scaled = scaler.transform(processed_data['X_test'])

# 保存原始特征数量
n_features = processed_data['X_train'].shape[1]

# 应用仅平方项特征工程
X_train_engineered = engineer_rosenbrock_features_squared_only(X_train_scaled)
X_test_engineered = engineer_rosenbrock_features_squared_only(X_test_scaled)

# 更新处理后的数据
processed_data['X_train'] = X_train_engineered
processed_data['X_test'] = X_test_engineered
processed_data['n_original_features'] = n_features

# 获取特征信息
feature_names = get_feature_names(n_features, squared_only=True)

logger.info(f"仅平方项特征工程后的数据形状: X_train={processed_data['X_train'].shape}")
logger.info(f"特征数量: {len(feature_names)}")

# 添加特征名称到数据字典
processed_data['feature_names'] = feature_names

logger.info("开始贝叶斯优化...")
results = bayesian_optimization(
    processed_data,
    n_trials=50,  # 设置试验次数为50轮
    timeout=7200,  # 设置超时时间为2小时
    use_gpu=True,  # 启用GPU加速
    random_state=42
)

# 添加特征名称到结果字典
results['feature_names'] = feature_names

# 保存训练数据到结果字典（用于生成训练集预测分析图）
results['X_train'] = processed_data['X_train']
results['y_train'] = processed_data['y_train']
results['X_test'] = processed_data['X_test']
results['y_test'] = processed_data['y_test']

# 确保test_metrics包含特征数量信息
if 'test_metrics' in results:
    results['test_metrics']['feature_count'] = processed_data['X_train'].shape[1]
else:
    # 如果没有test_metrics，创建一个
    y_pred_test = results['best_model'].predict(processed_data['X_test'])
    mse_test = mean_squared_error(processed_data['y_test'], y_pred_test)
    r2_test = r2_score(processed_data['y_test'], y_pred_test)
    pearson_test, _ = pearsonr(processed_data['y_test'], y_pred_test)
    
    results['test_metrics'] = {
        'test_mse': mse_test,
        'test_r2': r2_test,
        'test_pearson': pearson_test,
        'feature_count': processed_data['X_train'].shape[1]
    }
    
    logger.info(f"测试集评估结果: MSE={mse_test:.4f}, R²={r2_test:.4f}, Pearson={pearson_test:.4f}")

logger.info("保存优化结果...")
# 使用专门的目录保存结果
save_tuning_results(results, 'models/tuning_results/squared_features')

# 输出特征重要性
if 'best_model' in results:
    importances = results['best_model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("特征重要性排序（前10个）:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        logger.info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
END

echo "贝叶斯优化完成，结果保存在 models/tuning_results/squared_features" 