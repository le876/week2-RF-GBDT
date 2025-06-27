#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p models/tuning_results/bayesian_opt
mkdir -p visualizations

# 运行贝叶斯优化脚本
echo "开始Rosenbrock随机森林模型的贝叶斯优化（含特征工程）..."

# 导入模型调优模块并运行贝叶斯优化
python - <<END
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data.load_data import load_data, preprocess_data, get_feature_names
from models.model_tuning import bayesian_optimization, save_tuning_results

logger.info("加载数据并应用特征工程...")
data = load_data()
processed_data = preprocess_data(data)

# 获取特征信息
X_train = processed_data['X_train']
n_original_features = processed_data.get('n_original_features', X_train.shape[1])
feature_names = get_feature_names(n_original_features)

logger.info(f"特征工程后的数据形状: X_train={X_train.shape}")
logger.info(f"特征数量: {len(feature_names)}")

# 添加特征名称到数据字典
processed_data['feature_names'] = feature_names

logger.info("开始贝叶斯优化...")
results = bayesian_optimization(
    processed_data,
    n_trials=150,  # 增加试验次数
    timeout=10800,  # 延长至3小时
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

logger.info("保存优化结果...")
save_tuning_results(results, 'models/tuning_results/bayesian_opt')

# 输出特征重要性
if 'best_model' in results:
    importances = results['best_model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("特征重要性排序（前10个）:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        logger.info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
END

echo "贝叶斯优化完成，结果保存在 models/tuning_results/bayesian_opt" 