#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rosenbrock数据集的随机森林模型训练主脚本
"""

import os
import logging
import argparse
import numpy as np
from data.load_data import load_data, preprocess_data, get_feature_names
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Rosenbrock数据集的随机森林模型')
    parser.add_argument('--n_estimators', type=int, default=500, help='树的数量')
    parser.add_argument('--max_depth', type=int, default=None, help='树的最大深度')
    parser.add_argument('--min_samples_split', type=int, default=2, help='分裂内部节点所需的最小样本数')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='叶节点所需的最小样本数')
    parser.add_argument('--max_features', type=str, default='sqrt', help='寻找最佳分割时考虑的特征数量')
    parser.add_argument('--bootstrap', action='store_true', default=True, help='是否使用bootstrap样本')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='models/results', help='结果保存目录')
    return parser.parse_args()

def plot_feature_importance(model, feature_names, save_dir: str) -> None:
    """绘制特征重要性图"""
    plt.figure(figsize=(12, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title('特征重要性排序')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                           save_dir: str, prefix: str = '') -> None:
    """绘制预测分析图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 预测值与实际值对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'预测值 vs 实际值 (Pearson: {pearsonr(y_true, y_pred)[0]:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}prediction_vs_actual.png'))
    plt.close()
    
    # 2. 误差分布图
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('预测误差')
    plt.ylabel('频数')
    plt.title('误差分布')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}error_distribution.png'))
    plt.close()
    
    # 3. 残差图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分布')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}residuals.png'))
    plt.close()

def plot_detailed_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                                   save_dir: str, prefix: str = '') -> None:
    """绘制详细的预测分析图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 预测值与实际值的散点图（带置信区间）
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='预测点')
    
    # 添加回归线和置信区间
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "r--", label=f'拟合线 (斜率={z[0]:.4f})')
    
    # 计算Pearson相关系数和R²
    pearson_corr = pearsonr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k-', label='理想线 (y=x)')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'预测值 vs 实际值\nPearson: {pearson_corr:.4f}, R²: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}prediction_vs_actual_detailed.png'))
    plt.close()
    
    # 2. 残差分析图（带统计信息）
    errors = y_pred - y_true
    plt.figure(figsize=(12, 8))
    
    # 主残差图
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分布')
    plt.grid(True)
    
    # 残差直方图
    plt.subplot(2, 2, 2)
    sns.histplot(errors, kde=True)
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title(f'残差分布\n均值={np.mean(errors):.4f}, 标准差={np.std(errors):.4f}')
    
    # Q-Q图
    plt.subplot(2, 2, 3)
    from scipy.stats import probplot
    probplot(errors, dist="norm", plot=plt)
    plt.title('残差Q-Q图')
    
    # 残差的绝对值与预测值的关系（异方差性检验）
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, np.abs(errors), alpha=0.5)
    plt.xlabel('预测值')
    plt.ylabel('残差绝对值')
    plt.title('异方差性检验')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}residuals_detailed.png'))
    plt.close()
    
    # 3. 预测误差的分布特征
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=errors)
    plt.ylabel('预测误差')
    plt.title('预测误差箱线图')
    
    plt.subplot(1, 2, 2)
    sns.violinplot(y=errors)
    plt.ylabel('预测误差')
    plt.title('预测误差小提琴图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}error_distribution_detailed.png'))
    plt.close()

def save_results(model, params, metrics, X_train, y_train, X_test, y_test, feature_names, base_save_dir):
    """保存模型结果和可视化图表"""
    # 创建以时间戳命名的文件夹
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    # 保存参数
    params_path = os.path.join(save_dir, 'params.json')
    pd.Series(params).to_json(params_path)
    
    # 保存评估指标
    metrics_path = os.path.join(save_dir, 'metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    # 绘制特征重要性图
    plot_feature_importance(model, feature_names, save_dir)
    
    # 生成训练集预测分析图
    y_pred_train = model.predict(X_train)
    plot_prediction_analysis(y_train, y_pred_train, save_dir, 'train_')
    plot_detailed_prediction_analysis(y_train, y_pred_train, save_dir, 'train_')
    
    # 生成测试集预测分析图
    y_pred_test = model.predict(X_test)
    plot_prediction_analysis(y_test, y_pred_test, save_dir, 'test_')
    plot_detailed_prediction_analysis(y_test, y_pred_test, save_dir, 'test_')
    
    logger.info(f"模型已保存到 {model_path}")
    logger.info(f"结果和可视化图表已保存到 {save_dir}")
    
    return save_dir

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("开始训练Rosenbrock数据集的随机森林模型")
    
    # 解析max_features参数
    max_features = args.max_features
    if max_features.lower() == 'none':
        max_features = None
    
    # 构建模型参数
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': max_features,
        'bootstrap': args.bootstrap,
        'random_state': args.random_state,
        'n_jobs': -1  # 使用所有CPU核心
    }
    
    logger.info(f"模型参数: {params}")
    
    # 加载和预处理数据
    logger.info("加载数据并应用特征工程...")
    data = load_data()
    processed_data = preprocess_data(data)
    
    X_train, y_train = processed_data['X_train'], processed_data['y_train']
    X_test, y_test = processed_data['X_test'], processed_data['y_test']
    
    # 获取特征名称（包含工程特征）
    n_original_features = processed_data.get('n_original_features', X_train.shape[1])
    feature_names = get_feature_names(n_original_features)
    
    logger.info(f"特征工程后的数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
    logger.info(f"特征数量: {len(feature_names)}")
    
    # 训练模型
    logger.info("开始训练随机森林模型...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # 评估模型
    logger.info("评估模型性能...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    train_pearson, _ = pearsonr(y_train, y_pred_train)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_pearson, _ = pearsonr(y_test, y_pred_test)
    
    metrics = {
        'train_mse': train_mse,
        'train_r2': train_r2,
        'train_pearson': train_pearson,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'test_pearson': test_pearson
    }
    
    logger.info(f"训练集评估: MSE={train_mse:.4f}, R²={train_r2:.4f}, Pearson={train_pearson:.4f}")
    logger.info(f"测试集评估: MSE={test_mse:.4f}, R²={test_r2:.4f}, Pearson={test_pearson:.4f}")
    
    # 保存结果
    save_dir = save_results(model, params, metrics, X_train, y_train, X_test, y_test, feature_names, args.save_dir)
    
    # 输出特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("特征重要性排序:")
    for i in range(min(10, len(feature_names))):  # 只显示前10个重要特征
        idx = indices[i]
        logger.info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    logger.info(f"训练完成，结果已保存到 {save_dir}")

if __name__ == "__main__":
    main() 