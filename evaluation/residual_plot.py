#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
残差图和其他可视化模块，实现模型评估可视化功能。
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_prediction_vs_actual(model: Any, X: np.ndarray, y_true: np.ndarray, 
                             title: str = "Predicted vs Actual Values", 
                             save_path: Optional[str] = None) -> None:
    """
    绘制预测值与实际值的对比图。
    
    Args:
        model: 训练好的模型
        X: 特征数据
        y_true: 真实标签
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    # 预测
    y_pred = model.predict(X)
    
    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 设置标题和标签
    plt.title(f"{title}\nPearson Correlation: {pearson_corr:.4f}", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    
    # 添加注释
    plt.annotate(f"Pearson: {pearson_corr:.4f}", 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加目标线
    if pearson_corr >= 0.85:
        status = "✓ Target Achieved"
        color = "green"
    else:
        status = "✗ Target Not Achieved"
        color = "red"
    
    plt.annotate(f"Target (>0.85): {status}", 
                xy=(0.05, 0.89), 
                xycoords='axes fraction', 
                fontsize=12,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction vs Actual plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_residuals(model: Any, X: np.ndarray, y_true: np.ndarray, 
                  title: str = "Residual Analysis", 
                  save_path: Optional[str] = None) -> None:
    """
    绘制残差分析图。
    
    Args:
        model: 训练好的模型
        X: 特征数据
        y_true: 真实标签
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    # 预测
    y_pred = model.predict(X)
    
    # 计算残差
    residuals = y_true - y_pred
    
    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 残差 vs 预测值
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title("Residuals vs Predicted Values", fontsize=12)
    ax1.set_xlabel("Predicted Values", fontsize=10)
    ax1.set_ylabel("Residuals", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 残差分布
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title("Residuals Distribution", fontsize=12)
    ax2.set_xlabel("Residuals", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 设置总标题
    fig.suptitle(f"{title}\nPearson Correlation: {pearson_corr:.4f}", fontsize=14)
    
    # 添加注释
    ax1.annotate(f"Pearson: {pearson_corr:.4f}", 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加目标线
    if pearson_corr >= 0.85:
        status = "✓ Target Achieved"
        color = "green"
    else:
        status = "✗ Target Not Achieved"
        color = "red"
    
    ax1.annotate(f"Target (>0.85): {status}", 
                xy=(0.05, 0.89), 
                xycoords='axes fraction', 
                fontsize=10,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual analysis plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_error_distribution(model: Any, X: np.ndarray, y_true: np.ndarray, 
                           title: str = "Prediction Error Distribution", 
                           save_path: Optional[str] = None) -> None:
    """
    绘制预测误差分布图。
    
    Args:
        model: 训练好的模型
        X: 特征数据
        y_true: 真实标签
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    # 预测
    y_pred = model.predict(X)
    
    # 计算误差
    errors = np.abs(y_true - y_pred)
    
    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制误差分布
    sns.histplot(errors, kde=True)
    
    # 添加均值线
    plt.axvline(x=errors.mean(), color='r', linestyle='--', 
                label=f'Mean Error: {errors.mean():.4f}')
    
    # 设置标题和标签
    plt.title(f"{title}\nPearson Correlation: {pearson_corr:.4f}", fontsize=14)
    plt.xlabel("Absolute Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # 添加注释
    plt.annotate(f"Mean Error: {errors.mean():.4f}\nPearson: {pearson_corr:.4f}", 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加目标线
    if pearson_corr >= 0.85:
        status = "✓ Target Achieved"
        color = "green"
    else:
        status = "✗ Target Not Achieved"
        color = "red"
    
    plt.annotate(f"Target (>0.85): {status}", 
                xy=(0.05, 0.89), 
                xycoords='axes fraction', 
                fontsize=12,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction error distribution plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model: RandomForestRegressor, 
                           feature_names: Optional[List[str]] = None,
                           title: str = "Feature Importance", 
                           save_path: Optional[str] = None) -> None:
    """
    绘制特征重要性图。
    
    Args:
        model: 训练好的随机森林模型
        feature_names: 特征名称列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    # 获取特征重要性
    importances = model.feature_importances_
    
    # 如果没有提供特征名称，使用索引
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制条形图
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    
    # 设置标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def generate_all_plots(model: RandomForestRegressor, data: Dict[str, np.ndarray], 
                      save_dir: Optional[str] = None) -> None:
    """
    生成所有评估图表。
    
    Args:
        model: 训练好的随机森林模型
        data: 包含'X_train', 'y_train', 'X_test', 'y_test'的数据字典
        save_dir: 保存目录，如果为None则显示图表
    """
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 训练集评估图表
    logger.info("Generating training set evaluation plots...")
    
    # 预测值 vs 实际值
    plot_prediction_vs_actual(
        model, 
        data['X_train'], 
        data['y_train'],
        title="Training Set: Predicted vs Actual Values",
        save_path=os.path.join(save_dir, 'train_prediction_vs_actual.png') if save_dir else None
    )
    
    # 残差分析
    plot_residuals(
        model, 
        data['X_train'], 
        data['y_train'],
        title="Training Set: Residual Analysis",
        save_path=os.path.join(save_dir, 'train_residuals.png') if save_dir else None
    )
    
    # 误差分布
    plot_error_distribution(
        model, 
        data['X_train'], 
        data['y_train'],
        title="Training Set: Prediction Error Distribution",
        save_path=os.path.join(save_dir, 'train_error_distribution.png') if save_dir else None
    )
    
    # 特征重要性
    feature_names = [f'Feature {i}' for i in range(data['X_train'].shape[1])]
    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="Feature Importance",
        save_path=os.path.join(save_dir, 'feature_importance.png') if save_dir else None
    )
    
    # 测试集评估图表
    if 'X_test' in data and 'y_test' in data:
        logger.info("Generating test set evaluation plots...")
        
        # 预测值 vs 实际值
        plot_prediction_vs_actual(
            model, 
            data['X_test'], 
            data['y_test'],
            title="Test Set: Predicted vs Actual Values",
            save_path=os.path.join(save_dir, 'test_prediction_vs_actual.png') if save_dir else None
        )
        
        # 残差分析
        plot_residuals(
            model, 
            data['X_test'], 
            data['y_test'],
            title="Test Set: Residual Analysis",
            save_path=os.path.join(save_dir, 'test_residuals.png') if save_dir else None
        )
        
        # 误差分布
        plot_error_distribution(
            model, 
            data['X_test'], 
            data['y_test'],
            title="Test Set: Prediction Error Distribution",
            save_path=os.path.join(save_dir, 'test_error_distribution.png') if save_dir else None
        ) 