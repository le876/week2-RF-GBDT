#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
残差可视化模块，用于可视化模型预测结果。
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from scipy.stats import pearsonr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_prediction_vs_actual(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             title: str = "Prediction vs Actual",
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制预测值与真实值的散点图。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 图像保存路径，如果为None则不保存
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 计算Pearson相关系数
    pearson, _ = pearsonr(y_true, y_pred)
    
    # 绘制散点图
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    
    # 添加对角线（理想预测线）
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Prediction Line")
    
    # 添加标题和标签
    plt.title(f"{title}\nPearson Coefficient: {pearson:.4f}\nGBDT Optimized Ackley Model", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction vs Actual plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  title: str = "Residual Analysis",
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    绘制残差分析图。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 图像保存路径，如果为None则不保存
        figsize: 图像大小
    """
    residuals = y_true - y_pred
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{title}\nGBDT Optimized Ackley Model", fontsize=16)
    
    # 1. 残差散点图
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title("Residuals vs Predicted Values")
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差直方图
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    
    # 3. Q-Q图（检验残差的正态性）
    from scipy import stats
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title("Residual Q-Q Plot")
    
    # 4. 残差的绝对值 vs 预测值（检验方差齐性）
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6)
    axes[1, 1].set_title("Absolute Residuals vs Predicted Values")
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Absolute Residuals")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    fig.text(0.5, 0.01, f"Mean Residual: {mean_residual:.4f}, Std Residual: {std_residual:.4f}", 
             ha='center', fontsize=12)
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual analysis plot saved to {save_path}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_feature_importance(model, feature_names: Optional[List[str]] = None, 
                           title: str = "Feature Importance",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    绘制特征重要性图。
    
    Args:
        model: 训练好的GBDT模型
        feature_names: 特征名称列表，如果为None则使用索引
        title: 图表标题
        save_path: 图像保存路径，如果为None则不保存
        figsize: 图像大小
    """
    # 获取特征重要性
    importances = model.feature_importances_
    
    # 如果没有提供特征名称，则使用索引
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # 创建特征重要性DataFrame
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    bars = plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
    
    # 为条形图添加颜色渐变
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(bars)))
    
    plt.yticks(range(len(sorted_importances)), sorted_feature_names)
    plt.title(f"{title}\nGBDT Optimized Ackley Model", fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_prediction_error_distribution(y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      title: str = "Prediction Error Distribution",
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    绘制预测误差分布图。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 图像保存路径，如果为None则不保存
        figsize: 图像大小
    """
    # 计算相对误差（百分比）
    epsilon = 1e-10  # 避免除以零
    relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100
    
    plt.figure(figsize=figsize)
    
    # 绘制相对误差直方图
    sns.histplot(relative_errors, kde=True, bins=30)
    
    # 添加统计信息
    mean_error = np.mean(relative_errors)
    median_error = np.median(relative_errors)
    plt.axvline(x=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.2f}%')
    plt.axvline(x=median_error, color='g', linestyle='-.', label=f'Median Error: {median_error:.2f}%')
    
    # 添加标题和标签
    plt.title(f"{title}\nGBDT Optimized Ackley Model", fontsize=14)
    plt.xlabel("Relative Error (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction error distribution plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def generate_all_plots(model, data: Dict[str, np.ndarray], 
                      save_dir: Optional[str] = None) -> None:
    """
    生成所有评估图表。
    
    Args:
        model: 训练好的模型
        data: 包含'X_train', 'y_train', 'X_test', 'y_test'的数据字典
        save_dir: 图像保存目录，如果为None则不保存
    """
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 获取训练集和测试集的预测值
    X_train, y_train = data['X_train'], data['y_train']
    y_train_pred = model.predict(X_train)
    
    # 训练集上的图表
    logger.info("Generating training set evaluation plots...")
    
    # 预测值vs真实值图
    plot_prediction_vs_actual(
        y_train, y_train_pred, 
        title="Training Set: Prediction vs Actual",
        save_path=os.path.join(save_dir, 'train_prediction_vs_actual.png') if save_dir else None
    )
    
    # 残差分析图
    plot_residuals(
        y_train, y_train_pred, 
        title="Training Set: Residual Analysis",
        save_path=os.path.join(save_dir, 'train_residuals.png') if save_dir else None
    )
    
    # 预测误差分布图
    plot_prediction_error_distribution(
        y_train, y_train_pred, 
        title="Training Set: Prediction Error Distribution",
        save_path=os.path.join(save_dir, 'train_error_distribution.png') if save_dir else None
    )
    
    # 特征重要性图
    plot_feature_importance(
        model, 
        feature_names=[f'Feature {i}' for i in range(X_train.shape[1])],
        title="Feature Importance",
        save_path=os.path.join(save_dir, 'feature_importance.png') if save_dir else None
    )
    
    # 如果有测试集，也生成测试集上的图表
    if 'X_test' in data and 'y_test' in data:
        X_test, y_test = data['X_test'], data['y_test']
        y_test_pred = model.predict(X_test)
        
        logger.info("Generating test set evaluation plots...")
        
        # 预测值vs真实值图
        plot_prediction_vs_actual(
            y_test, y_test_pred, 
            title="Test Set: Prediction vs Actual",
            save_path=os.path.join(save_dir, 'test_prediction_vs_actual.png') if save_dir else None
        )
        
        # 残差分析图
        plot_residuals(
            y_test, y_test_pred, 
            title="Test Set: Residual Analysis",
            save_path=os.path.join(save_dir, 'test_residuals.png') if save_dir else None
        )
        
        # 预测误差分布图
        plot_prediction_error_distribution(
            y_test, y_test_pred, 
            title="Test Set: Prediction Error Distribution",
            save_path=os.path.join(save_dir, 'test_error_distribution.png') if save_dir else None
        )

if __name__ == "__main__":
    # 示例用法
    import sys
    sys.path.append('..')
    from data.load_data import load_data, preprocess_data
    from models.baseline_model import load_model
    
    # 加载和预处理数据
    data = load_data()
    processed_data = preprocess_data(data)
    
    # 假设已经有训练好的模型
    try:
        model = load_model("../trained_models/best_model.pkl")
        
        # 创建可视化目录
        os.makedirs("../visualizations", exist_ok=True)
        
        # 生成所有评估图表
        generate_all_plots(model, processed_data, save_dir="../visualizations")
        
    except FileNotFoundError:
        print("模型文件不存在，请先训练模型。") 