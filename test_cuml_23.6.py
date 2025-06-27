#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试降级后的cuML 23.6.0是否能在GTX 1080 Ti上正常工作
"""

import time
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cuml_compatibility():
    """测试cuML 23.6.0是否兼容GTX 1080 Ti"""
    try:
        # 导入cuML
        import cuml
        from cuml.ensemble import RandomForestRegressor as cuRFR
        from sklearn.ensemble import RandomForestRegressor as skRFR
        import cupy as cp
        
        # 显示版本信息
        logger.info("cuML版本: %s", cuml.__version__)
        logger.info("CuPy版本: %s", cp.__version__)
        logger.info("CUDA版本: %s", cp.cuda.runtime.runtimeGetVersion())
        
        # 获取GPU信息
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        logger.info("GPU型号: %s", gpu_info["name"].decode())
        logger.info("计算能力: %d.%d", gpu_info["major"], gpu_info["minor"])
        
        # 创建测试数据
        n_samples = 50000
        n_features = 20
        
        logger.info("生成测试数据: %d样本, %d特征", n_samples, n_features)
        
        # 在GPU上生成数据
        X_gpu = cp.random.rand(n_samples, n_features)
        y_gpu = cp.random.rand(n_samples)
        
        # 转换为CPU数据用于scikit-learn
        X_cpu = cp.asnumpy(X_gpu)
        y_cpu = cp.asnumpy(y_gpu)
        
        # 使用cuML训练模型
        logger.info("开始cuML训练测试...")
        start_time = time.time()
        
        cuml_model = cuRFR(n_estimators=100, max_depth=10)
        cuml_model.fit(X_gpu, y_gpu)
        
        cuml_time = time.time() - start_time
        logger.info("cuML GPU训练耗时: %.4f秒", cuml_time)
        
        # 使用scikit-learn训练模型
        logger.info("开始scikit-learn训练测试...")
        start_time = time.time()
        
        sklearn_model = skRFR(n_estimators=100, max_depth=10, n_jobs=-1)
        sklearn_model.fit(X_cpu, y_cpu)
        
        sklearn_time = time.time() - start_time
        logger.info("scikit-learn CPU训练耗时: %.4f秒", sklearn_time)
        
        # 计算加速比
        speedup = sklearn_time / cuml_time
        logger.info("GPU加速比: %.2f倍", speedup)
        
        # 预测测试
        logger.info("测试预测功能...")
        X_test_gpu = cp.random.rand(1000, n_features)
        X_test_cpu = cp.asnumpy(X_test_gpu)
        
        y_pred_gpu = cuml_model.predict(X_test_gpu)
        y_pred_cpu = sklearn_model.predict(X_test_cpu)
        
        logger.info("预测功能正常")
        
        return True, f"cuML 23.6.0测试成功，GPU加速比: {speedup:.2f}倍"
    
    except Exception as e:
        logger.error("cuML测试失败: %s", str(e))
        return False, f"cuML测试失败: {str(e)}"

if __name__ == "__main__":
    success, message = test_cuml_compatibility()
    if success:
        logger.info("✅ %s", message)
    else:
        logger.error("❌ %s", message) 