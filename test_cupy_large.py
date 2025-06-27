#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CuPy大矩阵性能测试脚本
"""

import time
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_large_matrix():
    """测试大矩阵计算性能"""
    try:
        import cupy as cp
        
        # 显示CuPy配置
        logger.info("CuPy版本: %s", cp.__version__)
        logger.info("CUDA版本: %s", cp.cuda.runtime.runtimeGetVersion())
        
        # 测试不同大小的矩阵
        sizes = [5000, 8000, 10000, 15000]
        
        for size in sizes:
            logger.info(f"测试 {size}x{size} 矩阵...")
            
            # GPU测试
            logger.info("开始GPU计算...")
            start_time = time.time()
            
            # 在GPU上创建大型矩阵
            x_gpu = cp.random.rand(size, size)
            y_gpu = cp.random.rand(size, size)
            
            # 矩阵乘法
            z_gpu = cp.matmul(x_gpu, y_gpu)
            cp.cuda.Stream.null.synchronize()  # 确保GPU操作完成
            
            gpu_time = time.time() - start_time
            logger.info("GPU计算耗时: %.4f秒", gpu_time)
            
            # CPU测试
            logger.info("开始CPU计算...")
            start_time = time.time()
            
            # 在CPU上创建相同大小的矩阵
            x_cpu = np.random.rand(size, size)
            y_cpu = np.random.rand(size, size)
            
            # 矩阵乘法
            z_cpu = np.matmul(x_cpu, y_cpu)
            
            cpu_time = time.time() - start_time
            logger.info("CPU计算耗时: %.4f秒", cpu_time)
            
            # 计算加速比
            speedup = cpu_time / gpu_time
            logger.info("GPU加速比: %.2f倍", speedup)
            logger.info("-" * 50)
            
            # 释放内存
            del x_gpu, y_gpu, z_gpu, x_cpu, y_cpu, z_cpu
            cp.get_default_memory_pool().free_all_blocks()
            
    except Exception as e:
        logger.error("测试失败: %s", str(e))

if __name__ == "__main__":
    test_large_matrix() 