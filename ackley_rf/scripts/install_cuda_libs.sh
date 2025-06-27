#!/bin/bash

# 安装CUDA运行时库的脚本

echo "=========================================="
echo "开始安装CUDA运行时库"
echo "=========================================="

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    echo "请使用root权限运行此脚本"
    echo "例如: sudo bash scripts/install_cuda_libs.sh"
    exit 1
fi

# 检测CUDA版本
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2)
echo "检测到CUDA版本: $CUDA_VERSION"

# 创建软链接目录
mkdir -p /usr/local/cuda-12.6/lib64
mkdir -p /usr/local/cuda-12.6/include

# 查找libnvrtc.so文件
echo "查找libnvrtc.so文件..."
NVRTC_PATHS=$(find /usr -name "libnvrtc.so*" 2>/dev/null)

if [ -z "$NVRTC_PATHS" ]; then
    echo "未找到libnvrtc.so文件，尝试安装CUDA运行时库..."
    
    # 添加NVIDIA仓库
    apt-get update
    apt-get install -y software-properties-common
    apt-get install -y apt-transport-https ca-certificates gnupg
    
    # 安装CUDA运行时库
    apt-get install -y --no-install-recommends \
        cuda-nvrtc-12-6 \
        cuda-nvrtc-dev-12-6 \
        cuda-cudart-12-6 \
        cuda-cudart-dev-12-6
    
    # 再次查找libnvrtc.so文件
    NVRTC_PATHS=$(find /usr -name "libnvrtc.so*" 2>/dev/null)
    
    if [ -z "$NVRTC_PATHS" ]; then
        echo "❌ 安装失败，未找到libnvrtc.so文件"
        echo "请尝试手动安装CUDA运行时库"
        exit 1
    fi
fi

# 创建软链接
echo "创建软链接..."
for NVRTC_PATH in $NVRTC_PATHS; do
    FILENAME=$(basename "$NVRTC_PATH")
    ln -sf "$NVRTC_PATH" "/usr/local/cuda-12.6/lib64/$FILENAME"
    echo "已创建软链接: $NVRTC_PATH -> /usr/local/cuda-12.6/lib64/$FILENAME"
done

# 更新库缓存
echo "更新库缓存..."
ldconfig

# 验证安装
echo "验证安装..."
if [ -f "/usr/local/cuda-12.6/lib64/libnvrtc.so" ]; then
    echo "✅ 安装成功，libnvrtc.so文件已可用"
else
    echo "❌ 安装失败，libnvrtc.so文件不可用"
    echo "请尝试手动安装CUDA运行时库"
    exit 1
fi

# 创建环境变量设置脚本
echo "创建环境变量设置脚本..."
cat > /tmp/cuda_env.sh << EOF
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
EOF

echo "环境变量设置脚本已创建: /tmp/cuda_env.sh"
echo "请运行以下命令设置环境变量:"
echo "source /tmp/cuda_env.sh"

echo "=========================================="
echo "CUDA运行时库安装完成"
echo "==========================================" 