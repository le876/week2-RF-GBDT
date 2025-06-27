import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保实验可重复性

    Args:
        seed (int, optional): 随机种子. 默认为 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device() -> torch.device:
    """获取当前可用的设备（GPU/CPU）

    Returns:
        torch.device: 可用设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pt") -> None:
    """保存模型检查点

    Args:
        state (dict): 要保存的状态字典
        is_best (bool): 是否是最佳模型
        checkpoint_dir (str): 保存检查点的目录
        filename (str, optional): 检查点文件名. 默认为 "checkpoint.pt"
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pt")
        torch.save(state, best_filepath) 