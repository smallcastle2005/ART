import os
import torch

class Config:
    # 数据集配置
    DATA_ROOT = os.path.join('data', 'ADE20K')  # 改为相对路径
    IMAGE_SIZE = 256  # 图像大小
    NUM_CLASSES = 150  # ADE20K的类别数
    
    # 模型配置
    MODEL = {
        'patch_size': 16,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 2048,
        'dropout': 0.1,
        'window_optimization_steps': 3
    }
    
    # 训练配置
    TRAIN = {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 3,
        'lr_factor': 0.5,
        'early_stopping_patience': 5
    }
    
    # 数据加载配置
    DATA = {
        'num_workers': 4,
        'pin_memory': True
    }
    
    # 数据增强配置
    AUGMENTATION = {
        'random_flip': True,
        'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet 均值
        'normalize_std': [0.229, 0.224, 0.225]     # ImageNet 标准差
    }
    
    # 保存和日志配置
    SAVE = {
        'checkpoint_dir': 'checkpoints',
        'save_frequency': 5  # 每多少个epoch保存一次
    }
    
    # 设备配置
    DEVICE = {
        'cuda': False,  # 设置为True如果使用GPU
        'gpu_id': 0
    }
    
    @property
    def device(self):
        if self.DEVICE['cuda'] and torch.cuda.is_available():
            return torch.device(f"cuda:{self.DEVICE['gpu_id']}")
        return torch.device('cpu')

# 创建全局配置实例
config = Config()