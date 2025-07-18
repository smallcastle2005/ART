import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from config import config

class ADE20KDataset(Dataset):
    """ADE20K语义分割数据集加载器"""
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): 数据集根目录
            split (string): 数据集分割 ('train' 或 'val')
            transform (callable, optional): 应用于样本的可选变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'annotations')
        
        # 获取图像文件名
        self.image_files = [f for f in os.listdir(self.image_dir) 
                           if f.endswith('.jpg') or f.endswith('.png')]
        
        # 验证数据集
        if len(self.image_files) == 0:
            raise RuntimeError(f"在 {self.image_dir} 中没有找到图像文件")
        print(f"找到 {len(self.image_files)} 个{split}图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # 加载掩码
        base_name = os.path.splitext(self.image_files[idx])[0]
        mask_extensions = ['.png', '.jpg', '.jpeg']
        mask_name = None
        
        for ext in mask_extensions:
            possible_mask = os.path.join(self.mask_dir, f"{base_name}{ext}")
            if os.path.exists(possible_mask):
                mask_name = possible_mask
                break
        
        if mask_name is None:
            raise FileNotFoundError(f"找不到对应的掩码文件: {base_name}")
        
        mask = Image.open(mask_name)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 关键修复：确保掩码保持原始类别值
        mask = mask.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # 确保掩码值在有效范围内 [0, num_classes-1]
        mask = torch.clamp(mask, min=0, max=config.NUM_CLASSES-1)
        
        return image, mask

# 训练和验证数据加载器的创建
def get_data_loaders():
    # 数据增强配置
    augment_config = config.AUGMENTATION
    normalize = transforms.Normalize(mean=augment_config['normalize_mean'],
                                    std=augment_config['normalize_std'])
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE+20, config.IMAGE_SIZE+20)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    # 验证/测试数据增强
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    # 创建数据集
    print(f"从 {config.DATA_ROOT} 加载数据集...")
    train_dataset = ADE20KDataset(
        root_dir=config.DATA_ROOT,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ADE20KDataset(
        root_dir=config.DATA_ROOT,
        split='val',
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=True,
        num_workers=config.DATA['num_workers'],
        pin_memory=config.DATA['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=False,
        num_workers=config.DATA['num_workers'],
        pin_memory=config.DATA['pin_memory'],
        drop_last=False
    )
    
    # 打印数据集统计信息
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 检查掩码值范围
    all_mask_values = []
    for _, mask in train_dataset:
        all_mask_values.extend(torch.unique(mask).tolist())
    
    unique_values = set(all_mask_values)
    print(f"掩码中的唯一值: {sorted(unique_values)}")
    print(f"最小值: {min(unique_values)}, 最大值: {max(unique_values)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 测试数据加载器
    train_loader, val_loader = get_data_loaders()
    
    # 获取一个批次并检查形状
    images, masks = next(iter(train_loader))
    print(f"图像形状: {images.shape}")
    print(f"掩码形状: {masks.shape}")
    print(f"图像值范围: {images.min().item():.4f} - {images.max().item():.4f}")
    print(f"掩码值范围: {masks.min().item()} - {masks.max().item()}")