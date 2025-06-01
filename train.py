import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import config
from data.preprocess import get_data_loaders
from models.transformer import ImageTransformer
from tqdm import tqdm

def calculate_accuracy(output, target):
    """计算像素级准确率"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total

def train_model():
    """训练图像Transformer模型"""
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = get_data_loaders()
    
    # 初始化模型
    model = ImageTransformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.MODEL['patch_size'],
        d_model=config.MODEL['d_model'],
        num_heads=config.MODEL['num_heads'],
        num_layers=config.MODEL['num_layers'],
        d_ff=config.MODEL['d_ff'],
        dropout=config.MODEL['dropout'],
        window_optimization_steps=config.MODEL['window_optimization_steps'],
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")
    
    # 损失函数和优化器
    # 损失函数 - 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 使用AdamW优化器
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # 学习率调度器
    # 余弦退火学习率
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,           # 5个epoch重启周期
        T_mult=1,
        eta_min=1e-6
    )
    
    # 创建检查点目录
    os.makedirs(config.SAVE['checkpoint_dir'], exist_ok=True)
    
    # 训练变量
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # 训练循环
    for epoch in range(config.TRAIN['num_epochs']):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_window_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        # 创建并正确使用进度条包装器
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for images, masks in train_loader_tqdm:  
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs, window_loss = model(images, masks)
            
            # 计算损失
            seg_loss = criterion(outputs, masks)
            total_loss = seg_loss + window_loss * 0.1  # 窗口损失加权
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()  # 只保留一个优化步骤
            
            # 计算当前批次的准确率
            batch_acc = calculate_accuracy(outputs, masks)
            
            # 更新统计信息
            train_loss += total_loss.item()
            train_window_loss += window_loss.item()
            train_acc += batch_acc
            train_batches += 1
            
            # 更新进度条显示信息
            train_loader_tqdm.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'window_loss': f'{window_loss.item():.4f}',
                'acc': f'{batch_acc*100:.2f}%'  # 使用已计算的准确率
            })

        scheduler.step()
        
        # 计算平均训练指标
        avg_train_loss = train_loss / train_batches
        avg_train_window_loss = train_window_loss / train_batches
        avg_train_acc = train_acc / train_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, masks)
                
                # 更新统计信息
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, masks)
                val_batches += 1
        
        # 计算平均验证指标
        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印统计信息
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{config.TRAIN['num_epochs']} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Window Loss: {avg_train_window_loss:.4f} | Acc: {avg_train_acc*100:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc*100:.2f}%")
        
        # 保存检查点
        if avg_val_loss < best_val_loss or avg_val_acc > best_val_acc:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_acc,
                'train_loss': avg_train_loss,
                'window_loss': avg_train_window_loss,
                'train_accuracy': avg_train_acc
            }
            torch.save(checkpoint, os.path.join(config.SAVE['checkpoint_dir'], 'best_model.pth'))
            print(" 保存最佳模型检查点")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.TRAIN['early_stopping_patience']:
                print(f" 早停：验证损失在 {epochs_no_improve} 个周期内没有改善")
                break
        
        # 定期保存模型
        if (epoch + 1) % config.SAVE['save_frequency'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_acc,
                'train_loss': avg_train_loss,
                'window_loss': avg_train_window_loss,
                'train_accuracy': avg_train_acc
            }
            torch.save(checkpoint, os.path.join(config.SAVE['checkpoint_dir'], f'model_epoch_{epoch+1}.pth'))
            print(f" 保存周期 {epoch+1} 检查点")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()