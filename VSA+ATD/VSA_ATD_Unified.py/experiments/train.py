import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 修复：导入正确的类名
from model import VSA_ATD_VisionTransformer
from configs.config import VSA_ATD_Config

class TrainingMonitor:
    """训练监控器 - 检测损失变化、零梯度占比和训练错误"""
    
    def __init__(self, model, patience=5):
        self.model = model
        self.patience = patience
        
        # 损失历史记录
        self.loss_history = []
        self.loss_change_history = []
        
        # 梯度统计
        self.zero_grad_ratios = []
        self.grad_norm_history = []
        
        # 训练状态监控
        self.stagnant_epochs = 0
        self.error_flags = []
        
        # 异常检测阈值
        self.min_loss_change = 1e-6  # 最小损失变化阈值
        self.max_zero_grad_ratio = 0.8  # 最大零梯度比例
        self.max_grad_norm = 10.0  # 最大梯度范数
        self.min_grad_norm = 1e-8  # 最小梯度范数
        
    def check_loss_change(self, current_loss):
        """检查损失变化"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 2:
            return True, "首次记录损失"
            
        # 计算损失变化
        loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
        self.loss_change_history.append(loss_change)
        
        # 检查损失是否停滞
        if loss_change < self.min_loss_change:
            self.stagnant_epochs += 1
            if self.stagnant_epochs >= self.patience:
                return False, f"损失连续{self.stagnant_epochs}轮变化过小 (<{self.min_loss_change})"
        else:
            self.stagnant_epochs = 0
            
        return True, f"损失变化: {loss_change:.6f}"
    
    def check_gradient_health(self):
        """检查梯度健康状况"""
        total_params = 0
        zero_grad_params = 0
        total_grad_norm = 0.0
        nan_grad_params = 0
        inf_grad_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                
                if param.grad is None:
                    zero_grad_params += 1
                else:
                    grad_norm = param.grad.norm().item()
                    
                    # 检查NaN和Inf
                    if torch.isnan(param.grad).any():
                        nan_grad_params += 1
                    elif torch.isinf(param.grad).any():
                        inf_grad_params += 1
                    elif grad_norm < self.min_grad_norm:
                        zero_grad_params += 1
                    else:
                        total_grad_norm += grad_norm
        
        # 计算零梯度比例
        zero_grad_ratio = zero_grad_params / max(total_params, 1)
        avg_grad_norm = total_grad_norm / max(total_params - zero_grad_params, 1)
        
        self.zero_grad_ratios.append(zero_grad_ratio)
        self.grad_norm_history.append(avg_grad_norm)
        
        # 检查异常情况
        errors = []
        
        if zero_grad_ratio > self.max_zero_grad_ratio:
            errors.append(f"零梯度比例过高: {zero_grad_ratio:.2%}")
            
        if avg_grad_norm > self.max_grad_norm:
            errors.append(f"梯度范数过大: {avg_grad_norm:.2e}")
            
        if avg_grad_norm < self.min_grad_norm:
            errors.append(f"梯度范数过小: {avg_grad_norm:.2e}")
            
        if nan_grad_params > 0:
            errors.append(f"发现{nan_grad_params}个NaN梯度参数")
            
        if inf_grad_params > 0:
            errors.append(f"发现{inf_grad_params}个Inf梯度参数")
        
        return {
            'zero_grad_ratio': zero_grad_ratio,
            'avg_grad_norm': avg_grad_norm,
            'total_params': total_params,
            'zero_grad_params': zero_grad_params,
            'nan_grad_params': nan_grad_params,
            'inf_grad_params': inf_grad_params,
            'errors': errors
        }
    
    def detect_training_errors(self, epoch, batch_idx, loss, accuracy=None):
        """检测训练错误"""
        errors = []
        warnings = []
        
        # 1. 检查损失值异常
        if torch.isnan(torch.tensor(loss)):
            errors.append("损失值为NaN")
        elif torch.isinf(torch.tensor(loss)):
            errors.append("损失值为Inf")
        elif loss < 0:
            errors.append(f"损失值为负数: {loss:.6f}")
        elif loss > 100:
            warnings.append(f"损失值过大: {loss:.6f}")
            
        # 2. 检查准确率异常
        if accuracy is not None:
            if accuracy < 0 or accuracy > 100:
                errors.append(f"准确率异常: {accuracy:.2f}%")
            elif accuracy == 0 and epoch > 2:
                warnings.append("准确率为0，可能存在标签问题")
                
        # 3. 检查损失变化
        loss_ok, loss_msg = self.check_loss_change(loss)
        if not loss_ok:
            warnings.append(loss_msg)
            
        # 4. 检查梯度健康状况
        grad_stats = self.check_gradient_health()
        errors.extend(grad_stats['errors'])
        
        # 5. 检查训练趋势
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            if all(l1 <= l2 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                warnings.append("损失连续5轮上升，可能过拟合")
                
        return {
            'errors': errors,
            'warnings': warnings,
            'grad_stats': grad_stats,
            'loss_change': loss_msg
        }
    
    def print_status(self, epoch, batch_idx, loss, accuracy=None):
        """打印训练状态"""
        status = self.detect_training_errors(epoch, batch_idx, loss, accuracy)
        
        print(f"\n📊 训练监控 - Epoch {epoch}, Batch {batch_idx}")
        print("-" * 60)
        
        # 基本信息
        print(f"💡 当前状态:")
        print(f"  损失: {loss:.6f}")
        if accuracy is not None:
            print(f"  准确率: {accuracy:.2f}%")
        print(f"  {status['loss_change']}")
        
        # 梯度统计
        grad_stats = status['grad_stats']
        print(f"\n🔍 梯度统计:")
        print(f"  零梯度比例: {grad_stats['zero_grad_ratio']:.2%} ({grad_stats['zero_grad_params']}/{grad_stats['total_params']})")
        print(f"  平均梯度范数: {grad_stats['avg_grad_norm']:.2e}")
        
        if grad_stats['nan_grad_params'] > 0:
            print(f"  NaN梯度参数: {grad_stats['nan_grad_params']}")
        if grad_stats['inf_grad_params'] > 0:
            print(f"  Inf梯度参数: {grad_stats['inf_grad_params']}")
        
        # 错误和警告
        if status['errors']:
            print(f"\n🚨 错误:")
            for error in status['errors']:
                print(f"  ❌ {error}")
                
        if status['warnings']:
            print(f"\n⚠️  警告:")
            for warning in status['warnings']:
                print(f"  🟡 {warning}")
                
        if not status['errors'] and not status['warnings']:
            print(f"\n✅ 训练状态正常")
            
        return len(status['errors']) == 0
    
    def get_summary(self):
        """获取训练总结"""
        if not self.loss_history:
            return "无训练数据"
            
        summary = {
            'total_epochs': len(self.loss_history),
            'final_loss': self.loss_history[-1],
            'loss_reduction': self.loss_history[0] - self.loss_history[-1] if len(self.loss_history) > 1 else 0,
            'avg_zero_grad_ratio': np.mean(self.zero_grad_ratios) if self.zero_grad_ratios else 0,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0,
            'stagnant_periods': self.stagnant_epochs
        }
        
        return summary

# ... existing code ...

def train_model():
    """训练模型主函数"""
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 数据加载
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 模型配置和初始化
    config = VSA_ATD_Config()
    model = VSA_ATD_VisionTransformer(config).to(device)
    
    # 初始化训练监控器
    monitor = TrainingMonitor(model, patience=3)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # 训练参数
    num_epochs = 20
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\n🚀 开始训练 - 包含智能训练监控")
    print("=" * 100)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs} - 训练阶段")
        print("-" * 60)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 计算当前准确率
            _, predicted = output.max(1)
            batch_correct = predicted.eq(target).sum().item()
            batch_acc = 100. * batch_correct / target.size(0)
            
            # 训练监控（每50个batch检查一次）
            if batch_idx % 50 == 0:
                training_ok = monitor.print_status(epoch+1, batch_idx, loss.item(), batch_acc)
                
                # 如果检测到严重错误，停止训练
                if not training_ok:
                    print(f"\n🛑 检测到严重训练错误，停止训练")
                    return None
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            train_total += target.size(0)
            train_correct += batch_correct
            
            # 进度显示
            if batch_idx % 100 == 0:
                current_acc = 100. * train_correct / train_total
                print(f'  Batch {batch_idx:3d}: Loss={loss.item():.6f}, Acc={current_acc:.2f}%')
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} - 验证阶段")
        print("-" * 60)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 输出epoch结果
        epoch_time = time.time() - epoch_start_time
        print(f"\n📋 Epoch {epoch+1}/{num_epochs} 结果:")
        print(f"  训练 - 损失: {avg_train_loss:.6f}, 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {avg_val_loss:.6f}, 准确率: {val_acc:.2f}%")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.8f}")
        print(f"  耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config.__dict__,
                'train_history': {
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'val_losses': val_losses,
                    'val_accs': val_accs
                }
            }, 'best_vsa_atd_vit.pth')
            print(f"  ✅ 保存最佳模型 (验证准确率: {best_acc:.2f}%)")
        
        print("=" * 100)
    
    # 训练完成总结
    print(f"\n🎉 训练完成!")
    print(f"📊 最终结果:")
    print(f"  最佳验证准确率: {best_acc:.2f}%")
    print(f"  最终训练准确率: {train_accs[-1]:.2f}%")
    print(f"  最终验证损失: {val_losses[-1]:.6f}")
    
    # 打印训练监控总结
    summary = monitor.get_summary()
    print(f"\n📈 训练监控总结:")
    print(f"  总训练轮数: {summary['total_epochs']}")
    print(f"  损失下降: {summary['loss_reduction']:.6f}")
    print(f"  平均零梯度比例: {summary['avg_zero_grad_ratio']:.2%}")
    print(f"  平均梯度范数: {summary['avg_grad_norm']:.2e}")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.subplot(1, 3, 3)
    if monitor.zero_grad_ratios:
        plt.plot(monitor.zero_grad_ratios, label='零梯度比例')
        plt.xlabel('检查点')
        plt.ylabel('比例')
        plt.legend()
        plt.title('零梯度比例变化')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print(f"📈 训练曲线已保存为 training_curves.png")
    
    return {
        'best_acc': best_acc,
        'train_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        },
        'monitor_summary': summary
    }

if __name__ == '__main__':
    try:
        results = train_model()
        if results:
            print(f"\n✅ 训练成功完成!")
            print(f"最佳验证准确率: {results['best_acc']:.2f}%")
        else:
            print(f"\n❌ 训练因错误而终止")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()