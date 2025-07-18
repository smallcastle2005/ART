import torch
import torch.nn as nn
import torch.optim as optim
from configs.config import VSA_ATD_Config
from model import VSA_ATD_VisionTransformer

# 1. 创建测试模型
model = VSA_ATD_VisionTransformer(config=VSA_ATD_Config())

# 2. 生成模拟输入数据 (CIFAR100格式)
dummy_x = torch.rand(4, 3, 32, 32) * 0.5 + 0.5  # 模拟归一化后的图像数据
# 修改目标格式
dummy_target = torch.randint(0, 100, (4,))  # 生成有效的类别标签

# 在forward前添加
print(f"输入数据范围: {dummy_x.min().item():.3f}-{dummy_x.max().item():.3f}")
print(f"目标类别: {dummy_target}")
dummy_target = torch.randn(4, 100)  # 假设100类分类

# 3. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_x = dummy_x.to(device)
dummy_target = dummy_target.to(device)

# 5. 前向传播
optimizer.zero_grad()
output = model(dummy_x)
loss = criterion(output, dummy_target.argmax(dim=1))

# 6. 反向传播
loss.backward()

# 7. 梯度检查
print("\n--- 梯度检查结果 ---")
zero_grad_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"参数: {name:<50} | 梯度范数: {grad_norm:.8f}")
            if grad_norm < 1e-8:
                zero_grad_params.append(name)
        else:
            print(f"参数: {name:<50} | 梯度: None")
            zero_grad_params.append(name)

if zero_grad_params:
    print("\n警告：以下参数梯度异常:")
    for p in zero_grad_params:
        print(f"- {p}")
