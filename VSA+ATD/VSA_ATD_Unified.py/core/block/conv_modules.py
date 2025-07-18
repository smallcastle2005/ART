import torch
import torch.nn as nn
import math


class dwconv(nn.Module):
    """深度可分离卷积模块 - ATD风格"""
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 
                     kernel_size=kernel_size, stride=1, 
                     padding=(kernel_size - 1) // 2, dilation=1,
                     groups=hidden_features), 
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        B, N, C = x.shape
        
        # 处理包含cls token的情况
        if isinstance(x_size, tuple):
            H, W = x_size
            # 检查是否包含cls token
            if N == H * W + 1:  # 包含cls token
                # 分离cls token和patch tokens
                cls_token = x[:, 0:1, :]  # [B, 1, C]
                patch_tokens = x[:, 1:, :]  # [B, H*W, C]
                
                # 只对patch tokens进行卷积操作
                patch_tokens = patch_tokens.transpose(1, 2).view(B, C, H, W).contiguous()
                patch_tokens = self.depthwise_conv(patch_tokens)
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2).contiguous()
                
                # 重新组合cls token和patch tokens
                x = torch.cat([cls_token, patch_tokens], dim=1)
            else:  # 不包含cls token
                x = x.transpose(1, 2).view(B, C, H, W).contiguous()
                x = self.depthwise_conv(x)
                x = x.flatten(2).transpose(1, 2).contiguous()
        else:
            # 处理单维度情况
            if N == x_size + 1:  # 包含cls token
                H = W = int(math.sqrt(x_size))
                cls_token = x[:, 0:1, :]
                patch_tokens = x[:, 1:, :]
                
                patch_tokens = patch_tokens.transpose(1, 2).view(B, C, H, W).contiguous()
                patch_tokens = self.depthwise_conv(patch_tokens)
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2).contiguous()
                
                x = torch.cat([cls_token, patch_tokens], dim=1)
            else:
                H = W = int(math.sqrt(N))
                x = x.transpose(1, 2).view(B, C, H, W).contiguous()
                x = self.depthwise_conv(x)
                x = x.flatten(2).transpose(1, 2).contiguous()
        
        return x


class ConvFFN(nn.Module):
    """卷积前馈网络模块 - ATD风格"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=self.hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, x, x_size):
        # 增加维度自动推导
        if isinstance(x_size, tuple):
            H, W = x_size
        else:  # 处理单维度分辨率
            H = W = int(math.sqrt(x_size - 1))  # 64=8x8
        
        # 正确的前向传播顺序
        x = self.fc1(x)  # 先通过第一个线性层
        x = self.act(x)  # 激活
        x = x + self.dwconv(x, x_size)  # ATD风格的残差连接
        x = self.fc2(x)  # 最后通过第二个线性层
        return x


# 向后兼容
DWConv = dwconv