'''
本代码主要实现的是全局注意力机制：结合令牌字典，实现全局特征的注意力计算。
主要功能有：
1. 接受输入特征 x 和令牌字典 td
2. 通过独立的线性层生成 Q, K, V
3. 计算归一化余弦相似度
4. 应用自适应缩放
5. Softmax归一化得到最终的注意力权重
6. 加权求和，计算输出特征
7. 返回更新后的特征和原始相似度矩阵
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class AdaptiveTokenDictionaryAttention(nn.Module):
    """自适应令牌字典交叉注意力机制 (ATD-CA)

    该模块实现了原始ATD中的全局注意力，其核心是计算输入特征与
    一个可学习的、全局共享的令牌字典之间的跨注意力。

    Args:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        num_tokens (int): 令牌字典中的令牌数量.
        reducted_dim (int): Q和K向量的降维维度.
        qkv_bias (bool, optional): 是否为Q, K, V线性层添加偏置. 默认为 True.
    """

    def __init__(self, dim: int, input_resolution: tuple, num_tokens: int, reducted_dim: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.reducted_dim = reducted_dim
        self.qkv_bias = qkv_bias

        # 为Q, K, V创建独立的线性变换层
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        # 可学习的缩放参数，用于自适应调整注意力得分
        self.scale = nn.Parameter(torch.ones(self.num_tokens) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        # 添加参数初始化
        self._init_weights()
        # 初始化缩放参数
        nn.init.constant_(self.scale, 0.5)

    def forward(self, x: torch.Tensor, td: torch.Tensor, x_size: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (b, n, c)，其中 n 是序列长度.
            td (torch.Tensor): 来自TokenDictionary的令牌字典，形状为 (b, m, c)，其中 m 是令牌数量.
            x_size (tuple): 输入特征的空间尺寸 (h, w).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x_attn (torch.Tensor): 经过注意力计算后的输出特征，形状为 (b, n, c).
                - attn (torch.Tensor): 未经Softmax归一化的原始相似度矩阵，形状为 (b, n, m).
        """
        h, w = x_size
        b, n, c = x.shape

        # 1. 生成Query, Key, Value
        q = self.wq(x)       # (b, n, reducted_dim)
        k = self.wk(td)      # (b, m, reducted_dim)
        v = self.wv(td)      # (b, m, c)

        # 2. 计算归一化余弦相似度
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        attn = q_norm @ k_norm.transpose(-2, -1)  # (b, n, m)
        # attn这里也是通过特征与令牌点积计算的相似度矩阵
    
        # 3. 应用自适应缩放，先将scale限制在[0,1]范围内
        scale = torch.clamp(self.scale, 0, 1)
        # 使用 unsqueeze 确保维度匹配 (1, 1, m) 以进行广播
        attn = attn * (1 + scale.unsqueeze(0).unsqueeze(0) * np.log(self.num_tokens))

        # 4. Softmax归一化得到最终的注意力权重
        attn_weights = self.softmax(attn)

        # 5. 加权求和，计算输出特征
        x_attn = (attn_weights @ v).reshape(b, n, c)   

        return x_attn, attn

    def _init_weights(self):
        """初始化模型参数"""
        # 初始化线性层权重
        for m in [self.wq, self.wk, self.wv]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
                
    def flops(self, n):
        """
        计算此模块的FLOPs。

        Args:
            n (int): 输入序列的长度。

        Returns:
            int: FLOPs的总数。
        """
        flops = 0
        # q = self.q_proj(x)
        flops += n * self.dim * self.reducted_dim
        # k = self.k_proj(td)
        flops += self.num_tokens * self.dim * self.reducted_dim
        # v = self.v_proj(td)
        flops += self.num_tokens * self.dim * self.dim
        # attn = q_norm @ k_norm.transpose(-2, -1)
        flops += n * self.reducted_dim * self.num_tokens
        # x_attn = attn @ v
        flops += n * self.num_tokens * self.dim
        return flops