'''
本代码实现类别注意力机制（AC-MSA）：基于特征相似度的自适应分组注意力计算。
主要功能：
1. 根据相似度矩阵将特征分为不同类别组
2. 在每个类别组内进行多头自注意力计算
3. 通过分组实现稀疏注意力，降低计算复杂度
4. 支持可学习的注意力缩放参数
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import index_reverse, feature_shuffle

class CategoryAttention(nn.Module):
    """自适应类别多头自注意力机制 (AC-MSA)
    
    该模块实现基于特征相似度的分组注意力计算，通过将特征按相似度分类
    并在每个类别组内进行自注意力，实现稀疏且高效的注意力机制。
    
    Args:
        dim (int): 输入特征的维度
        num_heads (int): 多头注意力的头数，默认为4
        category_size (int): 每个类别组的大小，默认为128
        qkv_bias (bool): 是否为QKV线性层添加偏置，默认为True
        proj_bias (bool): 是否为输出投影层添加偏置，默认为True
    """
    
    def __init__(self, 
                 dim: int,
                 input_resolution: tuple,
                 num_tokens: int = 64,
                 num_heads: int = 4,
                 category_size: int = 128,
                 qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens  
        self.num_heads = num_heads
        self.category_size = category_size
        
        # 添加head_dim计算
        self.head_dim = dim // num_heads
        
        # 只包含输出投影层
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 将注意力缩放参数设置为可学习参数
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1, 1))), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        
        # 添加参数初始化
        self._init_weights()

    def forward(self, 
                qkv: torch.Tensor, 
                similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qkv: QKV特征张量，形状为 (batch_size, sequence_length, 3*dim)
            similarity_matrix: 特征与令牌的相似度矩阵，形状为 (batch_size, sequence_length, num_tokens)
            
        Returns:
            output: 经过类别注意力计算后的特征，形状为 (batch_size, sequence_length, dim)
        """
        b, n, c3 = qkv.shape
        c = c3 // 3  # 单个特征的维度
        b, n, m = similarity_matrix.shape
        
        # 确定每组的大小和组数
        gs = min(n, self.category_size)  # 每组的大小
        ng = (n + gs - 1) // gs  # 组数
        
        # 1. 根据相似度矩阵对特征进行分类
        # 找到每个特征最相似的令牌ID
        token_ids = torch.argmax(similarity_matrix, dim=-1, keepdim=False)  # (b, n)
        
        # 2. 按令牌ID对特征进行排序
        sorted_values, sort_indices = torch.sort(token_ids, dim=-1, stable=False)
        sort_indices_reverse = index_reverse(sort_indices)
        
        # 3. 重新排列QKV特征
        shuffled_qkv = feature_shuffle(qkv, sort_indices)  # (b, n, c3)
        
        # 4. 处理padding（如果序列长度不能被组大小整除）
        pad_n = ng * gs - n # 计算所需要的padding大小
        if pad_n > 0:
            # 使用翻转的方式进行padding
            padded_qkv = torch.cat((
                shuffled_qkv, 
                torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])
            ), dim=1)
        else:
            padded_qkv = shuffled_qkv
            
        # 5. 重塑为分组形式并分离QKV
        y = padded_qkv.reshape(b, ng, gs, c3)
        qkv_grouped = y.reshape(b, ng, gs, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv_grouped[0], qkv_grouped[1], qkv_grouped[2]  # (b, ng, nh, gs, head_dim)
        
        # 6. 计算分组内的注意力
        # Q @ K^T
        attn = q @ k.transpose(-2, -1)  # (b, ng, nh, gs, gs)
        
        # 7. 应用可学习的缩放参数
        logit_scale = torch.clamp(
            self.logit_scale, 
            max=torch.log(torch.tensor(1. / 0.01)).to(qkv.device)
        ).exp()
        attn = attn * logit_scale
        
        # 8. Softmax归一化
        attn = self.softmax(attn)  # (b, ng, nh, gs, gs)
        
        # 9. 注意力加权求和
        output = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]
        
        # 10. 恢复原始特征顺序
        output = feature_shuffle(output, sort_indices_reverse)
        
        # 11. 输出投影
        output = self.proj(output)
        
        return output

    def _init_weights(self):
        """初始化模型参数"""
        # 初始化投影层
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)
        
        # 初始化缩放参数
        nn.init.constant_(self.logit_scale, torch.log(torch.tensor(10.0)))

        
    def flops(self, sequence_length: int) -> int:
        """
        计算此模块的FLOPs
        
        Args:
            sequence_length: 输入序列长度
            
        Returns:
            flops: 总FLOPs数量
        """
        flops = 0
        
        # Q @ K^T 计算
        flops += sequence_length * self.dim * self.category_size
        
        # Attn @ V 计算
        flops += sequence_length * self.dim * self.category_size
        
        # 输出投影
        flops += sequence_length * self.dim * self.dim
        
        return flops