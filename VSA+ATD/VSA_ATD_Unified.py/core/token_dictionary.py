'''
本代码主要是用于实现自适应令牌字典（ATD）的更新计算的部分
核心功能主要有：
1 令牌字典定义
2 自适应令牌更新
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class TokenDictionary(nn.Module):
    """ATD令牌字典模块
    
    负责管理和维护全局语义令牌，提供令牌的初始化、更新和检索功能
    """
    
    def __init__(self, 
                 num_tokens: int = 64,
                 token_dim: int = 512,
                 enable_adaptive_update: bool = True): # 设置enable_adaptive_update参数是因为并不是所有层都需要更新令牌字典
        """
        Args:
            num_tokens: 令牌字典中令牌的数量
            token_dim: 每个令牌的维度
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.enable_adaptive_update = enable_adaptive_update
        
        # 令牌字典的生成
        self.tokens_dictionary = nn.Parameter(torch.empty(num_tokens, token_dim))
        
        # 自适应更新所需的组件
        if enable_adaptive_update:
            self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True) # 对相似度矩阵进行归一化
            self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True) # 控制新旧令牌融合比例的可学习参数
            self.softmax = nn.Softmax(dim=-1) # 将归一化后的相似度转换为注意力权重
            self.sigmoid = nn.Sigmoid() # 将注意力权重转换为0-1之间的值
        
        self._init_tokens()
    
    def _init_tokens(self):
        """初始化令牌字典"""
        nn.init.normal_(self.tokens_dictionary, std=0.02)
    
    def get_batched_tokens(self, batch_size: int) -> torch.Tensor:
        """根据批次大小获取令牌字典
        Args:
            batch_size: 批次大小
        Returns:
            tokens: [B, num_tokens, token_dim]
        """
        return self.tokens_dictionary.unsqueeze(0).repeat(batch_size, 1, 1)  # 为每一个批次提供对应的令牌字典，便于后续的批量处理实现并行计算

    def adaptive_token_refinement(self, 
                                current_tokens: torch.Tensor,
                                input_features: torch.Tensor, 
                                attention_similarity: torch.Tensor) -> torch.Tensor:
        """自适应令牌更新
        实现ATD中的核心令牌更新机制：（注意这里只是定义更新的计算过程）
        td = s*td + (1-s)*torch.einsum('btn,bnc->btc', mask_soft, mask_x)
        Args:
            current_tokens: 当前令牌字典 [B, num_tokens, token_dim]
            input_features: 输入特征 [B, N, features_dim]
            attention_similarity: 特征与令牌的相似度 [B, N, num_tokens]
        Returns:
            refined_tokens: 更新后的令牌字典 [B, num_tokens, token_dim]
        """
        if not self.enable_adaptive_update:
            return current_tokens
            
        # b是批次大小，c代表特征维度
        b, N, c = input_features.shape # 这里的N是特征的长度
        b, n, c = current_tokens.shape # 这里的n是令牌的数量 注：由于einsum计算方法会自动获取张量维度，所以这里的b,n,c实际上并没有被直接使用
        
        # 使用已定义的组件进行自适应更新
        # 1. 对相似度矩阵进行归一化和softmax
        mask_soft = self.softmax(self.norm3(attention_similarity.transpose(-1, -2)))
        
        # 2. 重塑输入特征
        mask_x = input_features.reshape(b, N, c)
        
        # 3. 计算融合权重
        s = self.sigmoid(self.sigma)  # [num_tokens, 1]
        
        # 4. ATD的核心更新公式
        refined_tokens = s * current_tokens + (1 - s) * torch.einsum('btn,bnc->btc', mask_soft, mask_x)
        
        return refined_tokens

    def update_tokens(self, input_features, attention_similarity, is_last_layer=False, method='adaptive'):
        """
        更新令牌字典（按照原始ATD设计）
        Args:
            input_features: 输入特征 [batch_size, sequence_length, feature_dim]
            attention_similarity: 注意力相似度矩阵 [batch_size, num_tokens, sequence_length]
            is_last_layer: 是否为最后一层
            method: 更新方法 ('adaptive', 'identity')
        Returns:
            updated_tokens: 更新后的令牌字典 [batch_size, num_tokens, token_dim]
            注意：这里返回的是临时更新的批次令牌，全局令牌字典通过梯度学习
        """
        batch_size = input_features.shape[0]
        if method == 'adaptive' and self.enable_adaptive_update and not is_last_layer:
            # 使用自适应更新机制（仅用于当前层计算）
            current_tokens = self.get_batched_tokens(batch_size)
            updated_tokens = self.adaptive_token_refinement(
                current_tokens=current_tokens,
                input_features=input_features,
                attention_similarity=attention_similarity
            )
            # 全局令牌字典只通过梯度反传学习，不进行直接数据更新
            return updated_tokens  # 返回临时更新的批次令牌字典
        else:
            # 恒等映射
            return self.get_batched_tokens(batch_size)
    

