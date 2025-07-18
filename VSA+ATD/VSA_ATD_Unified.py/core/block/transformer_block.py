import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from ..token_dictionary import TokenDictionary
from ..attention.category_attention import CategoryAttention
from ..attention.vsa_local_attention import VSALocalAttention
from ..attention.atd_attention import AdaptiveTokenDictionaryAttention

# 导入卷积前馈网络模块
from .conv_modules import ConvFFN

# 导入窗口操作工具函数
from ..attention.utils import window_partition, window_reverse


class UnifiedTransformerLayer(nn.Module):
    """统一的Transformer层，整合多种注意力机制"""
    
    def __init__(self,
                 dim: int,
                 idx: int,
                 input_resolution: Tuple[int, int],
                 num_heads: int,
                 window_size: int,
                 shift_size: int,
                 category_size: int,
                 num_tokens: int,
                 reducted_dim: int,
                 convffn_kernel_size: int,
                 mlp_ratio: float,
                 qkv_bias: bool = True,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 is_last: bool = False,
                 use_vsa: bool = True,
                 attention_fusion_weights: Optional[Tuple[float, float, float]] = None):
        """
        统一Transformer层初始化
        
        Args:
            dim: 输入通道数
            idx: 层索引
            input_resolution: 输入分辨率
            num_heads: 注意力头数
            window_size: 窗口大小
            shift_size: 移位大小
            category_size: 类别大小
            num_tokens: 令牌字典中的令牌数量
            reducted_dim: 降维后的维度
            convffn_kernel_size: 卷积FFN的核大小
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: 是否使用QKV偏置
            act_layer: 激活层
            norm_layer: 归一化层
            is_last: 是否为最后一层
            use_vsa: 是否使用VSA注意力
            attention_fusion_weights: 注意力融合权重 (vsa_weight, atd_weight, category_weight)
        """
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.reducted_dim = reducted_dim
        self.is_last = is_last
        self.use_vsa = use_vsa
        
        # 注意力融合权重
        # 在__init__方法中，替换第82-83行
        if attention_fusion_weights is None:
            attention_fusion_weights = (0.4, 0.3, 0.3)  # 默认权重
        
        # 将权重改为可学习参数
        self.vsa_weight = nn.Parameter(torch.tensor(attention_fusion_weights[0], dtype=torch.float32))
        self.atd_weight = nn.Parameter(torch.tensor(attention_fusion_weights[1], dtype=torch.float32))
        self.category_weight = nn.Parameter(torch.tensor(attention_fusion_weights[2], dtype=torch.float32))
        
        # 激活函数和归一化
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if not is_last:
            self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True)
            self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True)
        
        # QKV投影 - 为所有注意力机制生成统一的查询、键、值
        # 全局注意力(ATD)：使用输入特征x直接计算
        # 类别注意力：直接使用原始qkv进行分组注意力
        # 局部注意力(VSA-Window)：使用经过空间变换和自适应采样的qkv
        # QKV投影 - 确保requires_grad=True
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        # 显式设置requires_grad
        for param in self.wqkv.parameters():
            param.requires_grad = True
        
        # 三种注意力机制
        # 1. VSA局部注意力 (主要的局部注意力机制)
        if use_vsa:
            self.attn_vsa = VSALocalAttention(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,  # 添加缺失的参数
                use_vsa_sampling=True,
                out_dim=None,  # 使用默认值
                relative_pos_embedding=True,
                qkv_bias=qkv_bias,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0
            )
        
        # 2. ATD交叉注意力 (全局注意力)
        self.attn_atd = AdaptiveTokenDictionaryAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            qkv_bias=qkv_bias
        )
        
        # 3. 类别注意力
        self.attn_category = CategoryAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias
        )
        
        # 卷积前馈网络
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            kernel_size=convffn_kernel_size, 
            act_layer=act_layer
        )
        
        # 注意力融合层
        # 注意力融合层 - 修复输出维度
        self.attention_fusion = nn.Linear(dim * 3, 3)  # 输出3个权重值，而不是dim维度
    
    def forward(self, x: torch.Tensor, td: torch.Tensor, x_size: tuple, token_dict: TokenDictionary) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入特征 (b, n, c)
            td: 令牌字典 (b, m, c)
            x_size: 输入尺寸 (h, w)
            params: 参数字典，包含attn_mask等
            
        Returns:
            x: 输出特征 (b, n, c)
            td: 更新后的令牌字典 (b, m, c)
        """
        h, w = x_size
        b, n, c = x.shape
        
        # 残差连接
        shortcut = x
        x_norm = self.norm1(x)
        
        # 统一生成QKV
        qkv = self.wqkv(x_norm)

        # 1. ATD交叉注意力 (全局注意力)
        x_atd, sim_atd = self.attn_atd(x_norm, td, x_size)
        # 得到x_atd(atd注意力输出特征)和sim_atd(atd注意力相似度矩阵)
        
        # 2. 类别注意力 - 需要QKV输入
        # 为类别注意力生成QKV
        x_category = self.attn_category(qkv, sim_atd)
        
        # 3. VSA局部注意力 - 重新启用
        if self.use_vsa:
            x_vsa = self.attn_vsa(qkv, x_size)
            
            # 确保维度一致性 - 处理cls token
            if x_vsa.shape[1] != x_atd.shape[1]:
                # 如果VSA包含cls token而ATD不包含，需要对齐
                if x_vsa.shape[1] == x_atd.shape[1] + 1:
                    # VSA有cls token，ATD没有 - 移除VSA的cls token进行融合
                    x_vsa_for_fusion = x_vsa[:, 1:, :]  # 移除cls token
                else:
                    # 其他情况，调整到相同维度
                    min_len = min(x_vsa.shape[1], x_atd.shape[1], x_category.shape[1])
                    x_vsa_for_fusion = x_vsa[:, :min_len, :]
                    x_atd = x_atd[:, :min_len, :]
                    x_category = x_category[:, :min_len, :]
            else:
                x_vsa_for_fusion = x_vsa
            
            # 使用注意力融合层动态调整权重
            fusion_input = torch.cat([x_vsa_for_fusion.mean(dim=1, keepdim=True), 
                                     x_atd.mean(dim=1, keepdim=True), 
                                     x_category.mean(dim=1, keepdim=True)], dim=-1)  # (b, 1, dim*3)
            
            fusion_logits = self.attention_fusion(fusion_input)  # (b, 1, 3)
            fusion_weights = F.softmax(fusion_logits, dim=-1)    # (b, 1, 3)
            
            # 分离权重
            vsa_w = fusion_weights[:, :, 0:1]    # (b, 1, 1)
            atd_w = fusion_weights[:, :, 1:2]    # (b, 1, 1) 
            cat_w = fusion_weights[:, :, 2:3]    # (b, 1, 1)
            
            # 应用动态权重 - 现在维度匹配了
            x = vsa_w * x_vsa_for_fusion + atd_w * x_atd + cat_w * x_category
            
            # 如果原始VSA有cls token，需要重新添加
            if x_vsa.shape[1] != x_vsa_for_fusion.shape[1]:
                cls_token = x_vsa[:, 0:1, :]  # 提取cls token
                x = torch.cat([cls_token, x], dim=1)  # 重新添加cls token
        else:
            x = self.atd_weight * x_atd + self.category_weight * x_category
        
        # 注意力特征融合 - 恢复完整的三种注意力融合
        # 注意力特征融合 - 修复维度一致性
        if self.use_vsa:
            weights_sum = self.vsa_weight + self.atd_weight + self.category_weight
            normalized_vsa = self.vsa_weight / weights_sum
            normalized_atd = self.atd_weight / weights_sum
            normalized_category = self.category_weight / weights_sum
            
            # 确保维度一致性
            if x_vsa.shape[1] != x_atd.shape[1]:
                # 处理cls token的情况
                if x_vsa.shape[1] == x_atd.shape[1] + 1:
                    x_vsa_align = x_vsa[:, 1:, :]  # 移除cls token
                    cls_token = x_vsa[:, 0:1, :]
                else:
                    min_len = min(x_vsa.shape[1], x_atd.shape[1], x_category.shape[1])
                    x_vsa_align = x_vsa[:, :min_len, :]
                    x_atd = x_atd[:, :min_len, :]
                    x_category = x_category[:, :min_len, :]
                    cls_token = None
            else:
                x_vsa_align = x_vsa
                cls_token = None
            
            # 注意力特征融合
            x_attention = (normalized_vsa * x_vsa_align + 
                          normalized_atd * x_atd + 
                          normalized_category * x_category)
            
            # 重新添加cls token（如果存在）
            if cls_token is not None:
                x_attention = torch.cat([cls_token, x_attention], dim=1)
        else:
            # 只使用ATD和Category
            weights_sum = self.atd_weight + self.category_weight
            normalized_atd = self.atd_weight / weights_sum
            normalized_category = self.category_weight / weights_sum
            
            x_attention = (normalized_atd * x_atd + 
                          normalized_category * x_category)
        
        # 残差连接
        x = shortcut + x_attention
        
        # 前馈网络
        x = x + self.convffn(self.norm2(x), x_size)
        
        # 令牌更新 - 使用传入的token_dict实例
        if not self.is_last:
            td = token_dict.adaptive_token_refinement(
                current_tokens=td,
                input_features=x,
                attention_similarity=sim_atd
            )
        
        return x, td
    
    def flops(self, input_resolution: Optional[Tuple[int, int]] = None) -> int:
        """计算FLOPs"""
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        
        # 各种注意力机制的FLOPs
        if self.use_vsa:
            flops += self.attn_vsa.flops((h, w))
        flops += self.attn_atd.flops(h * w)
        # 类别注意力FLOPs (简化计算)
        flops += h * w * self.dim * self.category_size
        
        # FFN
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * self.convffn_kernel_size ** 2 * self.mlp_ratio
        
        return flops


class UnifiedTransformerBlock(nn.Module):
    """统一的Transformer块，包含多个Transformer层"""
    
    def __init__(self,
                 dim: int,
                 input_resolution: Tuple[int, int],
                 depth: int,
                 num_heads: int,
                 window_size: int,
                 category_size: int,
                 num_tokens: int,
                 reducted_dim: int,
                 convffn_kernel_size: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_vsa: bool = True):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # 构建多个Transformer层
        self.layers = nn.ModuleList([
            UnifiedTransformerLayer(
                dim=dim,
                idx=i,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                category_size=category_size,
                num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_last=(i == depth - 1),
                use_vsa=use_vsa
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor, td: torch.Tensor, x_size: Tuple[int, int], token_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        h, w = x_size
        
        # 逐层前向传播 - 修复参数传递方式
        for layer in self.layers:
            x, td = layer(x, td, x_size, token_dict)  # 传递token_dict实例
            
        return x, td
    
    def flops(self) -> int:
        """计算总FLOPs"""
        flops = 0
        for layer in self.layers:
            flops += layer.flops()
        return flops
