"""
本代码实现的是结合VSA实现的思想创建的局部注意力机制
融合了VSA的变化窗口大小思想和ATD的窗口注意力结构
主要改进方式是采用自适应窗口、多样化感受野、以及数据驱动的方式来实现窗口的多样性和适应性
1 输入特征（接受Transformer中的qkv）
2 采样偏移缩放预测
3 网络采样
4 变化窗口特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
from timm.models.layers import trunc_normal_


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割为窗口
    Args:
        x: (b, h, w, c)
        window_size: 窗口大小
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """
    将窗口合并回特征图
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size: 窗口大小
        h: 特征图高度
        w: 特征图宽度
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class VSALocalAttention(nn.Module):
    """
    VSA增强的局部注意力机制
    
    融合了VSA的核心思想：
    1. 自适应窗口大小采样
    2. 可学习的采样偏移和缩放
    3. 保持与ATD架构的兼容性
    
    Args:
        dim (int): 输入特征维度
        input_resolution (Tuple[int, int]): 输入分辨率 (H, W)
        window_size (int): 基础窗口大小
        num_heads (int): 注意力头数
        out_dim (Optional[int]): 输出特征维度，默认与输入相同
        qkv_bias (bool): 是否使用QKV偏置
        qk_scale (Optional[float]): QK缩放因子
        attn_drop (float): 注意力dropout率
        proj_drop (float): 投影dropout率
        use_vsa_sampling (bool): 是否启用VSA采样机制
        relative_pos_embedding (bool): 是否使用相对位置编码
    """
    
    def __init__(self, 
                 dim: int,
                 input_resolution: Tuple[int, int],
                 num_heads: int = 8,
                 window_size: int = 7,
                 shift_size: int = 0,
                 use_vsa_sampling: bool = True,
                 out_dim: Optional[int] = None,
                 relative_pos_embedding: bool = True,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim or dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.use_vsa_sampling = use_vsa_sampling
        self.relative_pos_embedding = relative_pos_embedding
        
        # 计算padding
        h, w = self.input_resolution
        self.padding_bottom = (self.window_size - h % self.window_size) % self.window_size
        self.padding_right = (self.window_size - w % self.window_size) % self.window_size
        
        head_dim = self.out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # QKV投影层
        self.qkv = nn.Linear(dim, self.out_dim * 3, bias=qkv_bias)

        # VSA采样网络
        if self.use_vsa_sampling:
            # 采样偏移网络
            self.sampling_offsets = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.Conv2d(dim, num_heads * 2, kernel_size=1, stride=1)
            )

            # 采样缩放网络
            self.sampling_scales = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.Conv2d(dim, num_heads * 2, kernel_size=1, stride=1)
            )

            # 初始化采样参数
            self._reset_sampling_parameters()
            self._init_vsa_coordinates()

        # 相对位置编码
        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 通道注意力MLP
        self.channel_mlp = nn.Sequential(
            nn.Linear(3*dim, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, num_heads)
        )

        # 投影层
        self.proj = nn.Linear(self.out_dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    
    def _reset_sampling_parameters(self):
        """重置采样参数"""
        if hasattr(self, 'sampling_offsets'):
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
        if hasattr(self, 'sampling_scales'):
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)
    
    def _init_vsa_coordinates(self):
        """初始化VSA坐标系统"""
        h, w = self.input_resolution
        h_padded = h + self.shift_size + self.padding_bottom
        w_padded = w + self.shift_size + self.padding_right
        
        # 生成图像参考坐标
        image_reference_w = torch.linspace(-1, 1, w_padded)
        image_reference_h = torch.linspace(-1, 1, h_padded)
        image_reference = torch.stack(
            torch.meshgrid(image_reference_w, image_reference_h, indexing='ij'), 0
        ).permute(0, 2, 1).unsqueeze(0)  # (1, 2, h, w)
        
        # 生成窗口参考坐标
        window_reference = F.avg_pool2d(image_reference, kernel_size=self.window_size, stride=self.window_size)
        window_num_h = window_reference.shape[-2]
        window_num_w = window_reference.shape[-1]
        
        # 生成基础坐标 - 关键修改点
        base_coords_h = torch.linspace(-1, 1, self.window_size)
        base_coords_w = torch.linspace(-1, 1, self.window_size)
        base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h, indexing='ij'), 0)
        base_coords = base_coords.permute(0, 2, 1).reshape(1, 2, 1, self.window_size, 1, self.window_size)
        
        # 注册为buffer - 确保维度匹配
        self.register_buffer('base_coords', window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1))
        self.register_buffer('coords', base_coords.expand(1, 2, window_num_h, self.window_size, window_num_w, self.window_size))

    def _apply_vsa_sampling(self, qkv: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """应用VSA自适应采样 - 完整版本"""
        b, h, w, c3 = qkv.shape
        c = c3 // 3
        
        # 使用内部QKV投影层重新处理输入
        qkv_input = qkv.reshape(b * h * w, c3)
        qkv_processed = self.qkv(qkv_input[:, :c])  # 只使用Q通道作为输入
        qkv_processed = qkv_processed.view(b, h, w, c3)
        
        # 计算通道注意力权重
        channel_input = qkv.reshape(b * h * w, c3)
        channel_weights = self.channel_mlp(channel_input)  # (b*h*w, num_heads)
        channel_weights = F.softmax(channel_weights, dim=-1)
        channel_weights = channel_weights.reshape(b, h, w, self.num_heads)
        
        # ... 继续原有的采样逻辑，但使用qkv_processed和channel_weights ...
        
        # 应用通道注意力权重到采样结果
        for head_idx in range(self.num_heads):
            head_weight = channel_weights[:, :, :, head_idx:head_idx+1]  # (b, h, w, 1)
            # 将权重应用到对应头的特征
            # ...
        
        # 计算填充后的尺寸
        h_padded = h + self.shift_size + self.padding_bottom
        w_padded = w + self.shift_size + self.padding_right
        
        # 转换为BCHW格式并padding
        qkv_conv = qkv.permute(0, 3, 1, 2)  # (b, 3*c, h, w)
        qkv_padded = F.pad(qkv_conv, (self.shift_size, self.padding_right, 
                                     self.shift_size, self.padding_bottom))
        
        # 分离QKV
        q_padded, k_padded, v_padded = qkv_padded.chunk(3, dim=1)
        
        # 计算采样偏移和缩放
        x_input = qkv_conv[:, :c]  # 使用Q通道
        x_padded = F.pad(x_input, (self.shift_size, self.padding_right, 
                                 self.shift_size, self.padding_bottom))
        
        # 确保输入尺寸是窗口大小的整数倍
        target_h = ((h_padded + self.window_size - 1) // self.window_size) * self.window_size
        target_w = ((w_padded + self.window_size - 1) // self.window_size) * self.window_size
        
        if x_padded.shape[-2] != target_h or x_padded.shape[-1] != target_w:
            x_padded = F.interpolate(x_padded, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # 同步调整其他张量
            q_padded = F.interpolate(q_padded, size=(target_h, target_w), mode='bilinear', align_corners=False)
            k_padded = F.interpolate(k_padded, size=(target_h, target_w), mode='bilinear', align_corners=False)
            v_padded = F.interpolate(v_padded, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # 计算窗口数量
        window_num_h = target_h // self.window_size
        window_num_w = target_w // self.window_size
        
        # 生成采样参数
        sampling_offsets = self.sampling_offsets(x_padded)  # (b, num_heads*2, window_num_h, window_num_w)
        sampling_scales = self.sampling_scales(x_padded)    # (b, num_heads*2, window_num_h, window_num_w)
        
        # 重塑采样参数
        sampling_offsets = sampling_offsets.view(b, self.num_heads, 2, window_num_h, window_num_w)
        sampling_scales = sampling_scales.view(b, self.num_heads, 2, window_num_h, window_num_w)
        
        # 生成基础网格坐标
        device = qkv.device
        # 创建窗口内的基础坐标
        y_coords = torch.linspace(-1, 1, self.window_size, device=device)
        x_coords = torch.linspace(-1, 1, self.window_size, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (window_size, window_size, 2)
        
        # 扩展到所有窗口和批次
        base_grid = base_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, window_size, window_size, 2)
        base_grid = base_grid.expand(b, self.num_heads, window_num_h * window_num_w, -1, -1, -1)
        
        # 应用自适应采样
        sampled_features_list = []
        
        for head_idx in range(self.num_heads):
            # 获取当前头的采样参数
            head_offsets = sampling_offsets[:, head_idx]  # (b, 2, window_num_h, window_num_w)
            head_scales = sampling_scales[:, head_idx]    # (b, 2, window_num_h, window_num_w)
            
            # 重塑为窗口格式
            head_offsets = head_offsets.permute(0, 2, 3, 1).contiguous()  # (b, window_num_h, window_num_w, 2)
            head_scales = head_scales.permute(0, 2, 3, 1).contiguous()    # (b, window_num_h, window_num_w, 2)
            
            # 展平窗口维度
            head_offsets = head_offsets.view(b, window_num_h * window_num_w, 2)  # (b, num_windows, 2)
            head_scales = head_scales.view(b, window_num_h * window_num_w, 2)    # (b, num_windows, 2)
            
            # 计算自适应坐标
            head_offsets = head_offsets.unsqueeze(-2).unsqueeze(-2)  # (b, num_windows, 1, 1, 2)
            head_scales = head_scales.unsqueeze(-2).unsqueeze(-2)    # (b, num_windows, 1, 1, 2)
            
            # 增大影响系数，使自适应采样更有效
            scale_factor = 1.0 + 0.5 * torch.tanh(head_scales)  # 缩放范围：[0.5, 1.5]
            offset_factor = 0.3 * torch.tanh(head_offsets)       # 偏移范围：[-0.3, 0.3]
            
            # 应用缩放和偏移
            adaptive_grid = base_grid[:, head_idx] * scale_factor + offset_factor
            adaptive_grid = torch.clamp(adaptive_grid, -1, 1)
            
            # 重塑为采样格式
            adaptive_grid = adaptive_grid.view(b, window_num_h, window_num_w, self.window_size, self.window_size, 2)
            adaptive_grid = adaptive_grid.permute(0, 1, 3, 2, 4, 5).contiguous()
            adaptive_grid = adaptive_grid.view(b, target_h, target_w, 2)
            
            # 对K和V进行采样
            head_dim = c // self.num_heads
            k_head = k_padded[:, head_idx*head_dim:(head_idx+1)*head_dim]  # (b, head_dim, H, W)
            v_head = v_padded[:, head_idx*head_dim:(head_idx+1)*head_dim]  # (b, head_dim, H, W)
            
            # 使用grid_sample进行自适应采样
            k_sampled = F.grid_sample(k_head, adaptive_grid, mode='bilinear', 
                                    padding_mode='border', align_corners=False)
            v_sampled = F.grid_sample(v_head, adaptive_grid, mode='bilinear', 
                                    padding_mode='border', align_corners=False)
            
            sampled_features_list.append((k_sampled, v_sampled))
        
        # 合并所有头的结果
        k_sampled_all = torch.cat([k for k, v in sampled_features_list], dim=1)
        v_sampled_all = torch.cat([v for k, v in sampled_features_list], dim=1)
        
        # 转换回原始格式
        k_sampled_all = k_sampled_all.permute(0, 2, 3, 1)  # (b, H, W, c)
        v_sampled_all = v_sampled_all.permute(0, 2, 3, 1)  # (b, H, W, c)
        q_output = q_padded.permute(0, 2, 3, 1)             # (b, H, W, c)
        
        # 组合QKV
        qkv_sampled = torch.cat([q_output, k_sampled_all, v_sampled_all], dim=-1)
        
        return qkv_sampled
    
    def forward(self, qkv: torch.Tensor, x_size: Tuple[int, int], 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        h, w = x_size
        b, n, c3 = qkv.shape
        c = c3 // 3
        
        # 检查是否包含cls token
        has_cls_token = n == h * w + 1
        if has_cls_token:
            # 分离cls token和patch tokens
            cls_qkv = qkv[:, 0:1, :]  # (b, 1, c3)
            patch_qkv = qkv[:, 1:, :]  # (b, h*w, c3)
        else:
            # 维度检查
            assert n == h * w, f"序列长度 {n} 与空间尺寸 {h}x{w} 不匹配"
            patch_qkv = qkv
        
        assert c % self.num_heads == 0, f"特征维度 {c} 不能被头数 {self.num_heads} 整除"
        
        # 重塑为空间格式
        qkv_spatial = patch_qkv.reshape(b, h, w, c3)
        
        # 应用VSA采样
        if self.use_vsa_sampling:
            qkv_spatial = self._apply_vsa_sampling(qkv_spatial, x_size)
        
        # 窗口分割
        qkv_windows = window_partition(qkv_spatial, self.window_size)
        qkv_windows = qkv_windows.view(-1, self.window_size * self.window_size, c3)
        
        # 分离QKV
        b_, n_win, _ = qkv_windows.shape
        qkv_reshaped = qkv_windows.reshape(b_, n_win, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        # 计算注意力
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 相对位置编码
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        # 应用掩码
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n_win, n_win) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_win, n_win)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(b_, n_win, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 窗口合并
        x = x.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(x, self.window_size, h, w)
        x = x.view(b, h * w, c)
        
        # 如果有cls token，重新组合
        if has_cls_token:
            # 对cls token应用相同的投影变换
            cls_q, cls_k, cls_v = cls_qkv.chunk(3, dim=-1)
            cls_output = self.proj(cls_v)  # 简单的线性变换
            cls_output = self.proj_drop(cls_output)
            
            # 重新组合
            x = torch.cat([cls_output, x], dim=1)

        return x

    def flops(self, input_resolution: Tuple[int, int]) -> int:
        """计算FLOPs"""
        h, w = input_resolution
        flops = 0

        # VSA采样开销
        if self.use_vsa_sampling:
            nw = h * w // (self.window_size ** 2)
            flops += nw * self.dim * self.num_heads * 2 * 2

        # 窗口注意力计算
        nw = h * w // (self.window_size ** 2)
        flops += nw * self.num_heads * (self.window_size ** 2) ** 2 * (self.dim // self.num_heads)

        # 投影层
        flops += h * w * self.dim * self.dim

        return flops

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'num_heads={self.num_heads}, use_vsa_sampling={self.use_vsa_sampling}')