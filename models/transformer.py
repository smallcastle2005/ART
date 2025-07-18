import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)  # 缩放因子
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 分头处理
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 应用注意力掩码
        if mask is not None:
            mask = mask.unsqueeze(1).to(dtype=torch.bool)
            fill_value = torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
            scores = scores.masked_fill(~mask, fill_value)
        
        # 计算注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)
        
        # 应用注意力权重到值向量
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(output)

# 前馈网络+注意力机制——网络层的定义
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, attention_dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + 层归一化
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# 关键组件1：注意力窗口预测器
# 工作原理：
# 1. 输入特征张量x。
# 2. 对特征张量进行平均池化，得到全局特征。
# 3. 全局特征通过两层全连接网络，预测窗口的参数。
# 4. 输出窗口参数，包括中心点和尺寸。
# 5. 窗口参数通过sigmoid函数约束在合理范围内。
# 6. 输出窗口参数，用于注意力机制的计算。
class WindowPredictor(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model // 2, 4)  # 输出4个参数: [center_x, center_y, width, height]
        
        # 初始化窗口参数在合理范围内
        with torch.no_grad():
            # 中心点初始化在0.5附近，窗口大小初始化在0.4附近
            self.fc2.bias.data = torch.tensor([0.5, 0.5, 0.4, 0.4])
            self.fc2.weight.data *= 0.1
    # 前向传播，最后输出窗口参数用于学习
    def forward(self, x, target=None):
        # 平均池化获取全局特征
        features = torch.mean(x, dim=1)
        
        # 训练时添加轻微噪声增强鲁棒性
        if self.training:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        # 两层全连接网络预测窗口参数
        x = self.fc1(features)
        x = self.relu(x)
        window_params = self.fc2(x)
        
        # 使用sigmoid约束输出范围
        center_params = torch.sigmoid(window_params[:, :2]) * 0.4 + 0.3  # 中心点范围[0.3, 0.7]
        size_params = torch.sigmoid(window_params[:, 2:]) * 0.3 + 0.3    # 尺寸范围[0.3, 0.6]
        # 这样约束是为了整个矩形窗口还是能够位于图片较为中心的区域
        # 同时窗口参数的范围限制在[0.3, 0.7]，确保窗口不会超出图像边界
        
        return torch.cat([center_params, size_params], dim=1)

# 关键组件2：注意力窗口优化器
# 工作原理：
# 1. 输入当前特征张量x和当前窗口参数。
# 2. 计算当前窗口参数的梯度。
# 3. 引入动量机制，更新窗口参数。
# 4. 输出优化后的窗口参数。 
class WindowOptimizer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.refine_net = nn.Sequential(
            nn.Linear(d_model + 4, d_model // 2),  # 输入: 特征 + 当前窗口参数
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)             # 输出: 窗口参数更新量
        )
    
    def forward(self, features, current_window, loss):
        # 确保梯度计算
        if not current_window.requires_grad:
            current_window.requires_grad_(True)
        
        # 计算窗口参数梯度
        try:
            window_grad = torch.autograd.grad(loss, current_window, create_graph=True)[0]
        except RuntimeError as e:
            print(f"警告：窗口梯度计算失败: {e}")
            return current_window.detach()
        
        loss.backward(retain_graph=True)
        window_grad = current_window.grad
        # 裁剪梯度
        window_grad = torch.clamp(window_grad, -0.1, 0.1)
        
        # 平均特征并与当前窗口拼接
        features_avg = torch.mean(features, dim=1)
        input_features = torch.cat([features_avg, current_window], dim=-1)
        
        # 预测窗口更新量
        window_delta = self.refine_net(input_features)
        window_delta = torch.tanh(window_delta) * 0.1  # 限制更新幅度
        
        # 应用更新并约束范围
        new_window = current_window + window_delta
        new_window = torch.sigmoid(new_window)
        return new_window

# 完整的Transformer模型
class ImageTransformer(nn.Module):
    def __init__(
        self,
        num_classes=None,
        patch_size=None,
        d_model=None,
        num_heads=None,
        num_layers=None,
        d_ff=None,
        dropout=None,
        window_optimization_steps=None, 
        image_size=None,
    ):
        super().__init__()
        
        # 从配置中获取参数或使用默认值
        self.num_classes = num_classes if num_classes is not None else config.NUM_CLASSES
        self.patch_size = patch_size if patch_size is not None else config.MODEL['patch_size']
        self.d_model = d_model if d_model is not None else config.MODEL['d_model']
        self.num_heads = num_heads if num_heads is not None else config.MODEL['num_heads']
        self.num_layers = num_layers if num_layers is not None else config.MODEL['num_layers']
        self.d_ff = d_ff if d_ff is not None else config.MODEL['d_ff']
        self.dropout = dropout if dropout is not None else config.MODEL['dropout']
        self.window_optimization_steps = (
            window_optimization_steps
            if window_optimization_steps is not None
            else config.MODEL['window_optimization_steps']
        )
        self.image_size = image_size if image_size is not None else config.IMAGE_SIZE
        
        # 计算图像块数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)
        

         # 增加层归一化和残差连接
        self.transformer_blocks = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout), # 补充参数
                nn.LayerNorm(d_model)  # 新增层归一化
            ) for _ in range(num_layers)
        ])

        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 图像嵌入层
        self.embedding = nn.Sequential(
            nn.Conv2d(3, self.d_model, kernel_size=self.patch_size, stride=self.patch_size),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(),
            nn.Flatten(2)
        )
        
        # 解码器头
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model*2),  # 扩大容量
            nn.GELU(),                      # 替换ReLU
            nn.Dropout(dropout+0.1),         # 增加Dropout
            nn.Linear(d_model*2, num_classes)
        )
        
        # 上采样层
        self.upsample = nn.ConvTranspose2d(
            self.num_classes, self.num_classes,
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # 窗口预测器和优化器
        self.window_predictor = WindowPredictor(self.d_model, self.num_classes)
        self.window_optimizer = WindowOptimizer(self.d_model)
        
        # 窗口参数统计
        self.register_buffer('window_params_sum', torch.zeros(4))
        self.window_params_count = 0
        
        # 初始化权重
        self._init_weights()
    
    # 初始化权重
    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
        self.apply(init_weights)
    
    # 创建注意力编码
    def apply_attention_window(self, x, window_params):
        batch_size, seq_len, _ = x.shape
        size = int(math.sqrt(seq_len))  # 网格尺寸
        
        # 创建归一化网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, size, device=x.device),
            torch.linspace(0, 1, size, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).flatten(1, 2)  # [2, seq_len]
        
        # 提取中心点和尺寸
        centers = window_params[:, :2].unsqueeze(2)  # [batch_size, 2, 1]
        sizes = window_params[:, 2:].unsqueeze(2)    # [batch_size, 2, 1]
        
        # 计算每个网格点到窗口中心的归一化距离
        distances = torch.norm(
            (grid.unsqueeze(0) - centers) / (sizes + 1e-6),
            dim=1 
        )
        
        # 创建掩码: 距离小于1.0的位置设为True
        mask = (distances <= 1.0).float()
        
        # 扩展为注意力掩码 [batch_size, seq_len, seq_len]
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        return mask
    
    # 关键函数：计算窗口损失
    # 工作原理：
    # 1. 输入特征张量x和目标标签target。
    # 2. 预测窗口参数。
    # 3. 应用注意力窗口到特征。
    # 4. 计算交叉熵损失。
    # 5. 输出窗口损失。
    # 这样就能通过window_loss这一参数来评估注意力窗口的好坏，再进一步利用优化器优化窗口
    def compute_window_loss(self, window_params, features, target):
        batch_size = features.size(0)
        h_patches = w_patches = self.image_size // self.patch_size

        # 添加类别权重平衡
        class_weights = 1.0 / (torch.bincount(target.flatten()) + 1e-6)
        class_weights = class_weights / class_weights.sum()
        
        if target.dim() == 4 and target.size(1) > 1:  # one-hot编码
            target_high_res = target.argmax(dim=1)
        else:
            target_high_res = target
    
        # 确保目标张量是长整型
        target_high_res = target_high_res.long()
    
        # 创建全类别权重张量
        class_weights = torch.ones(self.num_classes, device=target_high_res.device)

        # 计算当前批次中出现的类别权重
        unique_classes = torch.unique(target_high_res)
        class_counts = torch.bincount(target_high_res.flatten(), minlength=self.num_classes)

        # 避免除零错误
        class_counts = class_counts.float() + 1e-6
    
        # 计算逆频率权重
        inverse_freq = 1.0 / class_counts
        inverse_freq[class_counts == 0] = 0  # 未出现的类别权重设为0
    
        # 归一化权重
        class_weights = inverse_freq / inverse_freq.sum()
        
        # 创建patch级目标
        if target.dim() == 4:
            target_patch = F.adaptive_max_pool2d(target_high_res.float(), (h_patches, w_patches)).long()
        else:
            target_patch = F.adaptive_max_pool2d(target.float(), (h_patches, w_patches)).long()
        
        # 预测分类结果
        pred_logits = self.decoder_head(features)  # [B, seq_len, C]
        
        # 计算注意力权重
        attention_mask = self.apply_attention_window(features, window_params)
        window_weights = attention_mask.mean(dim=1)  # 平均权重 [B, seq_len]
        
        # Patch级损失
        patch_loss = F.cross_entropy(
            pred_logits.transpose(1, 2).reshape(-1, self.num_classes),
            target_patch.reshape(-1).long(),
            weight=class_weights,
            reduction='none'
        ).view(batch_size, h_patches, w_patches)
        
        # 像素级损失
        high_res_output = self.upsample(
            pred_logits.transpose(1, 2).view(batch_size, self.num_classes, h_patches, w_patches)
        )
        pixel_loss = F.cross_entropy(
            high_res_output,
            target_high_res.long(),
            weight=class_weights,
            reduction='none'
        )
        
        # 加权损失组合
        weighted_patch_loss = (patch_loss * window_weights.view(batch_size, h_patches, w_patches)).mean()
        weighted_pixel_loss = pixel_loss.mean()
        
        # 窗口尺寸惩罚项
        size_penalty = torch.mean(window_params[:, 2:] - 0.4).abs()
        
        # 组合损失: 50%像素损失 + 30%块损失 + 20%尺寸惩罚
        return 0.5 * weighted_pixel_loss + 0.3 * weighted_patch_loss + 0.2 * size_penalty

    def forward(self, x, target=None):
        batch_size = x.size(0)
        
        # 1. 特征提取 
        x = self.embedding(x)  # [B, d_model, H*W]
        x = x.transpose(1, 2)  # [B, H*W, d_model]
        x = x + self.pos_encoding[:, :x.size(1), :]  # 添加位置编码
        
        # 2. 预测初始窗口 
        initial_features = x.clone()
        window_params = self.window_predictor(initial_features)
        all_window_losses = []
        
        # 3. 窗口优化循环 (仅训练时) 
        if self.training and target is not None:
            for step in range(self.window_optimization_steps):
                # 3.1 应用当前窗口创建注意力掩码
                attention_mask = self.apply_attention_window(x, window_params)
                
                # 3.2 使用当前掩码进行特征提取
                current_features = x.clone()
                for transformer in self.transformer_blocks:
                    current_features = transformer(current_features, mask=attention_mask)
                
                # 3.3 计算窗口损失
                window_loss = self.compute_window_loss(window_params, current_features, target)
                all_window_losses.append(window_loss)
                
                # 3.4 优化窗口参数
                window_params = self.window_optimizer(current_features, window_params, window_loss)
                
                # 3.5 记录窗口统计信息
                with torch.no_grad():
                    self.window_params_sum += window_params.detach().mean(dim=0)
                    self.window_params_count += 1
        
        # 4. 最终特征提取 
        # 使用最终优化的窗口参数
        final_mask = self.apply_attention_window(x, window_params)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=final_mask)
        
        # === 5. 解码和上采样 ===
        x = self.decoder_head(x)  # [B, seq_len, C]
        h = w = self.image_size // self.patch_size
        x = x.transpose(1, 2).view(batch_size, -1, h, w)
        output = self.upsample(x)  # [B, C, H, W]
        output = F.softmax(output, dim=1)
        # 返回结果和损失
        if self.training and target is not None and all_window_losses:
            total_window_loss = sum(all_window_losses) / len(all_window_losses)
            return output, total_window_loss
        
        return output

    # 统计窗口参数
    def get_average_window_params(self):
        if self.window_params_count == 0:
            return None
        avg_params = self.window_params_sum / self.window_params_count
        self.window_params_sum.zero_()
        self.window_params_count = 0
        return avg_params.cpu()

    # 获取当前窗口信息
    def get_current_window_stats(self):
        if self.window_params_count == 0:
            return None
        return {
            'center_x': float(self.window_params_sum[0] / self.window_params_count),
            'center_y': float(self.window_params_sum[1] / self.window_params_count),
            'width': float(self.window_params_sum[2] / self.window_params_count),
            'height': float(self.window_params_sum[3] / self.window_params_count),
            'count': self.window_params_count
        }