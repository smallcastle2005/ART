import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import VSA_ATD_Config
from core.block.transformer_block import UnifiedTransformerBlock
from core.token_dictionary import TokenDictionary

class PatchEmbed(nn.Module):
    """图像到Patch嵌入 - 针对CIFAR100优化"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 修改为序列结构
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.norm=nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

# 修改VSA_ATD_VisionTransformer类中的blocks构建部分
class VSA_ATD_VisionTransformer(nn.Module):
    def __init__(self, config = VSA_ATD_Config()):
        super().__init__()
        self.config = config
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim
        )
        
        # 计算输入分辨率
        patch_res = self.patch_embed.img_size // self.patch_embed.patch_size
        self.input_resolution = (patch_res, patch_res) 
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, config.embed_dim)
        )
        
        # 类别令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        
        # 构建Transformer层
        self.blocks = nn.ModuleList()
        total_blocks = len(config.depths)  # 4个block
        
        for i in range(total_blocks):
            block = UnifiedTransformerBlock(
                dim=config.embed_dim,
                input_resolution=self.input_resolution,
                depth=config.depths[i],
                num_heads=config.num_heads[i],
                window_size=config.vsa_window_size,
                category_size=config.category_size,
                num_tokens=config.num_tokens,
                reducted_dim=config.reducted_dim,
                convffn_kernel_size=3,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_vsa=True  # 从False改为True
            )
            self.blocks.append(block)
        
        # 最终层归一化
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 分类头
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 初始化令牌字典
        from core.token_dictionary import TokenDictionary
        self.token_dict = TokenDictionary(
            num_tokens=config.num_tokens, 
            token_dim=config.embed_dim
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x):
        # 1. 保留梯度关键点
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, x.shape[2])  # 匹配第三维
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 2. 令牌字典处理（确保梯度流动）
        td = self.token_dict.get_batched_tokens(x.size(0))
        
        # 3. Transformer块处理
        for block in self.blocks:
            # 使用梯度检查点（可选）
            x, td = torch.utils.checkpoint.checkpoint(block, x, td, self.input_resolution, self.token_dict)
            
        # 4. 分类头（确保最后层梯度）
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits