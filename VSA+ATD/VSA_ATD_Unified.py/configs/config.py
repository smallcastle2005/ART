class VSA_ATD_Config:
    # CIFAR100优化配置
    img_size = 32              # CIFAR100原始尺寸
    patch_size = 4             # 小patch适合小图像 (8×8 patches)
    embed_dim = 192            # 增加嵌入维度提升表达能力
    num_classes = 100          # CIFAR100类别数
    
    # VSA相关 - 针对小图像优化
    vsa_window_size = 8        # 适合32×32图像的窗口大小
    adaptive_window = True     # 启用自适应窗口
    
    # ATD相关 - 针对细粒度分类优化
    category_size = 256        # 增加令牌字典以处理100类
    num_tokens = 128          # 增加令牌数量
    reducted_dim = 16         # 适当的降维维度
    
    # 模型结构 - 平衡性能和效率
    depths = [2, 2, 6, 2]     # 中间层加深处理复杂特征
    num_heads = [3, 6, 12, 24] # 逐层增加注意力头
    mlp_ratio = 4             # 标准MLP比率
    
    # 训练相关
    drop_rate = 0.1           # Dropout率
    attn_drop_rate = 0.1      # 注意力Dropout
    drop_path_rate = 0.1      # DropPath率