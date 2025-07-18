import torch
import torch.nn.functional as F

# 自适应类别注意力所需工具函数AC_MSA
def index_reverse(index):
    """索引反转函数"""
    """生成反向索引，用于恢复原始顺序"""
    index_r = torch.zeros_like(index) # 创建一个与index相同形状的零张量
    ind = torch.arange(0, index.shape[-1]).to(index.device) # 创建从0到索引最后一维的张量（如tensor([[2, 0, 1], [1, 2, 0]])）得到的ind为tensor([0,1,2])
    for i in range(index.shape[0]): # 遍历索引的第一维
        index_r[i, index[i, :]] = ind # 将ind中的元素按照index[i, :]的顺序放入index_r[i, :]中，如[2,0,1]会变为[1,2,0]
    return index_r # 最终索引的顺序就会反过来

def feature_shuffle(x, index):
    """特征重排序函数"""
    """特征分组排序，用过相似度图将特征按类别重新排序"""
    """同时在分组后将特征恢复到原始排列"""
    dim = index.dim() # 获取索引的维度
    assert x.shape[:dim] == index.shape
    
    for _ in range(x.dim() - index.dim()): # 为索引添加维度，使其与特征的维度相同
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    
    shuffled_x = torch.gather(x, dim=dim-1, index=index) # 使用gather根据索引重排序特征
    return shuffled_x # 最后token就会根据索引的顺序重新排列

# 窗口注意力所需工具函数（注意：这里是原windowattention的工具函数）
def window_partition(x, window_size):
    """窗口分割"""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """窗口合并"""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x