# Transformer(decoder-only) & nGPT
![图片缺失](/figure/nGPT.png)

Original Transformer is on the left, nGPT on the right. It's note that input of nGPT should be normalized. 

## RoPE
There's no more specific description than original paper. You can get more information about RoPE by paper website (https://arxiv.org/pdf/2104.09864).
```python
def build_rope_matrix(max_seq_len, dim, device):
    """
    构建RoPE的旋转矩阵
    :param max_seq_len: 序列长度
    :param dim: 嵌入维度
    :param device: 设备（CPU或GPU）
    :return: 旋转矩阵，形状为 (max_seq_len, dim, dim)
    """
    # 定义旋转角度
    theta = 10000 ** (-torch.arange(0, dim // 2, device=device) * 2 / dim)
    # 初始化旋转矩阵
    rope_matrix = torch.zeros((max_seq_len, dim, dim), device=device)
    # 构建旋转矩阵
    for m in range(max_seq_len):
        for i in range(dim // 2):
            rope_matrix[m, 2 * i, 2 * i] = torch.cos(m * theta[i])
            rope_matrix[m, 2 * i, 2 * i + 1] = -torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i] = torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i + 1] = torch.cos(m * theta[i])
    return rope_matrix
```


## Self-attention of nGPT
1. You need to normalize $W_q$、$W_k$、$W_v$、$W_o$ so that the computed dot products with $h$ can be viewed as cosine similarity between unit norm vectors bounded in [-1, 1]; 
2. You should also normalize $q$、$k$ ensuring that the dot product of every query and key is under control:
$$q\leftarrow Norm(q)s_{qk}$$
$$q\leftarrow Norm(k)s_{qk}$$
where $s_{qk}\in R^{dk}$ is a vector of trainable scaling factors for the i-th head.

```python
import math
import torch
import torch.nn as nn

def build_rope_matrix(max_seq_len, dim, device='cpu'):
    """
    构建RoPE的旋转矩阵
    :param max_seq_len: 序列长度
    :param dim: 嵌入维度
    :param device: 设备（CPU或GPU）
    :return: 旋转矩阵，形状为 (max_seq_len, dim, dim)
    """
    # 定义旋转角度
    theta = 10000 ** (-torch.arange(0, dim // 2, device=device) * 2 / dim)
    # 初始化旋转矩阵
    rope_matrix = torch.zeros((max_seq_len, dim, dim), device=device)
    # 构建旋转矩阵
    for m in range(max_seq_len):
        for i in range(dim // 2):
            rope_matrix[m, 2 * i, 2 * i] = torch.cos(m * theta[i])
            rope_matrix[m, 2 * i, 2 * i + 1] = -torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i] = torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i + 1] = torch.cos(m * theta[i])
    return rope_matrix

class l2_normalize(nn.Module):
    """
    L2正则化
    """
    def __init__(self, dim=-1):
        super(l2_normalize, self).__init__()
        self.dim = dim
    
    def forward(self, data):
        out = data / torch.sqrt((data**2).sum(dim=self.dim, keepdim=True) + 1e-8)
        return out

class Normalize_attention(nn.Module):
    """
    标准化自注意力
    """
    def __init__(self, d_model, n_head, max_seq_len, device='cpu', lora=None):
        """
        这里的lora是我个人的定制化模型组件(这里你用不到)
        """
        super(Normalize_attention, self).__init__()
        self.W_q = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head)).to(device)
        self.W_k = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head)).to(device)
        self.W_v = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head)).to(device)
        self.W_o = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head)).to(device)

        self.S = nn.Parameter(torch.randn(1, n_head, 1, d_model // n_head)).to(device)

        self.norm_layer_1 = l2_normalize(dim=-2)
        self.norm_layer_2 = l2_normalize(dim=-1)

        self.n_head = n_head
        self.softmax = nn.Softmax(dim=-1)

        # 初始化RoPE张量
        self.rope_matrix = build_rope_matrix(max_seq_len, d_model // n_head, device).transpose(-2, -1)
    
    def forward(self, data):
        """
        data: [batch_size, length, d_model], 输入之前已经标准化好了的
        """
        # 标准化权重
        W_q = self.norm_layer_1(self.W_q)   # [n_head, d_model, d_model / n_head]
        W_k = self.norm_layer_1(self.W_k)
        W_v = self.norm_layer_1(self.W_v)
        W_o = self.norm_layer_1(self.W_o)

        batch_size, length, d_model = data.size()
        data = data.view(batch_size, 1, length, d_model)

        q = (data@W_q).view(batch_size, self.n_head, length, 1, d_model // self.n_head)
        q = (q@self.rope_matrix).view(batch_size, self.n_head, length, d_model // self.n_head)    # RoPE位置嵌入

        k = (data@W_k).view(batch_size, self.n_head, length, 1, d_model // self.n_head)
        k = (k@self.rope_matrix).view(batch_size, self.n_head, length, d_model // self.n_head)    # RoPE位置嵌入

        v = data@W_v

        # 标准化q、k、v
        q = self.norm_layer_2(q) * self.S
        k = self.norm_layer_2(k) * self.S

        score = q@k.transpose(-2, -1) * math.sqrt(d_model // self.n_head)
        mask = torch.tril(torch.ones(length, length), diagonal=0).to(data.device)
        score = score.masked_fill(mask==0, -1e8)
        v = score@v
        W_o = W_o.transpose(0, 1).contiguous().view(d_model, d_model)
        v = v.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return v @ W_o
```

## MLP of nGPT
1. You should normalize $W_u$、$W_v$、$W_o$; 
2. When calculating $u$ and $v$,  we introduce scaling factors $s_u\in R^{d_{MLP}}$ and $s_v\in R^{d_{MLP}}$. 

```python
import math
import torch
from torch import nn

class l2_normalize(nn.Module):
    """
    L2正则化
    """
    def __init__(self, dim=-1):
        super(l2_normalize, self).__init__()
        self.dim = dim
    
    def forward(self, data):
        out = data / torch.sqrt((data**2).sum(dim=self.dim, keepdim=True) + 1e-8)
        return out

class MLP(nn.Module):
    def __init__(self, d_model):
        super(MLP, self).__init__()
        self.W_u = nn.Parameter(torch.randn(d_model, d_model * 4))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model * 4))
        self.W_o = nn.Parameter(torch.randn(d_model * 4, d_model))

        self.S_u = nn.Parameter(torch.randn(1, 1, d_model * 4))
        self.S_v = nn.Parameter(torch.randn(1, 1, d_model * 4))

        self.norm_layer_1 = l2_normalize(dim=-2)
    
    def forward(self, data):
        """
        data: 依旧是已经标准化好了的
        """
        b, l, d_model = data.size()
        W_u = self.norm_layer_1(self.W_u)
        W_v = self.norm_layer_1(self.W_v)
        W_o = self.norm_layer_1(self.W_o)

        u = (data@W_u) * self.S_u
        v = (data@W_v) * self.S_v * math.sqrt(d_model)

        out = v / (1 + torch.exp(-v)) * u
        out = out@W_o
        return out
```

# nGPT model

```python
import math
import torch
from torch import nn

# MLP
class MLP(nn.Module):
    def __init__(self, d_model):
        super(MLP, self).__init__()
        self.W_u = nn.Parameter(torch.randn(d_model, d_model * 4))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model * 4))
        self.W_o = nn.Parameter(torch.randn(d_model * 4, d_model))

        self.S_u = nn.Parameter(torch.randn(1, 1, d_model * 4))
        self.S_v = nn.Parameter(torch.randn(1, 1, d_model * 4))

        self.norm_layer_1 = l2_normalize(dim=-2)
    
    def forward(self, data):
        """
        data: 依旧是已经标准化好了的
        """
        b, l, d_model = data.size()
        W_u = self.norm_layer_1(self.W_u)
        W_v = self.norm_layer_1(self.W_v)
        W_o = self.norm_layer_1(self.W_o)

        u = (data@W_u) * self.S_u
        v = (data@W_v) * self.S_v * math.sqrt(d_model)

        out = v / (1 + torch.exp(-v)) * u
        out = out@W_o
        return out

# nomalization
class l2_normalize(nn.Module):
    """
    L2正则化
    """
    def __init__(self, dim=-1):
        super(l2_normalize, self).__init__()
        self.dim = dim
    
    def forward(self, data):
        out = data / torch.sqrt((data**2).sum(dim=self.dim, keepdim=True) + 1e-8)
        return out

# Attention
def build_rope_matrix(max_seq_len, dim, device='cpu'):
    """
    构建RoPE的旋转矩阵
    :param max_seq_len: 序列长度
    :param dim: 嵌入维度
    :param device: 设备（CPU或GPU）
    :return: 旋转矩阵，形状为 (max_seq_len, dim, dim)
    """
    # 定义旋转角度
    theta = 10000 ** (-torch.arange(0, dim // 2, device=device) * 2 / dim)
    # 初始化旋转矩阵
    rope_matrix = torch.zeros((max_seq_len, dim, dim), device=device)
    # 构建旋转矩阵
    for m in range(max_seq_len):
        for i in range(dim // 2):
            rope_matrix[m, 2 * i, 2 * i] = torch.cos(m * theta[i])
            rope_matrix[m, 2 * i, 2 * i + 1] = -torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i] = torch.sin(m * theta[i])
            rope_matrix[m, 2 * i + 1, 2 * i + 1] = torch.cos(m * theta[i])
    return rope_matrix

class Normalize_attention(nn.Module):
    """
    标准化自注意力
    """
    def __init__(self, d_model, n_head, max_seq_len, lora=None):
        """
        这里的lora是我个人的定制化模型组件(这里你用不到)
        """
        super(Normalize_attention, self).__init__()
        self.W_q = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head))
        self.W_k = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head))
        self.W_v = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head))
        self.W_o = nn.Parameter(torch.randn(n_head, d_model, d_model // n_head))

        self.S = nn.Parameter(torch.randn(1, n_head, 1, d_model // n_head))

        self.norm_layer_1 = l2_normalize(dim=-2)
        self.norm_layer_2 = l2_normalize(dim=-1)

        self.n_head = n_head
        self.softmax = nn.Softmax(dim=-1)

        # 初始化RoPE张量
        self.rope_matrix = build_rope_matrix(max_seq_len, d_model // n_head, self.W_o.device).transpose(-2, -1)
    
    def forward(self, data):
        """
        data: [batch_size, length, d_model], 输入之前已经标准化好了的
        """
        # 标准化权重
        W_q = self.norm_layer_1(self.W_q)   # [n_head, d_model, d_model / n_head]
        W_k = self.norm_layer_1(self.W_k)
        W_v = self.norm_layer_1(self.W_v)
        W_o = self.norm_layer_1(self.W_o)

        batch_size, length, d_model = data.size()
        data = data.view(batch_size, 1, length, d_model)

        q = (data@W_q).view(batch_size, self.n_head, length, 1, d_model // self.n_head)
        q = (q@self.rope_matrix).view(batch_size, self.n_head, length, d_model // self.n_head)    # RoPE位置嵌入

        k = (data@W_k).view(batch_size, self.n_head, length, 1, d_model // self.n_head)
        k = (k@self.rope_matrix).view(batch_size, self.n_head, length, d_model // self.n_head)    # RoPE位置嵌入

        v = data@W_v

        # 标准化q、k、v
        q = self.norm_layer_2(q) * self.S
        k = self.norm_layer_2(k) * self.S

        score = q@k.transpose(-2, -1) * math.sqrt(d_model // self.n_head)
        mask = torch.tril(torch.ones(length, length), diagonal=0).to(data.device)
        score = score.masked_fill(mask==0, -1e8)
        v = score@v
        W_o = W_o.transpose(0, 1).contiguous().view(d_model, d_model)
        v = v.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return v @ W_o

# One Block of nGPT
class Block(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, lora=None):
        super(Block, self).__init__()

        self.norm_layer = l2_normalize(dim=-1)
        self.atten = Normalize_attention(
            d_model=d_model, 
            n_head=n_head, 
            max_seq_len=max_seq_len, 
            lora=lora
        )
        self.mlp = MLP(d_model=d_model)

        self.a_1 = nn.Parameter(torch.randn(1, 1, d_model))
        self.a_2 = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, data):
        # 数据标准化
        h = self.norm_layer(data)

        # 注意力 + 标准化层
        h_A = self.norm_layer(self.atten(h))
        h = self.norm_layer(h + self.a_1 * (h_A - h))

        # 全连接层
        h_M = self.norm_layer(self.mlp(h))
        h = self.norm_layer(h + self.a_2 * (h_A - h))
        return h 

# nGPT
class nGPT(nn.Module):
    def __init__(
        self, 
        d_model, 
        n_head, 
        max_seq_len, 
        max_voc_dim, 
        block_num, 
        lora=None, 
        device='cpu'
    ):
        """
        :params d_model: 词向量维度
        :params n_head: 注意力机制的头数
        :params max_seq_len: 序列的最大长度
        :params max_voc_dim: 词的总数
        :params block_num: 解码器的个数
        :params lora: 是否启用lora层
        :params device: 在什么设备上训练
        """
        super().__init__()
        # 词嵌入
        self.embedding_layer = nn.Embedding(int(max_voc_dim), d_model).to(device)

        # 区块堆叠
        self.decoders = nn.Sequential(
            *[Block(d_model=d_model, n_head=n_head, max_seq_len=max_seq_len, lora=lora) for _ in range(block_num)]
        ).to(device)

        # 输出
        self.outlayer = nn.Linear(d_model, int(max_voc_dim)).to(device)
    
    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.decoders(out)
        out = self.outlayer(out)
        return out
```