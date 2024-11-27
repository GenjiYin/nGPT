# Transformer(decoder-only) & nGPT
![图片缺失](/figure/nGPT.png)

The original Transformer is on the left, and nGPT is on the right. It's important to note that the input for nGPT should be normalized. 

You can read the paper on website: https://arxiv.org/pdf/2410.01131
## RoPE
There is no more detailed description than what is provided in the original paper. You can obtain further information about RoPE by visiting the paper's website (https://arxiv.org/pdf/2104.09864).
```python
# RoPE
def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos
    return q, k
```


## Self-attention of nGPT
You should also normalize $`q`$、$`k`$ ensuring that the dot product of every query and key is under control:

$$q\leftarrow Norm(q)s_{qk}$$

$$q\leftarrow Norm(k)s_{qk}$$

where $`s_{qk}\in R^{dk}`$ is a vector of trainable scaling factors for the i-th head.

```python
def sinusoidal_position_embedding...

def RoPE...

def l2_norm(x, dim=-1):
    down = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return x / down

class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()

        self.n_head = n_head

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.S_qk = nn.Parameter(torch.ones(d_model, dtype=torch.float32))


    def forward(self, data):
        B, T, C = data.size()

        q = self.q(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = RoPE(q, k)
        S_qk = self.S_qk.view(1, 1, self.n_head, C // self.n_head).transpose(1, 2)
        q = S_qk * l2_norm(q)
        k = S_qk * l2_norm(k)

        score = q@k.transpose(-2, -1) * math.sqrt(C // self.n_head)
        mask = torch.tril(torch.ones(T, T), diagonal=0).to(data.device)
        score = score.masked_fill(mask==0, -1e8)
        v = score@v
        v = v.transpose(1, 2).contiguous().view(B, T, C)

        return self.o(v)
```

## MLP of nGPT
1. You should normalize $W_u$、$`W_v`$、$`W_o`$; 
2. When calculating $`u`$ and $`v`$,  we introduce scaling factors $s_u\in R^{d_{MLP}}$ and $`s_v\in R^{d_{MLP}}`$. 

```python
import math
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_model):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, 2*4*d_model, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4*d_model, d_model, bias=False)

        self.S_uv = nn.Parameter(torch.ones(2*4*d_model, dtype=torch.float32))

    def forward(self, data):
        uv = self.fc1(data)
        S_uv = self.S_uv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        SwiGLU = u * self.silu(v)
        out = self.fc2(SwiGLU)
        return out
```

# nGPT model

```python
import torch
import math
from torch import nn

# RoPE
def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos
    return q, k

def l2_norm(x, dim=-1):
    down = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return x / down

class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()

        self.n_head = n_head

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.S_qk = nn.Parameter(torch.ones(d_model, dtype=torch.float32))


    def forward(self, data):
        B, T, C = data.size()

        q = self.q(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v(data).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = RoPE(q, k)
        S_qk = self.S_qk.view(1, 1, self.n_head, C // self.n_head).transpose(1, 2)
        q = S_qk * l2_norm(q)
        k = S_qk * l2_norm(k)

        score = q@k.transpose(-2, -1) * math.sqrt(C // self.n_head)
        mask = torch.tril(torch.ones(T, T), diagonal=0).to(data.device)
        score = score.masked_fill(mask==0, -1e8)
        v = score@v
        v = v.transpose(1, 2).contiguous().view(B, T, C)

        return self.o(v)

        
class MLP(nn.Module):
    def __init__(self, d_model):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, 2*4*d_model, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4*d_model, d_model, bias=False)

        self.S_uv = nn.Parameter(torch.ones(2*4*d_model, dtype=torch.float32))

    def forward(self, data):
        uv = self.fc1(data)
        S_uv = self.S_uv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        SwiGLU = u * self.silu(v)
        out = self.fc2(SwiGLU)
        return out

class Block(nn.Module):
    def __init__(self, d_model, n_head, device='cpu'):
        super(Block, self).__init__()

        self.atten = Attention(d_model, n_head).to(device)
        self.mlp = MLP(d_model).to(device)

        self.lr_atten = nn.Parameter(torch.ones(d_model, dtype=torch.float32)).to(device)
        self.lr_mlp = nn.Parameter(torch.ones(d_model, dtype=torch.float32)).to(device)
    
    def forward(self, data):
        h_att = self.atten(data)
        lr = torch.abs(self.lr_atten)

        h = l2_norm(data)
        h_att = l2_norm(h_att)

        res = h + lr * h_att
        h = l2_norm(res)

        h_mlp = self.mlp(h)
        lr = torch.abs(self.lr_mlp)
        h_mlp = l2_norm(h_mlp)
        res = h + lr * h_mlp
        return l2_norm(res)
    
class nGPT(nn.Module):
    def __init__(
        self, 
        d_model, 
        n_head,
        max_voc_dim, 
        block_num, 
        lora=None, 
        device='cpu'
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(int(max_voc_dim), d_model).to(device)

        # 区块堆叠
        self.decoders = nn.Sequential(
            *[Block(d_model=d_model, n_head=n_head, device=device) for _ in range(block_num)]
        )

        self.outlayer = nn.Linear(d_model, int(max_voc_dim)).to(device)

        self.apply(self._custom_init)
    
    def _custom_init(self, m):
        if isinstance(m, nn.Linear):
            weight = m.weight.data
            norm = torch.sqrt(torch.sum(weight**2, dim=-1, keepdim=True))
            m.weight.data = weight / norm
        

    def forward(self, data):
        out = self.embedding_layer(data)
        out = self.decoders(out)
        out = self.outlayer(out)
        return out
```
