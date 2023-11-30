import torch
from torch import nn, einsum
import torch.nn.functional as F 
from einops import rearrange


def l2norm(t):
    return F.normalize(t, dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
    
    def forward(self, x):
        if x.dtype == torch.float32:
            eps = 1e-5
        else:
            eps = 1e-3
        
        V = torch.var(x, dim=1, unbiased=False, keepdim=True)
        E = torch.mean(x, dim=1, keepdim=True)
        
        return (x - E) * (V + eps).rsqrt() * self.g


class CustomLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 3, d, dh, w
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        # 3, d, dh / head, w * dim_head
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', 
                                          h = self.heads))
        
        # activate
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        
        # attention value
        v = v / (h * w)
        
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        
        # alignment
        out = rearrange(out,'b h c (x y) -> b (h c) x y',
                            h=self.heads,
                            x=h,
                            y=w)
        
        return self.to_out(out)
    
class CustomAttention(nn.Module):
        def __init__(self, dim, heads=4, dim_head=32):
            super().__init__()
            self.scale = dim_head ** -0.5
            self.heads = heads 
            hidden_dim = dim_head * heads
            
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
         
        def forward(self, x):
            b, c, h, w = x.shape
        
            # 3, d, dh, w
            qkv = self.to_qkv(x).chunk(3, dim=1)
            
            # 3, d, dh / head, w * dim_head
            # print(qkv[0].size())
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
            
            q = q * self.scale 
            
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            
            out = rearrange(out,'b h (x y) d -> b (h d) x y',
                                x=h, y=w)
            
            return self.to_out(out)
            
            



        