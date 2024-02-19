import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_k, x_v):
        q = self.to_q(x_q)
        k = self.to_k(x_k)
        v = self.to_v(x_v)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.atten = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x_q, x_k, x_v):
        x_q = self.norm(x_q)
        x_k = self.norm(x_k)
        x_v = self.norm(x_v)
        x = self.atten(x_q, x_k, x_v) + x_v
        x = self.ff(x) + x
        return x


class ViT_fusion(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding_q = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_k = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_v = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, heads, dim_head, mlp_dim, dropout)

        self.return_linear = nn.Linear(dim, patch_dim)
        self.reshape = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_height // self.patch_height,
                                 w=self.image_width // self.patch_width, p1=self.patch_height, p2=self.patch_width)

    def forward(self, x_q, x_k, x_v):
        b, c, h, w = x_q.shape
        x_q = self.to_patch_embedding_q(x_q)
        x_k = self.to_patch_embedding_k(x_k)
        x_v = self.to_patch_embedding_v(x_v)

        x_q += self.pos_embedding
        x_k += self.pos_embedding
        x_v += self.pos_embedding
        x_q = self.dropout(x_q)
        x_k = self.dropout(x_k)
        x_v = self.dropout(x_v)

        x = self.transformer(x_q, x_k, x_v)
        x = self.return_linear(x)
        x = self.reshape(x)
        return x


if __name__ == '__main__':
    net = ViT_fusion(image_size=(10, 6), patch_size=(5, 3), dim=1024, heads=1, mlp_dim=2048, channels=256, dim_head=64)
    x_q = torch.rand((1, 256, 10, 6))
    x_k = torch.rand((1, 256, 10, 6))
    x_v = torch.rand((1, 256, 10, 6))
    out = net(x_q, x_k, x_v)
    print(out.size())
