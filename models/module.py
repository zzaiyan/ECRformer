"""ECRformer building blocks and attention modules.

Contains all attention mechanisms, feed-forward networks, and transformer blocks
used in the ECRformer architecture.
"""

import functools
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath
from einops import rearrange

try:
    from models.module_util import PreNorm, LayerNorm, RestormerLayerNorm
except ImportError:
    from module_util import PreNorm, LayerNorm, RestormerLayerNorm


def NonLinearity(inplace=False):
    return nn.GELU()


# ---------------------------------------------------------------------------
# Linear Attention
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class LinearAttnBlock(nn.Module):
    def __init__(self, feats, scale=0.01):
        super().__init__()
        self.attn = nn.Sequential(LayerNorm(feats), LinearAttention(feats))
        self.scale = scale

    def forward(self, x):
        return self.attn(x) * self.scale + x


# ---------------------------------------------------------------------------
# CBAM: Channel & Spatial Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """CBAM Channel Attention."""

    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            NonLinearity(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out * x


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=padding, padding_mode='reflect', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        return out * x


# ---------------------------------------------------------------------------
# Self-Attention & UNet Bottleneck
# ---------------------------------------------------------------------------

class SelfAttention2D(nn.Module):
    """Self-attention Layer."""

    def __init__(self, in_chans, down_factor=8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_chans, in_chans // down_factor, kernel_size=1)
        self.key_conv = nn.Conv2d(in_chans, in_chans // down_factor, kernel_size=1)
        self.value_conv = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = rearrange(self.query_conv(x), 'b c h w -> b (h w) c')
        k_T = rearrange(self.key_conv(x), 'b c h w -> b c (h w)')
        v = rearrange(self.value_conv(x), 'b c h w -> b (h w) c')
        energy = torch.matmul(q, k_T)
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return self.gamma * out


class UNetBottleneck(nn.Module):
    def __init__(self, in_chans, out_chans, conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d, attn_type=SelfAttention2D):
        super().__init__()
        self.attn = PreNorm(in_chans, attn_type(in_chans))
        self.ffn = PreNorm(in_chans, nn.Sequential(
            conv_type(in_chans, out_chans, kernel_size=1),
            NonLinearity(True),
            conv_type(in_chans, out_chans, kernel_size=1),
        ))
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.attn(x) * self.scale + x
        x = self.ffn(x) * self.scale + x
        return x


# ---------------------------------------------------------------------------
# Top-K Selective Attention (DRSFormer)
# ---------------------------------------------------------------------------

class TopkAttention(nn.Module):
    """Top-K Selective Attention.

    Reference: DRSFormer (https://github.com/cschenxiang/DRSformer)
    """

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out = (attn1 @ v) * self.attn1 + (attn2 @ v) * self.attn2 + \
              (attn3 @ v) * self.attn3 + (attn4 @ v) * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TopkAttnBlock(nn.Module):
    def __init__(self, feats, scale=0.01, mlp_ratio=2):
        super().__init__()
        self.attn = PreNorm(feats, TopkAttention(feats, num_heads=8, bias=True))
        self.ffn = PreNorm(feats, nn.Sequential(
            nn.Conv2d(feats, feats * mlp_ratio, kernel_size=1),
            NonLinearity(inplace=True),
            nn.Conv2d(feats * mlp_ratio, feats, kernel_size=1),
        ))
        self.scale = scale

    def forward(self, x):
        res = x
        x = self.attn(x) * self.scale + x
        x = self.ffn(x) * self.scale + x
        return x + res


# ---------------------------------------------------------------------------
# Multi-DConv Head Transposed Self-Attention (MDTA)
# ---------------------------------------------------------------------------

class TransposedAttention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)."""

    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1,
                              bias=bias, padding_mode='reflect')
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                     padding=1, groups=dim * 3, bias=bias,
                                     padding_mode='reflect')
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


# ---------------------------------------------------------------------------
# Gated-DConv Feed-Forward Network (GDFN)
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features * 2, bias=bias,
            padding_mode='reflect')
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ---------------------------------------------------------------------------
# Restormer Block
# ---------------------------------------------------------------------------

class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2., bias=False,
                 LayerNorm_type='WithBia', scale=0.01, **kwargs):
        super().__init__()
        self.norm1 = RestormerLayerNorm(dim, LayerNorm_type)
        self.attn = TransposedAttention(dim, num_heads, bias)
        self.norm2 = RestormerLayerNorm(dim, LayerNorm_type)
        self.ffn = GatedFFN(dim, ffn_expansion_factor, bias)
        self.scale_1 = nn.Parameter(scale * torch.ones(1))
        self.scale_2 = nn.Parameter(scale * torch.ones(1))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.scale_1
        x = x + self.ffn(self.norm2(x)) * self.scale_2
        return x


# ---------------------------------------------------------------------------
# Dilate Attention & Multi-Dilate Window Attention (MDWA)
# ---------------------------------------------------------------------------

class DilateAttention(nn.Module):
    """Implementation of Dilate-attention."""

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Sequential(
            nn.ReflectionPad2d(dilation * (kernel_size - 1) // 2),
            nn.Unfold(kernel_size, dilation)
        )
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else nn.Identity()

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        n_heads = d // self.head_dim

        q = rearrange(
            q, 'b (nHeads headDim) h w -> b nHeads (h w) 1 headDim',
            nHeads=n_heads, headDim=self.head_dim
        )
        k = self.unfold(k)
        k = rearrange(
            k, 'b (nHeads headDim ksq) hw -> b nHeads hw headDim ksq',
            nHeads=n_heads, headDim=self.head_dim, ksq=self.kernel_size ** 2
        )
        v = self.unfold(v)
        v = rearrange(
            v, 'b (nHeads headDim ksq) hw -> b nHeads hw ksq headDim',
            nHeads=n_heads, headDim=self.head_dim, ksq=self.kernel_size ** 2
        )
        x = (q @ k) * self.scale
        x = x.softmax(dim=-1)
        x = self.attn_drop(x)
        x = x @ v
        x = rearrange(
            x, 'b nHeads (h w) 1 headDim -> b h w (nHeads headDim)', h=H, w=W)
        return x


class MultiDilateWindowAttention(nn.Module):
    """Multi-Dilate Window Attention (MDWA)."""

    def __init__(self, dim, num_heads=8, bias=False, qk_scale=None, pos_embed=True,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, \
            f"num_heads {num_heads} must be divisible by num_dilation {self.num_dilation}"
        self.pos_embed = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim) if pos_embed else None
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.pos_embed:
            x = x + self.pos_embed(x)
        qkv = self.qkv(x.clone())
        qkv = rearrange(
            qkv,
            'b (three nd c2) h w -> nd three b c2 h w',
            three=3,
            nd=self.num_dilation,
            c2=C // self.num_dilation
        )
        x = torch.stack([self.dilate_attention[i](
            qkv[i, 0], qkv[i, 1], qkv[i, 2]) for i in range(self.num_dilation)])
        x = rearrange(x, 'nd b h w c2 -> b (nd c2) h w')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# ECRformer Block
# ---------------------------------------------------------------------------

class ECRformerBlock(nn.Module):
    """ECRformer Block: TransposedAttention + MultiDilateWindowAttention + FFN."""

    def __init__(self, dim: int, kernel_size: int = 3, dilation: list[int] = [1, ],
                 drop_path_rate=0., bias=False, conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d, **kwargs):
        super().__init__()
        if not isinstance(dilation, list):
            dilation = [dilation]

        num_heads = min(4, dim // 48)
        if dim % 3 != 0:
            num_heads = min(4, dim // 32)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        attn_norm_type = functools.partial(RestormerLayerNorm, LayerNorm_type='WithBia')

        self.attn_1 = nn.Sequential(
            attn_norm_type(dim),
            TransposedAttention(dim, num_heads, bias),
        )

        # Compute MDWA dilation and head count from dim
        dilation = [1, 2, 3, 4] if dim > 48 else [1, 2]
        cell_dim = 24
        if dim % 3 != 0:
            dilation = [1, 2, 3, 4] if dim > 32 else [1, 2]
            cell_dim = 16
        num_heads = dim // cell_dim

        self.attn_2 = nn.Sequential(
            attn_norm_type(dim),
            MultiDilateWindowAttention(dim, num_heads=num_heads, bias=bias,
                                       kernel_size=kernel_size, dilation=dilation),
        )

        self.ffn_2 = nn.Sequential(
            attn_norm_type(dim),
            GatedFFN(dim=dim, bias=bias),
        )

        scale = 0.01
        self.scale_1 = nn.Parameter(scale * torch.ones(1))
        self.scale_2 = nn.Parameter(scale * torch.ones(1))
        self.scale_3 = nn.Parameter(scale * torch.ones(1))

    def forward(self, x):
        x = self.drop_path(self.attn_1(x)) * self.scale_1 + x
        x = self.drop_path(self.attn_2(x)) * self.scale_2 + x
        x = self.drop_path(self.ffn_2(x)) * self.scale_3 + x
        return x
