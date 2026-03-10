"""Utility modules and helper functions for ECRformer.

Contains LayerNorm variants, PreNorm wrapper, and miscellaneous helpers.
"""

import numbers
import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def split_integer(n, k, weights=None):
    """Split integer n into k parts, optionally with weights."""
    if weights is None:
        quotient, remainder = divmod(n, k)
        return [quotient + 1 if i < remainder else quotient for i in range(k)]
    else:
        total = sum(weights)
        parts = []
        fractions = []
        for w in weights:
            exact = n * w / total
            integer_part = int(exact)
            frac_part = exact - integer_part
            parts.append(integer_part)
            fractions.append(frac_part)
        remainder = n - sum(parts)
        frac_indices = sorted(range(k), key=lambda i: fractions[i], reverse=True)
        for i in range(remainder):
            parts[frac_indices[i]] += 1
        return parts


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ---------------------------------------------------------------------------
# Channel-wise LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Channel-wise LayerNorm."""
    def __init__(self, dim, *args, **kwargs):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


# ---------------------------------------------------------------------------
# Restormer-style LayerNorm
# Reference: Restormer (https://arxiv.org/abs/2111.09881)
# ---------------------------------------------------------------------------

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + eps) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + eps) * self.weight + self.bias


class RestormerLayerNorm(nn.Module):
    """Restormer LayerNorm with BiasFree / WithBias variants."""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PostFn(nn.Module):
    def __init__(self, fn, post_fn):
        super().__init__()
        self.fn = fn
        self.post_fn = post_fn

    def forward(self, *args, **kwargs):
        res = self.fn(*args, **kwargs)
        return self.post_fn(res)
