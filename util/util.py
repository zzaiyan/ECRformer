"""Utility functions for model training and evaluation."""

import torch
import torch.nn as nn
import torch.nn.init as init
from .pytorch_ssim import ssim as calc_ssim
from typing import Literal


def count_parameters(model: torch.nn.Module, mode: Literal['return', 'print'] = 'print') -> int | None:
    """计算模型参数数量。"""
    counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if mode == 'return':
        return counts
    elif mode == 'print':
        units = ['', 'K', 'M', 'G']
        unit_idx = 0
        while counts > 1000:
            counts /= 1000
            unit_idx += 1
        print(f'Number of parameters: {counts:.4f}{units[unit_idx]}')
    else:
        raise ValueError('Invalid mode, must be "return" or "print"')


@torch.no_grad()
def compute_metric(pred, target, size_average=False, eps=1e-9):
    """Compute image quality metrics: RMSE, MAE, PSNR, SAM, SSIM."""
    rmse = torch.sqrt(torch.mean(torch.square(pred - target), dim=(1, 2, 3)))
    psnr = 20 * torch.log10(1 / (rmse + eps))
    mae = torch.mean(torch.abs(pred - target), dim=(1, 2, 3))

    # Spectral Angle Mapper
    mat = torch.mean(pred * target, 1) / (torch.sqrt(torch.mean(pred * pred, 1)) *
                                          torch.sqrt(torch.mean(target * target, 1)) + eps)
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1)),
                     dim=(1, 2)) * 180 / torch.pi

    ssim = calc_ssim(pred, target, size_average=False)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'PSNR': psnr,
        'SAM': sam,
        'SSIM': ssim,
    }
    post_fn = lambda x: x.mean() if size_average else x
    metrics = {k: post_fn(v.cpu()) for k, v in metrics.items()}

    return metrics


def initialize_weights(net_l, scale=1.):
    """Initialize network weights using Kaiming initialization."""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            params = list(m.parameters())
            if len(params) == 0:
                continue
            if all(not p.requires_grad for p in params):
                continue
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
