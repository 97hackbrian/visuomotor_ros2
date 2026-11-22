"""Aumentos de imagen en GPU, vectorizados por muestra.

``torchvision.transforms.ColorJitter`` en los workers de CPU reducía el DataLoader de ~35 a ~3.5 lotes/s (64 x 2
imágenes 256x256): se aplica aquí, en GPU, sobre el lote ya transferido. Los mismos parámetros se usan para todo el
historial de observación de una muestra (coherencia temporal), y distintos entre muestras.

Rangos UMI (Chi et al. 2024, App. E.4): brillo 0.3, contraste 0.4, saturación 0.5, tono 0.08.
El tono se rota en el plano de crominancia YIQ (aproximación estándar y diferenciable de la rotación HSV).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _u(n: int, lo: float, hi: float, device, dtype):
    return torch.empty(n, device=device, dtype=dtype).uniform_(lo, hi)


@torch.no_grad()
def color_jitter_gpu(imgs: torch.Tensor, brightness: float = 0.3, contrast: float = 0.4, saturation: float = 0.5,
                     hue: float = 0.08) -> torch.Tensor:
    """``imgs [B, T, 3, H, W]`` (o ``[B, 3, H, W]``) en [0, 1] -> imágenes aumentadas en [0, 1]."""
    squeeze = imgs.dim() == 4
    x = imgs[:, None] if squeeze else imgs
    b = x.shape[0]
    dev, dt = x.device, x.dtype
    shape = (b, 1, 1, 1, 1)
    if brightness > 0:
        x = x * _u(b, max(0.0, 1 - brightness), 1 + brightness, dev, dt).view(shape)
    gray_w = torch.tensor([0.299, 0.587, 0.114], device=dev, dtype=dt).view(1, 1, 3, 1, 1)
    if contrast > 0:
        mean = (x * gray_w).sum(2, keepdim=True).mean(dim=(1, 3, 4), keepdim=True)
        c = _u(b, max(0.0, 1 - contrast), 1 + contrast, dev, dt).view(shape)
        x = (x - mean) * c + mean
    if saturation > 0:
        gray = (x * gray_w).sum(2, keepdim=True)
        s = _u(b, max(0.0, 1 - saturation), 1 + saturation, dev, dt).view(shape)
        x = gray + (x - gray) * s
    if hue > 0:
        theta = _u(b, -hue, hue, dev, dt) * 2.0 * math.pi
        cos, sin = torch.cos(theta), torch.sin(theta)
        rot = torch.zeros(b, 3, 3, device=dev, dtype=dt)
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1], rot[:, 1, 2] = cos, -sin
        rot[:, 2, 1], rot[:, 2, 2] = sin, cos
        m = _YIQ2RGB.to(dev, dt) @ rot @ _RGB2YIQ.to(dev, dt)          # [B, 3, 3]
        x = torch.einsum("bij,btjhw->btihw", m, x)
    x = x.clamp(0.0, 1.0)
    return x[:, 0] if squeeze else x


def make_gpu_augment(cfg: Dict | None):
    if not cfg:
        return None
    return lambda imgs: color_jitter_gpu(imgs, **cfg)
