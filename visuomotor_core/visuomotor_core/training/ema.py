"""Media móvil exponencial (EMA) de los pesos, como en Diffusion Policy / UMI (``diffusers.EMAModel``).

decay(step) = clamp(1 - (1 + step / inv_gamma) ** -power, min_value, max_value)
Con power = 0.75: 0.998 a 4k pasos, 0.9997 a 42k y el tope 0.9999 hacia ~215k (Chi et al. 2023; UMI 2024).
GroupNorm es requisito para que la EMA funcione (DP §3.2): el encoder del repo ya la usa.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel:
    def __init__(self, model: nn.Module, power: float = 0.75, inv_gamma: float = 1.0,
                 min_value: float = 0.0, max_value: float = 0.9999, update_after_step: int = 0):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.power = float(power)
        self.inv_gamma = float(inv_gamma)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.update_after_step = int(update_after_step)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, step: int) -> float:
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** -self.power
        return float(min(self.max_value, max(self.min_value, value)))

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)
        ema_params = [p for p in self.model.parameters()]
        new_params = [p.detach() for p in model.parameters()]
        if self.decay == 0.0:
            torch._foreach_copy_(ema_params, new_params)
        else:
            torch._foreach_lerp_(ema_params, new_params, 1.0 - self.decay)
        # buffers (estadísticas de normalización, rejillas de SpatialSoftmax): copia directa
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)
        self.optimization_step += 1
