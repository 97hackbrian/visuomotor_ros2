"""
PPODiffusionImgBuffer — Replay Buffer especializado para DPPO con Observaciones Multimodales.

Almacena rollouts de interacción en simulación y organiza los datos para el MDP de denoising:
- Almacena cadenas de difusión intermedias (chains) y log-probabilidades.
- Calcula retornos y ventajas mediante GAE(gamma_macro, lambda).
- Normaliza ventajas y aplica quantile clipping.
- Genera minibatches combinando pasos de entorno y pasos de difusión mediante índices desenrollados.
- Libre de dependencias de ROS (Python Puro / PyTorch).
"""

from typing import Dict, Generator, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


class RunningRewardScaler:
    """
    Escalador adaptativo de recompensas basado en la desviación estándar
    móvil de los retornos empíricos descontados (similar a ReinFlow).
    """
    def __init__(self, gamma: float = 0.99, eps: float = 1e-4):
        self.gamma = gamma
        self.eps = eps
        self.running_r = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        scaled_rewards = np.zeros_like(rewards)
        for i, (r, d) in enumerate(zip(rewards, dones)):
            self.running_r = self.running_r * self.gamma + r
            self.count += 1
            delta = self.running_r - 0.0
            self.var += (delta ** 2 - self.var) / self.count
            std = np.sqrt(self.var) + self.eps
            scaled_rewards[i] = r / std
            if d:
                self.running_r = 0.0
        return scaled_rewards


class PPODiffusionImgBuffer:
    """
    Buffer para almacenar rollouts y estructurar minibatches del MDP de difusión.
    """
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        image_shape: Tuple[int, int, int],  # (C, H, W)
        n_obs_steps: int,
        action_dim: int,
        horizon: int,
        ft_denoising_steps: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_advantage_lower_quantile: float = 0.05,
        clip_advantage_upper_quantile: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.image_shape = image_shape
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.horizon = horizon
        self.ft_denoising_steps = ft_denoising_steps
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.q_low = clip_advantage_lower_quantile
        self.q_high = clip_advantage_upper_quantile

        self.reward_scaler = RunningRewardScaler(gamma=gamma)
        self.reset()

    def reset(self):
        """Reinicia los buffers para una nueva iteración de recolección."""
        self.ptr = 0
        self.full = False

        # Observaciones sensoriales
        self.obs_state = torch.zeros((self.buffer_size, self.n_obs_steps, self.obs_dim), dtype=torch.float32)
        c, h, w = self.image_shape
        self.obs_image = torch.zeros((self.buffer_size, self.n_obs_steps, c, h, w), dtype=torch.float32)

        # Cadenas de difusión y probabilidades
        self.chains = torch.zeros(
            (self.buffer_size, self.ft_denoising_steps + 1, self.horizon, self.action_dim),
            dtype=torch.float32
        )
        self.logprobs = torch.zeros(
            (self.buffer_size, self.ft_denoising_steps, self.horizon, self.action_dim),
            dtype=torch.float32
        )

        # Señales de RL
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size,), dtype=torch.float32)

    def add(
        self,
        obs_state: Tensor,
        obs_image: Tensor,
        chains: Tensor,
        logprobs: Tensor,
        reward: float,
        value: float,
        done: bool,
    ):
        """Agrega un paso de interacción al buffer."""
        assert self.ptr < self.buffer_size, "Buffer overflow: intente vaciar antes de agregar más pasos."

        # Guardar tensores en CPU/RAM para no agotar la VRAM durante el rollout
        self.obs_state[self.ptr] = obs_state.detach().cpu().squeeze(0)
        self.obs_image[self.ptr] = obs_image.detach().cpu().squeeze(0)
        self.chains[self.ptr] = chains.detach().cpu().squeeze(0)
        self.logprobs[self.ptr] = logprobs.detach().cpu().squeeze(0)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float, gamma_macro: float):
        """
        Calcula las ventajas generalizadas (GAE) y los retornos de Bellman.
        gamma_macro representa gamma^(act_steps) para descontar macro-pasos.
        """
        n_steps = self.ptr
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma_macro * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma_macro * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]

    def get_sampler(
        self, batch_size: int, update_epochs: int
    ) -> Generator[Dict[str, Tensor], None, None]:
        """
        Generador que baraja el espacio aplanado [N_steps * ft_denoising_steps]
        y produce minibatches listos para la función de pérdida de DPPO.
        """
        n_steps = self.ptr
        total_samples = n_steps * self.ft_denoising_steps

        # 1. Normalización y Quantile Clipping de Ventajas
        advs = self.advantages[:n_steps]
        adv_mean = advs.mean()
        adv_std = advs.std() + 1e-8
        norm_advs = (advs - adv_mean) / adv_std

        if self.q_low is not None and self.q_high is not None:
            low_val = torch.quantile(norm_advs, self.q_low)
            high_val = torch.quantile(norm_advs, self.q_high)
            norm_advs = torch.clamp(norm_advs, low_val, high_val)

        # 2. Bucle de Épocas PPO
        for _ in range(update_epochs):
            permutation = torch.randperm(total_samples)

            for start_idx in range(0, total_samples, batch_size):
                batch_indices = permutation[start_idx : start_idx + batch_size]

                # Desenrollar índice plano a (step_idx, denoising_idx)
                step_inds = batch_indices // self.ft_denoising_steps
                denoise_inds = batch_indices % self.ft_denoising_steps

                # Empaquetar observación sensorial
                batch_obs = {
                    "observation.state": self.obs_state[step_inds].to(self.device, non_blocking=True),
                    "observation.image": self.obs_image[step_inds].to(self.device, non_blocking=True),
                }

                # Extraer par de transición en la cadena de difusión
                chains_step = self.chains[step_inds]  # [B, ft_steps+1, H, D]
                # Indexar para cada elemento su correspondiente denoise_inds
                b_range = torch.arange(len(step_inds))
                chains_prev = chains_step[b_range, denoise_inds].to(self.device, non_blocking=True)
                chains_next = chains_step[b_range, denoise_inds + 1].to(self.device, non_blocking=True)

                old_lps = self.logprobs[step_inds][b_range, denoise_inds].to(self.device, non_blocking=True)
                batch_returns = self.returns[step_inds].to(self.device, non_blocking=True)
                batch_values = self.values[step_inds].to(self.device, non_blocking=True)
                batch_advs = norm_advs[step_inds].to(self.device, non_blocking=True)
                batch_denoise_inds = denoise_inds.to(self.device, non_blocking=True)

                yield {
                    "obs": batch_obs,
                    "chains_prev": chains_prev,
                    "chains_next": chains_next,
                    "denoising_inds": batch_denoise_inds,
                    "returns": batch_returns,
                    "old_values": batch_values,
                    "advantages": batch_advs,
                    "old_logprobs": old_lps,
                }
