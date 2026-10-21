"""
DPPOPolicyAdapter — Bridge entre DiffusionPolicy (Chi) y DPPO (ReinFlow / Ren et al.).

Este módulo adapta la arquitectura DiffusionPolicy (1D U-Net + ResNet-18 GroupNorm + SpatialSoftmax)
para su optimización mediante Proximal Policy Optimization sobre el MDP de denoising,
manteniendo CERO dependencias de ROS (Off-ROS / Python Puro).

Principios Matemáticos:
- Inner MDP sobre las transiciones de denoising reversas x_k -> x_{k-1}.
- Suelo de varianza de exploración: min_sampling_denoising_std y min_logprob_denoising_std.
- Clipping PPO dinámico exponencial a lo largo del proceso de difusión.
- Descuento temporal interno gamma_denoising.
- Canonicidad estricta de cuaterniones (q_w >= 0).
- Regularización elástica de Behavior Cloning (BC) contra el modelo maestro congelado.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visuomotor_core.models.config import DiffusionConfig
from visuomotor_core.models.diffusion_policy import DiffusionPolicy, DiffusionModel


@dataclass
class DPPOConfig:
    """Hiperparámetros matemáticos para el entrenamiento DPPO."""
    # --- Denoising MDP ---
    denoising_steps: int = 10              # Pasos totales DDIM en inferencia
    ft_denoising_steps: int = 5            # Pasos finales a optimizar con gradiente PPO
    gamma_denoising: float = 0.99          # Descuento a través de los pasos de difusión
    
    # --- PPO Clipping Schedule ---
    clip_ploss_coef: float = 0.01          # epsilon final en k = ft_denoising_steps
    clip_ploss_coef_base: float = 0.001    # epsilon base en k = 0
    clip_ploss_coef_rate: float = 3.0      # Tasa exponencial de crecimiento
    clip_vloss_coef: Optional[float] = 0.5 # Clipping de la función de valor
    vf_coef: float = 0.5                   # Peso del loss del crítico
    
    # --- Regularización Anti-Catastrophic Forgetting ---
    use_bc_loss: bool = True               # Activa regularización hacia la política supervisada
    bc_loss_coef: float = 0.02             # Coeficiente lambda_BC
    
    # --- Exploración y Suelos de Varianza ---
    eta: float = 0.5                       # Estocasticidad DDIM (0 = determinista, 1 = DDPM)
    min_sampling_denoising_std: float = 0.08  # Suelo de desviación estándar en rollout
    min_logprob_denoising_std: float = 0.08   # Suelo de desviación en evaluación de log-probabilidad
    
    # --- Action Horizon Slicing ---
    act_steps: int = 3                     # Cantidad de acciones ejecutadas en el entorno
    
    # --- GAE & Ventajas ---
    gamma: float = 0.99                    # Descuento temporal del entorno
    gae_lambda: float = 0.95              # Factor lambda para GAE
    clip_advantage_lower_quantile: float = 0.05
    clip_advantage_upper_quantile: float = 0.95


class CriticObs(nn.Module):
    """
    Función de valor V(s) basada exclusivamente en el vector propioceptivo.
    Entrada: cond["state"] de dimensión [B, T_o * D_o].
    """
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256, 256], activation: str = "Mish"):
        super().__init__()
        layers = []
        in_dim = obs_dim
        act_cls = getattr(nn, activation)
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_cls()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Inicialización de sesgo optimista leve para evitar colapso inicial
        nn.init.constant_(self.network[-1].bias, 1.0)

    def forward(self, cond: Dict[str, Tensor]) -> Tensor:
        state = cond["state"]
        if state.dim() == 3:
            state = state.flatten(1)
        return self.network(state).squeeze(-1)


class ViTCriticAdapter(nn.Module):
    """
    Función de valor V(s_rgb, s_state) multimodal.
    Reutiliza el rgb_encoder preentrenado (ResNet-18 GroupNorm + SpatialSoftmax)
    congelado, garantizando que el espacio latente visual sea estacionario.
    """
    def __init__(self, rgb_encoder: nn.Module, state_dim: int, n_obs_steps: int,
                 hidden_dims: List[int] = [256, 256, 256]):
        super().__init__()
        # Clonar el encoder visual y congelar sus parámetros
        self.rgb_encoder = copy.deepcopy(rgb_encoder)
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        self.rgb_encoder.eval()

        self.n_obs_steps = n_obs_steps
        visual_dim = self.rgb_encoder.feature_dim * n_obs_steps
        total_in_dim = visual_dim + (state_dim * n_obs_steps)

        layers = []
        in_dim = total_in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Mish()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.constant_(self.mlp[-1].bias, 1.0)

    def forward(self, cond: Dict[str, Tensor]) -> Tensor:
        # rgb shape: [B, T_o, C, H, W]
        rgb = cond["rgb"]
        b, t_o, c, h, w = rgb.shape
        rgb_flat = rgb.view(b * t_o, c, h, w)

        with torch.no_grad():
            features = self.rgb_encoder(rgb_flat)  # [B * T_o, feature_dim]
        features = features.view(b, t_o * self.rgb_encoder.feature_dim)

        state = cond["state"]
        if state.dim() == 3:
            state = state.flatten(1)

        x = torch.cat([features, state], dim=-1)
        return self.mlp(x).squeeze(-1)


class DPPOPolicyAdapter(nn.Module):
    """
    Adapter que orquesta el Actor Preentrenado (Frozen), el Actor Trainable (Fine-Tuned)
    y el Crítico para el algoritmo DPPO.
    """
    def __init__(self, pretrained_path: Union[str, Path], dppo_config: DPPOConfig,
                 critic_type: str = "vit", device: str = "cuda"):
        super().__init__()
        self.cfg = dppo_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. Cargar política supervisada maestra
        self.base_policy = DiffusionPolicy.from_pretrained(Path(pretrained_path))
        self.base_policy.eval()

        # Extraer configuraciones y módulos de normalización originales
        self.policy_config = self.base_policy.config
        self.normalize_inputs = self.base_policy.normalize_inputs
        self.normalize_targets = self.base_policy.normalize_targets
        self.unnormalize_outputs = self.base_policy.unnormalize_outputs

        # 2. Configurar Actor Maestro (Frozen) y Actor Fine-Tuned (Trainable)
        self.actor = self.base_policy.diffusion
        for p in self.actor.parameters():
            p.requires_grad = False
        self.actor.eval()

        self.actor_ft = copy.deepcopy(self.actor)
        for p in self.actor_ft.parameters():
            p.requires_grad = True

        # Congelar el backbone visual en actor_ft si se desea evitar sobreajuste
        for p in self.actor_ft.rgb_encoder.parameters():
            p.requires_grad = False
        self.actor_ft.rgb_encoder.eval()

        # 3. Inicializar Crítico
        state_dim = self.policy_config.input_shapes["observation.state"][0]
        n_obs = self.policy_config.n_obs_steps
        if critic_type == "vit":
            self.critic = ViTCriticAdapter(
                rgb_encoder=self.actor.rgb_encoder,
                state_dim=state_dim,
                n_obs_steps=n_obs
            )
        else:
            self.critic = CriticObs(obs_dim=state_dim * n_obs)

        # 4. Parámetros de Difusión y Schedules
        self.num_train_timesteps = self.policy_config.num_train_timesteps
        self.denoising_steps = self.cfg.denoising_steps
        self.ft_denoising_steps = self.cfg.ft_denoising_steps
        self.horizon = self.policy_config.horizon
        self.action_dim = self.policy_config.output_shapes["action"][0]

        # Configurar timesteps del scheduler DDIM
        self.actor.noise_scheduler.set_timesteps(self.denoising_steps)
        self.actor_ft.noise_scheduler.set_timesteps(self.denoising_steps)
        self.timesteps = self.actor.noise_scheduler.timesteps.to(self.device)

        self.to(self.device)

    def _canonicalize_quaternions(self, action_tensor: Tensor) -> Tensor:
        """Asegura que q_w >= 0 en las dimensiones de rotación de la acción [3:7]."""
        if action_tensor.shape[-1] >= 7:
            qw = action_tensor[..., 6:7]
            sign = torch.where(qw < 0.0, -1.0, 1.0)
            quat = action_tensor[..., 3:7] * sign
            action_tensor = torch.cat([action_tensor[..., :3], quat, action_tensor[..., 7:]], dim=-1)
        return action_tensor

    def _build_global_cond(self, model: DiffusionModel, batch: Dict[str, Tensor]) -> Tensor:
        """Codifica imagen + estado para formar el vector global_cond de la U-Net."""
        img = batch["observation.image"]
        state = batch["observation.state"]
        b, t_o = img.shape[:2]
        img_flat = img.view(b * t_o, *img.shape[2:])
        features = model.rgb_encoder(img_flat)
        features = features.view(b, -1)
        state_flat = state.view(b, -1)
        return torch.cat([features, state_flat], dim=-1)

    def sample_actions_with_chain(self, batch: Dict[str, Tensor], deterministic: bool = False
                                 ) -> Dict[str, Tensor]:
        """
        Ejecuta el proceso de denoising en reversa registrando la cadena completa
        x_{tau_K} -> x_{tau_0} y calculando log-probabilidades para el rollout.
        """
        norm_batch = self.normalize_inputs(batch)
        b = norm_batch["observation.state"].shape[0]

        with torch.no_grad():
            cond_frozen = self._build_global_cond(self.actor, norm_batch)
            cond_ft = self._build_global_cond(self.actor_ft, norm_batch)

            # Iniciar con ruido Gaussiano puro N(0, I)
            sample = torch.randn(size=(b, self.horizon, self.action_dim), device=self.device)
            chains = [sample.clone()]
            logprobs_list = []

            total_steps = len(self.timesteps)
            split_idx = total_steps - self.ft_denoising_steps

            for i, t in enumerate(self.timesteps):
                # Usar actor maestro para pasos iniciales, actor_ft para pasos fine-tuned
                is_ft_step = (i >= split_idx)
                current_unet = self.actor_ft.unet if is_ft_step else self.actor.unet
                current_cond = cond_ft if is_ft_step else cond_frozen

                t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                t_tensor = torch.full((b,), t_val, dtype=torch.long, device=sample.device)
                model_output = current_unet(sample, t_tensor, global_cond=current_cond)

                # Calcular media predicha limpia bajo DDIM (eta=0.0)
                step_res = self.actor.noise_scheduler.step(
                    model_output=model_output,
                    timestep=t_val,
                    sample=sample,
                    eta=0.0
                )
                pred_mean = step_res.prev_sample

                # Varianza analítica DDIM
                step_size = self.actor.noise_scheduler.config.num_train_timesteps // self.actor.noise_scheduler.num_inference_steps
                prev_t = t_val - step_size
                var_val = float(self.actor.noise_scheduler._get_variance(t_val, prev_t))
                std_val = self.cfg.eta * (var_val ** 0.5)

                if is_ft_step and not deterministic and self.cfg.eta > 0.0:
                    std_eff_scalar = max(std_val, self.cfg.min_sampling_denoising_std)
                    std_eff = torch.full_like(pred_mean, std_eff_scalar)
                    # Amortiguar fuertemente el ruido en las dimensiones de rotación [3:7]
                    # para preservar la orientación aprendida y evitar que el efector se tuerza
                    std_eff[:, :, 3:7] *= 0.05
                    noise = torch.randn_like(pred_mean)
                    next_sample = pred_mean + std_eff * noise
                else:
                    std_eff = torch.full_like(pred_mean, 1e-4)
                    next_sample = pred_mean

                if is_ft_step:
                    min_lp_std = torch.full_like(std_eff, self.cfg.min_logprob_denoising_std)
                    min_lp_std[:, :, 3:7] *= 0.05
                    std_lp = torch.maximum(std_eff, min_lp_std)
                    var = std_lp ** 2
                    logprob = -0.5 * (
                        ((next_sample - pred_mean) ** 2) / var
                        + 2.0 * torch.log(std_lp)
                        + math.log(2.0 * math.pi)
                    )
                    logprobs_list.append(logprob)

                sample = next_sample
                chains.append(sample.clone())

            # Desnormalizar acciones predichas finales
            final_actions_norm = chains[-1]
            final_actions_real = self.unnormalize_outputs({"action": final_actions_norm})["action"]

            # Cortar al horizonte de ejecución (act_steps) alineado a partir del paso t actual (n_obs_steps - 1)
            act_steps = self.cfg.act_steps
            start = self.policy_config.n_obs_steps - 1
            executed_actions = final_actions_real[:, start : start + act_steps]

            # Apilar cadenas de los pasos fine-tuned: shape [B, ft_denoising_steps + 1, H, D]
            ft_chains = torch.stack(chains[split_idx:], dim=1)
            ft_logprobs = torch.stack(logprobs_list, dim=1) if logprobs_list else torch.empty(0)

            return {
                "actions": executed_actions,               # [B, act_steps, D_a]
                "actions_full": final_actions_real,       # [B, H, D_a]
                "chains": ft_chains,                      # [B, ft_steps+1, H, D_a]
                "logprobs": ft_logprobs,                  # [B, ft_steps, H, D_a]
                "final_actions_norm": final_actions_norm, # [B, H, D_a]
            }

    def get_logprobs_subsample(self, obs: Dict[str, Tensor], chains_prev: Tensor,
                               chains_next: Tensor, denoising_inds: Tensor) -> Tensor:
        """
        Calcula la log-probabilidad de la transición (chains_prev -> chains_next)
        bajo el actor fine-tuned para un batch de entrenamiento.
        """
        norm_obs = self.normalize_inputs(obs)
        cond = self._build_global_cond(self.actor_ft, norm_obs)

        # Mapear denoising_inds a los timesteps reales correspondientes
        split_idx = len(self.timesteps) - self.ft_denoising_steps
        actual_timesteps = self.timesteps[split_idx + denoising_inds]

        model_output = self.actor_ft.unet(chains_prev, actual_timesteps, global_cond=cond)

        # Calcular media predicha del scheduler agrupando por pasos únicos en el batch
        pred_mean = torch.zeros_like(chains_prev)
        std_eff = torch.zeros_like(chains_prev)
        step_size = self.actor_ft.noise_scheduler.config.num_train_timesteps // self.actor_ft.noise_scheduler.num_inference_steps

        for d_idx in denoising_inds.unique():
            mask = (denoising_inds == d_idx)
            sub_output = model_output[mask]
            sub_sample = chains_prev[mask]
            sub_t = int(self.timesteps[split_idx + d_idx].item())

            step_res = self.actor_ft.noise_scheduler.step(
                model_output=sub_output,
                timestep=sub_t,
                sample=sub_sample,
                eta=0.0
            )
            pred_mean[mask] = step_res.prev_sample

            prev_t = sub_t - step_size
            var_val = float(self.actor_ft.noise_scheduler._get_variance(sub_t, prev_t))
            std_val = self.cfg.eta * (var_val ** 0.5)
            std_scalar = max(std_val, self.cfg.min_logprob_denoising_std)
            std_eff[mask] = std_scalar

        # Escalar rotación para coincidir exactamente con el sampling
        std_eff[:, :, 3:7] *= 0.05
        var = std_eff ** 2

        logprob = -0.5 * (
            ((chains_next - pred_mean) ** 2) / var
            + 2.0 * torch.log(std_eff)
            + math.log(2.0 * math.pi)
        )
        return logprob

    def compute_dppo_loss(self, obs: Dict[str, Tensor], chains_prev: Tensor, chains_next: Tensor,
                          denoising_inds: Tensor, returns: Tensor, old_values: Tensor,
                          advantages: Tensor, old_logprobs: Tensor, critic_only: bool = False) -> Dict[str, Tensor]:
        """
        Calcula el loss PPO recortado adaptativamente por paso de difusión,
        el loss del crítico y la regularización opcional de Behavior Cloning.
        Si critic_only=True, omite el forward pass del actor y entrena exclusivamente el crítico.
        """
        # 1. Critic Value Loss
        cond_critic = {
            "state": self.normalize_inputs(obs)["observation.state"],
            "rgb": self.normalize_inputs(obs)["observation.image"]
        }
        values = self.critic(cond_critic)
        if self.cfg.clip_vloss_coef is not None:
            v_clipped = old_values + torch.clamp(values - old_values, -self.cfg.clip_vloss_coef, self.cfg.clip_vloss_coef)
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (v_clipped - returns) ** 2
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            critic_loss = 0.5 * F.mse_loss(values, returns)

        if critic_only:
            zero_t = torch.tensor(0.0, device=self.device)
            return {
                "total_loss": self.cfg.vf_coef * critic_loss,
                "policy_loss": zero_t,
                "critic_loss": critic_loss,
                "bc_loss": zero_t,
                "approx_kl": zero_t,
                "clip_fraction": zero_t,
            }

        # 2. Nuevas log-probabilidades bajo actor_ft
        new_logprobs = self.get_logprobs_subsample(obs, chains_prev, chains_next, denoising_inds)

        # Truncar al horizonte efectivamente ejecutado (start:start+act_steps)
        act_steps = self.cfg.act_steps
        start = self.policy_config.n_obs_steps - 1
        end = start + act_steps
        new_lp_sliced = new_logprobs[:, start:end, :].mean(dim=(-1, -2))
        old_lp_sliced = old_logprobs[:, start:end, :].mean(dim=(-1, -2))

        # Estabilidad numérica: clamping de log-probs
        new_lp_sliced = torch.clamp(new_lp_sliced, min=-5.0, max=2.0)
        old_lp_sliced = torch.clamp(old_lp_sliced, min=-5.0, max=2.0)

        # 3. Ratio de PPO
        ratio = torch.exp(new_lp_sliced - old_lp_sliced)

        # 4. Descuento temporal a través de los pasos de denoising
        steps_from_end = self.ft_denoising_steps - 1 - denoising_inds
        denoise_discount = (self.cfg.gamma_denoising ** steps_from_end).unsqueeze(-1)
        discounted_advantages = advantages * denoise_discount

        # 5. Clipping dinámico exponencial de epsilon(k)
        t_norm = denoising_inds.float() / max(self.ft_denoising_steps - 1, 1)
        rate = self.cfg.clip_ploss_coef_rate
        exp_factor = (torch.exp(rate * t_norm) - 1.0) / (math.exp(rate) - 1.0)
        clip_coef = self.cfg.clip_ploss_coef_base + (self.cfg.clip_ploss_coef - self.cfg.clip_ploss_coef_base) * exp_factor
        clip_coef = clip_coef.unsqueeze(-1)

        # 6. PPO Clipped Surrogate Loss
        surr1 = ratio * discounted_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * discounted_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 7. Behavior Cloning Regularization (Anti-Catastrophic Forgetting)
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.use_bc_loss:
            split_idx = len(self.timesteps) - self.ft_denoising_steps
            actual_timesteps = self.timesteps[split_idx + denoising_inds]
            with torch.no_grad():
                cond_teacher = self._build_global_cond(self.actor, self.normalize_inputs(obs))
                teacher_output = self.actor.unet(chains_prev, actual_timesteps, global_cond=cond_teacher)
            cond_student = self._build_global_cond(self.actor_ft, self.normalize_inputs(obs))
            student_output = self.actor_ft.unet(chains_prev, actual_timesteps, global_cond=cond_student)
            bc_loss = F.mse_loss(student_output, teacher_output)

        total_loss = policy_loss + self.cfg.vf_coef * critic_loss + self.cfg.bc_loss_coef * bc_loss

        # Métricas de diagnóstico
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "bc_loss": bc_loss,
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
        }

    def export_pretrained(self, output_path: Union[str, Path]):
        """
        Exporta los pesos de actor_ft en formato completamente compatible con
        DiffusionPolicy.from_pretrained(), permitiendo su uso inmediato en inference.launch.py.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Crear instancia limpia para exportación
        export_policy = DiffusionPolicy(config=self.policy_config, dataset_stats=None)
        export_policy.diffusion.load_state_dict(self.actor_ft.state_dict())

        # Preservar buffers de normalización originales
        export_policy.normalize_inputs.load_state_dict(self.normalize_inputs.state_dict())
        export_policy.normalize_targets.load_state_dict(self.normalize_targets.state_dict())
        export_policy.unnormalize_outputs.load_state_dict(self.unnormalize_outputs.state_dict())

        export_policy.save_pretrained(output_path)
