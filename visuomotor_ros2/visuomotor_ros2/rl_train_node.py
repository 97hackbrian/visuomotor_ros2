"""
RLTrainNode — Nodo de ROS 2 para Entrenamiento y Fine-Tuning de Diffusion Policy con DPPO.

Orquesta el ciclo de aprendizaje por refuerzo:
1. Rollouts síncronos en Isaac Sim mediante GymROS2Env.
2. Almacenamiento en PPODiffusionImgBuffer y cálculo de GAE.
3. Warmup inicial del Crítico (5-10 iteraciones) para evitar desestabilizar la U-Net.
4. Optimización conjunta Actor-Crítico con clipping adaptativo y early stopping por KL.
5. Exportación periódica de checkpoints listos para inference.launch.py.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import torch

from visuomotor_core.models.dppo_adapter import DPPOConfig, DPPOPolicyAdapter
from visuomotor_core.models.dppo_buffer import PPODiffusionImgBuffer
from visuomotor_ros2.gym_ros2_env import GymROS2Env


class RLTrainNode(Node):
    def __init__(self):
        super().__init__("rl_train_node")
        self.get_logger().info("Iniciando nodo de entrenamiento DPPO RL...")

        # ── 1. Declaración de Parámetros desde YAML ────────────────────────
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("output_dir", "/ros/visuomotor_core/outputs/dppo_finetuned")
        self.declare_parameter("save_every_iterations", 10)
        self.declare_parameter("critic_type", "vit")

        # DPPO e Hiperparámetros de Difusión
        self.declare_parameter("denoising_steps", 10)
        self.declare_parameter("ft_denoising_steps", 5)
        self.declare_parameter("gamma_denoising", 0.99)
        self.declare_parameter("actor_lr", 1.0e-5)
        self.declare_parameter("critic_lr", 5.0e-4)
        self.declare_parameter("actor_weight_decay", 1.0e-6)
        self.declare_parameter("critic_weight_decay", 1.0e-4)
        self.declare_parameter("n_iterations", 200)
        self.declare_parameter("n_steps_per_iteration", 60)
        self.declare_parameter("update_epochs", 8)
        self.declare_parameter("batch_size", 30)
        self.declare_parameter("target_kl", 0.05)
        self.declare_parameter("n_critic_warmup_itr", 5)
        self.declare_parameter("clip_ploss_coef", 0.01)
        self.declare_parameter("clip_ploss_coef_base", 0.001)
        self.declare_parameter("clip_ploss_coef_rate", 3.0)
        self.declare_parameter("clip_vloss_coef", 0.5)
        self.declare_parameter("vf_coef", 0.5)
        self.declare_parameter("use_bc_loss", True)
        self.declare_parameter("bc_loss_coef", 0.02)
        self.declare_parameter("min_sampling_denoising_std", 0.08)
        self.declare_parameter("min_logprob_denoising_std", 0.08)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("gae_lambda", 0.95)

        # Entorno de Simulación
        self.declare_parameter("max_episode_steps", 300)
        self.declare_parameter("step_timeout_s", 1.5)
        self.declare_parameter("reset_stabilize_s", 2.0)
        self.declare_parameter("reset_duration_s", 4.0)
        self.declare_parameter("n_obs_steps", 2)
        self.declare_parameter("n_action_steps", 3)

        # Geometría y Objetos (Heredados de data_collection_params.yaml)
        self.declare_parameter("object_frame", "pick_target")
        self.declare_parameter("ee_frame", "gripper_base_link")
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_offset_z", 0.079)
        self.declare_parameter("hover_z_offset", 0.15)
        self.declare_parameter("lift_required_z", 0.08)
        self.declare_parameter("target_x", 0.25)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.40)

        # Pseudo-Sensor de Fuerza
        self.declare_parameter("pseudo_force_clamped_threshold", 0.0025)
        self.declare_parameter("gripper_binary_closed_threshold", -0.005)

        # Pesos de Recompensa
        self.declare_parameter("reward_approach_weight", 1.0)
        self.declare_parameter("reward_dist_sigma", 0.05)
        self.declare_parameter("reward_orient_weight", 0.2)
        self.declare_parameter("reward_grasp_clamped_bonus", 1.0)
        self.declare_parameter("reward_grasp_lifted_bonus", 5.0)
        self.declare_parameter("reward_empty_grasp_penalty", -0.5)
        self.declare_parameter("reward_transport_weight", 2.0)
        self.declare_parameter("reward_done_success_bonus", 15.0)
        self.declare_parameter("reward_early_stop_bonus", 0.05)
        self.declare_parameter("reward_false_done_penalty", -5.0)
        self.declare_parameter("reward_accel_penalty", 0.005)

        # Topics y Hardware
        self.declare_parameter("camera_topic", "/rgb")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("target_frame_topic", "target_frame_raw")
        self.declare_parameter("gripper_command_topic", "/position_controller/commands")
        self.declare_parameter("respawn_topic", "/respawn")
        self.declare_parameter("image_size", 256)
        self.declare_parameter("gripper_open_pos", 0.0)
        self.declare_parameter("gripper_closed_pos", -0.01)
        self.declare_parameter("gripper_joint_name_pattern", "gripper")
        self.declare_parameter("home_x", 0.25)
        self.declare_parameter("home_y", 0.0)
        self.declare_parameter("home_z", 0.40)
        self.declare_parameter("home_roll", 3.14159)
        self.declare_parameter("home_pitch", 0.0)
        self.declare_parameter("home_yaw", 0.0)

        # ── 2. Cargar Diccionario de Configuración para el Entorno ─────────
        env_config = {
            param_name: self.get_parameter(param_name).value
            for param_name in [
                "obs_dim", "act_dim", "image_size", "n_obs_steps", "max_episode_steps",
                "step_timeout_s", "reset_stabilize_s", "reset_duration_s", "base_frame", "ee_frame",
                "object_frame", "tcp_offset_z", "hover_z_offset", "lift_required_z",
                "target_x", "target_y", "target_z", "gripper_open_pos", "gripper_closed_pos",
                "pseudo_force_clamped_threshold", "gripper_binary_closed_threshold",
                "gripper_joint_name_pattern", "camera_topic", "joint_states_topic",
                "target_frame_topic", "gripper_command_topic", "respawn_topic",
                "home_x", "home_y", "home_z", "home_roll", "home_pitch", "home_yaw",
                "reward_approach_weight", "reward_dist_sigma", "reward_orient_weight",
                "reward_grasp_clamped_bonus", "reward_grasp_lifted_bonus",
                "reward_empty_grasp_penalty", "reward_transport_weight",
                "reward_done_success_bonus", "reward_early_stop_bonus",
                "reward_false_done_penalty", "reward_accel_penalty"
            ]
            if self.has_parameter(param_name)
        }
        # Inyectar defaults si no se declararon
        env_config.setdefault("obs_dim", 8)
        env_config.setdefault("act_dim", 9)
        env_config.setdefault("image_size", 256)

        # ── 3. Inicializar Entorno Gymnasium ──────────────────────────────
        self.env = GymROS2Env(node=self, config=env_config)

        # ── 4. Inicializar DPPO Adapter y Modelo de Difusión ──────────────
        ckpt_path = self.get_parameter("checkpoint_path").value
        assert ckpt_path and Path(ckpt_path).exists(), f"checkpoint_path inválido o inexistente: {ckpt_path}"

        dppo_cfg = DPPOConfig(
            denoising_steps=self.get_parameter("denoising_steps").value,
            ft_denoising_steps=self.get_parameter("ft_denoising_steps").value,
            gamma_denoising=self.get_parameter("gamma_denoising").value,
            clip_ploss_coef=self.get_parameter("clip_ploss_coef").value,
            clip_ploss_coef_base=self.get_parameter("clip_ploss_coef_base").value,
            clip_ploss_coef_rate=self.get_parameter("clip_ploss_coef_rate").value,
            clip_vloss_coef=self.get_parameter("clip_vloss_coef").value,
            vf_coef=self.get_parameter("vf_coef").value,
            use_bc_loss=self.get_parameter("use_bc_loss").value,
            bc_loss_coef=self.get_parameter("bc_loss_coef").value,
            min_sampling_denoising_std=self.get_parameter("min_sampling_denoising_std").value,
            min_logprob_denoising_std=self.get_parameter("min_logprob_denoising_std").value,
            act_steps=self.get_parameter("n_action_steps").value,
            gamma=self.get_parameter("gamma").value,
            gae_lambda=self.get_parameter("gae_lambda").value,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dppo = DPPOPolicyAdapter(
            pretrained_path=ckpt_path,
            dppo_config=dppo_cfg,
            critic_type=self.get_parameter("critic_type").value,
            device=str(self.device),
        )
        self.get_logger().info(f"Modelo cargado exitosamente en dispositivo: {self.device}")

        # ── 5. Inicializar Optimizadores ──────────────────────────────────
        self.actor_optimizer = torch.optim.AdamW(
            self.dppo.actor_ft.parameters(),
            lr=self.get_parameter("actor_lr").value,
            weight_decay=self.get_parameter("actor_weight_decay").value,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.dppo.critic.parameters(),
            lr=self.get_parameter("critic_lr").value,
            weight_decay=self.get_parameter("critic_weight_decay").value,
        )

        # ── 6. Inicializar Replay Buffer ──────────────────────────────────
        n_steps = self.get_parameter("n_steps_per_iteration").value
        self.buffer = PPODiffusionImgBuffer(
            buffer_size=n_steps,
            obs_dim=env_config["obs_dim"],
            image_shape=(3, env_config["image_size"], env_config["image_size"]),
            n_obs_steps=self.get_parameter("n_obs_steps").value,
            action_dim=env_config["act_dim"],
            horizon=self.dppo.horizon,
            ft_denoising_steps=dppo_cfg.ft_denoising_steps,
            device=self.device,
            gamma=dppo_cfg.gamma,
            gae_lambda=dppo_cfg.gae_lambda,
        )

        # ── 7. Iniciar Bucle de Entrenamiento en Hilo Secundario ──────────
        self._training_thread = threading.Thread(target=self._run_training_loop, daemon=True)
        self._training_thread.start()

    def _run_training_loop(self):
        """Bucle principal de ejecución de DPPO."""
        time.sleep(2.0)  # Breve pausa para asegurar suscripciones de ROS 2
        self.get_logger().info("¡Arrancando bucle principal de entrenamiento DPPO!")

        n_iterations = self.get_parameter("n_iterations").value
        n_steps = self.get_parameter("n_steps_per_iteration").value
        n_warmup = self.get_parameter("n_critic_warmup_itr").value
        update_epochs = self.get_parameter("update_epochs").value
        batch_size = self.get_parameter("batch_size").value
        target_kl = self.get_parameter("target_kl").value
        output_dir = Path(self.get_parameter("output_dir").value)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_every = self.get_parameter("save_every_iterations").value

        best_mean_reward = -float("inf")
        act_steps = self.get_parameter("n_action_steps").value
        gamma_macro = self.get_parameter("gamma").value ** act_steps

        for itr in range(1, n_iterations + 1):
            iter_start_time = time.time()
            self.get_logger().info(f"\n==================== ITERACIÓN RL {itr}/{n_iterations} ====================")
            obs, _ = self.env.reset()
            self.buffer.reset()
            episode_rewards = []
            cur_ep_reward = 0.0

            # ── FASE 1: RECOLECCIÓN DE ROLLOUTS ────────────────────────────
            self.dppo.eval()
            for step_i in range(n_steps):
                # Preparar tensores para la red
                obs_tensors = {
                    "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(self.device),
                    "observation.image": torch.from_numpy(obs["observation.image"]).float().unsqueeze(0).to(self.device),
                }

                # Estimar valor con el crítico
                with torch.no_grad():
                    norm_obs = self.dppo.normalize_inputs(obs_tensors)
                    val = self.dppo.critic({
                        "state": norm_obs["observation.state"],
                        "rgb": norm_obs["observation.image"]
                    }).item()

                    # Muestrear acciones estocásticas y cadenas de difusión
                    sample_dict = self.dppo.sample_actions_with_chain(obs_tensors, deterministic=False)

                actions_to_exec = sample_dict["actions"].squeeze(0).cpu().numpy()  # [act_steps, D_a]
                chains = sample_dict["chains"]                                    # [1, ft_steps+1, H, D_a]
                logprobs = sample_dict["logprobs"]                                # [1, ft_steps, H, D_a]

                # Ejecutar el chunk de act_steps en el simulador
                step_accum_reward = 0.0
                done = False
                for a_sub in actions_to_exec:
                    next_obs, r, term, trunc, info = self.env.step(a_sub)
                    step_accum_reward += r
                    cur_ep_reward += r
                    if term or trunc:
                        done = True
                        episode_rewards.append(cur_ep_reward)
                        cur_ep_reward = 0.0
                        next_obs, _ = self.env.reset()
                        break

                # Guardar en el buffer
                self.buffer.add(
                    obs_state=obs_tensors["observation.state"],
                    obs_image=obs_tensors["observation.image"],
                    chains=chains,
                    logprobs=logprobs,
                    reward=step_accum_reward,
                    value=val,
                    done=done,
                )
                obs = next_obs

            # Último valor para GAE
            with torch.no_grad():
                last_obs_tensors = {
                    "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(self.device),
                    "observation.image": torch.from_numpy(obs["observation.image"]).float().unsqueeze(0).to(self.device),
                }
                norm_last = self.dppo.normalize_inputs(last_obs_tensors)
                last_val = self.dppo.critic({
                    "state": norm_last["observation.state"],
                    "rgb": norm_last["observation.image"]
                }).item()

            # ── FASE 2: CÁLCULO DE VENTAJAS (GAE) ──────────────────────────
            self.buffer.compute_gae(last_value=last_val, gamma_macro=gamma_macro)

            # ── FASE 3: OPTIMIZACIÓN PPO ───────────────────────────────────
            self.dppo.train()
            is_warmup = (itr <= n_warmup)
            if is_warmup:
                self.get_logger().info(f"[Warmup Crítico {itr}/{n_warmup}] Congelando U-Net; entrenando solo V(s)...")

            sampler = self.buffer.get_sampler(batch_size=batch_size, update_epochs=update_epochs)
            losses_tracker = []
            stopped_early = False

            for batch in sampler:
                losses = self.dppo.compute_dppo_loss(
                    obs=batch["obs"],
                    chains_prev=batch["chains_prev"],
                    chains_next=batch["chains_next"],
                    denoising_inds=batch["denoising_inds"],
                    returns=batch["returns"],
                    old_values=batch["old_values"],
                    advantages=batch["advantages"],
                    old_logprobs=batch["old_logprobs"],
                    critic_only=is_warmup,
                )

                if is_warmup:
                    # En warmup solo optimizamos el crítico
                    loss_to_backprop = losses["critic_loss"]
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss_to_backprop.backward()
                    torch.nn.utils.clip_grad_norm_(self.dppo.critic.parameters(), max_norm=1.0)
                    self.critic_optimizer.step()
                else:
                    # Actualización conjunta
                    loss_to_backprop = losses["total_loss"]
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss_to_backprop.backward()
                    torch.nn.utils.clip_grad_norm_(self.dppo.actor_ft.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.dppo.critic.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                losses_tracker.append({k: v.item() for k, v in losses.items()})

                # Early stopping si la divergencia KL supera el umbral
                if not is_warmup and losses["approx_kl"].item() > target_kl:
                    self.get_logger().warn(
                        f"Divergencia KL superada ({losses['approx_kl'].item():.4f} > {target_kl}). "
                        "Deteniendo épocas de esta iteración."
                    )
                    stopped_early = True
                    break

            # ── FASE 4: MÉTRICAS Y CHECKPOINTING ───────────────────────────
            mean_ret = float(np.mean(episode_rewards)) if episode_rewards else cur_ep_reward
            mean_p_loss = float(np.mean([l["policy_loss"] for l in losses_tracker]))
            mean_c_loss = float(np.mean([l["critic_loss"] for l in losses_tracker]))
            mean_kl = float(np.mean([l["approx_kl"] for l in losses_tracker]))
            elapsed = time.time() - iter_start_time

            self.get_logger().info(
                f"Iteración {itr} en {elapsed:.1f}s | Reward Medio: {mean_ret:.2f} | "
                f"Policy Loss: {mean_p_loss:.4f} | Critic Loss: {mean_c_loss:.4f} | KL: {mean_kl:.4f}"
            )

            # Exportar checkpoints compatibles con inference.launch.py
            if not is_warmup and mean_ret > best_mean_reward:
                best_mean_reward = mean_ret
                best_path = output_dir / "best"
                self.dppo.export_pretrained(best_path)
                self.get_logger().info(f"¡Nuevo récord de recompensa ({best_mean_reward:.2f})! Exportado a: {best_path}")

            # Exportar último checkpoint
            last_path = output_dir / "last"
            self.dppo.export_pretrained(last_path)

            if itr % save_every == 0:
                periodic_path = output_dir / f"checkpoints/iter_{itr}"
                self.dppo.export_pretrained(periodic_path)
                self.get_logger().info(f"Checkpoint periódico guardado en: {periodic_path}")

        self.get_logger().info("¡Entrenamiento DPPO completado con éxito!")


def main(args=None):
    rclpy.init(args=args)
    node = RLTrainNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
