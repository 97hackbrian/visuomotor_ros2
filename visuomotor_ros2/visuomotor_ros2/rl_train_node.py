"""
RLTrainNode — Nodo de ROS 2 para Entrenamiento y Fine-Tuning de Diffusion Policy con DPPO.

Orquesta el ciclo de aprendizaje por refuerzo:
1. Rollouts síncronos en Isaac Sim mediante GymROS2Env.
2. Almacenamiento en PPODiffusionImgBuffer y cálculo de GAE.
3. Warmup inicial del Crítico (5-10 iteraciones) para evitar desestabilizar la U-Net.
4. Optimización conjunta Actor-Crítico con clipping adaptativo y early stopping por KL.
5. Exportación periódica de checkpoints listos para inference.launch.py.
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use('Agg')  # Backend headless obligatorio para Docker sin servidor X11
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32
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
        self.declare_parameter("n_steps_per_iteration", 300)
        self.declare_parameter("update_epochs", 8)
        self.declare_parameter("batch_size", 60)
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
        self.declare_parameter("max_episode_steps", 900)
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
        self.declare_parameter("reward_transport_sigma", 0.20)
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

        # Parámetros de Cuadrícula de Respawn Aleatorio (Halton QMC)
        self.declare_parameter("random_spawn", True)
        self.declare_parameter("respawn_mode", "on_success")
        self.declare_parameter("spawn_x_min", -0.35)
        self.declare_parameter("spawn_x_max", -0.52)
        self.declare_parameter("spawn_y_min", -0.56)
        self.declare_parameter("spawn_y_max", -1.08)
        self.declare_parameter("spawn_z", 0.44)
        self.declare_parameter("spawn_yaw_min", -3.1415)
        self.declare_parameter("spawn_yaw_max", 3.1415)
        self.declare_parameter("spawn_safe_margin_x", 0.02)
        self.declare_parameter("spawn_safe_margin_y", 0.02)

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
                "random_spawn", "respawn_mode", "spawn_x_min", "spawn_x_max",
                "spawn_y_min", "spawn_y_max", "spawn_z", "spawn_yaw_min", "spawn_yaw_max",
                "spawn_safe_margin_x", "spawn_safe_margin_y",
                "reward_approach_weight", "reward_dist_sigma", "reward_orient_weight",
                "reward_grasp_clamped_bonus", "reward_grasp_lifted_bonus",
                "reward_empty_grasp_penalty", "reward_transport_weight", "reward_transport_sigma",
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

        # ── Telemetría y Métricas en Tiempo Real (para rqt_plot) ──────────
        self.declare_parameter("enable_metrics_publish", True)
        self.declare_parameter("metrics_prefix", "/rl/metrics")
        self.enable_metrics = self.get_parameter("enable_metrics_publish").value
        prefix = self.get_parameter("metrics_prefix").value

        self.pub_instant_reward = self.create_publisher(Float32, f"{prefix}/instant_reward", 10)
        self.pub_episode_reward = self.create_publisher(Float32, f"{prefix}/episode_reward", 10)
        self.pub_critic_value = self.create_publisher(Float32, f"{prefix}/critic_value", 10)
        self.pub_done_flag = self.create_publisher(Float32, f"{prefix}/done_flag", 10)
        self.pub_pseudo_force = self.create_publisher(Float32, f"{prefix}/pseudo_force", 10)
        self.pub_dist_hover = self.create_publisher(Float32, f"{prefix}/dist_hover", 10)
        self.pub_dist_target = self.create_publisher(Float32, f"{prefix}/dist_target", 10)
        self.pub_r_approach = self.create_publisher(Float32, f"{prefix}/r_approach", 10)
        self.pub_r_orient = self.create_publisher(Float32, f"{prefix}/r_orient", 10)
        self.pub_r_grasp = self.create_publisher(Float32, f"{prefix}/r_grasp", 10)
        self.pub_r_transport = self.create_publisher(Float32, f"{prefix}/r_transport", 10)
        self.pub_iter_mean_reward = self.create_publisher(Float32, f"{prefix}/iter_mean_reward", 10)
        self.pub_iter_policy_loss = self.create_publisher(Float32, f"{prefix}/iter_policy_loss", 10)
        self.pub_iter_critic_loss = self.create_publisher(Float32, f"{prefix}/iter_critic_loss", 10)
        self.pub_iter_approx_kl = self.create_publisher(Float32, f"{prefix}/iter_approx_kl", 10)

        # Historial persistente para gráficos offline y archivo CSV
        self.metrics_history = []

        # ── 4. Inicializar DPPO Adapter y Modelo de Difusión ──────────────
        ckpt_path = self.get_parameter("checkpoint_path").value
        self.ckpt_path = ckpt_path
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
            task_success_in_iter = False

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

                    # Publicar métricas en tiempo real a tópicos ROS 2 para rqt_plot
                    self._publish_rl_step_metrics(r, cur_ep_reward, val, a_sub, info)

                    if term or trunc:
                        done = True
                        episode_rewards.append(cur_ep_reward)

                        is_success = bool(term or info.get("task_success", False))
                        if is_success:
                            task_success_in_iter = True
                            self.get_logger().info(
                                f"\n{'='*75}\n"
                                f"🎉 [RL EPISODIO {itr}] ¡TAREA COMPLETADA CON ÉXITO!\n"
                                f"   • Duración: Paso {info.get('step', self.env.current_step)}/{self.env.max_episode_steps}\n"
                                f"   • Retorno del Episodio: {cur_ep_reward:.2f}\n"
                                f"   • Agarre: Clamped={info.get('is_clamped', False)} | Elevación={info.get('delta_z_obj', 0.0):.3f}m | Fuerza={info.get('pseudo_force', 0.0):.4f}\n"
                                f"   • Destino Home: Dist Obj={info.get('dist_target', 0.0)*100:.1f} cm | Dist TCP={info.get('dist_ee_home', 0.0)*100:.1f} cm\n"
                                f"   • Done Flag Neuronal: {info.get('done_flag', 0.0):.4f}\n"
                                f"{'='*75}"
                            )
                        else:
                            self.get_logger().warn(
                                f"\n{'-'*75}\n"
                                f"⚠️  [RL EPISODIO {itr}] TAREA NO COMPLETADA (Episodio Finalizado)\n"
                                f"   • Motivo: {'Límite de tiempo agotado (Truncado)' if trunc else 'Parada prematura en falso'}\n"
                                f"   • Paso alcanzado: {info.get('step', self.env.current_step)}/{self.env.max_episode_steps}\n"
                                f"   • Retorno del Episodio: {cur_ep_reward:.2f}\n"
                                f"   • Agarre: Clamped={info.get('is_clamped', False)} | Elevación={info.get('delta_z_obj', 0.0):.3f}m | PseudoFuerza={info.get('pseudo_force', 0.0):.4f}\n"
                                f"   • Distancia restante a Home: Obj={info.get('dist_target', 0.0)*100:.1f} cm | TCP={info.get('dist_ee_home', 0.0)*100:.1f} cm\n"
                                f"   • Done Flag Neuronal: {info.get('done_flag', 0.0):.4f}\n"
                                f"{'-'*75}"
                            )

                        cur_ep_reward = 0.0
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

                if done:
                    break

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

            task_status_txt = "¡COMPLETADA (ÉXITO)!" if task_success_in_iter else "NO COMPLETADA"
            self.get_logger().info(
                f"Iteración {itr} en {elapsed:.1f}s | Tarea: {task_status_txt} | Reward Medio: {mean_ret:.2f} | "
                f"Policy Loss: {mean_p_loss:.4f} | Critic Loss: {mean_c_loss:.4f} | KL: {mean_kl:.4f}"
            )

            # Publicar métricas de la iteración en tiempo real para rqt_plot
            self._publish_rl_iter_metrics(mean_ret, mean_p_loss, mean_c_loss, mean_kl)

            # Guardar registros en CSV y actualizar curvas gráficas offline
            self._save_rl_metrics_and_plots(
                output_dir=output_dir,
                itr=itr,
                elapsed=elapsed,
                mean_ret=mean_ret,
                mean_p_loss=mean_p_loss,
                mean_c_loss=mean_c_loss,
                mean_kl=mean_kl,
                is_warmup=is_warmup,
                task_success=task_success_in_iter,
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

    def _publish_rl_step_metrics(self, r: float, cur_ep_reward: float, critic_val: float, action_sub: np.ndarray, info: Dict[str, Any]):
        """Publica métricas escalares paso a paso durante los rollouts de RL para monitoreo en rqt_plot."""
        if not self.enable_metrics:
            return

        self.pub_instant_reward.publish(Float32(data=float(r)))
        self.pub_episode_reward.publish(Float32(data=float(cur_ep_reward)))
        self.pub_critic_value.publish(Float32(data=float(critic_val)))

        done_flag = float(action_sub[8]) if len(action_sub) >= 9 else 0.0
        self.pub_done_flag.publish(Float32(data=done_flag))

        if info:
            if "pseudo_force" in info:
                self.pub_pseudo_force.publish(Float32(data=float(info["pseudo_force"])))
            if "dist_hover" in info:
                self.pub_dist_hover.publish(Float32(data=float(info["dist_hover"])))
            if "dist_target" in info:
                self.pub_dist_target.publish(Float32(data=float(info["dist_target"])))
            if "r_approach" in info:
                self.pub_r_approach.publish(Float32(data=float(info["r_approach"])))
            if "r_orient" in info:
                self.pub_r_orient.publish(Float32(data=float(info["r_orient"])))
            if "r_grasp" in info:
                self.pub_r_grasp.publish(Float32(data=float(info["r_grasp"])))
            if "r_transport" in info:
                self.pub_r_transport.publish(Float32(data=float(info["r_transport"])))

    def _publish_rl_iter_metrics(self, mean_ret: float, mean_p_loss: float, mean_c_loss: float, mean_kl: float):
        """Publica el resumen de optimización por iteración a tópicos ROS 2 para rqt_plot."""
        if not self.enable_metrics:
            return
        self.pub_iter_mean_reward.publish(Float32(data=float(mean_ret)))
        self.pub_iter_policy_loss.publish(Float32(data=float(mean_p_loss)))
        self.pub_iter_critic_loss.publish(Float32(data=float(mean_c_loss)))
        self.pub_iter_approx_kl.publish(Float32(data=float(mean_kl)))

    def _save_rl_metrics_and_plots(
        self, output_dir: Path, itr: int, elapsed: float, mean_ret: float,
        mean_p_loss: float, mean_c_loss: float, mean_kl: float, is_warmup: bool,
        task_success: bool = False
    ):
        """Registra métricas en un archivo .csv y genera gráficos de entrenamiento con matplotlib."""
        csv_path = output_dir / "rl_training_metrics.csv"
        file_exists = csv_path.exists()

        row = {
            "iteration": itr,
            "elapsed_s": round(elapsed, 2),
            "mean_reward": round(mean_ret, 4),
            "task_completed": int(task_success),
            "policy_loss": round(mean_p_loss, 6),
            "critic_loss": round(mean_c_loss, 6),
            "approx_kl": round(mean_kl, 6),
            "is_warmup": int(is_warmup),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.metrics_history.append(row)

        try:
            with open(csv_path, mode="a" if file_exists else "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
        except Exception as e:
            self.get_logger().warn(f"Error escribiendo rl_training_metrics.csv: {e}")

        # Generar gráficas con matplotlib (4 paneles de alto impacto)
        try:
            itrs = [r["iteration"] for r in self.metrics_history]
            rewards = [r["mean_reward"] for r in self.metrics_history]
            p_losses = [r["policy_loss"] for r in self.metrics_history]
            c_losses = [r["critic_loss"] for r in self.metrics_history]
            kls = [r["approx_kl"] for r in self.metrics_history]

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Dinámica de Entrenamiento DPPO RL (Online Fine-Tuning)", fontsize=16, fontweight='bold')

            # 1. Recompensa Media Episódica
            axs[0, 0].plot(itrs, rewards, color="#1f77b4", marker='o', markersize=3, alpha=0.6, label="Reward Medio")
            if len(rewards) >= 5:
                smooth = np.convolve(rewards, np.ones(5)/5, mode='valid')
                axs[0, 0].plot(itrs[4:], smooth, color="#d62728", linewidth=2.0, label="Media Móvil (5 iters)")
            axs[0, 0].set_title("Retorno Episódico (Reward)")
            axs[0, 0].set_xlabel("Iteración")
            axs[0, 0].set_ylabel("Recompensa Total")
            axs[0, 0].grid(True, linestyle="--", alpha=0.6)
            axs[0, 0].legend()

            # 2. Pérdida del Crítico V(s)
            axs[0, 1].plot(itrs, c_losses, color="#ff7f0e", marker='s', markersize=3, label="Critic Loss (MSE)")
            axs[0, 1].set_title("Pérdida del Crítico $V(s)$")
            axs[0, 1].set_xlabel("Iteración")
            axs[0, 1].set_ylabel("MSE Loss")
            axs[0, 1].grid(True, linestyle="--", alpha=0.6)
            axs[0, 1].legend()

            # 3. Pérdida de la Política
            axs[1, 0].plot(itrs, p_losses, color="#2ca02c", marker='^', markersize=3, label="Policy Loss")
            axs[1, 0].set_title("Pérdida de la Política (Actor PPO + BC)")
            axs[1, 0].set_xlabel("Iteración")
            axs[1, 0].set_ylabel("Pérdida")
            axs[1, 0].grid(True, linestyle="--", alpha=0.6)
            axs[1, 0].legend()

            # 4. Divergencia KL
            axs[1, 1].plot(itrs, kls, color="#9467bd", marker='d', markersize=3, label="Approx KL")
            target_kl = self.get_parameter("target_kl").value
            axs[1, 1].axhline(y=target_kl, color="r", linestyle="--", label=f"Target KL ({target_kl})")
            axs[1, 1].set_title("Divergencia KL (Desplazamiento de Política)")
            axs[1, 1].set_xlabel("Iteración")
            axs[1, 1].set_ylabel("KL")
            axs[1, 1].grid(True, linestyle="--", alpha=0.6)
            axs[1, 1].legend()

            plt.tight_layout()
            plot_png = output_dir / "rl_training_curves.png"
            plot_pdf = output_dir / "rl_training_curves.pdf"
            plt.savefig(plot_png, dpi=200, bbox_inches="tight")
            plt.savefig(plot_pdf, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            self.get_logger().warn(f"Error generando gráficas de RL con matplotlib: {e}")

        # Hook Relacional SQLite Fail-Safe y No Intrusivo
        try:
            from visuomotor_core.utils.tracker import record_rl_run
            success_count = sum(r.get("task_completed", 0) for r in self.metrics_history)
            sr_percent = (success_count / len(self.metrics_history)) * 100.0 if self.metrics_history else 0.0

            record_rl_run(
                run_id=f"rl_{output_dir.name}",
                base_checkpoint=str(getattr(self, "ckpt_path", "")),
                iterations_completed=itr,
                mean_reward=float(mean_ret),
                policy_loss=float(mean_p_loss),
                critic_loss=float(mean_c_loss),
                approx_kl=float(mean_kl),
                task_success_rate=round(sr_percent, 2),
                duration_s=round(elapsed, 2),
                checkpoint_dir=str(output_dir),
                metadata={
                    "respawn_mode": str(self.get_parameter("respawn_mode").value) if self.has_parameter("respawn_mode") else "unknown",
                    "target_kl": float(self.get_parameter("target_kl").value) if self.has_parameter("target_kl") else 0.05,
                    "ft_denoising_steps": int(self.get_parameter("ft_denoising_steps").value) if self.has_parameter("ft_denoising_steps") else 5,
                    "is_warmup": bool(is_warmup),
                },
            )
        except Exception as e:
            self.get_logger().debug(f"[DB Tracker Fail-Safe] Error no crítico al registrar corrida RL en SQLite: {e}")


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
