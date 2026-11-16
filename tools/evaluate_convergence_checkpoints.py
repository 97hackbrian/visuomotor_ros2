#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Evaluador Físico Multi-Checkpoint de Convergencia en Robótica Visuomotora:
Evalúa sistemáticamente múltiples checkpoints de entrenamiento (epoch_10, epoch_20,
epoch_30, latest/epoch_34) sobre el simulador físico NVIDIA Isaac Sim vía ROS 2 Jazzy.

Objetivo Científico:
1. Determinar empíricamente la tasa de éxito físico de agarre (Success Rate %) y precisión
   TCP (79.0 mm offset) en función del número de épocas de entrenamiento.
2. Identificar la 'Cantidad Efectiva de Épocas' (E*) necesaria para considerar un modelo
   físicamente óptimo.
3. Contrastar y calibrar los métodos de criterios de parada matemáticos (Kneedle, Flatness,
   RSD, Prechelt, EMA Gap) contra la realidad del contacto físico robótico.
4. Exportar telemetría tabular CSV, reportes JSON y figuras de calibración a 300 DPI (PNG y PDF).
"""

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch
import yaml

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# Imports locales del workspace
sys.path.append(str(Path(__file__).parent.parent / "visuomotor_core"))
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from visuomotor_ros2.gym_ros2_env import GymROS2Env


# Paleta científica accesible por checkpoint
CHECKPOINT_COLORS = {
    "epoch_10": "#1f77b4",       # Azul estándar
    "epoch_20": "#2ca02c",       # Verde bosque
    "epoch_30": "#ff7f0e",       # Naranja ámbar
    "latest_ep34": "#9467bd",    # Púrpura sobrio
}


def load_rl_params(params_path: Optional[Path] = None) -> Dict:
    """Carga los parámetros de RL y límites de spawn desde rl_training_params.yaml."""
    candidates = []
    if params_path:
        candidates.append(Path(params_path))
    candidates.extend([
        Path("/ros/visuomotor_ros2/config/rl_training_params.yaml"),
        Path(__file__).parent.parent / "visuomotor_ros2" / "config" / "rl_training_params.yaml",
        Path("/ros/install/visuomotor_ros2/share/visuomotor_ros2/config/rl_training_params.yaml"),
        Path(__file__).parent.parent / "install" / "visuomotor_ros2" / "share" / "visuomotor_ros2" / "config" / "rl_training_params.yaml",
    ])
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if "rl_train_node" in data and "ros__parameters" in data["rl_train_node"]:
                        return data["rl_train_node"]["ros__parameters"]
            except Exception:
                pass
    return {}


class CheckpointEvaluationRunner(Node):
    def __init__(
        self,
        out_dir: Path,
        max_steps_per_episode: int = 2400,
        cfg_override: Optional[Dict] = None,
        save_video: bool = True,
    ):
        super().__init__("checkpoint_evaluation_runner")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps_per_episode
        self.save_video = save_video

        env_config = {
            "obs_dim": 8,
            "act_dim": 9,
            "image_size": 256,
            "n_obs_steps": 2,
            "max_episode_steps": self.max_steps,
            "step_timeout_s": 3.0,
            "reset_stabilize_s": 2.0,
            "reset_duration_s": 4.0,
            "base_frame": "link_base",
            "ee_frame": "gripper_base_link",
            "object_frame": "pick_target",
            "target_x": 0.25,
            "target_y": 0.0,
            "target_z": 0.40,
            "gripper_open_pos": 0.0,
            "gripper_closed_pos": -0.01,
            "pseudo_force_clamped_threshold": 0.0007,
            "gripper_binary_closed_threshold": -0.001,
            "gripper_joint_name_pattern": "gripper",
            "camera_topic": "/rgb",
            "joint_states_topic": "/joint_states",
            "target_frame_topic": "/target_frame_raw",
            "gripper_command_topic": "/position_controller/commands",
            "respawn_topic": "/respawn",
            "random_spawn": True,
            "respawn_mode": "manual",
        }

        # Aplicar overrides de rl_training_params.yaml (spawn limits, thresholds, etc.)
        if cfg_override:
            for k in [
                "spawn_x_min", "spawn_x_max", "spawn_y_min", "spawn_y_max",
                "spawn_z", "spawn_yaw_min", "spawn_yaw_max",
                "spawn_safe_margin_x", "spawn_safe_margin_y",
                "pseudo_force_clamped_threshold", "gripper_binary_closed_threshold",
                "target_x", "target_y", "target_z", "lift_required_z",
                "tcp_offset_z", "hover_z_offset", "camera_topic"
            ]:
                if k in cfg_override:
                    env_config[k] = cfg_override[k]

        self.env = GymROS2Env(node=self, config=env_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Checkpoint Benchmark Runner inicializado en dispositivo: {self.device} (save_video={self.save_video})")

    def _draw_hud(
        self,
        img: np.ndarray,
        model_name: str,
        epoch: int,
        step: int,
        elapsed_s: float,
        dist_tcp_mm: float,
        is_clamped: bool,
        gripper_cmd: float,
        done_val: float,
    ) -> np.ndarray:
        """Dibuja una elegante capa HUD de telemetría sobre el fotograma de la cámara."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, h - 26), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"{model_name} (Ep {epoch}) | Paso {step} ({elapsed_s:.1f}s)"
        cv2.putText(img, title, (8, 16), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        dist_str = f"TCP: {dist_tcp_mm:.1f}mm" if dist_tcp_mm < 999.0 else "TCP: --"
        clamp_str = "CLAMP: OK" if is_clamped else "CLAMP: NO"
        clamp_color = (0, 255, 0) if is_clamped else (180, 180, 180)
        cv2.putText(img, dist_str, (8, 34), font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, clamp_str, (110, 34), font, 0.38, clamp_color, 1, cv2.LINE_AA)

        grip_str = "GRIP: CERR" if gripper_cmd <= -0.001 else "GRIP: ABIE"
        done_str = f"Done: {done_val:.3f}"
        cv2.putText(img, grip_str, (8, h - 9), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, done_str, (120, h - 9), font, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
        return img

    def run_single_episode(
        self,
        policy: DiffusionPolicy,
        model_name: str,
        epoch_num: int,
        ep_idx: int,
        spawn_pose: Tuple[float, float, float, float],
    ) -> Dict:
        """Ejecuta un episodio cerrado de inferencia en simulación y extrae telemetría."""
        tx, ty, tz, tyaw = spawn_pose
        self.get_logger().info(
            f"   ▶ [{model_name} (Época {epoch_num})] Ep {ep_idx} | Pose: X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}, Yaw={tyaw:.2f} rad"
        )

        obs, _ = self.env.reset(options={"spawn_pose": (tx, ty, tz, tyaw)})
        policy.reset()

        ep_start_time = time.time()
        tracking_errors = []
        final_info = {}
        auto_stop_triggered = False
        final_done_val = 0.0
        home_target = np.array([0.25, 0.0, 0.40], dtype=np.float32)
        min_dist_ee_obj = float("inf")
        min_dist_tcp_obj = float("inf")
        recorded_frames = []

        if self.save_video and self.env.latest_cv_img is not None:
            # self.env.latest_cv_img es RGB8; convertir a BGR para dibujo OpenCV y VideoWriter nativo
            bgr_img = cv2.cvtColor(self.env.latest_cv_img, cv2.COLOR_RGB2BGR)
            init_frame = self._draw_hud(
                bgr_img,
                model_name,
                epoch_num,
                0,
                0.0,
                float("inf"),
                False,
                0.0,
                0.0,
            )
            recorded_frames.append(init_frame)

        consecutive_timeouts = 0
        for step_i in range(self.max_steps):
            last_img = obs["observation.image"][-1]
            last_state = obs["observation.state"][-1]

            img_tensor = torch.from_numpy(last_img).float().to(self.device).unsqueeze(0)
            state_tensor = torch.from_numpy(last_state).float().to(self.device).unsqueeze(0)

            obs_dict = {
                "observation.image": img_tensor,
                "observation.state": state_tensor,
            }

            with torch.inference_mode():
                action_out = policy.select_action(obs_dict)
                action_np = action_out.squeeze().cpu().numpy()

            done_flag = float(action_np[8]) if len(action_np) >= 9 else 0.0
            final_done_val = done_flag

            # Error de seguimiento cartesiano
            cmd_pos = action_np[:3]
            actual_pos = last_state[:3]
            err_xyz = float(np.linalg.norm(cmd_pos - actual_pos))
            tracking_errors.append(err_xyz)

            # Ejecutar acción física en Isaac Sim
            next_obs, reward, term, trunc, info = self.env.step(action_np)
            final_info = info
            obs = next_obs

            # Distancias cinemáticas
            cur_dist_ee = float(info.get("dist_ee_obj", np.linalg.norm(actual_pos - np.array([tx, ty, tz]))))
            cur_dist_tcp = float(info.get("dist_tcp_obj", cur_dist_ee))
            if cur_dist_ee < min_dist_ee_obj:
                min_dist_ee_obj = cur_dist_ee
            if cur_dist_tcp < min_dist_tcp_obj:
                min_dist_tcp_obj = cur_dist_tcp

            # Grabación de video con capa HUD (conversión explícita RGB a BGR para VideoWriter)
            if self.save_video and self.env.latest_cv_img is not None:
                bgr_img = cv2.cvtColor(self.env.latest_cv_img, cv2.COLOR_RGB2BGR)
                hud_frame = self._draw_hud(
                    bgr_img,
                    model_name,
                    epoch_num,
                    step_i + 1,
                    time.time() - ep_start_time,
                    cur_dist_tcp * 1000.0,
                    bool(final_info.get("is_clamped", False)),
                    float(action_np[7]) if len(action_np) >= 8 else 0.0,
                    done_flag,
                )
                recorded_frames.append(hud_frame)

            if info.get("timeout", False):
                consecutive_timeouts += 1
                if consecutive_timeouts >= 5:
                    self.get_logger().error("     [ABORT] 5 timeouts consecutivos de simulación DDS. Abortando episodio.")
                    break
                continue
            consecutive_timeouts = 0

            dist_home = float(info.get("dist_ee_home", 1.0))
            is_clamped_now = bool(info.get("is_clamped", False))
            arrival_thresh = 0.05 if is_clamped_now else 0.08
            if (term and not info.get("timeout", False)) or (done_flag >= 0.50 and dist_home <= arrival_thresh):
                auto_stop_triggered = True
                self.get_logger().info(
                    f"     [Paso {step_i}] ¡Auto-Stop Neuronal en Destino! "
                    f"(done_flag={done_flag:.4f}, dist_home={dist_home*100:.1f}cm, clamped={is_clamped_now})"
                )
                break

            if trunc:
                break

        ep_duration = time.time() - ep_start_time
        obj_data = self.env._get_object_pose()
        obj_pos = obj_data[0] if obj_data is not None else np.array([tx, ty, tz])
        ee_pos = self.env.latest_state[:3] if self.env.latest_state is not None else home_target

        # Objetivo físico del cubo sostenido en Home: considera el offset TCP de pinza (79 mm hacia abajo)
        expected_obj_target = home_target - np.array([0.0, 0.0, 0.079], dtype=np.float32)

        min_dist_ee_obj_mm = float(min_dist_ee_obj * 1000.0) if min_dist_ee_obj != float("inf") else 0.0
        min_dist_tcp_obj_mm = float(min_dist_tcp_obj * 1000.0) if min_dist_tcp_obj != float("inf") else 0.0
        terminal_error_obj_mm = float(np.linalg.norm(obj_pos - expected_obj_target) * 1000.0)
        terminal_error_tcp_mm = float(np.linalg.norm(ee_pos - home_target) * 1000.0)
        mean_track_err_mm = float(np.mean(tracking_errors) * 1000.0) if tracking_errors else 0.0

        is_clamped = bool(final_info.get("is_clamped", False))
        delta_z = float(final_info.get("delta_z_obj", 0.0))
        task_success = bool(final_info.get("task_success", False)) or (
            is_clamped and (terminal_error_obj_mm <= 100.0 or terminal_error_tcp_mm <= 60.0)
        )

        # Exportar video MP4, transcodificar H.264 y generar GIF animado
        if self.save_video and recorded_frames:
            video_dir = self.out_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            raw_video_path = video_dir / f"{model_name}_ep{ep_idx}_raw.mp4"
            h264_video_path = video_dir / f"{model_name}_ep{ep_idx}.mp4"
            gif_path = video_dir / f"{model_name}_ep{ep_idx}.gif"

            try:
                h_f, w_f = recorded_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_vid = cv2.VideoWriter(str(raw_video_path), fourcc, 20.0, (w_f, h_f))
                for f_img in recorded_frames:
                    out_vid.write(f_img)
                out_vid.release()

                os.system(f"ffmpeg -y -i '{raw_video_path}' -vcodec libx264 -pix_fmt yuv420p '{h264_video_path}' >/dev/null 2>&1")
                if raw_video_path.exists():
                    raw_video_path.unlink()

                from PIL import Image as PILImage
                gif_imgs = [PILImage.fromarray(cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)) for f_img in recorded_frames[::3]]
                if gif_imgs:
                    gif_imgs[0].save(str(gif_path), save_all=True, append_images=gif_imgs[1:], duration=150, loop=0)

                # Sincronizar automáticamente hacia artifacts para visualización del usuario
                artifact_dir = Path("/home/hackbrian/.gemini/antigravity/brain/9e8fb575-52ae-460a-9baf-633ef92688f4")
                if artifact_dir.exists():
                    if h264_video_path.exists():
                        shutil.copy2(h264_video_path, artifact_dir / h264_video_path.name)
                    if gif_path.exists():
                        shutil.copy2(gif_path, artifact_dir / gif_path.name)
                self.get_logger().info(f"   🎬 Video H.264 exportado: {h264_video_path.name} | GIF: {gif_path.name}")
            except Exception as e_vid:
                self.get_logger().warn(f"   [AVISO] No se pudo procesar video: {e_vid}")

        status_str = "ÉXITO" if task_success else "FALLO"
        self.get_logger().info(
            f"   ✔ [{model_name}] Resultado: {status_str} | Duración: {ep_duration:.1f}s | Pasos: {len(tracking_errors)}\n"
            f"     • Distancia Contacto TCP-Objeto: {min_dist_tcp_obj_mm:.1f} mm\n"
            f"     • Error Obj: {terminal_error_obj_mm:.1f} mm | Clamped: {is_clamped} | Done Flag: {final_done_val:.4f}"
        )

        return {
            "model": model_name,
            "epoch": epoch_num,
            "episode": ep_idx,
            "spawn_x": round(tx, 4),
            "spawn_y": round(ty, 4),
            "spawn_z": round(tz, 4),
            "spawn_yaw": round(tyaw, 4),
            "task_success": int(task_success),
            "is_clamped": int(is_clamped),
            "min_dist_tcp_obj_mm": round(min_dist_tcp_obj_mm, 2),
            "min_dist_ee_obj_mm": round(min_dist_ee_obj_mm, 2),
            "delta_z_cm": round(delta_z * 100.0, 2),
            "terminal_error_obj_mm": round(terminal_error_obj_mm, 2),
            "terminal_error_tcp_mm": round(terminal_error_tcp_mm, 2),
            "tracking_rmse_mm": round(mean_track_err_mm, 2),
            "duration_s": round(ep_duration, 2),
            "auto_stop_triggered": int(auto_stop_triggered),
            "final_done_flag": round(final_done_val, 4),
        }


def generate_shared_poses(
    n: int,
    seed: int = 42,
    x_bounds: Tuple[float, float] = (-0.38, -0.42),
    y_bounds: Tuple[float, float] = (-0.56, -1.08),
    yaw_bounds: Tuple[float, float] = (-3.1415, 3.1415),
    spawn_z: float = 0.44,
    safe_margins: Tuple[float, float] = (0.02, 0.02),
) -> List[Tuple[float, float, float, float]]:
    """Genera n poses cuasi-aleatorias Halton dentro de los límites de spawn calibrados."""
    sampler = qmc.Halton(d=3, scramble=True, seed=seed)
    samples = sampler.random(n)
    poses = []
    x_low, x_high = min(x_bounds), max(x_bounds)
    y_low, y_high = min(y_bounds), max(y_bounds)
    yaw_low, yaw_high = min(yaw_bounds), max(yaw_bounds)
    margin_x, margin_y = safe_margins

    for s in samples:
        x = float(x_low + s[0] * (x_high - x_low))
        y = float(y_low + s[1] * (y_high - y_low))
        dist_x = min(abs(x - x_low), abs(x - x_high))
        dist_y = min(abs(y - y_low), abs(y - y_high))
        if dist_x < margin_x or dist_y < margin_y:
            yaw = 0.0
        else:
            yaw = float(yaw_low + s[2] * (yaw_high - yaw_low))
        poses.append((round(x, 4), round(y, 4), spawn_z, round(yaw, 4)))
    return poses


def plot_checkpoint_calibration(
    df: pd.DataFrame,
    out_dir: Path,
    epoch_metrics_path: Optional[Path] = None,
    stopping_summary_path: Optional[Path] = None,
):
    """
    Genera figuras científicas de calibración que cruzan:
    Tasa de éxito físico vs. Épocas, pérdida de entrenamiento y líneas de corte
    de los criterios de parada matemáticos.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Agrupar por checkpoint / época
    models = sorted(df["model"].unique(), key=lambda m: df[df["model"] == m]["epoch"].iloc[0])
    epochs = [int(df[df["model"] == m]["epoch"].iloc[0]) for m in models]
    sr_vals = [float(df[df["model"] == m]["task_success"].mean() * 100.0) for m in models]
    clamp_vals = [float(df[df["model"] == m]["is_clamped"].mean() * 100.0) for m in models]
    tcp_dist_mean = [float(df[df["model"] == m]["min_dist_tcp_obj_mm"].mean()) for m in models]
    tcp_dist_std = [float(df[df["model"] == m]["min_dist_tcp_obj_mm"].std()) for m in models]
    auto_stop_vals = [float(df[df["model"] == m]["auto_stop_triggered"].mean() * 100.0) for m in models]

    # Leer historial de pérdida por época si existe
    epoch_losses = {}
    if epoch_metrics_path and epoch_metrics_path.exists():
        try:
            ep_df = pd.read_csv(epoch_metrics_path)
            for _, r in ep_df.iterrows():
                epoch_losses[int(r["epoch"])] = float(r["avg_loss"])
        except Exception as e:
            print(f"[Aviso] No se pudo leer epoch_metrics.csv: {e}")

    # =========================================================================
    # FIGURA 1: CURVA PARETO (Tasa de Éxito Físico vs. Épocas vs. Loss MSE)
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Frontera de Calibración: Tasa de Éxito Físico (%) vs. Épocas de Entrenamiento (Diffusion Policy)",
        fontsize=14,
        fontweight="bold",
    )

    # Eje izquierdo: Tasa de Éxito Físico y Agarre Firme
    ax1.plot(epochs, sr_vals, color="#2ca02c", marker="o", linewidth=3.0, markersize=9, label="Tasa de Éxito Físico (%)")
    ax1.plot(epochs, clamp_vals, color="#1f77b4", marker="s", linestyle="--", linewidth=2.2, markersize=8, label="Agarre Firme Clamped (%)")
    ax1.set_xlabel("Épocas de Entrenamiento", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Tasa de Éxito / Agarre (%)", fontweight="bold", fontsize=12, color="#2ca02c")
    ax1.set_ylim(-5, 115)
    ax1.set_xticks(epochs)
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for ep, sr in zip(epochs, sr_vals):
        ax1.annotate(
            f"{sr:.0f}%",
            (ep, sr),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=11,
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0fff0", edgecolor="#2ca02c", alpha=0.9),
        )

    # Eje derecho: Curva de Pérdida MSE (si existe)
    if epoch_losses:
        all_eps = sorted(epoch_losses.keys())
        all_losses = [epoch_losses[e] for e in all_eps]
        ax2 = ax1.twinx()
        ax2.plot(all_eps, all_losses, color="#d62728", linestyle=":", linewidth=2.0, alpha=0.8, label="Loss MSE Promedio (Época)")
        ax2.set_ylabel("Pérdida MSE (Escala Log)", fontweight="bold", fontsize=12, color="#d62728")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    # Superposición de cortes de criterios de parada matemáticos si existe summary
    if stopping_summary_path and stopping_summary_path.exists():
        try:
            stop_df = pd.read_csv(stopping_summary_path)
            crit_colors = {
                "kneedle_elbow": "#e377c2",
                "gradient_flatness": "#8c564b",
                "variance_stabilization": "#9467bd",
                "prechelt_progress": "#17becf",
                "consensus": "#bcbd22",
            }
            for _, r in stop_df.iterrows():
                m_id = str(r["method_id"])
                rec_ep = float(r["recommended_epoch"])
                if m_id in crit_colors and rec_ep <= max(epochs) * 1.5:
                    c = crit_colors[m_id]
                    name = str(r["method_name"])
                    ax1.axvline(x=rec_ep, color=c, linestyle="-.", linewidth=1.5, alpha=0.85)
                    ax1.text(
                        rec_ep,
                        105 - (list(crit_colors.keys()).index(m_id) * 8),
                        f" {name}: Ép. {rec_ep:.1f}",
                        color=c,
                        fontweight="bold",
                        fontsize=8.5,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor=c, alpha=0.8),
                    )
        except Exception as e:
            print(f"[Aviso] No se pudieron superponer criterios de parada: {e}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if epoch_losses:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.95)
    else:
        ax1.legend(lines1, labels1, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    fig_png = out_dir / "checkpoint_success_rate_vs_epochs.png"
    fig_pdf = out_dir / "checkpoint_success_rate_vs_epochs.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Curva Pareto exportada en: {fig_png} y {fig_pdf}")

    # =========================================================================
    # FIGURA 2: PANEL DIAGNÓSTICO 4-CANALES (Cinemática y Destreza de Contacto)
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Diagnóstico Físico Multi-Checkpoint: Contacto, Error Terminal y Auto-Stop", fontsize=15, fontweight="bold")

    # A) Contacto TCP (mm) vs Época
    axs[0, 0].errorbar(epochs, tcp_dist_mean, yerr=tcp_dist_std, fmt="-o", color="#1f77b4", ecolor="#aec7e8", elinewidth=2, capsize=5, linewidth=2.5, markersize=8)
    axs[0, 0].axhspan(0.0, 5.0, color="#2ca02c", alpha=0.15, label="Zona de Contacto Físico Real (0 - 5 mm)")
    axs[0, 0].set_title("A) Proximidad de Contacto TCP-Cubo (mm)", fontweight="bold")
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("Distancia Mínima TCP (mm)")
    axs[0, 0].set_xticks(epochs)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="upper right")

    # B) Error Terminal Objeto (mm)
    term_means = [float(df[df["model"] == m]["terminal_error_obj_mm"].mean()) for m in models]
    term_stds = [float(df[df["model"] == m]["terminal_error_obj_mm"].std()) for m in models]
    axs[0, 1].errorbar(epochs, term_means, yerr=term_stds, fmt="-s", color="#ff7f0e", ecolor="#ffbb78", elinewidth=2, capsize=5, linewidth=2.5, markersize=8)
    axs[0, 1].axhline(y=60.0, color="#d62728", linestyle="--", linewidth=1.8, label="Tolerancia Máx. Entrega (60 mm)")
    axs[0, 1].set_title("B) Error Terminal en Zona de Entrega (mm)", fontweight="bold")
    axs[0, 1].set_xlabel("Épocas")
    axs[0, 1].set_ylabel("Error Terminal Objeto (mm)")
    axs[0, 1].set_xticks(epochs)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend(loc="upper left")

    # C) Tracking RMSE Cartesiano (mm)
    rmse_means = [float(df[df["model"] == m]["tracking_rmse_mm"].mean()) for m in models]
    rmse_stds = [float(df[df["model"] == m]["tracking_rmse_mm"].std()) for m in models]
    axs[1, 0].errorbar(epochs, rmse_means, yerr=rmse_stds, fmt="-^", color="#2ca02c", ecolor="#98df8a", elinewidth=2, capsize=5, linewidth=2.5, markersize=8)
    axs[1, 0].set_title("C) Fidelidad Cinemática Cartesiana RMSE (mm)", fontweight="bold")
    axs[1, 0].set_xlabel("Épocas")
    axs[1, 0].set_ylabel("Tracking RMSE (mm)")
    axs[1, 0].set_xticks(epochs)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # D) Auto-Stop Neuronal (%)
    axs[1, 1].bar([str(e) for e in epochs], auto_stop_vals, color="#9467bd", alpha=0.85, edgecolor="black", width=0.45)
    axs[1, 1].set_title("D) Tasa de Activación del Auto-Stop Neuronal (%)", fontweight="bold")
    axs[1, 1].set_xlabel("Épocas")
    axs[1, 1].set_ylabel("Auto-Stop Correcto (%)")
    axs[1, 1].set_ylim(0, 115)
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for idx, v in enumerate(auto_stop_vals):
        axs[1, 1].text(idx, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    diag_png = out_dir / "checkpoint_kinematics_calibration_4panel.png"
    diag_pdf = out_dir / "checkpoint_kinematics_calibration_4panel.pdf"
    plt.savefig(diag_png, dpi=300, bbox_inches="tight")
    plt.savefig(diag_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Panel 4-canales exportado en: {diag_png} y {diag_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación física comparativa de checkpoints en Isaac Sim.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="/ros/visuomotor_core/outputs/train_v8_continuous_convergence",
        help="Directorio del entrenamiento (busca checkpoints/epoch_N y latest)",
    )
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Episodios por cada checkpoint (defecto: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla QMC Halton determinista compartida")
    parser.add_argument("--max_steps", type=int, default=2400, help="Pasos máximos por episodio (120s a 20Hz, defecto: 2400)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directorio de salida para CSVs y gráficos")
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Lista de épocas separadas por coma a evaluar (ej. '10,50,100,150,200,250,300')",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=0,
        help="Selecciona automáticamente una grilla equidistante de N checkpoints (ej. 7)",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="/ros/visuomotor_ros2/config/rl_training_params.yaml",
        help="Archivo YAML de configuración con límites de spawn y parámetros de simulación",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Graba video MP4 y animación GIF de cada episodio con HUD de telemetría (por defecto: True)",
    )
    parser.add_argument(
        "--no_video",
        dest="save_video",
        action="store_false",
        help="Desactiva la grabación de videos para máxima velocidad",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "physical_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar parámetros de configuración de RL (incluyendo límites de spawn)
    rl_params = load_rl_params(Path(args.params_file) if args.params_file else None)
    x_bounds = (float(rl_params.get("spawn_x_min", -0.38)), float(rl_params.get("spawn_x_max", -0.42)))
    y_bounds = (float(rl_params.get("spawn_y_min", -0.56)), float(rl_params.get("spawn_y_max", -1.08)))
    yaw_bounds = (float(rl_params.get("spawn_yaw_min", -3.1415)), float(rl_params.get("spawn_yaw_max", 3.1415)))
    spawn_z = float(rl_params.get("spawn_z", 0.44))
    safe_margins = (float(rl_params.get("spawn_safe_margin_x", 0.02)), float(rl_params.get("spawn_safe_margin_y", 0.02)))

    # Identificar automáticamente los checkpoints a evaluar
    ckpt_dir = run_dir / "checkpoints"
    available_ckpts = []

    if ckpt_dir.exists():
        subdirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
                         key=lambda p: int(p.name.split("_")[1]))
        for d in subdirs:
            ep_n = int(d.name.split("_")[1])
            available_ckpts.append((d.name, ep_n, d))

    # Añadir checkpoint latest si existe y no está duplicado
    if (run_dir / "model.safetensors").exists() or (run_dir / "checkpoints" / "latest").exists():
        # Leer última época de epoch_metrics.csv
        latest_ep = 34
        ep_csv = run_dir / "epoch_metrics.csv"
        if ep_csv.exists():
            try:
                ep_df = pd.read_csv(ep_csv)
                latest_ep = int(ep_df["epoch"].max())
            except Exception:
                pass
        existing_eps = [ep for _, ep, _ in available_ckpts]
        if latest_ep not in existing_eps:
            available_ckpts.append((f"latest_ep{latest_ep}", latest_ep, run_dir))

    # Filtrar según --epochs o --grid si se especificó
    if args.epochs:
        target_eps = set([int(x.strip()) for x in args.epochs.split(",") if x.strip()])
        available_ckpts = [c for c in available_ckpts if c[1] in target_eps]
        available_ckpts.sort(key=lambda c: c[1])
    elif args.grid and args.grid > 1:
        if len(available_ckpts) > args.grid:
            indices = np.round(np.linspace(0, len(available_ckpts) - 1, args.grid)).astype(int)
            available_ckpts = [available_ckpts[i] for i in indices]

    if not available_ckpts:
        print(f"[ERROR] No se encontraron checkpoints válidos en: {run_dir}")
        sys.exit(1)

    print("\n" + "="*85)
    print(f"🎯 CAMPAÑA DE EVALUACIÓN MULTI-CHECKPOINT ({len(available_ckpts)} MODELOS x {args.episodes} EPISODIOS)")
    print("="*85)
    print(f"  • Límites de Spawn X: [{min(x_bounds):.3f}, {max(x_bounds):.3f}] m (Profundidad)")
    print(f"  • Límites de Spawn Y: [{min(y_bounds):.3f}, {max(y_bounds):.3f}] m (Lateral Mesa)")
    print(f"  • Altura Spawn Z:     {spawn_z:.3f} m | Margen de Seguridad: {safe_margins} m")
    print(f"  • Grabación de Video: {'ACTIVADA (H.264 + GIF + HUD)' if args.save_video else 'DESACTIVADA'}")
    print("-"*85)
    for name, ep, path in available_ckpts:
        print(f"  • Modelo: {name:15s} | Época: {ep:2d} | Ruta: {path}")
    print("="*85)

    rclpy.init()
    runner = CheckpointEvaluationRunner(
        out_dir=out_dir,
        max_steps_per_episode=args.max_steps,
        cfg_override=rl_params,
        save_video=args.save_video,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(runner)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(2.0)

    # Cargar todos los modelos en GPU
    loaded_policies = []
    for name, ep, path in available_ckpts:
        runner.get_logger().info(f"Cargando pesos de {name} desde {path}...")
        policy = DiffusionPolicy.from_pretrained(path).to(runner.device).eval()
        loaded_policies.append((name, ep, policy))

    # Generar N poses cuasi-aleatorias Halton compartidas usando los límites calibrados
    shared_poses = generate_shared_poses(
        n=args.episodes,
        seed=args.seed,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        yaw_bounds=yaw_bounds,
        spawn_z=spawn_z,
        safe_margins=safe_margins,
    )
    runner.get_logger().info(
        f"Generadas {len(shared_poses)} poses aleatorias idénticas usando límites calibrados: "
        f"X=[{min(x_bounds):.3f}, {max(x_bounds):.3f}], Y=[{min(y_bounds):.3f}, {max(y_bounds):.3f}]."
    )

    all_results = []

    # Evaluación pareada secuencial
    for ep_i, pose in enumerate(shared_poses, start=1):
        print(f"\n" + "-"*85)
        print(f"🎲 EPISODIO {ep_i}/{args.episodes} (Pose Compartida: X={pose[0]:.3f}, Y={pose[1]:.3f}, Yaw={pose[3]:.2f} rad)")
        print("-"*85)

        for name, ep_num, policy in loaded_policies:
            res = runner.run_single_episode(policy, name, ep_num, ep_i, pose)
            all_results.append(res)
            # Guardado incremental inmediato para persistencia continua y fail-safe
            csv_path = out_dir / "checkpoint_benchmark_episodes.csv"
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
                writer.writeheader()
                writer.writerows(all_results)

    # Guardar CSV de episodios
    csv_path = out_dir / "checkpoint_benchmark_episodes.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n📁 Resultados detallados guardados en: {csv_path}")

    # DataFrame consolidado
    df = pd.DataFrame(all_results)

    # Guardar resumen tabular por modelo
    summary_rows = []
    print("\n" + "="*85)
    print("🏆 RESUMEN COMPARATIVO POR ÉPOCA DE ENTRENAMIENTO")
    print("="*85)
    print(f"{'Modelo':15s} | {'Época':5s} | {'Éxito %':7s} | {'Clamp %':7s} | {'TCP Dist (mm)':16s} | {'Error Obj (mm)':16s} | {'Auto-Stop %':11s}")
    print("-"*85)

    effective_epoch = None
    max_sr = 0.0

    for name, ep_num, _ in loaded_policies:
        m_df = df[df["model"] == name]
        sr = float(m_df["task_success"].mean() * 100.0)
        clamp_rate = float(m_df["is_clamped"].mean() * 100.0)
        tcp_dist_mean = float(m_df["min_dist_tcp_obj_mm"].mean())
        tcp_dist_std = float(m_df["min_dist_tcp_obj_mm"].std())
        term_mean = float(m_df["terminal_error_obj_mm"].mean())
        term_std = float(m_df["terminal_error_obj_mm"].std())
        auto_stop_rate = float(m_df["auto_stop_triggered"].mean() * 100.0)

        print(f"{name:15s} | {ep_num:5d} | {sr:6.1f}% | {clamp_rate:6.1f}% | {tcp_dist_mean:6.2f} ± {tcp_dist_std:5.2f} | {term_mean:6.2f} ± {term_std:5.2f} | {auto_stop_rate:10.1f}%")

        summary_rows.append({
            "model": name,
            "epoch": ep_num,
            "episodes": len(m_df),
            "success_rate_pct": sr,
            "clamped_rate_pct": clamp_rate,
            "mean_min_dist_tcp_mm": round(tcp_dist_mean, 2),
            "std_min_dist_tcp_mm": round(tcp_dist_std, 2),
            "mean_terminal_error_obj_mm": round(term_mean, 2),
            "std_terminal_error_obj_mm": round(term_std, 2),
            "auto_stop_rate_pct": auto_stop_rate,
        })

        if sr > max_sr:
            max_sr = sr
            effective_epoch = ep_num
        elif sr >= max_sr and effective_epoch is None:
            effective_epoch = ep_num

    print("="*85)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "checkpoint_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"📊 Tabla de resumen guardada en: {summary_csv}")

    # Generar gráficos científicos de calibración
    plot_checkpoint_calibration(
        df=df,
        out_dir=out_dir,
        epoch_metrics_path=run_dir / "epoch_metrics.csv",
        stopping_summary_path=run_dir / "convergence" / "convergence_summary.csv",
    )

    # Exportar recomendación formal de calibración en JSON
    calibration_report = {
        "dataset": "demonstrations_v8.zarr",
        "eval_episodes_per_checkpoint": args.episodes,
        "checkpoints_evaluated": [c[0] for c in available_ckpts],
        "effective_epoch_recommended": effective_epoch,
        "max_success_rate_pct": max_sr,
        "conclusion": (
            f"La cantidad efectiva de épocas para alcanzar una tasa favorable de éxito ({max_sr:.1f}%) "
            f"es de {effective_epoch} épocas. Más allá de esta cota, la ganancia física se satura "
            f"y los criterios de parada asintótica pueden calibrarse para detener el entrenamiento en dicho punto."
        ),
    }
    with open(out_dir / "calibration_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)

    print(f"\n💡 [Conclusión]: Cantidad Efectiva de Épocas recomendada = {effective_epoch} épocas (Tasa de Éxito: {max_sr:.1f}%)")
    print(f"📄 Reporte de calibración exportado a: {out_dir / 'calibration_report.json'}")

    executor.shutdown()
    runner.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
