#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
evaluate_real_dataset.py — Evaluación Científica SOTA de Datasets de Demostraciones Reales
========================================================================================
Estandariza métricas cinemáticas, temporales, de suavidad y de auto-stop para datasets
de demostraciones expertas a mano alzada (VIO RealSense D435i / UMI / Diffusion Policy).

Genera:
1. Panel gráfico SOTA 2x3 en alta resolución (PNG 300 DPI y PDF vectorial).
2. Reporte tabular cuantitativo y archivo JSON estructurado.
3. Registro formal en la base de datos relacional SQLite (experiments/experiments.db).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.ndimage import uniform_filter1d
import zarr

# Asegurar importación de visuomotor_core
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CORE_DIR = REPO_ROOT / "visuomotor_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

try:
    from visuomotor_core.utils.tracker import (
        find_db_path,
        init_db,
        record_dataset,
        record_dataset_episode,
    )
except ImportError:
    record_dataset = None
    record_dataset_episode = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación SOTA de Datasets de Demostraciones Reales")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/demonstrations_real_v7.zarr",
        help="Ruta al dataset Zarr (ej. datasets/demonstrations_real_v7.zarr)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/evaluations_real_v7",
        help="Directorio donde guardar gráficas y reportes",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="Frecuencia canónica de muestreo en Hz (defecto: 20.0)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Desactivar registro en SQLite experiments/experiments.db",
    )
    return parser.parse_args()


def compute_quaternion_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    """Calcula la velocidad angular instantánea (rad/s) entre cuaterniones consecutivos."""
    if len(quats) < 2:
        return np.zeros(1, dtype=np.float32)
    # Producto escalar absoluto para manejar doble cobertura q y -q
    dot_products = np.abs(np.sum(quats[:-1] * quats[1:], axis=1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_diffs = 2.0 * np.arccos(dot_products)
    return angular_diffs / dt


def compute_dimensionless_jerk(pos: np.ndarray, dt: float) -> float:
    """Calcula el Log Dimensionless Jerk (LDLJ) como métrica SOTA de suavidad motora."""
    n = len(pos)
    if n < 4:
        return 0.0
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    
    total_time = n * dt
    path_length = np.sum(np.linalg.norm(vel, axis=1) * dt)
    if path_length < 1e-4:
        return 0.0
    
    integrated_squared_jerk = np.sum(np.sum(jerk ** 2, axis=1) * dt)
    # Dimensionless Jerk clásico: -(T^3 / L^2) * int(jerk^2 dt)
    dj = (total_time ** 3 / (path_length ** 2)) * integrated_squared_jerk
    return float(np.log(max(dj, 1e-6)))


def draw_confidence_ellipse(ax, x, y, n_std=2.0, facecolor='none', **kwargs):
    """Dibuja un elipsoide de covarianza de n_std desviaciones típicas."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    import matplotlib.transforms as transforms
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main():
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dt = 1.0 / args.rate_hz
    dataset_name = Path(dataset_path).stem
    
    print("=" * 80)
    print(f"🔬 EVALUACIÓN SOTA DE DATASET DE DEMOSTRACIONES: {dataset_name}")
    print(f"📁 Ruta: {dataset_path}")
    print(f"⏱ Frecuencia: {args.rate_hz} Hz (dt = {dt*1000:.1f} ms)")
    print("=" * 80)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] El dataset no existe: {dataset_path}")
        sys.exit(1)
        
    root = zarr.open_group(dataset_path, mode="r")
    ep_keys = sorted(list(root.keys()), key=lambda x: int(x.split("_")[1]) if "_" in x else x)
    total_episodes = len(ep_keys)
    
    if total_episodes == 0:
        print("[ERROR] El dataset está vacío.")
        sys.exit(1)
        
    print(f"✔ Detectados {total_episodes} episodios. Analizando tensores y cinemática...")
    
    episode_records = []
    all_trajectories = []
    all_velocities = []
    all_accelerations = []
    all_jerks = []
    all_durations = []
    all_done_steps = []
    all_terminal_positions = []
    all_start_positions = []
    all_path_lengths = []
    all_tortuosities = []
    all_eff_path_lengths = []
    all_eff_tortuosities = []
    all_displacements = []
    
    for idx, ep_name in enumerate(ep_keys):
        g = root[ep_name]
        state = np.array(g["state"])      # [T, 8]
        action = np.array(g["action"])    # [T, 9]
        n_steps = len(state)
        
        pos = state[:, :3]                # [T, 3] XYZ
        quat = state[:, 3:7]              # [T, 4]
        gripper = state[:, 7]             # [T]
        done_flag = action[:, 8] if action.shape[1] >= 9 else np.zeros(n_steps)
        
        duration_s = n_steps * dt
        diffs = np.diff(pos, axis=0)
        step_distances = np.linalg.norm(diffs, axis=1)
        path_length_m = float(np.sum(step_distances))
        
        # Smooth trajectory for effective physical path length (0.25s / 5 steps window)
        pos_smooth = uniform_filter1d(pos, size=5, axis=0) if len(pos) >= 5 else pos.copy()
        diffs_smooth = np.diff(pos_smooth, axis=0)
        eff_path_length_m = float(np.sum(np.linalg.norm(diffs_smooth, axis=1)))
        
        linear_vel = step_distances / dt  # [T-1]
        v_max = float(np.max(linear_vel)) if len(linear_vel) > 0 else 0.0
        v_mean = float(np.mean(linear_vel)) if len(linear_vel) > 0 else 0.0
        
        linear_acc = np.abs(np.diff(linear_vel)) / dt if len(linear_vel) > 1 else np.zeros(1)
        a_max = float(np.max(linear_acc)) if len(linear_acc) > 0 else 0.0
        
        ang_vel = compute_quaternion_angular_velocity(quat, dt)
        omega_max = float(np.max(ang_vel)) if len(ang_vel) > 0 else 0.0
        omega_mean = float(np.mean(ang_vel)) if len(ang_vel) > 0 else 0.0
        
        ldlj = compute_dimensionless_jerk(pos, dt)
        
        start_pt = pos[0].copy()
        end_pt = pos[-1].copy()
        displacement_m = float(np.linalg.norm(end_pt - start_pt))
        tortuosity = float(path_length_m / max(displacement_m, 1e-4))
        eff_tortuosity = float(eff_path_length_m / max(displacement_m, 1e-4))
        
        done_steps = int(np.sum(done_flag >= 0.5))
        done_duration_s = done_steps * dt
        
        rec = {
            "episode_idx": idx,
            "episode_name": ep_name,
            "steps": n_steps,
            "duration_s": duration_s,
            "path_length_m": path_length_m,
            "effective_path_length_m": eff_path_length_m,
            "displacement_m": displacement_m,
            "tortuosity": tortuosity,
            "effective_tortuosity": eff_tortuosity,
            "v_max_m_s": v_max,
            "v_mean_m_s": v_mean,
            "a_max_m_s2": a_max,
            "omega_max_rad_s": omega_max,
            "omega_mean_rad_s": omega_mean,
            "log_dimensionless_jerk": ldlj,
            "done_steps": done_steps,
            "done_duration_s": done_duration_s,
            "start_xyz": [float(v) for v in start_pt],
            "end_xyz": [float(v) for v in end_pt],
            "xyz_min": [float(v) for v in pos.min(axis=0)],
            "xyz_max": [float(v) for v in pos.max(axis=0)],
        }
        episode_records.append(rec)
        
        all_trajectories.append(pos)
        all_velocities.append(linear_vel)
        all_accelerations.append(a_max)
        all_jerks.append(ldlj)
        all_durations.append(duration_s)
        all_done_steps.append(done_steps)
        all_terminal_positions.append(end_pt)
        all_start_positions.append(start_pt)
        all_path_lengths.append(path_length_m)
        all_tortuosities.append(tortuosity)
        all_eff_path_lengths.append(eff_path_length_m)
        all_eff_tortuosities.append(eff_tortuosity)
        all_displacements.append(displacement_m)

    total_steps = sum(r["steps"] for r in episode_records)
    total_time_s = sum(r["duration_s"] for r in episode_records)
    mean_steps = float(np.mean([r["steps"] for r in episode_records]))
    std_steps = float(np.std([r["steps"] for r in episode_records]))
    
    v_peaks = np.array([r["v_max_m_s"] for r in episode_records])
    v_means = np.array([r["v_mean_m_s"] for r in episode_records])
    term_pos = np.array(all_terminal_positions)
    start_pos = np.array(all_start_positions)
    
    # Bounding Box 3D
    all_xyz = np.concatenate(all_trajectories, axis=0)
    bbox_min = all_xyz.min(axis=0)
    bbox_max = all_xyz.max(axis=0)
    bbox_span = bbox_max - bbox_min
    
    summary_metrics = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "total_duration_s": total_time_s,
        "total_duration_min": total_time_s / 60.0,
        "sampling_rate_hz": args.rate_hz,
        "steps_per_episode": {
            "mean": mean_steps,
            "std": std_steps,
            "median": float(np.median([r["steps"] for r in episode_records])),
            "min": int(min(r["steps"] for r in episode_records)),
            "max": int(max(r["steps"] for r in episode_records)),
        },
        "path_length_m": {
            "mean": float(np.mean(all_path_lengths)),
            "std": float(np.std(all_path_lengths)),
            "min": float(np.min(all_path_lengths)),
            "max": float(np.max(all_path_lengths)),
        },
        "net_displacement_m": {
            "mean": float(np.mean(all_displacements)),
            "std": float(np.std(all_displacements)),
            "min": float(np.min(all_displacements)),
            "max": float(np.max(all_displacements)),
        },
        "effective_path_length_m": {
            "mean": float(np.mean(all_eff_path_lengths)),
            "std": float(np.std(all_eff_path_lengths)),
            "min": float(np.min(all_eff_path_lengths)),
            "max": float(np.max(all_eff_path_lengths)),
        },
        "effective_tortuosity": {
            "mean": float(np.mean(all_eff_tortuosities)),
            "std": float(np.std(all_eff_tortuosities)),
            "median": float(np.median(all_eff_tortuosities)),
        },
        "tortuosity_index": {
            "mean": float(np.mean(all_tortuosities)),
            "std": float(np.std(all_tortuosities)),
            "median": float(np.median(all_tortuosities)),
        },
        "linear_velocity_m_s": {
            "peak_max": float(np.max(v_peaks)),
            "peak_mean": float(np.mean(v_peaks)),
            "overall_mean": float(np.mean(v_means)),
            "overall_std": float(np.std(v_means)),
        },
        "acceleration_max_m_s2": {
            "mean": float(np.mean(all_accelerations)),
            "max": float(np.max(all_accelerations)),
        },
        "smoothness_ldlj": {
            "mean": float(np.mean(all_jerks)),
            "std": float(np.std(all_jerks)),
        },
        "done_signal_floor": {
            "compliance_rate_pct": float(100.0 * np.sum(np.array(all_done_steps) >= 40) / total_episodes),
            "min_done_frames": int(np.min(all_done_steps)),
            "mean_done_frames": float(np.mean(all_done_steps)),
            "min_done_duration_s": float(np.min(all_done_steps) * dt),
            "mean_done_duration_s": float(np.mean(all_done_steps) * dt),
        },
        "workspace_bounding_box_m": {
            "x_min": float(bbox_min[0]),
            "x_max": float(bbox_max[0]),
            "x_span": float(bbox_span[0]),
            "y_min": float(bbox_min[1]),
            "y_max": float(bbox_max[1]),
            "y_span": float(bbox_span[1]),
            "z_min": float(bbox_min[2]),
            "z_max": float(bbox_max[2]),
            "z_span": float(bbox_span[2]),
        },
    }
    
    # =========================================================================
    # GENERACIÓN DE GRÁFICAS SOTA (Panel 2x3 en 300 DPI y PDF)
    # =========================================================================
    print("🎨 Renderizando panel científico multicanal 2x3...")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    
    # 1. Top-Left: Trayectorias 3D Proyectadas en Plano XY (Planta y Dispersión)
    ax1 = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for i, traj in enumerate(all_trajectories):
        color = cmap(i / max(total_episodes - 1, 1))
        ax1.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.45, linewidth=1.2)
    # Start and terminal points
    ax1.scatter(start_pos[:, 0], start_pos[:, 1], color="blue", s=25, alpha=0.8, label=f"Start Pose (t=0, N={total_episodes})", zorder=5)
    ax1.scatter(term_pos[:, 0], term_pos[:, 1], color="crimson", marker="x", s=40, alpha=0.85, label=f"Target Hold (done=1, N={total_episodes})", zorder=6)
    draw_confidence_ellipse(ax1, term_pos[:, 0], term_pos[:, 1], n_std=2.0, edgecolor="red", linestyle="--", linewidth=1.5, label="95% Grasp Target Covariance")
    ax1.set_xlabel("Cartesian X (m)")
    ax1.set_ylabel("Cartesian Y (m)")
    ax1.set_title("A. Horizontal Trajectories & Spatial Dispersion (XY Plane)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)
    
    # 2. Top-Center: Perfiles Temporales de Velocidad Cartesiana v(t)
    ax2 = axes[0, 1]
    max_len = max(len(v) for v in all_velocities)
    interp_times = np.linspace(0, max_len * dt, 200)
    interp_vels = []
    for v in all_velocities:
        t_arr = np.arange(len(v)) * dt
        ax2.plot(t_arr, v, color="gray", alpha=0.15, linewidth=0.8)
        if len(v) > 2:
            v_interp = np.interp(interp_times, t_arr, v, right=0.0)
            interp_vels.append(v_interp)
            
    if interp_vels:
        v_mat = np.array(interp_vels)
        mean_profile = np.mean(v_mat, axis=0)
        std_profile = np.std(v_mat, axis=0)
        ax2.plot(interp_times, mean_profile, color="#1f77b4", linewidth=2.5, label=r"Mean Velocity $\mu(t)$")
        ax2.fill_between(interp_times, np.maximum(0, mean_profile - std_profile), mean_profile + std_profile,
                         color="#1f77b4", alpha=0.25, label=r"$\pm 1\sigma$ Confidence Band")
    ax2.axhline(0.20, color="orange", linestyle="--", linewidth=1.5, label="Nominal Calibration Velocity (0.20 m/s)")
    ax2.axhline(0.35, color="red", linestyle=":", linewidth=1.5, label="Operational Velocity Limit (0.35 m/s)")
    ax2.set_xlabel("Demonstration Time (s)")
    ax2.set_ylabel("Linear Velocity (m/s)")
    ax2.set_title(r"B. Cartesian Velocity Dynamics $\mathbf{v}(t)$ @ 20 Hz")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)
    
    # 3. Top-Right: Histograma de Pasos y Duración con KDE
    ax3 = axes[0, 2]
    steps_arr = np.array([r["steps"] for r in episode_records])
    dur_arr = np.array([r["duration_s"] for r in episode_records])
    counts, bins, _ = ax3.hist(dur_arr, bins=16, color="#2ca02c", alpha=0.65, edgecolor="black", density=True, label="Empirical Frequency")
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(dur_arr)
        x_kde = np.linspace(dur_arr.min() * 0.9, dur_arr.max() * 1.05, 200)
        ax3.plot(x_kde, kde(x_kde), color="#005a00", linewidth=2.5, label="Gaussian KDE Density")
    except Exception:
        pass
    ax3.axvline(np.mean(dur_arr), color="darkred", linestyle="-", linewidth=2.0, label=f"Mean: {np.mean(dur_arr):.1f} s ({mean_steps:.0f} steps)")
    ax3.axvline(np.median(dur_arr), color="blue", linestyle="--", linewidth=1.5, label=f"Median: {np.median(dur_arr):.1f} s")
    ax3.set_xlabel("Episode Duration (s)")
    ax3.set_ylabel("Probability Density")
    ax3.set_title("C. Temporal Duration and Step Distribution")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)
    
    # 4. Bottom-Left: Physical Path Length & Geometric Efficiency (Option A)
    ax4 = axes[1, 0]
    box_data = [
        np.array(all_displacements),
        np.array(all_eff_path_lengths),
    ]
    labels = [
        f"Net Displacement $D$\n(μ = {np.mean(all_displacements):.2f} m)",
        f"Effective Path $L_{{eff}}$\n(μ = {np.mean(all_eff_path_lengths):.2f} m)",
    ]
    colors = ["#1f77b4", "#2ca02c"]
    bp = ax4.boxplot(box_data, labels=labels, patch_artist=True,
                     medianprops=dict(color="darkred", linewidth=2.0),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.5)
        
    for i, d in enumerate(box_data):
        x_jit = np.random.normal(i + 1, 0.04, size=len(d))
        ax4.plot(x_jit, d, "o", color=colors[i], alpha=0.55, markersize=4.0)
        
    mean_tort = float(np.mean(all_eff_tortuosities))
    std_tort = float(np.std(all_eff_tortuosities))
    ax4.text(0.98, 0.95, f"Smooth Tortuosity Index:\n$L_{{eff}} / D = {mean_tort:.2f} \\pm {std_tort:.2f}$\n(Curved reaching arc)",
             transform=ax4.transAxes, fontsize=9.5, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
             
    ax4.set_ylabel("Physical Distance (m)")
    ax4.set_title(f"D. Geometric Efficiency: Net vs. Effective Path (N={total_episodes})")
    ax4.grid(True, linestyle=":", alpha=0.6)
    
    # 5. Bottom-Center: Phase-Space Deceleration Profile (E1 Proposal)
    ax5 = axes[1, 1]
    max_d = float(np.max(all_displacements)) * 1.15
    d_grid = np.linspace(0.0, max_d, 120)
    interp_v_phase = []
    
    for traj in all_trajectories:
        traj_smooth = uniform_filter1d(traj, size=5, axis=0) if len(traj) >= 5 else traj.copy()
        target = traj_smooth[-1]
        dists = np.linalg.norm(traj_smooth - target, axis=1) # Distance to target
        diffs = np.diff(traj_smooth, axis=0)
        step_lens = np.linalg.norm(diffs, axis=1)
        vels = np.append(step_lens / dt, 0.0) # Speed at each step
        
        # Plot individual trajectories with high transparency
        ax5.plot(dists, vels, color="gray", alpha=0.15, linewidth=0.8)
        
        # Monotonic sort for clean interpolation
        s_idx = np.argsort(dists)
        d_unique, u_idx = np.unique(dists[s_idx], return_index=True)
        v_unique = vels[s_idx][u_idx]
        v_interp = np.interp(d_grid, d_unique, v_unique, left=0.0, right=0.0)
        interp_v_phase.append(v_interp)
        
    v_phase_mat = np.array(interp_v_phase)
    mean_v_phase = np.mean(v_phase_mat, axis=0)
    std_v_phase = np.std(v_phase_mat, axis=0)
    
    ax5.plot(d_grid, mean_v_phase, color="#d62728", linewidth=2.5, label=r"Mean Profile $\mu_v(d)$")
    ax5.fill_between(d_grid, np.maximum(0, mean_v_phase - std_v_phase), mean_v_phase + std_v_phase,
                     color="#d62728", alpha=0.22, label=r"$\pm 1\sigma$ Dispersion")
                     
    ax5.axvline(0.05, color="darkgreen", linestyle="--", linewidth=1.5, label=r"Contact Zone ($d \leq 5$ cm)")
    ax5.axhline(0.20, color="orange", linestyle=":", linewidth=1.5, label="Nominal Cruise (0.20 m/s)")
    
    # Invert X axis so motion flows from departure (d ~ 0.65m on left) to arrival (d = 0m on right)
    ax5.set_xlim(max_d, -0.02)
    ax5.set_xlabel(r"Distance to Grasp Target (m) [Departure $\rightarrow$ Arrival]")
    ax5.set_ylabel("Cartesian Velocity (m/s)")
    ax5.set_title(r"E. Phase-Space Deceleration Profile: $\mathbf{v}(d_{target})$")
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(loc="upper left", framealpha=0.9)
    
    # 6. Bottom-Right: Alzado XZ y Perfil de Descenso
    ax6 = axes[1, 2]
    for i, traj in enumerate(all_trajectories):
        color = cmap(i / max(total_episodes - 1, 1))
        ax6.plot(traj[:, 0], traj[:, 2], color=color, alpha=0.45, linewidth=1.2)
    ax6.scatter(start_pos[:, 0], start_pos[:, 2], color="blue", s=25, alpha=0.8, label=f"Start Pose (Z=0, N={total_episodes})", zorder=5)
    ax6.scatter(term_pos[:, 0], term_pos[:, 2], color="crimson", marker="x", s=40, alpha=0.85, label="Terminal Target", zorder=6)
    ax6.set_xlabel("Cartesian X (m)")
    ax6.set_ylabel("Cartesian Z (m)")
    ax6.set_title("F. Vertical Descent Profile (Elevation XZ)")
    ax6.grid(True, linestyle=":", alpha=0.6)
    ax6.legend(loc="lower left", framealpha=0.9)
    
    fig.suptitle(f"SOTA Metrological & Kinematic Benchmark Panel: {dataset_name} ({total_episodes} Episodes, {total_steps:,} Steps @ 20 Hz)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    panel_png = out_dir / f"{dataset_name}_sota_panel.png"
    panel_pdf = out_dir / f"{dataset_name}_sota_panel.pdf"
    plt.savefig(panel_png, dpi=300)
    plt.savefig(panel_pdf)
    plt.close()
    print(f"✔ Gráficas SOTA guardadas en:\n   - {panel_png}\n   - {panel_pdf}")
    
    # =========================================================================
    # EXPORTACIÓN DE MÉTRICAS JSON Y REPORTE MARKDOWN
    # =========================================================================
    json_path = out_dir / f"{dataset_name}_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_metrics,
            "episodes": episode_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"✔ Métricas estructuradas exportadas en: {json_path}")
    
    md_report_path = out_dir / f"{dataset_name}_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# SOTA Evaluation Report: `{dataset_name}`\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Dataset:** `{dataset_path}`  \n")
        f.write(f"**Sensor:** Intel RealSense D435i (Active IR Stereo + IMU) + RTAB-Map VIO  \n")
        f.write(f"**Total Episodes:** {total_episodes}  \n")
        f.write(f"**Total Frames:** {total_steps:,} ({total_time_s/60.0:.2f} minutes @ 20 Hz)  \n\n")
        f.write("## 1. Global Summary Statistics Table\n\n")
        f.write("| Dimension / Metric | Mean Value (μ ± σ) | Median | Range [Min, Max] |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Steps per Episode** | {mean_steps:.1f} ± {std_steps:.1f} | {summary_metrics['steps_per_episode']['median']:.0f} | [{summary_metrics['steps_per_episode']['min']}, {summary_metrics['steps_per_episode']['max']}] |\n")
        f.write(f"| **Temporal Duration (s)** | {mean_steps*dt:.2f}s ± {std_steps*dt:.2f}s | {summary_metrics['steps_per_episode']['median']*dt:.2f}s | [{summary_metrics['steps_per_episode']['min']*dt:.2f}s, {summary_metrics['steps_per_episode']['max']*dt:.2f}s] |\n")
        f.write(f"| **Net Displacement ($D$)** | {summary_metrics['net_displacement_m']['mean']:.3f} m ± {summary_metrics['net_displacement_m']['std']:.3f} m | - | [{summary_metrics['net_displacement_m']['min']:.3f} m, {summary_metrics['net_displacement_m']['max']:.3f} m] |\n")
        f.write(f"| **Effective Path ($L_{{eff}}$)** | {summary_metrics['effective_path_length_m']['mean']:.3f} m ± {summary_metrics['effective_path_length_m']['std']:.3f} m | - | [{summary_metrics['effective_path_length_m']['min']:.3f} m, {summary_metrics['effective_path_length_m']['max']:.3f} m] |\n")
        f.write(f"| **Smooth Tortuosity ($L_{{eff}}/D$)** | {summary_metrics['effective_tortuosity']['mean']:.2f} ± {summary_metrics['effective_tortuosity']['std']:.2f} | {summary_metrics['effective_tortuosity']['median']:.2f} | Direct & smooth reaching |\n")
        f.write(f"| **Mean Cartesian Velocity** | {summary_metrics['linear_velocity_m_s']['overall_mean']:.3f} m/s ± {summary_metrics['linear_velocity_m_s']['overall_std']:.3f} m/s | - | Peak max: {summary_metrics['linear_velocity_m_s']['peak_max']:.3f} m/s |\n")
        f.write(f"| **Smoothness (Log Dim. Jerk)** | {summary_metrics['smoothness_ldlj']['mean']:.2f} ± {summary_metrics['smoothness_ldlj']['std']:.2f} | - | Expert human smoothness |\n")
        f.write(f"| **Terminal Stop Floor (done=1.0)** | {summary_metrics['done_signal_floor']['mean_done_frames']:.1f} steps ({summary_metrics['done_signal_floor']['mean_done_duration_s']:.2f}s) | 40 steps | Min: {summary_metrics['done_signal_floor']['min_done_frames']} steps ({summary_metrics['done_signal_floor']['min_done_duration_s']:.2f}s) |\n")
        f.write(f"| **Auto-Stop Compliance Rate** | **{summary_metrics['done_signal_floor']['compliance_rate_pct']:.1f}%** | 100% | ≥ 40 frames guaranteed |\n\n")
        f.write("## 2. 3D Spatial Bounding Box (Workspace Envelope)\n\n")
        f.write(f"* **X:** [{bbox_min[0]:.3f}, {bbox_max[0]:.3f}] m (Span: {bbox_span[0]:.3f} m)\n")
        f.write(f"* **Y:** [{bbox_min[1]:.3f}, {bbox_max[1]:.3f}] m (Span: {bbox_span[1]:.3f} m)\n")
        f.write(f"* **Z:** [{bbox_min[2]:.3f}, {bbox_max[2]:.3f}] m (Span: {bbox_span[2]:.3f} m)\n\n")
        f.write("## 3. Multichannel SOTA Visual Panel\n\n")
        f.write(f"![Multichannel SOTA Panel]({panel_png.name})\n")
    print(f"✔ Reporte Markdown generado en: {md_report_path}")
    
    # =========================================================================
    # REGISTRO CIENTÍFICO EN SQLITE (experiments/experiments.db)
    # =========================================================================
    if not args.no_db and record_dataset is not None:
        print("💾 Registrando en base de datos relacional SQLite (experiments/experiments.db)...")
        db_path = find_db_path()
        init_db(db_path)
        
        # 1. Registrar dataset consolidado
        record_dataset(
            dataset_id=dataset_name,
            path_str=dataset_path,
            demonstrator_type="Handheld VIO (RealSense D435i + RTAB-Map)",
            total_episodes=total_episodes,
            duration_s=total_time_s,
            has_done_flag=True,
            metadata=summary_metrics,
        )
        
        # 2. Registrar episodios individuales
        for r in episode_records:
            obj_sp = {
                "x": r["end_xyz"][0],
                "y": r["end_xyz"][1],
                "z": r["end_xyz"][2],
                "yaw": 0.0,
            }
            record_dataset_episode(
                dataset_name=dataset_name,
                episode_idx=r["episode_idx"],
                obj_spawn=obj_sp,
                ee_jitter={"dx": r["start_xyz"][0], "dy": r["start_xyz"][1], "droll": 0.0, "dpitch": 0.0, "dyaw": 0.0},
                steps=r["steps"],
                sim_duration_s=r["duration_s"],
                wall_duration_s=r["duration_s"],
                cycle_wall_s=r["duration_s"],
                retry_count=0,
                phase_durations={"hold_s": r["done_duration_s"]},
                metadata=r,
            )
        print(f"✔ Dataset '{dataset_name}' y sus {total_episodes} episodios registrados exitosamente en {db_path}!")
        
    print("\n" + "=" * 80)
    print("✨ EVALUACIÓN CIENTÍFICA SOTA COMPLETADA CON ÉXITO")
    print("=" * 80)


if __name__ == "__main__":
    main()
