#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Recreador de curvas de Dinámica de Entrenamiento RL (DPPO):
Extrae el historial de entrenamiento de dppo_v7_rl_finetuned desde iter_1 a iter_100,
destacando los puntos donde se guardaron los checkpoints periódicos (iter_10 a iter_100).
Genera figuras formales a 300 DPI y PDF vectorial para el informe de avance.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def recreate_curves(csv_path: Path, out_dir: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts or parts[0] == "iteration":
                continue
            try:
                itr = int(parts[0])
                elapsed = float(parts[1])
                reward = float(parts[2])
                if len(parts) == 9:
                    p_loss = float(parts[4])
                    c_loss = float(parts[5])
                    kl = float(parts[6])
                else:
                    p_loss = float(parts[3])
                    c_loss = float(parts[4])
                    kl = float(parts[5])
                rows.append({
                    "iteration": itr,
                    "elapsed": elapsed,
                    "reward": reward,
                    "policy_loss": p_loss,
                    "critic_loss": c_loss,
                    "kl": kl
                })
            except Exception:
                pass

    # Tomar las primeras 100 iteraciones del entrenamiento continuo
    df_100 = rows[:100]
    itrs = [r["iteration"] for r in df_100]
    rewards = [r["reward"] for r in df_100]
    p_losses = [r["policy_loss"] for r in df_100]
    c_losses = [r["critic_loss"] for r in df_100]
    kls = [r["kl"] for r in df_100]

    checkpoint_itrs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dinámica de Entrenamiento DPPO RL (Fine-Tuning Online: iter_1 a iter_100)", fontsize=16, fontweight="bold")

    # 1. Retorno Episódico (Reward)
    axs[0, 0].plot(itrs, rewards, color="#1f77b4", marker="o", markersize=3, alpha=0.5, label="Reward Medio por Iteración")
    if len(rewards) >= 5:
        smooth = np.convolve(rewards, np.ones(5)/5, mode="valid")
        axs[0, 0].plot(itrs[4:], smooth, color="#d62728", linewidth=2.2, label="Media Móvil (5 iters)")
    # Destacar checkpoints periódicos
    ckpt_rewards = [rewards[i-1] for i in checkpoint_itrs if i <= len(rewards)]
    axs[0, 0].scatter(checkpoint_itrs, ckpt_rewards, color="#9467bd", s=50, zorder=5, label="Checkpoints Exportados (iter_10..100)")
    axs[0, 0].set_title("A) Retorno Episódico Acumulado ($R$)", fontweight="bold")
    axs[0, 0].set_xlabel("Iteración de RL")
    axs[0, 0].set_ylabel("Recompensa Total")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend(loc="lower right", fontsize=9)

    # 2. Pérdida del Crítico V(s)
    axs[0, 1].plot(itrs, c_losses, color="#ff7f0e", marker="s", markersize=3, alpha=0.7, label="Critic Loss (MSE)")
    if len(c_losses) >= 5:
        smooth_c = np.convolve(c_losses, np.ones(5)/5, mode="valid")
        axs[0, 1].plot(itrs[4:], smooth_c, color="#b25b08", linewidth=2.0, label="Tendencia Filtrada")
    axs[0, 1].set_title("B) Pérdida de Estimación del Crítico $V_\\phi(s)$", fontweight="bold")
    axs[0, 1].set_xlabel("Iteración de RL")
    axs[0, 1].set_ylabel("Error Cuadrático Medio (MSE)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend(loc="upper right", fontsize=9)

    # 3. Pérdida del Actor (PPO + BC)
    axs[1, 0].plot(itrs, p_losses, color="#2ca02c", marker="^", markersize=3, alpha=0.7, label="Policy Loss (L_CLIP + L_BC)")
    if len(p_losses) >= 5:
        smooth_p = np.convolve(p_losses, np.ones(5)/5, mode="valid")
        axs[1, 0].plot(itrs[4:], smooth_p, color="#145a14", linewidth=2.0, label="Tendencia")
    axs[1, 0].axhline(y=0.0, color="gray", linestyle=":", alpha=0.7)
    axs[1, 0].set_title("C) Pérdida de la Política Generativa (Actor DPPO)", fontweight="bold")
    axs[1, 0].set_xlabel("Iteración de RL")
    axs[1, 0].set_ylabel("Pérdida de Política")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend(loc="upper right", fontsize=9)

    # 4. Divergencia KL Aproximada
    axs[1, 1].plot(itrs, kls, color="#9467bd", marker="d", markersize=3, alpha=0.6, label="Approximate KL Divergence")
    axs[1, 1].axhline(y=0.05, color="red", linestyle="--", linewidth=1.8, label="Target KL Threshold (0.05)")
    axs[1, 1].set_title("D) Desplazamiento Estocástico de Política (Divergencia KL)", fontweight="bold")
    axs[1, 1].set_xlabel("Iteración de RL")
    axs[1, 1].set_ylabel("Divergencia KL")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "rl_training_curves_iter100.png"
    pdf_path = out_dir / "rl_training_curves_iter100.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Curvas de Dinámica de RL recreadas exitosamente:\n   • {png_path}\n   • {pdf_path}")


if __name__ == "__main__":
    csv_file = Path("/ros/visuomotor_core/outputs/dppo_v7_rl_finetuned/rl_training_metrics.csv")
    out_folder = Path("/ros/visuomotor_core/outputs/dppo_v7_rl_finetuned")
    recreate_curves(csv_file, out_folder)
