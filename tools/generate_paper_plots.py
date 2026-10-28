#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Generador de figuras de alta resolución (PNG 300 DPI y PDF vectorial) para
artículos de investigación científica (IEEE Transactions / Robotics Conferences).
Compara métricas entre Solo DP y DP + RL, y entre controladores demostradores.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Paleta científica accesible (Colorblind-friendly)
COLORS = {
    "SOLO_DP": "#1f77b4",        # Azul sobrio
    "DP_PLUS_RL": "#2ca02c",     # Verde éxito
    "PID_PILOT": "#ff7f0e",      # Naranja
    "MPC_CUROBO": "#9467bd",     # Púrpura
    "CARTESIAN_IMPEDANCE": "#8c564b", # Marrón
}


def load_episode_summary(csv_path: Path) -> List[Dict]:
    """Carga los datos de resumen de episodios."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "episode_idx": int(r["episode_idx"]),
                "task_completed": bool(int(r["task_completed"])),
                "duration_s": float(r["duration_s"]),
                "mean_tracking_error_m": float(r["mean_tracking_error_m"]),
                "terminal_error_mm": float(r["terminal_error_mm"]),
                "slip_occurred": bool(int(r["slip_occurred"])),
            })
    return rows


def plot_paper_comparison(
    exp_dp_dir: Path,
    exp_rl_dir: Path,
    out_dir: Path,
    exp_label_dp: str = "Solo Diffusion Policy",
    exp_label_rl: str = "Diffusion Policy + RL (DPPO)",
) -> None:
    """
    Genera un panel comparativo 2x2 formal listo para paper que contrasta
    Solo DP frente a DP + RL en:
    1. Tasa de Éxito (%)
    2. Error Terminal de Entrega (mm)
    3. Error Cuadrático Medio de Seguimiento (mm)
    4. Tiempo de Ciclo de Episodio (s)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_episodes = load_episode_summary(exp_dp_dir / "episode_summary.csv")
    rl_episodes = load_episode_summary(exp_rl_dir / "episode_summary.csv")

    if not dp_episodes or not rl_episodes:
        print(f"[WARN] Datos insuficientes para generar comparativa: DP={len(dp_episodes)}, RL={len(rl_episodes)}")
        return

    # Extraer métricas
    sr_dp = np.mean([e["task_completed"] for e in dp_episodes]) * 100.0
    sr_rl = np.mean([e["task_completed"] for e in rl_episodes]) * 100.0

    term_err_dp = [e["terminal_error_mm"] for e in dp_episodes]
    term_err_rl = [e["terminal_error_mm"] for e in rl_episodes]

    track_err_dp = [e["mean_tracking_error_m"] * 1000.0 for e in dp_episodes]
    track_err_rl = [e["mean_tracking_error_m"] * 1000.0 for e in rl_episodes]

    dur_dp = [e["duration_s"] for e in dp_episodes]
    dur_rl = [e["duration_s"] for e in rl_episodes]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Evaluación Experimental Comparativa: Solo DP vs. DP + RL (DPPO)", fontsize=15, fontweight="bold")

    # Subplot 1: Tasa de Éxito (Success Rate)
    labels = [exp_label_dp, exp_label_rl]
    srs = [sr_dp, sr_rl]
    bars = axs[0, 0].bar(labels, srs, color=[COLORS["SOLO_DP"], COLORS["DP_PLUS_RL"]], width=0.55, edgecolor="black", alpha=0.85)
    axs[0, 0].set_ylabel("Tasa de Éxito (%)", fontweight="bold")
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Subplot 2: Error Terminal (mm) - Boxplot
    box = axs[0, 1].boxplot(
        [term_err_dp, term_err_rl],
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for patch, color in zip(box["boxes"], [COLORS["SOLO_DP"], COLORS["DP_PLUS_RL"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axs[0, 1].set_ylabel("Error Terminal de Entrega (mm)", fontweight="bold")
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 3: Tracking Error (mm) - Violin Plot
    violin = axs[1, 0].violinplot([track_err_dp, track_err_rl], showmeans=True, showextrema=True)
    for pc, color in zip(violin["bodies"], [COLORS["SOLO_DP"], COLORS["DP_PLUS_RL"]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
    axs[1, 0].set_xticks([1, 2])
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].set_ylabel("RMSE de Seguimiento Cartesiano (mm)", fontweight="bold")
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 4: Duración de Episodio (s)
    box_dur = axs[1, 1].boxplot(
        [dur_dp, dur_rl],
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(box_dur["boxes"], [COLORS["SOLO_DP"], COLORS["DP_PLUS_RL"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axs[1, 1].set_ylabel("Duración de Episodio (s)", fontweight="bold")
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    out_png = out_dir / "paper_comparison_dp_vs_rl.png"
    out_pdf = out_dir / "paper_comparison_dp_vs_rl.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Figuras de publicación generadas en:")
    print(f"   • PNG 300 DPI: {out_png}")
    print(f"   • PDF Vectorial: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de figuras de publicación para papers")
    parser.add_argument("--dp_dir", type=str, required=True, help="Directorio de resultados del experimento Solo DP")
    parser.add_argument("--rl_dir", type=str, required=True, help="Directorio de resultados del experimento DP + RL")
    parser.add_argument("--out_dir", type=str, default="experiments/figures_paper", help="Directorio de salida para figuras")
    args = parser.parse_args()

    plot_paper_comparison(Path(args.dp_dir), Path(args.rl_dir), Path(args.out_dir))
