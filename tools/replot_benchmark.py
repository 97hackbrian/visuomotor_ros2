#!/usr/bin/env python3
"""
Script utilitario para re-graficar el panel de benchmark comparativo 2x3
directamente desde benchmark_episodes.csv sin requerir simulación.
Diseño de publicación científica: maneja outliers sin aplastar las cajas (boxplots),
incorpora escala logarítmica para distancia TCP, límites enfocados con badges para
outliers y superposición de puntos dispersos (jitter scatter).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

COLORS = {
    "Solo DP": "#1f77b4",          # Azul clásico
    "DP + RL (Iter 100)": "#2ca02c", # Verde éxito
}


def replot(csv_path: Path, out_dir: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_data = df[df["model"] == "Solo DP"]
    rl_data = df[df["model"] == "DP + RL (Iter 100)"]

    sr_dp = dp_data["task_success"].mean() * 100.0
    sr_rl = rl_data["task_success"].mean() * 100.0
    clamp_dp = dp_data["is_clamped"].mean() * 100.0
    clamp_rl = rl_data["is_clamped"].mean() * 100.0

    np.random.seed(42)  # Jitter reproducible
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"Evaluación Experimental Comparativa (N = {len(dp_data)} episodios pareados por modelo)", fontsize=16, fontweight="bold")

    labels = ["Solo DP", "DP + RL (Iter 100)"]

    # =========================================================================
    # A) Tasa de Éxito Global y Agarre Firme (%)
    # =========================================================================
    x = np.arange(len(labels))
    w = 0.35
    b1 = axs[0, 0].bar(x - w/2, [sr_dp, sr_rl], w, label="Tasa de Éxito (%)", color=[COLORS["Solo DP"], COLORS["DP + RL (Iter 100)"]], alpha=0.9, edgecolor="black")
    b2 = axs[0, 0].bar(x + w/2, [clamp_dp, clamp_rl], w, label="Agarre Firme (%)", color=["#aec7e8", "#98df8a"], alpha=0.9, edgecolor="black", hatch="//")
    axs[0, 0].set_ylabel("Porcentaje (%)", fontweight="bold")
    axs[0, 0].set_ylim(0, 118)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, fontweight="bold")
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.6)
    axs[0, 0].legend(loc="upper right", framealpha=0.9)
    axs[0, 0].set_title("A) Tasa de Éxito y Agarre Firme (%)", fontweight="bold")
    for b in b1:
        axs[0, 0].text(b.get_x() + b.get_width()/2.0, b.get_height() + 2.0, f"{b.get_height():.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for b in b2:
        axs[0, 0].text(b.get_x() + b.get_width()/2.0, b.get_height() + 2.0, f"{b.get_height():.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # =========================================================================
    # B) Mínima Distancia TCP a Cubo (mm) - Escala Logarítmica
    # =========================================================================
    b_vals = [dp_data["min_dist_tcp_obj_mm"].values, rl_data["min_dist_tcp_obj_mm"].values]
    box_b = axs[0, 1].boxplot(b_vals, labels=labels, patch_artist=True, showmeans=True,
                               medianprops=dict(color="black", linewidth=2.0),
                               meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=7))
    box_b["boxes"][0].set_facecolor(COLORS["Solo DP"])
    box_b["boxes"][1].set_facecolor(COLORS["DP + RL (Iter 100)"])
    for b in box_b["boxes"]:
        b.set_alpha(0.75)
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_ylim(0.4, 200)
    axs[0, 1].set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    axs[0, 1].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    for idx, vals in enumerate(b_vals, start=1):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        axs[0, 1].scatter(idx + jitter, vals, color="black", alpha=0.65, s=35, zorder=5, edgecolor="white", linewidth=0.5)
    axs[0, 1].set_ylabel("Mínima Distancia TCP a Cubo (mm) [Escala Log]", fontweight="bold")
    axs[0, 1].set_title("B) Precisión de Aproximación: Contacto TCP (mm)", fontweight="bold")
    axs[0, 1].grid(True, which="both", linestyle="--", alpha=0.5)

    # =========================================================================
    # C) Error Terminal de Entrega en Home (mm) - Enfoque en rango activo + Outlier Badge
    # =========================================================================
    c_vals = [dp_data["terminal_error_obj_mm"].values, rl_data["terminal_error_obj_mm"].values]
    box_c = axs[0, 2].boxplot(c_vals, labels=labels, patch_artist=True, showmeans=True, showfliers=False,
                               medianprops=dict(color="black", linewidth=2.0),
                               meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=7))
    box_c["boxes"][0].set_facecolor(COLORS["Solo DP"])
    box_c["boxes"][1].set_facecolor(COLORS["DP + RL (Iter 100)"])
    for b in box_c["boxes"]:
        b.set_alpha(0.75)
    axs[0, 2].set_ylim(105, 172)
    for idx, vals in enumerate(c_vals, start=1):
        in_range = [v for v in vals if v <= 165]
        jitter = np.random.normal(0, 0.04, size=len(in_range))
        axs[0, 2].scatter(idx + jitter, in_range, color="black", alpha=0.65, s=35, zorder=5, edgecolor="white", linewidth=0.5)
    # Badge explicativo del outlier
    axs[0, 2].text(0.5, 0.94, "▲ Outlier Ep 3: 390.7 mm (sin agarre, cubo en mesa)", transform=axs[0, 2].transAxes,
                   ha="center", va="top", fontsize=9.5, fontweight="bold", color="#d62728",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0", edgecolor="#d62728", linewidth=1.2, alpha=0.95))
    axs[0, 2].set_ylabel("Error Terminal de Entrega (mm)", fontweight="bold")
    axs[0, 2].set_title("C) Precisión Terminal en Destino Home (mm)", fontweight="bold")
    axs[0, 2].grid(True, linestyle="--", alpha=0.6)

    # =========================================================================
    # D) Fidelidad Cinemática Cartesiana / RMSE (mm)
    # =========================================================================
    d_vals = [dp_data["tracking_rmse_mm"].values, rl_data["tracking_rmse_mm"].values]
    box_d = axs[1, 0].boxplot(d_vals, labels=labels, patch_artist=True, showmeans=True,
                               medianprops=dict(color="black", linewidth=2.0),
                               meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=7))
    box_d["boxes"][0].set_facecolor(COLORS["Solo DP"])
    box_d["boxes"][1].set_facecolor(COLORS["DP + RL (Iter 100)"])
    for b in box_d["boxes"]:
        b.set_alpha(0.75)
    for idx, vals in enumerate(d_vals, start=1):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        axs[1, 0].scatter(idx + jitter, vals, color="black", alpha=0.65, s=35, zorder=5, edgecolor="white", linewidth=0.5)
    axs[1, 0].set_ylabel("Error RMSE (mm)", fontweight="bold")
    axs[1, 0].set_title("D) Fidelidad Cinemática RMSE (mm)", fontweight="bold")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # =========================================================================
    # E) Elevación Vertical del Objeto ΔZ (cm) - Enfoque en vuelos exitosos + Outlier Badge
    # =========================================================================
    e_vals = [dp_data["delta_z_cm"].values, rl_data["delta_z_cm"].values]
    box_e = axs[1, 1].boxplot(e_vals, labels=labels, patch_artist=True, showmeans=True, showfliers=False,
                               medianprops=dict(color="black", linewidth=2.0),
                               meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=7))
    box_e["boxes"][0].set_facecolor(COLORS["Solo DP"])
    box_e["boxes"][1].set_facecolor(COLORS["DP + RL (Iter 100)"])
    for b in box_e["boxes"]:
        b.set_alpha(0.75)
    axs[1, 1].set_ylim(21, 33)
    for idx, vals in enumerate(e_vals, start=1):
        in_range = [v for v in vals if v >= 20]
        jitter = np.random.normal(0, 0.04, size=len(in_range))
        axs[1, 1].scatter(idx + jitter, in_range, color="black", alpha=0.65, s=35, zorder=5, edgecolor="white", linewidth=0.5)
    # Badge explicativo del outlier
    axs[1, 1].text(0.5, 0.06, "▼ Outlier Ep 3: 0.0 cm (fallo de agarre)", transform=axs[1, 1].transAxes,
                   ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#d62728",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0", edgecolor="#d62728", linewidth=1.2, alpha=0.95))
    axs[1, 1].set_ylabel(r"Elevación Vertical $\Delta Z$ (cm)", fontweight="bold")
    axs[1, 1].set_title(r"E) Elevación Vertical del Objeto $\Delta Z$ (cm)", fontweight="bold")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    # =========================================================================
    # F) Duración de Ciclo por Episodio (s) - Enfoque en tareas concluidas + Outlier Badge
    # =========================================================================
    f_vals = [dp_data["duration_s"].values, rl_data["duration_s"].values]
    box_f = axs[1, 2].boxplot(f_vals, labels=labels, patch_artist=True, showmeans=True, showfliers=False,
                               medianprops=dict(color="black", linewidth=2.0),
                               meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=7))
    box_f["boxes"][0].set_facecolor(COLORS["Solo DP"])
    box_f["boxes"][1].set_facecolor(COLORS["DP + RL (Iter 100)"])
    for b in box_f["boxes"]:
        b.set_alpha(0.75)
    axs[1, 2].set_ylim(36, 88)
    for idx, vals in enumerate(f_vals, start=1):
        in_range = [v for v in vals if v <= 85]
        jitter = np.random.normal(0, 0.04, size=len(in_range))
        axs[1, 2].scatter(idx + jitter, in_range, color="black", alpha=0.65, s=35, zorder=5, edgecolor="white", linewidth=0.5)
    # Badge explicativo de los timeouts
    axs[1, 2].text(0.5, 0.94, "▲ Timeouts: Ep 3 (116s) y Ep 9 (121s)", transform=axs[1, 2].transAxes,
                   ha="center", va="top", fontsize=9.5, fontweight="bold", color="#d62728",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0", edgecolor="#d62728", linewidth=1.2, alpha=0.95))
    axs[1, 2].set_ylabel("Tiempo de Ciclo (s)", fontweight="bold")
    axs[1, 2].set_title("F) Duración de Ciclo por Episodio (s)", fontweight="bold")
    axs[1, 2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    png_path = out_dir / "benchmark_comparison_plot.png"
    pdf_path = out_dir / "benchmark_comparison_plot.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Figuras optimizadas con calidad de publicación en:\n   • {png_path}\n   • {pdf_path}")


if __name__ == "__main__":
    csv_file = Path("/ros/visuomotor_core/outputs/benchmark_comparison/benchmark_episodes.csv")
    out_dir = Path("/ros/visuomotor_core/outputs/benchmark_comparison")
    replot(csv_file, out_dir)
