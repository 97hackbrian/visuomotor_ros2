"""Figuras de entrenamiento (en inglés, backend headless Agg, PNG 300 DPI + PDF vectorial).

Regla del repo: eje X inferior en pasos de optimización y superior en épocas (1 época = N pasos), y el nombre del
dataset siempre parametrizado (nunca fijo en el código).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paleta categórica validada (skill dataviz, orden fijo): azul, naranja, aqua; gris para referencias.
C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linestyle": "-", "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False})


def _epoch_axis(ax, steps_per_epoch: int):
    if steps_per_epoch > 0:
        sec = ax.secondary_xaxis("top", functions=(lambda s: s / float(steps_per_epoch),
                                                   lambda e: e * float(steps_per_epoch)))
        sec.set_xlabel(f"Epoch (1 epoch = {steps_per_epoch:,} steps)", fontsize=9)


def _save(fig, out_base: Path, dpi: int = 300):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(out_dir: Path, step_history: List[Dict], epoch_history: List[Dict],
                         steps_per_epoch: int, dataset_name: str = "", ema_history: List[Dict] | None = None,
                         dpi: int = 300):
    if not step_history:
        return
    steps = np.array([r["step"] for r in step_history])
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    title = "Supervised training dynamics: Diffusion Policy (DDIM + 1D U-Net)"
    if dataset_name:
        title += f" on {dataset_name}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax = axs[0, 0]
    ax.plot(steps, [r["loss"] for r in step_history], color=C1, alpha=0.25, lw=1.0, label="Batch loss")
    ax.plot(steps, [r["loss_ema"] for r in step_history], color=C2, lw=2.0, label="Loss EMA (α=0.05)")
    ax.set_yscale("log")
    ax.set_title("Denoising loss per optimization step", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("MSE loss (log)")
    ax.grid(True)
    ax.legend(loc="upper right")
    _epoch_axis(ax, steps_per_epoch)

    ax = axs[0, 1]
    if epoch_history:
        ax.plot([r["epoch"] for r in epoch_history], [r["avg_loss"] for r in epoch_history], color=C1,
                marker="o", ms=3, lw=1.8, label="Mean train loss / epoch")
        if any("val_loss" in r and np.isfinite(r.get("val_loss", np.nan)) for r in epoch_history):
            ev = [r for r in epoch_history if np.isfinite(r.get("val_loss", np.nan))]
            ax.plot([r["epoch"] for r in ev], [r["val_loss"] for r in ev], color=C2, marker="s", ms=4,
                    lw=1.8, label="Held-out denoising loss (EMA weights)")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
    ax.set_title("Epoch-level convergence", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log)")
    ax.grid(True)

    ax = axs[1, 0]
    ax.plot(steps, [r["lr"] for r in step_history], color=C1, lw=2.0, label="Learning rate")
    # (la EMA decay queda en training_metrics.csv; no se superpone con un segundo eje Y)
    ax.set_title("Learning-rate schedule (warmup + cosine)", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.grid(True)
    ax.legend(loc="upper right")

    ax = axs[1, 1]
    ax.plot(steps, [r["vram_peak_mb"] for r in step_history], color=C1, lw=1.8, label="Peak VRAM (MB)")
    ax.set_title("GPU memory (peak allocated)", fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("MB")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save(fig, Path(out_dir) / "losses" / "training_curves", dpi)


_BASELINES = {"ade_mm": ("hold_ade_mm", "cv_ade_mm"), "ade_exec_mm": ("hold_ade_exec_mm", "cv_ade_exec_mm"),
              "fde_mm": ("hold_fde_mm", "cv_fde_mm")}


def _finite(v) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def plot_eval_curves(out_dir: Path, eval_history: List[Dict], steps_per_epoch: int, dataset_name: str = "",
                     dpi: int = 300):
    """Métricas de muestreo completo por checkpoint (held-out vs. subconjunto de entrenamiento)."""
    if not eval_history:
        return
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    title = "Offline policy evaluation (full DDIM sampling, EMA weights)"
    if dataset_name:
        title += f" — {dataset_name}"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    splits = {"train": (C1, "Train subset"), "val": (C2, "Held-out episodes")}
    panels = [
        ("ade_mm", "A) Average displacement error (horizon)", "ADE (mm)"),
        ("ade_exec_mm", "B) ADE over executed steps", "ADE exec. (mm)"),
        ("fde_mm", "C) Final displacement error (t+H-1)", "FDE (mm)"),
        ("rot_deg", "D) Mean geodesic orientation error", "Rotation error (deg)"),
        ("done_f1", "E) Done-flag F1 (first executed step)", "F1"),
        ("progress_ratio", "F) Progress ratio (pred./demo displacement)", "Ratio"),
    ]
    for ax, (key, ttl, ylab) in zip(axs.ravel(), panels):
        for split, (color, label) in splits.items():
            rows = [r for r in eval_history if r["split"] == split and key in r]
            if rows:
                ax.plot([r["step"] for r in rows], [r[key] for r in rows], color=color, marker="o", ms=4, lw=1.8,
                        label=label)
        if key in _BASELINES:
            # Líneas base sobre las mismas ventanas held-out (no dependen del checkpoint: se usa la última)
            for bk, ls, color, label in zip(_BASELINES[key], ("--", "-."), (GRAY, INK),
                                            ("Baseline: hold current pose (held-out)",
                                             "Baseline: constant velocity (held-out)")):
                rows = [r for r in eval_history if r["split"] == "val" and _finite(r.get(bk))]
                if rows:
                    ax.axhline(float(rows[-1][bk]), color=color, ls=ls, lw=1.5, label=label)
        if key == "progress_ratio":
            ax.axhline(1.0, color=GRAY, ls="--", lw=1.5, label="Ideal")
        ax.set_title(ttl, fontweight="bold")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend(loc="best", fontsize=9)
        _epoch_axis(ax, steps_per_epoch)
    plt.tight_layout()
    _save(fig, Path(out_dir) / "evaluation" / "offline_eval_curves", dpi)
