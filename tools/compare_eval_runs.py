#!/usr/bin/env python3
"""Compara corridas de entrenamiento por sus métricas offline de muestreo completo (``eval_metrics.csv``).

Figura (inglés, PNG 300 DPI + PDF) con 6 paneles sobre los episodios de validación (held-out): ADE, ADE ejecutado,
FDE, error de orientación, F1 del flag done y razón de progreso, con las líneas base "quedarse quieto" y "velocidad
constante" (esta última solo en corridas evaluadas después de añadirla a ``evaluation.py``). Solo lectura.

Uso (contenedor):
    python3 tools/compare_eval_runs.py --runs visuomotor_core/outputs/run_a visuomotor_core/outputs/run_b \
        --labels "relative_ee (UMI)" "absolute" --out visuomotor_core/outputs/pilot_comparison/pilot_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]  # orden categórico fijo (skill dataviz)
GRAY, INK = "#8a8984", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10, "axes.edgecolor": INK})


def load(run: Path, split: str):
    rows = []
    with open(run / "eval_metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            rows.append({k: (float(v) if v not in ("", "None", "nan") else float("nan")) if k != "split" else v
                         for k, v in r.items() if k != "timestamp"})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="ruta base de salida (sin extensión)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    if len(args.runs) != len(args.labels) or len(args.runs) > len(PALETTE):
        raise SystemExit(f"Se requieren tantas etiquetas como corridas (máx. {len(PALETTE)})")

    data = [load(Path(r), args.split) for r in args.runs]
    meta = json.load(open(Path(args.runs[0]) / "training_meta.json"))
    ds = meta.get("dataset_name", "")
    n_val = len(meta.get("val_episodes", []))
    panels = [("ade_mm", "A) Average displacement error", "ADE (mm)", ("hold_ade_mm", "cv_ade_mm")),
              ("ade_exec_mm", "B) ADE over executed steps", "ADE exec. (mm)", ("hold_ade_exec_mm", "cv_ade_exec_mm")),
              ("fde_mm", "C) Final displacement error", "FDE (mm)", ("hold_fde_mm", "cv_fde_mm")),
              ("rot_deg", "D) Geodesic orientation error", "deg", None),
              ("done_f1", "E) Done-flag F1 (first executed step)", "F1", None),
              ("progress_ratio", "F) Progress ratio (pred./demo)", "ratio", None)]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(args.title or f"Held-out offline evaluation ({n_val} episodes) — {ds}", fontsize=14, fontweight="bold")
    for ax, (key, title, ylab, base) in zip(axs.ravel(), panels):
        for rows, label, color in zip(data, args.labels, PALETTE):
            pts = [(r["epoch"], r[key]) for r in rows if key in r]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", ms=5, lw=2, label=label)
        for bk, ls, color, blabel in zip(base or (), ("--", "-."), (GRAY, INK),
                                         ("Baseline: hold current pose", "Baseline: constant velocity")):
            vals = [r.get(bk) for rows in data for r in rows if r.get(bk) == r.get(bk) and r.get(bk) is not None]
            if vals:
                ax.axhline(vals[-1], color=color, ls=ls, lw=1.4, label=blabel)
        if key == "progress_ratio":
            ax.axhline(1.0, color=GRAY, ls="--", lw=1.4, label="Ideal")
        if key in ("ade_mm", "ade_exec_mm", "fde_mm"):
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend(fontsize=9)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    summary = {label: rows[-1] if rows else {} for label, rows in zip(args.labels, data)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
