#!/usr/bin/env python3
"""Informe final de un modelo entrenado con demostraciones reales para el xArm5 (figura en inglés + MODEL_CARD.md).

Reúne, en solo lectura: curvas de entrenamiento (``training_metrics.csv``), métricas offline sobre episodios de
validación por checkpoint (``eval_metrics.csv``), la reproducción de la ruta de inferencia de la Jetson
(``replay_report.json`` de ``replay_policy_offline.py``, con el barrido umbral x paciencia del auto-stop) y la prueba
ROS 2 en lazo cerrado con el puente simulado (``closed_loop_check.json`` + ``tcp_trace.npy``).

Uso (contenedor):
    python3 tools/xarm5/report_real_model.py --run-dir visuomotor_core/outputs/<run> --epoch 150 \
        --replay <dir>/replay_report.json --closed-loop <dir> --out-dir <dir> [--model-card deploy/jetson/model/MODEL_CARD.md]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10, "axes.edgecolor": INK})


def read_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epoch", type=int, required=True, help="época del checkpoint seleccionado")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--closed-loop", default=None, help="directorio de check_policy_closed_loop_sim.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-card", default=None)
    ap.add_argument("--autostop-threshold", type=float, default=None)
    ap.add_argument("--autostop-patience", type=int, default=None)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    meta = json.load(open(run / "training_meta.json"))
    spe = int(meta["steps_per_epoch"])
    steps = read_csv(run / "training_metrics.csv")
    evals = read_csv(run / "eval_metrics.csv")
    rep = json.load(open(args.replay))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = meta.get("dataset_name", "")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"Real-robot model report — {run.name} (checkpoint epoch {args.epoch}, {ds})",
                 fontsize=14, fontweight="bold")

    # A) pérdida
    ax = axs[0, 0]
    s = np.array([fnum(r["step"]) for r in steps])
    ax.plot(s, [fnum(r["loss"]) for r in steps], color=C1, alpha=0.25, lw=0.8, label="Batch loss")
    ax.plot(s, [fnum(r["loss_ema"]) for r in steps], color=C2, lw=1.8, label="Loss EMA (α=0.05)")
    ax.axvline(args.epoch * spe, color=GRAY, ls="--", lw=1.2, label="Selected checkpoint")
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Denoising MSE (log)")
    ax.set_title("A) Training loss", fontweight="bold")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / spe, lambda e: e * spe))
    sec.set_xlabel(f"Epoch (1 epoch = {spe} steps)", fontsize=9)
    ax.grid(True)
    ax.legend(fontsize=8.5)

    val = [r for r in evals if r["split"] == "val"]
    tr = [r for r in evals if r["split"] == "train"]
    ep_v = [fnum(r["epoch"]) for r in val]

    # B) ADE ejecutado y FDE (validación)
    ax = axs[0, 1]
    ax.plot([fnum(r["epoch"]) for r in tr], [fnum(r["ade_exec_mm"]) for r in tr], color=C3, ls="--", marker="^",
            ms=3.5, lw=1.4, label="Train-subset ADE (executed)")
    ax.plot(ep_v, [fnum(r["ade_exec_mm"]) for r in val], color=C1, marker="o", ms=3.5, label="Held-out ADE (executed steps)")
    ax.plot(ep_v, [fnum(r["fde_mm"]) for r in val], color=C2, marker="s", ms=3.5, label="Held-out FDE (horizon end)")
    hold = [fnum(r.get("hold_ade_exec_mm")) for r in val if r.get("hold_ade_exec_mm") not in (None, "")]
    if hold:
        ax.axhline(hold[-1], color=GRAY, ls=":", lw=1.4, label="Baseline: hold pose (executed)")
    ax.axvline(args.epoch, color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mm")
    ax.set_title("B) Offline displacement errors", fontweight="bold")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    # C) F1 del done y razón de progreso
    ax = axs[0, 2]
    ax.plot(ep_v, [fnum(r["done_f1"]) for r in val], color=C1, marker="o", ms=3.5, label="Held-out done-flag F1")
    ax.plot(ep_v, [fnum(r["progress_ratio"]) for r in val], color=C2, marker="s", ms=3.5, label="Held-out progress ratio")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.4, label="Ideal progress")
    ax.axvline(args.epoch, color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_title("C) Termination and progress", fontweight="bold")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    # D) reproducción por episodio
    ax = axs[1, 0]
    eps = rep["episodes"]
    x = np.arange(len(eps))
    ax.bar(x - 0.2, [e["cmd_err_mm_mean"] for e in eps], width=0.4, color=C1, label="Command error, mean (mm)")
    ax.bar(x + 0.2, [e["stop_to_goal_mm"] for e in eps], width=0.4, color=C2, label="Auto-stop to goal (mm, default 0.5 / 1 step)")
    ax.set_xticks(x)
    ax.set_xticklabels([e["episode"].replace("episode_", "ep") for e in eps], rotation=0)
    ax.set_ylabel("mm")
    ax.set_title("D) Replay of the deployed inference path (held-out)", fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend(fontsize=8.5)

    # E) barrido del auto-stop
    ax = axs[1, 1]
    sw = rep.get("autostop_sweep", [])
    thr = sorted({r["threshold"] for r in sw})
    pat = sorted({r["patience"] for r in sw})
    grid = np.full((len(thr), len(pat)), np.nan)
    rate = np.full_like(grid, np.nan)
    for r in sw:
        i, j = thr.index(r["threshold"]), pat.index(r["patience"])
        grid[i, j] = r["stop_to_goal_mm_median"]
        rate[i, j] = r["stop_rate"]
    # Solo son válidas las configuraciones que detienen el 100% de los episodios: en las demás la distancia se mide al
    # final del episodio (el robot nunca paró) y parecería engañosamente pequeña.
    valid = np.where(rate >= 1.0, grid, np.nan)
    ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=8, aspect="auto")
    im = ax.imshow(np.ma.masked_invalid(valid), cmap="Blues", aspect="auto", vmin=0.0,
                   vmax=max(1.0, float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0))
    vmean = 0.5 * float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.0
    for i in range(len(thr)):
        for j in range(len(pat)):
            if np.isfinite(valid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f} mm\n100% stop", ha="center", va="center", fontsize=8,
                        color="white" if valid[i, j] > vmean else INK)
            else:
                ax.text(j, i, f"stops only\n{rate[i, j] * 100:.0f}% of ep.", ha="center", va="center", fontsize=8,
                        color=INK)
            if args.autostop_threshold == thr[i] and args.autostop_patience == pat[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C2, lw=2.5))
    ax.set_xticks(range(len(pat)))
    ax.set_xticklabels(pat)
    ax.set_yticks(range(len(thr)))
    ax.set_yticklabels(thr)
    ax.set_xlabel("Patience (consecutive steps ≥ threshold)")
    ax.set_ylabel("Done threshold")
    ax.set_title("E) Auto-stop calibration: median stop-to-goal distance", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")

    # F) lazo cerrado ROS 2
    ax = axs[1, 2]
    if args.closed_loop and (Path(args.closed_loop) / "tcp_trace.npy").exists():
        cl = json.load(open(Path(args.closed_loop) / "closed_loop_check.json"))
        tcp = np.load(Path(args.closed_loop) / "tcp_trace.npy")
        demo = np.array(cl["demo_displacement_m"])
        h = tcp[0, 1:4]
        ax.plot(tcp[:, 1], tcp[:, 3], color=C2, lw=2, label="Simulated robot TCP (ROS 2 chain)")
        ax.scatter([h[0]], [h[2]], marker="*", s=160, color=INK, label="HOME3")
        ax.plot([h[0], h[0] + demo[0]], [h[2], h[2] + demo[2]], color=C1, lw=2, ls="--",
                label="Demonstrated net displacement")
        ax.scatter([h[0] + demo[0]], [h[2] + demo[2]], marker="o", s=40, color=C1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"F) Closed-loop integration (cos sim {cl['displacement_cosine_similarity']:.2f}, "
                     f"auto-stop {'yes' if cl['neural_autostop'] else 'no'})", fontweight="bold")
        ax.legend(fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "Closed-loop check not provided", ha="center", va="center")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out / "final_model_report.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "final_model_report.pdf", bbox_inches="tight")
    plt.close(fig)

    sel = [r for r in val if int(fnum(r["epoch"])) == args.epoch]
    sel = sel[-1] if sel else (val[-1] if val else {})
    summ = rep["summary"]
    if args.model_card:
        card = [
            f"# Model card — `{run.name}` (epoch {args.epoch})", "",
            "**Status: trained and verified offline / in ROS 2 simulation only. NOT yet evaluated on the physical xArm5.**", "",
            "| Item | Value |", "|---|---|",
            f"| Dataset | `{ds}` ({len(meta['train_episodes'])} train / {len(meta['val_episodes'])} held-out episodes) |",
            f"| Task | approach the pruning shears on the table from HOME3 (no grasp) |",
            f"| Action representation | `{meta['action_representation']}` (UMI-style, relative to the current EE pose) |",
            f"| Quaternion convention | `{meta['quaternion_hemisphere']}` (reference {np.round(meta['quaternion_reference'], 4).tolist() if meta.get('quaternion_reference') else '-'}) |",
            f"| Training | {meta['epochs']} epochs x {spe} steps, batch {meta['batch_size']}, lr {meta['lr']:.1e} warmup+cosine, EMA {meta['use_ema']}, ColorJitter {bool(meta.get('color_jitter'))} |",
            f"| Held-out ADE / ADE executed / FDE | {fnum(sel.get('ade_mm')):.1f} / {fnum(sel.get('ade_exec_mm')):.1f} / {fnum(sel.get('fde_mm')):.1f} mm (hold-pose baseline {fnum(sel.get('hold_ade_mm')):.1f} / {fnum(sel.get('hold_ade_exec_mm')):.1f} mm) |",
            f"| Held-out orientation error | {fnum(sel.get('rot_deg')):.2f} deg |",
            f"| Held-out done F1 / progress ratio | {fnum(sel.get('done_f1')):.2f} / {fnum(sel.get('progress_ratio')):.2f} |",
            f"| Replay (Jetson path) command error | {summ['cmd_err_mm_mean']:.1f} mm mean, {summ['cmd_err_mm_p95']:.1f} mm p95, {summ['rot_err_deg_mean']:.2f} deg |",
            f"| Auto-stop (replay, default params) | rate {summ['autostop_rate'] * 100:.0f}%, median delay {summ['autostop_delay_s_median']} s, stop-to-goal {summ['stop_to_goal_mm']:.1f} mm |",
        ]
        if args.autostop_threshold is not None:
            card.append(f"| Recommended auto-stop | threshold {args.autostop_threshold}, patience {args.autostop_patience} steps |")
        if args.closed_loop and (Path(args.closed_loop) / "closed_loop_check.json").exists():
            cl = json.load(open(Path(args.closed_loop) / "closed_loop_check.json"))
            card.append(f"| ROS 2 closed loop (simulated bridge) | direction cos sim {cl['displacement_cosine_similarity']:.2f}, "
                        f"auto-stop {cl['neural_autostop']}, bridge rejections {cl['bridge_rejections']} |")
        card += ["", "Files: `model.safetensors`, `config.json` (carries action representation and quaternion convention), "
                 "`training_meta.json`. Report figure: `final_model_report.png`.", ""]
        Path(args.model_card).write_text("\n".join(card))
    print(json.dumps({"selected_epoch": args.epoch, "val": {k: sel.get(k) for k in
                      ("ade_mm", "ade_exec_mm", "fde_mm", "rot_deg", "done_f1", "progress_ratio")}, "replay": summ},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
