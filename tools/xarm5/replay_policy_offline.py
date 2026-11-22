#!/usr/bin/env python3
"""Reproducción offline de la ruta de inferencia de ``policy_node`` sobre episodios reales (sin ROS ni robot).

Replica, paso a paso a la frecuencia del dataset, lo que hace ``policy_node`` en la Jetson con los MISMOS parámetros
(``deploy/jetson/config/jetson_policy_params.yaml``): colas de observación de DiffusionPolicy, ``n_action_steps``,
ensemble temporal, recomposición ``relative_ee``, convención de hemisferio del cuaternión y auto-stop neuronal.

La observación (imagen + estado) se toma de la demostración (*teacher forcing*): mide el error de cada comando frente
a la pose demostrada siguiente y el instante de auto-stop, pero no el error compuesto de lazo cerrado (requeriría
renderizar la escena). Es la prueba más cercana al robot disponible sin hardware.

Uso (contenedor):
    python3 tools/xarm5/replay_policy_offline.py --checkpoint visuomotor_core/outputs/<run>/checkpoints/epoch_N \
        --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr --episodes val --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_core"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visuomotor_core.models.diffusion_policy import DiffusionPolicy  # noqa: E402

C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10})
DEFAULT_PARAMS = REPO / "deploy/jetson/config/jetson_policy_params.yaml"


def load_policy(ckpt: Path, params: dict, device):
    policy = DiffusionPolicy.from_pretrained(ckpt)
    policy.diffusion.num_inference_steps = int(params["num_inference_steps"])
    policy.config.n_action_steps = int(params["n_action_steps"])
    policy.config.use_temporal_ensemble = bool(params["use_temporal_ensemble"])
    policy.config.temporal_ensemble_k = float(params["temporal_ensemble_k"])
    policy.reset()
    policy.eval().to(device)
    return policy


@torch.no_grad()
def replay_episode(policy, group, params, device, seed=0):
    """Devuelve comandos [T, A] (marco base - home) y la verdad terreno siguiente [T, A]."""
    torch.manual_seed(seed)
    policy.reset()
    if hasattr(policy, "active_chunks"):
        policy.active_chunks = []
    images = np.asarray(group["image"])
    states = np.asarray(group["state"], dtype=np.float32)
    actions = np.asarray(group["action"], dtype=np.float32)
    thr = float(params["neural_auto_stop_threshold"])
    patience = max(1, int(params.get("neural_auto_stop_patience_steps", 1)))
    cmds, stop_t, streak = [], None, 0
    for t in range(len(states)):
        img = torch.from_numpy(images[t]).to(torch.float32).permute(2, 0, 1)[None] / 255.0
        obs = {"observation.image": img.to(device), "observation.state": torch.from_numpy(states[t])[None].to(device)}
        a = policy.select_action(obs).squeeze(0).float().cpu().numpy()
        cmds.append(a)
        if a.shape[-1] >= 9:
            streak = streak + 1 if a[8] >= thr else 0
            if stop_t is None and streak >= patience:
                stop_t = t
    cmds = np.array(cmds)
    gt_next = np.concatenate([actions[1:], actions[-1:]], axis=0)
    return cmds, gt_next, stop_t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", default="val", help="'val' (de training_meta.json), 'all' o lista separada por comas")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    import zarr

    params = yaml.safe_load(open(args.params))["policy_node"]["ros__parameters"]
    ckpt = Path(args.checkpoint)
    root = zarr.open_group(args.dataset, mode="r")
    all_eps = sorted(root.group_keys(), key=lambda k: int(k.split("_")[1]))
    if args.episodes == "val":
        meta = None
        for cand in (ckpt / "training_meta.json", ckpt.parent.parent / "training_meta.json"):
            if cand.exists():
                meta = json.load(open(cand))
                break
        if meta is None or not meta.get("val_episodes"):
            sys.exit("No se encontraron val_episodes en training_meta.json")
        eps = meta["val_episodes"]
    elif args.episodes == "all":
        eps = all_eps
    else:
        eps = [e.strip() for e in args.episodes.split(",")]
    if args.max_episodes > 0:
        eps = eps[:args.max_episodes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, params, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    for e in eps:
        g = root[e]
        cmds, gt, stop_t = replay_episode(policy, g, params, device)
        err = np.linalg.norm(cmds[:, :3] - gt[:, :3], axis=1) * 1000.0
        q1 = cmds[:, 3:7] / np.linalg.norm(cmds[:, 3:7], axis=1, keepdims=True)
        rot = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q1 * gt[:, 3:7], 1)), 0, 1)))
        done_gt = np.asarray(g["action"][:, 8]) >= 0.5
        t_done_gt = int(np.argmax(done_gt)) if done_gt.any() else None
        end_gt = np.asarray(g["action"][-1, :3])
        stop_pos = cmds[stop_t, :3] if stop_t is not None else cmds[-1, :3]
        moving = ~done_gt
        rows.append({
            "episode": e, "frames": int(len(cmds)),
            "cmd_err_mm_mean": float(err[moving].mean()), "cmd_err_mm_p95": float(np.percentile(err[moving], 95)),
            "rot_err_deg_mean": float(rot[moving].mean()),
            "autostop_step": stop_t, "demo_done_step": t_done_gt,
            "autostop_delay_s": None if (stop_t is None or t_done_gt is None) else (stop_t - t_done_gt) / 20.0,
            "stop_to_goal_mm": float(np.linalg.norm(stop_pos - end_gt) * 1000.0),
            "quat_min_norm": float(np.linalg.norm(cmds[:, 3:7], axis=1).min()),
        })
        traces[e] = (cmds, np.asarray(g["action"]), stop_t, t_done_gt)
        print(f"{e}: cmd err {rows[-1]['cmd_err_mm_mean']:.1f} mm (p95 {rows[-1]['cmd_err_mm_p95']:.1f}) | rot "
              f"{rows[-1]['rot_err_deg_mean']:.2f}° | auto-stop t={stop_t} vs demo done t={t_done_gt} | "
              f"stop→goal {rows[-1]['stop_to_goal_mm']:.1f} mm", flush=True)

    summ = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in
            ("cmd_err_mm_mean", "cmd_err_mm_p95", "rot_err_deg_mean", "stop_to_goal_mm")}
    delays = [r["autostop_delay_s"] for r in rows if r["autostop_delay_s"] is not None]
    summ["autostop_rate"] = float(np.mean([r["autostop_step"] is not None for r in rows]))
    summ["autostop_delay_s_median"] = float(np.median(delays)) if delays else None
    summ["premature_stops"] = int(sum(1 for d in delays if d < -1.0))
    # Barrido offline de la parada automática (umbral x paciencia) sobre las mismas predicciones
    sweep = []
    for thr in (0.5, 0.6, 0.7, 0.8):
        for pat in (1, 3, 5, 10):
            st, d2g = [], []
            for e, (cmds, demo, _, t_done) in traces.items():
                above = cmds[:, 8] >= thr
                run, t_stop = 0, None
                for t, ab in enumerate(above):
                    run = run + 1 if ab else 0
                    if run >= pat:
                        t_stop = t
                        break
                st.append(None if (t_stop is None or t_done is None) else (t_stop - t_done) / 20.0)
                pos = cmds[t_stop, :3] if t_stop is not None else cmds[-1, :3]
                d2g.append(float(np.linalg.norm(pos - demo[-1, :3]) * 1000.0))
            ok = [d for d in st if d is not None]
            sweep.append({"threshold": thr, "patience": pat, "stop_rate": len(ok) / len(st),
                          "delay_s_median": float(np.median(ok)) if ok else None,
                          "premature_gt_1s": int(sum(d < -1.0 for d in ok)),
                          "stop_to_goal_mm_median": float(np.median(d2g))})
    report = {"checkpoint": str(ckpt), "dataset": args.dataset, "params": args.params, "summary": summ,
              "autostop_sweep": sweep, "episodes": rows}
    (out / "replay_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summ, indent=2))
    print("auto-stop sweep (thr, patience): " + " | ".join(
        f"{r['threshold']:.1f}/{r['patience']}: rate {r['stop_rate']:.2f} delay {r['delay_s_median']} "
        f"early {r['premature_gt_1s']} d2g {r['stop_to_goal_mm_median']:.0f}mm" for r in sweep))

    # Figura: trayectorias comandadas frente a la demostración + flag done
    n = len(traces)
    fig, axs = plt.subplots(n, 3, figsize=(16, 3.2 * n), squeeze=False)
    title = "Offline replay of the deployed inference path (teacher-forced observations)"
    fig.suptitle(f"{title}\n{args.label or ckpt.name} — {Path(args.dataset).stem}", fontsize=13, fontweight="bold")
    for i, (e, (cmds, demo, stop_t, t_done)) in enumerate(traces.items()):
        tt = np.arange(len(demo)) / 20.0
        for j, (a, b, lab) in enumerate(((0, 1, "y"), (0, 2, "z"))):
            ax = axs[i, j]
            ax.plot(demo[:, a], demo[:, b], color=C1, lw=2.0, label="Demonstration")
            ax.plot(cmds[:, a], cmds[:, b], color=C2, lw=1.4, label="Policy command")
            if stop_t is not None:
                ax.scatter([cmds[stop_t, a]], [cmds[stop_t, b]], marker="X", s=80, color=INK, zorder=5,
                           label="Neural auto-stop")
            ax.set_xlabel("x - home (m)")
            ax.set_ylabel(f"{lab} - home (m)")
            ax.set_title(f"{e}: {'top' if b == 1 else 'side'} view", fontsize=10, fontweight="bold")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True)
        ax = axs[i, 2]
        ax.plot(tt, demo[:, 8], color=C1, lw=2.0, label="Demo done flag")
        ax.plot(tt, cmds[:, 8], color=C2, lw=1.4, label="Predicted done flag")
        ax.axhline(float(params["neural_auto_stop_threshold"]), color=GRAY, ls="--", lw=1.2, label="Auto-stop threshold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Done flag")
        ax.set_title(f"{e}: termination signal", fontsize=10, fontweight="bold")
        ax.grid(True)
        if i == 0:
            for a_ in axs[0]:
                a_.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "replay_trajectories.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "replay_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
