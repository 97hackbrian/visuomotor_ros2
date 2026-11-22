#!/usr/bin/env python3
"""Diagnóstico offline de checkpoints de Diffusion Policy frente a líneas base, con ablación visual (solo lectura).

Responde, antes de llevar un checkpoint al robot, dos preguntas que la pérdida no contesta:

1. ¿Supera a las líneas base sin aprendizaje? "Quedarse quieto" (vara baja) y "velocidad constante" (vara exigente:
   extrapola la velocidad de los dos últimos estados y no usa la imagen). Se reporta el perfil de error por paso del
   horizonte: dónde gana la política (intención a 0.5-0.75 s) y dónde gana la cinemática (primeros pasos).
2. ¿Usa la imagen? Ablación: cada ventana recibe la imagen de otra ventana del lote (en su mayoría de otro episodio)
   con el mismo estado. Si el error no sube, la política ignora la visión. En las ventanas que arrancan desde el
   reposo el estado no dice hacia dónde ir (todas empiezan en HOME): se compara la dirección predicha con la
   demostrada, con la de las imágenes cruzadas y con la dirección media de arranque de los episodios de entrenamiento.

Origen: diagnóstico de train_real_v7x5v2_relative_ee_e300 (Fase 22, 2026-11-22). La línea base de velocidad constante
también se calcula en cada evaluación de train.py (visuomotor_core/training/evaluation.py, columnas cv_*).

Salidas en --out-dir: baseline_diagnostics.json (todo), baseline_diagnostics.csv (una fila por checkpoint, sirve de
tabla de la figura) y baseline_diagnostics.png/.pdf (figura en inglés, 300 DPI).

Uso (contenedor, GPU; un checkpoint tarda ~1 min con 2.6k ventanas):
    python3 tools/diagnose_policy_baselines.py --run-dir visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300 \\
        --epochs 100,150,200,250,300 --highlight 200 \\
        --out-dir visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300/baseline_diagnostics
    python3 tools/diagnose_policy_baselines.py --checkpoint deploy/jetson/model \\
        --meta visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300/training_meta.json --out-dir <dir>
    python3 tools/diagnose_policy_baselines.py --replot --out-dir <dir>     # solo la figura, desde el JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "visuomotor_core"))

from visuomotor_core.dataset.zarr_dataset import ZarrDataset, list_episodes  # noqa: E402
from visuomotor_core.models.diffusion_policy import DiffusionPolicy  # noqa: E402
from visuomotor_core.training.evaluation import (constant_velocity_prediction,  # noqa: E402
                                                 estimate_action_offset)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica validada (skill dataviz; orden fijo): política, imágenes cruzadas, dirección media.
# Neutros para las líneas base: gris = quedarse quieto, tinta = velocidad constante.
C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10})


# ----------------------------------------------------------------------------------------------------------------
def resolve_checkpoints(args) -> Tuple[List[Tuple[str, Optional[int], Path]], dict]:
    """Lista ``(etiqueta, época, directorio)`` y metadatos de entrenamiento (episodios train/val, dataset)."""
    ckpts: List[Tuple[str, Optional[int], Path]] = []
    meta_path: Optional[Path] = Path(args.meta) if args.meta else None
    if args.run_dir:
        run = Path(args.run_dir)
        meta_path = meta_path or run / "training_meta.json"
        for tok in [t.strip() for t in args.epochs.split(",") if t.strip()]:
            if tok == "final":
                ckpts.append(("final", None, run))
            else:
                ckpts.append((f"epoch {int(tok)}", int(tok), run / "checkpoints" / f"epoch_{int(tok)}"))
    for c in args.checkpoint or []:
        c = Path(c)
        ckpts.append((c.name, None, c))
        if meta_path is None:
            for cand in (c / "training_meta.json", c.parent.parent / "training_meta.json"):
                if cand.is_file():
                    meta_path = cand
                    break
    if not ckpts:
        raise SystemExit("Indicar --run-dir/--epochs o --checkpoint")
    for _, _, d in ckpts:
        if not (d / "model.safetensors").is_file():
            raise SystemExit(f"No existe {d / 'model.safetensors'}")
    meta = json.load(open(meta_path)) if meta_path and meta_path.is_file() else {}
    return ckpts, meta


def onset_mean_direction(group, episodes: List[str], horizon: int, fs: float, onset_speed: float,
                         min_disp: float) -> Tuple[Optional[np.ndarray], int]:
    """Dirección media (unitaria) del desplazamiento demostrado en el horizonte, en ventanas que arrancan del reposo."""
    dirs = []
    for e in episodes:
        s = np.asarray(group[e]["state"][:, :3], dtype=np.float64)
        a = np.asarray(group[e]["action"][:, :3], dtype=np.float64)
        n = len(s)
        for i in range(1, n):
            disp = a[min(i + horizon - 1, n - 1)] - s[i]
            if np.linalg.norm(s[i] - s[i - 1]) * fs < onset_speed and np.linalg.norm(disp) > min_disp:
                dirs.append(disp / np.linalg.norm(disp))
    if not dirs:
        return None, 0
    m = np.mean(dirs, axis=0)
    return m / max(np.linalg.norm(m), 1e-12), len(dirs)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def _err_summary(err: np.ndarray, k: int) -> Dict[str, float]:
    return {"ade_mm": float(err.mean()), "ade_exec_mm": float(err[:, :k].mean()), "fde_mm": float(err[:, -1].mean())}


# ----------------------------------------------------------------------------------------------------------------
def run(args) -> dict:
    ckpts, meta = resolve_checkpoints(args)
    dataset = args.dataset or meta.get("dataset_path")
    if not dataset:
        raise SystemExit("Sin --dataset ni dataset_path en training_meta.json")
    group = zarr.open_group(dataset, mode="r")
    all_eps = list_episodes(group)
    if args.episodes == "val":
        episodes = list(meta.get("val_episodes") or [])
        if not episodes:
            raise SystemExit("training_meta.json no tiene val_episodes: usar --episodes all o una lista")
    elif args.episodes == "all":
        episodes = all_eps
    else:
        episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    train_eps = [e for e in (meta.get("train_episodes") or all_eps) if e not in set(episodes)]
    fs = float(group[episodes[0]].attrs.get("sampling_rate_hz", args.fs))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = []
    for label, epoch, path in ckpts:
        p = DiffusionPolicy.from_pretrained(path).eval().to(dev)
        if args.num_inference_steps:
            p.diffusion.num_inference_steps = int(args.num_inference_steps)
        policies.append((label, epoch, p))
    cfg = policies[0][2].config
    offset = estimate_action_offset(group, train_eps)
    mean_dir, n_dir = onset_mean_direction(group, train_eps, cfg.horizon, fs, args.onset_speed, args.min_disp)

    ds = ZarrDataset(dataset, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon, episodes=episodes,
                     action_representation="absolute", stats={}, proprio_dropout_prob=0.0, color_jitter=None)
    order = np.random.default_rng(args.seed).permutation(len(ds))       # mezcla episodios: el "roll" cruza imágenes
    if args.max_windows > 0:
        order = order[:args.max_windows]
    ep_of = np.array([ds.frames[i][0] for i in order])
    loader = DataLoader(Subset(ds, order.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    acc = {lab: {"err": [], "err_swap": [], "pd": [], "pd_swap": []} for lab, _, _ in policies}
    com = {"err_hold": [], "err_cv": [], "gt_disp": [], "speed": [], "swap_other_ep": []}
    with torch.no_grad():
        for b, batch in enumerate(loader):
            img = batch["observation.image"].to(dev, non_blocking=True)
            st = batch["observation.state"].to(dev)
            gt = batch["action"].to(dev)[:, 1:, :3].float()                # pasos ejecutables t+1 .. t+H-1
            n = st.shape[0]
            shift = n // 2 if n > 1 else 0
            cur = st[:, -1, :3].float()
            h = gt.shape[1]
            for lab, _, pol in policies:
                modes = (("err", "pd", img),) + ((("err_swap", "pd_swap", img.roll(shift, dims=0)),)
                                                  if not args.no_swap else ())
                for ek, pk, im in modes:
                    torch.manual_seed(args.seed + b)
                    pred = pol.predict_chunk({"observation.image": im, "observation.state": st}).float()[..., :3]
                    h = min(h, pred.shape[1])
                    acc[lab][ek].append(((pred[:, :h] - gt[:, :h]).norm(dim=-1) * 1000.0).cpu().numpy())
                    acc[lab][pk].append((pred[:, h - 1] - cur).cpu().numpy())
            g = gt[:, :h]
            cv = constant_velocity_prediction(st, h, offset)
            com["err_cv"].append(((cv - g).norm(dim=-1) * 1000.0).cpu().numpy())
            com["err_hold"].append(((cur[:, None, :] - g).norm(dim=-1) * 1000.0).cpu().numpy())
            com["gt_disp"].append((g[:, -1] - cur).cpu().numpy())
            com["speed"].append(((st[:, -1, :3] - st[:, -2, :3]).norm(dim=-1) * fs).cpu().numpy()
                                if st.shape[1] >= 2 else np.zeros(n))
            eps_b = ep_of[b * args.batch_size: b * args.batch_size + n]
            com["swap_other_ep"].append(eps_b != np.roll(eps_b, shift))
            if args.verbose:
                print(f"lote {b + 1}/{len(loader)}", flush=True)

    C = {k: np.concatenate(v) for k, v in com.items()}
    A = {lab: {k: np.concatenate(v) if v else None for k, v in d.items()} for lab, d in acc.items()}
    k_exec = int(args.n_action_steps)
    gd = C["gt_disp"]
    valid = np.linalg.norm(gd, axis=1) > args.min_disp
    groups = {"onset_from_rest": valid & (C["speed"] < args.onset_speed),
              "moving": valid & (C["speed"] >= args.onset_speed)}

    out = {
        "dataset": str(dataset), "episodes": episodes, "n_windows": int(len(ep_of)), "sampling_rate_hz": fs,
        "n_action_steps_exec": k_exec, "horizon_steps": int(C["err_cv"].shape[1]), "action_offset_steps": offset,
        "swap_from_other_episode_fraction": float(C["swap_other_ep"].mean()),
        "onset_speed_m_s": args.onset_speed, "min_displacement_m": args.min_disp,
        "train_onset_mean_direction": {"n_windows": n_dir, "dir": None if mean_dir is None else mean_dir.tolist()},
        "baselines": {"hold": _err_summary(C["err_hold"], k_exec), "constant_velocity": _err_summary(C["err_cv"], k_exec)},
        "baseline_error_profile_mm": {"hold": C["err_hold"].mean(0).tolist(), "constant_velocity": C["err_cv"].mean(0).tolist()},
        "checkpoints": {},
    }
    for lab, epoch, _ in policies:
        d = A[lab]
        res = {"epoch": epoch, "policy": _err_summary(d["err"], k_exec),
               "error_profile_mm": {"policy": d["err"].mean(0).tolist()}}
        if d["err_swap"] is not None:
            res["swapped_images"] = _err_summary(d["err_swap"], k_exec)
            res["error_profile_mm"]["swapped_images"] = d["err_swap"].mean(0).tolist()
        for gname, mask in groups.items():
            ms = mask & C["swap_other_ep"]
            gr = {"n_windows": int(mask.sum()), "fde_mm": {
                "policy": float(d["err"][mask, -1].mean()) if mask.any() else float("nan"),
                "hold": float(C["err_hold"][mask, -1].mean()) if mask.any() else float("nan"),
                "constant_velocity": float(C["err_cv"][mask, -1].mean()) if mask.any() else float("nan")},
                "direction_acc_cos_gt_0.9": {}, "direction_cos_median": {}}
            series = [("policy", d["pd"], mask)]
            if d["pd_swap"] is not None:
                series.append(("swapped_images", d["pd_swap"], ms))
                gr["fde_mm"]["swapped_images"] = float(d["err_swap"][ms, -1].mean()) if ms.any() else float("nan")
            for name, pdisp, m in series:
                c = _cos(pdisp[m], gd[m]) if m.any() else np.array([np.nan])
                gr["direction_acc_cos_gt_0.9"][name] = float(np.mean(c > 0.9)) if m.any() else float("nan")
                gr["direction_cos_median"][name] = float(np.median(c))
            if mean_dir is not None and mask.any():
                c = _cos(np.broadcast_to(mean_dir, gd[mask].shape), gd[mask])
                gr["direction_acc_cos_gt_0.9"]["train_mean_direction"] = float(np.mean(c > 0.9))
                gr["direction_cos_median"]["train_mean_direction"] = float(np.median(c))
            res[gname] = gr
        per_ep = {}
        for e in episodes:
            m = ep_of == e
            if not m.any():
                continue
            per_ep[e] = {"n": int(m.sum()), "policy_fde_mm": float(d["err"][m, -1].mean()),
                         "cv_fde_mm": float(C["err_cv"][m, -1].mean()), "hold_fde_mm": float(C["err_hold"][m, -1].mean()),
                         "policy_ade_exec_mm": float(d["err"][m, :k_exec].mean()),
                         "cv_ade_exec_mm": float(C["err_cv"][m, :k_exec].mean())}
            if d["err_swap"] is not None:
                per_ep[e]["swap_fde_mm"] = float(d["err_swap"][m, -1].mean())
        res["per_episode"] = per_ep
        res["episodes_policy_beats_cv_fde"] = int(sum(v["policy_fde_mm"] < v["cv_fde_mm"] for v in per_ep.values()))
        res["episodes_policy_beats_cv_ade_exec"] = int(sum(v["policy_ade_exec_mm"] < v["cv_ade_exec_mm"]
                                                           for v in per_ep.values()))
        out["checkpoints"][lab] = res
    return out


# ----------------------------------------------------------------------------------------------------------------
CSV_COLS = ["checkpoint", "epoch", "n_windows", "ade_mm", "ade_exec_mm", "fde_mm", "swap_ade_mm", "swap_ade_exec_mm",
            "swap_fde_mm", "hold_ade_mm", "hold_ade_exec_mm", "hold_fde_mm", "cv_ade_mm", "cv_ade_exec_mm", "cv_fde_mm",
            "dir_acc_moving", "dir_acc_moving_swap", "dir_acc_moving_mean_dir", "dir_acc_onset", "dir_acc_onset_swap",
            "dir_acc_onset_mean_dir", "n_onset", "n_moving", "episodes_beat_cv_fde", "episodes_beat_cv_ade_exec"]


def write_csv(out: dict, path: Path):
    hb, cb = out["baselines"]["hold"], out["baselines"]["constant_velocity"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for lab, r in out["checkpoints"].items():
            sw = r.get("swapped_images", {})
            mv, on = r["moving"]["direction_acc_cos_gt_0.9"], r["onset_from_rest"]["direction_acc_cos_gt_0.9"]
            w.writerow([lab, r["epoch"], out["n_windows"], r["policy"]["ade_mm"], r["policy"]["ade_exec_mm"],
                        r["policy"]["fde_mm"], sw.get("ade_mm"), sw.get("ade_exec_mm"), sw.get("fde_mm"),
                        hb["ade_mm"], hb["ade_exec_mm"], hb["fde_mm"], cb["ade_mm"], cb["ade_exec_mm"], cb["fde_mm"],
                        mv.get("policy"), mv.get("swapped_images"), mv.get("train_mean_direction"), on.get("policy"),
                        on.get("swapped_images"), on.get("train_mean_direction"), r["onset_from_rest"]["n_windows"],
                        r["moving"]["n_windows"], r["episodes_policy_beats_cv_fde"],
                        r["episodes_policy_beats_cv_ade_exec"]])


def plot(out: dict, highlight: str, path_base: Path, dataset_name: str):
    ck = out["checkpoints"]
    hi = ck[highlight]
    dt = 1.0 / out["sampling_rate_hz"]
    k = out["n_action_steps_exec"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    n_eps = len(out["episodes"])
    fig.suptitle(f"Policy vs. baselines and visual ablation — {dataset_name} "
                 f"({n_eps} held-out episodes, {out['n_windows']} windows)", fontsize=14, fontweight="bold")

    # A) perfil de error por paso del horizonte
    ax = axs[0, 0]
    prof = hi["error_profile_mm"]
    t = (np.arange(len(prof["policy"])) + 1) * dt
    ax.axvspan(0.5 * dt, (k + 0.5) * dt, color="#d9d8d4", alpha=0.45, lw=0)
    ax.plot(t, prof["policy"], color=C1, marker="o", ms=4, label=f"Policy ({highlight})")
    if "swapped_images" in prof:
        ax.plot(t, prof["swapped_images"], color=C2, marker="s", ms=4, label="Policy, images from another episode")
    ax.plot(t, out["baseline_error_profile_mm"]["constant_velocity"], color=INK, ls="-.", label="Baseline: constant velocity")
    ax.plot(t, out["baseline_error_profile_mm"]["hold"], color=GRAY, ls="--", label="Baseline: hold current pose")
    ymax = ax.get_ylim()[1]
    ax.text(0.5 * dt + 0.005, ymax * 0.97, f"executed\n({k} steps)", va="top", ha="left", fontsize=9, color=INK)
    ax.set_title("A) Mean position error along the predicted horizon", fontweight="bold", loc="left")
    ax.set_xlabel("Time ahead of the observation (s)")
    ax.set_ylabel("Position error (mm)")
    ax.set_ylim(0, ymax)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9)

    # B) FDE por checkpoint
    ax = axs[0, 1]
    labs = list(ck.keys())
    xs = [ck[lab]["epoch"] if ck[lab]["epoch"] is not None else i for i, lab in enumerate(labs)]
    ax.plot(xs, [ck[lab]["policy"]["fde_mm"] for lab in labs], color=C1, marker="o", ms=7, label="Policy")
    if all("swapped_images" in ck[lab] for lab in labs):
        ax.plot(xs, [ck[lab]["swapped_images"]["fde_mm"] for lab in labs], color=C2, marker="s", ms=7,
                label="Policy, images from another episode")
    ax.axhline(out["baselines"]["constant_velocity"]["fde_mm"], color=INK, ls="-.", lw=1.6,
               label="Baseline: constant velocity")
    ax.axhline(out["baselines"]["hold"]["fde_mm"], color=GRAY, ls="--", lw=1.6, label="Baseline: hold current pose")
    hx = xs[labs.index(highlight)]
    ax.annotate(f"{ck[highlight]['policy']['fde_mm']:.1f} mm", (hx, ck[highlight]["policy"]["fde_mm"]),
                textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=INK)
    ax.set_title(f"B) Final displacement error (t + {len(t) * dt:.2f} s) per checkpoint", fontweight="bold", loc="left")
    ax.set_xlabel("Epoch" if all(ck[lab]["epoch"] is not None for lab in labs) else "Checkpoint")
    if not all(ck[lab]["epoch"] is not None for lab in labs):
        ax.set_xticks(xs)
        ax.set_xticklabels(labs)
    ax.set_ylabel("FDE (mm)")
    ax.set_ylim(0, None)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=9)

    # C) exactitud de dirección
    ax = axs[1, 0]
    groups = [("onset_from_rest", "Starting from rest"), ("moving", "Already moving")]
    series = [("policy", "Policy", C1), ("swapped_images", "Policy, images from another episode", C2),
              ("train_mean_direction", "Always the mean training direction", C3)]
    width = 0.26
    for j, (key, label, color) in enumerate(series):
        vals = [hi[g]["direction_acc_cos_gt_0.9"].get(key, np.nan) for g, _ in groups]
        pos = np.arange(len(groups)) + (j - 1) * width
        bars = ax.bar(pos, [100 * v for v in vals], width=width - 0.02, color=color, label=label)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, 100 * v + 1.2, f"{100 * v:.0f}%", ha="center",
                        fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{name}\n(n = {hi[g]['n_windows']} windows)" for g, name in groups])
    ax.set_ylabel("Windows with correct direction (cos > 0.9, %)")
    ax.set_ylim(0, 105)
    ax.set_title(f"C) Direction of the predicted displacement ({highlight})", fontweight="bold", loc="left")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    # D) FDE por episodio
    ax = axs[1, 1]
    pe = hi["per_episode"]
    eps = list(pe.keys())
    x = np.arange(len(eps))
    marks = [("policy_fde_mm", "Policy", C1, "o"), ("swap_fde_mm", "Policy, images from another episode", C2, "s"),
             ("cv_fde_mm", "Baseline: constant velocity", INK, "D"), ("hold_fde_mm", "Baseline: hold current pose", GRAY, "^")]
    for key, label, color, m in marks:
        if all(key in pe[e] for e in eps):
            ax.scatter(x, [pe[e][key] for e in eps], color=color, marker=m, s=46, label=label, zorder=3,
                       edgecolors="white", linewidths=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("episode_", "ep ") for e in eps])
    ax.set_ylabel("FDE (mm)")
    top = max(max(pe[e].get(key, 0.0) for e in eps) for key, _, _, _ in marks)
    ax.set_ylim(0, 1.32 * top)                                   # deja la franja superior para la leyenda
    ax.set_title(f"D) FDE per held-out episode ({highlight}): policy < constant velocity in "
                 f"{hi['episodes_policy_beats_cv_fde']}/{len(eps)}", fontweight="bold", loc="left")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", help="directorio de una corrida de train.py (usa checkpoints/epoch_N)")
    ap.add_argument("--epochs", default="final", help="épocas separadas por comas (o 'final') con --run-dir")
    ap.add_argument("--checkpoint", nargs="*", help="directorios de checkpoint (model.safetensors + config.json)")
    ap.add_argument("--meta", help="training_meta.json (episodios train/val y dataset); por defecto, el de la corrida")
    ap.add_argument("--dataset", help="dataset Zarr; por defecto dataset_path de training_meta.json")
    ap.add_argument("--episodes", default="val", help="'val' (de training_meta.json), 'all' o lista separada por comas")
    ap.add_argument("--highlight", help="checkpoint destacado en las figuras A, C y D (época o etiqueta)")
    ap.add_argument("--n-action-steps", type=int, default=3, help="pasos ejecutados por inferencia (Jetson: 3)")
    ap.add_argument("--num-inference-steps", type=int, default=0, help="pasos DDIM (0 = los del checkpoint)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = todas las ventanas")
    ap.add_argument("--onset-speed", type=float, default=0.01, help="m/s: ventana 'desde el reposo' por debajo")
    ap.add_argument("--min-disp", type=float, default=0.01, help="m de desplazamiento demostrado para medir dirección")
    ap.add_argument("--fs", type=float, default=20.0, help="Hz si el dataset no guarda sampling_rate_hz")
    ap.add_argument("--no-swap", action="store_true", help="omitir la ablación de imágenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--replot", action="store_true", help="solo regenerar la figura desde baseline_diagnostics.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    if args.replot:
        out = json.loads((Path(args.out_dir) / "baseline_diagnostics.json").read_text())
        hl = args.highlight and next((lab for lab in out["checkpoints"]
                                      if lab in (args.highlight, f"epoch {args.highlight}")), None)
        hl = hl or out.get("highlight") or list(out["checkpoints"])[-1]
        plot(out, hl, Path(args.out_dir) / "baseline_diagnostics", Path(out["dataset"]).name.replace(".zarr", ""))
        print(f"Figura regenerada en {args.out_dir}")
        return 0
    out = run(args)
    labs = list(out["checkpoints"].keys())
    hl = labs[-1]
    if args.highlight:
        cand = [lab for lab in labs if lab == args.highlight or lab == f"epoch {args.highlight}"]
        if not cand:
            raise SystemExit(f"--highlight {args.highlight!r} no está entre {labs}")
        hl = cand[0]
    out["highlight"] = hl
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_diagnostics.json").write_text(json.dumps(out, indent=1))
    write_csv(out, out_dir / "baseline_diagnostics.csv")
    plot(out, hl, out_dir / "baseline_diagnostics", Path(out["dataset"]).name.replace(".zarr", ""))
    r = out["checkpoints"][hl]
    cvb, hb = out["baselines"]["constant_velocity"], out["baselines"]["hold"]
    print(f"[{hl}] {out['n_windows']} ventanas | ADE exec({out['n_action_steps_exec']}) política "
          f"{r['policy']['ade_exec_mm']:.2f} mm vs. vel. cte {cvb['ade_exec_mm']:.2f} vs. quieto {hb['ade_exec_mm']:.2f}"
          f" | FDE política {r['policy']['fde_mm']:.1f} vs. vel. cte {cvb['fde_mm']:.1f} vs. quieto {hb['fde_mm']:.1f}"
          + (f" | FDE imágenes cruzadas {r['swapped_images']['fde_mm']:.1f}" if "swapped_images" in r else ""))
    print(f"Resultados en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
