#!/usr/bin/env python3
"""Exactitud de la odometría de la D435i contra el tablero ChArUco (y contra la verdad, en bags sintéticos).

Por cada corrida (un bag con la imagen de color, ``camera_info`` y ``/tf_static``):
1. **Referencia:** pose del marco óptico de color en el marco del tablero, en cada imagen donde el tablero se ve con
   suficientes esquinas (PnP, ``visuomotor_bringup/charuco.py``) y bajo error de reproyección.
2. **Estimaciones:** una o más odometrías (``nav_msgs/Odometry``), la del mismo bag (en vivo) o las de bags
   reprocesados (``replay_odometry_candidates.py``). Se llevan al marco óptico de color con la TF estática.
3. **Pares por sello:** la odometría de RTAB-Map lleva el sello de la imagen que la produjo. Métricas con
   ``evaluate_paired``: ATE anclado al primer par (la métrica operativa, porque las demos usan la pose relativa al
   inicio), SE(3)/Sim(3), escala, RPE por tiempo y pérdidas.
4. **Resumen por estimador:** mediana, p90 y máximo entre corridas, nunca el mejor caso. Salidas: JSON, Markdown y
   figura en inglés.

El tablero debe estar medido con calibre (``measured: true`` en ``charuco_board.yaml``). Con medidas nominales la
escala de la referencia hereda el error de impresión; ``--allow-nominal`` lo permite y lo marca en el informe.

Uso (contenedor con ROS 2 cargado):
  python3 tools/hardware/evaluate_pose_estimates.py --run datasets/vio_bags/run_01 --run datasets/vio_bags/run_02 \\
      --out-dir experiments/vio_measurement/2026-11-25
  # odometrías reprocesadas ({run} = nombre del bag de la corrida):
  ... --estimate "bundle_adjustment=experiments/vio_replay/{run}/bundle_adjustment:/replay/odom"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "hardware"))
sys.path.insert(0, str(REPO / "visuomotor_bringup"))
sys.path.insert(0, str(REPO / "visuomotor_core"))

from vio_bag_utils import (StaticTF, bag_topics, decode_image, iter_messages, make_T, matrix_to_quat,  # noqa: E402
                           odometry_from_bag, quat_to_matrix, rpy_to_matrix, stamp_of)
from visuomotor_core.analysis.trajectory_metrics import evaluate_paired  # noqa: E402

DEFAULT_CONFIG = REPO / "visuomotor_bringup" / "config" / "realsense" / "vio_measurement.yaml"
FLAT_KEYS = [("ate_anchored_mm", "median"), ("ate_anchored_mm", "p90"), ("ate_anchored_mm", "max"),
             ("final_error_anchored_mm", None), ("ate_sim3_mm", "rmse"), ("scale_error_pct", None),
             ("rot_anchored_deg", "median"), ("rpe_0.5s_trans_mm", "median"), ("rpe_0.5s_rot_deg", "median"),
             ("lost_pct", None), ("coverage_pct", None)]


def load_config(path=DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())["vio_measurement"]


# ---------------------------------------------------------------------------------------------------------------------
# Referencia del tablero
# ---------------------------------------------------------------------------------------------------------------------
def board_reference(bag: Path, cfg: dict, allow_nominal: bool = False) -> Dict:
    """Poses ``T_tablero_óptico`` por imagen donde el tablero se ve bien."""
    import cv2
    from visuomotor_bringup.charuco import CharucoConfig, estimate_pose, make_board
    bcfg = CharucoConfig.load(REPO / cfg["board_config"])
    if not bcfg.measured and not allow_nominal:
        raise SystemExit("El tablero no está medido (measured: false en charuco_board.yaml). Medirlo con calibre o "
                         "usar --allow-nominal (la escala de la referencia hereda el error de impresión).")
    board = make_board(bcfg)
    topics = bag_topics(bag)
    color = next((t for t in cfg["color_topics"] if t in topics), None)
    if color is None:
        raise SystemExit(f"{bag}: no hay imagen de color ({cfg['color_topics']})")
    k = dist = None
    t, Ts, ncorn, reproj = [], [], [], []
    n_img = 0
    for topic, m, _ in iter_messages(bag, [cfg["camera_info_topic"], color]):
        if topic == cfg["camera_info_topic"]:
            if k is None:
                k = np.asarray(m.k, dtype=np.float64).reshape(3, 3)
                dist = np.asarray(m.d, dtype=np.float64) if len(m.d) else None
            continue
        if k is None:
            continue
        n_img += 1
        img = decode_image(m)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pose = estimate_pose(gray, board, bcfg, k, dist, min_corners=int(cfg["board_min_corners"]))
        if pose is None or pose.reprojection_error_px > float(cfg["board_max_reprojection_px"]):
            continue
        r, tt = pose.camera_in_board()
        t.append(stamp_of(m))
        Ts.append(make_T(r, tt))
        ncorn.append(pose.n_corners)
        reproj.append(pose.reprojection_error_px)
    order = np.argsort(t)
    return {"t": np.asarray(t)[order], "T": np.asarray(Ts).reshape(-1, 4, 4)[order],
            "n_corners": np.asarray(ncorn)[order], "reprojection_px": np.asarray(reproj)[order],
            "n_images": n_img, "color_topic": color, "board_measured": bool(bcfg.measured)}


def ground_truth(bag: Path, topic: str) -> Optional[Dict]:
    t, Ts = [], []
    for _, m, _ in iter_messages(bag, [topic]):
        o, p = m.pose.orientation, m.pose.position
        t.append(stamp_of(m))
        Ts.append(make_T(quat_to_matrix([o.x, o.y, o.z, o.w]), [p.x, p.y, p.z]))
    if not t:
        return None
    order = np.argsort(t)
    return {"t": np.asarray(t)[order], "T": np.asarray(Ts)[order]}


# ---------------------------------------------------------------------------------------------------------------------
# Estimaciones
# ---------------------------------------------------------------------------------------------------------------------
def link_to_optical(run_bag: Path, child: str, cfg: dict) -> Tuple[np.ndarray, str]:
    tf = StaticTF()
    for _, m, _ in iter_messages(run_bag, ["/tf_static"]):
        tf.add_message(m)
    T = tf.lookup(child, cfg["optical_frame"])
    if T is not None:
        return T, "tf_static"
    return make_T(rpy_to_matrix(*cfg["default_link_to_optical_rpy"]), cfg["default_link_to_optical_xyz"]), \
        "valor típico de la D435i (el bag no trae /tf_static)"


def estimate_in_optical(est_bag: Path, topic: str, run_bag: Path, cfg: dict) -> Dict:
    od = odometry_from_bag(est_bag, topic)
    if len(od["t"]) == 0:
        raise ValueError(f"{est_bag}: sin mensajes en {topic}")
    T_child_opt, src = link_to_optical(run_bag, od["child"] or "camera_link", cfg)
    return {"t": od["t"], "T": od["T"] @ T_child_opt, "lost": od["lost"], "tf_source": src,
            "frame": od["frame"], "child": od["child"]}


def pair_by_stamp(t_a: np.ndarray, t_b: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Índices ``(i, j)`` con ``|t_a[i] - t_b[j]| <= tol`` (``t_b`` ordenado), uno por cada ``t_a``."""
    if len(t_a) == 0 or len(t_b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    j = np.searchsorted(t_b, t_a)
    c0, c1 = np.clip(j - 1, 0, len(t_b) - 1), np.clip(j, 0, len(t_b) - 1)
    c = np.where(np.abs(t_b[c1] - t_a) < np.abs(t_b[c0] - t_a), c1, c0)
    ok = np.abs(t_b[c] - t_a) <= tol
    return np.nonzero(ok)[0], c[ok]


def compare(ref: Dict, est: Dict, cfg: dict, tol: Optional[float] = None) -> Dict:
    """Métricas de ``est`` contra ``ref`` en los pares por sello."""
    tol = float(cfg["pair_tolerance_s"]) if tol is None else tol
    i, j = pair_by_stamp(ref["t"], est["t"], tol)
    if len(i) < 3:
        raise ValueError(f"solo {len(i)} pares entre la referencia y la estimación")
    lost = est["lost"][j] if "lost" in est else np.zeros(len(j), bool)
    out = evaluate_paired(ref["t"][i], est["T"][j, :3, 3], np.stack([matrix_to_quat(r) for r in est["T"][j, :3, :3]]),
                          ref["T"][i, :3, 3], np.stack([matrix_to_quat(r) for r in ref["T"][i, :3, :3]]),
                          lost=lost, rpe_horizons_s=cfg["rpe_horizons_s"])
    est_valid = int((~est["lost"]).sum()) if "lost" in est else len(est["t"])
    out["pairs"] = int(len(i))
    out["coverage_pct"] = float(100.0 * len(i) / max(1, len(est["t"])))
    out["est_frames"], out["est_valid_frames"] = int(len(est["t"])), est_valid
    out["est_lost_pct_all"] = float(100.0 * (1 - est_valid / max(1, len(est["t"]))))
    return out


def flatten(m: Dict) -> Dict[str, float]:
    out = {}
    for key, sub in FLAT_KEYS:
        if key.startswith("rpe_"):
            h, what = key[4:].split("_", 1)
            v = m["rpe"].get(h, {}).get("trans_mm" if what.startswith("trans") else "rot_deg", {}).get(sub)
        else:
            v = m.get(key) if sub is None else m.get(key, {}).get(sub)
        out[key if sub is None else f"{key}.{sub}"] = float("nan") if v is None else float(v)
    out["lost_pct_all"] = float(m.get("est_lost_pct_all", float("nan")))
    return out


def summarize(flat_runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = flat_runs[0].keys() if flat_runs else []
    out = {}
    for k in keys:
        a = np.array([r[k] for r in flat_runs], dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a):
            out[k] = {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                      "max": float(a.max()), "min": float(a.min())}
    return out


def acceptance(summary: Dict, acc: dict) -> Dict[str, bool]:
    def med(k):
        return summary.get(k, {}).get("median", float("nan"))
    return {
        f"ATE anclado mediana <= {acc['ate_anchored_median_mm']} mm": med("ate_anchored_mm.median") <= acc["ate_anchored_median_mm"],
        f"ATE anclado p90 <= {acc['ate_anchored_p90_mm']} mm": med("ate_anchored_mm.p90") <= acc["ate_anchored_p90_mm"],
        f"RPE 0,5 s mediana <= {acc['rpe_0p5s_median_mm']} mm": med("rpe_0.5s_trans_mm.median") <= acc["rpe_0p5s_median_mm"],
        f"|escala| <= {acc['scale_error_abs_pct']} %": abs(med("scale_error_pct")) <= acc["scale_error_abs_pct"],
        f"rotación mediana <= {acc['rot_anchored_median_deg']}°": med("rot_anchored_deg.median") <= acc["rot_anchored_median_deg"],
        f"pérdidas <= {acc['lost_pct']} %": summary.get("lost_pct_all", {}).get("max", 0.0) <= acc["lost_pct"],
    }


# ---------------------------------------------------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------------------------------------------------
def write_report(out_dir: Path, per_run: Dict, summaries: Dict, accept: Dict, notes: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {run: {lab: {k: v for k, v in m.items() if k != "error_series"} for lab, m in d.items()}
            for run, d in per_run.items()}
    (out_dir / "pose_evaluation.json").write_text(json.dumps(
        {"runs": slim, "summary": summaries, "acceptance": accept, "notes": notes}, indent=2, ensure_ascii=False))
    md = ["# Odometría de la D435i contra el tablero ChArUco", "",
          f"Corridas: {len(per_run)}. Cifras por estimador: **mediana** [p90, máx] entre corridas; nunca el mejor "
          "caso.", ""]
    md += [f"- {n}" for n in notes] + [""]
    cols = ["ate_anchored_mm.median", "ate_anchored_mm.p90", "final_error_anchored_mm", "scale_error_pct",
            "rot_anchored_deg.median", "rpe_0.5s_trans_mm.median", "lost_pct_all", "coverage_pct"]
    names = ["ATE anclado mediana (mm)", "ATE anclado p90 (mm)", "error final (mm)", "escala (%)",
             "rotación mediana (°)", "RPE 0,5 s (mm)", "pérdidas (%)", "pares con tablero (%)"]
    md.append("| Estimador | " + " | ".join(names) + " | Cumple |")
    md.append("|---|" + "---|" * (len(cols) + 1))
    for lab, s in summaries.items():
        cells = []
        for c in cols:
            v = s.get(c)
            cells.append("—" if v is None else f"**{v['median']:.2f}** [{v['p90']:.2f}, {v['max']:.2f}]")
        ok = accept.get(lab, {})
        md.append(f"| {lab} | " + " | ".join(cells) + f" | {sum(ok.values())}/{len(ok)} |")
    md += ["", "## Criterios (propuestos en el plan; se fijan tras la medición base)", ""]
    for lab, ok in accept.items():
        md.append(f"- **{lab}:** " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in ok.items()))
    (out_dir / "pose_evaluation.md").write_text("\n".join(md) + "\n")


def plot(out_dir: Path, per_run: Dict, summaries: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summaries)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d6a12a", "#d62a6b", "#2ab9d6", "#6b6b6b"]
    runs = list(per_run)[:6]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    for li, lab in enumerate(labels):
        for ri, run in enumerate(runs):
            m = per_run[run].get(lab)
            if not m:
                continue
            s = m["error_series"]
            ax.plot(s["t"], s["ate_anchored_mm"], color=colors[li % len(colors)], lw=1.4, alpha=0.85,
                    label=lab if ri == 0 else None)
    ax.set_title("A) Anchored position error vs. ChArUco board (each line = one run)", fontweight="bold")
    ax.set_xlabel("Time since first paired frame (s)")
    ax.set_ylabel("Anchored position error (mm)")
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    data = [[per_run[r][lab]["ate_anchored_mm"]["median"] for r in per_run if lab in per_run[r]] for lab in labels]
    ax.boxplot(data, vert=True, showfliers=True)
    for li, d in enumerate(data):
        ax.scatter(np.full(len(d), li + 1) + np.linspace(-0.08, 0.08, len(d)), d, s=18,
                   color=colors[li % len(colors)], zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("B) Per-run median anchored error", fontweight="bold")
    ax.set_ylabel("Median anchored error (mm)")
    ax.grid(True, axis="y", alpha=0.4)
    for sp in ("top", "right"):
        axs[0].spines[sp].set_visible(False)
        axs[1].spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "pose_evaluation.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "pose_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="bag de una corrida (repetible)")
    ap.add_argument("--estimate", action="append", default=[],
                    help="ETIQUETA=BAG:TÓPICO; {run} se reemplaza por el nombre del bag de la corrida")
    ap.add_argument("--no-live", action="store_true", help="no evaluar la odometría en vivo del propio bag")
    ap.add_argument("--ground-truth-topic", default="", help="bags sintéticos: evaluar también contra la verdad")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--allow-nominal", action="store_true")
    ap.add_argument("--board-config", default=None, help="YAML del tablero (por defecto, el de vio_measurement.yaml)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.board_config:
        cfg["board_config"] = str(Path(a.board_config).resolve())
    notes: List[str] = []
    per_run: Dict[str, Dict[str, Dict]] = {}
    for run in a.run:
        run = Path(run)
        name = run.name
        ref = board_reference(run, cfg, a.allow_nominal)
        if not ref["board_measured"]:
            notes.append("Tablero con medidas NOMINALES: la escala de la referencia hereda el error de impresión.")
        cov = 100.0 * len(ref["t"]) / max(1, ref["n_images"])
        print(f"[{name}] tablero en {len(ref['t'])}/{ref['n_images']} imágenes ({cov:.0f} %), reproyección mediana "
              f"{np.median(ref['reprojection_px']) if len(ref['t']) else float('nan'):.2f} px")
        estimates: List[Tuple[str, Path, str]] = []
        if not a.no_live and cfg["live_odom_topic"] in bag_topics(run):
            estimates.append(("live", run, cfg["live_odom_topic"]))
        for spec in a.estimate:
            lab, rest = spec.split("=", 1)
            bag, topic = rest.rsplit(":", 1)
            estimates.append((lab, Path(bag.replace("{run}", name)), topic))
        per_run[name] = {}
        gt = ground_truth(run, a.ground_truth_topic) if a.ground_truth_topic else None
        if gt is not None:
            m = compare(gt, {"t": ref["t"], "T": ref["T"]}, cfg)       # la propia referencia contra la verdad
            per_run[name]["board_vs_truth"] = m
            print(f"[{name}] referencia del tablero vs verdad: ATE anclado mediana "
                  f"{m['ate_anchored_mm']['median']:.2f} mm, escala {m['scale_error_pct']:+.2f} %")
        for lab, bag, topic in estimates:
            try:
                est = estimate_in_optical(bag, topic, run, cfg)
            except (ValueError, OSError, RuntimeError) as e:
                print(f"[{name}] {lab}: sin datos ({e})")
                continue
            if est["tf_source"] != "tf_static":
                notes.append(f"{name}/{lab}: TF cámara-óptica = {est['tf_source']}.")
            try:
                m = compare(ref, est, cfg)
            except ValueError as e:
                print(f"[{name}] {lab}: {e}")
                continue
            per_run[name][lab] = m
            print(f"[{name}] {lab:>18}: ATE anclado mediana {m['ate_anchored_mm']['median']:6.2f} mm, p90 "
                  f"{m['ate_anchored_mm']['p90']:6.2f}, final {m['final_error_anchored_mm']:6.2f} mm, escala "
                  f"{m['scale_error_pct']:+.2f} %, pérdidas {m['est_lost_pct_all']:.1f} %, pares {m['pairs']}")
            if gt is not None:
                mg = compare(gt, est, cfg)
                per_run[name][f"{lab}_vs_truth"] = mg
    labels = sorted({lab for d in per_run.values() for lab in d})
    summaries = {lab: summarize([flatten(d[lab]) for d in per_run.values() if lab in d]) for lab in labels}
    accept = {lab: acceptance(s, cfg["acceptance"]) for lab, s in summaries.items()}
    out = Path(a.out_dir)
    if not out.is_absolute():
        out = REPO / out
    write_report(out, per_run, summaries, accept, sorted(set(notes)))
    plot(out, per_run, summaries)
    print(f"-> {out}/pose_evaluation.(json|md|png|pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
