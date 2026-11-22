#!/usr/bin/env python3
"""Espacio de trabajo del xArm5 (XF1305) frente a las demostraciones convertidas (solo lectura).

Separa tres cosas que se confunden al hablar de "alcance":

1. Especificación del fabricante (bio/xarm5_ufactory_XF1305/): alcance nominal 700 mm y rango cartesiano X/Y ±700 mm,
   Z -400..951.5 mm (Hardware Manual v2.7.1 §8.1, Developer Manual v2.0.1 §4.1); envolvente de los eslabones radio
   770 mm, +1037/-416 mm (Hardware Manual §2.2.2, "end-effector not included"). Son cotas globales: no dicen hasta
   dónde llega la brida a una altura y con una inclinación dadas.
2. Alcance cinemático a la altura y con la inclinación de la tarea: la muñeca (J4) debe quedar a <= L2 + L3 del hombro
   (J2, z = 0.267 m); con el codo recto hay singularidad (C24), por eso el puente usa el 95 % (workspace_max_reach_m).
   Se evalúa con la misma ``ReachLimit`` del puente (xarm5_interface.py).
3. Caja de seguridad configurada en el puente (xarm5_real_params.yaml): conservadora desde la integración inicial
   ("ampliar tras medir la celda real"). Es una decisión de seguridad de la celda, no un límite del brazo.

Las posiciones del dataset son relativas a HOME (relative_coordinates): se suman ``home_xyz_m`` de los atributos.
No modela los límites articulares (requiere IK); el guard de joint_limit_margin_rad del puente los cubre.

Salidas en --out-dir: workspace_summary.json y fig_xarm5_workspace_vs_demos.png/.pdf (inglés, 300 DPI).

Uso (contenedor):
    python3 tools/xarm5/analyze_xarm5_workspace.py --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr \\
        --out-dir datasets/evaluations_real_v7_xarm5_v2 [--propose-y-max 0.45]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))

from visuomotor_bringup.xarm5_interface import (ReachLimit, WorkspaceLimits, arm_azimuth,  # noqa: E402
                                                arm_plane_params)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFAULT_BRIDGE = REPO / "visuomotor_bringup" / "config" / "xarm5" / "xarm5_real_params.yaml"
# Paleta categórica validada (skill dataviz; orden fijo): alcance, caja del puente, demostraciones. Neutros: specs.
C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10})


def load_limits(path: Path, y_override: float | None = None):
    b = yaml.safe_load(open(path))["xarm5_bridge"]["ros__parameters"]
    y_min, y_max = (-y_override, y_override) if y_override else (b["workspace_y_min"], b["workspace_y_max"])
    box = WorkspaceLimits(b["workspace_x_min"], b["workspace_x_max"], y_min, y_max, b["workspace_z_min"],
                          b["workspace_z_max"], b["workspace_min_radius_xy"], b["workspace_max_radius_xy"])
    reach = ReachLimit(b["workspace_shoulder_xyz_m"], b["workspace_wrist_offset_m"], b["workspace_max_reach_m"],
                       b["workspace_min_reach_m"])
    return box, reach, b


def load_demos(dataset: str):
    g = zarr.open_group(dataset, mode="r")
    names = sorted(g.group_keys(), key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else n)
    demos = []
    for n in names:
        ep = g[n]
        st = np.asarray(ep["state"][:], dtype=np.float64)
        home = np.asarray(ep.attrs.get("home_xyz_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        p = st[:, :3] + (home if ep.attrs.get("relative_coordinates", False) else 0.0)
        q = st[:, 3:7]
        pitch = np.array([arm_plane_params(qi, arm_azimuth(pi))[0] for pi, qi in zip(p, q)])
        demos.append({"name": n, "p": p, "pitch": pitch})
    return demos


def lateral_limit(reach: ReachLimit, x: float, z: float, pitch: float, max_d: float) -> float:
    """Mayor |y| con la muñeca a <= max_d del hombro, a profundidad x y altura z (bisección; simétrico en y)."""
    if reach.reach([x, 0.0, z], pitch) > max_d:
        return float("nan")
    lo, hi = 0.0, 1.2
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if reach.reach([x, mid, z], pitch) <= max_d:
            lo = mid
        else:
            hi = mid
    return lo


def box_outline(box: WorkspaceLimits, n: int = 400):
    """Contorno (x, y) de caja ∩ anillo radial del puente."""
    t = np.linspace(0, 1, n)
    pts = np.concatenate([
        np.stack([box.x_min + (box.x_max - box.x_min) * t, np.full(n, box.y_min)], 1),
        np.stack([np.full(n, box.x_max), box.y_min + (box.y_max - box.y_min) * t], 1),
        np.stack([box.x_max - (box.x_max - box.x_min) * t, np.full(n, box.y_max)], 1),
        np.stack([np.full(n, box.x_min), box.y_max - (box.y_max - box.y_min) * t], 1)])
    out = np.array([box.clamp([x, y, 0.5 * (box.z_min + box.z_max)])[0][:2] for x, y in pts])
    return out


def analyze(args) -> dict:
    box, reach, b = load_limits(Path(args.bridge_params))
    L = float(args.straight_elbow_m)
    demos = load_demos(args.dataset)
    allp = np.concatenate([d["p"] for d in demos])
    allpitch = np.concatenate([d["pitch"] for d in demos])
    d_all = np.array([reach.reach(p, a) for p, a in zip(allp, allpitch)])
    clamp = [box.clamp(p) for p in allp]
    clamp_dist = np.array([np.linalg.norm(c[0] - p) for c, p in zip(clamp, allp)])
    tol = float(args.tolerance_m)
    ends = np.array([d["p"][-1] for d in demos])
    end_pitch = np.array([d["pitch"][-1] for d in demos])
    end_out = np.array([np.linalg.norm(box.clamp(p)[0] - p) > tol for p in ends])
    lat95 = np.array([lateral_limit(reach, p[0], p[2], a, reach.max_reach) for p, a in zip(ends, end_pitch)])
    lat100 = np.array([lateral_limit(reach, p[0], p[2], a, L) for p, a in zip(ends, end_pitch)])
    lat_box = np.array([min(box.y_max, math.sqrt(max(box.max_radius_xy ** 2 - p[0] ** 2, 0.0))) for p in ends])
    home = demos[0]["p"][0]
    beyond_eps = {}
    for dm in demos:
        dd = np.array([reach.reach(p, a) for p, a in zip(dm["p"], dm["pitch"])])
        dm["wrist"] = np.array([reach.wrist_center(p, a) for p, a in zip(dm["p"], dm["pitch"])])
        dm["beyond"] = dd > reach.max_reach
        if dm["beyond"].any():
            beyond_eps[dm["name"]] = {"frames": int(dm["beyond"].sum()),
                                      "max_excess_mm": float(1000.0 * (dd.max() - reach.max_reach)),
                                      "beyond_straight_elbow": bool(dd.max() > L)}
    out = {
        "dataset": str(args.dataset), "episodes": len(demos), "frames": int(len(allp)),
        "manufacturer_spec": {
            "model": "UFACTORY xArm 5, XF1305", "nominal_reach_m": args.spec_reach_m,
            "cartesian_range_m": {"x": [-0.70, 0.70], "y": [-0.70, 0.70], "z": [-0.40, 0.9515]},
            "link_envelope_m": {"radius": args.spec_envelope_m, "above_base": 1.037, "below_base": 0.416},
            "joint_ranges_deg": {"J1": [-360, 360], "J2": [-117, 116], "J3": [-219, 10], "J4": [-97, 180],
                                 "J5": [-360, 360]},
            "sources": ["bio/xarm5_ufactory_XF1305/xArm_Hardware_Manual_v2.7.1.pdf (Preface, §2.2.2, §8.1-8.2)",
                        "bio/xarm5_ufactory_XF1305/xArm_Developer_Manual_v2.0.1.pdf (§4.1)"]},
        "kinematic_model": {"shoulder_xyz_m": list(map(float, reach.shoulder_xyz)),
                            "wrist_offset_m": list(map(float, reach.wrist_offset)),
                            "straight_elbow_m": L, "bridge_max_reach_m": reach.max_reach,
                            "bridge_reach_fraction": reach.max_reach / L, "min_reach_m": reach.min_reach},
        "bridge_box": {"x": [box.x_min, box.x_max], "y": [box.y_min, box.y_max], "z": [box.z_min, box.z_max],
                       "radius_xy": [box.min_radius_xy, box.max_radius_xy]},
        "demos": {
            "home_xyz_m": home.tolist(),
            "x_range_m": [float(allp[:, 0].min()), float(allp[:, 0].max())],
            "y_range_m": [float(allp[:, 1].min()), float(allp[:, 1].max())],
            "z_range_m": [float(allp[:, 2].min()), float(allp[:, 2].max())],
            "radius_xy_max_m": float(np.hypot(allp[:, 0], allp[:, 1]).max()),
            "endpoint_z_median_m": float(np.median(ends[:, 2])),
            "endpoint_pitch_median_rad": float(np.median(end_pitch)),
            "shoulder_wrist_max_m": float(d_all.max()), "straight_elbow_utilization_max": float(d_all.max() / L),
            "frames_outside_bridge_box_fraction": float(np.mean(clamp_dist > tol)),
            "frames_outside_by_y_fraction": float(np.mean((allp[:, 1] < box.y_min - tol) | (allp[:, 1] > box.y_max + tol))),
            "frames_beyond_bridge_reach_fraction": float(np.mean(d_all > reach.max_reach)),
            "endpoints_outside_bridge_box": int(end_out.sum()),
            "frames_beyond_straight_elbow_fraction": float(np.mean(d_all > L)),
            "episodes_beyond_bridge_reach": beyond_eps,
            "endpoint_lateral_limit_m": {
                "bridge_reach_95pct_median": float(np.nanmedian(lat95)), "bridge_reach_95pct_min": float(np.nanmin(lat95)),
                "straight_elbow_median": float(np.nanmedian(lat100)),
                "bridge_box_and_radius_median": float(np.median(lat_box)),
                "demo_abs_y_max": float(np.abs(ends[:, 1]).max())}},
    }
    if args.propose_y_max:
        pbox, _, _ = load_limits(Path(args.bridge_params), args.propose_y_max)
        pd = np.array([np.linalg.norm(pbox.clamp(p)[0] - p) for p in allp])
        out["proposed_box"] = {"y": [-args.propose_y_max, args.propose_y_max],
                               "frames_outside_fraction": float(np.mean(pd > tol)),
                               "endpoints_outside": int(sum(np.linalg.norm(pbox.clamp(p)[0] - p) > tol for p in ends))}
    plot(out, demos, box, reach, L, ends, end_out, Path(args.out_dir) / "fig_xarm5_workspace_vs_demos", args)
    return out


def plot(out, demos, box, reach, L, ends, end_out, path_base: Path, args):
    z_t = out["demos"]["endpoint_z_median_m"]
    pitch_t = out["demos"]["endpoint_pitch_median_rad"]
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 9.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    fig.suptitle(f"xArm 5 (XF1305) reach vs. bridge safety box vs. demonstrations — "
                 f"{Path(args.dataset).name.replace('.zarr', '')} ({len(demos)} episodes)", fontsize=14,
                 fontweight="bold")

    # A) vista superior de la brida a la altura de aproximación (región exacta para esa altura e inclinación)
    ax = axs[0]
    X, Y = np.meshgrid(np.linspace(-0.35, 0.85, 241), np.linspace(-0.85, 1.0, 371))
    D = np.vectorize(lambda x, y: reach.reach([x, y, z_t], pitch_t))(X, Y)
    ok = (D <= reach.max_reach) & (D >= reach.min_reach)
    ax.contourf(X, Y, ok.astype(float), levels=[0.5, 1.5], colors=[C1], alpha=0.16)
    ax.contour(X, Y, D, levels=[reach.max_reach], colors=[C1], linewidths=2.0)
    ax.contour(X, Y, D, levels=[L], colors=[C1], linewidths=1.2, linestyles="--")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(args.spec_reach_m * np.cos(th), args.spec_reach_m * np.sin(th), color=GRAY, ls=":", lw=1.6)
    ob = box_outline(box)
    ax.plot(ob[:, 0], ob[:, 1], color=C2, lw=2.2)
    for d in demos:
        ax.plot(d["p"][:, 0], d["p"][:, 1], color=C3, lw=0.7, alpha=0.35)
    ax.scatter(ends[~end_out, 0], ends[~end_out, 1], s=26, color=C3, edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(ends[end_out, 0], ends[end_out, 1], s=70, color=C3, marker="X", edgecolors=INK, linewidths=0.8, zorder=5)
    home = out["demos"]["home_xyz_m"]
    ax.scatter([home[0]], [home[1]], s=180, marker="*", color=INK, zorder=6)
    ax.scatter([0], [0], s=60, marker="s", color=INK, zorder=6)
    ax.annotate("base (J1)", (0, 0), xytext=(-6, -14), textcoords="offset points", fontsize=9, color=INK, ha="right")
    lat = out["demos"]["endpoint_lateral_limit_m"]
    ax.set_title(f"A) Top view of the flange at the approach height (z = {z_t:.2f} m, tool pitch "
                 f"{math.degrees(pitch_t):.1f}°)", fontweight="bold", loc="left", fontsize=11)
    ax.set_xlabel("x in link_base (m)")
    ax.set_ylabel("y in link_base (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 0.85)
    ax.set_ylim(-0.85, 1.0)
    ax.grid(True)
    ax.text(0.98, 0.985, f"Lateral limit at the demo endpoints (median):\n"
                         f"  kinematic, bridge 95%: ±{lat['bridge_reach_95pct_median']:.2f} m\n"
                         f"  configured box ∩ radius: ±{lat['bridge_box_and_radius_median']:.2f} m\n"
                         f"  demonstrations reached: |y| ≤ {lat['demo_abs_y_max']:.2f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d9d8d4"))
    handles = [Patch(facecolor=C1, alpha=0.25, edgecolor=C1, label=f"Reachable at this height (bridge: wrist ≤ "
                                                                     f"{reach.max_reach:.3f} m = 95% straight elbow)"),
               Line2D([], [], color=C1, ls="--", lw=1.2, label=f"Straight elbow ({L:.3f} m, singular)"),
               Line2D([], [], color=GRAY, ls=":", lw=1.6,
                      label=f"Datasheet nominal reach ({args.spec_reach_m * 1000:.0f} mm, any height)"),
               Line2D([], [], color=C2, lw=2.2, label=f"Bridge safety box (configured: |y| ≤ {box.y_max:.2f} m, "
                                                       f"r ≤ {box.max_radius_xy:.2f} m)"),
               Line2D([], [], color=C3, lw=1.0, marker="o", ms=5, label="Demonstrations and endpoints"),
               Line2D([], [], color=C3, marker="X", ls="", ms=8, markeredgecolor=INK,
                      label=f"Endpoint outside the box ({int(end_out.sum())}/{len(ends)})"),
               Line2D([], [], color=INK, marker="*", ls="", ms=12, label="HOME3 (start of every demo)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)

    # B) plano del brazo en el espacio de la MUÑECA (J4): alcanzable <=> dentro del círculo centrado en el hombro.
    #    Exacto para cualquier inclinación de la herramienta (la brida depende de la inclinación; la muñeca no).
    ax = axs[1]
    sh = reach.shoulder_xyz
    th = np.linspace(-np.pi / 2, np.pi / 2, 300)
    xs, zs = np.cos(th), sh[2] + np.sin(th)
    ax.fill(np.concatenate([reach.max_reach * xs, reach.min_reach * xs[::-1]]),
            np.concatenate([sh[2] + reach.max_reach * np.sin(th), sh[2] + reach.min_reach * np.sin(th)[::-1]]),
            color=C1, alpha=0.16, lw=0)
    ax.plot(reach.max_reach * xs, sh[2] + reach.max_reach * np.sin(th), color=C1, lw=2.0)
    ax.plot(L * xs, sh[2] + L * np.sin(th), color=C1, lw=1.2, ls="--")
    n_beyond = 0
    for d in demos:
        rw = np.hypot(d["wrist"][:, 0], d["wrist"][:, 1])
        ax.plot(rw, d["wrist"][:, 2], color=C3, lw=0.7, alpha=0.35)
        if d["beyond"].any():
            n_beyond += int(d["beyond"].sum())
            ax.scatter(rw[d["beyond"]], d["wrist"][d["beyond"], 2], s=14, marker="x", color=INK, zorder=5,
                       linewidths=1.0)
    we = np.array([d["wrist"][-1] for d in demos])
    ax.scatter(np.hypot(we[:, 0], we[:, 1]), we[:, 2], s=26, color=C3, edgecolors="white", linewidths=0.8, zorder=4)
    w0 = demos[0]["wrist"][0]
    ax.scatter([math.hypot(w0[0], w0[1])], [w0[2]], s=180, marker="*", color=INK, zorder=6)
    ax.scatter([0], [sh[2]], s=60, marker="o", color=INK, zorder=6)
    ax.annotate("shoulder (J2)", (0, sh[2]), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_title("B) Arm plane, wrist centre (J4): reachable inside the shoulder circle", fontweight="bold",
                 loc="left", fontsize=11)
    ax.set_xlabel("Horizontal distance of the wrist from the base axis (m)")
    ax.set_ylabel("Wrist height above the base (m)")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(True)
    frac = out["demos"]["frames_beyond_bridge_reach_fraction"]
    handles = [Patch(facecolor=C1, alpha=0.25, edgecolor=C1, label=f"Reachable (bridge: {reach.min_reach:.2f}-"
                                                                     f"{reach.max_reach:.3f} m from the shoulder)"),
               Line2D([], [], color=C1, ls="--", lw=1.2, label=f"Straight elbow ({L:.3f} m, singular)"),
               Line2D([], [], color=C3, lw=1.0, marker="o", ms=5, label="Demonstrations (wrist) and endpoints"),
               Line2D([], [], color=INK, marker="x", ls="", ms=7,
                      label=f"Frames beyond the bridge reach ({n_beyond}, {100 * frac:.1f}%)"),
               Line2D([], [], color=INK, marker="*", ls="", ms=12, label="HOME3 (start)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="dataset convertido al xArm5 (salida de convert_vio_dataset_to_xarm5)")
    ap.add_argument("--bridge-params", default=str(DEFAULT_BRIDGE))
    ap.add_argument("--out-dir", required=True)
    # URDF xarm5: |J2->J3| = |(0.0535, 0.2845)| = 0.28949 m, |J3->J4| = |(0.0775, 0.3425)| = 0.35116 m
    ap.add_argument("--straight-elbow-m", type=float, default=0.6406, help="L2 + L3 (hombro-muñeca con codo recto)")
    ap.add_argument("--spec-reach-m", type=float, default=0.70, help="alcance nominal (manual §8.1 / §4.1)")
    ap.add_argument("--spec-envelope-m", type=float, default=0.77, help="radio de la envolvente (manual §2.2.2)")
    ap.add_argument("--tolerance-m", type=float, default=0.03, help="recortes menores no cuentan (= conversor)")
    ap.add_argument("--propose-y-max", type=float, default=0.0, help="evaluar una caja con |y| <= este valor")
    args = ap.parse_args(argv)
    out = analyze(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "workspace_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("demos", "bridge_box", "kinematic_model")} |
                     ({"proposed_box": out["proposed_box"]} if "proposed_box" in out else {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
