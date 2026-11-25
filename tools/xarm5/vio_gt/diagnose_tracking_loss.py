#!/usr/bin/env python3
"""¿Por qué se pierde la odometría? Profundidad válida, distancia a la mesa y textura frente a las pérdidas.

Para una corrida de ``run_vio_gt.sh``, en el tiempo:
* fracción de píxeles con profundidad válida (``aligned_depth_to_color``, 0 = sin dato) y su mediana;
* altura de la cámara sobre la mesa según el ROBOT (TCP · extrínseca medida; mesa = plano z del tablero);
* esquinas GFTT en la imagen de color (proxy de textura; mismos parámetros por defecto de OpenCV);
* pérdidas de la odometría en vivo (y de un candidato reprocesado con ``--odom-bag``).

Hipótesis H4 del plan: al final de la aproximación la cámara queda más cerca de la mesa que el mínimo de profundidad
de la D435i, la profundidad desaparece y la odometría RGB-D se pierde. Salida: ``tracking_loss.json`` y ``.png``.

Uso (contenedor, con ROS):
    python3 tools/xarm5/vio_gt/diagnose_tracking_loss.py --run experiments/vio_gt/<fecha>/runs/<corrida> \
        --hand-eye experiments/vio_gt/<fecha>/runs/<static>/hand_eye.json --table-z -0.016
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("tools/hardware", "visuomotor_core"):
    sys.path.insert(0, str(REPO / p))
from vio_bag_utils import decode_image, iter_messages, odometry_from_bag, stamp_of  # noqa: E402
from visuomotor_core.analysis.trajectory_metrics import interpolate_poses, quat_to_matrix  # noqa: E402

DEPTH = "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
COLOR = "/camera/camera/color/image_raw/compressed"


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hand-eye", required=True)
    ap.add_argument("--table-z", type=float, default=-0.016, help="z de la mesa en link_base (plano del tablero)")
    ap.add_argument("--every", type=int, default=6, help="analizar 1 de cada N imágenes")
    ap.add_argument("--odom-bag", default="")
    ap.add_argument("--odom-topic", default="/replay/odom")
    ap.add_argument("--clock-offset-s", type=float, default=None, help="desfase extra (giróscopo) en s")
    a = ap.parse_args(argv)
    run = Path(a.run)
    bag = REPO / (run / "bag_path.txt").read_text().strip()
    offset = json.loads((run / "clock.json").read_text())["offset_pc_minus_jetson_s"]
    extra = a.clock_offset_s
    if extra is None:
        rep = run / "vio_vs_robot_live.json"
        extra = json.loads(rep.read_text()).get("time_offset_used_s", 0.0) if rep.exists() else 0.0
    d = np.load(run / "robot.npz")
    t_r = d["tcp"][:, 0] + offset + extra
    X = np.asarray(json.loads(Path(a.hand_eye).read_text())["T_tcp_optical"])
    rows = {"depth": [], "color": []}
    k = {"depth": 0, "color": 0}
    for topic, m, _ in iter_messages(bag, [DEPTH, COLOR]):
        key = "depth" if topic == DEPTH else "color"
        k[key] += 1
        if k[key] % a.every:
            continue
        img = decode_image(m)
        ts = stamp_of(m)
        if key == "depth":
            v = img > 0
            rows["depth"].append((ts, float(v.mean()), float(np.median(img[v])) / 1000.0 if v.any() else 0.0))
        else:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            c = cv2.goodFeaturesToTrack(g, maxCorners=1000, qualityLevel=0.01, minDistance=7)
            rows["color"].append((ts, 0 if c is None else len(c)))
    D = np.array(rows["depth"])
    C = np.array(rows["color"])
    # Altura del centro óptico sobre la mesa según el robot
    p, q, valid = interpolate_poses(D[:, 0], t_r, d["tcp"][:, 1:4], d["tcp"][:, 4:8])
    R = quat_to_matrix(q)
    cam = p + np.einsum("nij,j->ni", R, X[:3, 3])
    height = cam[:, 2] - a.table_z
    live = odometry_from_bag(bag, "/rtabmap/odom")
    lost_live = np.interp(D[:, 0], live["t"], live["lost"].astype(float)) > 0.5
    out = {"run": run.name, "n_depth": int(len(D)), "n_color": int(len(C))}
    lost_alt = None
    if a.odom_bag:
        alt = odometry_from_bag(REPO / a.odom_bag, a.odom_topic)
        lost_alt = np.interp(D[:, 0], alt["t"], alt["lost"].astype(float)) > 0.5
    # Pérdidas por franja de altura y de profundidad válida
    bins = [0.0, 0.25, 0.30, 0.35, 0.45, 0.60, 2.0]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = valid & (height >= lo) & (height < hi)
        if m.sum() == 0:
            continue
        row = {"height_m": [lo, hi], "n": int(m.sum()), "valid_depth_pct": float(100 * D[m, 1].mean()),
               "lost_live_pct": float(100 * lost_live[m].mean())}
        if lost_alt is not None:
            row["lost_alt_pct"] = float(100 * lost_alt[m].mean())
        table.append(row)
    out["by_height"] = table
    tc = np.interp(D[:, 0], C[:, 0], C[:, 1])
    out["gftt_when_lost_median"] = float(np.median(tc[lost_live])) if lost_live.any() else None
    out["gftt_when_tracking_median"] = float(np.median(tc[~lost_live]))
    out["valid_depth_when_lost_pct"] = float(100 * D[lost_live, 1].mean()) if lost_live.any() else None
    out["valid_depth_when_tracking_pct"] = float(100 * D[~lost_live, 1].mean())
    out["min_height_m"] = float(np.nanmin(np.where(valid, height, np.nan)))
    (run / "tracking_loss.json").write_text(json.dumps(out, indent=2))
    print(f"{run.name}: altura mínima de la cámara sobre la mesa {out['min_height_m']:.3f} m")
    print("  altura (m)     n   prof. válida  pérdidas vivo" + ("  pérdidas alt" if lost_alt is not None else ""))
    for r in table:
        print(f"  {r['height_m'][0]:.2f}-{r['height_m'][1]:.2f}  {r['n']:5d}  {r['valid_depth_pct']:8.1f} %  "
              f"{r['lost_live_pct']:9.1f} %" + (f"  {r['lost_alt_pct']:9.1f} %" if lost_alt is not None else ""))
    print(f"  esquinas GFTT: {out['gftt_when_tracking_median']:.0f} rastreando, {out['gftt_when_lost_median']} perdida | "
          f"prof. válida: {out['valid_depth_when_tracking_pct']:.1f} % rastreando, {out['valid_depth_when_lost_pct']} % perdida")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = D[:, 0] - D[0, 0]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    ax[0].plot(t, np.where(valid, height, np.nan), lw=1.5)
    ax[0].set(ylabel="camera height above table (m)")
    ax[1].plot(t, 100 * D[:, 1], lw=1.2, label="valid depth %")
    ax[1].set(ylabel="valid depth (%)")
    ax[2].plot(np.interp(t + D[0, 0], C[:, 0], C[:, 0]) - D[0, 0], tc, lw=1.2)
    ax[2].set(ylabel="GFTT corners", xlabel="time (s)")
    for axi in ax:
        y0, y1 = axi.get_ylim()
        axi.fill_between(t, y0, y1, where=lost_live, color="C3", alpha=0.25, step="mid", label="live lost")
        if lost_alt is not None:
            axi.fill_between(t, y0, y0 + 0.1 * (y1 - y0), where=lost_alt, color="C1", alpha=0.6, step="mid",
                             label="replay lost")
    ax[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Tracking loss diagnosis — {run.name}")
    fig.tight_layout()
    fig.savefig(run / "tracking_loss.png", dpi=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
