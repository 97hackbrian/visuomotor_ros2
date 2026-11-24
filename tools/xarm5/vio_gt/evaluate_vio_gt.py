#!/usr/bin/env python3
"""Odometría de la D435i contra el xArm5 (referencia) y calibración cámara-brida con el tablero.

Para una corrida de ``run_vio_gt.sh`` (carpeta con robot.npz, plan.json, clock.json y bag_path.txt):

1. **Cámara-brida** (planes ``static``): en cada parada, pose media del TCP (robot quieto) y del tablero (PnP en
   las imágenes de la ventana). ``cv2.calibrateHandEye`` (todos los métodos) da ``T_tcp_óptico``. La escala del
   tablero se ajusta con el robot: el factor que minimiza la dispersión del tablero en la base (equivale a medir los
   cuadros con el robot).
2. **Odometría vs. robot:** la pose de ``camera_link`` según el robot (``T_base_tcp · T_tcp_link``) se interpola en
   los sellos de la odometría. Relojes: desfase PC-Jetson medido por ssh y refinado por correlación de velocidades.
   Métricas de ``evaluate_paired`` (ATE anclada al primer frame, Sim(3) con escala, RPE a 0,25/0,5/1 s, pérdidas).

Uso (contenedor, con ROS para leer el bag):
    python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --run experiments/vio_gt/<fecha>/runs/<corrida> --allow-nominal
    python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --run <corrida> --hand-eye <corrida_static>/hand_eye.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("tools/hardware", "tools/xarm5", "visuomotor_bringup", "visuomotor_core"):
    sys.path.insert(0, str(REPO / p))

from vio_bag_utils import (StaticTF, decode_image, inv_T, iter_messages, make_T, matrix_to_quat,  # noqa: E402
                           odometry_from_bag, quat_to_matrix, stamp_of)
from calibrate_hand_eye import consistency, board_in_base, rotation_diversity_deg, METHODS  # noqa: E402
from visuomotor_core.analysis.trajectory_metrics import quat_to_matrix as tm_quat_to_matrix  # noqa: E402
from visuomotor_core.analysis.trajectory_metrics import (evaluate_paired,  # noqa: E402
                                                         interpolate_poses)

ODOM = "/rtabmap/odom"
COLOR = "/camera/camera/color/image_raw/compressed"
INFO = "/camera/camera/color/camera_info"


def mean_T(Ts):
    u, _, vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    return make_T(u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt, np.mean([T[:3, 3] for T in Ts], 0))


def robot_series(run: Path, offset: float):
    d = np.load(run / "robot.npz")
    tcp = d["tcp"]
    return tcp[:, 0] + offset, tcp[:, 1:4], tcp[:, 4:8], float(d["t0"]) + offset


def static_samples(run, bag, plan, t_pc, p, q, t0_pc, cfg, board, min_corners, max_reproj):
    from visuomotor_bringup.charuco import estimate_pose
    import cv2
    wins = [(t0_pc + s + 1.5, t0_pc + e - 0.3, name) for s, e, name in plan["segments"] if name.startswith("static")]
    K = D = None
    per = {w[2]: [] for w in wins}
    for topic, m, _ in iter_messages(bag, [INFO, COLOR]):
        if topic == INFO:
            if K is None:
                K, D = np.array(m.k).reshape(3, 3), np.array(m.d)
            continue
        ts = stamp_of(m)
        w = next((w for w in wins if w[0] <= ts <= w[1]), None)
        if w is None or K is None or len(per[w[2]]) >= 12:
            continue
        g = cv2.cvtColor(decode_image(m), cv2.COLOR_RGB2GRAY)
        bp = estimate_pose(g, board, cfg, K, D, min_corners=min_corners)
        if bp is not None and bp.reprojection_error_px <= max_reproj:
            per[w[2]].append((make_T(bp.rotation, bp.translation), bp.n_corners, bp.reprojection_error_px))
    samples = []
    for t_a, t_b, name in wins:
        # El registro del TCP solo guarda cambios: quieto, la pose vigente es la última anterior a la ventana
        m = (t_pc >= t_a) & (t_pc <= t_b)
        before = np.nonzero(t_pc < t_a)[0]
        if len(before):
            m[before[-1]] = True
        if m.sum() < 1 or len(per[name]) < 3:
            print(f"  {name}: descartada (TCP {int(m.sum())} muestras, tablero {len(per[name])} imágenes)")
            continue
        spread = np.ptp(p[m], axis=0).max() * 1000
        if spread > 0.5:
            print(f"  {name}: descartada, el TCP se movió {spread:.2f} mm en la ventana")
            continue
        Tb = mean_T([make_T(quat_to_matrix(qq), pp) for pp, qq in zip(p[m], q[m])])
        Tc = mean_T([x[0] for x in per[name]])
        jit = np.ptp([x[0][:3, 3] for x in per[name]], axis=0).max() * 1000
        samples.append({"name": name, "T_base_tcp": Tb, "T_cam_board": Tc, "n_img": len(per[name]),
                        "corners": int(np.median([x[1] for x in per[name]])),
                        "reproj_px": float(np.median([x[2] for x in per[name]])), "board_jitter_mm": float(jit)})
    return samples, K


def gyro_time_offset(bag, t_r, q_r, max_offset_s, grid_hz=100.0):
    """Desfase (s) entre el giróscopo de la cámara (reloj del PC) y el robot (reloj de la Jetson ya corrido por ssh),
    por correlación de la rapidez angular |w| (no depende de la extrínseca). Devuelve (desfase, correlación)."""
    tg, wg = [], []
    for _, m, _ in iter_messages(bag, ["/camera/camera/gyro/sample"]):
        tg.append(stamp_of(m))
        w = m.angular_velocity
        wg.append(np.sqrt(w.x * w.x + w.y * w.y + w.z * w.z))
    if len(tg) < 100:
        return 0.0, 0.0
    tg, wg = np.asarray(tg), np.asarray(wg)
    o = np.argsort(tg)
    tg, wg = tg[o], wg[o]
    # |w| del robot: rotación relativa entre muestras separadas ~0,1 s
    R = tm_quat_to_matrix(np.asarray(q_r))
    k = max(1, int(round(0.1 / max(np.median(np.diff(t_r)), 1e-3))))
    dR = np.einsum("nji,njk->nik", R[:-k], R[k:])
    ang = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))
    tw = 0.5 * (t_r[:-k] + t_r[k:])
    wr = ang / (t_r[k:] - t_r[:-k])
    t0, t1 = max(tg[0], tw[0]), min(tg[-1], tw[-1])
    g = np.arange(t0, t1, 1 / grid_hz)
    box = np.ones(int(0.1 * grid_hz)) / int(0.1 * grid_hz)
    a = np.convolve(np.interp(g, tg, wg), box, "same")
    b = np.interp(g, tw, wr)
    a, b = a - a.mean(), b - b.mean()
    K = int(max_offset_s * grid_hz)
    best = (-2.0, 0)
    for s in range(-K, K + 1):
        x = a[max(0, s):len(a) + min(0, s)]
        y = b[max(0, -s):len(b) + min(0, -s)]
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        if c > best[0]:
            best = (c, s)
    return best[1] / grid_hz, best[0]


def solve(samples):
    import cv2
    best = None
    B = [s["T_base_tcp"] for s in samples]
    for scale in np.linspace(0.95, 1.05, 101):
        C = [s["T_cam_board"].copy() for s in samples]
        for c in C:
            c[:3, 3] *= scale
        for name in METHODS:
            r, t = cv2.calibrateHandEye([b[:3, :3] for b in B], [b[:3, 3] for b in B], [c[:3, :3] for c in C],
                                        [c[:3, 3] for c in C], method=getattr(cv2, f"CALIB_HAND_EYE_{name}"))
            X = make_T(r, t.ravel())
            res = consistency(board_in_base(B, X, C))
            key = res["pos_median_mm"] + 2.0 * res["rot_median_deg"]
            if best is None or key < best[0]:
                best = (key, scale, name, X, res)
    _, scale, name, X, res = best
    # Residuo con la escala nominal y el mismo método, para comparar
    C1 = [s["T_cam_board"] for s in samples]
    r, t = cv2.calibrateHandEye([b[:3, :3] for b in B], [b[:3, 3] for b in B], [c[:3, :3] for c in C1],
                                [c[:3, 3] for c in C1], method=getattr(cv2, f"CALIB_HAND_EYE_{name}"))
    res_nom = consistency(board_in_base(B, make_T(r, t.ravel()), C1))
    return {"method": name, "board_scale": float(scale), "T_tcp_optical": X, "residual": res,
            "residual_nominal_scale": res_nom, "rotation_diversity_deg": rotation_diversity_deg(B)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hand-eye", default="", help="hand_eye.json de una corrida static (si no, se calcula aquí)")
    ap.add_argument("--board-config", default=str(REPO / "visuomotor_bringup/config/realsense/charuco_board.yaml"))
    ap.add_argument("--allow-nominal", action="store_true")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--max-reproj-px", type=float, default=1.0)
    ap.add_argument("--max-offset-s", type=float, default=1.0)
    ap.add_argument("--odom-bag", default="", help="bag con otra odometría (p. ej. un candidato reprocesado)")
    ap.add_argument("--odom-topic", default=ODOM)
    ap.add_argument("--label", default="live", help="sufijo de las salidas (vio_vs_robot_<label>.json/png)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    plan = json.loads((run / "plan.json").read_text())
    offset = json.loads((run / "clock.json").read_text())["offset_pc_minus_jetson_s"]
    bag = REPO / (run / "bag_path.txt").read_text().strip()
    t_r, p_r, q_r, t0_pc = robot_series(run, offset)
    stf = StaticTF()
    for _, m, _ in iter_messages(bag, ["/tf_static"]):
        stf.add_message(m)
    odo = odometry_from_bag(bag, ODOM)
    T_link_opt = stf.lookup(odo["child"] or "camera_link", "camera_color_optical_frame")
    if T_link_opt is None:
        raise SystemExit("sin tf_static camera_link -> camera_color_optical_frame en el bag")
    report = {"run": str(run), "plan": plan["name"], "odom_child": odo["child"], "clock_offset_s": offset}

    if a.hand_eye:
        he = json.loads(Path(a.hand_eye).read_text())
        X = np.asarray(he["T_tcp_optical"])
        report["hand_eye_from"] = a.hand_eye
    else:
        from visuomotor_bringup.charuco import CharucoConfig, make_board
        cfg = CharucoConfig.load(a.board_config)
        if not cfg.measured and not a.allow_nominal:
            raise SystemExit("tablero sin medir: medir y poner measured: true, o --allow-nominal")
        samples, _ = static_samples(run, bag, plan, t_r, p_r, q_r, t0_pc, cfg, make_board(cfg), a.min_corners,
                                    a.max_reproj_px)
        print(f"{len(samples)} paradas válidas para la calibración cámara-brida")
        he = solve(samples)
        X = he["T_tcp_optical"]
        T_tcp_link = X @ inv_T(T_link_opt)
        he_out = {"method": he["method"], "board_scale": he["board_scale"],
                  "square_length_from_robot_m": he["board_scale"] * cfg.square_length_m,
                  "T_tcp_optical": X.tolist(), "T_tcp_camera_link": T_tcp_link.tolist(),
                  "tcp_to_camera_link_xyz_m": T_tcp_link[:3, 3].tolist(),
                  "tcp_to_camera_link_quat_xyzw": matrix_to_quat(T_tcp_link[:3, :3]).tolist(),
                  "residual": he["residual"], "residual_nominal_scale": he["residual_nominal_scale"],
                  "rotation_diversity_deg": he["rotation_diversity_deg"], "n_samples": len(samples),
                  "samples": [{k: v for k, v in s.items() if not k.startswith("T_")} for s in samples]}
        (run / "hand_eye.json").write_text(json.dumps(he_out, indent=2))
        report["hand_eye"] = {k: v for k, v in he_out.items() if k != "samples"}
        print(json.dumps({k: he_out[k] for k in ("method", "board_scale", "square_length_from_robot_m",
                                                 "tcp_to_camera_link_xyz_m", "residual", "residual_nominal_scale",
                                                 "rotation_diversity_deg", "n_samples")}, indent=1))

    T_tcp_link = X @ inv_T(T_link_opt)
    if a.odom_bag:                                   # odometría reprocesada (replay_odometry_candidates.py)
        odo = odometry_from_bag(REPO / a.odom_bag, a.odom_topic)
        report["odom_bag"], report["odom_topic"] = a.odom_bag, a.odom_topic
    # Odometría desde el reinicio (primer mensaje) hasta el final del registro del robot
    t_o, T_o, lost = odo["t"], odo["T"], odo["lost"]
    keep = (t_o >= t_r[0] + 0.2) & (t_o <= t_r[-1] - 0.2)
    t_o, T_o, lost = t_o[keep], T_o[keep], lost[keep]

    def anchored(Ts):
        A = inv_T(Ts[0])
        return np.einsum("ij,njk->nik", A, Ts)
    # Relojes: el desfase por ssh se refina con el giróscopo de la cámara contra la velocidad angular del robot
    # (independiente de la odometría evaluada). Positivo: la cámara llega tarde respecto al robot.
    d, corr = gyro_time_offset(bag, t_r, q_r, a.max_offset_s)
    report["time_offset_gyro_s"], report["time_offset_gyro_corr"] = d, corr
    if corr < 0.6:
        print(f"AVISO: correlación del giróscopo baja ({corr:.2f}): se usa solo el desfase medido por ssh")
        d = 0.0
    report["time_offset_used_s"] = d
    p_rob, q_rob, _ = interpolate_poses(t_o - d, t_r, p_r, q_r)
    T_rob = np.einsum("nij,jk->nik", np.array([make_T(quat_to_matrix(qq), pp) for pp, qq in zip(p_rob, q_rob)]),
                      T_tcp_link)
    q_est = np.array([matrix_to_quat(T[:3, :3]) for T in T_o])
    q_ref = np.array([matrix_to_quat(T[:3, :3]) for T in T_rob])
    res = evaluate_paired(t_o, T_o[:, :3, 3], q_est, T_rob[:, :3, 3], q_ref, lost=lost)
    series = res.pop("error_series")
    report["odometry_vs_robot"] = res
    (run / f"vio_vs_robot_{a.label}.json").write_text(json.dumps(report, indent=2))
    r = res
    print(f"odometría vs robot ({plan['name']}): recorrido {r['path_length_m']:.2f} m, {r['duration_s']:.0f} s, "
          f"pérdidas {r['lost_pct']:.1f} %, desfase por giróscopo {d * 1000:+.0f} ms (corr {corr:.2f})")
    print(f"  ATE anclada: mediana {r['ate_anchored_mm']['median']:.1f} mm, p90 {r['ate_anchored_mm']['p90']:.1f}, "
          f"máx {r['ate_anchored_mm']['max']:.1f}, final {r['final_error_anchored_mm']:.1f} mm | rotación mediana "
          f"{r['rot_anchored_deg']['median']:.2f}°")
    print(f"  ATE Sim(3): mediana {r['ate_sim3_mm']['median']:.1f} mm | escala {r['scale']:.4f} "
          f"({r['scale_error_pct']:+.2f} %) | RPE 0,5 s mediana {r['rpe']['0.5s']['trans_mm']['median']:.2f} mm")
    # Figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A_o, A_r = anchored(T_o), anchored(T_rob)
    tt = t_o - t_o[0]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for k, lab in enumerate("xyz"):
        ax[0, 0].plot(tt, A_r[:, k, 3] * 1000, color=f"C{k}", lw=2, alpha=0.5, label=f"{lab} robot")
        ax[0, 0].plot(tt, A_o[:, k, 3] * 1000, color=f"C{k}", lw=1, ls="--", label=f"{lab} odometry")
    ax[0, 0].set(title="camera_link motion from the first frame", xlabel="time (s)", ylabel="mm")
    ax[0, 0].legend(ncol=2, fontsize=8)
    ax[0, 1].plot(series["t"], series["ate_anchored_mm"], lw=1)
    if lost.any():
        ax[0, 1].scatter(tt[lost], np.zeros(lost.sum()), color="C3", s=6, label="lost")
        ax[0, 1].legend()
    ax[0, 1].set(title="anchored position error", xlabel="time (s)", ylabel="mm")
    ax[1, 0].plot(A_r[:, 0, 3] * 1000, A_r[:, 1, 3] * 1000, lw=2, alpha=0.5, label="robot")
    ax[1, 0].plot(A_o[:, 0, 3] * 1000, A_o[:, 1, 3] * 1000, lw=1, ls="--", label="odometry")
    ax[1, 0].set(title="x-y (camera_link frame at start)", xlabel="mm", ylabel="mm", aspect="equal")
    ax[1, 0].legend()
    ax[1, 1].axis("off")
    ax[1, 1].text(0, 1, json.dumps({"ATE anchored median mm": round(r["ate_anchored_mm"]["median"], 1),
                                     "ATE anchored p90 mm": round(r["ate_anchored_mm"]["p90"], 1),
                                     "ATE anchored max mm": round(r["ate_anchored_mm"]["max"], 1),
                                     "scale error %": round(r["scale_error_pct"], 2),
                                     "RPE 0.5 s median mm": round(r["rpe"]["0.5s"]["trans_mm"]["median"], 2),
                                     "lost %": round(r["lost_pct"], 2), "path m": round(r["path_length_m"], 2),
                                     "gyro clock offset ms": round(d * 1000, 1)}, indent=1),
                  va="top", family="monospace", fontsize=9)
    fig.suptitle(f"D435i odometry ({a.label}) vs xArm5 — {plan['name']}")
    fig.tight_layout()
    fig.savefig(run / f"vio_vs_robot_{a.label}.png", dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
