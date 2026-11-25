#!/usr/bin/env python3
"""Calibra con el robot la posición de los ArUco de la mesa en ``link_base`` y valida el anclaje de inicio.

Entrada: una corrida ``static`` de ``run_vio_gt.sh`` (paradas del xArm5 en poses conocidas) exportada con
``export_bag_frames.py``, más la calibración cámara-brida (``hand_eye.json``).

1. En cada ventana quieta (HOME3 y paradas), la pose de la cámara es ``T_base_tcp · T_tcp_óptico`` (robot). Se toman
   las esquinas de los ArUco (mediana de varios frames, subpíxel), la profundidad alineada y el acelerómetro.
2. **Triangulación:** cada esquina se triangula en ``link_base`` con todas las ventanas. No hace falta medir los
   marcadores ni suponer que están en el plano del tablero.
3. **Gravedad en la base:** el "arriba" del acelerómetro, rotado a la base con la pose del robot y promediado, da la
   vertical real en ``link_base`` (desnivel de la base + sesgo del IMU).
4. **Validación dejando una ventana fuera** (triangulación y gravedad sin la ventana k), comparada contra el robot:
   * ``pnp``: PnP solo con los ArUco;
   * ``pnp_imu``: PnP, con la inclinación reemplazada por la del IMU;
   * ``rgbd``: alineación 3D-3D de las esquinas con la profundidad del sensor (× ``--depth-scale``);
   * ``hybrid``: rotación de ``pnp_imu`` y posición desde la profundidad (centroide de las esquinas en 3D).
   Junto a cada una va el error de suponer HOME3, como referencia.

Escribe ``<corrida>/start_anchor_validation.json`` y, con ``--write``,
``visuomotor_bringup/config/realsense/table_markers.yaml`` (esquinas y gravedad en la base).

Uso (contenedor):
    python3 tools/xarm5/vio_gt/calibrate_table_markers.py --run experiments/vio_gt/<fecha>/runs/<static> \
        --frames datasets/vio_frames/<static> [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("visuomotor_data_collection", "visuomotor_core"):
    sys.path.insert(0, str(REPO / p))
from visuomotor_core.analysis.trajectory_metrics import interpolate_poses, quat_to_matrix  # noqa: E402
from visuomotor_data_collection.start_anchor import (TableMarkers, camera_pose_from_markers,  # noqa: E402
                                                     camera_pose_from_markers_rgbd, correct_tilt_with_up, depth_at,
                                                     inv_T, pose_error, scale_intrinsics, tilt_from_up,
                                                     triangulate_point,
                                                     up_in_camera_from_accel)
from visuomotor_data_collection.vo_start_guide import detect_markers  # noqa: E402


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--hand-eye", default="", help="por defecto <run>/hand_eye.json")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--ids", default="12,14", help="ArUco de la mesa (los de home3_view_reference.yaml)")
    ap.add_argument("--depth-scale", type=float, default=0.986, help="corrección de escala de la profundidad")
    ap.add_argument("--frames-per-window", type=int, default=9)
    ap.add_argument("--v7-size", type=int, default=256, help="lado de la imagen de v7 para la variante pnp_v7 (0 = no)")
    ap.add_argument("--out", default=str(REPO / "visuomotor_bringup/config/realsense/table_markers.yaml"))
    ap.add_argument("--write", action="store_true", help="escribir el YAML de los marcadores")
    a = ap.parse_args(argv)
    run, fr = Path(a.run), REPO / a.frames
    keep_ids = {int(i) for i in a.ids.split(",")}
    cal = json.loads((fr / "calib.json").read_text())
    K, D = np.array(cal["K"]).reshape(3, 3), np.array(cal["D"])
    T_cam_imu = np.array(cal["T_opt_imu"])
    X = np.array(json.loads(Path(a.hand_eye or run / "hand_eye.json").read_text())["T_tcp_optical"])
    plan = json.loads((run / "plan.json").read_text())
    off = json.loads((run / "clock.json").read_text())["offset_pc_minus_jetson_s"]
    cache = json.loads((run / "eval_cache.json").read_text()) if (run / "eval_cache.json").exists() else {}
    d = cache.get("gyro_offset_max_1", [0.0])[0]                 # la cámara llega d s tarde respecto al robot
    rob = np.load(run / "robot.npz")
    t_r = rob["tcp"][:, 0] + off
    t0 = float(rob["t0"]) + off
    assoc = [line.split() for line in (fr / "associations.txt").read_text().splitlines() if line.strip()]
    tf = np.array([float(f[0]) for f in assoc])
    imu = np.loadtxt(fr / "imu.csv", delimiter=",", skiprows=1)
    wins = []
    for s, e, name in plan["segments"]:
        if not (name.startswith("static") or name.startswith("home3")):
            continue
        a0, a1 = t0 + s + 1.5, t0 + e - 0.3                      # tiempo del robot (reloj del PC)
        idx = np.nonzero((tf - d >= a0) & (tf - d <= a1))[0]
        if len(idx) < 3:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(a.frames_per_window, len(idx))).round().astype(int)]
        p, q, _ = interpolate_poses(tf[pick] - d, t_r, rob["tcp"][:, 1:4], rob["tcp"][:, 4:8])
        if np.ptp(p, axis=0).max() > 0.0005:
            print(f"  {name}: el robot se movió en la ventana, descartada")
            continue
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3], T_base_tcp[:3, 3] = quat_to_matrix(q[len(q) // 2]), p[len(p) // 2]
        obs, obs7, depths = {}, {}, []
        for k in pick:
            g = cv2.imread(str(fr / assoc[k][1]), cv2.IMREAD_GRAYSCALE)
            for i, c in detect_markers(g, a.dictionary).items():
                if i in keep_ids:
                    obs.setdefault(i, []).append(c)
            if a.v7_size:                                         # como el recolector de v7: 640x480 -> 256x256
                g7 = cv2.resize(g, (a.v7_size, a.v7_size), interpolation=cv2.INTER_AREA)
                for i, c in detect_markers(g7, a.dictionary).items():
                    if i in keep_ids:
                        obs7.setdefault(i, []).append(c)
            depths.append(cv2.imread(str(fr / assoc[k][3]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0)
        corners = {i: np.median(np.stack(v), axis=0) for i, v in obs.items() if len(v) >= max(2, len(pick) // 2)}
        # v7 guardaba UNA imagen por instante: para su variante se usa un solo frame (el del medio)
        corners7 = {i: v[len(v) // 2] for i, v in obs7.items()}
        dstack = np.stack(depths)
        dmed = np.where((dstack > 0).sum(0) > len(depths) // 2, np.median(np.where(dstack > 0, dstack, np.nan), 0), 0)
        im = (imu[:, 0] - d >= a0) & (imu[:, 0] - d <= a1)
        wins.append({"name": name, "T_base_cam": T_base_tcp @ X, "corners": corners, "corners7": corners7,
                     "depth": np.nan_to_num(dmed, nan=0.0),
                     "up_imu": up_in_camera_from_accel(imu[im, 4:7], T_cam_imu) if im.sum() > 20 else None})
    ids = sorted({i for w in wins for i in w["corners"]})
    print(f"{len(wins)} ventanas quietas; marcadores vistos: " +
          ", ".join(f"{i} en {sum(i in w['corners'] for w in wins)}" for i in ids))

    def triangulate(use):
        out, errs = {}, []
        for i in ids:
            vs = [w for w in use if i in w["corners"]]
            if len(vs) < 2:
                continue
            Ps = [K @ inv_T(w["T_base_cam"])[:3, :] for w in vs]
            pts = []
            for c in range(4):
                Xp, e = triangulate_point(Ps, [w["corners"][i][c] for w in vs])
                pts.append(Xp)
                errs.append(e)
            out[i] = np.array(pts)
        return out, errs

    def gravity_base(use):
        ups = [w["T_base_cam"][:3, :3] @ w["up_imu"] for w in use if w["up_imu"] is not None]
        g = np.mean(ups, axis=0)
        return g / np.linalg.norm(g), ups

    full, errs = triangulate(wins)
    g_full, ups = gravity_base(wins)
    spread = [np.degrees(np.arccos(np.clip(np.dot(u, g_full), -1, 1))) for u in ups]
    table = TableMarkers(a.dictionary, full)
    print("lado de los marcadores (mm): " + ", ".join(f"{i}: {v * 1000:.1f}" for i, v in table.side_lengths_m().items())
          + f" | reproyección de la triangulación {np.sqrt(np.mean(np.square(errs))):.2f} px")
    print(f"vertical real en la base: {np.round(g_full, 4)} ({np.degrees(np.arccos(g_full[2])):.2f}° de z); "
          f"dispersión del IMU entre ventanas: mediana {np.median(spread):.2f}°, máx {np.max(spread):.2f}°")
    home = next(w for w in wins if w["name"] == "home3")
    rows = []
    for k, w in enumerate(wins):
        others = [v for j, v in enumerate(wins) if j != k]
        tri, _ = triangulate(others)
        tab = TableMarkers(a.dictionary, tri)
        g_loo, _ = gravity_base(others)
        est = camera_pose_from_markers(w["corners"], tab, K, D)
        if est is None:
            continue
        r = {"window": w["name"], "markers": est["markers"], "reproj_px": est["reprojection_px"]}
        r["assume_home3"] = pose_error(home["T_base_cam"], w["T_base_cam"])
        r["pnp"] = pose_error(est["T_base_cam"], w["T_base_cam"])
        if a.v7_size and w["corners7"]:                          # resolución de v7, sin IMU (v7 no lo guardó)
            K7 = scale_intrinsics(K, (cal["width"], cal["height"]), (a.v7_size, a.v7_size))
            e7 = camera_pose_from_markers(w["corners7"], tab, K7, D)
            if e7 is not None:
                r["pnp_v7"] = pose_error(e7["T_base_cam"], w["T_base_cam"])
        T_pi = est["T_base_cam"]
        if w["up_imu"] is not None:
            r["imu_tilt_vs_robot_deg"] = tilt_from_up(w["T_base_cam"], w["up_imu"], g_loo)
            T_pi = correct_tilt_with_up(est["T_base_cam"], w["up_imu"], g_loo)
            r["pnp_imu"] = pose_error(T_pi, w["T_base_cam"])
        rg = camera_pose_from_markers_rgbd(w["corners"], w["depth"], tab, K, a.depth_scale)
        if rg is not None:
            r["rgbd"] = pose_error(rg["T_base_cam"], w["T_base_cam"])
            # Híbrido: rotación de pnp_imu; posición desde el centroide 3D de las esquinas (profundidad)
            Kinv = np.linalg.inv(K)
            src, dst = [], []
            for i in sorted(set(w["corners"]) & set(tri)):
                for c in range(4):
                    z = depth_at(w["depth"], w["corners"][i][c]) * a.depth_scale
                    if np.isfinite(z) and z > 0:
                        src.append(z * (Kinv @ np.r_[w["corners"][i][c], 1.0]))
                        dst.append(tri[i][c])
            src, dst = np.asarray(src), np.asarray(dst)
            T_h = T_pi.copy()
            T_h[:3, 3] = dst.mean(0) - T_pi[:3, :3] @ src.mean(0)
            r["hybrid"] = pose_error(T_h, w["T_base_cam"])
        rows.append(r)
    variants = ["assume_home3", "pnp", "pnp_imu", "pnp_v7", "rgbd", "hybrid"]
    print(f"\n{'ventana':11s} {'marc.':8s}" + "".join(f"{v:>16s}" for v in variants) + "   IMU°")
    for r in rows:
        print(f"{r['window']:11s} {str(r['markers']):8s}" + "".join(
            f"{r[v][0]:9.1f}/{r[v][1]:5.2f}" if v in r else f"{'-':>16s}" for v in variants)
            + f"  {r.get('imu_tilt_vs_robot_deg', float('nan')):5.2f}")
    summary = {"windows": len(rows), "side_length_mm": {i: v * 1000 for i, v in table.side_lengths_m().items()},
               "gravity_base": g_full.tolist(), "imu_spread_deg_median": float(np.median(spread))}
    for v in variants:
        vals = np.array([r[v] for r in rows if v in r and r["window"] not in ("home3", "home3_end")
                         or (v in r and v != "assume_home3")])
        if len(vals):
            summary[v] = {"mm_median": float(np.median(vals[:, 0])), "mm_max": float(vals[:, 0].max()),
                          "deg_median": float(np.median(vals[:, 1])), "deg_max": float(vals[:, 1].max())}
    two = [r for r in rows if len(r["markers"]) == 2]
    for v in variants[1:]:
        vals = np.array([r[v] for r in two if v in r])
        if len(vals):
            summary[f"{v}_two_markers"] = {"n": len(vals), "mm_median": float(np.median(vals[:, 0])),
                                           "deg_median": float(np.median(vals[:, 1]))}
    print("\nresumen (dejando una ventana fuera; mm/°):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    (run / "start_anchor_validation.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                                   default=float))
    if a.write:
        import yaml
        table.meta = {"gravity_base": np.round(g_full, 6).tolist(), "depth_scale_correction": a.depth_scale,
                      "source_run": str(run.relative_to(REPO)) if run.is_absolute() else str(run),
                      "windows": len(wins), "triangulation_rms_px": float(np.sqrt(np.mean(np.square(errs)))),
                      "validation": "start_anchor_validation.json de la corrida"}
        txt = ("# ArUco de la mesa: esquinas en link_base (m) triangulando con el robot, y vertical real en la base\n"
               "# (tools/xarm5/vio_gt/calibrate_table_markers.py). Vale mientras no se muevan los marcadores sobre el\n"
               "# mantel ni el robot respecto de la mesa (la guía de inicio lo comprueba en HOME3).\n")
        Path(a.out).write_text(txt + yaml.safe_dump(table.to_dict(), sort_keys=False))
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
