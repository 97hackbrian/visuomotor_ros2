#!/usr/bin/env python3
"""Pose REAL de la cámara al inicio de cada episodio (ArUco de la mesa + IMU), para ``convert_vio_dataset_to_xarm5.py
--start-poses``. Reemplaza el supuesto de que toda demo empieza exactamente en HOME3.

Fuentes:
* ``--zarr <dataset crudo v8>``: cada episodio trae su bag crudo (atributo ``raw_bag``, recolector v3); se exporta a
  ``datasets/vio_frames/<dataset>/<episodio>`` con ``export_bag_frames.py`` si no existe.
* ``--frames <carpeta export>`` (se puede repetir): carpetas ya exportadas, p. ej. las corridas con el robot. Con
  ``--robot-run <corrida>`` en el mismo orden, el resultado se compara con la pose del robot (validación).

Método (``visuomotor_data_collection/start_anchor.py``):
1. Ventana quieta del inicio: desde el primer frame hasta que el giróscopo pasa de ``--gyro-static-rad-s`` (máx.
   ``--max-static-s``).
2. PnP con las esquinas de los ArUco (``table_markers.yaml``, calibradas con el robot) en la mediana de esos frames.
3. Inclinación corregida con el acelerómetro y la vertical real de la base. Si el PnP y el IMU discrepan > 3°, el
   episodio se marca y el conversor usa HOME (``usable: false``).
Precisión medida contra el robot (2026-11-25, dejando una ventana fuera): con los dos marcadores, ~11 mm y 0,6°.

Uso (contenedor con ROS si hay que exportar bags):
    python3 tools/xarm5/estimate_start_poses.py --zarr datasets/demonstrations_real_v8.zarr --out <json>
    python3 tools/xarm5/estimate_start_poses.py --frames datasets/vio_frames/<c> --robot-run experiments/vio_gt/<f>/runs/<c> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for p in ("visuomotor_data_collection", "visuomotor_bringup", "tools/xarm5", "tools/xarm5/vio_gt"):
    sys.path.insert(0, str(REPO / p))
from visuomotor_data_collection.start_anchor import (TableMarkers, estimate_start_pose, inv_T,  # noqa: E402
                                                     pose_error)


def static_window(frames_t: np.ndarray, imu: np.ndarray, max_s: float, gyro_thr: float):
    """Instante en que termina la quietud inicial (giróscopo suavizado a 0,1 s por encima del umbral)."""
    t0 = frames_t[0]
    m = (imu[:, 0] >= t0) & (imu[:, 0] <= t0 + max_s)
    if m.sum() < 10:
        return t0 + min(max_s, 0.2)
    w = np.linalg.norm(imu[m, 1:4], axis=1)
    k = max(1, int(round(0.1 * len(w) / max(imu[m, 0][-1] - imu[m, 0][0], 1e-3))))
    ws = np.convolve(w, np.ones(k) / k, "same")
    moving = np.nonzero(ws > gyro_thr)[0]
    return float(imu[m, 0][moving[0]]) if len(moving) else t0 + max_s


def estimate_dir(fr: Path, table: TableMarkers, T_tcp_link: np.ndarray, T_base_home_link: np.ndarray, a,
                 episode_t0: float = None) -> dict:
    """``episode_t0``: sello del primer frame del episodio. Con el recolector v3.1 el bag empieza en ARMING (quieto),
    así que la ventana quieta es [inicio del bag, t0): se usa esa. Si no hay pre-roll, se detecta con el giróscopo."""
    import cv2
    cal = json.loads((fr / "calib.json").read_text())
    K, D = np.array(cal["K"]).reshape(3, 3), np.array(cal["D"])
    T_link_opt = np.array(cal["T_link_opt"])
    assoc = [line.split() for line in (fr / "associations.txt").read_text().splitlines() if line.strip()]
    tf = np.array([float(x[0]) for x in assoc])
    imu = np.loadtxt(fr / "imu.csv", delimiter=",", skiprows=1, ndmin=2)
    if episode_t0 is not None and episode_t0 - tf[0] >= a.min_static_s:
        t_end, source = episode_t0 - 0.03, "arming"                # pre-roll: la cámara quieta mientras se reinicia
    else:
        t_end, source = static_window(tf, imu, a.max_static_s, a.gyro_static_rad_s), "gyro"
    still = np.nonzero(tf <= t_end)[0]
    out = {"static_s": float(t_end - tf[0]), "static_source": source, "frames_used": 0, "usable": False}
    if (t_end - tf[0]) < a.min_static_s or len(still) < 3:
        out["flags"] = [f"sin quietud al inicio ({t_end - tf[0]:.2f} s)"]
        still = np.arange(min(3, len(tf)))
        accel = None
    else:
        im = (imu[:, 0] >= tf[0]) & (imu[:, 0] <= t_end)
        accel = imu[im, 4:7]
    pick = still[np.linspace(0, len(still) - 1, min(10, len(still))).round().astype(int)]
    grays = [cv2.imread(str(fr / assoc[k][1]), cv2.IMREAD_GRAYSCALE) for k in pick]
    g_base = table.meta.get("gravity_base", [0.0, 0.0, 1.0])
    res = estimate_start_pose(grays, table, K, D, accel=accel, T_cam_imu=np.array(cal["T_opt_imu"]), g_base=g_base,
                              max_tilt_disagreement_deg=a.max_tilt_disagreement_deg)
    out.update(frames_used=len(pick), markers=res["markers"], imu_used=res["imu_used"],
               flags=out.get("flags", []) + res["flags"], reprojection_px=res.get("reprojection_px"),
               tilt_pnp_vs_imu_deg=res.get("tilt_pnp_vs_imu_deg"))
    if res["T_base_cam"] is None:
        return out
    T_base_link = res["T_base_cam"] @ inv_T(T_link_opt)
    off_mm, off_deg = pose_error(T_base_link, T_base_home_link)
    critical = [f for f in out["flags"] if "discrepan" in f or "sin marcadores" in f]
    out.update(T_base_camera_link=T_base_link.tolist(), T_base_tcp=(T_base_link @ inv_T(T_tcp_link)).tolist(),
               offset_from_home_mm=off_mm, offset_from_home_deg=off_deg, usable=not critical)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default="")
    ap.add_argument("--frames", action="append", default=[])
    ap.add_argument("--robot-run", action="append", default=[], help="corrida del robot (validación), mismo orden")
    ap.add_argument("--table", default=str(REPO / "visuomotor_bringup/config/realsense/table_markers.yaml"))
    ap.add_argument("--alignment", default=str(REPO / "visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml"))
    ap.add_argument("--policy-params", default=str(REPO / "deploy/jetson/config/jetson_policy_params.yaml"))
    ap.add_argument("--max-static-s", type=float, default=1.0)
    ap.add_argument("--min-static-s", type=float, default=0.2)
    ap.add_argument("--gyro-static-rad-s", type=float, default=0.05)
    ap.add_argument("--max-tilt-disagreement-deg", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from convert_vio_dataset_to_xarm5 import pose_matrix, rpy_to_quat
    table = TableMarkers.load(a.table)
    al = yaml.safe_load(open(a.alignment))["xarm5_dataset_alignment"]
    T_tcp_link = pose_matrix(al["tcp_to_camera_xyz_m"], rpy_to_quat(*al["tcp_to_camera_rpy_rad"]))
    pol = yaml.safe_load(open(a.policy_params))["policy_node"]["ros__parameters"]
    T_base_home = pose_matrix([pol["home_x"], pol["home_y"], pol["home_z"]],
                              rpy_to_quat(pol["home_roll"], pol["home_pitch"], pol["home_yaw"]))
    T_base_home_link = T_base_home @ T_tcp_link
    jobs = []
    if a.zarr:
        import zarr
        from export_bag_frames import main as export
        z = zarr.open_group(str(REPO / a.zarr), mode="r")
        root = REPO / "datasets/vio_frames" / Path(a.zarr).stem
        for key in sorted(z.group_keys(), key=lambda k: int(k.split("_")[-1])):
            bag = z[key].attrs.get("raw_bag")
            if not bag:
                print(f"  {key}: sin bag crudo (raw_bag_enabled: false al grabar)")
                continue
            fr = root / key
            if not fr.exists():
                export(["--bag", str(bag), "--out", str(fr.relative_to(REPO))])
            ts = z[key]["timestamp"][:1] if "timestamp" in z[key] else []
            t0 = float(ts[0]) if len(ts) and z[key].attrs.get("raw_bag_starts_at") == "arming" else None
            jobs.append((key, fr, None, t0))
    for i, f in enumerate(a.frames):
        jobs.append((Path(f).name, REPO / f, a.robot_run[i] if i < len(a.robot_run) else None, None))
    episodes, errs = {}, []
    for key, fr, run, t0 in jobs:
        r = estimate_dir(fr, table, T_tcp_link, T_base_home_link, a, episode_t0=t0)
        if run and r.get("T_base_camera_link") is not None:
            # Validación: la pose del robot al inicio de la corrida (primer registro del TCP) con la extrínseca medida
            rob = np.load(REPO / run / "robot.npz")
            from visuomotor_core.analysis.trajectory_metrics import quat_to_matrix
            T_bt = np.eye(4)
            T_bt[:3, :3], T_bt[:3, 3] = quat_to_matrix(rob["tcp"][0, 4:8]), rob["tcp"][0, 1:4]
            r["error_vs_robot_mm"], r["error_vs_robot_deg"] = pose_error(np.asarray(r["T_base_tcp"]), T_bt)
            errs.append((r["error_vs_robot_mm"], r["error_vs_robot_deg"]))
        episodes[key] = r
        print(f"{key:28s} marcadores {str(r.get('markers')):9s} IMU {'sí' if r.get('imu_used') else 'no'} | "
              f"desvío de HOME3 {r.get('offset_from_home_mm', float('nan')):6.1f} mm "
              f"{r.get('offset_from_home_deg', float('nan')):5.2f}°"
              + (f" | error vs robot {r['error_vs_robot_mm']:5.1f} mm {r['error_vs_robot_deg']:4.2f}°"
                 if 'error_vs_robot_mm' in r else "") + (f" | {r['flags']}" if r.get("flags") else ""))
    summary = {"episodes": len(episodes), "usable": sum(bool(v.get("usable")) for v in episodes.values())}
    if errs:
        e = np.array(errs)
        summary.update(error_vs_robot_mm_median=float(np.median(e[:, 0])), error_vs_robot_mm_max=float(e[:, 0].max()),
                       error_vs_robot_deg_median=float(np.median(e[:, 1])), error_vs_robot_deg_max=float(e[:, 1].max()))
    Path(a.out).write_text(json.dumps({"table": a.table, "summary": summary, "episodes": episodes}, indent=1))
    print("resumen:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
