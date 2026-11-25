#!/usr/bin/env python3
"""Salidas de vio_runner (ORB-SLAM3) -> TUM en el marco óptico de color, para evaluate_vio_gt.py --traj-tum.

* ``<prefijo>_live.txt`` -> ``<prefijo>_live.tum``: lo que el sistema entrega en cada frame (uso en vivo).
  - Válidos: estado OK (2), OK_KLT (5) y "recién perdido" (3, pose predicha, con IMU).
  - Desde el primer LOST (4), o si vuelve a "sin inicializar" (1) tras haber inicializado, ORB-SLAM3 sigue en un mapa
    NUEVO con otro origen: esos frames y todos los siguientes cuentan como perdidos (criterio conservador).
* ``_final_tum.txt`` (RGB-D, cámara) o ``_final_euroc.txt`` (inerciales, pose del IMU en ns) -> ``_final.tum``:
  trayectoria tras el ajuste global (uso offline). La del IMU se pasa a la cámara con ``T_opt_imu`` de calib.json.

Uso: python3 orbslam3_to_tum.py <prefijo> <calib.json>
"""
import json
import sys
from pathlib import Path

import numpy as np


def q2m(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def write_tum(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:.6f}" if i == 0 else f"{v:.8f}" for i, v in enumerate(r)) + "\n")


def main():
    prefix, cal = sys.argv[1], json.load(open(sys.argv[2]))
    info = {}
    live = np.loadtxt(prefix + "_live.txt", ndmin=2)
    st = live[:, 1].astype(int)
    initialized = np.cumsum(np.isin(st, (2, 3, 5))) > 0
    broken = (st == 4) | ((st == 1) & initialized)
    cut = int(np.argmax(broken)) if broken.any() else len(st)
    valid = np.isin(st, (2, 3, 5)) & (np.arange(len(st)) < cut)
    write_tum(prefix + "_live.tum", np.c_[live[valid, 0], live[valid, 2:9]])
    info["live"] = {"frames": int(len(st)), "valid": int(valid.sum()),
                    "first_break_s": float(live[cut, 0] - live[0, 0]) if cut < len(st) else None,
                    "recently_lost_frames": int((st == 3).sum()), "lost_frames": int((st == 4).sum())}
    tum, euroc = Path(prefix + "_final_tum.txt"), Path(prefix + "_final_euroc.txt")
    if tum.exists():
        rows = np.loadtxt(tum, ndmin=2)
        write_tum(prefix + "_final.tum", rows)
        info["final"] = {"poses": int(len(rows)), "frame": "camera"}
    elif euroc.exists():
        rows = np.loadtxt(euroc, ndmin=2)
        T_imu_opt = np.linalg.inv(np.array(cal["T_opt_imu"]))
        out = []
        for r in rows:
            Twb = np.eye(4)
            Twb[:3, :3], Twb[:3, 3] = q2m(r[4:8]), r[1:4]
            Twc = Twb @ T_imu_opt
            out.append(np.r_[r[0] * 1e-9, Twc[:3, 3], m2q(Twc[:3, :3])])
        write_tum(prefix + "_final.tum", out)
        info["final"] = {"poses": int(len(rows)), "frame": "imu->camera"}
    Path(prefix + "_info.json").write_text(json.dumps(info, indent=1))
    print(prefix, json.dumps(info))


if __name__ == "__main__":
    main()
