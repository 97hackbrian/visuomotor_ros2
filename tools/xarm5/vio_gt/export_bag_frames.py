#!/usr/bin/env python3
"""Exporta un bag de medición a una carpeta tipo TUM RGB-D, para probar otros frameworks (ORB-SLAM3, DROID-SLAM, DPVO,
Open3D...) fuera de ROS.

Escribe en ``<out>/``:
* ``rgb/<t>.jpg``: los bytes JPEG del bag tal cual (sin recomprimir) y ``rgb.txt`` (``t ruta``);
* ``depth/<t>.png``: profundidad de 16 bits en mm (el PNG de compressedDepth sin su cabecera de 12 bytes) y
  ``depth.txt``;
* ``associations.txt``: ``t_rgb rgb t_depth depth``, pares con |dt| <= 5 ms (aligned_depth_to_color es del mismo
  instante que el color);
* ``imu.csv``: ``t, wx, wy, wz, ax, ay, az`` de ``/camera/camera/imu`` (marco camera_imu_optical_frame);
* ``calib.json``: K, D, tamaño, escala de profundidad y extrínsecas de fábrica (tf_static): ``T_link_opt`` (color
  óptico en camera_link), ``T_link_imu`` (IMU en camera_link) y ``T_opt_imu`` (IMU en el marco óptico de color).
Los tiempos son los sellos del bag (reloj del PC), los mismos de la odometría: ``evaluate_vio_gt.py --traj-tum`` los
compara contra el robot igual que a RTAB-Map. No modifica el bag. Se niega a escribir sobre una carpeta existente.

Uso (contenedor con ROS):
    python3 tools/xarm5/vio_gt/export_bag_frames.py --bag datasets/vio_bags/<bag> --out datasets/vio_frames/<nombre>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/hardware"))
from vio_bag_utils import StaticTF, inv_T, iter_messages, stamp_of  # noqa: E402

COLOR = "/camera/camera/color/image_raw/compressed"
DEPTH = "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
INFO = "/camera/camera/color/camera_info"
IMU = "/camera/camera/imu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dt-s", type=float, default=0.005)
    a = ap.parse_args(argv)
    bag, out = REPO / a.bag, REPO / a.out
    if out.exists():
        raise SystemExit(f"{out} ya existe: no se sobrescribe")
    (out / "rgb").mkdir(parents=True)
    (out / "depth").mkdir()
    stf = StaticTF()
    rgb, dep, imu = [], [], []
    calib = None
    for topic, m, _ in iter_messages(bag, [COLOR, DEPTH, INFO, IMU, "/tf_static"]):
        if topic == "/tf_static":
            stf.add_message(m)
            continue
        if topic == INFO:
            if calib is None:
                calib = {"K": list(map(float, m.k)), "D": list(map(float, m.d)), "distortion_model": m.distortion_model,
                         "width": int(m.width), "height": int(m.height), "depth_scale": 1000.0,
                         "color_frame": m.header.frame_id}
            continue
        t = stamp_of(m)
        if topic == IMU:
            w, acc = m.angular_velocity, m.linear_acceleration
            imu.append((t, w.x, w.y, w.z, acc.x, acc.y, acc.z))
            continue
        name = f"{t:.6f}"
        if topic == COLOR:
            if "jpeg" not in m.format.lower() and "jpg" not in m.format.lower():
                raise SystemExit(f"formato de color no JPEG: {m.format}")
            (out / "rgb" / f"{name}.jpg").write_bytes(bytes(m.data))
            rgb.append((t, f"rgb/{name}.jpg"))
        else:
            (out / "depth" / f"{name}.png").write_bytes(bytes(m.data)[12:])
            dep.append((t, f"depth/{name}.png"))
    rgb.sort()
    dep.sort()
    imu.sort()
    (out / "rgb.txt").write_text("".join(f"{t:.6f} {p}\n" for t, p in rgb))
    (out / "depth.txt").write_text("".join(f"{t:.6f} {p}\n" for t, p in dep))
    td = np.array([d[0] for d in dep])
    assoc = []
    for t, p in rgb:
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) <= a.max_dt_s:
            assoc.append(f"{t:.6f} {p} {td[j]:.6f} {dep[j][1]}\n")
    (out / "associations.txt").write_text("".join(assoc))
    (out / "imu.csv").write_text("t,wx,wy,wz,ax,ay,az\n" + "".join(",".join(f"{v:.9f}" for v in r) + "\n" for r in imu))
    T_link_opt = stf.lookup("camera_link", "camera_color_optical_frame")
    T_link_imu = stf.lookup("camera_link", "camera_imu_optical_frame")
    calib.update(T_link_opt=T_link_opt.tolist(), T_link_imu=T_link_imu.tolist(),
                 T_opt_imu=(inv_T(T_link_opt) @ T_link_imu).tolist(), source_bag=a.bag,
                 n_rgb=len(rgb), n_depth=len(dep), n_assoc=len(assoc), n_imu=len(imu),
                 rgb_rate_hz=float((len(rgb) - 1) / (rgb[-1][0] - rgb[0][0])),
                 imu_rate_hz=float((len(imu) - 1) / (imu[-1][0] - imu[0][0])))
    (out / "calib.json").write_text(json.dumps(calib, indent=2))
    print(f"{out.name}: {len(rgb)} rgb, {len(dep)} depth, {len(assoc)} pares, {len(imu)} imu "
          f"({calib['imu_rate_hz']:.0f} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
