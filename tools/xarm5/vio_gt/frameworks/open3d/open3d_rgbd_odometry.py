#!/usr/bin/env python3
"""Odometría RGB-D densa de Open3D (Steinbrücker/Park, término híbrido) contra un keyframe, sobre un export TUM.

Cada frame se alinea con el keyframe vigente (no con el anterior: menos deriva). Se crea un keyframe nuevo cuando la
cámara se aleja > ``--kf-trans-m`` o gira > ``--kf-rot-deg``, o si la alineación falla. Si falla, el frame cuenta
como perdido: no se escribe pose, y el siguiente se intenta contra el mismo keyframe. Tras ``--max-fail``
fallos seguidos, el frame actual pasa a ser keyframe con la última pose conocida (como RTAB-Map al recuperar).
Salida: TUM con la pose de la cámara de color (marco óptico) en el mundo = primer frame.

Uso (imagen visuomotor_open3d:0.18): python3 open3d_rgbd_odometry.py <carpeta_export> <salida.tum>
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def rot_deg(T):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(T[:3, :3]) - 1) / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--kf-trans-m", type=float, default=0.05)
    ap.add_argument("--kf-rot-deg", type=float, default=5.0)
    ap.add_argument("--max-fail", type=int, default=5)
    ap.add_argument("--depth-trunc", type=float, default=3.0)
    a = ap.parse_args()
    d = Path(a.data)
    cal = json.loads((d / "calib.json").read_text())
    K = np.array(cal["K"]).reshape(3, 3)
    intr = o3d.camera.PinholeCameraIntrinsic(cal["width"], cal["height"], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    pairs = [line.split() for line in (d / "associations.txt").read_text().splitlines() if line.strip()]
    opt = o3d.pipelines.odometry.OdometryOption()
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    def load(p):
        c = o3d.io.read_image(str(d / p[1]))
        z = o3d.io.read_image(str(d / p[3]))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            c, z, depth_scale=cal["depth_scale"], depth_trunc=a.depth_trunc, convert_rgb_to_intensity=True)

    kf, T_kf = load(pairs[0]), np.eye(4)        # keyframe y su pose en el mundo
    T_prev_rel = np.eye(4)                      # pose del frame anterior respecto al keyframe (semilla)
    out, fails, lost, t0 = [], 0, 0, time.time()
    out.append((float(pairs[0][0]), T_kf.copy()))
    for i, p in enumerate(pairs[1:], 1):
        cur = load(p)
        # compute_rgbd_odometry devuelve T tal que p_kf = T · p_cur (pose del actual en el marco del keyframe)
        ok, T_rel, _ = o3d.pipelines.odometry.compute_rgbd_odometry(cur, kf, intr, T_prev_rel, jac, opt)
        if ok:
            fails = 0
            T_w = T_kf @ T_rel
            out.append((float(p[0]), T_w))
            T_prev_rel = T_rel
            if np.linalg.norm(T_rel[:3, 3]) > a.kf_trans_m or rot_deg(T_rel) > a.kf_rot_deg:
                kf, T_kf, T_prev_rel = cur, T_w, np.eye(4)
        else:
            lost += 1
            fails += 1
            if fails >= a.max_fail:             # se rehace el keyframe con la última pose conocida
                kf, T_kf, T_prev_rel, fails = cur, out[-1][1], np.eye(4), 0
        if i % 500 == 0:
            print(f"frame {i}/{len(pairs)}  perdidos {lost}  {time.time() - t0:.0f} s", flush=True)
    with open(a.out, "w") as f:
        for t, T in out:
            R = T[:3, :3]
            w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            qx = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
            qy = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
            qz = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
            f.write(f"{t:.6f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {w:.8f}\n")
    print(f"{len(out)} poses, {lost} perdidos de {len(pairs)} ({100 * lost / len(pairs):.1f} %), "
          f"{time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
