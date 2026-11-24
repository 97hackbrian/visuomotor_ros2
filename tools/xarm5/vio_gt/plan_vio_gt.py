#!/usr/bin/env python3
"""Planes de movimiento del xArm5 para medir la odometría de la D435i con el robot como referencia (Fase 2.2).

Genera trayectorias densas a 20 Hz (``poses``: x y z qx qy qz qw en ``link_base``) que
``jetson/vio_gt_motion.py`` envía al puente por ``/target_frame_raw``. Todas empiezan y terminan en HOME3.

Tipos de plan:
* ``static``: paradas mirando el tablero ChArUco (calibración cámara-brida y error estático de la odometría).
  Varían el radio, la altura, el acimut, la inclinación y el giro de la herramienta.
* ``approach``: aproximaciones reales de v7 (``demonstrations_real_v7_xarm5_v2.zarr``, ya proyectadas a lo que el
  brazo puede ejecutar), escaladas en el tiempo a ``--v-max``, con quietud al inicio y al final, y regreso a HOME3.

Cada pose se valida con la MISMA caja y el mismo alcance que el puente (``xarm5_real_params.yaml``); la orientación
se construye con ``arm_plane_quat`` (alcanzable por construcción). Las inclinaciones se limitan al rango que ya
ejecutó el robot (HOME3 y las demos) con un margen. No mueve nada: solo escribe JSON.

Uso (contenedor, sin ROS):
    python3 tools/xarm5/vio_gt/plan_vio_gt.py --board docs/hardware/.board_plan.json --out-dir experiments/vio_gt/plans
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))
from visuomotor_bringup.xarm5_interface import (  # noqa: E402
    ReachLimit, WorkspaceLimits, arm_azimuth, arm_plane_params, arm_plane_quat, clamp_to_workspace)

RATE = 20.0


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def qangle(q0, q1):
    return 2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def segment(p0, q0, p1, q1, v_max, w_max):
    """Movimiento de reposo a reposo con perfil cosenoidal (velocidad pico = pi/2 · media)."""
    T = max(np.linalg.norm(np.subtract(p1, p0)) / v_max, qangle(q0, q1) / w_max) * math.pi / 2 + 0.5
    n = max(2, int(round(T * RATE)))
    out = []
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)
        out.append(np.r_[np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0)), slerp(q0, q1, s)])
    return out


def hold(pose, seconds):
    return [np.asarray(pose, float).copy() for _ in range(int(round(seconds * RATE)))]


def load_limits(path):
    g = yaml.safe_load(open(path))["xarm5_bridge"]["ros__parameters"]
    box = WorkspaceLimits(g["workspace_x_min"], g["workspace_x_max"], g["workspace_y_min"], g["workspace_y_max"],
                          g["workspace_z_min"], g["workspace_z_max"], g["workspace_min_radius_xy"],
                          g["workspace_max_radius_xy"])
    reach = ReachLimit(g["workspace_shoulder_xyz_m"], g["workspace_wrist_offset_m"], g["workspace_max_reach_m"],
                       g["workspace_min_reach_m"])
    return box, reach, g


def check(poses, box, reach, pitch_lim, z_min):
    """Cada pose dentro de la caja y del alcance del puente (sin recorte), inclinación en rango y sobre el piso."""
    worst = 0.0
    for p in poses:
        az = arm_azimuth(p[:3])
        pitch, _ = arm_plane_params(p[3:7], az)
        out, _, valid = clamp_to_workspace(p[:3], pitch, box, reach)
        # HOME3 medido queda en el techo de la caja (z = 0.75001): se toleran recortes < 2 mm
        if not valid or np.linalg.norm(out - p[:3]) > 0.002:
            raise ValueError(f"pose fuera de la caja o del alcance del puente: {np.round(p[:3], 3)}")
        if not pitch_lim[0] - 1e-6 <= pitch <= pitch_lim[1] + 1e-6:
            raise ValueError(f"inclinación {math.degrees(pitch):.1f}° fuera de {np.degrees(pitch_lim).round(1)}")
        if p[2] < z_min:
            raise ValueError(f"z={p[2]:.3f} bajo el mínimo {z_min}")
        worst = max(worst, math.degrees(pitch))
    return True


def finish(name, kind, poses, segments, meta):
    P = np.array(poses)
    v = np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1) * RATE
    w = np.array([qangle(P[i, 3:], P[i + 1, 3:]) for i in range(len(P) - 1)]) * RATE
    return {"name": name, "kind": kind, "rate_hz": RATE, "poses": np.round(P, 6).tolist(), "segments": segments,
            "duration_s": round(len(P) / RATE, 1), "v_max_m_s": round(float(v.max()), 4),
            "w_max_rad_s": round(float(w.max()), 4), "z_min_m": round(float(P[:, 2].min()), 3), **meta}


def static_plan(home_p, home_q, board, box, reach, a):
    b = np.asarray(board["center"])
    phi_b = math.atan2(b[1], b[0])
    _, yaw_home = arm_plane_params(home_q, arm_azimuth(home_p))
    # (radio, z, delta acimut, desfase de puntería vertical, delta giro): variedad para la calibración cámara-brida
    spec = [(0.45, 0.62, 0.00, 0.00, 0.0), (0.45, 0.62, 0.00, 0.00, 0.6), (0.45, 0.62, 0.00, 0.00, -0.6),
            (0.55, 0.50, 0.10, 0.08, 0.3), (0.55, 0.50, -0.10, 0.08, -0.3), (0.60, 0.40, 0.00, -0.06, 0.0),
            (0.60, 0.40, 0.08, 0.00, 0.5), (0.60, 0.40, -0.08, 0.00, -0.5), (0.52, 0.36, 0.00, 0.10, 0.0),
            (0.40, 0.48, 0.12, 0.00, 0.0), (0.40, 0.48, -0.12, 0.00, 0.4), (0.35, 0.66, 0.00, 0.10, -0.4),
            (0.50, 0.58, 0.06, -0.08, 0.7), (0.58, 0.46, -0.05, 0.05, -0.7)]
    poses = hold(np.r_[home_p, home_q], a.hold_s)
    segments = [[0.0, len(poses) / RATE, "home3"]]
    last = poses[-1]
    for i, (r, z, dphi, daim, dyaw) in enumerate(spec):
        phi = phi_b + dphi
        p = np.array([r * math.cos(phi), r * math.sin(phi), z])
        d_r = float(np.dot(b[:2], [math.cos(phi), math.sin(phi)])) - r
        d_z = b[2] - z
        pitch = math.atan2(-d_r, -d_z) + daim
        pitch = min(max(pitch, a.pitch_min), a.pitch_max)
        q = arm_plane_quat(phi, pitch, yaw_home + dyaw)
        poses += segment(last[:3], last[3:], p, q, a.v_max, a.w_max)
        t0 = len(poses) / RATE
        poses += hold(np.r_[p, q], a.hold_s)
        segments.append([t0, len(poses) / RATE, f"static_{i:02d}"])
        last = poses[-1]
    poses += segment(last[:3], last[3:], home_p, home_q, a.v_max, a.w_max)
    t0 = len(poses) / RATE
    poses += hold(np.r_[home_p, home_q], a.hold_s)
    segments.append([t0, len(poses) / RATE, "home3_end"])
    check(poses, box, reach, (a.pitch_min, a.pitch_max), a.z_floor)
    return finish("static_board", "static", poses, segments, {"board_center": b.tolist()})


def approach_plans(home_p, home_q, box, reach, a):
    import zarr
    z = zarr.open(str(REPO / a.dataset), "r")
    keys = sorted(z.keys(), key=lambda k: int(k.split("_")[-1]))
    ends = []
    for k in keys:
        s = z[k]["state"][:]
        ends.append((k, s[-1, :3] + np.asarray(z[k].attrs["home_xyz_m"])))
    # Episodios repartidos por el destino final (lateral y profundidad): orden por y y se toman equiespaciados
    ends.sort(key=lambda e: (e[1][1], e[1][0]))
    plans, used = [], set()
    for target in np.linspace(0, len(ends) - 1, a.n_approach).round().astype(int):
        # El más cercano a la posición objetivo (en el orden lateral) que el puente ejecute sin recortes
        for j in sorted(range(len(ends)), key=lambda j: abs(j - target)):
            k = ends[j][0]
            if k in used:
                continue
            try:
                plans.append(_approach_plan(z, k, home_p, home_q, box, reach, a))
                used.add(k)
                break
            except ValueError as e:
                print(f"  {k} descartado: {e}")
    return plans


def _approach_plan(z, k, home_p, home_q, box, reach, a):
    if True:
        s = z[k]["state"][:]
        P = s[:, :3] + np.asarray(z[k].attrs["home_xyz_m"])
        Q = s[:, 3:7] / np.linalg.norm(s[:, 3:7], axis=1, keepdims=True)
        # Proyección exacta a lo alcanzable (como el puente) e inclinación dentro del rango permitido
        Qr = []
        for p, q in zip(P, Q):
            az = arm_azimuth(p)
            pitch, yaw = arm_plane_params(q, az)
            Qr.append(arm_plane_quat(az, min(max(pitch, a.pitch_min), a.pitch_max), yaw))
        Qr = np.array(Qr)
        for i in range(1, len(Qr)):
            if np.dot(Qr[i], Qr[i - 1]) < 0:
                Qr[i] = -Qr[i]
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) * RATE
        w = np.array([qangle(Qr[i], Qr[i + 1]) for i in range(len(Qr) - 1)]) * RATE
        scale = max(np.percentile(v, 99.5) / a.v_max, np.percentile(w, 99.5) / a.w_max, 1.0)
        t_src = np.arange(len(P)) / RATE * scale
        t_new = np.arange(0, t_src[-1] + 1e-9, 1 / RATE)
        Pn = np.stack([np.interp(t_new, t_src, P[:, j]) for j in range(3)], 1)
        idx = np.clip(np.searchsorted(t_src, t_new) - 1, 0, len(P) - 2)
        frac = np.clip((t_new - t_src[idx]) / (t_src[idx + 1] - t_src[idx]), 0, 1)
        Qn = np.array([slerp(Qr[i], Qr[i + 1], f) for i, f in zip(idx, frac)])
        start = np.r_[home_p, home_q]
        poses = hold(start, a.hold_s)
        poses += segment(home_p, home_q, Pn[0], Qn[0], a.v_max, a.w_max)   # HOME3 medido -> HOME3 de la demo
        t_move = len(poses) / RATE
        poses += [np.r_[p, q] for p, q in zip(Pn, Qn)]
        t_end = len(poses) / RATE
        poses += hold(poses[-1], a.hold_s)
        t_ret = len(poses) / RATE
        last = poses[-1]
        poses += segment(last[:3], last[3:], home_p, home_q, a.v_max, a.w_max)
        poses += hold(start, a.hold_s)
        segments = [[0.0, t_move, "home3"], [t_move, t_end, "approach"], [t_end, t_ret, "end_hold"],
                    [t_ret, len(poses) / RATE, "return_home3"]]
        check(poses, box, reach, (a.pitch_min, a.pitch_max), a.z_floor)
        return finish(f"approach_{k}", "approach", poses, segments,
                      {"source_episode": k, "time_scale": round(float(scale), 2),
                       "end_xyz": np.round(Pn[-1], 3).tolist()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True, help="JSON con center del tablero en link_base")
    ap.add_argument("--home3", default="", help="JSON con xyz y q del TCP medido en HOME3 (si no, policy params)")
    ap.add_argument("--bridge-params", default="visuomotor_bringup/config/xarm5/xarm5_real_params.yaml")
    ap.add_argument("--dataset", default="datasets/demonstrations_real_v7_xarm5_v2.zarr")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--v-max", type=float, default=0.035, help="m/s (puente: 0.05)")
    ap.add_argument("--w-max", type=float, default=0.30, help="rad/s (puente: 0.5)")
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--pitch-min", type=float, default=math.radians(-36.0), help="HOME3 = -24.6°")
    ap.add_argument("--pitch-max", type=float, default=math.radians(20.0), help="demos v7: p99 = +18.6°")
    ap.add_argument("--z-floor", type=float, default=0.22, help="TCP mínimo de los planes (vigilante: 0.15)")
    ap.add_argument("--n-approach", type=int, default=8)
    a = ap.parse_args(argv)
    box, reach, _ = load_limits(REPO / a.bridge_params)
    h = json.load(open(a.home3))
    home_p, home_q = np.asarray(h["xyz"], float), np.asarray(h["q"], float) / np.linalg.norm(h["q"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plans = [static_plan(home_p, home_q, json.load(open(a.board)), box, reach, a)]
    plans += approach_plans(home_p, home_q, box, reach, a)
    for p in plans:
        (out / f"{p['name']}.json").write_text(json.dumps(p))
        print(f"{p['name']:24s} {p['duration_s']:6.1f} s  v_max {p['v_max_m_s']:.3f} m/s  w_max "
              f"{p['w_max_rad_s']:.2f} rad/s  z_min {p['z_min_m']:.3f} m"
              + (f"  fin {p['end_xyz']}  x{p['time_scale']}" if p["kind"] == "approach" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
