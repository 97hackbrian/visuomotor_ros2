#!/usr/bin/env python3
"""Convierte demostraciones reales (cámara a mano + RTAB-Map VIO) al espacio del xArm5 para entrenar Diffusion Policy.

Problema que resuelve
---------------------
``vo_collector_node`` guarda la pose de la CÁMARA en el marco ``odom`` de RTAB-Map (``frame_id: camera_link``,
origen y ejes arbitrarios). En el robot, ``policy_node`` observa el TCP del xArm5 en ``link_base`` (TF
``link_base -> xarm_tcp``) y resta ``home_x/y/z``. Sin esta conversión, el checkpoint aprende un espacio que no
corresponde con el del robot. Además el xArm5 (5 GDL) solo alcanza inclinaciones en el plano vertical del brazo:
las acciones se proyectan con el MISMO modo y límites que aplica ``xarm5_bridge``, para que lo aprendido sea
ejecutable tal cual.

Por cada frame (state y action):
    [opcional] filtro de la trayectoria VIO (Hampel + Butterworth de fase cero, ``trajectory_filter`` en la
               alineación): elimina el ruido del VIO (~8 mm RMS en v7) que inflaba 4x la velocidad aparente
    T_rel(t)      = T_odom_cam(0)^-1 · T_odom_cam(t)                    (movimiento relativo al inicio)
    T_base_tcp(t) = T_base_tcp_home · T_tcp_cam · T_rel(t) · T_tcp_cam^-1
    orientación   -> proyección alcanzable (orientation_mode del puente, p. ej. arm_plane)
    posición      -> menos home_x/y/z si relative_coordinates (igual que policy_node)
    signo del cuaternión -> ``quaternion_hemisphere`` del policy_node ('home': dot(q, q_home) >= 0, continuo;
                            'qw_positive': convención histórica, discontinua con la garra hacia abajo)
El gripper (state[7], action[7]), done (action[8]) y las imágenes se copian sin cambios.

Seguridad de datos: el dataset de origen se abre en solo lectura; el destino debe NO existir (nunca se sobrescribe).

Uso:
    python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_v6.zarr
        [--dst datasets/demonstrations_real_v6_xarm5.zarr] [--report-only] [--drop-flagged] [--allow-uncalibrated]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Optional
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))
sys.path.insert(0, str(REPO / "visuomotor_core"))

from visuomotor_bringup.xarm5_interface import (  # noqa: E402
    ReachLimit,
    WorkspaceLimits,
    arm_azimuth,
    arm_plane_params,
    arm_plane_quat,
    clamp_to_workspace,
    clamp_yaw,
    matrix_to_quat,
    project_arm_plane,
    project_orientation,
    quat_to_matrix,
    rotation_angle,
    rpy_to_quat,
)
from visuomotor_core.dataset.trajectory_filters import TrajectoryFilterConfig, smooth_pose_trajectory  # noqa: E402
from visuomotor_core.utils.rotations import HEMISPHERE_MODES, canonicalize  # noqa: E402

CONVERTER_VERSION = 3  # 2: filtro + hemisferio + auditoría de saltos; 3: anclaje opcional con la pose de inicio medida (--start-poses)

DEFAULT_BRIDGE_PARAMS = REPO / "visuomotor_bringup/config/xarm5/xarm5_real_params.yaml"
DEFAULT_ALIGNMENT = REPO / "visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml"
DEFAULT_POLICY_PARAMS = REPO / "deploy/jetson/config/jetson_policy_params.yaml"


def pose_matrix(xyz, quat) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(quat)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def inv(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = -r @ t[:3, 3]
    return out


class Converter:
    def __init__(self, bridge: dict, alignment: dict, policy: dict):
        self.mode = bridge["orientation_mode"]
        if self.mode not in ("arm_plane", "top_down_yaw"):
            raise ValueError(f"orientation_mode={self.mode!r}: solo 'arm_plane' o 'top_down_yaw' son entrenables")
        self.fixed_rpy = (float(bridge["top_down_roll"]), float(bridge["top_down_pitch"]), float(bridge["fixed_yaw"]))
        self.pitch_limits = (float(bridge.get("arm_plane_pitch_min", 0.0)), float(bridge.get("arm_plane_pitch_max", 0.0)))
        self.yaw_limits = (float(bridge["yaw_min"]), float(bridge["yaw_max"]))
        self.workspace = WorkspaceLimits(
            bridge["workspace_x_min"], bridge["workspace_x_max"], bridge["workspace_y_min"], bridge["workspace_y_max"],
            bridge["workspace_z_min"], bridge["workspace_z_max"], bridge["workspace_min_radius_xy"],
            bridge["workspace_max_radius_xy"])
        self.reach = ReachLimit(bridge["workspace_shoulder_xyz_m"], bridge["workspace_wrist_offset_m"],
                                bridge["workspace_max_reach_m"], bridge["workspace_min_reach_m"])
        self.max_lin = float(bridge["max_linear_speed_m_s"])
        self.max_ang = float(bridge["max_angular_speed_rad_s"])

        self.home_xyz = np.array([policy["home_x"], policy["home_y"], policy["home_z"]], dtype=np.float64)
        self.home_rpy = (float(policy["home_roll"]), float(policy["home_pitch"]), float(policy["home_yaw"]))
        self.relative = bool(policy["relative_coordinates"])
        self.t_base_home = pose_matrix(self.home_xyz, rpy_to_quat(*self.home_rpy))

        self.t_tcp_cam = pose_matrix(alignment["tcp_to_camera_xyz_m"], rpy_to_quat(*alignment["tcp_to_camera_rpy_rad"]))
        self.t_cam_tcp = inv(self.t_tcp_cam)
        self.calibrated = bool(alignment["extrinsic_calibrated"])
        self.max_dev = float(alignment["max_orientation_deviation_rad"])
        self.max_ws_frames = int(alignment["max_workspace_violation_frames"])
        self.speed_ratio = float(alignment["speed_tolerance_ratio"])
        self.speed_window = max(1, int(alignment["speed_window_steps"]))
        # Opcionales (ausentes = comportamiento de la versión 1)
        tf = alignment.get("trajectory_filter")
        self.filter_cfg = TrajectoryFilterConfig.from_dict(tf) if tf else TrajectoryFilterConfig(enabled=False)
        self.max_raw_step = float(alignment.get("max_raw_step_m", float("inf")))
        self.max_ws_fraction = float(alignment.get("max_workspace_violation_fraction", 1.0))
        # recorte del puente por debajo de esta distancia no cuenta como violación (0 = cualquier recorte, v1)
        self.ws_tolerance = float(alignment.get("workspace_tolerance_m", 0.0))
        # convención del dataset: alineación (fuente propia); compatibilidad: policy YAML si declara un modo concreto
        pol_mode = str(policy.get("quaternion_hemisphere", "qw_positive"))
        self.hemisphere = str(alignment.get("quaternion_hemisphere",
                                            pol_mode if pol_mode in HEMISPHERE_MODES else "qw_positive"))
        if self.hemisphere not in HEMISPHERE_MODES:
            raise ValueError(f"quaternion_hemisphere={self.hemisphere!r}; opciones: {HEMISPHERE_MODES}")
        self.q_ref = rpy_to_quat(*self.home_rpy)

    def project(self, xyz: np.ndarray, quat) -> np.ndarray:
        """Orientación ejecutable con la misma secuencia que xarm5_bridge (sin yaw_offset: es calibración del robot)."""
        az = arm_azimuth(xyz)
        if self.mode == "arm_plane":
            pitch, yaw, _ = project_arm_plane(quat, az, self.pitch_limits[0], self.pitch_limits[1])
            return arm_plane_quat(az, pitch, clamp_yaw(yaw, *self.yaw_limits))
        roll, pitch, yaw = project_orientation(quat, "top_down_yaw", self.fixed_rpy)
        return rpy_to_quat(roll, pitch, clamp_yaw(yaw, *self.yaw_limits))

    def to_base(self, poses: np.ndarray, ref_inv: np.ndarray, t_base_start: Optional[np.ndarray] = None):
        """poses [T, 7] (xyz + quat en odom) -> (xyz [T,3] TCP en link_base, quat original [T,4] en link_base).

        ``t_base_start``: pose MEDIDA del TCP al inicio del episodio (ArUco + IMU, ``estimate_start_poses.py``). Si es
        ``None`` se supone que el episodio empieza en HOME (comportamiento de las versiones 1 y 2)."""
        t_start = self.t_base_home if t_base_start is None else np.asarray(t_base_start, dtype=np.float64)
        xyz = np.zeros((len(poses), 3))
        quats = np.zeros((len(poses), 4))
        for i, p in enumerate(poses):
            t = t_start @ self.t_tcp_cam @ (ref_inv @ pose_matrix(p[:3], p[3:7])) @ self.t_cam_tcp
            xyz[i] = t[:3, 3]
            quats[i] = matrix_to_quat(t[:3, :3])
        return xyz, quats

    def speeds(self, xyz: np.ndarray, quats: np.ndarray, rate_hz: float):
        w = self.speed_window
        if len(xyz) <= w:
            return 0.0, 0.0
        lin = np.linalg.norm(xyz[w:] - xyz[:-w], axis=1) * rate_hz / w
        ang = np.array([rotation_angle(quats[i], quats[i + w]) for i in range(len(quats) - w)]) * rate_hz / w
        return float(lin.max()), float(ang.max())

    def filter_poses(self, poses: np.ndarray, rate_hz: float):
        """Filtra ``poses [T, 7]`` (xyz + quat en odom) si ``trajectory_filter.enabled``; devuelve (poses, stats)."""
        if not self.filter_cfg.enabled:
            return np.asarray(poses, dtype=np.float64), {}
        p, q, fstats = smooth_pose_trajectory(poses[:, :3], poses[:, 3:7], rate_hz, self.filter_cfg)
        return np.concatenate([p, q], axis=1), fstats

    def convert_episode(self, state: np.ndarray, action: np.ndarray, rate_hz: float,
                        t_base_start: Optional[np.ndarray] = None):
        raw_steps = np.linalg.norm(np.diff(np.asarray(state[:, :3], dtype=np.float64), axis=0), axis=1)
        poses = {}
        filter_stats = {}
        for name, arr in (("state", state), ("action", action)):
            poses[name], fs = self.filter_poses(np.asarray(arr[:, :7], dtype=np.float64), rate_hz)
            if name == "state":
                filter_stats = fs
        # cámara al inicio del episodio (ya filtrada) = cámara en HOME
        ref_inv = inv(pose_matrix(poses["state"][0, :3], poses["state"][0, 3:7]))
        out = {}
        stats = {}
        for name, arr in (("state", state), ("action", action)):
            xyz, q_req = self.to_base(poses[name], ref_inv, t_base_start)
            q_proj = np.array([self.project(xyz[i], q_req[i]) for i in range(len(xyz))])
            dev = np.array([rotation_angle(q_req[i], q_proj[i]) for i in range(len(xyz))])
            q_proj = canonicalize(q_proj, self.hemisphere, self.q_ref)
            new = arr.astype(np.float64).copy()
            new[:, :3] = xyz - self.home_xyz if self.relative else xyz
            new[:, 3:7] = q_proj
            out[name] = new.astype(np.float32)
            if name == "action":
                clamp_dist = np.array([
                    np.linalg.norm(clamp_to_workspace(p, arm_plane_params(qp, arm_azimuth(p))[0], self.workspace,
                                                      self.reach)[0] - p) for p, qp in zip(xyz, q_proj)])
                ws = int((clamp_dist > self.ws_tolerance).sum())
                lin, ang = self.speeds(xyz, q_proj, rate_hz)
                q_steps = np.sum(q_proj[1:] * q_proj[:-1], axis=1)
                stats = {
                    "frames": int(len(xyz)),
                    "raw_max_step_m": float(raw_steps.max()) if len(raw_steps) else 0.0,
                    "trajectory_filter": filter_stats,
                    "quaternion_sign_flips": int((q_steps < 0.0).sum()),
                    "workspace_violation_fraction": float(ws) / max(1, len(xyz)),
                    "workspace_max_clamp_m": float(clamp_dist.max()) if len(clamp_dist) else 0.0,
                    "max_orientation_deviation_rad": float(dev.max()),
                    "mean_orientation_deviation_rad": float(dev.mean()),
                    "frames_over_deviation": int((dev > self.max_dev).sum()),
                    "workspace_violation_frames": int(ws),
                    "max_linear_speed_m_s": lin,
                    "max_angular_speed_rad_s": ang,
                    "tcp_xyz_min": [float(v) for v in xyz.min(axis=0)],
                    "tcp_xyz_max": [float(v) for v in xyz.max(axis=0)],
                    "start_offset_from_home_m": float(np.linalg.norm(xyz[0] - self.home_xyz)),
                    "start_anchor": "measured" if t_base_start is not None else "home",
                }
                reasons = []
                if stats["max_orientation_deviation_rad"] > self.max_dev:
                    reasons.append("orientación fuera de lo alcanzable por el xArm5")
                if ws > self.max_ws_frames or float(ws) / max(1, len(xyz)) > self.max_ws_fraction:
                    reasons.append("TCP fuera del volumen de trabajo o del alcance del brazo")
                if stats["raw_max_step_m"] > self.max_raw_step:
                    reasons.append(f"salto de tracking VIO {stats['raw_max_step_m']:.3f} m > {self.max_raw_step:.3f} m")
                if lin > self.speed_ratio * self.max_lin:
                    reasons.append(f"velocidad lineal {lin:.3f} > {self.speed_ratio * self.max_lin:.3f} m/s")
                if ang > self.speed_ratio * self.max_ang:
                    reasons.append(f"velocidad angular {ang:.3f} > {self.speed_ratio * self.max_ang:.3f} rad/s")
                stats["flags"] = reasons
        return out, stats


def episode_keys(root):
    keys = [k for k in root.group_keys() if k.startswith("episode_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dataset Zarr de origen (solo lectura)")
    ap.add_argument("--dst", default=None, help="dataset Zarr de destino (no debe existir); defecto: <src>_xarm5.zarr")
    ap.add_argument("--bridge-params", default=str(DEFAULT_BRIDGE_PARAMS))
    ap.add_argument("--alignment", default=str(DEFAULT_ALIGNMENT))
    ap.add_argument("--policy-params", default=str(DEFAULT_POLICY_PARAMS))
    ap.add_argument("--report-only", action="store_true", help="solo audita y escribe el informe JSON")
    ap.add_argument("--drop-flagged", action="store_true", help="no copia los episodios marcados en la auditoría")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="permite convertir con extrinsic_calibrated: false (solo pruebas)")
    ap.add_argument("--start-poses", default="",
                    help="JSON de tools/xarm5/estimate_start_poses.py: ancla cada episodio con su pose de inicio "
                         "MEDIDA (ArUco + IMU) en vez de suponer HOME. Episodios sin pose fiable: HOME")
    args = ap.parse_args(argv)
    starts = {}
    if args.start_poses:
        for k, v in json.loads(Path(args.start_poses).read_text())["episodes"].items():
            if v.get("T_base_camera_link") is not None and v.get("usable", False):
                starts[k] = np.asarray(v["T_base_camera_link"], dtype=np.float64)

    import zarr  # import tardío: --help funciona sin zarr

    bridge = yaml.safe_load(open(args.bridge_params))["xarm5_bridge"]["ros__parameters"]
    alignment = yaml.safe_load(open(args.alignment))["xarm5_dataset_alignment"]
    policy = yaml.safe_load(open(args.policy_params))["policy_node"]["ros__parameters"]
    conv = Converter(bridge, alignment, policy)
    if not conv.calibrated and not args.allow_uncalibrated and not args.report_only:
        sys.exit("ERROR: extrinsic_calibrated: false en la alineación. Mide la extrínseca cámara-TCP "
                 f"({args.alignment}) o usa --allow-uncalibrated solo para pruebas.")

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve() if args.dst else src.with_name(src.stem + "_xarm5.zarr")
    if dst == src:
        sys.exit("ERROR: el destino no puede ser el origen.")
    if not args.report_only and dst.exists():
        sys.exit(f"ERROR: {dst} ya existe; nunca se sobrescribe un dataset.")

    root = zarr.open_group(str(src), mode="r")
    out_root = None if args.report_only else zarr.open_group(str(dst), mode="w-")
    report = {"source": src.name, "destination": None if args.report_only else dst.name,
              "timestamp": time.time(), "orientation_mode": conv.mode, "pitch_limits_rad": list(conv.pitch_limits),
              "yaw_limits_rad": list(conv.yaw_limits), "home_xyz_m": conv.home_xyz.tolist(),
              "home_rpy_rad": list(conv.home_rpy), "relative_coordinates": conv.relative,
              "tcp_to_camera_xyz_m": alignment["tcp_to_camera_xyz_m"],
              "tcp_to_camera_rpy_rad": alignment["tcp_to_camera_rpy_rad"],
              "extrinsic_calibrated": conv.calibrated, "converter_version": CONVERTER_VERSION,
              "trajectory_filter": conv.filter_cfg.to_dict(), "quaternion_hemisphere": conv.hemisphere,
              "quaternion_reference_xyzw": [float(v) for v in conv.q_ref],
              "start_poses": args.start_poses or None, "episodes": {}}
    n_out = 0
    for key in episode_keys(root):
        ep = root[key]
        attrs = dict(ep.attrs)
        if attrs.get("discarded"):                            # descartada al grabar ([U] o a pedido): se omite
            report["episodes"][key] = {"skipped": "discarded", "reason": attrs.get("discard_reason", "")}
            print(f"  {key}: descartada ({attrs.get('discard_reason', '')}): se omite")
            continue
        rate = float(attrs.get("sampling_rate_hz", 20.0))
        state, action = np.asarray(ep["state"]), np.asarray(ep["action"])
        if state.shape[1] != 8 or action.shape[1] < 8:
            sys.exit(f"ERROR: {key}: state {state.shape} / action {action.shape} (se espera 8 / 9 dimensiones)")
        # Pose medida de camera_link al inicio -> TCP: T_base_tcp = T_base_cam · T_cam_tcp (misma extrínseca del YAML)
        t_start = starts[key] @ conv.t_cam_tcp if key in starts else None
        data, stats = conv.convert_episode(state, action, rate, t_start)
        stats["source_episode"] = key
        flagged = bool(stats["flags"])
        written = None
        if out_root is not None and not (flagged and args.drop_flagged):
            written = f"episode_{n_out}"
            g = out_root.create_group(written)
            img = ep["image"]
            g.create_dataset("image", data=np.asarray(img), chunks=(1, *img.shape[1:]), dtype="uint8")
            g.create_dataset("state", data=data["state"], chunks=(1, data["state"].shape[1]), dtype="float32")
            g.create_dataset("action", data=data["action"], chunks=(1, data["action"].shape[1]), dtype="float32")
            for extra in ep.array_keys():
                if extra not in ("image", "state", "action"):
                    g.create_dataset(extra, data=np.asarray(ep[extra]))
            attrs.update({"xarm5_aligned": True, "source_dataset": src.name, "source_episode": key,
                          "frame": bridge["base_frame"], "ee_frame": bridge["tcp_frame"],
                          "orientation_mode": conv.mode, "relative_coordinates": conv.relative,
                          "home_xyz_m": conv.home_xyz.tolist(), "audit_flags": stats["flags"],
                          "converter_version": CONVERTER_VERSION, "quaternion_hemisphere": conv.hemisphere,
                          "start_anchor": stats["start_anchor"],
                          "trajectory_filter": conv.filter_cfg.to_dict(),
                          "max_orientation_deviation_rad": stats["max_orientation_deviation_rad"]})
            g.attrs.update(attrs)
            n_out += 1
        stats["written_as"] = written
        report["episodes"][key] = stats
        mark = "MARCADO: " + "; ".join(stats["flags"]) if flagged else "ok"
        print(f"{key} -> {written or '-'} | {stats['frames']} frames | desv. orient. máx "
              f"{math.degrees(stats['max_orientation_deviation_rad']):.1f}° | v máx {stats['max_linear_speed_m_s']:.3f} m/s"
              f" | fuera de volumen {stats['workspace_violation_frames']} | {mark}")

    if out_root is not None:
        out_root.attrs.update({"xarm5_conversion": {k: v for k, v in report.items() if k != "episodes"}})
    report_path = (dst.with_name(dst.stem + "_conversion_report.json") if not args.report_only
                   else src.with_name(src.stem + "_xarm5_audit_report.json"))
    if report_path.exists():
        report_path = report_path.with_name(report_path.stem + f"_{int(time.time())}.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    n_flag = sum(1 for e in report["episodes"].values() if e.get("flags"))
    n_skip = sum(1 for e in report["episodes"].values() if e.get("skipped"))
    print(f"\n{len(report['episodes'])} episodios, {n_skip} descartados al grabar (omitidos), {n_flag} marcados, "
          f"{n_out} escritos"
          + ("" if args.report_only else f" en {dst}") + f". Informe: {report_path}")
    if not conv.calibrated:
        print("AVISO: extrínseca cámara-TCP sin calibrar (extrinsic_calibrated: false): no entrenar con este resultado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
