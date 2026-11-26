"""Monitor del alcance del xArm5 durante la recolección A MANO (2026-11-25).

Las demos v8 se graban moviendo la cámara con la mano. La mano puede llevarla a poses que el brazo no alcanza. En la
primera toma, el 64 % de las demos pasó por alguna (mediana del 101 % del codo recto, frente al 88 % de v7):
sobre todo al avanzar ALTO hacia las celdas lejanas antes de bajar, y en algunos finales de esas celdas. El conversor
recorta esas poses, así que la acción del dataset deja de coincidir con la imagen.

Este monitor calcula, con la pose relativa de la cámara desde el inicio (odometría en vivo) y suponiendo que la demo
empezó en HOME3 (la guía de inicio lo asegura: 6-15 mm), dónde estaría el TCP. Usa la MISMA fórmula que el
conversor:
    T_base_tcp(t) = T_base_home · T_tcp_cam · T_rel(t) · T_cam_tcp
y el MISMO modelo del puente: caja de trabajo, alcance hombro-muñeca y proyección de 5 GDL. Sin ROS: lo usan
``vo_collector_node`` y sus pruebas.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from visuomotor_bringup.xarm5_interface import (ReachLimit, WorkspaceLimits, arm_azimuth,  # noqa: F401
                                                arm_plane_params)


def pose_matrix(xyz: Sequence[float], rotation: Rotation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation.as_matrix()
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def rpy_rotation(roll: float, pitch: float, yaw: float) -> Rotation:
    """Convención de ROS (y de ``convert_vio_dataset_to_xarm5.rpy_to_quat``): R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw])


class ReachMonitor:
    """``check(T_rel)`` -> dónde estaría el TCP y si el robot llega (caja + alcance), con el motivo."""

    def __init__(self, bridge_yaml, alignment_yaml, home_yaml, warn_ratio: float = 0.97, z_tolerance_m: float = 0.02):
        g = yaml.safe_load(open(bridge_yaml))["xarm5_bridge"]["ros__parameters"]
        self.box = WorkspaceLimits(g["workspace_x_min"], g["workspace_x_max"], g["workspace_y_min"],
                                   g["workspace_y_max"], g["workspace_z_min"], g["workspace_z_max"],
                                   g["workspace_min_radius_xy"], g["workspace_max_radius_xy"])
        self.reach = ReachLimit(g["workspace_shoulder_xyz_m"], g["workspace_wrist_offset_m"],
                                g["workspace_max_reach_m"], g["workspace_min_reach_m"])
        self.pitch_limits = (float(g["arm_plane_pitch_min"]), float(g["arm_plane_pitch_max"]))
        # Giro de la herramienta (J5) que acepta el puente: sin cruzar ±pi (2026-11-26: las demos con las hojas a 180°
        # y la cámara alineada piden ±180°, fuera de ±2,6 rad; el conversor las recorta y marca)
        self.yaw_limits = (float(g["yaw_min"]), float(g["yaw_max"]))
        al = yaml.safe_load(open(alignment_yaml))["xarm5_dataset_alignment"]
        self.T_tcp_cam = pose_matrix(al["tcp_to_camera_xyz_m"], rpy_rotation(*al["tcp_to_camera_rpy_rad"]))
        self.T_cam_tcp = np.linalg.inv(self.T_tcp_cam)
        pol = yaml.safe_load(open(home_yaml))["policy_node"]["ros__parameters"]
        self.T_base_home = pose_matrix([pol["home_x"], pol["home_y"], pol["home_z"]],
                                       rpy_rotation(pol["home_roll"], pol["home_pitch"], pol["home_yaw"]))
        self.warn_ratio = float(warn_ratio)
        self.z_tol = float(z_tolerance_m)

    def tcp_pose(self, T_rel: np.ndarray) -> np.ndarray:
        return self.T_base_home @ self.T_tcp_cam @ np.asarray(T_rel, dtype=np.float64) @ self.T_cam_tcp

    def check(self, T_rel: np.ndarray) -> Dict:
        T = self.tcp_pose(T_rel)
        p = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        pitch, yaw = arm_plane_params(q, arm_azimuth(p))
        pitch = min(max(pitch, self.pitch_limits[0]), self.pitch_limits[1])
        ratio = self.reach.reach(p, pitch) / self.reach.max_reach
        reasons: List[str] = []
        if ratio > 1.0:
            reasons.append("brazo demasiado estirado")
        if p[0] > self.box.x_max or math.hypot(p[0], p[1]) > self.box.max_radius_xy:
            reasons.append("demasiado lejos de la base")
        if abs(p[1]) > self.box.y_max:
            reasons.append("demasiado al costado")
        if p[2] > self.box.z_max + self.z_tol:
            reasons.append("demasiado alto")
        if p[2] < self.box.z_min:
            reasons.append("demasiado bajo")
        if not (self.yaw_limits[0] <= yaw <= self.yaw_limits[1]):
            reasons.append("cámara demasiado girada")
        outside = bool(reasons)
        return {"outside": outside, "near": (not outside) and ratio > self.warn_ratio, "ratio": float(ratio),
                "reasons": reasons, "tcp_xyz": p.tolist(), "yaw_rad": float(yaw)}


def relative_transform(p0, q0, p, q) -> np.ndarray:
    """T_rel = T(p0, q0)^-1 · T(p, q) con cuaterniones xyzw."""
    T0 = pose_matrix(p0, Rotation.from_quat(np.asarray(q0, dtype=np.float64)))
    T1 = pose_matrix(p, Rotation.from_quat(np.asarray(q, dtype=np.float64)))
    return np.linalg.inv(T0) @ T1


def advice(reasons: Sequence[str]) -> str:
    """Qué hacer, en pocas palabras, para volver al alcance del robot (a lo más dos acciones, la más importante
    primero: girar menos la cámara, después la posición)."""
    acts = []
    if "cámara demasiado girada" in reasons:
        acts.append("gira menos la cámara")
    if "brazo demasiado estirado" in reasons or "demasiado lejos de la base" in reasons:
        acts.append("baja primero y después avanza")
    elif "demasiado alto" in reasons:
        acts.append("baja un poco")
    if "demasiado al costado" in reasons:
        acts.append("acércate al centro")
    if "demasiado bajo" in reasons:
        acts.append("sube un poco")
    return " y ".join(acts[:2]) if acts else "baja primero y después avanza"


def summarize(stats: Optional[Dict]) -> Dict:
    stats = stats or {}
    n = max(1, int(stats.get("frames", 0)))
    return {"frames_checked": int(stats.get("frames", 0)), "frames_outside": int(stats.get("outside", 0)),
            "fraction_outside": round(int(stats.get("outside", 0)) / n, 3),
            "max_reach_ratio": round(float(stats.get("max_ratio", 0.0)), 3)}
