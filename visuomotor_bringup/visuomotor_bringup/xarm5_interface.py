"""Lógica pura (sin ROS) del puente entre la política visuomotora y el xArm5 real de UFACTORY.

Se mantiene separada del nodo ROS para poder verificarla con pytest sin hardware ni ``rclpy``.

Convenciones
------------
* ROS / política: posiciones en metros, orientación como cuaternión ``[qx, qy, qz, qw]``.
* Controlador xArm (``xarm_api`` con ``is_radian=true``): posiciones en **milímetros** y orientación
  como roll/pitch/yaw en **radianes** sobre ejes fijos X-Y-Z, es decir ``R = Rz(yaw) · Ry(pitch) · Rx(roll)``.

Restricción cinemática del xArm5 (5 GDL)
----------------------------------------
J1 gira sobre el eje vertical, J2-J4 son ejes paralelos horizontales y J5 gira sobre el eje de la
herramienta. Por eso no se puede alcanzar una orientación arbitraria (3 GDL) junto con una posición
arbitraria (3 GDL). La configuración "garra vertical hacia abajo con giro libre" sí es alcanzable en
todo el volumen de trabajo y es el modo por defecto (``top_down_yaw``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

ORIENTATION_MODES = ("top_down_yaw", "fixed", "passthrough")
GRIPPER_TYPES = ("xarm_gripper", "vacuum", "none")

_MM_PER_M = 1000.0


# ----------------------------------------------------------------------------------------------
# Ángulos y rotaciones
# ----------------------------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Normaliza un ángulo a [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quat(q: Sequence[float]) -> np.ndarray:
    """Normaliza ``[qx, qy, qz, qw]`` y fuerza ``qw >= 0`` (misma convención que ``observation.state``)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("Cuaternión de norma nula")
    q = q / n
    if q[3] < 0.0:
        q = -q
    return q


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY sobre ejes fijos (convención xArm) a cuaternión ``[qx, qy, qz, qw]`` con ``qw >= 0``."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    return normalize_quat(q)


def quat_to_rpy(q: Sequence[float]) -> Tuple[float, float, float]:
    """Cuaternión a RPY (convención xArm). En gimbal lock (pitch = ±90°) se fija roll = 0."""
    r = quat_to_matrix(q)
    sp = -r[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) > 1.0 - 1e-9:
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return wrap_angle(roll), pitch, wrap_angle(yaw)


def tool_yaw(q: Sequence[float]) -> Optional[float]:
    """Giro de la herramienta alrededor del eje vertical (proyección del eje X de la herramienta).

    Devuelve ``None`` si el eje X de la herramienta es casi vertical (yaw indefinido).
    """
    r = quat_to_matrix(q)
    hx, hy = r[0, 0], r[1, 0]
    if math.hypot(hx, hy) < 1e-3:
        return None
    return math.atan2(hy, hx)


def project_orientation(
    q: Sequence[float],
    mode: str,
    fixed_rpy: Sequence[float],
    yaw_offset: float = 0.0,
    fallback_yaw: float = 0.0,
) -> Tuple[float, float, float]:
    """Proyecta la orientación que pide la política a una orientación ejecutable por el xArm5.

    * ``top_down_yaw``: roll/pitch fijos (``fixed_rpy[0:2]``, garra vertical) y yaw tomado de la política.
    * ``fixed``: orientación constante ``fixed_rpy`` (ignora la orientación de la política).
    * ``passthrough``: RPY de la política sin modificar (solo depuración: el controlador puede rechazarla).
    """
    if mode == "top_down_yaw":
        yaw = tool_yaw(q)
        if yaw is None:
            yaw = fallback_yaw
        return float(fixed_rpy[0]), float(fixed_rpy[1]), wrap_angle(yaw + yaw_offset)
    if mode == "fixed":
        return float(fixed_rpy[0]), float(fixed_rpy[1]), float(fixed_rpy[2])
    if mode == "passthrough":
        roll, pitch, yaw = quat_to_rpy(q)
        return roll, pitch, wrap_angle(yaw + yaw_offset)
    raise ValueError(f"orientation_mode desconocido: {mode!r}. Opciones: {ORIENTATION_MODES}")


# ----------------------------------------------------------------------------------------------
# Límites cartesianos
# ----------------------------------------------------------------------------------------------
@dataclass
class WorkspaceLimits:
    """Caja de trabajo en ``link_base`` (metros) más límites radiales en el plano XY."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    min_radius_xy: float
    max_radius_xy: float

    def __post_init__(self):
        for lo, hi, name in ((self.x_min, self.x_max, "x"), (self.y_min, self.y_max, "y"),
                             (self.z_min, self.z_max, "z"), (self.min_radius_xy, self.max_radius_xy, "radius_xy")):
            if lo > hi:
                raise ValueError(f"Límite inválido en {name}: min={lo} > max={hi}")

    def clamp(self, xyz: Sequence[float]) -> Tuple[np.ndarray, bool]:
        """Recorta un punto al volumen permitido. Devuelve ``(punto, fue_recortado)``.

        Primero se aplica el límite radial (evita la singularidad sobre el eje de J1 y puntos fuera de
        alcance) y después la caja, que es la restricción dura final.
        """
        p = np.asarray(xyz, dtype=np.float64).copy()
        orig = p.copy()
        r = math.hypot(p[0], p[1])
        if r > self.max_radius_xy and r > 1e-12:
            p[:2] *= self.max_radius_xy / r
        elif r < self.min_radius_xy:
            if r > 1e-12:
                p[:2] *= self.min_radius_xy / r
            else:
                p[0] = self.min_radius_xy
        p[0] = min(max(p[0], self.x_min), self.x_max)
        p[1] = min(max(p[1], self.y_min), self.y_max)
        p[2] = min(max(p[2], self.z_min), self.z_max)
        return p, bool(np.any(np.abs(p - orig) > 1e-9))


def limit_step(prev_xyz: Sequence[float], target_xyz: Sequence[float], max_step: float) -> np.ndarray:
    """Avanza desde ``prev_xyz`` hacia ``target_xyz`` como máximo ``max_step`` metros (norma euclídea)."""
    prev = np.asarray(prev_xyz, dtype=np.float64)
    delta = np.asarray(target_xyz, dtype=np.float64) - prev
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-12:
        return prev + delta
    return prev + delta * (max_step / dist)


def limit_angle_step(prev: float, target: float, max_step: float, wrap: bool = True) -> float:
    """Igual que :func:`limit_step` para un ángulo.

    ``wrap=True``: camino más corto (puede cruzar ±pi). ``wrap=False``: interpolación lineal sin cruzar ±pi,
    usada para el yaw cuando está acotado a ``[yaw_min, yaw_max]`` (evita giros de 2*pi en la muñeca).
    """
    if wrap:
        delta = wrap_angle(target - prev)
        delta = max(-max_step, min(max_step, delta))
        return wrap_angle(prev + delta)
    delta = float(target) - float(prev)
    delta = max(-max_step, min(max_step, delta))
    return float(prev) + delta


def clamp_yaw(yaw: float, yaw_min: float, yaw_max: float) -> float:
    """Acota el yaw (normalizado a [-pi, pi)) al intervalo ``[yaw_min, yaw_max]`` dentro de (-pi, pi)."""
    if not (-math.pi <= yaw_min < yaw_max <= math.pi):
        raise ValueError(f"Rango de yaw inválido: [{yaw_min}, {yaw_max}]")
    return min(max(wrap_angle(yaw), yaw_min), yaw_max)


def is_finite_pose(xyz: Sequence[float], quat: Sequence[float]) -> bool:
    """``True`` si posición y cuaternión son finitos y el cuaternión no es nulo."""
    values = list(xyz) + list(quat)
    if not all(math.isfinite(float(v)) for v in values):
        return False
    return float(np.linalg.norm(np.asarray(quat, dtype=np.float64))) > 1e-9


# ----------------------------------------------------------------------------------------------
# Conversiones de unidades ROS <-> controlador xArm
# ----------------------------------------------------------------------------------------------
def to_servo_pose(xyz_m: Sequence[float], rpy: Sequence[float]) -> list:
    """Pose para ``set_servo_cartesian``: ``[x, y, z]`` en mm y ``[roll, pitch, yaw]`` en rad."""
    return [float(xyz_m[0]) * _MM_PER_M, float(xyz_m[1]) * _MM_PER_M, float(xyz_m[2]) * _MM_PER_M,
            float(rpy[0]), float(rpy[1]), float(rpy[2])]


def from_robot_pose(pose6: Sequence[float]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """``RobotMsg.pose`` (mm, rad) a ``(xyz en m, (roll, pitch, yaw))``."""
    if len(pose6) < 6:
        raise ValueError(f"Se esperaban 6 valores de pose, llegaron {len(pose6)}")
    xyz = np.array([pose6[0], pose6[1], pose6[2]], dtype=np.float64) / _MM_PER_M
    return xyz, (float(pose6[3]), float(pose6[4]), float(pose6[5]))


# ----------------------------------------------------------------------------------------------
# Frenos y habilitación de motores (RobotMsg.mt_brake / mt_able)
# ----------------------------------------------------------------------------------------------
def joints_with_bit_cleared(mask: int, dof: int) -> list:
    """Articulaciones (1..dof) cuyo bit en ``mask`` vale 0.

    Semántica del SDK UFACTORY (``xarm_report.cc``): ``motor_brake_states[i] = (mt_brake >> i) & 1`` y el robot solo
    está listo si todos los bits de sus ejes valen 1. Por tanto, en ``mt_brake`` un bit 0 = freno APLICADO
    (no liberado) y en ``mt_able`` un bit 0 = motor deshabilitado.
    """
    if dof < 1:
        raise ValueError("dof debe ser >= 1")
    mask = int(mask)
    return [j + 1 for j in range(dof) if not (mask >> j) & 0x01]


def brake_report(mt_brake: int, mt_able: int, dof: int) -> Tuple[bool, list, list]:
    """``(todo_ok, juntas_con_freno_aplicado, juntas_con_motor_deshabilitado)`` para los ejes 1..dof."""
    stuck = joints_with_bit_cleared(mt_brake, dof)
    disabled = joints_with_bit_cleared(mt_able, dof)
    return (not stuck and not disabled), stuck, disabled


# ----------------------------------------------------------------------------------------------
# Gripper
# ----------------------------------------------------------------------------------------------
class GripperBinarizer:
    """Convierte el comando continuo de la política en abrir/cerrar con histéresis.

    En la política ``0.0`` es abierto y ``<= close_threshold`` (p. ej. -0.001) es cerrado.
    Para volver a abrir, el comando debe superar ``close_threshold + hysteresis``.
    """

    def __init__(self, close_threshold: float, hysteresis: float, initially_closed: bool = False):
        if hysteresis < 0.0:
            raise ValueError("hysteresis debe ser >= 0")
        self.close_threshold = float(close_threshold)
        self.hysteresis = float(hysteresis)
        self.closed = bool(initially_closed)

    def update(self, command: float) -> bool:
        if self.closed:
            if command > self.close_threshold + self.hysteresis:
                self.closed = False
        elif command <= self.close_threshold:
            self.closed = True
        return self.closed


def linear_map(value: float, src_a: float, src_b: float, dst_a: float, dst_b: float) -> float:
    """Mapea linealmente ``value`` de ``[src_a, src_b]`` a ``[dst_a, dst_b]`` recortando al rango."""
    if abs(src_b - src_a) < 1e-12:
        raise ValueError("Rango de origen degenerado")
    t = (float(value) - src_a) / (src_b - src_a)
    t = min(max(t, 0.0), 1.0)
    return dst_a + t * (dst_b - dst_a)
