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
todo el volumen de trabajo (``top_down_yaw``).

Orientaciones alcanzables (modo ``arm_plane``): el eje de la herramienta solo puede inclinarse dentro del plano
vertical del brazo, de acimut ``phi = atan2(y, x)`` (J1), y J5 gira la herramienta sobre su propio eje. Con la
garra hacia abajo como referencia::

    R(phi, pitch, yaw) = Rz(phi) · Ry(pitch) · Rz(yaw - phi) · Rx(pi)

``pitch`` = inclinación en el plano del brazo (= J2 + J3 + J4 en el controlador) y ``yaw`` = giro de la
herramienta (con ``pitch = 0`` coincide con el yaw de ``top_down_yaw``: rpy = (pi, 0, yaw)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

ORIENTATION_MODES = ("top_down_yaw", "arm_plane", "fixed", "passthrough")
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


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Matriz de rotación 3x3 a cuaternión ``[qx, qy, qz, qw]`` con ``qw >= 0``."""
    r = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return normalize_quat(q)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


_RX_PI = np.diag([1.0, -1.0, -1.0])  # Rx(pi): garra hacia abajo


def rotation_angle(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    """Ángulo (rad, en [0, pi]) de la rotación que lleva ``q_a`` a ``q_b``."""
    d = abs(float(np.dot(normalize_quat(q_a), normalize_quat(q_b))))
    return 2.0 * math.acos(min(1.0, d))


def arm_azimuth(xyz: Sequence[float]) -> float:
    """Acimut del plano vertical del brazo que contiene el punto ``xyz`` (``link_base``)."""
    return math.atan2(float(xyz[1]), float(xyz[0]))


def arm_plane_matrix(azimuth: float, pitch: float, yaw: float) -> np.ndarray:
    """``R = Rz(phi) · Ry(pitch) · Rz(yaw - phi) · Rx(pi)`` (ver docstring del módulo)."""
    return _rz(azimuth) @ _ry(pitch) @ _rz(yaw - azimuth) @ _RX_PI


def arm_plane_quat(azimuth: float, pitch: float, yaw: float) -> np.ndarray:
    """Cuaternión alcanzable por el xArm5 para (acimut del brazo, inclinación en su plano, giro)."""
    return matrix_to_quat(arm_plane_matrix(azimuth, pitch, yaw))


def arm_plane_params(q: Sequence[float], azimuth: float, fallback_pitch: float = 0.0) -> Tuple[float, float]:
    """Descompone una orientación cualquiera en ``(pitch, yaw)`` alcanzables con el brazo en ``azimuth``.

    Proyecta el eje de la herramienta al plano vertical del brazo (``pitch``) y toma el giro alrededor del eje
    proyectado (``yaw``, en [-pi, pi)). Si la orientación ya es alcanzable, la descomposición es exacta. Si el eje
    de la herramienta es perpendicular al plano (inclinación indefinida) usa ``fallback_pitch``.
    """
    n = _rz(-azimuth) @ quat_to_matrix(q) @ _RX_PI      # = Ry(pitch) · Rz(yaw - phi) si es alcanzable
    sx, cz = float(n[0, 2]), float(n[2, 2])
    pitch = math.atan2(sx, cz) if math.hypot(sx, cz) > 1e-6 else float(fallback_pitch)
    k = _ry(-pitch) @ n                                 # ~ Rz(yaw - phi)
    return pitch, wrap_angle(azimuth + math.atan2(float(k[1, 0]), float(k[0, 0])))


def project_arm_plane(
    q: Sequence[float],
    azimuth: float,
    pitch_min: float,
    pitch_max: float,
    yaw_offset: float = 0.0,
    fallback_pitch: float = 0.0,
) -> Tuple[float, float, float]:
    """Orientación alcanzable más cercana a ``q``: ``(pitch, yaw, desviación_rad)``.

    ``pitch`` se satura a ``[pitch_min, pitch_max]``; ``desviación_rad`` es el ángulo entre la orientación pedida
    y la ejecutable (fuera del plano + saturación), útil para auditar datasets y vigilar la política.
    """
    if not pitch_min <= pitch_max:
        raise ValueError(f"Rango de pitch inválido: [{pitch_min}, {pitch_max}]")
    pitch, yaw = arm_plane_params(q, azimuth, fallback_pitch)
    pitch = min(max(pitch, float(pitch_min)), float(pitch_max))
    deviation = rotation_angle(q, arm_plane_quat(azimuth, pitch, yaw))
    return pitch, wrap_angle(yaw + yaw_offset), deviation


def continuous_rpy(rpy: Sequence[float], ref_rpy: Sequence[float]) -> Tuple[float, float, float]:
    """Expresa roll y yaw con el valor equivalente (±2*pi) más cercano a ``ref_rpy``, sin salir de [-pi, pi].

    Evita alternar entre +pi y -pi en el roll (garra hacia abajo) mientras el valor siga dentro del rango que
    acepta el controlador; si el equivalente más cercano saldría de [-pi, pi] se usa el ángulo normalizado.
    """
    def near(a: float, ref: float) -> float:
        v = float(ref) + wrap_angle(float(a) - float(ref))
        return v if abs(v) <= math.pi else wrap_angle(v)

    return near(rpy[0], ref_rpy[0]), float(rpy[1]), near(rpy[2], ref_rpy[2])


def project_orientation(
    q: Sequence[float],
    mode: str,
    fixed_rpy: Sequence[float],
    yaw_offset: float = 0.0,
    fallback_yaw: float = 0.0,
    azimuth: Optional[float] = None,
    pitch_limits: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float, float]:
    """Proyecta la orientación que pide la política a una orientación ejecutable por el xArm5.

    * ``top_down_yaw``: roll/pitch fijos (``fixed_rpy[0:2]``, garra vertical) y yaw tomado de la política.
    * ``arm_plane``: inclinación en el plano del brazo (acotada a ``pitch_limits``) + giro; requiere ``azimuth``.
    * ``fixed``: orientación constante ``fixed_rpy`` (ignora la orientación de la política).
    * ``passthrough``: RPY de la política sin modificar (solo depuración: el controlador puede rechazarla).
    """
    if mode == "arm_plane":
        if azimuth is None:
            raise ValueError("orientation_mode 'arm_plane' requiere el acimut del brazo")
        pitch, yaw, _ = project_arm_plane(q, azimuth, pitch_limits[0], pitch_limits[1], yaw_offset)
        return quat_to_rpy(arm_plane_quat(azimuth, pitch, yaw))
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


@dataclass
class ReachLimit:
    """Alcance del brazo: distancia hombro (J2) - muñeca (J4) acotada, lejos del codo recto (singularidad, C24).

    La caja de trabajo no basta: la altura alcanzable baja con el radio (p. ej. ~0.75 m cerca de la base y ~0.38 m
    a 0.65 m con la garra vertical). La muñeca se calcula desde la pose de la brida y la inclinación en el plano del
    brazo: ``wrist = p + Rz(phi)·Ry(pitch)·Rx(pi) · wrist_offset`` (``wrist_offset`` constante en ese marco; xArm5:
    (-0.076, 0, -0.097) m según la URDF), por eso el límite también es válido con la garra inclinada.
    ``min_reach`` excluye el brazo muy plegado junto a la base (inalcanzable por los límites de J3/J4).
    """

    shoulder_xyz: Sequence[float]
    wrist_offset: Sequence[float]
    max_reach: float
    min_reach: float = 0.0

    def __post_init__(self):
        self.shoulder_xyz = np.asarray(self.shoulder_xyz, dtype=np.float64)
        self.wrist_offset = np.asarray(self.wrist_offset, dtype=np.float64)
        if self.shoulder_xyz.shape != (3,) or self.wrist_offset.shape != (3,) or not 0.0 <= self.min_reach < self.max_reach:
            raise ValueError("ReachLimit: shoulder_xyz y wrist_offset de 3 elementos y 0 <= min_reach < max_reach")
        if abs(float(self.shoulder_xyz[0])) > 1e-9 or abs(float(self.shoulder_xyz[1])) > 1e-9:
            raise ValueError("ReachLimit: el hombro debe estar sobre el eje de J1 (x = y = 0)")

    def _offset(self, xyz, pitch: float) -> np.ndarray:
        return (_rz(arm_azimuth(xyz)) @ _ry(pitch) @ _RX_PI) @ self.wrist_offset

    def wrist_center(self, xyz: Sequence[float], pitch: float = 0.0) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz + self._offset(xyz, pitch)

    def reach(self, xyz: Sequence[float], pitch: float = 0.0) -> float:
        return float(np.linalg.norm(self.wrist_center(xyz, pitch) - self.shoulder_xyz))

    def clamp(self, xyz: Sequence[float], pitch: float = 0.0) -> Tuple[np.ndarray, bool]:
        """Lleva la muñeca a ``[min_reach, max_reach]`` del hombro en línea recta (mismo acimut) y recalcula la brida."""
        xyz = np.asarray(xyz, dtype=np.float64)
        offset = self._offset(xyz, pitch)
        v = xyz + offset - self.shoulder_xyz
        d = float(np.linalg.norm(v))
        if self.min_reach <= d <= self.max_reach:
            return xyz, False
        if d < 1e-9:  # muñeca sobre el hombro: alejarla en horizontal según el acimut del objetivo
            az = arm_azimuth(xyz)
            v, d = np.array([math.cos(az), math.sin(az), 0.0]), 1.0
        target = self.max_reach if d > self.max_reach else self.min_reach
        # el hombro está sobre el eje de J1: escalar v conserva el acimut, y con él el desplazamiento de la muñeca
        return self.shoulder_xyz + v * (target / d) - offset, True


def clamp_to_workspace(xyz: Sequence[float], pitch: float, box: "WorkspaceLimits",
                       reach: Optional[ReachLimit]) -> Tuple[np.ndarray, bool, bool]:
    """Caja + alcance (alternando hasta 4 veces). Devuelve ``(xyz, recortado, válido)``.

    ``válido`` es ``False`` si no existe un punto que cumpla ambos a la vez cerca del pedido (p. ej. esquina de la caja
    fuera de alcance): el llamador debe mantener la posición en vez de enviarlo.
    """
    out, clamped = box.clamp(xyz)
    if reach is None:
        return out, clamped, True
    for _ in range(4):
        out, c_reach = reach.clamp(out, pitch)
        out, c_box = box.clamp(out)
        clamped = clamped or c_reach or c_box
        if not c_box:
            break
    d = reach.reach(out, pitch)
    valid = reach.min_reach - 1e-6 <= d <= reach.max_reach + 1e-6 and not box.clamp(out)[1]
    return out, clamped, valid


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


def clamp_yaw_near(yaw: float, current: float, yaw_min: float, yaw_max: float) -> float:
    """Como :func:`clamp_yaw`, pero si el yaw pedido queda FUERA del rango se usa el límite más cercano al yaw
    ``current`` (el comandado en el ciclo anterior, o el del frame anterior), no el del lado del pedido.

    Cerca de ±pi el pedido salta entre +pi y -pi de un ciclo a otro (2026-11-26: demos con las hojas a 180° y la
    cámara alineada). Con :func:`clamp_yaw` el objetivo alternaría entre los dos límites, y como el yaw se interpola
    sin cruzar ±pi, la muñeca intentaría dar casi una vuelta entera ida y vuelta. Dentro del rango es idéntico a
    :func:`clamp_yaw` (también los giros largos sin cruzar ±pi)."""
    y = wrap_angle(yaw)
    if yaw_min <= y <= yaw_max:
        return clamp_yaw(y, yaw_min, yaw_max)
    clamp_yaw(0.0, yaw_min, yaw_max)                                   # valida el rango
    d_min, d_max = abs(float(current) - yaw_min), abs(float(current) - yaw_max)
    if abs(d_min - d_max) < 1e-9:                                      # equidistante: el lado del pedido
        return yaw_max if y >= 0.0 else yaw_min
    return yaw_min if d_min < d_max else yaw_max


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
# Límites articulares
# ----------------------------------------------------------------------------------------------
def joints_near_limits(
    joints: Sequence[float], lower: Sequence[float], upper: Sequence[float], margin: float
) -> list:
    """Articulaciones a menos de ``margin`` rad de su límite (o fuera de él).

    Devuelve ``[(numero_junta, valor, limite_cercano), ...]`` (juntas numeradas desde 1). Con servo cartesiano el
    controlador resuelve la IK cerca de la configuración actual: si una junta está pegada a su límite (p. ej. J1 en
    +2*pi, el caso real del xArm5), cualquier paso que la empuje más allá termina en error C23 (Joint Angle Exceed Limit).
    """
    if margin < 0.0:
        raise ValueError("margin debe ser >= 0")
    n = min(len(joints), len(lower), len(upper))
    near = []
    for i in range(n):
        q, lo, hi = float(joints[i]), float(lower[i]), float(upper[i])
        if lo >= hi:
            raise ValueError(f"Límites inválidos en J{i + 1}: [{lo}, {hi}]")
        if q - lo < margin:
            near.append((i + 1, q, lo))
        elif hi - q < margin:
            near.append((i + 1, q, hi))
    return near


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
