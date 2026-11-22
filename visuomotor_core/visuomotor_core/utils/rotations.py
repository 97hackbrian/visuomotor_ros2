"""Utilidades de cuaterniones sin dependencias pesadas (solo NumPy), compartidas por la conversión de datasets,
el entrenamiento y la inferencia (``policy_node``, Python 3.8 en la Jetson).

Convención: ``[qx, qy, qz, qw]``.

Por qué existe la convención de hemisferio ``"home"``
----------------------------------------------------
``q`` y ``-q`` representan la misma rotación. La convención clásica ``qw >= 0`` es DISCONTINUA cerca de rotaciones
de ~pi (``qw ~ 0``). Con la garra del xArm5 hacia abajo (``Rx(pi)``) se cumple exactamente ese caso: en la
proyección ``arm_plane`` ``qw = -sin(pitch/2)·sin((yaw - 2·acimut)/2)``, que cambia de signo con movimientos de
milímetros alrededor de HOME3. Con ``qw >= 0`` el cuaternión entero salta (``qx`` de +0.98 a -0.98) y la red debe
aprender un objetivo discontinuo; además, promediar cuaterniones de signo opuesto (ensemble temporal) los anula.
Con ``"home"`` se elige el signo tal que ``dot(q, q_home) >= 0``: es continuo para toda rotación a menos de pi de
HOME, que cubre de sobra el volumen de trabajo.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

HEMISPHERE_MODES = ("qw_positive", "home")


def normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza cuaterniones ``[..., 4]``; los de norma ~0 se devuelven sin cambios (el llamador decide)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > eps, q / np.maximum(n, eps), q)


def canonicalize(q: np.ndarray, mode: str = "qw_positive", q_ref: Optional[Sequence[float]] = None) -> np.ndarray:
    """Elige el signo de cada cuaternión ``[..., 4]`` según ``mode``.

    * ``"qw_positive"``: ``qw >= 0`` (convención histórica del repo; discontinua cerca de ``qw = 0``).
    * ``"home"``: ``dot(q, q_ref) >= 0`` (continua alrededor de ``q_ref``, p. ej. la orientación de HOME).
    """
    q = np.asarray(q, dtype=np.float64)
    if mode == "qw_positive":
        sign = np.where(q[..., 3:4] < 0.0, -1.0, 1.0)
    elif mode == "home":
        if q_ref is None:
            raise ValueError("canonicalize(mode='home') requiere q_ref")
        ref = normalize(np.asarray(q_ref, dtype=np.float64))
        sign = np.where(np.sum(q * ref, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    else:
        raise ValueError(f"Modo de hemisferio desconocido: {mode!r}. Opciones: {HEMISPHERE_MODES}")
    return q * sign


def weighted_average(quats: np.ndarray, weights: np.ndarray, q_ref: Sequence[float],
                     min_norm: float = 0.5) -> Optional[np.ndarray]:
    """Promedio ponderado de cuaterniones ``[N, 4]`` alineando antes sus signos con ``q_ref``.

    Devuelve ``None`` si la norma del promedio cae bajo ``min_norm`` (orientaciones contradictorias): el llamador debe
    conservar la última orientación válida en vez de normalizar un vector casi nulo.
    """
    q = canonicalize(np.asarray(quats, dtype=np.float64), "home", q_ref)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    avg = (q * w).sum(axis=0) / max(float(w.sum()), 1e-12)
    n = float(np.linalg.norm(avg))
    if n < min_norm:
        return None
    return avg / n


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY sobre ejes fijos (``R = Rz·Ry·Rx``, convención ROS/xArm) a ``[qx, qy, qz, qw]`` (sin canonizar)."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


# ----------------------------------------------------------------------------------------------
# Poses relativas (representación de acción "relative_ee", estilo UMI)
# ----------------------------------------------------------------------------------------------
def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto de Hamilton ``a ⊗ b`` para cuaterniones ``[..., 4]`` (``[qx, qy, qz, qw]``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rota vectores ``v [..., 3]`` con cuaterniones unitarios ``q [..., 4]`` (``q v q*``)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def relative_pose(p_ref: np.ndarray, q_ref: np.ndarray, p: np.ndarray, q: np.ndarray):
    """Pose ``(p, q)`` expresada en el marco de ``(p_ref, q_ref)``: ``T_ref^-1 · T``.

    Devuelve ``(dp [..., 3], dq [..., 4])`` con ``dq`` en el hemisferio ``qw >= 0`` (continuo: las rotaciones
    relativas dentro de un horizonte de acción son pequeñas, lejos de pi).
    """
    qc = quat_conjugate(normalize(q_ref))
    dp = quat_rotate(qc, np.asarray(p, dtype=np.float64) - np.asarray(p_ref, dtype=np.float64))
    dq = canonicalize(normalize(quat_multiply(qc, normalize(q))), "qw_positive")
    return dp, dq


def compose_pose(p_ref: np.ndarray, q_ref: np.ndarray, dp: np.ndarray, dq: np.ndarray):
    """Inversa de :func:`relative_pose`: ``T_ref · T_rel`` -> ``(p, q)`` (``q`` sin canonizar)."""
    qr = normalize(q_ref)
    p = np.asarray(p_ref, dtype=np.float64) + quat_rotate(qr, dp)
    q = normalize(quat_multiply(qr, normalize(dq)))
    return p, q
