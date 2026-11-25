"""Pose REAL de la cámara al inicio de una demostración, con los ArUco de la mesa y el IMU (sin ROS).

Problema: el conversor supone que cada demo empieza exactamente en HOME3. Si la cámara empieza más alta, más baja o
inclinada, todo el episodio queda desplazado y rotado. En v7, el inicio estaba a 32° ± 6° de inclinación frente a los
24,6° de HOME3, y 5° sobre una aproximación de 0,6 m son ~5 cm al final.

Solución:
* **ArUco:** las esquinas de los marcadores de la mesa tienen posición conocida en ``link_base``, calibrada una vez
  con el robot (``tools/xarm5/vio_gt/calibrate_table_markers.py``). El PnP con todas las esquinas visibles da la pose
  de la cámara al inicio con los 6 grados de libertad.
* **IMU:** con la cámara quieta, el acelerómetro da la gravedad, es decir, la inclinación absoluta (no el rumbo ni la
  posición). Sirve para verificar la inclinación del PnP y como respaldo si los ArUco no se ven.

Convenciones: ``T_a_b`` = pose de ``b`` en ``a`` (``p_a = T_a_b · p_b``). Cámara = marco óptico de color (OpenCV).
Las esquinas de un ArUco siguen el orden de OpenCV (sup-izq, sup-der, inf-der, inf-izq).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Corners = Dict[int, np.ndarray]          # id -> (4, 2) píxeles


def inv_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def triangulate_point(Ps: Sequence[np.ndarray], uvs: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Triangulación lineal (DLT) de un punto visto por varias cámaras con ``P = K [R | t]`` (mundo -> imagen).
    Devuelve el punto y el error de reproyección RMS (px)."""
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1, :3] / vt[-1, 3]
    err = []
    for P, uv in zip(Ps, uvs):
        x = P @ np.r_[X, 1.0]
        err.append(np.linalg.norm(x[:2] / x[2] - uv))
    return X, float(np.sqrt(np.mean(np.square(err))))


@dataclass
class TableMarkers:
    """Esquinas 3D de los ArUco de la mesa en ``link_base`` (calibradas con el robot)."""

    dictionary: str
    corners_base: Dict[int, np.ndarray]                 # id -> (4, 3) m
    meta: Dict = field(default_factory=dict)

    def side_lengths_m(self) -> Dict[int, float]:
        return {i: float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))
                for i, c in self.corners_base.items()}

    def to_dict(self) -> Dict:
        return {"table_markers": {"dictionary": self.dictionary,
                                  "corners_base_m": {int(i): np.round(c, 6).tolist()
                                                     for i, c in sorted(self.corners_base.items())},
                                  "side_length_m": {int(i): round(v, 5) for i, v in self.side_lengths_m().items()},
                                  **self.meta}}

    @classmethod
    def from_dict(cls, d: Dict) -> "TableMarkers":
        v = d["table_markers"]
        meta = {k: x for k, x in v.items() if k not in ("dictionary", "corners_base_m", "side_length_m")}
        return cls(str(v["dictionary"]), {int(i): np.asarray(c, float) for i, c in v["corners_base_m"].items()}, meta)

    @classmethod
    def load(cls, path) -> "TableMarkers":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def camera_pose_from_markers(live: Corners, table: TableMarkers, K: np.ndarray, D: Optional[np.ndarray] = None,
                             min_markers: int = 1) -> Optional[Dict]:
    """PnP con todas las esquinas de los marcadores visibles y calibrados. Devuelve ``T_base_cam``, el número de
    marcadores y la reproyección RMS (px), o ``None`` si no hay suficientes."""
    import cv2
    ids = sorted(i for i in live if i in table.corners_base)
    if len(ids) < min_markers:
        return None
    obj = np.concatenate([table.corners_base[i] for i in ids]).astype(np.float64)
    img = np.concatenate([np.asarray(live[i], float).reshape(4, 2) for i in ids]).astype(np.float64)
    D = np.zeros(5) if D is None else np.asarray(D, float)
    # IPPE para un plano chico; con varios marcadores, SQPnP (solución global) y después refinamiento LM. El PnP
    # iterativo sin estimación inicial cae en la solución "volteada" con esquinas imprecisas (imágenes de 256 px).
    flag = cv2.SOLVEPNP_IPPE if len(ids) == 1 else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
    if not ok:
        return None
    if len(ids) > 1:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3], T_cam_base[:3, 3] = R, tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return {"T_base_cam": inv_T(T_cam_base), "markers": ids, "reprojection_px": rms}


def depth_at(depth_m: np.ndarray, uv: np.ndarray, half: int = 3) -> float:
    """Mediana de la profundidad válida (> 0) en una ventana de (2·half+1)² píxeles alrededor de ``uv``."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth_m.shape
    win = depth_m[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
    good = win[win > 0]
    return float(np.median(good)) if good.size >= 3 else float("nan")


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transformación rígida ``T`` (4x4) que minimiza ||T·src - dst|| (puntos Nx3)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def camera_pose_from_markers_rgbd(live: Corners, depth_m: np.ndarray, table: TableMarkers, K: np.ndarray,
                                  depth_scale_correction: float = 1.0, min_points: int = 4) -> Optional[Dict]:
    """Pose de la cámara por alineación 3D-3D: las esquinas se llevan a 3D con la profundidad del sensor (corregida
    por ``depth_scale_correction``) y se alinean con sus posiciones en ``link_base``. Fija mejor la DISTANCIA que el
    PnP con marcadores chicos. ``depth_m`` está alineada a la imagen de color, en metros."""
    src, dst = [], []
    Kinv = np.linalg.inv(K)
    for i in sorted(i for i in live if i in table.corners_base):
        for c in range(4):
            z = depth_at(depth_m, live[i][c]) * depth_scale_correction
            if np.isfinite(z) and z > 0:
                src.append(z * (Kinv @ np.r_[live[i][c], 1.0]))
                dst.append(table.corners_base[i][c])
    if len(src) < min_points:
        return None
    src, dst = np.asarray(src), np.asarray(dst)
    T = kabsch(src, dst)
    res = np.linalg.norm((T[:3, :3] @ src.T).T + T[:3, 3] - dst, axis=1)
    return {"T_base_cam": T, "points": int(len(src)), "residual_mm": float(np.sqrt(np.mean(res ** 2)) * 1000)}


def up_in_camera_from_accel(accel: np.ndarray, T_cam_imu: np.ndarray) -> np.ndarray:
    """Dirección "arriba" (unitaria) en el marco de la cámara desde muestras del acelerómetro con la cámara quieta.
    Quieto, el acelerómetro mide la fuerza específica, que apunta hacia ARRIBA."""
    a = np.asarray(accel, float).reshape(-1, 3).mean(axis=0)
    up = T_cam_imu[:3, :3] @ a
    return up / np.linalg.norm(up)


def tilt_from_up(T_base_cam: np.ndarray, up_cam: np.ndarray, up_base=(0.0, 0.0, 1.0)) -> float:
    """Ángulo (°) entre el "arriba" medido por el IMU y el que predice ``T_base_cam`` (base con z vertical)."""
    pred = T_base_cam[:3, :3].T @ np.asarray(up_base, float)
    c = float(np.clip(np.dot(pred / np.linalg.norm(pred), up_cam), -1.0, 1.0))
    return math.degrees(math.acos(c))


def correct_tilt_with_up(T_base_cam: np.ndarray, up_cam: np.ndarray, up_base=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rota la orientación de ``T_base_cam`` lo mínimo para que su "arriba" coincida con el del IMU. El rumbo y la
    posición no cambian (el IMU no los observa)."""
    pred = T_base_cam[:3, :3].T @ np.asarray(up_base, float)
    pred /= np.linalg.norm(pred)
    v = np.cross(pred, up_cam)
    s, c = np.linalg.norm(v), float(np.dot(pred, up_cam))
    if s < 1e-12:
        return T_base_cam.copy()
    k = v / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R_fix = np.eye(3) + s * Kx + (1 - c) * Kx @ Kx          # en el marco de la cámara: lleva pred -> up_cam
    out = T_base_cam.copy()
    out[:3, :3] = T_base_cam[:3, :3] @ R_fix.T
    return out


def scale_intrinsics(K: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """K para una imagen redimensionada de ``src_size`` (ancho, alto) a ``dst_size`` SIN recorte (como el recolector
    v1/v2: 640x480 -> 256x256, escala distinta en cada eje)."""
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    K2 = np.asarray(K, float).copy()
    K2[0, 0] *= sx
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5
    K2[1, 1] *= sy
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5
    return K2


def estimate_start_pose(grays: Sequence[np.ndarray], table: TableMarkers, K: np.ndarray, D=None,
                        accel: Optional[np.ndarray] = None, T_cam_imu: Optional[np.ndarray] = None,
                        g_base: Optional[Sequence[float]] = None, max_tilt_disagreement_deg: float = 3.0,
                        detect=None) -> Dict:
    """Pose de la cámara al inicio de un episodio con la cámara QUIETA: esquinas (mediana de ``grays``) + PnP, y la
    inclinación corregida con el acelerómetro si se da (``accel`` Nx3 en el marco del IMU, ``T_cam_imu`` y la vertical
    real ``g_base``). Devuelve ``T_base_cam`` (marco óptico) y su diagnóstico; ``T_base_cam`` es ``None`` si no se ve
    ningún marcador calibrado.

    Control de calidad (``flags``): un solo marcador (posición menos precisa, hasta ~3 cm en vistas oblicuas), IMU y
    PnP en desacuerdo de inclinación > ``max_tilt_disagreement_deg`` (cámara moviéndose o marcador mal detectado)."""
    if detect is None:
        from visuomotor_data_collection.vo_start_guide import detect_markers as detect
    obs: Dict[int, list] = {}
    for g in grays:
        for i, c in detect(g, table.dictionary).items():
            if i in table.corners_base:
                obs.setdefault(i, []).append(c)
    need = max(1, len(grays) // 2)
    corners = {i: np.median(np.stack(v), axis=0) for i, v in obs.items() if len(v) >= need}
    out: Dict = {"T_base_cam": None, "markers": sorted(corners), "flags": [], "imu_used": False}
    est = camera_pose_from_markers(corners, table, K, D)
    if est is None:
        out["flags"].append("sin marcadores de la mesa")
        return out
    T = est["T_base_cam"]
    out.update(reprojection_px=est["reprojection_px"], T_base_cam_pnp=T)
    if len(est["markers"]) < 2:
        out["flags"].append("un solo marcador")
    if accel is not None and T_cam_imu is not None and len(accel) >= 10:
        up = up_in_camera_from_accel(accel, T_cam_imu)
        gb = np.asarray(g_base if g_base is not None else (0.0, 0.0, 1.0), float)
        dis = tilt_from_up(T, up, gb)
        out["tilt_pnp_vs_imu_deg"] = dis
        if dis > max_tilt_disagreement_deg:
            out["flags"].append(f"PnP e IMU discrepan {dis:.1f}° en la inclinación")
        else:
            T = correct_tilt_with_up(T, up, gb)
            out["imu_used"] = True
    out["T_base_cam"] = T
    return out


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(traslación mm, rotación °) entre dos poses."""
    dR = T_a[:3, :3].T @ T_b[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0), ang
