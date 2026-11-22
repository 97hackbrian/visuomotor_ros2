"""Filtrado de trayectorias de pose ruidosas (odometría visual-inercial de demostraciones a mano).

Diagnóstico que lo motiva (``demonstrations_real_v7``, 2026-11-22): la PSD de la posición tiene la energía del
movimiento humano bajo ~0.8 Hz y un piso de ruido plano desde ~0.9 Hz; el residuo frente a una trayectoria suave es
~8 mm RMS (isótropo: ruido del VIO, no artefacto de muestreo) con saltos aislados de 3-9 cm por frame. Ese ruido
infla 4x la velocidad aparente (0.20 m/s frente a ~0.05 m/s reales) y se aprendería como objetivo de la política.

Procedimiento (todo de fase cero, sin retardo):
1. Filtro de Hampel (mediana móvil + MAD) que reemplaza picos aislados por la mediana local.
2. Butterworth paso bajo aplicado con ``filtfilt`` (ida y vuelta) con relleno constante en los bordes: el inicio
   (reposo en HOME) y el final (retención con ``done = 1``) son estáticos, así que el borde no se deforma.
La orientación se filtra en el espacio tangente (vector de rotación) relativo a la primera pose del episodio, que es
continuo mientras la rotación acumulada sea < pi.

SciPy se importa de forma diferida: este módulo no se usa en inferencia.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TrajectoryFilterConfig:
    enabled: bool = True
    hampel_half_window: int = 3          # ventana total 2k+1 muestras (7 = 0.35 s a 20 Hz)
    hampel_n_sigma: float = 3.0
    lowpass_cutoff_hz: float = 1.5
    lowpass_order: int = 4
    filter_orientation: bool = True

    @classmethod
    def from_dict(cls, d: Dict | None) -> "TrajectoryFilterConfig":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def to_dict(self) -> Dict:
        return asdict(self)


def hampel(x: np.ndarray, half_window: int = 3, n_sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Filtro de Hampel por columna sobre ``x [T, D]``. Devuelve ``(filtrado, máscara_de_atípicos [T, D])``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    k = int(half_window)
    if k < 1 or len(x) < 2 * k + 1:
        return x.copy(), np.zeros_like(x, dtype=bool)
    padded = np.pad(x, ((k, k), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1, axis=0)  # [T, D, W]
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    outlier = np.abs(x - med) > n_sigma * np.maximum(mad, 1e-9)
    return np.where(outlier, med, x), outlier


def lowpass_zero_phase(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Butterworth paso bajo de fase cero (``filtfilt``) con relleno constante en los bordes."""
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=np.float64)
    wn = float(cutoff_hz) / (0.5 * float(fs))
    if not 0.0 < wn < 1.0:
        raise ValueError(f"cutoff_hz={cutoff_hz} fuera de (0, fs/2={fs / 2})")
    b, a = butter(int(order), wn)
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    if padlen < 1:
        return x.copy()
    return filtfilt(b, a, x, axis=0, padtype="constant", padlen=padlen)


def unwrap_rotvec(rv: np.ndarray) -> np.ndarray:
    """Hace continuo un vector de rotación ``[T, 3]``: ``r`` y ``r - 2pi·r/|r|`` son la misma rotación; en cada frame se
    elige la representación más cercana a la anterior (evita el salto al cruzar pi, que el filtro suavizaría como un
    giro inexistente)."""
    out = np.array(rv, dtype=np.float64, copy=True)
    for t in range(1, len(out)):
        n = float(np.linalg.norm(out[t]))
        if n < 1e-9:
            continue
        axis = out[t] / n
        cands = [out[t] + 2.0 * np.pi * k * axis for k in (-2, -1, 0, 1, 2)]
        out[t] = min(cands, key=lambda c: float(np.linalg.norm(c - out[t - 1])))
    return out


def smooth_pose_trajectory(positions: np.ndarray, quats: np.ndarray, fs: float,
                           cfg: TrajectoryFilterConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Filtra una trayectoria de poses ``positions [T,3]`` + ``quats [T,4]`` (``[qx,qy,qz,qw]``).

    Devuelve ``(pos_filtrada, quat_filtrado, estadísticas)``. Las estadísticas cuantifican lo eliminado (RMS en mm,
    picos reemplazados) para auditar que el filtro no borra movimiento real.
    """
    from scipy.spatial.transform import Rotation

    p = np.asarray(positions, dtype=np.float64)
    q = np.asarray(quats, dtype=np.float64)
    stats: Dict = {"filter": cfg.to_dict()}
    if not cfg.enabled:
        return p.copy(), q.copy(), stats

    p_h, out_p = hampel(p, cfg.hampel_half_window, cfg.hampel_n_sigma)
    p_f = lowpass_zero_phase(p_h, fs, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
    stats["position_spikes_replaced"] = int(out_p.any(axis=1).sum())
    stats["position_removed_rms_mm"] = float(np.sqrt(((p - p_f) ** 2).sum(axis=1).mean()) * 1000.0)

    if cfg.filter_orientation:
        r0 = Rotation.from_quat(q[0])
        rel = unwrap_rotvec((r0.inv() * Rotation.from_quat(q)).as_rotvec())   # [T,3] continuo también al cruzar pi
        stats["orientation_max_rel_angle_rad"] = float(np.linalg.norm(rel, axis=1).max())
        rel_h, out_r = hampel(rel, cfg.hampel_half_window, cfg.hampel_n_sigma)
        rel_f = lowpass_zero_phase(rel_h, fs, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
        q_f = (r0 * Rotation.from_rotvec(rel_f)).as_quat()
        ang = (Rotation.from_quat(q).inv() * Rotation.from_quat(q_f)).magnitude()
        stats["orientation_spikes_replaced"] = int(out_r.any(axis=1).sum())
        stats["orientation_removed_rms_deg"] = float(np.degrees(np.sqrt((ang ** 2).mean())))
    else:
        q_f = q.copy()
    return p_f, q_f, stats
