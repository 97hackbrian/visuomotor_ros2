"""Lógica pura (sin ROS) del recolector de demostraciones a mano con la D435i (``vo_collector_node``, v2).

Corrige los defectos medidos en la grabación v7 (``context.md`` lección 43; plan
``docs/plans/plan_d435i_herramienta_demostraciones.md``, Fase 4):

* **Poses nulas.** Al perder el rastreo, RTAB-Map publica una odometría nula: cuaternión cero y covarianza 9999. El
  recolector v1 la guardaba como orientación identidad y posición 0. Aquí se marca como PERDIDA y nunca se convierte
  en una pose.
* **Sin sellos de tiempo.** El v1 tomaba "la última imagen" y "la última pose" en un timer. Ahora cada frame del
  dataset tiene su instante: la pose se interpola (lineal + SLERP) en una rejilla regular de ``1/rate`` sobre los
  sellos de la odometría, y la imagen es la de sello más cercano. RTAB-Map publica la odometría con el sello de la
  imagen que la produjo.
* **Signo del cuaternión.** Se mantiene continuo con el frame anterior, en vez de forzar ``qw >= 0``, que es
  discontinuo con la cámara apuntando hacia abajo.
* **Etiqueta done.** Va de 1 desde el inicio del HOLD, y se verifica que ese tramo esté quieto y dure lo mínimo. Ya
  no se fuerza en frames en movimiento.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

RTABMAP_BAD_COVARIANCE = 9999.0


# ---------------------------------------------------------------------------------------------------------------------
# Odometría
# ---------------------------------------------------------------------------------------------------------------------
def is_null_odometry(quat_xyzw: Sequence[float], covariance_diag: Optional[Sequence[float]] = None,
                     bad_covariance: float = RTABMAP_BAD_COVARIANCE) -> bool:
    """True si el mensaje es la odometría nula de RTAB-Map: cuaternión de norma ~0 o covarianza ≥ ``bad_covariance``
    (``publish_null_when_lost``)."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
        return True
    if covariance_diag is not None:
        c = np.asarray(covariance_diag, dtype=np.float64)
        if np.any(c >= bad_covariance):
            return True
    return False


def continuous_quat(q: Sequence[float], q_prev: Optional[Sequence[float]]) -> np.ndarray:
    """Cuaternión unitario con el signo más cercano a ``q_prev`` (continuidad); sin referencia, ``qw >= 0``."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q_prev is None:
        return q if q[3] >= 0 else -q
    return q if float(np.dot(q, q_prev)) >= 0 else -q


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


@dataclass
class PoseSample:
    stamp: float
    p: np.ndarray
    q: np.ndarray
    lost: bool = False
    inliers: int = -1


class PoseBuffer:
    """Poses de odometría ordenadas por sello, con interpolación y marca de pérdida."""

    def __init__(self, maxlen: int = 600):
        self._s: Deque[PoseSample] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._s)

    def add(self, sample: PoseSample) -> None:
        if self._s and sample.stamp <= self._s[-1].stamp:
            if sample.stamp == self._s[-1].stamp:          # duplicado (p. ej. el mismo frame por otra vía)
                return
            items = list(self._s) + [sample]                  # llegada desordenada: reordenar
            items.sort(key=lambda s: s.stamp)
            self._s = deque(items, maxlen=self._s.maxlen)
            return
        self._s.append(sample)

    def set_inliers(self, stamp: float, inliers: int, lost: bool, tol: float = 1e-4) -> None:
        """Asocia la información de ``odom_info`` (mismo sello que la odometría)."""
        for s in reversed(self._s):
            if abs(s.stamp - stamp) <= tol:
                s.inliers = int(inliers)
                s.lost = s.lost or bool(lost)
                return
            if s.stamp < stamp - tol:
                return

    @property
    def latest_stamp(self) -> Optional[float]:
        return self._s[-1].stamp if self._s else None

    def latest(self) -> Optional[PoseSample]:
        return self._s[-1] if self._s else None

    def window(self, t_from: float) -> List[PoseSample]:
        """Muestras con sello >= ``t_from``."""
        return [s for s in self._s if s.stamp >= t_from]

    def valid_after(self, t: float) -> int:
        """Número de poses válidas con sello > ``t``."""
        return sum(1 for s in self._s if s.stamp > t and not s.lost)

    def interpolate(self, t: float, max_gap_s: float) -> Optional[Tuple[np.ndarray, np.ndarray, bool, float, int]]:
        """Pose en ``t``: ``(p, q, lost, sello_más_cercano, inliers_mín)``. ``lost`` es True si un vecino está perdido
        o el hueco entre vecinos supera ``max_gap_s``. Devuelve None si ``t`` está fuera del rango del búfer."""
        if len(self._s) < 2 or t < self._s[0].stamp or t > self._s[-1].stamp:
            if len(self._s) >= 1 and abs(t - self._s[-1].stamp) < 1e-9:
                s = self._s[-1]
                return s.p.copy(), s.q.copy(), s.lost, s.stamp, s.inliers
            return None
        stamps = [s.stamp for s in self._s]
        j = bisect.bisect_left(stamps, t)
        if j < len(stamps) and stamps[j] == t:
            s = self._s[j]
            return s.p.copy(), s.q.copy(), s.lost, s.stamp, s.inliers
        a, b = self._s[j - 1], self._s[j]
        u = (t - a.stamp) / (b.stamp - a.stamp)
        lost = a.lost or b.lost or (b.stamp - a.stamp) > max_gap_s
        p = a.p + u * (b.p - a.p)
        q = slerp(a.q, b.q, u)
        near = a.stamp if u < 0.5 else b.stamp
        inl = min(x for x in (a.inliers, b.inliers)) if a.inliers >= 0 and b.inliers >= 0 else max(a.inliers, b.inliers)
        return p, q, lost, near, inl


class StampedBuffer:
    """Mensajes (p. ej. imágenes ya redimensionadas) por sello, con búsqueda del más cercano."""

    def __init__(self, maxlen: int = 90):
        self._t: Deque[float] = deque(maxlen=maxlen)
        self._v: Deque[object] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._t)

    def add(self, stamp: float, value) -> None:
        """Inserta en orden. Con un ejecutor multihilo, dos callbacks de imágenes consecutivas pueden terminar en
        orden inverso: descartar la que llega tarde perdía imágenes (hallado con tests/test_vo_collector_node_sim.py)."""
        if not self._t or stamp > self._t[-1]:
            self._t.append(stamp)
            self._v.append(value)
            return
        ts = list(self._t)
        j = bisect.bisect_left(ts, stamp)
        if j < len(ts) and ts[j] == stamp:
            return                                             # duplicado
        vs = list(self._v)
        ts.insert(j, stamp)
        vs.insert(j, value)
        self._t = deque(ts, maxlen=self._t.maxlen)
        self._v = deque(vs, maxlen=self._v.maxlen)

    def nearest(self, t: float, max_dt: float):
        """``(sello, valor)`` más cercano a ``t`` si está a ≤ ``max_dt``; si no, None."""
        if not self._t:
            return None
        ts = list(self._t)
        j = bisect.bisect_left(ts, t)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(ts) and (best is None or abs(ts[k] - t) < abs(ts[best] - t)):
                best = k
        if best is None or abs(ts[best] - t) > max_dt:
            return None
        return ts[best], self._v[best]


# ---------------------------------------------------------------------------------------------------------------------
# Episodio
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    timestamp: float
    image_stamp: float
    pose_stamp: float
    p: np.ndarray                 # posición absoluta de odom (m)
    q: np.ndarray                 # cuaternión continuo
    image: np.ndarray
    lost: bool
    inliers: int
    gripper: float
    hold: bool


@dataclass
class EpisodeSampler:
    """Genera los frames del episodio en una rejilla regular de tiempo a partir de los búferes."""
    rate_hz: float
    max_image_dt_s: float
    max_pose_gap_s: float
    t_next: Optional[float] = None
    frames: List[Frame] = field(default_factory=list)
    _q_prev: Optional[np.ndarray] = None

    def start(self, t0: float) -> None:
        self.t_next = t0
        self.frames = []
        self._q_prev = None

    def step(self, poses: PoseBuffer, images: StampedBuffer, gripper: float, hold: bool,
             image_wait_s: float = 0.0) -> int:
        """Añade todos los frames cuya pose ya está disponible (``t_next <= último sello de odometría``) y cuya imagen
        más cercana ya llegó (se espera ``image_wait_s`` tras ``t_next`` antes de aceptar una imagen anterior).
        Devuelve cuántos frames añadió."""
        if self.t_next is None or poses.latest_stamp is None:
            return 0
        added = 0
        while self.t_next <= poses.latest_stamp:
            r = poses.interpolate(self.t_next, self.max_pose_gap_s)
            if r is None:                                     # t_next anterior al búfer: saltar
                self.t_next += 1.0 / self.rate_hz
                continue
            img = images.nearest(self.t_next, self.max_image_dt_s)
            if img is None:
                newest = images.nearest(self.t_next + 10.0, 1e9)
                if newest is not None and newest[0] < self.t_next + image_wait_s:
                    break                                      # la imagen de este instante aún puede llegar
                lost_img = True
                img = images.nearest(self.t_next, 1e9)         # se guarda la más cercana, marcada como inválida
                if img is None:
                    break
            else:
                lost_img = False
            p, q, lost, pose_stamp, inl = r
            q = continuous_quat(q, self._q_prev)
            self._q_prev = q
            self.frames.append(Frame(timestamp=self.t_next, image_stamp=img[0], pose_stamp=pose_stamp, p=p, q=q,
                                     image=img[1], lost=bool(lost or lost_img), inliers=int(inl), gripper=gripper,
                                     hold=hold))
            self.t_next += 1.0 / self.rate_hz
            added += 1
        return added


def smoothed_speeds(p: np.ndarray, rate_hz: float, window_s: float) -> np.ndarray:
    """Velocidad (m/s) como desplazamiento en una ventana centrada de ``window_s`` (robusta al ruido por frame)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    k = max(1, int(round(window_s * rate_hz / 2.0)))
    v = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if b > a:
            v[i] = np.linalg.norm(p[b] - p[a]) / ((b - a) / rate_hz)
    return v


def detect_hold_start(p: np.ndarray, rate_hz: float, static_speed: float, window_s: float) -> Optional[int]:
    """Primer índice del tramo final quieto (todas las velocidades suavizadas ≤ ``static_speed`` hasta el final)."""
    v = smoothed_speeds(p, rate_hz, window_s)
    if len(v) == 0 or v[-1] > static_speed:
        return None
    i = len(v) - 1
    while i > 0 and v[i - 1] <= static_speed:
        i -= 1
    return i


def trailing_static_s(p: np.ndarray, rate_hz: float, static_speed: float, window_s: float) -> float:
    """Duración (s) del tramo final quieto; 0 si el último frame se mueve."""
    i = detect_hold_start(p, rate_hz, static_speed, window_s)
    return 0.0 if i is None else (len(p) - i) / rate_hz


def max_travel(p: np.ndarray) -> float:
    """Distancia máxima (m) alcanzada desde la posición inicial."""
    p = np.asarray(p, dtype=np.float64)
    return float(np.linalg.norm(p - p[0], axis=1).max()) if len(p) else 0.0


def camera_still(points, rate_hz: float, window_s: float, static_speed_m_s: float) -> bool:
    """¿La cámara está QUIETA en los últimos ``window_s``? (desplazamiento entre los extremos de la ventana dividido
    por su duración, bajo ``static_speed_m_s``; el mismo criterio de quietud que el guardado automático). Con menos
    frames que la ventana: no."""
    n = max(2, int(round(window_s * rate_hz)))
    if len(points) < n:
        return False
    p = np.asarray(points[-n:], dtype=np.float64)
    return float(np.linalg.norm(p[-1] - p[0])) / ((n - 1) / rate_hz) < static_speed_m_s


def imu_still_since(t: np.ndarray, gyro: np.ndarray, gyro_thr: float, smooth_s: float = 0.15) -> Optional[float]:
    """Inicio del tramo FINAL quieto según el giróscopo (|w| suavizado < ``gyro_thr``), o ``None`` si al final se mueve.

    No depende de la imagen: sirve cuando la odometría visual se pierde al final de la aproximación, cerca de la mesa,
    donde la pose queda congelada y parecería quieta aunque la mano se siga moviendo."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return None
    g = np.asarray(gyro, float)
    w = np.linalg.norm(g, axis=1) if g.ndim == 2 else np.abs(g)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    k = max(1, int(round(smooth_s / max(dt, 1e-4))))
    ws = np.convolve(w, np.ones(k) / k, mode="same")
    moving = np.nonzero(ws >= gyro_thr)[0]
    if len(moving) == 0:
        return float(t[0])
    if moving[-1] >= len(t) - 1:
        return None
    return float(t[moving[-1] + 1])


def imu_moving_fraction(t_frames, imu_t, gyro, gyro_thr: float, smooth_s: float = 0.15) -> Optional[float]:
    """Fracción de los instantes ``t_frames`` en que la cámara GIRA según el giróscopo (|w| suavizado > ``gyro_thr``).
    ``None`` si el IMU no cubre esos instantes.

    Control del tramo final independiente de la imagen (2026-11-25): cerca de la mesa, y con la vista final en
    diagonal, la odometría en vivo se pierde y sus poses saltan, así que una cámara quieta parecía moverse y el
    control rechazaba demos buenas."""
    t_frames = np.asarray(t_frames, float)
    t = np.asarray(imu_t, float)
    if len(t) < 5 or len(t_frames) == 0 or t[0] > t_frames[0] + 1e-6 or t[-1] < t_frames[-1] - 0.1:
        return None
    g = np.asarray(gyro, float)
    w = np.linalg.norm(g, axis=1) if g.ndim == 2 else np.abs(g)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(smooth_s / max(dt, 1e-4))))
    ws = np.convolve(w, np.ones(k) / k, mode="same")
    return float((np.interp(t_frames, t, ws) > gyro_thr).mean())


def auto_save_ready(p: np.ndarray, rate_hz: float, min_travel_m: float, min_hold_s: float, margin_s: float,
                    static_speed: float, window_s: float) -> bool:
    """Guardado automático: la cámara ya se alejó del inicio al menos ``min_travel_m`` (al empezar también está
    quieta) y lleva quieta al final ``min_hold_s + margin_s``."""
    if len(p) < 2 or max_travel(p) < min_travel_m:
        return False
    return trailing_static_s(p, rate_hz, static_speed, window_s) >= min_hold_s + margin_s


def check_done_segment(p: np.ndarray, hold: Sequence[bool], rate_hz: float, min_hold_s: float, static_speed: float,
                       window_s: float, max_moving_frac: float = 0.0) -> Dict:
    """Verifica el tramo done: existe, está al final, dura ≥ ``min_hold_s`` y está quieto. Devuelve las etiquetas
    done (1 desde el primer frame en HOLD) y la lista de problemas (vacía = correcto).

    Quieto: a lo más ``max_moving_frac`` de los frames del tramo superan ``static_speed``. Las velocidades se calculan
    sobre TODA la trayectoria, igual que en ``detect_hold_start``: recortando primero el tramo, la ventana quedaba
    de un solo lado en sus bordes, y frames que la detección daba por quietos salían "en movimiento" (2026-11-25: 1-3
    frames de unos 50 rechazaban demos a mano)."""
    hold = np.asarray(hold, dtype=bool)
    n = len(hold)
    done = np.zeros(n, dtype=np.float32)
    problems: List[str] = []
    if not hold.any():
        return {"done": done, "problems": ["no se marcó HOLD (llegada al destino)"], "hold_frames": 0,
                "moving_frames": 0}
    i0 = int(np.argmax(hold))
    if not hold[i0:].all():
        problems.append("el HOLD no es un tramo continuo hasta el final")
    done[i0:] = 1.0
    hold_frames = n - i0
    if hold_frames < int(round(min_hold_s * rate_hz)):
        problems.append(f"HOLD de {hold_frames / rate_hz:.2f} s < {min_hold_s:.2f} s: mantener la cámara quieta más tiempo")
    v = smoothed_speeds(np.asarray(p), rate_hz, window_s)[i0:]       # misma ventana que la detección
    # Los primeros medio-ventana frames del tramo promedian también el movimiento de ANTES de detenerse (HOLD marcado
    # con [H] o por el giróscopo justo al parar): no cuentan.
    k = max(1, int(round(window_s * rate_hz / 2.0)))
    core = v[k:] if len(v) > 2 * k else v
    moving = int((core > static_speed).sum())
    if moving > max_moving_frac * len(core):
        problems.append(f"{moving} frames del HOLD se mueven a más de {static_speed * 100:.1f} cm/s")
    return {"done": done, "problems": problems, "hold_frames": int(hold_frames), "moving_frames": moving}


def episode_quality(frames: Sequence[Frame], rate_hz: float) -> Dict:
    """Resumen de calidad del episodio (se guarda en los atributos)."""
    if not frames:
        return {}
    t = np.array([f.timestamp for f in frames])
    img_t = np.array([f.image_stamp for f in frames])
    inl = np.array([f.inliers for f in frames])
    inl = inl[inl >= 0]
    dt = np.diff(t)
    return {
        "frames": len(frames), "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "lost_frames": int(sum(f.lost for f in frames)),
        "duplicate_images": int((np.diff(img_t) == 0).sum()) if len(img_t) > 1 else 0,
        "max_image_offset_ms": float(np.abs(img_t - t).max() * 1000.0),
        "dt_mean_ms": float(dt.mean() * 1000.0) if len(dt) else 0.0,
        "dt_std_ms": float(dt.std() * 1000.0) if len(dt) else 0.0,
        "inliers_min": int(inl.min()) if len(inl) else -1,
        "inliers_median": float(np.median(inl)) if len(inl) else -1.0,
        "rate_hz": float(rate_hz),
    }
