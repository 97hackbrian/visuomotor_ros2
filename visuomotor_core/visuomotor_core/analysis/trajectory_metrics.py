"""Exactitud de una trayectoria estimada (odometría de la D435i) frente a una referencia (xArm5 o tablero ChArUco).

Solo NumPy. Convenciones: posiciones ``[N, 3]`` en metros, cuaterniones ``[N, 4]`` como ``[qx, qy, qz, qw]`` y tiempos
en segundos.

Métricas
--------
* **ATE anclado al primer frame**: la estimación se lleva a la referencia con la transformación rígida que hace
  coincidir su PRIMERA pose. Es la métrica operativa: el pipeline de demostraciones usa la pose relativa al inicio del
  episodio, así que todo error acumulado desde t = 0 llega al dataset.
* **ATE SE(3) / Sim(3)** (Umeyama, 1991): mejor alineación rígida o con escala de toda la trayectoria. La escala de
  Sim(3) da el error de escala; la diferencia con el ATE anclado muestra cuánto pesa el error inicial.
* **RPE** a horizontes de tiempo: error de la pose relativa entre ``t`` y ``t + Δ`` (deriva local), en traslación y
  rotación.
* **Desfase temporal**: correlación cruzada de los perfiles de velocidad (latencia de la estimación).

Nunca se resume una campaña con el mejor intento: ``summarize_runs`` da mediana, p90, máximo y n.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# Rotaciones (vectorizadas)
# ---------------------------------------------------------------------------------------------------------------------
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``[N, 4]`` o ``[4]`` ``[qx, qy, qz, qw]`` -> matrices ``[N, 3, 3]`` o ``[3, 3]`` (se normaliza)."""
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r[0] if single else r


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Matrices ``[N, 3, 3]`` o ``[3, 3]`` -> cuaterniones ``[qx, qy, qz, qw]`` con ``qw >= 0``."""
    r = np.asarray(r, dtype=np.float64)
    single = r.ndim == 2
    r = r.reshape(-1, 3, 3)
    out = np.empty((len(r), 4))
    for i, m in enumerate(r):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            out[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            out[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            out[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            out[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    out[out[:, 3] < 0] *= -1
    return out[0] if single else out


def rotation_angle_deg(r: np.ndarray) -> np.ndarray:
    """Ángulo (grados) de rotaciones ``[N, 3, 3]`` o ``[3, 3]``."""
    r = np.asarray(r, dtype=np.float64)
    tr = np.trace(r, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Interpolación esférica fila a fila entre ``q0`` y ``q1`` (``[N, 4]``) con fracciones ``u`` (``[N]``)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    u = np.asarray(u, dtype=np.float64)[:, None]
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1[dot[:, 0] < 0] *= -1
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta[:, 0] < 1e-6
    s = np.sin(theta)
    s[small] = 1.0
    a = np.where(small[:, None], 1.0 - u, np.sin((1.0 - u) * theta) / s)
    b = np.where(small[:, None], u, np.sin(u * theta) / s)
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------------------------------
# Asociación temporal e interpolación
# ---------------------------------------------------------------------------------------------------------------------
def interpolate_poses(t_query: np.ndarray, t: np.ndarray, p: np.ndarray, q: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poses en ``t_query`` (posición lineal, orientación SLERP). Devuelve ``(p, q, valid)``; ``valid`` es False fuera
    del rango de ``t`` (no se extrapola)."""
    t = np.asarray(t, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("los tiempos de la referencia deben ser estrictamente crecientes")
    valid = (tq >= t[0]) & (tq <= t[-1])
    idx = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2)
    u = np.clip((tq - t[idx]) / (t[idx + 1] - t[idx]), 0.0, 1.0)
    p = np.asarray(p, dtype=np.float64)
    pq = p[idx] + u[:, None] * (p[idx + 1] - p[idx])
    qq = slerp(np.asarray(q, dtype=np.float64)[idx], np.asarray(q, dtype=np.float64)[idx + 1], u)
    return pq, qq, valid


# ---------------------------------------------------------------------------------------------------------------------
# Alineaciones
# ---------------------------------------------------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` que minimiza ``sum ||dst_i - (s R src_i + t)||^2`` (Umeyama, 1991). ``s = 1`` sin escala."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        raise ValueError("Umeyama necesita al menos 3 puntos")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1.0
    r = u @ sgn @ vt
    s = float(np.trace(np.diag(d) @ sgn) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * r @ mu_s
    return s, r, t


def apply_transform(s: float, r: np.ndarray, t: np.ndarray, p: np.ndarray, rq: Optional[np.ndarray] = None):
    """Aplica ``x -> s R x + t`` a posiciones y ``R_i -> R R_i`` a orientaciones (matrices ``[N, 3, 3]``)."""
    p2 = s * (np.asarray(p, dtype=np.float64) @ r.T) + t
    return p2, (None if rq is None else np.einsum("ij,njk->nik", r, rq))


def anchor_to_first(p_est: np.ndarray, r_est: np.ndarray, p_ref: np.ndarray, r_ref: np.ndarray):
    """Transformación rígida que hace coincidir la PRIMERA pose estimada con la primera de referencia."""
    r = r_ref[0] @ r_est[0].T
    t = p_ref[0] - r @ p_est[0]
    return 1.0, r, t


# ---------------------------------------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------------------------------------
def _stats(x: np.ndarray, scale: float = 1.0) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64) * scale
    if len(x) == 0:
        return {"rmse": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {"rmse": float(np.sqrt(np.mean(x ** 2))), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def relative_pose_errors(p_est: np.ndarray, r_est: np.ndarray, p_ref: np.ndarray, r_ref: np.ndarray, delta: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """RPE entre los frames ``i`` e ``i + delta``: errores de traslación (m) y rotación (grados)."""
    n = len(p_est)
    if delta < 1 or delta >= n:
        return np.zeros(0), np.zeros(0)
    i = np.arange(n - delta)
    j = i + delta
    # Pose relativa en el marco del frame i: R_i^T (p_j - p_i), R_i^T R_j
    dp_ref = np.einsum("nji,nj->ni", r_ref[i], p_ref[j] - p_ref[i])
    dp_est = np.einsum("nji,nj->ni", r_est[i], p_est[j] - p_est[i])
    dr_ref = np.einsum("nji,njk->nik", r_ref[i], r_ref[j])
    dr_est = np.einsum("nji,njk->nik", r_est[i], r_est[j])
    # Error: (ΔT_ref)^-1 ΔT_est
    e_r = np.einsum("nji,njk->nik", dr_ref, dr_est)
    e_t = np.einsum("nji,nj->ni", dr_ref, dp_est - dp_ref)
    return np.linalg.norm(e_t, axis=1), rotation_angle_deg(e_r)


def estimate_time_offset(t_est: np.ndarray, p_est: np.ndarray, t_ref: np.ndarray, p_ref: np.ndarray,
                         max_offset_s: float = 0.3, grid_hz: float = 200.0) -> float:
    """Desfase ``d`` (s) tal que la estimación en ``t`` corresponde a la referencia en ``t - d`` (``d > 0``: la
    estimación llega tarde), por correlación cruzada de los perfiles de velocidad."""
    t0, t1 = max(t_est[0], t_ref[0]), min(t_est[-1], t_ref[-1])
    if t1 - t0 < 4 * max_offset_s:
        raise ValueError("el solape temporal es demasiado corto para estimar el desfase")
    g = np.arange(t0, t1, 1.0 / grid_hz)

    def velocity(t, p):
        pi = np.stack([np.interp(g, t, p[:, k]) for k in range(3)], axis=1)
        v = np.gradient(pi, axis=0) * grid_hz
        return v - v.mean(0)

    ve, vr = velocity(np.asarray(t_est), np.asarray(p_est)), velocity(np.asarray(t_ref), np.asarray(p_ref))
    k_max = int(round(max_offset_s * grid_hz))
    ks = np.arange(-k_max, k_max + 1)
    corr = np.empty(len(ks))
    for n, k in enumerate(ks):
        a = ve[max(0, k):len(ve) + min(0, k)].ravel()
        b = vr[max(0, -k):len(vr) + min(0, -k)].ravel()
        corr[n] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    i = int(np.argmax(corr))
    frac = 0.0
    if 0 < i < len(ks) - 1:                       # refinamiento parabólico del máximo
        den = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if abs(den) > 1e-12:
            frac = 0.5 * (corr[i - 1] - corr[i + 1]) / den
    return float((ks[i] + frac) / grid_hz)


def evaluate_trajectory(t_est: np.ndarray, p_est: np.ndarray, q_est: np.ndarray,
                        t_ref: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray,
                        rpe_horizons_s: Sequence[float] = (0.25, 0.5, 1.0),
                        lost: Optional[np.ndarray] = None, time_offset_s: float = 0.0) -> Dict:
    """Compara una trayectoria estimada con la referencia interpolada en los instantes de la estimación.

    ``lost`` marca frames sin estimación válida (pose nula, rastreo perdido): cuentan en ``lost_pct`` y se excluyen
    de las métricas. ``time_offset_s`` se resta a los tiempos estimados antes de asociar (latencia conocida).
    """
    t_est = np.asarray(t_est, dtype=np.float64) - time_offset_s
    p_est = np.asarray(p_est, dtype=np.float64)
    q_est = np.asarray(q_est, dtype=np.float64)
    lost = np.zeros(len(t_est), bool) if lost is None else np.asarray(lost, bool)
    pr, qr, inside = interpolate_poses(t_est, t_ref, p_ref, q_ref)
    ok = inside & ~lost
    if ok.sum() < 3:
        raise ValueError("menos de 3 frames válidos dentro del rango de la referencia")
    pe, re_ = p_est[ok], quat_to_matrix(q_est[ok])
    pr, rr = pr[ok], quat_to_matrix(qr[ok])
    te = t_est[ok]

    out: Dict = {"n_frames": int(len(t_est)), "n_valid": int(ok.sum()),
                 "lost_pct": float(100.0 * lost.mean()), "duration_s": float(te[-1] - te[0]),
                 "path_length_m": float(np.linalg.norm(np.diff(pr, axis=0), axis=1).sum())}
    # ATE anclado al primer frame (métrica operativa)
    s, r, t = anchor_to_first(pe, re_, pr, rr)
    pa, ra = apply_transform(s, r, t, pe, re_)
    out["ate_anchored_mm"] = _stats(np.linalg.norm(pa - pr, axis=1), 1000.0)
    out["rot_anchored_deg"] = _stats(rotation_angle_deg(np.einsum("nji,njk->nik", rr, ra)))
    out["final_error_anchored_mm"] = float(np.linalg.norm(pa[-1] - pr[-1]) * 1000.0)
    # ATE con la mejor alineación rígida y con escala
    s, r, t = umeyama(pe, pr, with_scale=False)
    out["ate_se3_mm"] = _stats(np.linalg.norm(apply_transform(s, r, t, pe)[0] - pr, axis=1), 1000.0)
    s, r, t = umeyama(pe, pr, with_scale=True)
    out["ate_sim3_mm"] = _stats(np.linalg.norm(apply_transform(s, r, t, pe)[0] - pr, axis=1), 1000.0)
    out["scale"] = float(1.0 / s)                      # longitudes estimadas / longitudes de referencia
    out["scale_error_pct"] = float(100.0 * (1.0 / s - 1.0))
    # RPE a horizontes de tiempo (frames consecutivos válidos; se usa el período mediano)
    dt = float(np.median(np.diff(te)))
    out["rpe"] = {}
    for h in rpe_horizons_s:
        d = max(1, int(round(h / dt)))
        et, er = relative_pose_errors(pe, re_, pr, rr, d)
        out["rpe"][f"{h:g}s"] = {"trans_mm": _stats(et, 1000.0), "rot_deg": _stats(er)}
    return out


def summarize_runs(results: Iterable[Dict], keys: Sequence[str] = ("ate_anchored_mm.rmse", "ate_anchored_mm.max",
                                                                      "ate_sim3_mm.rmse", "scale_error_pct",
                                                                      "rot_anchored_deg.median", "lost_pct")
                   ) -> Dict[str, Dict[str, float]]:
    """Distribución por corrida de cada métrica (``"a.b"`` = clave anidada): mediana, p90, máximo, mínimo y n."""
    results = list(results)
    out = {}
    for key in keys:
        vals: List[float] = []
        for r in results:
            v = r
            for part in key.split("."):
                v = v[part]
            vals.append(float(v))
        a = np.asarray(vals)
        out[key] = {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                    "max": float(a.max()), "min": float(a.min())}
    return out
