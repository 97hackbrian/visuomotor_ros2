"""Guía de inicio: empezar cada demostración con la vista de HOME3, que es con la que arranca la política en el xArm5.

Referencia: las esquinas de los ArUco de la mesa (DICT_4X4_50) en las imágenes que tomó la cámara montada en el robot
en HOME3 (``tools/hardware/make_home3_view_reference.py``). En los 13 intentos del 2026-11-23 se repitieron con
< 1 px de diferencia. Las demos v7 empezaron con escala 0,88-1,26 y corrimientos de hasta ~80 px (640 px de ancho)
respecto a esa vista.

La comparación es en la imagen: no necesita el robot, ni el tamaño real de los marcadores, ni la calibración de la
cámara. Se ajusta una semejanza 2D (escala, giro y traslación) de la referencia a lo observado:
* el corrimiento indica hacia dónde mover o girar la cámara;
* la escala indica si acercarla o alejarla;
* el giro indica si rotarla sobre su eje;
* el residuo es la diferencia de perspectiva (inclinación) que la semejanza no explica.

Sin ROS: la usan ``vo_collector_node`` y sus pruebas (``tests/test_vo_start_guide.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Corners = Dict[int, np.ndarray]          # id -> esquinas [4, 2] en píxeles (orden de OpenCV)


@dataclass
class ViewReference:
    dictionary: str
    image_size: Tuple[int, int]            # (ancho, alto) de las imágenes de referencia
    markers: Corners                       # id -> [4, 2] en píxeles de image_size
    repeatability_px: float = 0.0          # dispersión de las esquinas entre las imágenes de referencia
    source: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ViewReference":
        v = d["home3_view_reference"] if "home3_view_reference" in d else d
        return cls(dictionary=str(v["dictionary"]), image_size=(int(v["image_size"][0]), int(v["image_size"][1])),
                   markers={int(k): np.asarray(c, dtype=np.float64).reshape(4, 2) for k, c in v["markers"].items()},
                   repeatability_px=float(v.get("repeatability_px", 0.0)), source=str(v.get("source", "")),
                   extra={k: v[k] for k in v if k not in ("dictionary", "image_size", "markers", "repeatability_px",
                                                          "source")})

    @classmethod
    def load(cls, path) -> "ViewReference":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {"home3_view_reference": {
            "dictionary": self.dictionary, "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "markers": {int(k): np.round(c, 2).tolist() for k, c in sorted(self.markers.items())},
            "repeatability_px": round(float(self.repeatability_px), 3), "source": self.source, **self.extra}}


# ---------------------------------------------------------------------------------------------------------------------
# Detección (OpenCV 4.5 a 4.10: API antigua y ArucoDetector)
# ---------------------------------------------------------------------------------------------------------------------
def detect_markers(gray: np.ndarray, dictionary: str = "DICT_4X4_50") -> Corners:
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    new_api = hasattr(cv2.aruco, "ArucoDetector")
    prm = cv2.aruco.DetectorParameters() if new_api else cv2.aruco.DetectorParameters_create()
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX      # esquinas subpíxel (sin esto, enteras)
    if new_api:
        corners, ids, _ = cv2.aruco.ArucoDetector(d, prm).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=prm)
    if ids is None:
        return {}
    return {int(i): np.asarray(c, dtype=np.float64).reshape(4, 2) for c, i in zip(corners, ids.ravel())}


# ---------------------------------------------------------------------------------------------------------------------
# Alineación
# ---------------------------------------------------------------------------------------------------------------------
def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """Semejanza 2D ``dst ≈ s R(θ) src + t`` (Umeyama). Devuelve ``(s, θ [rad], t, residuo RMS)``.

    En coordenadas de imagen (y hacia abajo), ``θ > 0`` se ve como un giro HORARIO."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    var = float((a ** 2).sum() / len(a))
    cov = b.T @ a / len(a)
    u, sig, vt = np.linalg.svd(cov)
    d = np.diag([1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0])
    r = u @ d @ vt
    s = float(np.trace(np.diag(sig) @ d) / max(var, 1e-12))
    t = mu_d - s * r @ mu_s
    res = float(np.sqrt(np.mean(np.sum((dst - (s * src @ r.T + t)) ** 2, axis=1))))
    return s, float(math.atan2(r[1, 0], r[0, 0])), t, res


def view_alignment(live: Corners, live_size: Tuple[int, int], ref: ViewReference, tol_px: float = 8.0,
                   min_markers: int = 2) -> Dict:
    """Compara los marcadores vistos ahora con la referencia de HOME3.

    Todo se expresa en píxeles de la referencia (p. ej. 640x480), aunque la imagen en vivo tenga otro tamaño.
    ``aligned`` exige al menos ``min_markers`` marcadores en común y un error medio de esquina ≤ ``tol_px``. Las
    pistas (``hints``) solo nombran las componentes que aportan más de ``tol_px / 2``."""
    sx, sy = ref.image_size[0] / float(live_size[0]), ref.image_size[1] / float(live_size[1])
    common = sorted(i for i in live if i in ref.markers)
    out: Dict = {"n_markers": len(common), "aligned": False, "error_px": None, "shift_px": None, "scale": None,
                 "roll_deg": None, "residual_px": None, "tol_px": float(tol_px), "hints": []}
    if not common:
        out["hints"] = ["no se ven los ArUco de la mesa: apuntar la cámara a la mesa como en HOME3"]
        return out
    src = np.concatenate([ref.markers[i] for i in common])
    dst = np.concatenate([live[i] * np.array([sx, sy]) for i in common])
    err = float(np.linalg.norm(dst - src, axis=1).mean())
    shift = dst.mean(0) - src.mean(0)
    radius = float(np.linalg.norm(src - src.mean(0), axis=1).mean())
    s, th, _, res = fit_similarity(src, dst)                    # >= 4 esquinas: un marcador ya basta para ajustarla
    out.update(error_px=round(err, 2), shift_px=[round(float(shift[0]), 2), round(float(shift[1]), 2)],
               scale=round(s, 4), roll_deg=round(math.degrees(th), 2), residual_px=round(res, 2),
               markers=common)
    half = tol_px / 2.0
    hints: List[str] = []
    if len(common) < min_markers:
        hints.append(f"solo se ven {len(common)} de {len(ref.markers)} marcadores: encuadrar ambos")
    if abs(s - 1.0) * radius > half:
        hints.append(f"{'ALEJAR' if s > 1 else 'ACERCAR'} la cámara (la escena se ve {abs(s - 1) * 100:.0f} % "
                     f"{'más grande' if s > 1 else 'más chica'})")
    if abs(shift[0]) > half:
        hints.append(f"mover/girar hacia la {'DERECHA' if shift[0] > 0 else 'IZQUIERDA'} ({abs(shift[0]):.0f} px)")
    if abs(shift[1]) > half:
        hints.append(f"mover/inclinar hacia {'ABAJO (hacia la base)' if shift[1] > 0 else 'ARRIBA (lejos de la base)'} "
                     f"({abs(shift[1]):.0f} px)")
    if abs(th) * radius > half:
        hints.append(f"rotar la cámara en sentido {'HORARIO' if th > 0 else 'ANTIHORARIO'} (visto desde detrás de la "
                     f"cámara, {abs(math.degrees(th)):.1f}°)")
    if res > half:
        hints.append(f"inclinación distinta (perspectiva, {res:.0f} px): corregir el ángulo de la cámara")
    out["aligned"] = bool(len(common) >= min_markers and err <= tol_px)
    out["hints"] = hints if not out["aligned"] else []
    return out


class SteadyAlignment:
    """Inicio automático: la vista lleva ``hold_s`` alineada sin interrupción y quieta (el centro de los marcadores
    no se aleja más de ``max_motion_px`` de su media en esa ventana). La odometría no sirve para esto: su ruido en
    reposo (mediana 3 mm, hasta 6 mm) es del orden del umbral."""

    def __init__(self, hold_s: float = 1.0, max_motion_px: float = 3.0):
        self.hold_s, self.max_motion_px = float(hold_s), float(max_motion_px)
        self._buf: List[Tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self._buf = []

    def update(self, t: float, res: Dict, live: Corners) -> bool:
        if not res.get("aligned") or not res.get("markers"):
            self._buf = []
            return False
        c = np.concatenate([live[i] for i in res["markers"]]).mean(0)
        self._buf.append((float(t), c))
        self._buf = [(ti, ci) for ti, ci in self._buf if ti >= t - self.hold_s - 0.5]
        if self._buf[-1][0] - self._buf[0][0] < self.hold_s:
            return False
        recent = np.array([ci for ti, ci in self._buf if ti >= t - self.hold_s])
        dev = np.linalg.norm(recent - recent.mean(0), axis=1).max()
        return bool(dev <= self.max_motion_px)


def summary_line(res: Dict) -> str:
    """Una línea de estado para la terminal."""
    if res.get("error_px") is None:
        return "[INICIO] " + "; ".join(res.get("hints") or ["sin marcadores"])
    head = (f"[INICIO] {'ALINEADA' if res['aligned'] else 'no alineada'} | error {res['error_px']:.1f} px "
            f"(tol {res['tol_px']:.0f}) | escala {res['scale']:.3f} | giro {res['roll_deg']:+.1f}° | "
            f"marcadores {res['n_markers']}")
    return head + ("" if res["aligned"] else " | " + "; ".join(res["hints"]))


# ---------------------------------------------------------------------------------------------------------------------
# Imagen de guía (para rqt_image_view)
# ---------------------------------------------------------------------------------------------------------------------
def draw_target_cell(out: np.ndarray, polygon_px, k, label: str, yaw_deg: Optional[float] = None) -> None:
    """Dibuja la celda del programa (polígono en píxeles de la vista de HOME3) y, si se da, una flecha con la
    orientación de las HOJAS de la herramienta: 0° = hacia arriba de la imagen, positivo = antihorario."""
    import cv2
    poly = (np.asarray(polygon_px, float) * k).astype(np.int32)
    col = (0, 220, 255)
    cv2.polylines(out, [poly], True, col, 3, cv2.LINE_AA)
    c = poly.mean(axis=0)
    if yaw_deg is not None:
        t = math.radians(float(yaw_deg))
        d = np.array([-math.sin(t), -math.cos(t)]) * 0.35 * float(np.ptp(poly[:, 0]))
        cv2.arrowedLine(out, tuple(int(v) for v in c - d), tuple(int(v) for v in c + d), col, 3, cv2.LINE_AA,
                        tipLength=0.3)
    org = (int(poly[:, 0].min()), int(poly[:, 1].min()) - 6)
    cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def draw_guide(img_rgb: np.ndarray, live: Corners, ref: ViewReference, res: Dict,
               target: Optional[Dict] = None) -> np.ndarray:
    """Imagen BGR con la referencia (verde), lo observado (naranja), flechas y el estado. ``target`` (opcional):
    ``{"polygon_px", "label", "yaw_deg"}`` de la celda pedida por el programa, en píxeles de la vista de HOME3."""
    import cv2
    h, w = img_rgb.shape[:2]
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    k = np.array([w / float(ref.image_size[0]), h / float(ref.image_size[1])])
    if target is not None:
        draw_target_cell(out, target["polygon_px"], k, target.get("label", ""), target.get("yaw_deg"))
    ok = bool(res.get("aligned"))
    ref_col, live_col = (60, 200, 60), (0, 140, 255)
    for i, c in ref.markers.items():
        pts = (c * k).astype(np.int32)
        cv2.polylines(out, [pts], True, ref_col, 2, cv2.LINE_AA)
        if i in live:
            a = tuple(int(x) for x in (c * k).mean(0))
            b = tuple(int(x) for x in live[i].mean(0))
            cv2.arrowedLine(out, b, a, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
    for c in live.values():
        cv2.polylines(out, [c.astype(np.int32)], True, live_col, 2, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - 12, cy), (cx + 12, cy), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 12), (cx, cy + 12), (200, 200, 200), 1, cv2.LINE_AA)
    banner = (0, 150, 0) if ok else (0, 0, 180)
    cv2.rectangle(out, (0, 0), (w, 26), banner, -1)
    txt = ("ALINEADA CON HOME3" if ok else "NO ALINEADA") + (
        "" if res.get("error_px") is None else f"  error {res['error_px']:.1f}px  escala {res['scale']:.3f}  "
                                               f"giro {res['roll_deg']:+.1f}")
    cv2.putText(out, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for n, hint in enumerate((res.get("hints") or [])[:4]):
        cv2.putText(out, hint[:80], (8, h - 12 - 18 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, hint[:80], (8, h - 12 - 18 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Construcción de la referencia
# ---------------------------------------------------------------------------------------------------------------------
def build_reference(images: Sequence[np.ndarray], dictionary: str = "DICT_4X4_50", min_fraction: float = 0.8,
                    source: str = "") -> ViewReference:
    """Mediana de las esquinas de cada marcador presente en al menos ``min_fraction`` de las imágenes (RGB o gris)."""
    import cv2
    det: List[Corners] = []
    size = None
    for im in images:
        g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if size is None:
            size = (g.shape[1], g.shape[0])
        elif size != (g.shape[1], g.shape[0]):
            raise ValueError("todas las imágenes de referencia deben tener el mismo tamaño")
        det.append(detect_markers(g, dictionary))
    if not det:
        raise ValueError("sin imágenes de referencia")
    ids = sorted({i for d in det for i in d})
    keep = [i for i in ids if sum(i in d for d in det) >= min_fraction * len(det)]
    if not keep:
        raise ValueError("ningún marcador aparece en suficientes imágenes")
    markers, spread = {}, []
    for i in keep:
        stack = np.stack([d[i] for d in det if i in d])          # [n, 4, 2]
        med = np.median(stack, axis=0)
        markers[i] = med
        spread.append(np.linalg.norm(stack - med, axis=2).ravel())
    rep = float(np.percentile(np.concatenate(spread), 95))
    return ViewReference(dictionary=dictionary, image_size=size, markers=markers, repeatability_px=rep, source=source,
                         extra={"n_images": len(det), "n_images_per_marker": {int(i): int(sum(i in d for d in det))
                                                                               for i in keep}})
