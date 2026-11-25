#!/usr/bin/env python3
"""Ajustar a mano la grilla de celdas del programa v8, viéndola SOBRE LA MESA.

Muestra la grilla numerada con la homografía de los ArUco de la mesa (la misma que usa la guía del recolector):
* sobre la imagen de HOME3 tomada con la cámara en el TCP (por defecto), o
* sobre la cámara EN VIVO (``--live``); si en vivo no se ven los dos ArUco, vuelve a la imagen de HOME3.

Cuatro barras, en mm de ``link_base``: CERCA (borde del lado de la base), LEJOS (lado de los ArUco), IZQUIERDA (+y) y
DERECHA (-y, en positivo). Las filas y las columnas quedan iguales.

Cada celda se pinta según si el robot llega a su vista final: cámara a 25-30 cm sobre el centro y las esquinas de la
celda, con la orientación de HOME3. Usa la misma caja, alcance y proyección de 5 GDL que el puente
(``make_v8_cells.ReachModel``).
* verde = llega;
* rojo = no llega (esas demos se recortarían o descartarían al convertir).
La línea roja marca el límite de los ArUco (``marker_clearance_m``): la grilla no debe pasarla.

Teclas (con la ventana activa):
* [G] guardar: escribe ``x_edges_m`` y ``y_edges_m`` en ``v8_cells.yaml`` (conserva los comentarios) y regenera
  ``v8_cells_home3.yaml`` y las figuras con ``make_v8_cells.py``. Si hay celdas rojas o la grilla pasa el límite de
  los ArUco, pide confirmar con otra [G].
* [R] volver a lo guardado.
* [Q] o [Esc] salir sin guardar.
Después de guardar, reiniciar el recolector: lee la grilla al arrancar.

Uso (con pantalla; lo lanza ``tools/hardware/v8_tmux.sh grid``):
    python3 tools/hardware/adjust_v8_grid.py             # sobre la imagen de HOME3
    python3 tools/hardware/adjust_v8_grid.py --live      # cámara en vivo (contenedor de la cámara corriendo)
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in ("tools/hardware", "visuomotor_data_collection"):
    sys.path.insert(0, str(REPO / _p))

from make_v8_cells import ReachModel, default_paths  # noqa: E402
from visuomotor_data_collection.vo_schedule import cells_from_edges  # noqa: E402
from visuomotor_data_collection.vo_start_guide import (detect_markers, project_table_xy,  # noqa: E402
                                                       table_homography)

WIN = "Grilla v8 (G guardar, R restaurar, Q salir)"
# Barras: (nombre, mínimo en mm, recorrido en mm). OpenCV las cuenta desde 0: valor = mínimo + posición.
BARS = (("cerca x (mm)", 300, 280), ("lejos x (mm)", 480, 220), ("izquierda +y (mm)", 0, 450),
        ("derecha -y (mm)", 0, 450))


# ---------------------------------------------------------------------------------------------------------------------
# Sin interfaz (probado en tests/test_adjust_v8_grid.py)
# ---------------------------------------------------------------------------------------------------------------------
def edges_from_limits(near: float, far: float, left: float, right: float) -> Tuple[List[float], List[float]]:
    """Bordes de 3 filas y 3 columnas iguales. ``right`` es la distancia hacia -y (positiva)."""
    xe = [round(near + (far - near) * k / 3.0, 4) for k in range(4)]
    ye = [round(-right + (left + right) * k / 3.0, 4) for k in range(4)]
    return xe, ye


def limits_from_edges(xe: Sequence[float], ye: Sequence[float]) -> Tuple[float, float, float, float]:
    xe, ye = sorted(xe), sorted(ye)
    return float(xe[0]), float(xe[-1]), float(ye[-1]), float(-ye[0])


def check_limits(near: float, far: float, left: float, right: float, min_size: float = 0.06) -> Optional[str]:
    """Motivo por el que la grilla no es válida, o None."""
    if far - near < min_size:
        return f"la grilla debe medir al menos {min_size * 100:.0f} cm a lo alto"
    if left + right < min_size:
        return f"la grilla debe medir al menos {min_size * 100:.0f} cm a lo ancho"
    return None


def write_edges_yaml(text: str, xe: Sequence[float], ye: Sequence[float], note: str) -> str:
    """Reemplaza los valores de ``x_edges_m`` e ``y_edges_m`` en el texto del YAML, conservando la indentación y el
    resto del archivo (comentarios incluidos). ``note`` va como comentario al final de cada línea."""
    def fmt(v):
        return "[" + ", ".join(f"{float(x):g}" for x in v) + "]"

    out = text
    for key, vals in (("x_edges_m", xe), ("y_edges_m", ye)):
        pat = re.compile(rf"^(\s*){key}:.*$", re.M)
        if len(pat.findall(out)) != 1:
            raise ValueError(f"se esperaba una sola línea '{key}:' en el YAML")
        out = pat.sub(lambda m: f"{m.group(1)}{key}: {fmt(vals)}   # {note}", out)
    return out


def marker_limit_x(table_xy: Dict[int, np.ndarray], clearance: float) -> float:
    """x máxima de la grilla para no tapar los ArUco."""
    return float(min(c[:, 0].min() for c in table_xy.values())) - float(clearance)


# ---------------------------------------------------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------------------------------------------------
def _text(img, txt, org, scale=0.5, col=(255, 255, 255), thick=1):
    import cv2
    import unicodedata
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def draw_overlay(img_bgr: np.ndarray, H: np.ndarray, cells_m: Dict[int, Sequence[float]], reach: Dict[int, Tuple],
                 x_limit: float, lines: Sequence[str]) -> np.ndarray:
    """Grilla con celdas verdes (el robot llega) o rojas (no llega), número y medidas, y el límite de los ArUco."""
    import cv2
    out = img_bgr.copy()
    layer = out.copy()
    polys = {}
    for cid, (x0, x1, y0, y1) in cells_m.items():
        poly = project_table_xy(H, [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]).astype(np.int32)
        polys[cid] = poly
        ok = reach.get(cid, (0.0, 0.0))[0] >= 1.0
        cv2.fillPoly(layer, [poly], (60, 190, 60) if ok else (40, 40, 230))
    cv2.addWeighted(layer, 0.30, out, 0.70, 0, dst=out)
    for cid, poly in polys.items():
        cv2.polylines(out, [poly], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(out, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
        c = poly.mean(axis=0).astype(int)
        frac, clamp = reach.get(cid, (0.0, 0.0))
        _text(out, str(cid), (int(c[0]) - 7, int(c[1]) + 2), 0.7, (255, 255, 255), 2)
        if frac < 1.0:
            _text(out, f"no llega ({clamp * 1000:.0f} mm)", (int(c[0]) - 45, int(c[1]) + 20), 0.38, (180, 180, 255))
    ys = np.linspace(-0.5, 0.5, 60)
    lim = project_table_xy(H, np.c_[np.full_like(ys, x_limit), ys]).astype(np.int32)
    cv2.polylines(out, [lim], False, (0, 0, 255), 2, cv2.LINE_AA)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 18 * len(lines) + 8), (30, 30, 30), -1)
    for k, ln in enumerate(lines):
        _text(out, ln, (8, 18 + 18 * k), 0.48, (255, 255, 255) if k else (0, 230, 255))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Cámara en vivo (opcional)
# ---------------------------------------------------------------------------------------------------------------------
class LiveImage:
    """Última imagen del tópico de color (rclpy en un hilo); sin cv_bridge."""

    def __init__(self, topic: str):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        rclpy.init()
        self.node = rclpy.create_node("adjust_v8_grid")
        self.img, self.lock = None, threading.Lock()
        self.node.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _cb(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        bgr = a[:, :, ::-1].copy() if m.encoding.lower().startswith("rgb") else a.copy()
        with self.lock:
            self.img = bgr

    def latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.img is None else self.img.copy()


# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import cv2
    d = default_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("cells", "table", "image", "bridge", "alignment", "calib"):
        ap.add_argument(f"--{k}", default=str(d[k]))
    ap.add_argument("--live", action="store_true", help="cámara en vivo en vez de la imagen de HOME3")
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    a = ap.parse_args(argv)

    cells_path = Path(a.cells)
    cfg = yaml.safe_load(cells_path.read_text())["v8_cells"]
    model = ReachModel(a.bridge, a.alignment, a.calib, float(cfg["table_z_m"]), cfg["check_heights_m"],
                       cfg.get("check_camera_yaws_deg", [0.0]))
    mk = yaml.safe_load(open(a.table))["table_markers"]["corners_base_m"]
    table_xy = {int(i): np.asarray(c, float)[:, :2] for i, c in mk.items()}
    x_limit = marker_limit_x(table_xy, float(cfg.get("marker_clearance_m", 0.03)))
    ref_img = cv2.imread(a.image)
    if ref_img is None:
        print(f"ERROR: no se pudo leer {a.image}")
        return 1
    H_ref = table_homography(detect_markers(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)), table_xy)
    if H_ref is None:
        print("ERROR: no se ven los dos ArUco en la imagen de HOME3")
        return 1
    live = LiveImage(a.image_topic) if a.live else None

    saved = limits_from_edges(cfg["x_edges_m"], cfg["y_edges_m"])
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1000, 760)

    def set_bars(lims):
        for (name, lo, span), v in zip(BARS, lims):
            cv2.setTrackbarPos(name, WIN, int(round(min(max(v * 1000.0 - lo, 0), span))))

    for name, lo, span in BARS:
        cv2.createTrackbar(name, WIN, 0, span, lambda _v: None)
    set_bars(saved)

    cache: Dict[Tuple, Dict[int, Tuple]] = {}
    confirm_until, msg, msg_until = 0.0, "", 0.0
    print("Ajuste de la grilla v8: barras en la ventana; [G] guardar, [R] restaurar, [Q]/[Esc] salir.")
    while True:
        lims = tuple((lo + cv2.getTrackbarPos(name, WIN)) / 1000.0 for name, lo, _span in BARS)
        near, far, left, right = lims
        xe, ye = edges_from_limits(near, far, left, right)
        bad = check_limits(near, far, left, right)
        cells_m = cells_from_edges(xe, ye) if bad is None else {}
        key = (tuple(xe), tuple(ye))
        if bad is None and key not in cache:
            cache[key] = {cid: model.cell(*b) for cid, b in cells_m.items()}
        reach = cache.get(key, {})
        n_ok = sum(1 for r in reach.values() if r[0] >= 1.0)
        over_markers = far > x_limit + 1e-6

        frame, H, src = None, None, "imagen de HOME3 (camara en el TCP)"
        if live is not None:
            frame = live.latest()
            if frame is not None:
                H = table_homography(detect_markers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), table_xy)
                src = "camara EN VIVO" if H is not None else "en vivo sin los 2 ArUco: imagen de HOME3"
        if H is None:
            frame, H = ref_img, H_ref

        lines = [f"Grilla: alto {(far - near) * 100:.1f} cm (x {near * 1000:.0f}-{far * 1000:.0f} mm), ancho "
                 f"{(left + right) * 100:.1f} cm (y {-right * 1000:+.0f}..{left * 1000:+.0f} mm) | celda "
                 f"{(far - near) / 3 * 100:.1f} x {(left + right) / 3 * 100:.1f} cm",
                 (bad or f"El robot llega a {n_ok}/9 celdas" + (" | PASA EL LIMITE DE LOS ARUCO" if over_markers
                                                                 else "")) + f" | {src}",
                 "G guardar | R restaurar | Q salir" + (f" | {msg}" if time.time() < msg_until else "")]
        out = draw_overlay(frame, H, cells_m, reach, x_limit, lines) if cells_m else frame.copy()
        cv2.imshow(WIN, out)
        k = cv2.waitKey(40) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            print("Salida sin guardar.")
            break
        if k in (ord("r"), ord("R")):
            set_bars(saved)
            msg, msg_until = "restaurada", time.time() + 2.0
        if k in (ord("g"), ord("G")):
            if bad is not None:
                msg, msg_until = bad, time.time() + 3.0
                continue
            risky = n_ok < 9 or over_markers
            if risky and time.time() > confirm_until:
                confirm_until = time.time() + 4.0
                msg, msg_until = "hay celdas rojas o tapa los ArUco: G otra vez para guardar igual", confirm_until
                continue
            note = f"ajustada con adjust_v8_grid.py el {datetime.now():%Y-%m-%d %H:%M}"
            cells_path.write_text(write_edges_yaml(cells_path.read_text(), xe, ye, note))
            import make_v8_cells
            rc = make_v8_cells.main(["--cells", str(cells_path), "--table", a.table, "--image", a.image,
                                     "--bridge", a.bridge, "--alignment", a.alignment, "--calib", a.calib])
            saved = lims
            confirm_until = 0.0
            msg, msg_until = ("GUARDADA (reiniciar el recolector)" if rc == 0 else "guardada, pero falló "
                              "make_v8_cells.py"), time.time() + 4.0
            print(f"Guardada: x {xe} y {ye} ({n_ok}/9 celdas alcanzables). Reiniciar el recolector para usarla.")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
