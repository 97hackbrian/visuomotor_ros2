#!/usr/bin/env python3
"""Celdas de la mesa para el programa v8, en metros: alcanzabilidad y dibujo sobre la vista de HOME3.

Lee ``visuomotor_bringup/config/realsense/v8_cells.yaml`` (grilla 3x3 en ``link_base``) y:
1. **Comprueba que el robot llega** a la vista final de cada celda. Para cada esquina y el centro, con varias alturas
   y orientaciones de la herramienta, pone la cámara sobre el punto mirando hacia abajo, calcula el TCP con la
   extrínseca medida y aplica la MISMA caja, alcance y proyección de 5 GDL que el puente.
2. **Proyecta las celdas** en la vista de HOME3: robot en HOME3 (TCP medido) con la extrínseca cámara-brida medida.
3. **Comprueba la distancia a los ArUco** de la mesa (``table_markers.yaml``).
Escribe ``v8_cells_home3.yaml`` (polígonos en píxeles de HOME3 640x480, para la guía del recolector) y un PNG con la
grilla.

Uso (contenedor):
    python3 tools/hardware/make_v8_cells.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for p in ("visuomotor_bringup", "tools/xarm5", "visuomotor_data_collection"):
    sys.path.insert(0, str(REPO / p))
from visuomotor_bringup.xarm5_interface import (ReachLimit, WorkspaceLimits, arm_azimuth,  # noqa: E402
                                                arm_plane_params, arm_plane_quat, clamp_to_workspace)
from convert_vio_dataset_to_xarm5 import matrix_to_quat, pose_matrix, rpy_to_quat  # noqa: E402
from visuomotor_data_collection.start_anchor import inv_T  # noqa: E402
from visuomotor_data_collection.vo_schedule import CELL_NAMES  # noqa: E402

S = "experiments/vio_gt/2026-11-24"


def q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_looking_down(p, yaw_deg):
    """Cámara (óptica) en ``p`` mirando hacia abajo; ``yaw`` 0 = arriba de la imagen hacia +x (lejos de la base),
    positivo = antihorario visto desde arriba (convención de vo_schedule)."""
    s, c = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = np.c_[[s, -c, 0.0], [-c, -s, 0.0], [0.0, 0.0, -1.0]]
    T[:3, 3] = p
    return T


def cam_oblique(p_xy, table_z: float, dist: float, tilt_deg: float, yaw_deg: float) -> np.ndarray:
    """Cámara (óptica) a ``dist`` del punto de la mesa ``p_xy`` y MIRÁNDOLO, inclinada ``tilt_deg`` desde la vertical
    hacia la base del robot (en el plano vertical radial: así mira un brazo que se estira hacia afuera). Arriba de la
    imagen ~ dirección ``yaw_deg`` (convención de ``cam_looking_down``: 0 = +x, positivo = antihorario). Con
    ``tilt_deg = 0`` es exactamente ``cam_looking_down``."""
    t = math.radians(tilt_deg)
    r = np.array([p_xy[0], p_xy[1], 0.0])
    r = r / max(np.linalg.norm(r), 1e-9)
    pw = np.array([p_xy[0], p_xy[1], table_z])
    c = pw + dist * (-math.sin(t) * r + math.cos(t) * np.array([0.0, 0.0, 1.0]))
    z = (pw - c) / np.linalg.norm(pw - c)
    up = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    u = up - np.dot(up, z) * z
    u = u / np.linalg.norm(u)
    y = -u
    T = np.eye(4)
    T[:3, :3] = np.c_[np.cross(y, z), y, z]
    T[:3, 3] = c
    return T


class ReachModel:
    """¿Llega el robot a la vista final de un punto de la mesa? Pone la cámara a ``heights`` del punto, mirándolo:
    justo encima o, si no llega así, inclinada hacia la base hasta ``max_tilt_deg`` (de ``tilt_step_deg`` en
    ``tilt_step_deg``). Orientaciones de cámara ``cam_yaws`` con "+90 = derecha". Calcula el TCP con la extrínseca
    medida y aplica la MISMA caja, alcance y proyección de 5 GDL que el puente. Lo usan este script y
    ``adjust_v8_grid.py``.

    2026-11-25: antes se exigía la cámara JUSTO ENCIMA de cada esquina. Así, en los bordes de la mesa el TCP salía de
    la caja (|y| <= 0,30) o del radio de 0,65 m, y celdas a las que el robot sí llega (el usuario lo comprobó) salían
    "no llega". Mirando en diagonal desde más cerca de la base (como ya mira la vista de HOME3) llega a casi todas,
    sin tocar la caja."""

    def __init__(self, bridge_yaml, alignment_yaml, calib_json, table_z: float, heights, cam_yaws=(0.0,),
                 max_tilt_deg: float = 0.0, tilt_step_deg: float = 15.0):
        g = yaml.safe_load(open(bridge_yaml))["xarm5_bridge"]["ros__parameters"]
        self.box = WorkspaceLimits(g["workspace_x_min"], g["workspace_x_max"], g["workspace_y_min"],
                                   g["workspace_y_max"], g["workspace_z_min"], g["workspace_z_max"],
                                   g["workspace_min_radius_xy"], g["workspace_max_radius_xy"])
        self.reach = ReachLimit(g["workspace_shoulder_xyz_m"], g["workspace_wrist_offset_m"],
                                g["workspace_max_reach_m"], g["workspace_min_reach_m"])
        self.pl = (g["arm_plane_pitch_min"], g["arm_plane_pitch_max"])
        al = yaml.safe_load(open(alignment_yaml))["xarm5_dataset_alignment"]
        self.T_tcp_link = pose_matrix(al["tcp_to_camera_xyz_m"], rpy_to_quat(*al["tcp_to_camera_rpy_rad"]))
        self.T_link_opt = np.array(json.loads(Path(calib_json).read_text())["T_link_opt"])
        self.zt = float(table_z)
        # cam_looking_down usa "positivo = antihorario"; la configuración, "+90 = derecha"
        self.combos = [(float(h), -float(yw)) for h in heights for yw in cam_yaws]
        step = max(float(tilt_step_deg), 1.0)
        self.tilts = [0.0] + [float(t) for t in np.arange(step, float(max_tilt_deg) + 1e-9, step)]

    def _pose_ok(self, T_opt):
        """(alcanzable, recorte en m) del TCP para la cámara en ``T_opt``."""
        T_tcp = T_opt @ inv_T(self.T_link_opt) @ inv_T(self.T_tcp_link)
        xyz, q = T_tcp[:3, 3], matrix_to_quat(T_tcp[:3, :3])
        az = arm_azimuth(xyz)
        pitch, yaw = arm_plane_params(q, az)
        pitch_c = min(max(pitch, self.pl[0]), self.pl[1])
        Rq, Rr = T_tcp[:3, :3], q2m(arm_plane_quat(az, pitch_c, yaw))
        dev = math.degrees(math.acos(min(1.0, (np.trace(Rq.T @ Rr) - 1) / 2)))
        out, _, valid = clamp_to_workspace(xyz, pitch_c, self.box, self.reach)
        clamp = float(np.linalg.norm(out - xyz))
        return bool(valid and clamp < 0.01 and dev < 8.0), clamp

    def point_detail(self, p_xy):
        """(fracción de alturas×orientaciones alcanzables, recorte mínimo del peor caso en m, inclinación máxima
        necesaria en grados) para la vista final de ``p_xy``."""
        ok, worst, tilt_needed = 0, 0.0, 0.0
        for h, yw in self.combos:
            best = None
            for t in self.tilts:
                good, clamp = self._pose_ok(cam_oblique(p_xy, self.zt, h, t, yw))
                best = clamp if best is None else min(best, clamp)
                if good:
                    ok += 1
                    tilt_needed = max(tilt_needed, t)
                    best = 0.0 if t > 0 else clamp
                    break
            worst = max(worst, best)
        return ok / len(self.combos), worst, tilt_needed

    def point(self, p_xy):
        """(fracción de combinaciones alcanzables, recorte en m) para la vista final de ``p_xy``."""
        frac, worst, _ = self.point_detail(p_xy)
        return frac, worst

    def cell_detail(self, x_lo, x_hi, y_lo, y_hi) -> dict:
        """Centro y 4 esquinas: ``center`` (fracción del centro), ``worst`` (peor punto), ``clamp_m`` e
        ``tilt_deg`` (inclinación máxima necesaria)."""
        pts = [((x_lo + x_hi) / 2, (y_lo + y_hi) / 2), (x_lo, y_hi), (x_hi, y_hi), (x_hi, y_lo), (x_lo, y_lo)]
        fr = [self.point_detail(p) for p in pts]
        return {"center": fr[0][0], "worst": min(f[0] for f in fr), "clamp_m": max(f[1] for f in fr),
                "tilt_deg": max(f[2] for f in fr)}

    def cell(self, x_lo, x_hi, y_lo, y_hi):
        """Peor caso entre el centro y las 4 esquinas de la celda: (fracción alcanzable, recorte máximo en m)."""
        d = self.cell_detail(x_lo, x_hi, y_lo, y_hi)
        return d["worst"], d["clamp_m"]


def default_paths() -> dict:
    """Rutas por defecto (las mismas en este script y en adjust_v8_grid.py)."""
    return {"cells": REPO / "visuomotor_bringup/config/realsense/v8_cells.yaml",
            "table": REPO / "visuomotor_bringup/config/realsense/table_markers.yaml",
            "image": REPO / S / "home3_now/home3_0.png",
            "bridge": REPO / "visuomotor_bringup/config/xarm5/xarm5_real_params.yaml",
            "alignment": REPO / "visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml",
            "calib": REPO / "datasets/vio_frames/201446_static_board/calib.json"}


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=str(REPO / "visuomotor_bringup/config/realsense/v8_cells.yaml"))
    ap.add_argument("--table", default=str(REPO / "visuomotor_bringup/config/realsense/table_markers.yaml"))
    ap.add_argument("--home3-tcp", default=str(REPO / S / "home3_tcp.json"))
    ap.add_argument("--hand-eye", default=str(REPO / S / "runs/201446_static_board/hand_eye.json"))
    ap.add_argument("--calib", default=str(REPO / "datasets/vio_frames/201446_static_board/calib.json"))
    ap.add_argument("--image", default=str(REPO / S / "home3_now/home3_0.png"))
    ap.add_argument("--bridge", default=str(REPO / "visuomotor_bringup/config/xarm5/xarm5_real_params.yaml"))
    ap.add_argument("--alignment", default=str(REPO / "visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml"))
    ap.add_argument("--out", default=str(REPO / "visuomotor_bringup/config/realsense/v8_cells_home3.yaml"))
    ap.add_argument("--png", default=str(REPO / "experiments/collection_schedules/v8_cells_home3.png"))
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.cells))["v8_cells"]
    xe, ye, zt = cfg["x_edges_m"], cfg["y_edges_m"], float(cfg["table_z_m"])
    # Orientación de la CÁMARA al final según el protocolo (check_camera_yaws_deg; [0] = la de HOME3, sin girarla
    # con las hojas)
    model = ReachModel(a.bridge, a.alignment, a.calib, zt, cfg["check_heights_m"], cfg.get("check_camera_yaws_deg", [0.0]),
                       cfg.get("check_max_tilt_deg", 0.0), cfg.get("check_tilt_step_deg", 15.0))
    # Con la cámara alineada con las hojas, cada demo tiene UNA orientación: el alcance y la pista (diagonal, hacia el
    # centro) se guardan también por orientación, para que el recolector diga solo lo de esa demo.
    cam_yaws = [float(y) for y in cfg.get("check_camera_yaws_deg", [0.0])]
    by_yaw_models = ({y: ReachModel(a.bridge, a.alignment, a.calib, zt, cfg["check_heights_m"], [y],
                                    cfg.get("check_max_tilt_deg", 0.0), cfg.get("check_tilt_step_deg", 15.0))
                      for y in cam_yaws} if len(cam_yaws) > 1 else {})
    cal = json.loads(Path(a.calib).read_text())
    K = np.array(cal["K"]).reshape(3, 3)
    h3 = json.loads(Path(a.home3_tcp).read_text())
    T_base_tcp_h3 = np.eye(4)
    T_base_tcp_h3[:3, :3], T_base_tcp_h3[:3, 3] = q2m(h3["q"]), h3["xyz"]
    T_base_opt_h3 = T_base_tcp_h3 @ np.array(json.loads(Path(a.hand_eye).read_text())["T_tcp_optical"])
    markers = {int(k): np.array(v) for k, v in yaml.safe_load(open(a.table))["table_markers"]["corners_base_m"].items()}

    def project(pts):
        T_cb = inv_T(T_base_opt_h3)
        pc = (T_cb[:3, :3] @ np.asarray(pts).T).T + T_cb[:3, 3]
        uv = (K @ pc.T).T
        return uv[:, :2] / uv[:, 2:]

    img = cv2.imread(a.image)
    over = img.copy()
    cells = {}
    rows = [(0, (1, 2, 3)), (1, (4, 5, 6)), (2, (7, 8, 9))]           # fila de x: cerca, medio, lejos
    for ri, ids in rows:
        for ci, cid in enumerate(ids):                              # columna: izquierda (+y) -> derecha (-y)
            y_hi, y_lo = ye[3 - ci], ye[2 - ci]
            x_lo, x_hi = xe[ri], xe[ri + 1]
            corners = [(x_lo, y_hi), (x_hi, y_hi), (x_hi, y_lo), (x_lo, y_lo)]
            det = model.cell_detail(x_lo, x_hi, y_lo, y_hi)
            frac, max_clamp = det["worst"], det["clamp_m"]
            poly = project([[x, y, zt] for x, y in corners])
            cells[cid] = {"name": CELL_NAMES[cid], "x_m": [x_lo, x_hi], "y_m": [y_lo, y_hi],
                          "center_m": [round((x_lo + x_hi) / 2, 3), round((y_lo + y_hi) / 2, 3)],
                          "polygon_home3_px": np.round(poly, 1).tolist(), "reachable_fraction": round(frac, 3),
                          "center_reachable_fraction": round(det["center"], 3), "tilt_needed_deg": det["tilt_deg"],
                          "max_clamp_m": round(max_clamp, 4)}
            if by_yaw_models:
                cells[cid]["by_yaw"] = {}
                for y, my in by_yaw_models.items():
                    dy = my.cell_detail(x_lo, x_hi, y_lo, y_hi)
                    cells[cid]["by_yaw"][int(round(y))] = {"reachable_fraction": round(dy["worst"], 3),
                                                           "center_reachable_fraction": round(dy["center"], 3),
                                                           "tilt_needed_deg": dy["tilt_deg"]}
            color = (40, 170, 40) if frac == 1.0 else (0, 0, 220)
            cv2.polylines(over, [poly.astype(np.int32)], True, color, 2, cv2.LINE_AA)
            c = poly.mean(0).astype(int)
            cv2.putText(over, str(cid), (c[0] - 8, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            print(f"celda {cid} ({CELL_NAMES[cid]:34s}) x {x_lo:.2f}-{x_hi:.2f} y {y_lo:+.2f}..{y_hi:+.2f} | "
                  f"alcanzable {100 * frac:5.1f} % (centro {100 * det['center']:.0f} %) | inclinación {det['tilt_deg']:.0f}° | "
                  f"recorte máx {1000 * cells[cid]['max_clamp_m']:.0f} mm")
    for m in markers.values():
        cv2.polylines(over, [project(m).astype(np.int32)], True, (220, 120, 0), 2, cv2.LINE_AA)
    # Comprobación cruzada (2026-11-25): la guía en vivo proyecta la grilla con la homografía de los ArUco detectados
    # (vo_start_guide.table_homography), no con la pose de la cámara. En la imagen de referencia de HOME3 ambos métodos,
    # independientes, deben coincidir; si no, el orden de las esquinas de table_markers.yaml o la calibración está mal.
    from visuomotor_data_collection.vo_schedule import cells_from_edges, yaw_words
    from visuomotor_data_collection.vo_start_guide import (ViewReference, detect_markers, draw_cell_grid,
                                                           project_table_xy, table_homography)
    cells_m = cells_from_edges(xe, ye)
    table_xy = {i: m[:, :2] for i, m in markers.items()}
    live = detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    H = table_homography(live, table_xy)
    if H is None:
        print("ERROR: no se ven los dos ArUco en la imagen de referencia")
        return 1
    mk_err = max(float(np.abs(project_table_xy(H, table_xy[i]) - live[i]).max()) for i in table_xy if i in live)
    grid_err = max(float(np.abs(project_table_xy(H, [(x0, y1), (x1, y1), (x1, y0), (x0, y0)])
                                - np.array(cells[c]["polygon_home3_px"])).max())
                   for c, (x0, x1, y0, y1) in cells_m.items())
    print(f"Comprobación: homografía de los ArUco -> esquinas de los marcadores a {mk_err:.1f} px; esquinas de las "
          f"celdas a {grid_err:.1f} px de las proyectadas con la pose de la cámara en HOME3")
    ref_path = REPO / "visuomotor_bringup/config/realsense/home3_view_reference.yaml"
    if ref_path.is_file():
        ref = ViewReference.load(ref_path)
        H_ref = table_homography(ref.markers, table_xy)
        ref_err = max(float(np.abs(project_table_xy(H_ref, [(x0, y1), (x1, y1), (x1, y0), (x0, y0)])
                                   - np.array(cells[c]["polygon_home3_px"])).max())
                      for c, (x0, x1, y0, y1) in cells_m.items())
        print(f"             con la vista de referencia de la guía (mediana de 13 intentos): {ref_err:.1f} px")
    grid_png = Path(a.png).with_name(Path(a.png).stem.replace("cells", "grid") + ".png")
    ex_cell, ex_yaw = 5, 90.0
    g_img = img.copy()
    draw_cell_grid(g_img, H, cells_m, ex_cell, ex_yaw,
                   f"EJEMPLO: celda {ex_cell} ({CELL_NAMES[ex_cell]}) | hojas {yaw_words(ex_yaw).replace('°', ' grados')}")
    cv2.putText(g_img, "Grilla v8 sobre la imagen de HOME3 (camara en el TCP)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(grid_png), g_img)
    print(f"-> {grid_png}")
    out_img = cv2.addWeighted(over, 0.85, img, 0.15, 0)
    cv2.putText(out_img, "Celdas v8 (m, link_base) en la vista de HOME3 | verde = el robot llega", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.png, out_img)
    doc = {"v8_cells_home3": {"image_size": [img.shape[1], img.shape[0]], "source_cells": str(Path(a.cells).name),
                              "cells": {int(k): v for k, v in cells.items()}}}
    Path(a.out).write_text("# Generado por tools/hardware/make_v8_cells.py: no editar a mano.\n" +
                           yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"-> {a.out}\n-> {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
