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
    g = yaml.safe_load(open(a.bridge))["xarm5_bridge"]["ros__parameters"]
    box = WorkspaceLimits(g["workspace_x_min"], g["workspace_x_max"], g["workspace_y_min"], g["workspace_y_max"],
                          g["workspace_z_min"], g["workspace_z_max"], g["workspace_min_radius_xy"],
                          g["workspace_max_radius_xy"])
    reach = ReachLimit(g["workspace_shoulder_xyz_m"], g["workspace_wrist_offset_m"], g["workspace_max_reach_m"],
                       g["workspace_min_reach_m"])
    pl = (g["arm_plane_pitch_min"], g["arm_plane_pitch_max"])
    al = yaml.safe_load(open(a.alignment))["xarm5_dataset_alignment"]
    T_tcp_link = pose_matrix(al["tcp_to_camera_xyz_m"], rpy_to_quat(*al["tcp_to_camera_rpy_rad"]))
    cal = json.loads(Path(a.calib).read_text())
    K = np.array(cal["K"]).reshape(3, 3)
    T_link_opt = np.array(cal["T_link_opt"])
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

    def reachable(p_xy):
        ok, worst = 0, 0.0
        # Orientación de la CÁMARA al final según el protocolo (check_camera_yaws_deg; [0] = la de HOME3, sin girarla con
        # las hojas). cam_looking_down usa "positivo = antihorario"; la configuración, "+90 = derecha".
        combos = [(h, -float(yw)) for h in cfg["check_heights_m"] for yw in cfg.get("check_camera_yaws_deg", [0.0])]
        for h, yw in combos:
            T_opt = cam_looking_down([p_xy[0], p_xy[1], zt + h], yw)
            T_tcp = T_opt @ inv_T(T_link_opt) @ inv_T(T_tcp_link)
            xyz, q = T_tcp[:3, 3], matrix_to_quat(T_tcp[:3, :3])
            az = arm_azimuth(xyz)
            pitch, yaw = arm_plane_params(q, az)
            pitch_c = min(max(pitch, pl[0]), pl[1])
            Rq, Rr = T_tcp[:3, :3], q2m(arm_plane_quat(az, pitch_c, yaw))
            dev = math.degrees(math.acos(min(1.0, (np.trace(Rq.T @ Rr) - 1) / 2)))
            out, _, valid = clamp_to_workspace(xyz, pitch_c, box, reach)
            clamp = float(np.linalg.norm(out - xyz))
            good = valid and clamp < 0.01 and dev < 8.0
            ok += good
            worst = max(worst, clamp)
        return ok / len(combos), worst

    img = cv2.imread(a.image)
    over = img.copy()
    cells = {}
    rows = [(0, (1, 2, 3)), (1, (4, 5, 6)), (2, (7, 8, 9))]           # fila de x: cerca, medio, lejos
    for ri, ids in rows:
        for ci, cid in enumerate(ids):                              # columna: izquierda (+y) -> derecha (-y)
            y_hi, y_lo = ye[3 - ci], ye[2 - ci]
            x_lo, x_hi = xe[ri], xe[ri + 1]
            corners = [(x_lo, y_hi), (x_hi, y_hi), (x_hi, y_lo), (x_lo, y_lo)]
            pts = [((x_lo + x_hi) / 2, (y_lo + y_hi) / 2)] + corners
            fr = [reachable(p) for p in pts]
            frac = min(f[0] for f in fr)
            d_mark = min(np.min(np.abs(np.array([[m[:, 0].min() - x_hi, x_lo - m[:, 0].max()]]))) if
                         (m[:, 1].min() <= y_hi and m[:, 1].max() >= y_lo) else 1.0 for m in markers.values())
            poly = project([[x, y, zt] for x, y in corners])
            cells[cid] = {"name": CELL_NAMES[cid], "x_m": [x_lo, x_hi], "y_m": [y_lo, y_hi],
                          "center_m": [round((x_lo + x_hi) / 2, 3), round((y_lo + y_hi) / 2, 3)],
                          "polygon_home3_px": np.round(poly, 1).tolist(), "reachable_fraction": round(frac, 3),
                          "max_clamp_m": round(max(f[1] for f in fr), 4)}
            color = (40, 170, 40) if frac == 1.0 else (0, 0, 220)
            cv2.polylines(over, [poly.astype(np.int32)], True, color, 2, cv2.LINE_AA)
            c = poly.mean(0).astype(int)
            cv2.putText(over, str(cid), (c[0] - 8, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            print(f"celda {cid} ({CELL_NAMES[cid]:34s}) x {x_lo:.2f}-{x_hi:.2f} y {y_lo:+.2f}..{y_hi:+.2f} | "
                  f"alcanzable {100 * frac:5.1f} % | recorte máx {1000 * cells[cid]['max_clamp_m']:.0f} mm")
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
