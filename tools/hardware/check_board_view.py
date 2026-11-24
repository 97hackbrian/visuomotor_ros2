#!/usr/bin/env python3
"""Colocar el tablero ChArUco en la mesa: comprobación en vivo desde la vista de HOME3.

Con la cámara corriendo, informa a 2 Hz:
* la guía de inicio: si la vista coincide con HOME3, según los ArUco de la mesa;
* el tablero: esquinas detectadas de las posibles, distancia de la cámara al centro del tablero, ángulo de visión y
  error de reproyección.
También publica una imagen anotada (``/board_check/image``, verla con rqt_image_view).

Objetivo al colocar el tablero:
* desde la vista de HOME3 se ven TODAS las esquinas;
* no tapa los dos ArUco de la mesa ni la zona donde va la herramienta;
* sigue viéndose al bajar hacia la herramienta el mayor tiempo posible.

Uso (contenedor de la cámara con el driver corriendo):
    python3 tools/hardware/check_board_view.py
    python3 tools/hardware/check_board_view.py --snapshot docs/hardware/board_placement.png   # guarda una imagen al salir
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))

from visuomotor_bringup.charuco import CharucoConfig, chessboard_corners, detect_corners, estimate_pose, make_board  # noqa: E402,E501


def board_view_stats(gray: np.ndarray, board, cfg: CharucoConfig, k: np.ndarray, dist=None) -> Dict:
    """Esquinas vistas, distancia al centro del tablero, ángulo de visión y reproyección (sin ROS)."""
    total = (cfg.squares_x - 1) * (cfg.squares_y - 1)
    px, ids = detect_corners(gray, board, cfg, k, dist)
    out: Dict = {"corners": 0 if ids is None else int(len(ids)), "total": total, "distance_m": None,
                 "view_angle_deg": None, "reprojection_px": None, "corners_px": px}
    pose = estimate_pose(gray, board, cfg, k, dist, min_corners=6)
    if pose is not None:
        center = chessboard_corners(board).mean(0)
        c_cam = pose.rotation @ center + pose.translation
        normal = pose.rotation[:, 2]
        out.update(distance_m=float(np.linalg.norm(c_cam)),
                   view_angle_deg=float(math.degrees(math.acos(min(1.0, abs(float(normal @ c_cam)) /
                                                                   float(np.linalg.norm(c_cam)))))),
                   reprojection_px=float(pose.reprojection_error_px))
    return out


def summary(st: Dict, guide: Optional[Dict]) -> str:
    g = ""
    if guide is not None:
        g = "HOME3 " + ("ALINEADA" if guide.get("aligned") else "no alineada")
        if guide.get("error_px") is not None:
            g += f" ({guide['error_px']:.1f} px)"
        g += " | "
    b = f"tablero {st['corners']}/{st['total']} esquinas"
    if st["distance_m"] is not None:
        b += (f", a {st['distance_m']:.2f} m, ángulo {st['view_angle_deg']:.0f}°, reproyección "
              f"{st['reprojection_px']:.2f} px")
    ok = st["corners"] == st["total"]
    return g + b + ("  -> OK" if ok else "  -> faltan esquinas: mover o despejar el tablero")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    ap.add_argument("--board-config", default=str(REPO / "visuomotor_bringup/config/realsense/charuco_board.yaml"))
    ap.add_argument("--home3-reference",
                    default=str(REPO / "visuomotor_bringup/config/realsense/home3_view_reference.yaml"))
    ap.add_argument("--snapshot", default="", help="guardar la última imagen anotada al salir (Ctrl+C)")
    a = ap.parse_args(argv)

    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from visuomotor_data_collection.vo_start_guide import ViewReference, detect_markers, view_alignment

    cfg = CharucoConfig.load(a.board_config)
    board = make_board(cfg)
    ref = ViewReference.load(a.home3_reference) if Path(a.home3_reference).exists() else None
    rclpy.init()
    node = rclpy.create_node("check_board_view")
    bridge = CvBridge()
    state = {"img": None, "k": None, "d": None, "last": None}
    node.create_subscription(Image, a.image_topic, lambda m: state.__setitem__("img", m), qos_profile_sensor_data)

    def on_info(m):
        if state["k"] is None:
            state["k"] = np.asarray(m.k, float).reshape(3, 3)
            state["d"] = np.asarray(m.d, float) if len(m.d) else None
    node.create_subscription(CameraInfo, a.info_topic, on_info, 10)
    pub = node.create_publisher(Image, "/board_check/image", 2)
    print(f"Tablero {cfg.squares_x}x{cfg.squares_y} ({'medido' if cfg.measured else 'NOMINAL, sin medir'}); "
          f"imagen anotada en /board_check/image. Ctrl+C para salir.")
    t_next = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() < t_next or state["img"] is None or state["k"] is None:
                continue
            t_next = time.time() + 0.5
            rgb = bridge.imgmsg_to_cv2(state["img"], desired_encoding="rgb8")
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            st = board_view_stats(gray, board, cfg, state["k"], state["d"])
            guide = None
            if ref is not None:
                guide = view_alignment(detect_markers(gray, ref.dictionary), (rgb.shape[1], rgb.shape[0]), ref)
            line = summary(st, guide)
            print(line, flush=True)
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if st["corners_px"] is not None:
                for p in st["corners_px"]:
                    cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 200, 0), -1, cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 120, 0) if st["corners"] == st["total"] else (0, 0, 170), -1)
            cv2.putText(out, line[:95], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            state["last"] = out
            if pub.get_subscription_count() > 0:
                msg = bridge.cv2_to_imgmsg(out, encoding="bgr8")
                msg.header = state["img"].header
                pub.publish(msg)
    except KeyboardInterrupt:
        pass
    finally:
        if a.snapshot and state["last"] is not None:
            cv2.imwrite(a.snapshot, state["last"])
            print(f"Imagen guardada: {a.snapshot}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
