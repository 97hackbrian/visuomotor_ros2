#!/usr/bin/env python3
"""Calibración cámara–brida (eye-in-hand) de la D435i montada en el xArm5, con el tablero ChArUco fijo de la celda.

Plan: ``docs/plans/plan_d435i_herramienta_demostraciones.md``, Fase 2.1. Mide la rotación Y la traslación
cámara→TCP, que hasta ahora eran una suposición ([0, 0, 0] de traslación). De paso entrega la pose del tablero en
``link_base`` y la vista de referencia de HOME3 en el marco del tablero.

Subcomandos
-----------
``capture`` (ROS 2, con el brazo y la cámara en marcha):
    El usuario lleva el brazo a 15-25 poses distintas con el marcador de RViz (teleop existente) y pulsa una tecla
    en cada parada:
    * ``[C]``: guarda una muestra;
    * ``[H]``: guarda una muestra en HOME3, que sirve para comprobar la vista de referencia;
    * ``[Q]``: sale.
    Solo guarda si el brazo lleva ``--still-s`` quieto y el tablero se ve con suficientes esquinas y error de
    reproyección bajo. Las poses deben variar el giro de la base (J1), la inclinación y el giro de la muñeca (J5):
    sin rotaciones en al menos dos ejes, la calibración no está determinada.

``solve`` (sin ROS):
    Resuelve con los 5 métodos de ``cv2.calibrateHandEye`` y elige el de menor dispersión de la pose del tablero en
    la base. Esa dispersión es el residuo: con una calibración correcta, todas las muestras ven el tablero en el
    mismo sitio. Escribe ``hand_eye_result.json`` e imprime los valores para pegar en
    ``xarm5_dataset_alignment.yaml`` y ``charuco_board.yaml``. **No modifica ningún archivo de configuración.**

Uso:
    # Jetson o portátil con la cámara y el TF del robot visibles (ROS_DOMAIN_ID del robot):
    python3 tools/xarm5/calibrate_hand_eye.py capture --out-dir datasets/hand_eye/2026-11-24
    # Contenedor del portátil:
    python3 tools/xarm5/calibrate_hand_eye.py solve --samples datasets/hand_eye/2026-11-24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))

from visuomotor_bringup.charuco import DEFAULT_CONFIG as BOARD_CONFIG, CharucoConfig  # noqa: E402
from visuomotor_bringup.xarm5_interface import quat_to_matrix, quat_to_rpy, rpy_to_quat  # noqa: E402

POLICY_PARAMS = REPO / "deploy" / "jetson" / "config" / "jetson_policy_params.yaml"
METHODS = ("PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS")


# ---------------------------------------------------------------------------------------------------------------------
# Geometría (sin ROS)
# ---------------------------------------------------------------------------------------------------------------------
def to_t(r: np.ndarray, t: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return m


def inv_t(m: np.ndarray) -> np.ndarray:
    r, t = m[:3, :3], m[:3, 3]
    return to_t(r.T, -r.T @ t)


def rot_angle_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def matrix_to_quat_xyzw(r: np.ndarray) -> List[float]:
    """Matriz de rotación -> ``[qx, qy, qz, qw]`` con ``qw >= 0``."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


def mean_pose(ts: Sequence[np.ndarray]) -> np.ndarray:
    """Media de poses: posición media y rotación media cordal (proyección SVD de la suma)."""
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in ts))
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return to_t(r, np.mean([t[:3, 3] for t in ts], axis=0))


def board_in_base(base_tcp: Sequence[np.ndarray], tcp_cam: np.ndarray, cam_board: Sequence[np.ndarray]):
    """Pose del tablero en la base vista por cada muestra: ``T_base_tcp · T_tcp_cam · T_cam_board``."""
    return [b @ tcp_cam @ c for b, c in zip(base_tcp, cam_board)]


def consistency(ts: Sequence[np.ndarray]) -> Dict[str, float]:
    """Dispersión de un conjunto de poses que deberían ser iguales (mm y grados respecto a su media)."""
    m = mean_pose(ts)
    dp = np.array([np.linalg.norm(t[:3, 3] - m[:3, 3]) * 1000.0 for t in ts])
    dr = np.array([rot_angle_deg(m[:3, :3].T @ t[:3, :3]) for t in ts])
    return {"pos_median_mm": float(np.median(dp)), "pos_max_mm": float(dp.max()),
            "rot_median_deg": float(np.median(dr)), "rot_max_deg": float(dr.max())}


def rotation_diversity_deg(base_tcp: Sequence[np.ndarray]) -> float:
    """Mayor ángulo entre las orientaciones del TCP de dos muestras."""
    best = 0.0
    for i in range(len(base_tcp)):
        for j in range(i + 1, len(base_tcp)):
            best = max(best, rot_angle_deg(base_tcp[i][:3, :3].T @ base_tcp[j][:3, :3]))
    return best


def solve_hand_eye(base_tcp: Sequence[np.ndarray], cam_board: Sequence[np.ndarray], methods=METHODS) -> Dict:
    """Resuelve ``T_tcp_cam`` con cada método de OpenCV. Devuelve, por método, la transformación y su residuo
    (dispersión de la pose del tablero en la base)."""
    import cv2
    if len(base_tcp) < 3:
        raise ValueError("se necesitan al menos 3 muestras (recomendado 15-25)")
    out = {}
    for name in methods:
        r, t = cv2.calibrateHandEye([b[:3, :3] for b in base_tcp], [b[:3, 3] for b in base_tcp],
                                    [c[:3, :3] for c in cam_board], [c[:3, 3] for c in cam_board],
                                    method=getattr(cv2, f"CALIB_HAND_EYE_{name}"))
        x = to_t(r, t)
        out[name] = {"tcp_to_camera": x, "residual": consistency(board_in_base(base_tcp, x, cam_board))}
    return out


def pose_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    return to_t(quat_to_matrix(rpy_to_quat(*rpy)), xyz)


# ---------------------------------------------------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------------------------------------------------
def cmd_solve(args) -> int:
    import yaml
    d = Path(args.samples)
    data = json.loads((d / "samples.json").read_text())
    board = CharucoConfig.load(args.board_config)
    if not board.measured and not args.allow_nominal:
        print("charuco_board.yaml tiene medidas NOMINALES (measured: false): medir el tablero con calibre o pasar "
              "--allow-nominal (solo para pruebas; la traslación quedaría escalada).", file=sys.stderr)
        return 2
    samples = data["samples"]
    base_tcp = [np.asarray(s["T_base_tcp"]) for s in samples]
    cam_board = [np.asarray(s["T_cam_board"]) for s in samples]
    res = solve_hand_eye(base_tcp, cam_board)
    best = min(res, key=lambda k: res[k]["residual"]["pos_median_mm"])
    x_opt = res[best]["tcp_to_camera"]
    # camera_link (marco de la odometría y del dataset) en el TCP: T_tcp_link = T_tcp_opt · T_opt_link
    t_link_opt = np.asarray(data["T_cameralink_optical"]) if data.get("T_cameralink_optical") else None
    x_link = x_opt @ inv_t(t_link_opt) if t_link_opt is not None else None
    boards = board_in_base(base_tcp, x_opt, cam_board)
    t_base_board = mean_pose(boards)
    pol = yaml.safe_load(open(args.policy_params))["policy_node"]["ros__parameters"]
    t_base_tcp_home3 = pose_from_xyz_rpy([pol["home_x"], pol["home_y"], pol["home_z"]],
                                         [pol["home_roll"], pol["home_pitch"], pol["home_yaw"]])
    t_board_cam_home3 = inv_t(t_base_board) @ t_base_tcp_home3 @ x_opt
    home_checks = []
    for s, cb in zip(samples, cam_board):
        if s.get("home3"):
            measured = inv_t(cb)
            home_checks.append({"pos_mm": float(np.linalg.norm(measured[:3, 3] - t_board_cam_home3[:3, 3]) * 1000),
                                "rot_deg": rot_angle_deg(t_board_cam_home3[:3, :3].T @ measured[:3, :3])})
    div = rotation_diversity_deg(base_tcp)
    r = res[best]["residual"]
    ok = (r["pos_median_mm"] <= args.max_residual_mm and r["rot_median_deg"] <= args.max_residual_deg
          and div >= args.min_rotation_diversity_deg)
    result = {
        "samples": len(samples), "rotation_diversity_deg": div, "method": best,
        "methods": {k: {"tcp_to_camera_optical_xyz_m": v["tcp_to_camera"][:3, 3].tolist(),
                        "tcp_to_camera_optical_quat_xyzw": matrix_to_quat_xyzw(v["tcp_to_camera"][:3, :3]),
                        "residual": v["residual"]} for k, v in res.items()},
        "tcp_to_camera_optical": {"xyz_m": x_opt[:3, 3].tolist(), "quat_xyzw": matrix_to_quat_xyzw(x_opt[:3, :3])},
        "tcp_to_camera_link": None if x_link is None else {
            "xyz_m": x_link[:3, 3].tolist(), "quat_xyzw": matrix_to_quat_xyzw(x_link[:3, :3]),
            "rpy_rad": list(quat_to_rpy(matrix_to_quat_xyzw(x_link[:3, :3])))},
        "base_to_board": {"xyz_m": t_base_board[:3, 3].tolist(), "quat_xyzw": matrix_to_quat_xyzw(t_base_board[:3, :3])},
        "home3_board_to_camera": {"xyz_m": t_board_cam_home3[:3, 3].tolist(),
                                  "quat_xyzw": matrix_to_quat_xyzw(t_board_cam_home3[:3, :3])},
        "home3_checks": home_checks, "board_measured": board.measured,
        "acceptance": {"max_residual_mm": args.max_residual_mm, "max_residual_deg": args.max_residual_deg,
                       "min_rotation_diversity_deg": args.min_rotation_diversity_deg, "pass": bool(ok)},
    }
    (d / "hand_eye_result.json").write_text(json.dumps(result, indent=2))

    print(f"{len(samples)} muestras, diversidad de rotación {div:.1f}°")
    print(f"{'método':<11} {'pos. mediana':>13} {'pos. máx.':>10} {'rot. mediana':>13} {'rot. máx.':>10}")
    for k, v in res.items():
        rr = v["residual"]
        print(f"{k:<11} {rr['pos_median_mm']:>10.2f} mm {rr['pos_max_mm']:>7.2f} mm {rr['rot_median_deg']:>10.3f}° "
              f"{rr['rot_max_deg']:>8.3f}°{'  <- elegido' if k == best else ''}")
    for h in home_checks:
        print(f"Muestra en HOME3 frente a la vista calculada: {h['pos_mm']:.1f} mm, {h['rot_deg']:.2f}°")
    print("\nRESULTADO:", "PASA" if ok else "NO PASA (revisar muestras, tablero o medidas)", f"-> {d / 'hand_eye_result.json'}")
    if x_link is not None:
        rpy = result["tcp_to_camera_link"]["rpy_rad"]
        print("\n# xarm5_dataset_alignment.yaml (revisar antes de pegar; luego reconvertir el dataset):")
        print(f"  tcp_to_camera_xyz_m: [{', '.join(f'{v:.5f}' for v in x_link[:3, 3])}]")
        print(f"  tcp_to_camera_rpy_rad: [{', '.join(f'{v:.6f}' for v in rpy)}]")
    print("\n# charuco_board.yaml:")
    print(f"  base_to_board_xyz_m: [{', '.join(f'{v:.5f}' for v in t_base_board[:3, 3])}]")
    print(f"  base_to_board_quat_xyzw: [{', '.join(f'{v:.6f}' for v in result['base_to_board']['quat_xyzw'])}]")
    print(f"  home3_board_to_camera_xyz_m: [{', '.join(f'{v:.5f}' for v in t_board_cam_home3[:3, 3])}]")
    print(f"  home3_board_to_camera_quat_xyzw: "
          f"[{', '.join(f'{v:.6f}' for v in result['home3_board_to_camera']['quat_xyzw'])}]")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------
# capture (ROS 2)
# ---------------------------------------------------------------------------------------------------------------------
def cmd_capture(args) -> int:  # pragma: no cover - requiere ROS 2, cámara y robot
    import select
    import termios
    import threading
    import tty

    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    from visuomotor_bringup.charuco import estimate_pose, make_board

    board_cfg = CharucoConfig.load(args.board_config)
    board = make_board(board_cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = {"board": board_cfg.__dict__, "samples": [], "T_cameralink_optical": None,
             "frames": {"base": args.base_frame, "tcp": args.tcp_frame, "camera_link": args.camera_link_frame,
                        "optical": args.optical_frame}}

    def tf_to_t(tf) -> np.ndarray:
        q = tf.transform.rotation
        p = tf.transform.translation
        return to_t(quat_to_matrix([q.x, q.y, q.z, q.w]), [p.x, p.y, p.z])

    class Capture(Node):
        def __init__(self):
            super().__init__("hand_eye_capture")
            self.bridge = CvBridge()
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            self.image, self.k, self.dist = None, None, None
            self.history: List[Tuple[float, np.ndarray]] = []
            self.create_subscription(Image, args.image_topic, self._img, 5)
            self.create_subscription(CameraInfo, args.info_topic, self._info, 5)
            self.create_timer(0.1, self._track)

        def _img(self, msg):
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        def _info(self, msg):
            self.k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist = np.asarray(msg.d, dtype=np.float64)

        def _track(self):
            try:
                t = tf_to_t(self.tf.lookup_transform(args.base_frame, args.tcp_frame, rclpy.time.Time()))
            except Exception:
                return
            now = time.time()
            self.history = [(s, m) for s, m in self.history if now - s <= args.still_s + 0.5] + [(now, t)]

        def still(self) -> bool:
            now = time.time()
            win = [m for s, m in self.history if now - s <= args.still_s]
            if len(win) < max(3, int(args.still_s * 10) - 2):
                return False
            ref = win[-1]
            return all(np.linalg.norm(m[:3, 3] - ref[:3, 3]) * 1000 <= args.still_tol_mm and
                       rot_angle_deg(ref[:3, :3].T @ m[:3, :3]) <= 0.1 for m in win)

        def capture(self, home3: bool):
            if self.image is None or self.k is None or not self.history:
                self.get_logger().warn("Falta imagen, camera_info o el TF del robot.")
                return
            if not self.still():
                self.get_logger().warn(f"El brazo no lleva {args.still_s:.1f} s quieto: esperar y repetir.")
                return
            if store["T_cameralink_optical"] is None:
                try:
                    store["T_cameralink_optical"] = tf_to_t(self.tf.lookup_transform(
                        args.camera_link_frame, args.optical_frame, rclpy.time.Time())).tolist()
                except Exception as e:
                    self.get_logger().warn(f"Sin TF {args.camera_link_frame}->{args.optical_frame}: {e}")
            pose = estimate_pose(self.image, board, board_cfg, self.k, self.dist, min_corners=args.min_corners)
            if pose is None or pose.reprojection_error_px > args.max_reproj_px:
                self.get_logger().warn("Tablero no detectado con calidad suficiente "
                                       f"(mín. {args.min_corners} esquinas, error < {args.max_reproj_px} px).")
                return
            i = len(store["samples"])
            cv2.imwrite(str(out / f"sample_{i:03d}.png"), self.image)
            store["samples"].append({
                "T_base_tcp": self.history[-1][1].tolist(), "T_cam_board": to_t(pose.rotation, pose.translation).tolist(),
                "n_corners": pose.n_corners, "reprojection_error_px": pose.reprojection_error_px,
                "camera_matrix": self.k.tolist(), "dist_coeffs": self.dist.tolist(), "home3": home3,
                "stamp": time.time()})
            (out / "samples.json").write_text(json.dumps(store, indent=2))
            div = rotation_diversity_deg([np.asarray(s["T_base_tcp"]) for s in store["samples"]])
            self.get_logger().info(f"Muestra {i}{' (HOME3)' if home3 else ''}: {pose.n_corners} esquinas, "
                                   f"{pose.reprojection_error_px:.2f} px; diversidad de rotación {div:.0f}°")

    rclpy.init()
    node = Capture()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("[C] guardar muestra  [H] muestra en HOME3  [Q] salir. Mover el brazo con el marcador de RViz entre muestras.")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in ("c", "h"):
                    node.capture(home3=(ch == "h"))
                elif ch == "q":
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print(f"{len(store['samples'])} muestras en {out / 'samples.json'}")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capturar muestras (ROS 2)")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--board-config", default=str(BOARD_CONFIG))
    c.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    c.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    c.add_argument("--base-frame", default="link_base")
    c.add_argument("--tcp-frame", default="xarm_tcp")
    c.add_argument("--camera-link-frame", default="camera_link")
    c.add_argument("--optical-frame", default="camera_color_optical_frame")
    c.add_argument("--min-corners", type=int, default=12)
    c.add_argument("--max-reproj-px", type=float, default=1.0)
    c.add_argument("--still-s", type=float, default=1.0)
    c.add_argument("--still-tol-mm", type=float, default=0.3)
    s = sub.add_parser("solve", help="resolver la calibración (sin ROS)")
    s.add_argument("--samples", required=True, help="carpeta con samples.json")
    s.add_argument("--board-config", default=str(BOARD_CONFIG))
    s.add_argument("--policy-params", default=str(POLICY_PARAMS), help="HOME3 (home_x..home_yaw)")
    s.add_argument("--allow-nominal", action="store_true", help="aceptar medidas nominales del tablero (pruebas)")
    s.add_argument("--max-residual-mm", type=float, default=2.0)
    s.add_argument("--max-residual-deg", type=float, default=0.5)
    s.add_argument("--min-rotation-diversity-deg", type=float, default=30.0)
    args = ap.parse_args(argv)
    return cmd_capture(args) if args.cmd == "capture" else cmd_solve(args)


if __name__ == "__main__":
    sys.exit(main())
