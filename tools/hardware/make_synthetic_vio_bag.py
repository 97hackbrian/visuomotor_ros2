#!/usr/bin/env python3
"""Bag sintético (MCAP) de una aproximación con la D435i sobre una mesa con el tablero ChArUco, con verdad conocida.

Sirve para probar la cadena de medición sin cámara (``evaluate_pose_estimates.py`` y
``replay_odometry_candidates.py``). Contenido, con los mismos tópicos que graba la cámara real:
* color 640x480 JPEG (o crudo) y ``camera_info``;
* profundidad alineada en mm (cruda, o compressedDepth);
* IMU a 200 Hz y ``/tf_static`` con la extrínseca típica de la RealSense;
* ``/rtabmap/odom``: odometría "en vivo" simulada a partir de la verdad, con error de escala, deriva y un tramo de
  rastreo perdido (odometría nula de RTAB-Map);
* ``/ground_truth/optical_pose``: verdad del marco óptico de color en el marco de la mesa.

La escena es un plano texturado (z = 0) con el tablero en el centro. La cámara empieza alta e inclinada, como en
HOME3, y baja hacia la mesa.

Uso (en el contenedor, con ROS 2 cargado):
    python3 tools/hardware/make_synthetic_vio_bag.py --out /tmp/synthetic_run --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "hardware"))
sys.path.insert(0, str(REPO / "visuomotor_bringup"))

from vio_bag_utils import encode_compressed_depth, inv_T, make_T, matrix_to_quat  # noqa: E402

K = np.array([[605.0, 0, 320.0], [0, 605.0, 240.0], [0, 0, 1.0]])
W, H = 640, 480
RES = 0.0005                                     # m por píxel de textura (2 px/mm)
TEX_W, TEX_H = 2400, 1800                        # 1,2 m x 0,9 m
X0, Y0 = -0.6, 0.45                              # esquina superior izquierda de la textura en el marco de la mesa
R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # columnas: ejes ópticos en camera_link
T_LINK_OPT = make_T(R_LINK_OPT, [0.0, 0.015, 0.0])


def texture(seed: int, board_config=None) -> np.ndarray:
    """Mantel con textura aleatoria (esquinas para GFTT/ORB) y el tablero ChArUco nominal en el centro."""
    import cv2
    from visuomotor_bringup.charuco import CharucoConfig, board_image, make_board
    rng = np.random.default_rng(seed)
    tex = np.full((TEX_H, TEX_W), 205, np.uint8)
    noise = cv2.GaussianBlur(rng.normal(0, 25, (TEX_H, TEX_W)).astype(np.float32), (0, 0), 3)
    tex = np.clip(tex + noise, 0, 255).astype(np.uint8)
    for _ in range(700):
        c = int(rng.integers(40, 160))
        x, y = int(rng.integers(0, TEX_W)), int(rng.integers(0, TEX_H))
        if rng.random() < 0.5:
            w, h = int(rng.integers(8, 60)), int(rng.integers(8, 60))
            cv2.rectangle(tex, (x, y), (x + w, y + h), c, -1)
        else:
            cv2.circle(tex, (x, y), int(rng.integers(5, 30)), c, -1)
    cfg = CharucoConfig.load() if board_config is None else CharucoConfig.load(board_config)
    bw, bh = (int(round(v / RES)) for v in cfg.size_m)
    board = board_image(make_board(cfg), (bw, bh))
    x, y = TEX_W // 2 - bw // 2, TEX_H // 2 - bh // 2 - 160      # centro del tablero en (0, +0,08 m) de la mesa
    tex[y:y + bh, x:x + bw] = board
    return tex


def look_at(pos, target) -> np.ndarray:
    """``R_mesa_óptico``: z hacia el objetivo; arriba de la imagen hacia +y de la mesa (lejos de la base)."""
    z = np.asarray(target, float) - np.asarray(pos, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def trajectory(rate: float, t_still0: float = 1.0, t_move: float = 6.0, t_still1: float = 1.5):
    """Poses ``T_mesa_óptico`` a ``rate`` Hz: quieta, aproximación de mínimo jerk con giro, quieta."""
    n = int(round((t_still0 + t_move + t_still1) * rate))
    t = np.arange(n) / rate
    s = np.clip((t - t_still0) / t_move, 0, 1)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    p0, p1 = np.array([0.0, -0.30, 0.70]), np.array([0.06, 0.02, 0.32])
    g0, g1 = np.array([0.0, 0.12, 0.0]), np.array([0.07, 0.10, 0.0])
    Ts = []
    for k in range(n):
        pos = p0 + s[k] * (p1 - p0)
        tgt = g0 + s[k] * (g1 - g0)
        r = look_at(pos, tgt)
        roll = 0.15 * np.sin(np.pi * s[k])                         # giro de muñeca a mitad de camino
        cr, sr = np.cos(roll), np.sin(roll)
        r = r @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        Ts.append(make_T(r, pos))
    return t, np.stack(Ts)


def render(tex: np.ndarray, T_w_c: np.ndarray):
    """Imagen (gris) y profundidad (mm, uint16) del plano z = 0 visto desde ``T_mesa_óptico``."""
    import cv2
    T_c_w = inv_T(T_w_c)
    r, t = T_c_w[:3, :3], T_c_w[:3, 3]
    a = np.array([[RES, 0, X0], [0, -RES, Y0], [0, 0, 1.0]])         # píxel de textura -> (x, y, 1) de la mesa
    hm = K @ np.c_[r[:, 0], r[:, 1], t] @ a
    img = cv2.warpPerspective(tex, hm, (W, H), flags=cv2.INTER_AREA, borderValue=128)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], axis=-1)
    n_c = r[:, 2]
    den = rays @ n_c
    z = np.where(np.abs(den) > 1e-9, (n_c @ t) / den, 0.0)
    depth = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
    return img, depth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--color", choices=["compressed", "raw"], default="compressed")
    ap.add_argument("--depth", choices=["raw", "compressedDepth"], default="raw")
    ap.add_argument("--live-scale", type=float, default=1.03, help="escala de la odometría en vivo simulada")
    ap.add_argument("--live-lost", default="3.0:3.3", help="tramo 'a:b' (s) con odometría nula; '' = ninguno")
    ap.add_argument("--t0", type=float, default=1000.0, help="sello del primer frame (s)")
    ap.add_argument("--board-config", default=None, help="YAML del tablero (por defecto charuco_board.yaml)")
    a = ap.parse_args(argv)

    import cv2
    import rosbag2_py
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
    from tf2_msgs.msg import TFMessage

    out = Path(a.out)
    if out.exists():
        print(f"ERROR: {out} ya existe")
        return 1
    tex = texture(a.seed, a.board_config)
    t, T = trajectory(a.rate)
    rng = np.random.default_rng(a.seed + 1)

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(out), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    color_topic = "/camera/camera/color/image_raw" + ("/compressed" if a.color == "compressed" else "")
    depth_topic = "/camera/camera/aligned_depth_to_color/image_raw" + (
        "/compressedDepth" if a.depth == "compressedDepth" else "")
    topics = {
        color_topic: "sensor_msgs/msg/CompressedImage" if a.color == "compressed" else "sensor_msgs/msg/Image",
        depth_topic: "sensor_msgs/msg/CompressedImage" if a.depth == "compressedDepth" else "sensor_msgs/msg/Image",
        "/camera/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/camera/camera/imu": "sensor_msgs/msg/Imu",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/rtabmap/odom": "nav_msgs/msg/Odometry",
        "/ground_truth/optical_pose": "geometry_msgs/msg/PoseStamped",
    }
    for i, (name, typ) in enumerate(topics.items()):
        try:
            w.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr"))
        except TypeError:
            w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    def stamp(sec: float) -> Time:
        return Time(sec=int(sec), nanosec=int(round((sec - int(sec)) * 1e9)) % 1_000_000_000)

    def write(topic, msg, sec):
        w.write(topic, serialize_message(msg), int(round(sec * 1e9)))

    # TF estáticas
    tfm = TFMessage()
    for child in ("camera_color_optical_frame", "camera_imu_optical_frame"):
        tr = TransformStamped()
        tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp(a.t0 - 0.5), "camera_link", child
        q = matrix_to_quat(T_LINK_OPT[:3, :3])
        tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = T_LINK_OPT[:3, 3]
        tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = q
        tfm.transforms.append(tr)
    write("/tf_static", tfm, a.t0 - 0.5)

    # IMU a 200 Hz en el marco óptico (= camera_imu_optical_frame), desde la trayectoria
    t_imu = np.arange(t[0] - 0.5, t[-1], 1.0 / 200.0)
    p = T[:, :3, 3]
    pi = np.stack([np.interp(t_imu, t, p[:, k]) for k in range(3)], axis=1)
    acc_w = np.gradient(np.gradient(pi, t_imu, axis=0), t_imu, axis=0)
    omega = []                                              # velocidad angular (marco del cuerpo) por intervalo
    for k in range(len(t) - 1):
        dr = T[k, :3, :3].T @ T[k + 1, :3, :3]
        omega.append(np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) * 0.5 * a.rate)
    for k, ti in enumerate(t_imu):
        j = int(np.clip(np.searchsorted(t, ti) - 1, 0, len(t) - 2))
        r = T[j, :3, :3]
        f = r.T @ (acc_w[k] + np.array([0.0, 0.0, 9.81]))
        wv = omega[j]
        m = Imu()
        m.header.stamp, m.header.frame_id = stamp(a.t0 + ti), "camera_imu_optical_frame"
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (f + rng.normal(0, 0.02, 3))
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (wv + rng.normal(0, 0.002, 3))
        m.orientation_covariance[0] = -1.0
        write("/camera/camera/imu", m, a.t0 + ti)

    # Odometría "en vivo": escala, deriva lenta y un tramo perdido. Marco odom = camera_link del primer frame.
    T_w_link = T @ inv_T(T_LINK_OPT)
    lost_a, lost_b = (float(v) for v in a.live_lost.split(":")) if a.live_lost else (1e9, 1e9)
    p0 = T_w_link[0, :3, 3]
    info = CameraInfo()
    info.header.frame_id, info.width, info.height, info.distortion_model = "camera_color_optical_frame", W, H, "plumb_bob"
    info.k = K.ravel().tolist()
    info.d = [0.0] * 5
    info.r = np.eye(3).ravel().tolist()
    info.p = np.c_[K, np.zeros(3)].ravel().tolist()
    for k, tk in enumerate(t):
        sec = a.t0 + tk
        img, depth = render(tex, T[k])
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if a.color == "compressed":
            cm = CompressedImage()
            cm.header.stamp, cm.header.frame_id, cm.format = stamp(sec), "camera_color_optical_frame", "rgb8; jpeg compressed bgr8"
            cm.data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            write(color_topic, cm, sec)
        else:
            im = Image()
            im.header.stamp, im.header.frame_id = stamp(sec), "camera_color_optical_frame"
            im.height, im.width, im.encoding, im.step = H, W, "rgb8", W * 3
            im.data = rgb.tobytes()
            write(color_topic, im, sec)
        if a.depth == "compressedDepth":
            dm = CompressedImage()
            dm.header.stamp, dm.header.frame_id = stamp(sec), "camera_color_optical_frame"
            dm.format = "16UC1; compressedDepth png"
            dm.data = encode_compressed_depth(depth)
            write(depth_topic, dm, sec)
        else:
            dm = Image()
            dm.header.stamp, dm.header.frame_id = stamp(sec), "camera_color_optical_frame"
            dm.height, dm.width, dm.encoding, dm.step = H, W, "16UC1", W * 2
            dm.data = depth.tobytes()
            write(depth_topic, dm, sec)
        info.header.stamp = stamp(sec)
        write("/camera/camera/color/camera_info", info, sec)
        gt = PoseStamped()
        gt.header.stamp, gt.header.frame_id = stamp(sec), "table"
        gt.pose.position.x, gt.pose.position.y, gt.pose.position.z = T[k, :3, 3]
        q = matrix_to_quat(T[k, :3, :3])
        gt.pose.orientation.x, gt.pose.orientation.y, gt.pose.orientation.z, gt.pose.orientation.w = q
        write("/ground_truth/optical_pose", gt, sec)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp(sec), "odom", "camera_link"
        if lost_a <= tk <= lost_b:
            od.pose.covariance = [9999.0 if i % 7 == 0 else 0.0 for i in range(36)]   # odometría nula
        else:
            Tl = T_w_link[k].copy()
            Tl[:3, 3] = p0 + a.live_scale * (Tl[:3, 3] - p0) + np.array([0.0005, 0.0, 0.0]) * tk
            To = inv_T(T_w_link[0]) @ Tl
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = To[:3, 3]
            qo = matrix_to_quat(To[:3, :3])
            (od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z,
             od.pose.pose.orientation.w) = qo
            od.pose.covariance = [1e-4 if i % 7 == 0 else 0.0 for i in range(36)]
        write("/rtabmap/odom", od, sec)
    del w
    print(f"{len(t)} frames ({t[-1]:.1f} s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
