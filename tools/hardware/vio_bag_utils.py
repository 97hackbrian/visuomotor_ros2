"""Utilidades para medir la odometría de la D435i con bags de rosbag2 (MCAP): lectura, imágenes, TF estáticas y poses.

Requiere ROS 2 (``rosbag2_py``, ``rclpy``) con ``source /opt/ros/jazzy/setup.bash``. Lo usan
``evaluate_pose_estimates.py``, ``replay_odometry_candidates.py`` y ``make_synthetic_vio_bag.py``.

Convenciones: cuaterniones ``[qx, qy, qz, qw]``; matrices homogéneas 4x4; ``T_a_b`` lleva puntos del marco ``b`` al
marco ``a`` (pose de ``b`` en ``a``).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Rotaciones y transformaciones
# ---------------------------------------------------------------------------------------------------------------------


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("cuaternión nulo")
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    tr = np.trace(r)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def make_T(r: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    r, t = T[:3, :3], T[:3, 3]
    return make_T(r.T, -r.T @ t)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


# ---------------------------------------------------------------------------------------------------------------------
# Bags
# ---------------------------------------------------------------------------------------------------------------------
def _storage_id(path: Path) -> str:
    return "mcap" if any(Path(path).glob("*.mcap")) or str(path).endswith(".mcap") else "sqlite3"


def bag_topics(path) -> Dict[str, str]:
    """``{tópico: tipo}`` del bag."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=_storage_id(Path(path))),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def iter_messages(path, topics: Iterable[str]) -> Iterator[Tuple[str, object, int]]:
    """``(tópico, mensaje deserializado, tiempo de grabación ns)`` en orden de grabación."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=_storage_id(Path(path))),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in types]
    if not wanted:
        return
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    classes = {t: get_message(types[t]) for t in wanted}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, deserialize_message(data, classes[topic]), t_ns


def stamp_of(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def decode_image(msg) -> np.ndarray:
    """``sensor_msgs/Image`` o ``CompressedImage`` (jpeg/png, y compressedDepth) a un array: RGB uint8 [H, W, 3],
    gris [H, W] o profundidad uint16 [H, W]."""
    import cv2
    if hasattr(msg, "format"):                                  # CompressedImage
        data = bytes(msg.data)
        if "compressedDepth" in msg.format:
            data = data[12:]                                    # cabecera ConfigHeader (formato + 2 floats)
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint16 if enc in ("16uc1", "mono16") else np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step // 3 if msg.step else msg.width, 3)[:, :msg.width]
        return img[:, :, ::-1].copy() if enc == "bgr8" else img.copy()
    if enc in ("mono8", "8uc1"):
        return buf.reshape(msg.height, msg.step if msg.step else msg.width)[:, :msg.width].copy()
    if enc in ("16uc1", "mono16"):
        return buf.reshape(msg.height, (msg.step // 2) if msg.step else msg.width)[:, :msg.width].copy()
    raise ValueError(f"codificación de imagen no soportada: {msg.encoding}")


def encode_compressed_depth(depth_u16: np.ndarray) -> bytes:
    """Formato de ``compressed_depth_image_transport`` para 16UC1: cabecera de 12 bytes y PNG sin pérdida."""
    import cv2
    ok, png = cv2.imencode(".png", depth_u16)
    if not ok:
        raise RuntimeError("no se pudo codificar la profundidad")
    return struct.pack("<iff", 0, 0.0, 0.0) + png.tobytes()


# ---------------------------------------------------------------------------------------------------------------------
# TF estáticas
# ---------------------------------------------------------------------------------------------------------------------
class StaticTF:
    """Árbol de TF estáticas (de ``/tf_static``) con composición entre dos marcos cualesquiera."""

    def __init__(self):
        self.edges: Dict[Tuple[str, str], np.ndarray] = {}      # (padre, hijo) -> T_padre_hijo

    def add_message(self, msg) -> None:
        for tr in msg.transforms:
            q = tr.transform.rotation
            t = tr.transform.translation
            T = make_T(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
            self.edges[(tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))] = T

    def lookup(self, target: str, source: str) -> Optional[np.ndarray]:
        """``T_target_source`` (pose de ``source`` en ``target``) o None si no están conectados."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        adj: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for (a, b), T in self.edges.items():
            adj.setdefault(a, []).append((b, T))
            adj.setdefault(b, []).append((a, inv_T(T)))
        frontier, seen = [(target, np.eye(4))], {target}
        while frontier:
            node, T_t_node = frontier.pop()
            for nxt, T_node_nxt in adj.get(node, []):
                if nxt in seen:
                    continue
                T_t_nxt = T_t_node @ T_node_nxt
                if nxt == source:
                    return T_t_nxt
                seen.add(nxt)
                frontier.append((nxt, T_t_nxt))
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Odometría
# ---------------------------------------------------------------------------------------------------------------------
def odometry_from_bag(path, topic: str) -> Dict[str, np.ndarray]:
    """Poses de ``nav_msgs/Odometry``: ``t``, ``T`` [N, 4, 4] (pose del ``child_frame_id`` en el marco de odometría),
    ``lost`` (odometría nula de RTAB-Map: cuaternión cero o covarianza 9999), ``frame`` y ``child``."""
    ts, Ts, lost = [], [], []
    frame = child = ""
    last = np.eye(4)
    for _, m, _ in iter_messages(path, [topic]):
        o, p = m.pose.pose.orientation, m.pose.pose.position
        qn = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w
        bad = qn < 1e-9 or m.pose.covariance[0] >= 9999.0
        frame, child = m.header.frame_id, m.child_frame_id
        if not bad:
            last = make_T(quat_to_matrix([o.x, o.y, o.z, o.w]), [p.x, p.y, p.z])
        ts.append(stamp_of(m))
        Ts.append(last.copy())
        lost.append(bad)
    order = np.argsort(ts, kind="stable") if ts else np.zeros(0, int)
    return {"t": np.asarray(ts)[order], "T": np.asarray(Ts).reshape(-1, 4, 4)[order],
            "lost": np.asarray(lost, bool)[order], "frame": frame, "child": child}
