#!/usr/bin/env python3
"""Recolector de demostraciones a mano con la RealSense D435i/D415 y la odometría de RTAB-Map (v2).

Cambios respecto a v1 (``context.md`` lección 43; plan ``docs/plans/plan_d435i_herramienta_demostraciones.md``,
Fase 4). La lógica pura está en ``vo_episode_logic.py`` y tiene pruebas en ``tests/test_vo_episode_logic.py``.
* **Sellos de tiempo:** cada frame tiene su instante (``timestamp``). La pose se interpola en una rejilla regular sobre
  los sellos de la odometría, y la imagen es la de sello más cercano (``image_stamp``).
* **Calidad de rastreo:** la odometría nula de RTAB-Map (rastreo perdido) se marca como perdida y nunca se guarda como
  pose. Se usan ``odom_info`` (inliers) y ``min_inliers_tracking``.
* **Reinicio de odometría:** al empezar cada episodio se reinicia (``reset_odom``) con la cámara quieta, y se guarda
  la gravedad que mide el IMU al inicio.
* **Etiqueta done:** vale 1 desde el HOLD (manual o detectado por quietud) y se verifica que ese tramo esté quieto y
  dure lo mínimo. Ya no se fuerza en frames en movimiento.
* **Control de calidad al guardar:** si hay problemas (frames perdidos, imágenes que faltan, HOLD incorrecto o pocos
  inliers) no se guarda; se corrige con [D] o se fuerza con [F]. El resumen de calidad va a los atributos.
* **Grabación cruda** por episodio (rosbag2 MCAP, opcional) para reprocesar la odometría offline.
* **Metadatos:** versión del recolector, commit, sha256 de los YAML, celda del objeto (teclas 1–9).

Controles: [ESPACIO] empezar · [H] llegada (HOLD) · [S] guardar · [F] forzar guardado · [D] descartar ·
[G] gripper · [1-9] celda del objeto (0 = ninguna) · [Q] salir.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState

from visuomotor_msgs.srv import EpisodeTrigger
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage
from visuomotor_data_collection.vo_episode_logic import (
    EpisodeSampler, PoseBuffer, PoseSample, StampedBuffer, check_done_segment, continuous_quat, detect_hold_start,
    episode_quality, is_null_odometry)

COLLECTOR_VERSION = 2
IDLE, ARMING, RECORDING = "IDLE", "ARMING", "RECORDING"


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _git_commit(repo: Path) -> str:
    """Commit actual leyendo ``.git`` (sin ejecutar git: AGENTS.md prohíbe git dentro del contenedor)."""
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = repo / ".git" / ref
            if p.exists():
                return p.read_text().strip()
            packed = repo / ".git" / "packed-refs"
            for line in packed.read_text().splitlines() if packed.exists() else []:
                if line.endswith(" " + ref):
                    return line.split()[0]
            return "unknown"
        return head
    except OSError:
        return "unknown"


class VOCollectorNode(Node):
    def __init__(self):
        super().__init__('vo_collector_node')
        self.bridge = CvBridge()
        self.cb_data = ReentrantCallbackGroup()
        self.cb_ctrl = MutuallyExclusiveCallbackGroup()
        self.lock = threading.RLock()

        d = self.declare_parameter
        d('dataset_path', 'AUTO')
        d('dataset_prefix', 'demonstrations_real')
        d('sampling_rate_hz', 20.0)
        d('camera_topic', '/camera/camera/color/image_raw')
        d('odom_topic', '/rtabmap/odom')
        d('odom_info_topic', '/odom_info')
        d('imu_topic', '/camera/camera/imu')
        d('image_size', 256)
        d('min_hold_duration_s', 2.0)
        d('record_done_flag', True)
        d('min_inliers_tracking', 20)
        d('relative_coordinates', True)
        d('gripper_topic', '/visuomotor/gripper_joint_states')
        d('gripper_closed_val', -0.005)
        d('max_image_dt_s', 0.02)
        d('max_pose_gap_s', 0.10)
        d('image_wait_s', 0.05)
        d('reset_odom_on_start', True)
        d('reset_odom_service', '/reset_odom')
        d('reset_settle_frames', 10)
        d('reset_timeout_s', 3.0)
        d('static_speed_m_s', 0.02)
        d('static_window_s', 0.5)
        d('auto_hold', False)
        d('strict_qa', True)
        d('min_episode_frames', 20)
        d('raw_bag_enabled', False)
        d('raw_bag_dir', 'datasets/raw_bags')
        d('raw_bag_topics', ['/camera/camera/color/image_raw/compressed',
                             '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
                             '/camera/camera/color/camera_info', '/camera/camera/imu', '/rtabmap/odom',
                             '/odom_info', '/tf_static'])
        d('repo_root', '.')
        d('config_files_to_hash', ['visuomotor_bringup/config/realsense/d435i_camera_params.yaml',
                                   'visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml',
                                   'visuomotor_data_collection/config/vo_data_collection_params.yaml'])
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.p = g

        self.storage_path, is_auto = self._resolve_dataset_path(g('dataset_path'), g('dataset_prefix'))
        self.storage = ZarrStorage(self.storage_path)
        n_prev = self.storage.get_num_episodes()
        if not is_auto and n_prev > 0:
            self.get_logger().info(f"MODO AUMENTO: '{self.storage_path}' tiene {n_prev} episodios; los nuevos empiezan "
                                   f"en episode_{n_prev}.")

        self.rate = float(g('sampling_rate_hz'))
        self.poses = PoseBuffer(maxlen=int(30 * 20))
        self.images = StampedBuffer(maxlen=90)
        self.sampler = EpisodeSampler(rate_hz=self.rate, max_image_dt_s=float(g('max_image_dt_s')),
                                      max_pose_gap_s=float(g('max_pose_gap_s')))
        self.state = IDLE
        self.hold = False
        self.current_gripper = 0.0
        self.object_cell = 0
        self.t_reset = None
        self.t_arm = 0.0
        self.odom_reset_ok = False
        self.imu_acc = []
        self.last_q = None
        self.last_p = None
        self.n_null = 0
        self.pending_problems = None
        self.repo = Path(g('repo_root')).resolve()
        self.raw_bag = None
        if g('raw_bag_enabled'):
            from visuomotor_data_collection.vo_raw_recorder import RawEpisodeBag
            self.raw_bag = RawEpisodeBag(self, list(g('raw_bag_topics')))

        self.create_subscription(Image, g('camera_topic'), self._img_cb, 10, callback_group=self.cb_data)
        self.create_subscription(Odometry, g('odom_topic'), self._odom_cb, 50, callback_group=self.cb_data)
        self.create_subscription(Imu, g('imu_topic'), self._imu_cb, 50, callback_group=self.cb_data)
        self.create_subscription(JointState, g('gripper_topic'), self._gripper_cb, 10, callback_group=self.cb_data)
        try:
            from rtabmap_msgs.msg import OdomInfo
            self.create_subscription(OdomInfo, g('odom_info_topic'), self._odom_info_cb, 50,
                                     callback_group=self.cb_data)
        except ImportError:
            self.get_logger().warn("rtabmap_msgs no disponible: sin inliers por frame.")
        self.reset_client = None
        if g('reset_odom_on_start'):
            from std_srvs.srv import Empty
            self.reset_client = self.create_client(Empty, g('reset_odom_service'), callback_group=self.cb_ctrl)
        self.create_service(EpisodeTrigger, '~/trigger_episode', self._srv_trigger, callback_group=self.cb_ctrl)
        self.create_timer(1.0 / (2.0 * self.rate), self._tick, callback_group=self.cb_ctrl)
        self.create_timer(1.0, self._status, callback_group=self.cb_ctrl)
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        self.get_logger().info("=" * 72)
        self.get_logger().info(f" VO Collector v{COLLECTOR_VERSION} | {self.rate} Hz | {g('image_size')} px | "
                               f"{self.storage_path} ({n_prev} episodios)")
        self.get_logger().info(" [ESPACIO] empezar  [H] llegada  [S] guardar  [F] forzar  [D] descartar  [G] gripper  "
                               "[1-9] celda  [Q] salir")
        self.get_logger().info("=" * 72)

    # ------------------------------------------------------------------ dataset
    def _resolve_dataset_path(self, raw_path, prefix):
        base_dir = "datasets"
        os.makedirs(base_dir, exist_ok=True)
        raw = str(raw_path).strip() if raw_path is not None else ""
        if not raw or raw.upper() in ('AUTO', 'NONE', 'DEFAULT', '""', "''"):
            max_v = 0
            for p in glob.glob(os.path.join(base_dir, f"{prefix}_v*.zarr")):
                m = re.search(rf"{prefix}_v(\d+)\.zarr", p)
                if m:
                    max_v = max(max_v, int(m.group(1)))
            path = os.path.join(base_dir, f"{prefix}_v{max_v + 1}.zarr")
            self.get_logger().info(f"Modo AUTO: nuevo dataset '{path}'")
            return path, True
        clean = raw if raw.endswith('.zarr') else f"{raw}.zarr"
        if not os.path.isabs(clean) and not clean.startswith("datasets/"):
            clean = os.path.join(base_dir, clean)
        return clean, False

    # ------------------------------------------------------------------ callbacks de datos
    def _img_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            s = int(self.p('image_size'))
            if img.shape[0] != s or img.shape[1] != s:
                img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.images.add(_stamp(msg), img)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Imagen: {e}", throttle_duration_sec=2.0)

    def _odom_cb(self, msg):
        t = _stamp(msg)
        o, pos = msg.pose.pose.orientation, msg.pose.pose.position
        cov = msg.pose.covariance
        q = [o.x, o.y, o.z, o.w]
        with self.lock:
            if is_null_odometry(q, [cov[0], cov[7], cov[14], cov[21], cov[28], cov[35]]):
                self.n_null += 1
                if self.last_p is not None:                     # se conserva la última pose, marcada como perdida
                    self.poses.add(PoseSample(t, self.last_p.copy(), self.last_q.copy(), lost=True))
                return
            qc = continuous_quat(q, self.last_q)
            p = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            self.last_p, self.last_q = p, qc
            self.poses.add(PoseSample(t, p, qc))

    def _odom_info_cb(self, msg):
        with self.lock:
            low = msg.inliers < int(self.p('min_inliers_tracking'))
            self.poses.set_inliers(_stamp(msg), msg.inliers, bool(msg.lost))
            if low and self.state == RECORDING:
                self.get_logger().warn(f"Pocos inliers: {msg.inliers}", throttle_duration_sec=1.0)

    def _imu_cb(self, msg):
        if self.state in (ARMING,) or (self.state == RECORDING and len(self.imu_acc) < 40):
            a = msg.linear_acceleration
            self.imu_acc.append((a.x, a.y, a.z))
            self.imu_frame = msg.header.frame_id

    def _gripper_cb(self, msg: JointState):
        if msg.position:
            self.current_gripper = float(msg.position[0])

    # ------------------------------------------------------------------ máquina de estados
    def _start(self):
        with self.lock:
            if self.state != IDLE:
                self.get_logger().warn(f"Ya hay un episodio en curso ({self.state}).")
                return False
            if self.poses.latest() is None or len(self.images) == 0:
                self.get_logger().warn("No se puede empezar: falta imagen u odometría.")
                return False
            self.hold, self.current_gripper, self.pending_problems = False, 0.0, None
            self.imu_acc, self.imu_frame, self.n_null = [], "", 0
            self.state, self.t_arm, self.t_reset, self.odom_reset_ok = ARMING, time.time(), None, False
        if self.reset_client is not None:
            if self.reset_client.service_is_ready():
                from std_srvs.srv import Empty
                fut = self.reset_client.call_async(Empty.Request())
                fut.add_done_callback(self._reset_done)
                self.get_logger().info(">>> Reiniciando la odometría: mantener la cámara QUIETA en la pose de inicio...")
                return True
            self.get_logger().warn(f"Servicio {self.p('reset_odom_service')} no disponible: se empieza SIN reiniciar la "
                                   "odometría (queda registrado en los atributos).")
        with self.lock:
            self.t_reset = self.poses.latest_stamp
        return True

    def _reset_done(self, _fut):
        # Base de tiempo de la cámara (sellos de la odometría), no el reloj del nodo: así no depende de que la
        # RealSense use el tiempo global del host.
        with self.lock:
            self.t_reset = self.poses.latest_stamp
            self.odom_reset_ok = True

    def _tick(self):
        with self.lock:
            if self.state == ARMING:
                if self.t_reset is None:
                    if time.time() - self.t_arm > float(self.p('reset_timeout_s')):
                        self.get_logger().error("El reinicio de la odometría no respondió: episodio cancelado.")
                        self.state = IDLE
                    return
                if time.time() - self.t_arm > 2.0 * float(self.p('reset_timeout_s')) + 2.0:
                    self.get_logger().error("La odometría no entregó poses válidas tras el reinicio (¿rastreo "
                                            "perdido?): episodio cancelado.")
                    self.state = IDLE
                    return
                if self.poses.valid_after(self.t_reset) >= int(self.p('reset_settle_frames')):
                    t0 = self.poses.latest_stamp
                    self.sampler.start(t0)
                    self.state = RECORDING
                    if self.raw_bag is not None:
                        stem = Path(self.storage_path).stem
                        path = Path(self.p('raw_bag_dir')) / stem / time.strftime("ep_%Y%m%d_%H%M%S")
                        rec = self.raw_bag.start(path)
                        self.get_logger().info(f"[RAW BAG] {len(rec)} tópicos -> {path}")
                    self.get_logger().info(">>> [REC] Grabando. Mover la cámara hacia el objetivo; [H] al llegar.")
                return
            if self.state == RECORDING:
                self.sampler.step(self.poses, self.images, self.current_gripper, self.hold,
                                  image_wait_s=float(self.p('image_wait_s')))

    def _status(self):
        with self.lock:
            if self.state != RECORDING or not self.sampler.frames:
                return
            fr = self.sampler.frames
            lost = sum(f.lost for f in fr)
            inl = [f.inliers for f in fr[-20:] if f.inliers >= 0]
            p = np.array([f.p for f in fr[-11:]])
            v = np.linalg.norm(p[-1] - p[0]) / max(1e-6, (len(p) - 1) / self.rate) if len(p) > 1 else 0.0
        msg = (f"[REC] {len(fr) / self.rate:5.1f} s | perdidos {lost} | inliers {min(inl) if inl else -1} | "
               f"v {v * 100:4.1f} cm/s{' | HOLD' if self.hold else ''}{f' | celda {self.object_cell}' if self.object_cell else ''}")
        (self.get_logger().warn if lost else self.get_logger().info)(msg)

    def _hold(self):
        if self.state != RECORDING:
            self.get_logger().warn("HOLD solo durante la grabación.")
            return
        self.hold = True
        self.get_logger().info(f">>> [HOLD] Mantener la cámara QUIETA al menos {self.p('min_hold_duration_s')} s y "
                               "luego [S].")

    def _discard(self):
        with self.lock:
            n = len(self.sampler.frames)
            self.state, self.hold, self.pending_problems = IDLE, False, None
            self.sampler.start(None)
            self.sampler.t_next = None
        if self.raw_bag is not None:
            self.raw_bag.stop(delete=True)
        self.get_logger().info(f"[DISCARD] Episodio descartado ({n} frames).")
        return True, "Descartado"

    def _save(self, force: bool = False):
        with self.lock:
            if self.state != RECORDING:
                return False, "No hay grabación activa"
            # Completar los frames cuya pose ya llegó
            self.sampler.step(self.poses, self.images, self.current_gripper, self.hold, image_wait_s=0.0)
            frames = list(self.sampler.frames)
        if len(frames) < int(self.p('min_episode_frames')):
            self._discard()
            return False, f"Episodio demasiado corto ({len(frames)} frames): descartado"
        p_abs = np.array([f.p for f in frames])
        hold = np.array([f.hold for f in frames])
        auto = False
        if self.p('auto_hold') and not hold.any():
            i = detect_hold_start(p_abs, self.rate, float(self.p('static_speed_m_s')), float(self.p('static_window_s')))
            if i is not None:
                hold[i:] = True
                auto = True
        chk = check_done_segment(p_abs, hold, self.rate, float(self.p('min_hold_duration_s')),
                                 float(self.p('static_speed_m_s')), float(self.p('static_window_s')))
        qual = episode_quality(frames, self.rate)
        problems = list(chk["problems"])
        if qual["lost_frames"]:
            problems.append(f"{qual['lost_frames']} frames con rastreo perdido o sin imagen")
        if qual["duplicate_images"]:
            problems.append(f"{qual['duplicate_images']} imágenes repetidas en frames consecutivos")
        min_inl = int(self.p('min_inliers_tracking'))
        low = sum(1 for f in frames if 0 <= f.inliers < min_inl)
        if low:
            problems.append(f"{low} frames con menos de {min_inl} inliers")
        if problems and self.p('strict_qa') and not force:
            for pr in problems:
                self.get_logger().error(f"[QA] {pr}")
            self.get_logger().error("[QA] Episodio NO guardado: [D] descartar y repetir, o [F] forzar el guardado "
                                    "(queda marcado en los atributos).")
            return False, "QA: " + "; ".join(problems)

        rel = p_abs - p_abs[0] if self.p('relative_coordinates') else p_abs
        q = np.array([f.q for f in frames])
        grip = np.array([f.gripper for f in frames], dtype=np.float32)
        state = np.zeros((len(frames), 8), dtype=np.float32)
        state[:, 0:3], state[:, 3:7], state[:, 7] = rel, q, grip
        if self.p('record_done_flag'):
            action = np.zeros((len(frames), 9), dtype=np.float32)
            action[:, 8] = chk["done"]
        else:
            action = np.zeros((len(frames), 8), dtype=np.float32)
        action[:, 0:8] = state[:, 0:8]
        buffer = [{"image": f.image, "state": s, "action": a} for f, s, a in zip(frames, state, action)]
        extra = {
            "timestamp": np.array([f.timestamp for f in frames], dtype=np.float64),
            "image_stamp": np.array([f.image_stamp for f in frames], dtype=np.float64),
            "pose_stamp": np.array([f.pose_stamp for f in frames], dtype=np.float64),
            "inliers": np.array([f.inliers for f in frames], dtype=np.int32),
            "lost": np.array([f.lost for f in frames], dtype=bool),
        }
        bag_path = self.raw_bag.stop(delete=False) if self.raw_bag is not None else None
        g = np.mean(np.array(self.imu_acc), axis=0).tolist() if self.imu_acc else None
        attrs = {
            "collector_version": COLLECTOR_VERSION,
            "sensor": "Intel RealSense", "odometry": "RTAB-Map rgbd_odometry",
            "sampling_rate_hz": self.rate, "total_steps": len(frames), "hold_steps": int(chk["hold_frames"]),
            "hold_detected_auto": auto, "timestamp": time.time(),
            "relative_coordinates": bool(self.p('relative_coordinates')),
            "initial_pos_x": float(p_abs[0, 0]), "initial_pos_y": float(p_abs[0, 1]), "initial_pos_z": float(p_abs[0, 2]),
            "initial_quat_xyzw": q[0].tolist(),
            "odom_reset": bool(self.odom_reset_ok), "null_odometry_messages": int(self.n_null),
            "start_gravity_mps2": g, "start_gravity_frame": getattr(self, "imu_frame", ""),
            "quality": qual, "qa_problems": problems, "qa_forced": bool(force and problems),
            "object_cell": int(self.object_cell), "raw_bag": None if bag_path is None else str(bag_path),
            "git_commit": _git_commit(self.repo), "config_sha256": self._config_hashes(),
        }
        ep_idx, total = self.storage.append_episode(buffer, attrs=attrs, extra_arrays=extra, episode_chunks=True)
        with self.lock:
            self.state, self.hold = IDLE, False
            self.sampler.t_next = None
        self.get_logger().info(f"[SAVE] episode_{ep_idx}: {len(frames)} frames ({len(frames) / self.rate:.1f} s), "
                               f"done {chk['hold_frames']} frames{' (auto)' if auto else ''}"
                               f"{', FORZADO con problemas' if force and problems else ''}. Total {total}.")
        return True, f"Guardado episode_{ep_idx}"

    def _config_hashes(self):
        out = {}
        for rel in self.p('config_files_to_hash'):
            path = self.repo / rel
            try:
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                out[rel] = None
        return out

    def _toggle_gripper(self):
        closed = float(self.p('gripper_closed_val'))
        self.current_gripper = closed if abs(self.current_gripper) < 1e-4 else 0.0
        self.get_logger().info(f"[GRIPPER] {self.current_gripper:.4f}")

    # ------------------------------------------------------------------ control
    def _srv_trigger(self, req, res):
        cmd = req.command.strip().upper()
        if cmd == "START":
            res.success = self._start()
            res.message = "Grabación solicitada" if res.success else "No se pudo empezar"
        elif cmd == "HOLD":
            self._hold()
            res.success, res.message = self.state == RECORDING, "Hold"
        elif cmd in ("GRIPPER", "TOGGLE_GRIPPER"):
            self._toggle_gripper()
            res.success, res.message = True, f"Gripper={self.current_gripper}"
        elif cmd in ("SAVE", "FORCE_SAVE"):
            res.success, res.message = self._save(force=(cmd == "FORCE_SAVE"))
        elif cmd == "DISCARD":
            res.success, res.message = self._discard()
        elif cmd.startswith("CELL"):
            try:
                self.object_cell = int(cmd[4:].strip() or 0)
                res.success, res.message = True, f"Celda {self.object_cell}"
            except ValueError:
                res.success, res.message = False, "Formato: CELL <n>"
        elif cmd == "STATUS":
            res.success, res.message = True, f"{self.state} frames={len(self.sampler.frames)}"
        else:
            res.success, res.message = False, f"Comando desconocido: {req.command}"
        return res

    def _keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().info("stdin no es una terminal: usar el servicio ~/trigger_episode.")
            return
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                ch = sys.stdin.read(1).lower()
                if ch == ' ':
                    self._start()
                elif ch == 'h':
                    self._hold()
                elif ch == 's':
                    self._save()
                elif ch == 'f':
                    self._save(force=True)
                elif ch == 'd':
                    self._discard()
                elif ch == 'g':
                    self._toggle_gripper()
                elif ch.isdigit():
                    self.object_cell = int(ch)
                    self.get_logger().info(f"Celda del objeto: {self.object_cell or 'ninguna'}")
                elif ch == 'q':
                    rclpy.try_shutdown()
                    break
        except Exception:  # noqa: BLE001
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = VOCollectorNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.raw_bag is not None and node.raw_bag.active:
            node.raw_bag.stop(delete=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
