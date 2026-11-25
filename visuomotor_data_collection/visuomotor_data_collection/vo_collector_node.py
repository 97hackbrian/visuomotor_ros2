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

Cambios de v3 (2026-11-24, tras los 14 intentos en el xArm5; ``vo_start_guide.py`` y ``vo_schedule.py``):
* **Guía de inicio a HOME3:** compara los ArUco de la mesa con la vista que tiene la cámara del robot en HOME3. Publica
  una imagen de guía (``~/start_guide``, verla con rqt_image_view) y, con ``start_guide_required``, no deja empezar
  fuera de tolerancia ([I] empieza igual y queda registrado). Los episodios guardan ``start_view``.
* **Inicio y guardado automáticos (opcionales):** ``auto_start`` empieza cuando la vista lleva ``auto_start_hold_s``
  alineada y quieta, sin tocar el teclado (tocarlo mueve la cámara). ``auto_save`` guarda cuando la cámara ya se alejó
  del inicio y lleva quieta al final ``min_hold_duration_s`` + margen.
* **Velocidad:** aviso por encima de ``max_demo_speed_m_s``; la máxima y la media van a los atributos.
* **Programa de recolección** (``schedule_file``): muestra la siguiente celda y orientación de la herramienta y la
  guarda en el episodio ([N] salta una entrada).

Controles: [ESPACIO] empezar · [I] empezar ignorando la guía · [H] llegada (HOLD) · [S] guardar · [F] forzar guardado
· [D] descartar · [U] descartar la última demo guardada (la marca, no la borra) · [G] gripper · [1-9] celda del objeto (0 = ninguna) · [N] saltar entrada del programa · [Q] salir.
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
from collections import deque
from datetime import datetime
import tty
from pathlib import Path
from typing import Optional

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
    EpisodeSampler, PoseBuffer, PoseSample, StampedBuffer, auto_save_ready, check_done_segment, continuous_quat,
    imu_still_since,
    detect_hold_start, episode_quality, is_null_odometry, max_travel, smoothed_speeds)
from visuomotor_data_collection.vo_schedule import (
    YAW_CONVENTION, cells_from_edges, done_indices, load_csv as load_schedule, next_pending, placement_hint,
    schedule_id, yaw_words)
from visuomotor_data_collection.vo_start_guide import (
    SteadyAlignment, ViewReference, detect_markers, draw_guide, summary_line, view_alignment)

COLLECTOR_VERSION = 3
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
        d('reset_odom_service', '/rgbd_odometry/reset_odom')   # servicio privado de rgbd_odometry
        d('reset_settle_frames', 10)
        d('reset_timeout_s', 3.0)
        d('static_speed_m_s', 0.02)
        d('static_window_s', 0.5)
        d('auto_hold', False)
        d('strict_qa', True)
        # False (v8, con bag crudo: las poses del dataset salen de DROID offline): las pérdidas de la odometría EN VIVO
        # quedan como aviso (qa_warnings) y no bloquean el guardado.
        d('qa_tracking_blocking', True)
        # Quietud final por giróscopo cuando la odometría en vivo está perdida al final (la pose se congela).
        d('imu_still_gyro_rad_s', 0.08)
        d('imu_still_smooth_s', 0.15)
        d('manual_save_trim_s', 0.3)          # [S]/[F]: se quita el movimiento de apretar la tecla
        # Tramo done "quieto" si a lo más esta fracción de sus frames supera static_speed_m_s (picos de la mano y del
        # ruido de la odometría cerca de la mesa). 0 = ninguno, como antes del 2026-11-25.
        d('qa_hold_max_moving_frac', 0.2)
        # [S] es la decisión del operador: los problemas del tramo final (corto o con movimiento) quedan como avisos y
        # la demo se guarda. Si no hay ningún tramo quieto, el done se pone en los últimos min_hold_duration_s.
        d('manual_save_overrides_hold_qa', True)
        d('min_episode_frames', 20)
        d('raw_bag_enabled', False)
        d('raw_bag_dir', 'datasets/raw_bags')
        d('raw_bag_preroll_arming', True)   # el bag empieza en ARMING (cámara quieta): anclaje de inicio
        d('raw_bag_topics', ['/camera/camera/color/image_raw/compressed',
                             '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
                             '/camera/camera/color/camera_info', '/camera/camera/imu', '/rtabmap/odom',
                             '/odom_info', '/tf_static'])
        d('repo_root', '.')
        d('config_files_to_hash', ['visuomotor_bringup/config/realsense/d435i_camera_params.yaml',
                                   'visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml',
                                   'visuomotor_data_collection/config/vo_data_collection_params.yaml'])
        # v3: guía de inicio a HOME3, inicio/guardado automáticos, velocidad y programa de recolección
        d('start_guide_enabled', True)
        d('start_guide_reference', 'visuomotor_bringup/config/realsense/home3_view_reference.yaml')
        d('start_guide_tolerance_px', 8.0)
        d('start_guide_min_markers', 2)
        d('start_guide_required', True)
        d('start_guide_rate_hz', 5.0)
        d('start_guide_image_topic', '~/start_guide')
        d('end_view_box_frac', 0.5)
        d('auto_start', False)
        d('auto_start_hold_s', 1.0)
        d('auto_start_max_motion_px', 3.0)
        d('auto_start_cooldown_s', 3.0)
        d('auto_save', False)
        d('auto_save_min_travel_m', 0.10)
        d('auto_save_margin_s', 0.5)
        d('max_demo_speed_m_s', 0.15)
        d('terminal_bell', True)
        d('schedule_file', '')
        d('schedule_cells_file', 'visuomotor_bringup/config/realsense/v8_cells_home3.yaml')
        # Grilla en vivo (2026-11-25): celdas en metros y ArUco de la mesa en link_base; la guía la proyecta sobre la
        # mesa con la homografía de los marcadores detectados. Vacío = guía anterior (polígono de HOME3).
        d('schedule_cells_m_file', 'visuomotor_bringup/config/realsense/v8_cells.yaml')
        d('table_markers_file', 'visuomotor_bringup/config/realsense/table_markers.yaml')
        # Programa al que pertenecen los episodios grabados antes de guardar schedule_id (2026-11-25)
        d('schedule_legacy_id', 'v8_schedule')
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
        self.last_saved = None                  # (episode_N, índice del programa) para [U]
        # Sello del último mensaje de odometría, VÁLIDO O NULO. Si RTAB-Map ya estaba perdido al arrancar el
        # recolector, no hay ninguna pose válida guardada, pero sí llegan mensajes; el reinicio en ARMING la recupera.
        self.last_odom_stamp = None
        self.t_arm = 0.0
        self.odom_reset_ok = False
        self.imu_acc = []
        self.imu_buf = deque(maxlen=4000)                   # (t, wx, wy, wz): quietud final por giróscopo
        self.imu_hold_start = None
        self.last_q = None
        self.last_p = None
        self.n_null = 0
        self.pending_problems = None
        self.saving = False
        self.repo = Path(g('repo_root')).resolve()
        self.raw_bag = None
        if g('raw_bag_enabled'):
            from visuomotor_data_collection.vo_raw_recorder import RawEpisodeBag
            self.raw_bag = RawEpisodeBag(self, list(g('raw_bag_topics')))

        # Guía de inicio a HOME3 (v3)
        self.guide_ref = None
        self.guide_res = None                     # (time.time(), resultado de view_alignment)
        self.latest_full = None                   # (sello, imagen RGB a resolución completa)
        self.start_view = None
        self.steady = SteadyAlignment(float(g('auto_start_hold_s')), float(g('auto_start_max_motion_px')))
        self.t_idle_since = time.time()
        self._last_guide_print = 0.0
        self._last_guide_aligned = None
        self.auto_save_blocked = False
        if g('start_guide_enabled'):
            ref_path = Path(g('start_guide_reference'))
            ref_path = ref_path if ref_path.is_absolute() else self.repo / ref_path
            try:
                self.guide_ref = ViewReference.load(ref_path)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Guía de inicio DESACTIVADA: no se pudo leer {ref_path} ({e}).")
        self.guide_pub = self.create_publisher(Image, g('start_guide_image_topic'), 2)
        self.cb_guide = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / float(g('start_guide_rate_hz')), self._guide_tick, callback_group=self.cb_guide)

        # Programa de recolección (v3): entradas hechas = las que ya están en los atributos del dataset
        self.schedule, self.sched_done, self.sched_skipped = [], set(), set()
        self.sched_id = ""
        if str(g('schedule_file')).strip():
            sp = Path(g('schedule_file'))
            sp = sp if sp.is_absolute() else self.repo / sp
            self.schedule = load_schedule(sp)
            # Solo cuentan los episodios grabados con ESTE programa (al cambiarlo, los índices viejos no son los nuevos)
            self.sched_id = schedule_id(sp)
            self.sched_done = done_indices((dict(self.storage.root[k].attrs) for k in self.storage.root.group_keys()),
                                           self.sched_id, str(g('schedule_legacy_id')) or None)
            self.get_logger().info(f"Programa {sp.name}: {len(self.schedule)} entradas, {len(self.sched_done)} hechas.")
        # Grilla en metros + ArUco de la mesa (guía en vivo sobre la mesa)
        self.cells_m, self.table_xy = {}, {}
        cm = Path(str(g('schedule_cells_m_file')).strip() or "/nonexistent")
        tm = Path(str(g('table_markers_file')).strip() or "/nonexistent")
        cm, tm = (x if x.is_absolute() else self.repo / x for x in (cm, tm))
        if self.schedule and cm.is_file() and tm.is_file():
            import yaml
            cfg = yaml.safe_load(cm.read_text())["v8_cells"]
            self.cells_m = cells_from_edges(cfg["x_edges_m"], cfg["y_edges_m"])
            mk = yaml.safe_load(tm.read_text())["table_markers"]["corners_base_m"]
            self.table_xy = {int(i): np.asarray(c, float)[:, :2] for i, c in mk.items()}
        # Celdas en la vista de HOME3 (make_v8_cells.py): la guía dibuja dónde va la herramienta y su orientación
        self.cells_px, self.cells_info = {}, {}
        cp = Path(str(g('schedule_cells_file')).strip() or "/nonexistent")
        cp = cp if cp.is_absolute() else self.repo / cp
        if self.schedule and cp.is_file():
            import yaml
            cells = (yaml.safe_load(cp.read_text()) or {}).get("v8_cells_home3", {}).get("cells", {})
            self.cells_px = {int(k): v["polygon_home3_px"] for k, v in cells.items()}
            self.cells_info = {int(k): v for k, v in cells.items()}         # alcance: pista de la vista final

        self.create_subscription(Image, g('camera_topic'), self._img_cb, 10, callback_group=self.cb_data)
        self.create_subscription(Odometry, g('odom_topic'), self._odom_cb, 50, callback_group=self.cb_data)
        # La D435i publica el IMU best-effort: con la QoS por defecto (reliable) no llegaba ningún mensaje
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Imu, g('imu_topic'), self._imu_cb, qos_profile_sensor_data, callback_group=self.cb_data)
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
        self.get_logger().info(" [ESPACIO] empezar  [I] empezar ignorando la guía  [H] llegada  [S] guardar  "
                               "[F] forzar  [D] descartar  [U] descartar la última guardada  [G] gripper  [1-9] celda  [N] saltar  "
                               "[Q] salir")
        if self.guide_ref is not None:
            self.get_logger().info(f" Guía de inicio a HOME3: tolerancia {g('start_guide_tolerance_px')} px "
                                   f"({'obligatoria' if g('start_guide_required') else 'solo aviso'}); imagen en "
                                   f"{self.guide_pub.topic_name} (rqt_image_view)")
        self.get_logger().info(f" Inicio automático: {'SÍ' if g('auto_start') else 'no'} | guardado automático: "
                               f"{'SÍ' if g('auto_save') else 'no'}")
        self.get_logger().info("=" * 72)
        self._announce_schedule()

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
            self.latest_full = (_stamp(msg), img)                 # referencia (sin copia) para la guía
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
            self.last_odom_stamp = t
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
        w = msg.angular_velocity
        self.imu_buf.append((msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, w.x, w.y, w.z))
        if self.state in (ARMING,) or (self.state == RECORDING and len(self.imu_acc) < 40):
            a = msg.linear_acceleration
            self.imu_acc.append((a.x, a.y, a.z))
            self.imu_frame = msg.header.frame_id

    def _gripper_cb(self, msg: JointState):
        if msg.position:
            self.current_gripper = float(msg.position[0])

    # ------------------------------------------------------------------ máquina de estados
    def _start(self, override: bool = False, source: str = "manual"):
        with self.lock:
            if self.state != IDLE:
                self.get_logger().warn(f"Ya hay un episodio en curso ({self.state}).")
                return False
            if self.last_odom_stamp is None or len(self.images) == 0:
                self.get_logger().warn("No se puede empezar: falta imagen u odometría.")
                return False
            if self.poses.latest() is None and (self.reset_client is None or not self.reset_client.service_is_ready()):
                # Perdida desde el arranque y sin reinicio: nada la recuperaría en ARMING
                self.get_logger().warn("No se puede empezar: la odometría está perdida y no hay servicio de reinicio.")
                return False
            res = None
            if self.guide_ref is not None and self.guide_res is not None and time.time() - self.guide_res[0] < 1.0:
                res = self.guide_res[1]
            if self.guide_ref is not None and self.p('start_guide_required') and not override:
                if res is None or not res.get("aligned"):
                    self.get_logger().warn("No se empieza: la vista no coincide con HOME3. " +
                                           (summary_line(res) if res else "Sin resultado reciente de la guía.") +
                                           " [I] empieza igual (queda registrado).")
                    self._bell()
                    return False
            self.start_view = None if res is None else {
                k: res.get(k) for k in ("aligned", "error_px", "shift_px", "scale", "roll_deg", "residual_px",
                                        "n_markers", "tol_px")}
            if self.start_view is not None or override:
                self.start_view = dict(self.start_view or {}, override=bool(override), source=source)
            self.auto_save_blocked = False
            self.hold, self.current_gripper, self.pending_problems = False, 0.0, None
            self.imu_acc, self.imu_frame, self.n_null = [], "", 0
            self.state, self.t_arm, self.t_reset, self.odom_reset_ok = ARMING, time.time(), None, False
            self.imu_hold_start = None
            # v3.1: el bag crudo empieza YA, con la cámara quieta en la pose de inicio. Esos frames y el acelerómetro
            # dan la pose REAL de inicio (estimate_start_poses.py). Antes empezaba con el pitido de "Grabando", justo
            # cuando la mano empieza a moverse, y no quedaba ningún momento quieto.
            if bool(self.p('raw_bag_preroll_arming')):
                self._raw_bag_begin("arming")
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
            self.t_reset = self.last_odom_stamp
        return True

    def _raw_bag_begin(self, phase: str) -> None:
        """Abre el bag crudo del episodio si está activado y no está abierto ya (``phase``: arming | recording)."""
        if self.raw_bag is None or self.raw_bag.active:
            return
        stem = Path(self.storage_path).stem
        path = Path(self.p('raw_bag_dir')) / stem / time.strftime("ep_%Y%m%d_%H%M%S")
        rec = self.raw_bag.start(path)
        self.raw_bag_started_in = phase
        self.get_logger().info(f"[RAW BAG] {len(rec)} tópicos desde {phase.upper()} -> {path}")

    def _raw_bag_cancel(self) -> None:
        """Episodio cancelado antes de grabar: el bag abierto en ARMING se borra."""
        if self.raw_bag is not None and self.raw_bag.active:
            self.raw_bag.stop(delete=True)

    def _reset_done(self, _fut):
        # Base de tiempo de la cámara (sellos de la odometría), no el reloj del nodo: así no depende de que la
        # RealSense use el tiempo global del host.
        with self.lock:
            # Todo mensaje (válido o nulo) con sello <= este es anterior al reinicio
            self.t_reset = self.last_odom_stamp
            self.odom_reset_ok = True

    def _tick(self):
        save_now = False
        with self.lock:
            if self.state == ARMING:
                if self.t_reset is None:
                    if time.time() - self.t_arm > float(self.p('reset_timeout_s')):
                        self.get_logger().error("El reinicio de la odometría no respondió: episodio cancelado.")
                        self.state = IDLE
                        self._raw_bag_cancel()
                    return
                if time.time() - self.t_arm > 2.0 * float(self.p('reset_timeout_s')) + 2.0:
                    self.get_logger().error("La odometría no entregó poses válidas tras el reinicio (¿rastreo "
                                            "perdido?): episodio cancelado.")
                    self.state = IDLE
                    self._raw_bag_cancel()
                    return
                if self.poses.valid_after(self.t_reset) >= int(self.p('reset_settle_frames')):
                    t0 = self.poses.latest_stamp
                    self.sampler.start(t0)
                    self.state = RECORDING
                    self._raw_bag_begin("recording")             # sin pre-roll (o si no se abrió en ARMING)
                    self.get_logger().info(">>> [REC] Grabando. Mover la cámara hacia el objetivo; " +
                                           ("al llegar, quedarse QUIETA (guardado automático)." if self.p('auto_save')
                                            else "[H] al llegar."))
                    self._bell()
                return
            if self.state == RECORDING:
                self.sampler.step(self.poses, self.images, self.current_gripper, self.hold,
                                  image_wait_s=float(self.p('image_wait_s')))
                if self.p('auto_save') and not self.auto_save_blocked and len(self.sampler.frames) > 2:
                    fr = self.sampler.frames
                    p = np.array([f.p for f in fr])
                    need = float(self.p('min_hold_duration_s')) + float(self.p('auto_save_margin_s'))
                    tail = fr[-max(2, int(round(need * self.rate))):]
                    if not any(f.lost for f in tail):
                        save_now = auto_save_ready(p, self.rate, float(self.p('auto_save_min_travel_m')),
                                                   float(self.p('min_hold_duration_s')),
                                                   float(self.p('auto_save_margin_s')),
                                                   float(self.p('static_speed_m_s')), float(self.p('static_window_s')))
                    else:
                        # Odometría perdida al final: su pose congelada parecería quieta. La quietud se decide con el
                        # giróscopo (no depende de la imagen) y el recorrido con los frames que sí tenían pose.
                        ok = np.array([not f.lost for f in fr])
                        imu = np.array(self.imu_buf) if len(self.imu_buf) > 10 else None
                        if (imu is not None and ok.sum() >= 2
                                and max_travel(p[ok]) >= float(self.p('auto_save_min_travel_m'))):
                            t_still = imu_still_since(imu[:, 0], imu[:, 1:4], float(self.p('imu_still_gyro_rad_s')),
                                                      float(self.p('imu_still_smooth_s')))
                            if t_still is not None and imu[-1, 0] - t_still >= need:
                                save_now, self.imu_hold_start = True, t_still
        if save_now:                                           # fuera del lock: _save lo toma de nuevo
            ok, msg = self._save(auto_detect_hold=True, trigger="auto")
            if not ok:
                with self.lock:
                    self.auto_save_blocked = True
                self.get_logger().error(f"[AUTO] No se guardó ({msg}). [S] guardar igual o [D] descartar.")

    def _status(self):
        with self.lock:
            if self.state != RECORDING or not self.sampler.frames:
                return
            fr = self.sampler.frames
            lost = sum(f.lost for f in fr)
            inl = [f.inliers for f in fr[-20:] if f.inliers >= 0]
            p = np.array([f.p for f in fr[-11:]])
            v = np.linalg.norm(p[-1] - p[0]) / max(1e-6, (len(p) - 1) / self.rate) if len(p) > 1 else 0.0
        vmax = float(self.p('max_demo_speed_m_s'))
        fast = v > vmax
        msg = (f"[REC] {len(fr) / self.rate:5.1f} s | perdidos {lost} | inliers {min(inl) if inl else -1} | "
               f"v {v * 100:4.1f} cm/s{f' > {vmax * 100:.0f}: MÁS LENTO' if fast else ''}"
               f"{' | HOLD' if self.hold else ''}{f' | celda {self.object_cell}' if self.object_cell else ''}")
        # Una línea por severidad: rclpy lanza ValueError si la misma línea de log cambia de severidad entre llamadas,
        # y esa excepción detenía el ejecutor (el recolector v2 dejaba de recibir datos con el primer frame perdido).
        if lost or fast:
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

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
            self.imu_hold_start = None
            self.sampler.start(None)
            self.sampler.t_next = None
            self.t_idle_since = time.time()
            self.steady.reset()
        if self.raw_bag is not None:
            self.raw_bag.stop(delete=True)
        self.get_logger().info(f"[DISCARD] Episodio descartado ({n} frames).")
        return True, "Descartado"

    def _discard_last_saved(self, reason: str = "descartada por el operador con [U]"):
        """[U]: descarta la ÚLTIMA demo guardada en esta sesión (p. ej. el final no centró la herramienta). No borra
        nada: la marca ``discarded`` en sus atributos, así no cuenta para el programa (su entrada se vuelve a pedir) y
        el conversor y el armado del dataset la omiten. Los datos y el bag crudo quedan, por si se quiere revisar."""
        with self.lock:
            if self.state != IDLE:
                self.get_logger().warn("[U] solo entre demos (no durante una grabación).")
                return False, "En grabación"
            last = self.last_saved
            if last is None:
                self.get_logger().warn("[U] no hay ninguna demo guardada en esta sesión.")
                return False, "Nada que descartar"
            key, index = last
            ep = self.storage.root[key]
            if dict(ep.attrs).get("discarded"):
                self.get_logger().warn(f"[U] {key} ya estaba descartada.")
                return False, "Ya descartada"
            ep.attrs.update({"discarded": True, "discard_reason": reason,
                             "discarded_at": datetime.now().isoformat(timespec="seconds")})
            if index is not None:
                self.sched_done.discard(index)
            self.last_saved = None
        self.get_logger().info(f"[UNDO] {key} descartada: no cuenta para el programa y su entrada se vuelve a pedir "
                               "(los datos quedan, marcados).")
        self._announce_schedule()
        return True, f"{key} descartada"

    def _save(self, force: bool = False, auto_detect_hold: Optional[bool] = None, trigger: str = "manual"):
        with self.lock:
            if self.state != RECORDING:
                return False, "No hay grabación activa"
            if self.saving:                                   # [S] y el guardado automático a la vez
                return False, "Guardado en curso"
            self.saving = True
            # Completar los frames cuya pose ya llegó
            self.sampler.step(self.poses, self.images, self.current_gripper, self.hold, image_wait_s=0.0)
            frames = list(self.sampler.frames)
            # Guardado con tecla: los últimos manual_save_trim_s son el movimiento de apretarla (la mano sostiene la
            # cámara); se quitan para que el tramo final quieto quede limpio.
            trim = float(self.p('manual_save_trim_s'))
            if trigger == "manual" and trim > 0 and len(frames) > 2:
                t_cut = frames[-1].timestamp - trim
                frames = [f for f in frames if f.timestamp <= t_cut]
        try:
            auto_detect = bool(self.p('auto_hold')) if auto_detect_hold is None else bool(auto_detect_hold)
            return self._save_frames(frames, force, auto_detect, trigger)
        finally:
            with self.lock:
                self.saving = False

    def _save_frames(self, frames, force: bool, auto_detect: bool, trigger: str):
        if len(frames) < int(self.p('min_episode_frames')):
            self._discard()
            return False, f"Episodio demasiado corto ({len(frames)} frames): descartado"
        p_abs = np.array([f.p for f in frames])
        hold = np.array([f.hold for f in frames])
        auto, done_source = False, "odometry"
        t_imu = self.imu_hold_start
        if auto_detect and not hold.any() and t_imu is None and any(f.lost for f in frames[-10:]) \
                and len(self.imu_buf) > 10:
            # [S] manual con la odometría perdida al final: la quietud también la decide el giróscopo
            imu = np.array(self.imu_buf)
            imu = imu[imu[:, 0] <= frames[-1].timestamp]
            if len(imu) > 10:
                t_imu = imu_still_since(imu[:, 0], imu[:, 1:4], float(self.p('imu_still_gyro_rad_s')),
                                        float(self.p('imu_still_smooth_s')))
        if auto_detect and not hold.any() and t_imu is not None:
            # Quietud según el giróscopo (odometría perdida al final): el HOLD empieza donde se quedó quieta
            ts = np.array([f.timestamp for f in frames])
            hold = ts >= t_imu
            auto, done_source = bool(hold.any()), "imu"
        if auto_detect and not hold.any():
            i = detect_hold_start(p_abs, self.rate, float(self.p('static_speed_m_s')), float(self.p('static_window_s')))
            if i is not None:
                hold[i:] = True
                auto = True
        # [S] es la decisión del operador (2026-11-25: con la cámara en la mano, el control del tramo final rechazaba
        # demos buenas por 1-3 frames y ni [S] dejaba guardarlas). Sus problemas quedan como avisos; el done del
        # dataset final se recalcula offline con las poses de DROID.
        manual_ok = trigger == "manual" and bool(self.p('manual_save_overrides_hold_qa'))
        hold_warn = []
        if manual_ok and not hold.any() and len(frames) > 1:
            n_tail = max(1, int(round(float(self.p('min_hold_duration_s')) * self.rate)))
            hold[-n_tail:] = True
            done_source = "manual_tail"
            hold_warn.append(f"sin tramo quieto: done en los últimos {n_tail / self.rate:.1f} s (guardado con [S])")
        chk = check_done_segment(p_abs, hold, self.rate, float(self.p('min_hold_duration_s')),
                                 float(self.p('static_speed_m_s')), float(self.p('static_window_s')),
                                 float(self.p('qa_hold_max_moving_frac')))
        qual = episode_quality(frames, self.rate)
        problems = [] if manual_ok else list(chk["problems"])
        if manual_ok:
            hold_warn += list(chk["problems"])
        tracking, warnings = [], list(hold_warn)
        for wmsg in hold_warn:
            self.get_logger().warn(f"[QA] aviso (guardado con [S]): {wmsg}")
        if qual["lost_frames"]:
            tracking.append(f"{qual['lost_frames']} frames con rastreo perdido o sin imagen")
        if qual["duplicate_images"]:
            problems.append(f"{qual['duplicate_images']} imágenes repetidas en frames consecutivos")
        min_inl = int(self.p('min_inliers_tracking'))
        low = sum(1 for f in frames if 0 <= f.inliers < min_inl)
        if low:
            tracking.append(f"{low} frames con menos de {min_inl} inliers")
        if self.p('qa_tracking_blocking'):
            problems += tracking
        else:                                                 # v8: las poses salen de DROID desde el bag crudo
            warnings += tracking
            for wmsg in tracking:
                self.get_logger().warn(f"[QA] aviso (no bloquea; poses offline): {wmsg}")
        if problems and self.p('strict_qa') and not force:
            for pr in problems:
                self.get_logger().error(f"[QA] {pr}")
            self.get_logger().error("[QA] Episodio NO guardado: [S] guardar igual, [D] descartar y repetir, o [F] forzar "
                                    "(queda marcado en los atributos).")
            self._bell()
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
        i_hold = int(np.argmax(chk["done"] > 0.5)) if (chk["done"] > 0.5).any() else len(frames)
        v = smoothed_speeds(p_abs, self.rate, float(self.p('static_window_s')))
        entry = next_pending(self.schedule, self.sched_done, self.sched_skipped) if self.schedule else None
        cell = int(self.object_cell) or (entry.cell if entry is not None else 0)
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
            "qa_warnings": warnings, "done_source": done_source,
            "object_cell": cell, "raw_bag": None if bag_path is None else str(bag_path),
            "raw_bag_starts_at": getattr(self, "raw_bag_started_in", None),
            "git_commit": _git_commit(self.repo), "config_sha256": self._config_hashes(),
            # v3
            "start_view": self.start_view, "schedule": None if entry is None else dict(
                entry.as_attrs(), schedule_id=self.sched_id, yaw_convention=YAW_CONVENTION,
                # La grilla se puede ajustar (adjust_v8_grid.py): la celda en metros deja el episodio autocontenido
                cell_bounds_m=[float(v) for v in self.cells_m.get(entry.cell, ())]),
            "max_speed_m_s": float(v[:i_hold].max()) if i_hold > 0 else 0.0,
            "mean_speed_m_s": float(v[:i_hold].mean()) if i_hold > 0 else 0.0,
            "travel_m": max_travel(p_abs), "max_demo_speed_m_s": float(self.p('max_demo_speed_m_s')),
            "auto_start": bool(self.start_view and self.start_view.get("source") == "auto"),
            "save_trigger": trigger,
        }
        ep_idx, total = self.storage.append_episode(buffer, attrs=attrs, extra_arrays=extra, episode_chunks=True)
        self.imu_hold_start = None
        with self.lock:
            self.state, self.hold = IDLE, False
            self.sampler.t_next = None
            self.t_idle_since = time.time()
            self.steady.reset()
            if entry is not None:
                self.sched_done.add(entry.index)
            self.last_saved = (f"episode_{ep_idx}", None if entry is None else entry.index)
        self.get_logger().info(f"[SAVE] episode_{ep_idx}: {len(frames)} frames ({len(frames) / self.rate:.1f} s), "
                               f"done {chk['hold_frames']} frames{' (auto)' if auto else ''}"
                               f"{', FORZADO con problemas' if force and problems else ''}. Total {total}. "
                               f"Velocidad máx {attrs['max_speed_m_s'] * 100:.1f} cm/s.")
        self._bell()
        self._announce_schedule()
        return True, f"Guardado episode_{ep_idx}"

    # ------------------------------------------------------------------ guía de inicio (v3)
    def _guide_tick(self):
        """En IDLE: compara la vista con HOME3, informa y arranca solo (``auto_start``). En REC: retícula de llegada.
        La imagen de guía solo se dibuja si alguien la mira (rqt_image_view)."""
        lf = self.latest_full
        if lf is None:
            return
        stamp, img = lf
        state = self.state
        want_img = self.guide_pub.get_subscription_count() > 0
        if state == IDLE and self.guide_ref is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            live = detect_markers(gray, self.guide_ref.dictionary)
            res = view_alignment(live, (img.shape[1], img.shape[0]), self.guide_ref,
                                 tol_px=float(self.p('start_guide_tolerance_px')),
                                 min_markers=int(self.p('start_guide_min_markers')))
            now = time.time()
            with self.lock:
                self.guide_res = (now, res)
            if res["aligned"] != self._last_guide_aligned or now - self._last_guide_print > 2.0:
                if res["aligned"]:                             # una línea por severidad (ver _status)
                    self.get_logger().info(summary_line(res))
                else:
                    self.get_logger().warn(summary_line(res))
                self._last_guide_print, self._last_guide_aligned = now, res["aligned"]
            steady = self.steady.update(now, res, live)
            if (self.p('auto_start') and steady and now - self.t_idle_since >= float(self.p('auto_start_cooldown_s'))):
                self.get_logger().info(">>> [AUTO] Vista de HOME3 alineada y quieta: empezando.")
                self.steady.reset()
                self._start(source="auto")
            if want_img:
                target = None
                e = next_pending(self.schedule, self.sched_done, self.sched_skipped) if self.schedule else None
                if e is not None and e.cell in self.cells_m and self.table_xy:
                    # Grilla numerada sobre la mesa (homografía de los ArUco en vivo); texto sin tildes ni grados
                    # (las fuentes de OpenCV son ASCII)
                    target = {"cells_m": self.cells_m, "table_xy": self.table_xy, "cell": e.cell, "yaw_deg": e.yaw_deg,
                              "label": f"COLOCAR #{e.index}: celda {e.cell} | hojas {yaw_words(e.yaw_deg)}"
                                       + (f" | {hint}" if (hint := placement_hint(self.cells_info.get(e.cell), e.yaw_deg)) else "")}
                elif e is not None and e.cell in self.cells_px:
                    target = {"polygon_px": self.cells_px[e.cell], "yaw_deg": e.yaw_deg,
                              "label": f"#{e.index}: celda {e.cell}, hojas {yaw_words(e.yaw_deg).replace('°', ' grados')}"}
                self._publish_guide(draw_guide(img, live, self.guide_ref, res, target), stamp)
        elif state in (ARMING, RECORDING) and want_img:
            out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w = out.shape[:2]
            f = float(self.p('end_view_box_frac'))
            bw, bh = int(w * f / 2), int(h * f / 2)
            cv2.rectangle(out, (w // 2 - bw, h // 2 - bh), (w // 2 + bw, h // 2 + bh), (0, 200, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(out, (w // 2, h // 2), (0, 200, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            txt = "QUIETA: reiniciando odometria" if state == ARMING else (
                "REC  HOLD" if self.hold else "REC  final: herramienta centrada y HOJAS hacia ARRIBA")
            cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 170) if state == RECORDING else (0, 120, 200), -1)
            cv2.putText(out, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self._publish_guide(out, stamp)

    def _publish_guide(self, bgr, stamp):
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp.sec = int(stamp)
        msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9)) % 1_000_000_000
        msg.header.frame_id = "camera_color_optical_frame"
        self.guide_pub.publish(msg)

    def _bell(self):
        if self.p('terminal_bell'):
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except OSError:
                pass

    def _announce_schedule(self):
        if not self.schedule:
            return
        e = next_pending(self.schedule, self.sched_done, self.sched_skipped)
        if e is None:
            self.get_logger().info(f"[PROGRAMA] Completo: {len(self.sched_done)} demos hechas.")
        else:
            hint = placement_hint(self.cells_info.get(e.cell), e.yaw_deg)
            self.get_logger().info(f"[PROGRAMA] Siguiente {e.describe()} ({len(self.sched_done)}/{len(self.schedule)} "
                                   f"hechas). [N] saltar." + (f" Ojo: {hint}." if hint else ""))

    def _skip_schedule(self):
        e = next_pending(self.schedule, self.sched_done, self.sched_skipped) if self.schedule else None
        if e is None:
            self.get_logger().warn("No hay entrada del programa que saltar.")
            return False
        self.sched_skipped.add(e.index)
        self.get_logger().info(f"[PROGRAMA] Saltada {e.describe()} (vuelve a aparecer al reiniciar el recolector).")
        self._announce_schedule()
        return True

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
        if cmd in ("START", "START_OVERRIDE"):
            res.success = self._start(override=(cmd == "START_OVERRIDE"))
            res.message = "Grabación solicitada" if res.success else "No se pudo empezar (¿vista no alineada?)"
        elif cmd == "SKIP":
            res.success = self._skip_schedule()
            res.message = "Entrada saltada" if res.success else "Sin programa"
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
                elif ch == 'i':
                    self._start(override=True)
                elif ch == 'n':
                    self._skip_schedule()
                elif ch == 'h':
                    self._hold()
                elif ch == 's':
                    self._save()
                elif ch == 'f':
                    self._save(force=True)
                elif ch == 'd':
                    self._discard()
                elif ch == 'u':
                    self._discard_last_saved()
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
