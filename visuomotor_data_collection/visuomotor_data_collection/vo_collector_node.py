#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import glob
import time
import threading
import sys
import select
import termios
import tty
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from visuomotor_msgs.srv import EpisodeTrigger
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage

class VOCollectorNode(Node):
    def __init__(self):
        super().__init__('vo_collector_node')
        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()
        
        # Parámetros del nodo
        self.declare_parameter('dataset_path', 'AUTO')
        self.declare_parameter('dataset_prefix', 'demonstrations_real')
        self.declare_parameter('sampling_rate_hz', 20.0)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('odom_topic', '/rtabmap/odom')
        self.declare_parameter('image_size', 256)
        self.declare_parameter('min_hold_duration_s', 2.0)
        self.declare_parameter('record_done_flag', True)
        self.declare_parameter('min_inliers_tracking', 10)
        self.declare_parameter('relative_coordinates', True)
        
        # Resolución de ruta de dataset
        storage_path = self.get_parameter('dataset_path').value
        prefix = self.get_parameter('dataset_prefix').value
        if storage_path == 'AUTO':
            base_dir = "datasets"
            os.makedirs(base_dir, exist_ok=True)
            pattern = os.path.join(base_dir, f"{prefix}_v*.zarr")
            existing = glob.glob(pattern)
            max_v = 0
            for p in existing:
                match = re.search(rf"{prefix}_v(\d+)\.zarr", p)
                if match:
                    v = int(match.group(1))
                    if v > max_v:
                        max_v = v
            storage_path = os.path.join(base_dir, f"{prefix}_v{max_v + 1}.zarr")
            self.get_logger().info(f"[VO Collector] Modo AUTO: Creando nuevo dataset en {storage_path}")
        else:
            self.get_logger().info(f"[VO Collector] Usando dataset especificado: {storage_path}")
            
        self.storage = ZarrStorage(storage_path)
        
        # Variables de estado
        self.is_recording = False
        self.is_holding = False
        self.buffer = []
        
        self.latest_img = None
        self.latest_pose = None # [X, Y, Z, Qx, Qy, Qz, Qw]
        self.initial_pose = None
        
        self.image_size = self.get_parameter('image_size').value
        
        # Suscriptores
        camera_topic = self.get_parameter('camera_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        
        self.create_subscription(Image, camera_topic, self._img_cb, 10, callback_group=self.cbg)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10, callback_group=self.cbg)
        
        # Servicio de control de episodios
        self.create_service(EpisodeTrigger, '~/trigger_episode', self._srv_trigger, callback_group=self.cbg)
        
        # Timer de muestreo sincrónico
        rate = self.get_parameter('sampling_rate_hz').value
        self.sample_timer = self.create_timer(1.0 / rate, self._sample_step, callback_group=self.cbg)
        
        # Hilo interactivo de teclado en consola
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()
        
        self.get_logger().info("=================================================================")
        self.get_logger().info(" Nodo VO Collector Inicializado (RealSense D415 + RTAB-Map VO)")
        self.get_logger().info(f" Frecuencia: {rate} Hz | Imagen: {self.image_size}x{self.image_size} | Dataset: {storage_path}")
        self.get_logger().info(" Controles de Teclado:")
        self.get_logger().info("   [ESPACIO] -> Iniciar Grabación (START)")
        self.get_logger().info("   [H]       -> Llegada a Destino (HOLD_TARGET / done=1.0)")
        self.get_logger().info("   [S]       -> Guardar Episodio (SAVE)")
        self.get_logger().info("   [D]       -> Descartar Episodio (DISCARD)")
        self.get_logger().info("   [Q]       -> Salir")
        self.get_logger().info("=================================================================")

    def _img_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Redimensionar en RAM a 256x256 para optimizar almacenamiento y compatibilidad
            if cv_img.shape[0] != self.image_size or cv_img.shape[1] != self.image_size:
                cv_img = cv2.resize(cv_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            if self.latest_img is None:
                self.get_logger().info("✅ [Stream OK] Primera imagen recibida de la cámara.")
            self.latest_img = cv_img
        except Exception as e:
            self.get_logger().error(f"Error procesando imagen: {e}")

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        # Asegurar convención de semiesfera superior (Qw >= 0) y cuaternión unitario
        qw = float(ori.w)
        qx = float(ori.x)
        qy = float(ori.y)
        qz = float(ori.z)
        if qw < 0.0:
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz
            
        norm_q = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm_q > 1e-6:
            qx /= norm_q
            qy /= norm_q
            qz /= norm_q
            qw /= norm_q
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            
        if self.latest_pose is None:
            self.get_logger().info("✅ [Stream OK] Primera lectura recibida de odometría VIO (/rtabmap/odom).")
        self.latest_pose = np.array([pos.x, pos.y, pos.z, qx, qy, qz, qw], dtype=np.float32)

    def _sample_step(self):
        if not self.is_recording:
            return
            
        if self.latest_img is None or self.latest_pose is None:
            return
            
        # Transformación relativa respecto al inicio del episodio (Modo UMI / Delta-Home)
        current_pose = self.latest_pose.copy()
        use_relative = self.get_parameter('relative_coordinates').value

        if self.initial_pose is None:
            self.initial_pose = current_pose.copy()
            rel_pos = np.zeros(3, dtype=np.float32) if use_relative else current_pose[0:3]
        else:
            if use_relative:
                # Si es el primer paso del buffer, garantizar exactamente (0.0, 0.0, 0.0) sin jitter numérico
                if len(self.buffer) == 0:
                    rel_pos = np.zeros(3, dtype=np.float32)
                else:
                    rel_pos = current_pose[0:3] - self.initial_pose[0:3]
            else:
                rel_pos = current_pose[0:3]

        rel_ori = current_pose[3:7] # Cuaternión absoluto estabilizado (Qw >= 0)
        
        # 1. observation.state: 8 Dimensiones (X, Y, Z, Qx, Qy, Qz, Qw, Gripper)
        # Regla de oro: CERO fuga causal: done JAMÁS forma parte de observation.state
        state_vec = np.zeros(8, dtype=np.float32)
        state_vec[0:3] = rel_pos
        state_vec[3:7] = rel_ori
        state_vec[7] = 0.0 # Gripper abierto (manual approaching)
        
        # 2. action: 9 Dimensiones (Target_X, Target_Y, Target_Z, Target_Q, Target_Gripper, Done_Flag)
        record_done = self.get_parameter('record_done_flag').value
        if record_done:
            done_val = 1.0 if self.is_holding else 0.0
            action_vec = np.zeros(9, dtype=np.float32)
            action_vec[0:3] = rel_pos
            action_vec[3:7] = rel_ori
            action_vec[7] = 0.0
            action_vec[8] = done_val
        else:
            action_vec = np.zeros(8, dtype=np.float32)
            action_vec[0:3] = rel_pos
            action_vec[3:7] = rel_ori
            action_vec[7] = 0.0
        
        self.buffer.append({
            'image': self.latest_img.copy(),
            'state': state_vec,
            'action': action_vec
        })

    def _start_recording(self):
        if self.latest_img is None or self.latest_pose is None:
            img_status = "OK" if self.latest_img is not None else "FALTA (/camera/camera/color/image_raw)"
            odom_status = "OK" if self.latest_pose is not None else "FALTA (/rtabmap/odom)"
            self.get_logger().warn(f"⚠️ [REC RECHAZADO] No se puede iniciar grabación: faltan tópicos -> Imagen: {img_status} | Odometría: {odom_status}")
            return
        self.buffer.clear()
        self.initial_pose = None
        self.is_holding = False
        self.is_recording = True
        self.get_logger().info(">>> [REC] Grabando episodio... Mueva la cámara hacia el objetivo.")

    def _hold_target(self):
        if not self.is_recording:
            self.get_logger().warn("No se puede hacer HOLD si no se está grabando.")
            return
        self.is_holding = True
        self.get_logger().info(">>> [HOLD] En destino. Inyectando done=1.0. Mantenga quieta la cámara 2s.")

    def _save_episode(self):
        if not self.is_recording:
            self.get_logger().warn("No hay episodio activo para guardar.")
            return False, "No hay grabacion activa"
            
        self.is_recording = False
        self.is_holding = False
        total_frames = len(self.buffer)
        
        if total_frames < 20:
            self.get_logger().warn(f"Episodio demasiado corto ({total_frames} frames). Descartado.")
            self.buffer.clear()
            return False, "Episodio demasiado corto"
            
        # Garantía de piso mínimo de 2.0 segundos (40 frames a 20 Hz) con done=1.0
        rate = self.get_parameter('sampling_rate_hz').value
        min_hold_frames = int(rate * self.get_parameter('min_hold_duration_s').value)
        start_hold_idx = max(0, total_frames - min_hold_frames)
        if self.get_parameter('record_done_flag').value:
            for i in range(start_hold_idx, total_frames):
                self.buffer[i]['action'][8] = 1.0
            
        attrs = {
            'sensor': 'Intel RealSense',
            'odometry': 'RTAB-Map rgbd_odometry',
            'sampling_rate_hz': float(rate),
            'total_steps': total_frames,
            'hold_steps': (total_frames - start_hold_idx),
            'timestamp': time.time(),
            'relative_coordinates': bool(self.get_parameter('relative_coordinates').value),
        }
        if self.initial_pose is not None:
            attrs['initial_pos_x'] = float(self.initial_pose[0])
            attrs['initial_pos_y'] = float(self.initial_pose[1])
            attrs['initial_pos_z'] = float(self.initial_pose[2])
        
        self.storage.append_episode(self.buffer, attrs=attrs)
        self.get_logger().info(f"✔ [SAVE] Episodio guardado con éxito: {total_frames} frames ({total_frames/rate:.2f} s).")
        self.buffer.clear()
        return True, f"Guardado {total_frames} frames"

    def _discard_episode(self):
        self.is_recording = False
        self.is_holding = False
        count = len(self.buffer)
        self.buffer.clear()
        self.get_logger().info(f"✘ [DISCARD] Episodio descartado ({count} frames liberados).")
        return True, "Descartado"

    def _srv_trigger(self, req, res):
        cmd = req.command.strip().upper()
        if cmd == "START":
            self._start_recording()
            res.success = True
            res.message = "Grabacion iniciada"
        elif cmd == "HOLD":
            self._hold_target()
            res.success = True
            res.message = "Hold target activado"
        elif cmd == "SAVE":
            succ, msg = self._save_episode()
            res.success = succ
            res.message = msg
        elif cmd == "DISCARD":
            succ, msg = self._discard_episode()
            res.success = succ
            res.message = msg
        else:
            res.success = False
            res.message = f"Comando desconocido: {req.command}"
        return res

    def _keyboard_loop(self):
        # Solo activa captura de teclado interactivo si stdin es un tty
        if not sys.stdin.isatty():
            return
            
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        self._start_recording()
                    elif ch.lower() == 'h':
                        self._hold_target()
                    elif ch.lower() == 's':
                        self._save_episode()
                    elif ch.lower() == 'd':
                        self._discard_episode()
                    elif ch.lower() == 'q':
                        self.get_logger().info("Saliendo...")
                        rclpy.try_shutdown()
                        break
        except Exception:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = VOCollectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
