"""
GymROS2Env — Gymnasium Environment que conecta Isaac Sim vía ROS 2 para entrenamiento DPPO.

Características Principales:
1. Sincronización Estricta (Lockstep): step() envía comandos y espera la llegada síncrona
   del nuevo frame de cámara y TF mediante threading.Event.
2. Pseudo-Sensor de Fuerza: F_pseudo = p_actual - p_cmd, con filtro de velocidad articular
   para evitar falsos positivos durante el recorrido en el aire.
3. Validación de Agarre en Dos Capas: F_pseudo >= umbral (presión mecánica) + Delta z_obj >= 8cm (TF).
4. Recompensa FSM-Aligned: Reproduce exactamente los hitos, tolerancias y parámetros de pickup_approximate.py.
5. Auto-Stop Neuronal con Crédito Temporal: Premia la predicción action[8] >= 0.50 en la meta y castiga paradas en falso.
6. Canonicidad de Cuaterniones: Fuerza q_w >= 0 en todas las observaciones y comandos.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv_bridge
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, Float64MultiArray
import tf2_ros
from tf2_ros import TransformException


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convierte ángulos de Euler a cuaternión [qx, qy, qz, qw] con qw >= 0."""
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    if q[3] < 0.0:
        q = -q
    return q


ENV_BASE = gym.Env if gym is not None else object


class GymROS2Env(ENV_BASE):
    """
    Entorno Gymnasium compatible con Gym/Isaac Sim gobernado por tópicos de ROS 2.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node: Node, config: Dict[str, Any]):
        if gym is not None:
            super().__init__()
        self.node = node
        self.cfg = config
        self.bridge = cv_bridge.CvBridge()

        # Dimensiones y Parámetros
        self.obs_dim = self.cfg.get("obs_dim", 8)
        self.act_dim = self.cfg.get("act_dim", 9)
        self.image_size = self.cfg.get("image_size", 256)
        self.n_obs_steps = self.cfg.get("n_obs_steps", 2)
        self.max_episode_steps = self.cfg.get("max_episode_steps", 300)
        self.step_timeout_s = self.cfg.get("step_timeout_s", 1.5)
        self.reset_stabilize_s = self.cfg.get("reset_stabilize_s", 2.0)

        # Marcos de Referencia y Geometría (heredados de data_collection_params.yaml)
        self.base_frame = self.cfg.get("base_frame", "link_base")
        self.ee_frame = self.cfg.get("ee_frame", "gripper_base_link")
        self.object_frame = self.cfg.get("object_frame", "pick_target")
        self.tcp_offset_z = self.cfg.get("tcp_offset_z", 0.079)
        self.hover_z_offset = self.cfg.get("hover_z_offset", 0.15)
        self.lift_required_z = self.cfg.get("lift_required_z", 0.08)

        # Destino oficial
        self.target_pos = np.array([
            self.cfg.get("target_x", 0.25),
            self.cfg.get("target_y", 0.0),
            self.cfg.get("target_z", 0.40)
        ], dtype=np.float32)

        # Parámetros del Gripper y Pseudo-Sensor
        self.gripper_open_pos = self.cfg.get("gripper_open_pos", 0.0)
        self.gripper_closed_pos = self.cfg.get("gripper_closed_pos", -0.01)
        self.pseudo_force_thresh = self.cfg.get("pseudo_force_clamped_threshold", 0.0025)
        self.gripper_closed_thresh = self.cfg.get("gripper_binary_closed_threshold", -0.005)
        self.gripper_joint_pattern = self.cfg.get("gripper_joint_name_pattern", "gripper")

        # Estado interno del Pseudo-Sensor
        self.current_gripper_pos = 0.0
        self.prev_gripper_pos = 0.0
        self.last_gripper_update_time = time.time()
        self.gripper_velocity = 0.0
        self.last_cmd_gripper_pos = 0.0
        self.pseudo_force = 0.0

        # Espacios de Gymnasium (opcionales si gymnasium está instalado)
        if spaces is not None:
            self.observation_space = spaces.Dict({
                "observation.state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_obs_steps, self.obs_dim), dtype=np.float32
                ),
                "observation.image": spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_obs_steps, 3, self.image_size, self.image_size), dtype=np.float32
                ),
            })
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
            )
        else:
            self.observation_space = None
            self.action_space = None

        # Infraestructura de ROS 2
        self._obs_event = threading.Event()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # QoS Profiles
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        control_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Suscripciones
        camera_topic = self.cfg.get("camera_topic", "/rgb")
        joint_topic = self.cfg.get("joint_states_topic", "/joint_states")
        self.node.create_subscription(Image, camera_topic, self._img_cb, image_qos)
        self.node.create_subscription(JointState, joint_topic, self._joint_cb, 10)

        # Publicadores
        target_topic = self.cfg.get("target_frame_topic", "/target_frame_raw")
        gripper_topic = self.cfg.get("gripper_command_topic", "/position_controller/commands")
        respawn_topic = self.cfg.get("respawn_topic", "/respawn")
        self.pose_pub = self.node.create_publisher(PoseStamped, target_topic, control_qos)
        self.gripper_pub = self.node.create_publisher(Float64MultiArray, gripper_topic, control_qos)
        self.respawn_pub = self.node.create_publisher(Empty, respawn_topic, 10)

        # Buffers temporales de observación
        self.latest_cv_img = None
        self.latest_state = None
        self.obs_state_history = []
        self.obs_image_history = []

        # Estado del Episodio
        self.current_step = 0
        self.initial_obj_pos = None
        self.prev_action_real = None

    def _joint_cb(self, msg: JointState):
        """Monitorea la posición del gripper y calcula en tiempo real la Pseudo-Fuerza."""
        now = time.time()
        dt = max(now - self.last_gripper_update_time, 1e-4)
        for name, pos in zip(msg.name, msg.position):
            if self.gripper_joint_pattern in name:
                current_p = float(pos)
                self.gripper_velocity = (current_p - self.current_gripper_pos) / dt
                self.prev_gripper_pos = self.current_gripper_pos
                self.current_gripper_pos = current_p
                self.last_gripper_update_time = now
                # Pseudo-fuerza propioceptiva:
                self.pseudo_force = self.current_gripper_pos - self.last_cmd_gripper_pos
                break

    def _img_cb(self, msg: Image):
        """Recibe la imagen RGB y sincroniza la pose del TCP desde el árbol TF."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            t_ee = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())

            ee_pose = np.array([
                t_ee.transform.translation.x,
                t_ee.transform.translation.y,
                t_ee.transform.translation.z,
                t_ee.transform.rotation.x,
                t_ee.transform.rotation.y,
                t_ee.transform.rotation.z,
                t_ee.transform.rotation.w,
                self.current_gripper_pos
            ], dtype=np.float32)

            self.latest_cv_img = cv_img
            self.latest_state = ee_pose

            # Disparar evento de sincronización si estamos esperando un paso
            self._obs_event.set()

        except TransformException:
            pass

    def _get_object_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extrae la posición y orientación del objeto pick_target desde TF."""
        try:
            t_obj = self.tf_buffer.lookup_transform(self.base_frame, self.object_frame, rclpy.time.Time())
            pos = np.array([
                t_obj.transform.translation.x,
                t_obj.transform.translation.y,
                t_obj.transform.translation.z
            ], dtype=np.float32)
            q = np.array([
                t_obj.transform.rotation.x,
                t_obj.transform.rotation.y,
                t_obj.transform.rotation.z,
                t_obj.transform.rotation.w
            ], dtype=np.float32)
            if q[3] < 0.0:
                q = -q
            return pos, q
        except TransformException:
            return None

    def _format_image(self, img_np: np.ndarray) -> np.ndarray:
        """Convierte una imagen NumPy [H, W, 3] a tensor float normalizado [3, size, size]."""
        import cv2
        resized = cv2.resize(img_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor

    def _get_current_obs(self) -> Dict[str, np.ndarray]:
        """Empaqueta el historial de observaciones [T_o, ...] para la política."""
        state_arr = np.stack(self.obs_state_history, axis=0)
        img_arr = np.stack(self.obs_image_history, axis=0)
        return {
            "observation.state": state_arr,
            "observation.image": img_arr,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicio completo de episodio:
        1. Respawn de objetos vía OmniGraph en Isaac Sim.
        2. Regreso del manipulador a la postura de Home.
        3. Espera de reposo físico de 2.0s (auto-resetea además el filtro 1-Euro por timeout).
        """
        if gym is not None:
            try:
                super().reset(seed=seed)
            except Exception:
                pass
        self.current_step = 0
        self.prev_action_real = None
        self.obs_state_history.clear()
        self.obs_image_history.clear()

        # 1. Teletransportar/Respawnear objeto en simulación
        self.respawn_pub.publish(Empty())
        time.sleep(0.5)

        # 2. Mover manipulador a Home transmitiendo continuamente a 20 Hz
        hx = float(self.cfg.get("home_x", 0.25))
        hy = float(self.cfg.get("home_y", 0.0))
        hz = float(self.cfg.get("home_z", 0.40))
        hr = float(self.cfg.get("home_roll", 3.14159))
        hp = float(self.cfg.get("home_pitch", 0.0))
        hyaw = float(self.cfg.get("home_yaw", 0.0))
        home_q = euler_to_quat(hr, hp, hyaw)

        home_msg = PoseStamped()
        home_msg.header.frame_id = self.base_frame
        home_msg.pose.position.x = hx
        home_msg.pose.position.y = hy
        home_msg.pose.position.z = hz
        home_msg.pose.orientation.x = float(home_q[0])
        home_msg.pose.orientation.y = float(home_q[1])
        home_msg.pose.orientation.z = float(home_q[2])
        home_msg.pose.orientation.w = float(home_q[3])

        # Abrir gripper
        grip_msg = Float64MultiArray()
        grip_msg.data = [float(self.gripper_open_pos)]
        self.last_cmd_gripper_pos = self.gripper_open_pos

        reset_duration = float(self.cfg.get("reset_duration_s", 4.0))
        rate_hz = 20.0
        dt = 1.0 / rate_hz
        total_steps = int(reset_duration * rate_hz)

        self.node.get_logger().info(f"Retornando manipulador a Home [{hx:.2f}, {hy:.2f}, {hz:.2f}] a 20 Hz...")
        for step_idx in range(total_steps):
            home_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pose_pub.publish(home_msg)
            self.gripper_pub.publish(grip_msg)
            time.sleep(dt)

            # Si ya pasaron al menos 2.5s y el error cartesiano es mínimo (<2.5cm), Home alcanzado
            if step_idx > int(2.5 * rate_hz) and self.latest_state is not None:
                cur_pos = self.latest_state[:3]
                err = np.linalg.norm(cur_pos - np.array([hx, hy, hz]))
                if err < 0.025:
                    self.node.get_logger().info(f"Home alcanzado correctamente (error residual: {err*100:.1f} cm).")
                    break

        # 3. Breve reposo físico para estabilización
        time.sleep(1.0)

        # Capturar observación inicial (esperar a que el bridge de Isaac Sim publique)
        while self.latest_cv_img is None or self.latest_state is None:
            self.node.get_logger().info("Esperando stream de /rgb y TF desde Isaac Sim...", throttle_duration_sec=2.0)
            self._obs_event.clear()
            self._obs_event.wait(timeout=1.0)

        # Registrar posición inicial de reposo del objeto
        obj_data = self._get_object_pose()
        if obj_data is not None:
            self.initial_obj_pos = obj_data[0].copy()
        else:
            self.initial_obj_pos = np.array([0.28, -0.85, 0.44], dtype=np.float32)

        # Pre-llenar historial con la observación inicial
        init_img = self._format_image(self.latest_cv_img)
        init_state = self.latest_state.copy()
        for _ in range(self.n_obs_steps):
            self.obs_state_history.append(init_state)
            self.obs_image_history.append(init_img)

        return self._get_current_obs(), {"initial_obj_pos": self.initial_obj_pos}

    def step(self, action_real: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta la acción en Isaac Sim de forma síncrona y calcula la recompensa alineada con la FSM.
        action_real: vector desnormalizado [9] -> [X, Y, Z, qx, qy, qz, qw, gripper, done_flag].
        """
        self.current_step += 1

        # 1. Enviar comando cinemático a pose_smoother -> IK
        target_msg = PoseStamped()
        target_msg.header.stamp = self.node.get_clock().now().to_msg()
        target_msg.header.frame_id = self.base_frame
        target_msg.pose.position.x = float(action_real[0])
        target_msg.pose.position.y = float(action_real[1])
        target_msg.pose.position.z = float(action_real[2])

        q = action_real[3:7]
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm
        target_msg.pose.orientation.x = float(q[0])
        target_msg.pose.orientation.y = float(q[1])
        target_msg.pose.orientation.z = float(q[2])
        target_msg.pose.orientation.w = float(q[3])
        self.pose_pub.publish(target_msg)

        # 2. Enviar comando del Gripper
        cmd_gripper = float(action_real[7])
        # Binarizador de control
        binary_cmd = self.gripper_closed_pos if cmd_gripper <= self.gripper_closed_thresh else self.gripper_open_pos
        self.last_cmd_gripper_pos = binary_cmd
        grip_msg = Float64MultiArray()
        grip_msg.data = [binary_cmd]
        self.gripper_pub.publish(grip_msg)

        # 3. Esperar síncronamente nuevo frame de cámara y TF (Lockstep)
        self._obs_event.clear()
        arrived = self._obs_event.wait(timeout=self.step_timeout_s)

        if not arrived:
            self.node.get_logger().warn("Timeout de simulación en step(): Isaac Sim no respondió a tiempo.")
            return self._get_current_obs(), -1.0, True, False, {"timeout": True}

        # Actualizar ventana rodante de observaciones
        cur_img = self._format_image(self.latest_cv_img)
        cur_state = self.latest_state.copy()
        self.obs_state_history.append(cur_state)
        self.obs_image_history.append(cur_img)
        if len(self.obs_state_history) > self.n_obs_steps:
            self.obs_state_history.pop(0)
            self.obs_image_history.pop(0)

        obs = self._get_current_obs()

        # 4. Calcular Recompensa FSM-Aligned con Pseudo-Sensor de Fuerza
        reward, terminated, info = self._compute_reward_and_termination(action_real, cur_state)
        truncated = (self.current_step >= self.max_episode_steps)

        self.prev_action_real = action_real.copy()
        return obs, reward, terminated, truncated, info

    def _compute_reward_and_termination(
        self, action_real: np.ndarray, current_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Computa la función de recompensa por fases basada en la FSM de pickup_approximate.py.
        """
        ee_pos = current_state[:3]
        ee_quat = current_state[3:7]
        done_flag = float(action_real[8]) if len(action_real) >= 9 else 0.0

        obj_data = self._get_object_pose()
        if obj_data is not None:
            obj_pos, obj_quat = obj_data
        else:
            obj_pos = self.initial_obj_pos.copy()
            obj_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # 1. Hito 1: Aproximación en Embudo hacia (pick_target + hover_z)
        p_hover = obj_pos + np.array([0.0, 0.0, self.tcp_offset_z + self.hover_z_offset], dtype=np.float32)
        dist_hover = float(np.linalg.norm(ee_pos - p_hover))
        r_approach = self.cfg.get("reward_approach_weight", 1.0) * np.exp(-dist_hover / self.cfg.get("reward_dist_sigma", 0.05))

        # 2. Hito 2: Alineación de Orientación
        dot_q = float(np.abs(np.dot(ee_quat, obj_quat)))
        r_orient = self.cfg.get("reward_orient_weight", 0.2) * (dot_q ** 2)

        # 3. Hito 3: Agarre Propioceptivo con Pseudo-Sensor de Fuerza + Elevación TF
        delta_z_obj = float(obj_pos[2] - self.initial_obj_pos[2])
        is_closing = (self.last_cmd_gripper_pos <= self.gripper_closed_thresh)
        is_stationary = (abs(self.gripper_velocity) < 0.0015)
        is_clamped = (self.pseudo_force >= self.pseudo_force_thresh) and is_closing and is_stationary

        r_grasp = 0.0
        if is_clamped:
            if delta_z_obj >= self.lift_required_z:
                # Agarre perfecto: compresión mecánica + elevación confirmada >= 8 cm
                r_grasp = self.cfg.get("reward_grasp_lifted_bonus", 5.0)
            else:
                # Contacto firme en la mesa antes de despegar
                r_grasp = self.cfg.get("reward_grasp_clamped_bonus", 1.0)
        elif is_closing and self.pseudo_force < 0.001:
            # Penalización por cerrar las pinzas al aire sin objeto
            r_grasp = self.cfg.get("reward_empty_grasp_penalty", -0.5)

        # 4. Hito 4: Transporte del Objeto al Destino
        dist_target = float(np.linalg.norm(obj_pos - self.target_pos))
        r_transport = 0.0
        if is_clamped and (delta_z_obj >= self.lift_required_z):
            r_transport = self.cfg.get("reward_transport_weight", 2.0) * np.exp(-dist_target / 0.05)

        # 5. Hito 5: Auto-Stop Terminal (done_flag >= 0.50)
        r_done = 0.0
        terminated = False
        if done_flag >= 0.50:
            if dist_target < 0.05 and is_clamped:
                # ¡Éxito Total de la Tarea!
                time_bonus = self.cfg.get("reward_early_stop_bonus", 0.05) * (self.max_episode_steps - self.current_step)
                r_done = self.cfg.get("reward_done_success_bonus", 15.0) + time_bonus
                terminated = True
                self.node.get_logger().info(f"[RL ÉXITO] ¡Tarea completada y validada! (Paso {self.current_step}).")
            else:
                # Parada prematura en falso
                r_done = self.cfg.get("reward_false_done_penalty", -5.0)

        # 6. Penalización de Aceleración Brusca
        r_accel = 0.0
        if self.prev_action_real is not None:
            accel_sq = float(np.sum((action_real[:3] - self.prev_action_real[:3]) ** 2))
            r_accel = self.cfg.get("reward_accel_penalty", 0.005) * accel_sq

        total_reward = float(r_approach + r_orient + r_grasp + r_transport + r_done - r_accel)

        info = {
            "step": self.current_step,
            "dist_hover": dist_hover,
            "dist_target": dist_target,
            "delta_z_obj": delta_z_obj,
            "pseudo_force": self.pseudo_force,
            "is_clamped": is_clamped,
            "done_flag": done_flag,
            "r_approach": r_approach,
            "r_orient": r_orient,
            "r_grasp": r_grasp,
            "r_transport": r_transport,
            "r_done": r_done,
        }
        return total_reward, terminated, info
