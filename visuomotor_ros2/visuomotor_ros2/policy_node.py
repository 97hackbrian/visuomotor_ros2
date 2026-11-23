from __future__ import annotations
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from visuomotor_msgs.msg import ActionChunk
from visuomotor_ros2.action_executors import ActionExecutor
import torch
import numpy as np
import cv_bridge
from geometry_msgs.msg import Pose, PointStamped
from std_msgs.msg import Float32
import tf2_ros
from tf2_ros import TransformException
import time

# Import real policy
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from pathlib import Path

def euler_to_quat(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.cbg = ReentrantCallbackGroup()
        self.bridge = cv_bridge.CvBridge()
        
        self.declare_parameter('model_type', 'diffusion')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('inference_hz', 20.0)
        self.declare_parameter('camera_topic', '/rgb')
        self.declare_parameter('ee_frame', 'gripper_base_link')
        self.declare_parameter('base_frame', 'link_base')
        self.declare_parameter('num_inference_steps', 16)
        # CUDA Graphs: encoder + bucle DDIM en un grafo (mismo resultado, ~2x más rápido en la Jetson Orin Nano)
        self.declare_parameter('use_cuda_graph', False)
        self.declare_parameter('n_action_steps', 8)
        self.declare_parameter('use_temporal_ensemble', True)
        self.declare_parameter('temporal_ensemble_k', 0.01)
        self.declare_parameter('home_x', 0.25)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('home_z', 0.40)
        self.declare_parameter('home_roll', 3.14159)
        self.declare_parameter('home_pitch', 0.0)
        self.declare_parameter('home_yaw', 0.0)
        self.declare_parameter('relative_coordinates', True)
        
        # --- Configuración de Sensores, Topics y TF ---
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('action_chunk_topic', 'policy/action_chunk')
        
        # --- Configuración de Preprocesamiento de Imagen ---
        self.declare_parameter('image_size', 256)
        
        # --- Configuración del Gripper ---
        self.declare_parameter('gripper_open_pos', 0.0)
        self.declare_parameter('gripper_closed_pos', -0.01)
        self.declare_parameter('gripper_binary_threshold', -0.001)
        self.declare_parameter('gripper_joint_name_pattern', 'gripper')
        
        # --- Configuración de Reset ---
        self.declare_parameter('reset_duration_s', 5.0)
        
        # --- Parámetros de Auto-Stop ---
        self.declare_parameter('enable_neural_auto_stop', True)
        self.declare_parameter('neural_auto_stop_threshold', 0.5)
        # Pasos de inferencia CONSECUTIVOS con la 9ª dimensión >= umbral antes de detener (1 = comportamiento histórico)
        self.declare_parameter('neural_auto_stop_patience_steps', 1)
        self.declare_parameter('enable_heuristic_auto_stop', False)
        self.declare_parameter('enable_auto_stop', True)  # Retrocompatibilidad / interruptor general
        self.declare_parameter('auto_stop_variance_threshold', 0.0000001)
        self.declare_parameter('auto_stop_patience_steps', 20)

        # Convención de signo del cuaternión (estado observado y acciones): 'auto' = la del checkpoint (config.json),
        # 'qw_positive' (histórica) o 'home' (dot(q, q_home) >= 0, continua con la garra hacia abajo).
        self.declare_parameter('quaternion_hemisphere', 'auto')
        # Redimensionado de la imagen: 'torchvision' (histórico: bilineal con antialias) o 'cv2_area'
        # (cv2.INTER_AREA, idéntico a vo_collector_node para datasets reales).
        self.declare_parameter('image_resize_method', 'torchvision')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.current_gripper_state = 0.0
        joint_topic = self.get_parameter('joint_states_topic').value
        self.create_subscription(JointState, joint_topic, self._joint_cb, 10, callback_group=self.cbg)

        camera_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, camera_topic, self._img_cb, 10, callback_group=self.cbg)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        ckpt_path = self.get_parameter('checkpoint_path').value
        if ckpt_path:
            self.policy = DiffusionPolicy.from_pretrained(Path(ckpt_path))
            self.policy.diffusion.num_inference_steps = self.get_parameter('num_inference_steps').value
            
            # Sobreescribir el n_action_steps para controlar la reactividad (pasos ejecutados antes de replanificar)
            self.policy.config.n_action_steps = self.get_parameter('n_action_steps').value
            self.policy.config.use_temporal_ensemble = self.get_parameter('use_temporal_ensemble').value
            self.policy.config.temporal_ensemble_k = self.get_parameter('temporal_ensemble_k').value
            self.policy.reset() # Re-inicializar las colas con el nuevo tamaño
            
            self.policy.eval()
            self.policy.to(self.device)
            self.policy.reset()
            self.get_logger().info(f"Loaded {self.get_parameter('model_type').value} model from {ckpt_path}")
            if self.get_parameter('use_cuda_graph').value:
                t0 = time.perf_counter()
                try:
                    ok = self.policy.enable_cuda_graph(batch_size=1)
                    self.get_logger().info(
                        f"CUDA Graph {'capturado' if ok else 'NO disponible (sin CUDA): inferencia normal'} "
                        f"({self.policy.diffusion.num_inference_steps} pasos DDIM, "
                        f"{(time.perf_counter() - t0) * 1000:.0f} ms)")
                except Exception as e:  # noqa: BLE001 - fail-safe: la política sigue en modo normal
                    self.policy.diffusion.disable_cuda_graph()
                    self.get_logger().warn(f"No se pudo capturar el CUDA Graph ({e}); inferencia normal.")
        else:
            self.policy = None
            self.get_logger().warn("No checkpoint_path provided. Running in dummy mode.")

        self.q_mode, self.q_ref = self._resolve_quaternion_convention()
        self.last_valid_cmd_quat = None
        rep_mode = getattr(self.policy.config, 'action_representation', 'absolute') if self.policy is not None else '-'
        self.get_logger().info(f"Acción del checkpoint: {rep_mode} | cuaternión: {self.q_mode}"
                               + (f" ref={np.round(self.q_ref, 4).tolist()}" if self.q_ref is not None else ""))
        
        self.action_executor = ActionExecutor(self)
        chunk_topic = self.get_parameter('action_chunk_topic').value
        self.action_pub = self.create_publisher(ActionChunk, chunk_topic, 10)

        # --- Telemetría y Métricas en Tiempo Real (para rqt_plot) ---
        self.declare_parameter('enable_metrics_publish', True)
        self.declare_parameter('metrics_prefix', '/policy/metrics')
        self.enable_metrics = self.get_parameter('enable_metrics_publish').value
        prefix = self.get_parameter('metrics_prefix').value
        self.pub_done_flag = self.create_publisher(Float32, f'{prefix}/done_flag', 10)
        self.pub_done_threshold = self.create_publisher(Float32, f'{prefix}/done_threshold', 10)
        self.pub_gripper_cmd = self.create_publisher(Float32, f'{prefix}/gripper_cmd', 10)
        self.pub_gripper_state = self.create_publisher(Float32, f'{prefix}/gripper_state', 10)
        self.pub_tracking_error = self.create_publisher(Float32, f'{prefix}/tracking_error_xyz', 10)
        self.pub_tcp_speed = self.create_publisher(Float32, f'{prefix}/tcp_speed_m_s', 10)
        self.pub_action_variance = self.create_publisher(Float32, f'{prefix}/action_variance', 10)
        self.pub_dist_home = self.create_publisher(Float32, f'{prefix}/dist_to_home', 10)
        self.pub_tcp_cmd = self.create_publisher(PointStamped, f'{prefix}/tcp_cmd_xyz', 10)
        self.pub_tcp_actual = self.create_publisher(PointStamped, f'{prefix}/tcp_actual_xyz', 10)

        self.last_ee_pos = None
        self.last_pose_time = time.time()
        
        self.latest_img = None
        self.latest_state = None
        
        # State machine
        self.node_start_time = time.time()
        self.state = 'RESET'
        
        # Historial para detectar inactividad y hacer auto-stop
        self.action_history = []
        self.frozen_pose = None
        self.frozen_gripper = None
        self.done_trigger_value = None
        self.done_trigger_reason = ""
        self.done_streak = 0
        
        rate = self.get_parameter('inference_hz').value
        self.timer = self.create_timer(1.0 / rate, self._inference_step, callback_group=self.cbg)

    def _resolve_quaternion_convention(self):
        """Convención de signo del cuaternión: la del checkpoint ('auto') o la forzada por YAML (con aviso)."""
        requested = self.get_parameter('quaternion_hemisphere').value
        ckpt_mode = getattr(self.policy.config, 'quaternion_hemisphere', 'qw_positive') if self.policy is not None else 'qw_positive'
        ckpt_ref = getattr(self.policy.config, 'quaternion_reference', None) if self.policy is not None else None
        mode = ckpt_mode if requested == 'auto' else requested
        if mode not in ('qw_positive', 'home'):
            raise ValueError(f"quaternion_hemisphere inválido: {requested!r}")
        if self.policy is not None and requested != 'auto' and requested != ckpt_mode:
            # fallar en vez de advertir: con la garra hacia abajo el estado quedaría fuera de distribución
            raise RuntimeError(f"quaternion_hemisphere={requested} difiere del checkpoint ({ckpt_mode}); "
                               "usar 'auto' o reentrenar con la convención deseada.")
        if mode != 'home':
            return mode, None
        yaml_ref = np.array(euler_to_quat(self.get_parameter('home_roll').value, self.get_parameter('home_pitch').value,
                                          self.get_parameter('home_yaw').value), dtype=np.float64)
        yaml_ref /= np.linalg.norm(yaml_ref)
        if ckpt_ref is None:
            return mode, yaml_ref
        ref = np.asarray(ckpt_ref, dtype=np.float64)
        ref /= np.linalg.norm(ref)
        ang = np.degrees(2.0 * np.arccos(min(1.0, abs(float(ref @ yaml_ref)))))
        if ang > 2.0:
            self.get_logger().warn(f"El HOME del YAML difiere {ang:.1f}° del HOME con que se entrenó el checkpoint.")
        return mode, ref

    def _canonical_quat(self, q):
        """Normaliza y fija el signo según la convención; ``None`` si la norma es ~0."""
        q = np.asarray(q, dtype=np.float64)
        n = float(np.linalg.norm(q))
        if n < 1e-6:
            return None
        q = q / n
        if self.q_mode == 'home' and self.q_ref is not None:
            return -q if float(q @ self.q_ref) < 0.0 else q
        return -q if q[3] < 0.0 else q

    def _joint_cb(self, msg):
        try:
            pattern = self.get_parameter('gripper_joint_name_pattern').value
            for name, pos in zip(msg.name, msg.position):
                if pattern in name:
                    self.current_gripper_state = float(pos)
                    break
        except Exception:
            pass

    def _img_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        try:
            ee_frame = self.get_parameter('ee_frame').value
            base_frame = self.get_parameter('base_frame').value
            t = self.tf_buffer.lookup_transform(base_frame, ee_frame, rclpy.time.Time())
            
            # Cuaternión normalizado con la convención de signo del checkpoint (quaternion_hemisphere)
            q = self._canonical_quat([t.transform.rotation.x, t.transform.rotation.y,
                                      t.transform.rotation.z, t.transform.rotation.w])
            if q is None:
                self.get_logger().warn("Cuaternión de TF con norma ~0; se descarta la observación.",
                                       throttle_duration_sec=2.0)
                return
            qx, qy, qz, qw = (float(v) for v in q)

            ee_pose = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
                qx,
                qy,
                qz,
                qw,
                self.current_gripper_state
            ], dtype=np.float32)
            
            
            self.latest_img = cv_img
            self.latest_state = ee_pose
                
        except TransformException as e:
            self.get_logger().warn(f"TF Error en la inferencia: {e}", throttle_duration_sec=2.0)
        
    def _inference_step(self):
        if self.latest_img is None or self.latest_state is None:
            self.get_logger().info("Esperando imagen y tf para inferir...", throttle_duration_sec=2.0)
            return
            
        now = time.time()
        if self.state == 'RESET':
            reset_duration = self.get_parameter('reset_duration_s').value
            if now - self.node_start_time < reset_duration:
                hx = self.get_parameter('home_x').value
                hy = self.get_parameter('home_y').value
                hz = self.get_parameter('home_z').value
                self.get_logger().info(f"Yendo a posicion inicial [{hx:.2f}, {hy:.2f}, {hz:.2f}]...", throttle_duration_sec=1.0)
                msg = ActionChunk()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.get_parameter("base_frame").value
                p = Pose()
                p.position.x = hx
                p.position.y = hy
                p.position.z = hz
                
                # Orientación inicial desde los parámetros YAML
                r = self.get_parameter('home_roll').value
                p_pitch = self.get_parameter('home_pitch').value
                y = self.get_parameter('home_yaw').value
                home_q = euler_to_quat(r, p_pitch, y)
                
                p.orientation.x = float(home_q[0])
                p.orientation.y = float(home_q[1])
                p.orientation.z = float(home_q[2])
                p.orientation.w = float(home_q[3])
                
                msg.poses.append(p)
                msg.gripper_states.append(self.get_parameter('gripper_open_pos').value)
                
                self.action_pub.publish(msg)
                self.action_executor.execute(msg)
                return
            else:
                self.get_logger().info("RESET finalizado. ¡Cediendo el control a la Red Neuronal (Diffusion Policy)!")
                self.state = 'INFERENCE'
                
        if self.state == 'DONE':
            if self.frozen_pose is not None:
                msg = ActionChunk()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.get_parameter("base_frame").value
                msg.poses.append(self.frozen_pose)
                msg.gripper_states.append(self.frozen_gripper)
                self.action_pub.publish(msg)
                self.action_executor.execute(msg)
                trigger_info = f" ({self.done_trigger_reason})" if self.done_trigger_reason else ""
                self.get_logger().info(f"Tarea completada (Auto-Stop){trigger_info}. Manteniendo posición fija.", throttle_duration_sec=2.0)
                
                # Publicar telemetría constante en estado DONE
                if self.enable_metrics:
                    dummy_act = np.zeros(9, dtype=np.float32)
                    dummy_act[8] = float(self.done_trigger_value) if self.done_trigger_value is not None else 1.0
                    self._publish_telemetry(self.frozen_pose, dummy_act, self.frozen_gripper, self.get_parameter('neural_auto_stop_threshold').value, variance=0.0)
            return

        if self.policy is not None:
            import torchvision.transforms as transforms
            # Prepare image tensor (1, C, H, W)
            img_size = self.get_parameter('image_size').value
            if self.get_parameter('image_resize_method').value == 'cv2_area':
                import cv2
                img_np = self.latest_img
                if img_np.shape[:2] != (img_size, img_size):
                    img_np = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_AREA)
                img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).to(torch.float32).permute(2, 0, 1) / 255.0
            else:
                img_tensor = torch.from_numpy(self.latest_img).to(torch.float32).permute(2, 0, 1) / 255.0
                img_tensor = transforms.Resize((img_size, img_size), antialias=True)(img_tensor)
            img_tensor = img_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            
            # Prepare state tensor (1, dim)
            state_data = self.latest_state.copy()
            if self.get_parameter('relative_coordinates').value:
                home_pos = np.array([
                    self.get_parameter('home_x').value,
                    self.get_parameter('home_y').value,
                    self.get_parameter('home_z').value
                ], dtype=np.float32)
                self.home_pos = home_pos
                state_data[:3] -= home_pos

            state_tensor = torch.from_numpy(state_data).to(torch.float32).to(self.device).unsqueeze(0)
            
            observation = {
                "observation.state": state_tensor,
                "observation.image": img_tensor,
            }
            
            with torch.inference_mode():
                action_output = self.policy.select_action(observation)
                action_np = action_output.squeeze().cpu().numpy()
                
            # Si se usan coordenadas relativas (Modo UMI / Delta-Home), re-proyectar traslación a coordenadas absolutas de base_frame
            if self.get_parameter('relative_coordinates').value:
                home_pos = np.array([
                    self.get_parameter('home_x').value,
                    self.get_parameter('home_y').value,
                    self.get_parameter('home_z').value
                ], dtype=np.float32)
                self.home_pos = home_pos
                if action_np.ndim == 1:
                    action_np[:3] += home_pos
                else:
                    action_np[:, :3] += home_pos

            msg = ActionChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.get_parameter("base_frame").value
            
            # Soporte unificado para predicción escalar 1D [dim] o chunk 2D [H, dim]
            actions_list = action_np if action_np.ndim == 2 else [action_np]
            first_act = actions_list[0]
            
            gripper_open = self.get_parameter('gripper_open_pos').value
            gripper_closed = self.get_parameter('gripper_closed_pos').value
            gripper_thresh = self.get_parameter('gripper_binary_threshold').value

            for act in actions_list:
                p_step = Pose()
                p_step.position.x = float(act[0])
                p_step.position.y = float(act[1])
                p_step.position.z = float(act[2])
                
                if len(act) >= 7:
                    # Cuaternión predicho normalizado con la convención del checkpoint. Si la norma colapsa (p. ej.
                    # promedio de orientaciones contradictorias) se mantiene la última orientación válida o la medida,
                    # nunca la identidad (garra hacia arriba).
                    q_cmd = self._canonical_quat(act[3:7])
                    if q_cmd is not None:
                        self.last_valid_cmd_quat = q_cmd
                    elif self.last_valid_cmd_quat is not None:
                        q_cmd = self.last_valid_cmd_quat
                    else:
                        q_cmd = np.asarray(self.latest_state[3:7], dtype=np.float64)
                    p_step.orientation.x, p_step.orientation.y, p_step.orientation.z, p_step.orientation.w = (
                        float(v) for v in q_cmd)
                msg.poses.append(p_step)

                # Binarizador de Inferencia por paso
                if len(act) >= 8:
                    raw_g = float(act[7])
                    bin_g = gripper_closed if raw_g <= gripper_thresh else raw_g
                    msg.gripper_states.append(bin_g)
                else:
                    msg.gripper_states.append(gripper_open)
                
            self.action_pub.publish(msg)
            self.action_executor.execute(msg)
            
            p = msg.poses[0]
            raw_gripper = float(first_act[7]) if len(first_act) >= 8 else gripper_open
            binary_gripper = msg.gripper_states[0]
            gripper_val = binary_gripper
            gripper_str = "CERRADO" if gripper_val <= gripper_thresh else "ABIERTO"
            
            # --- Lógica de Auto-Stop Separada (Neuronal y Heurística) ---
            # 1. Chequeo por Bandera Neuronal Explícita (Dimensión 9)
            neural_stop_active = self.get_parameter('enable_neural_auto_stop').value and self.get_parameter('enable_auto_stop').value
            neural_threshold = self.get_parameter('neural_auto_stop_threshold').value
            
            done_flag = None
            if len(first_act) >= 9:
                done_flag = float(first_act[8])
                self.done_streak = self.done_streak + 1 if done_flag >= neural_threshold else 0
                patience = max(1, int(self.get_parameter('neural_auto_stop_patience_steps').value))
                if neural_stop_active and self.done_streak >= patience:
                    self.done_trigger_value = done_flag
                    self.done_trigger_reason = f"Neuronal: 9na dim = {done_flag:.4f} >= umbral {neural_threshold:.2f}"
                    self.get_logger().info(
                        f"¡Auto-Stop Neuronal Activado! La red predijo FIN DE TAREA "
                        f"(Valor de 9na dimensión: {done_flag:.4f} >= umbral: {neural_threshold:.2f})."
                    )
                    self.state = 'DONE'
                    self.frozen_pose = p
                    self.frozen_gripper = gripper_val
                    return

            # 2. Chequeo por Heurística de Varianza (Inactividad física)
            if self.get_parameter('enable_heuristic_auto_stop').value:
                # Guardar solo la posición 3D (X, Y, Z)
                pos_array = np.array([p.position.x, p.position.y, p.position.z])
                self.action_history.append(pos_array)
                
                patience = self.get_parameter('auto_stop_patience_steps').value
                if len(self.action_history) > patience:
                    self.action_history.pop(0)
                    
                    history_np = np.array(self.action_history)
                    variance = np.max(np.var(history_np, axis=0))
                    
                    if variance < self.get_parameter('auto_stop_variance_threshold').value:
                        self.done_trigger_value = done_flag
                        dim9_text = f" | 9na dim: {done_flag:.4f}" if done_flag is not None else ""
                        self.done_trigger_reason = f"Heurístico: varianza = {variance:.6f}{dim9_text}"
                        self.get_logger().info(
                            f"¡Auto-Stop Heurístico Activado! Varianza por debajo del umbral ({variance:.6f}){dim9_text}."
                        )
                        self.state = 'DONE'
                        self.frozen_pose = p
                        self.frozen_gripper = gripper_val
                        return
                        
            # Publicar telemetría en tiempo real para rqt_plot
            var_val = variance if 'variance' in locals() else None
            self._publish_telemetry(p, first_act, binary_gripper, neural_threshold, variance=var_val)

            # Log periódico (1 Hz) mostrando acciones, gripper y estado de la última dimensión
            log_msg = f"[IA] Accion: X={p.position.x:.3f} Y={p.position.y:.3f} Z={p.position.z:.3f} | Gripper: {raw_gripper:.7f} ({gripper_str})"
            if done_flag is not None:
                if neural_stop_active:
                    status_str = "ALCANZADO" if done_flag >= neural_threshold else "No alcanzado"
                    log_msg += f" | Dim 9 (Done): {done_flag:.4f} [Umbral: {neural_threshold:.2f} -> {status_str}]"
                else:
                    log_msg += f" | Dim 9 (Done): {done_flag:.4f} (Auto-Stop OFF)"
            elif len(action_np) > 0 and len(action_np) != 8:
                log_msg += f" | Ultima Dim ({len(action_np)}): {float(action_np[-1]):.4f}"
                
            self.get_logger().info(log_msg, throttle_duration_sec=1.0)

    def _publish_telemetry(self, target_pose, action_np, gripper_cmd, neural_threshold, variance=None):
        """Publica métricas escalares y poses en tópicos de ROS 2 para monitoreo con rqt_plot."""
        if not self.enable_metrics:
            return

        done_val = float(action_np[8]) if len(action_np) >= 9 else 0.0
        self.pub_done_flag.publish(Float32(data=done_val))
        self.pub_done_threshold.publish(Float32(data=float(neural_threshold)))
        self.pub_gripper_cmd.publish(Float32(data=float(gripper_cmd)))
        self.pub_gripper_state.publish(Float32(data=float(self.current_gripper_state)))

        if variance is not None:
            self.pub_action_variance.publish(Float32(data=float(variance)))

        if self.latest_state is not None:
            actual_pos = self.latest_state[:3]
            cmd_pos = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
            err = float(np.linalg.norm(cmd_pos - actual_pos))
            self.pub_tracking_error.publish(Float32(data=err))

            now_t = time.time()
            if self.last_ee_pos is not None:
                dt = max(now_t - self.last_pose_time, 1e-4)
                spd = float(np.linalg.norm(actual_pos - self.last_ee_pos) / dt)
                self.pub_tcp_speed.publish(Float32(data=spd))
            self.last_ee_pos = actual_pos.copy()
            self.last_pose_time = now_t

            hx = self.get_parameter('home_x').value
            hy = self.get_parameter('home_y').value
            hz = self.get_parameter('home_z').value
            dist_h = float(np.linalg.norm(actual_pos - np.array([hx, hy, hz])))
            self.pub_dist_home.publish(Float32(data=dist_h))

            now_msg = self.get_clock().now().to_msg()
            frame_id = self.get_parameter('base_frame').value

            ps_cmd = PointStamped()
            ps_cmd.header.stamp = now_msg
            ps_cmd.header.frame_id = frame_id
            ps_cmd.point.x, ps_cmd.point.y, ps_cmd.point.z = float(cmd_pos[0]), float(cmd_pos[1]), float(cmd_pos[2])
            self.pub_tcp_cmd.publish(ps_cmd)

            ps_act = PointStamped()
            ps_act.header.stamp = now_msg
            ps_act.header.frame_id = frame_id
            ps_act.point.x, ps_act.point.y, ps_act.point.z = float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])
            self.pub_tcp_actual.publish(ps_act)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
