import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from visuomotor_msgs.msg import ActionChunk
from visuomotor_ros2.action_executors import ActionExecutor
import torch
import numpy as np
import cv_bridge
from geometry_msgs.msg import Pose
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
        self.declare_parameter('n_action_steps', 8)
        self.declare_parameter('use_temporal_ensemble', True)
        self.declare_parameter('temporal_ensemble_k', 0.01)
        self.declare_parameter('home_x', 0.25)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('home_z', 0.40)
        self.declare_parameter('home_roll', 3.14159)
        self.declare_parameter('home_pitch', 0.0)
        self.declare_parameter('home_yaw', 0.0)
        
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
        self.declare_parameter('enable_heuristic_auto_stop', False)
        self.declare_parameter('enable_auto_stop', True)  # Retrocompatibilidad / interruptor general
        self.declare_parameter('auto_stop_variance_threshold', 0.0000001)
        self.declare_parameter('auto_stop_patience_steps', 20)
        
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
        else:
            self.policy = None
            self.get_logger().warn("No checkpoint_path provided. Running in dummy mode.")
        
        self.action_executor = ActionExecutor(self)
        chunk_topic = self.get_parameter('action_chunk_topic').value
        self.action_pub = self.create_publisher(ActionChunk, chunk_topic, 10)
        
        self.latest_img = None
        self.latest_state = None
        
        # State machine
        self.node_start_time = time.time()
        self.state = 'RESET'
        
        # Historial para detectar inactividad y hacer auto-stop
        self.action_history = []
        self.frozen_pose = None
        self.frozen_gripper = None
        
        rate = self.get_parameter('inference_hz').value
        self.timer = self.create_timer(1.0 / rate, self._inference_step, callback_group=self.cbg)

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
            
            ee_pose = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
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
                self.get_logger().info("Tarea completada (Auto-Stop). Manteniendo posición fija.", throttle_duration_sec=2.0)
            return

        if self.policy is not None:
            import torchvision.transforms as transforms
            # Prepare image tensor (1, C, H, W)
            img_size = self.get_parameter('image_size').value
            img_tensor = torch.from_numpy(self.latest_img).to(torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = transforms.Resize((img_size, img_size), antialias=True)(img_tensor)
            img_tensor = img_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            
            # Prepare state tensor (1, dim)
            state_tensor = torch.from_numpy(self.latest_state).to(torch.float32).to(self.device).unsqueeze(0)
            
            observation = {
                "observation.state": state_tensor,
                "observation.image": img_tensor,
            }
            
            with torch.inference_mode():
                action_output = self.policy.select_action(observation)
                action_np = action_output.squeeze().cpu().numpy()
                
            msg = ActionChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.get_parameter("base_frame").value
            
            p = Pose()
            p.position.x = float(action_np[0])
            p.position.y = float(action_np[1])
            p.position.z = float(action_np[2])
            
            if len(action_np) >= 7:
                # Normalizar el cuaternión predicho por la red neuronal!
                norm = np.linalg.norm(action_np[3:7])
                if norm > 1e-6:
                    p.orientation.x = float(action_np[3] / norm)
                    p.orientation.y = float(action_np[4] / norm)
                    p.orientation.z = float(action_np[5] / norm)
                    p.orientation.w = float(action_np[6] / norm)
                else:
                    p.orientation.x = float(action_np[3])
                    p.orientation.y = float(action_np[4])
                    p.orientation.z = float(action_np[5])
                    p.orientation.w = float(action_np[6])
            
            msg.poses.append(p)
            
            gripper_open = self.get_parameter('gripper_open_pos').value
            gripper_closed = self.get_parameter('gripper_closed_pos').value
            gripper_thresh = self.get_parameter('gripper_binary_threshold').value
            
            if len(action_np) >= 8:
                # Binarizador de Inferencia: asegurar que el motor del gripper 
                # reciba el booleano duro y no vibre con decimales de la red
                raw_gripper = float(action_np[7])
                binary_gripper = gripper_closed if raw_gripper <= gripper_thresh else raw_gripper
                msg.gripper_states.append(binary_gripper)
            else:
                msg.gripper_states.append(gripper_open)
                
            self.action_pub.publish(msg)
            self.action_executor.execute(msg)
            
            gripper_val = msg.gripper_states[0]
            gripper_str = "CERRADO" if gripper_val <= gripper_thresh else "ABIERTO"
            
            # --- Lógica de Auto-Stop Separada (Neuronal y Heurística) ---
            # 1. Chequeo por Bandera Neuronal Explícita (Dimensión 9)
            neural_stop_active = self.get_parameter('enable_neural_auto_stop').value and self.get_parameter('enable_auto_stop').value
            neural_threshold = self.get_parameter('neural_auto_stop_threshold').value
            
            done_flag = None
            if len(action_np) >= 9:
                done_flag = float(action_np[8])
                if neural_stop_active and done_flag >= neural_threshold:
                    self.get_logger().info(
                        f"¡Auto-Stop Neuronal Activado! La red predijo FIN DE TAREA "
                        f"(flag={done_flag:.4f} >= umbral={neural_threshold:.2f})."
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
                        self.get_logger().info(f"¡Auto-Stop Heurístico Activado! Varianza por debajo del umbral ({variance:.6f}).")
                        self.state = 'DONE'
                        self.frozen_pose = p
                        self.frozen_gripper = gripper_val
                        return
                        
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

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
