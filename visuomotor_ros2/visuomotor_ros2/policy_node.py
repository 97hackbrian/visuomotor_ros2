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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.current_gripper_state = 0.0
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10, callback_group=self.cbg)

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
            self.policy.reset() # Re-inicializar las colas con el nuevo tamaño
            
            self.policy.eval()
            self.policy.to(self.device)
            self.policy.reset()
            self.get_logger().info(f"Loaded {self.get_parameter('model_type').value} model from {ckpt_path}")
        else:
            self.policy = None
            self.get_logger().warn("No checkpoint_path provided. Running in dummy mode.")
        
        self.action_executor = ActionExecutor(self)
        self.action_pub = self.create_publisher(ActionChunk, 'policy/action_chunk', 10)
        
        self.latest_img = None
        self.latest_state = None
        
        # State machine
        self.node_start_time = time.time()
        self.state = 'RESET'
        self.reset_duration_s = 5.0
        self.initial_quat = None
        
        rate = self.get_parameter('inference_hz').value
        self.timer = self.create_timer(1.0 / rate, self._inference_step, callback_group=self.cbg)

    def _joint_cb(self, msg):
        try:
            for name, pos in zip(msg.name, msg.position):
                if 'gripper' in name:
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
            
            if self.initial_quat is None:
                self.initial_quat = ee_pose[3:7]
            
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
            if now - self.node_start_time < self.reset_duration_s:
                self.get_logger().info(f"Yendo a posicion inicial [0.25, 0.0, 0.40]...", throttle_duration_sec=1.0)
                msg = ActionChunk()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.get_parameter("base_frame").value
                p = Pose()
                p.position.x = 0.25
                p.position.y = 0.0
                p.position.z = 0.40
                
                # Mantener la orientación que tenía al iniciar el script
                if self.initial_quat is not None:
                    p.orientation.x = float(self.initial_quat[0])
                    p.orientation.y = float(self.initial_quat[1])
                    p.orientation.z = float(self.initial_quat[2])
                    p.orientation.w = float(self.initial_quat[3])
                
                msg.poses.append(p)
                msg.gripper_states.append(0.0) # Gripper abierto
                
                self.action_pub.publish(msg)
                self.action_executor.execute(msg)
                return
            else:
                self.get_logger().info("RESET finalizado. ¡Cediendo el control a la Red Neuronal (Diffusion Policy)!")
                self.state = 'INFERENCE'
                
        if self.policy is not None:
            import torchvision.transforms as transforms
            # Prepare image tensor (1, C, H, W)
            img_tensor = torch.from_numpy(self.latest_img).to(torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = transforms.Resize((256, 256), antialias=True)(img_tensor)
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
            
            if len(action_np) >= 8:
                # Binarizador de Inferencia: asegurar que el motor del gripper 
                # reciba el booleano duro y no vibre con decimales de la red
                raw_gripper = float(action_np[7])
                binary_gripper = -0.01 if raw_gripper <= -0.001 else raw_gripper
                msg.gripper_states.append(binary_gripper)
            else:
                msg.gripper_states.append(0.0)
                
            self.action_pub.publish(msg)
            self.action_executor.execute(msg)
            
            gripper_val = msg.gripper_states[0]
            gripper_str = "CERRADO" if gripper_val <= -0.001 else "ABIERTO"
            self.get_logger().info(f"[IA] Accion: X={p.position.x:.3f} Y={p.position.y:.3f} Z={p.position.z:.3f} | Gripper: {raw_gripper:.7f} ({gripper_str})", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
