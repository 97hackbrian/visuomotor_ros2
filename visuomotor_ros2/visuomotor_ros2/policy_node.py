import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from visuomotor_msgs.msg import ActionChunk
from visuomotor_ros2.observation_buffer import ObservationBuffer
from visuomotor_ros2.action_executors import ActionExecutor
import torch
import numpy as np

# Import real policy
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from pathlib import Path

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.cbg = ReentrantCallbackGroup()
        
        self.declare_parameter('model_type', 'diffusion')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('inference_hz', 10.0)
        self.declare_parameter('h_obs', 2)
        self.declare_parameter('h_act', 16)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load weights
        ckpt_path = self.get_parameter('checkpoint_path').value
        if ckpt_path:
            self.policy = DiffusionPolicy.from_pretrained(Path(ckpt_path))
            self.policy.diffusion.num_inference_steps = 10
            self.policy.eval()
            self.policy.to(self.device)
            self.policy.reset()
            self.get_logger().info(f"Loaded {self.get_parameter('model_type').value} model from {ckpt_path}")
        else:
            self.policy = None
            self.get_logger().warn("No checkpoint_path provided. Running in dummy mode.")
        
        self.obs_buffer = ObservationBuffer(self, self.get_parameter('h_obs').value)
        self.action_executor = ActionExecutor(self)
        
        self.action_pub = self.create_publisher(ActionChunk, 'policy/action_chunk', 10)
        
        rate = self.get_parameter('inference_hz').value
        self.timer = self.create_timer(1.0 / rate, self._inference_step, callback_group=self.cbg)
        
    def _inference_step(self):
        obs = self.obs_buffer.get_recent_observations()
        if obs is None:
            return
            
        if self.policy is not None:
            # Reconstruir tensor de imagen (B, T, C, H, W)
            # Y tensor de estado (B, T_state, dim)
            import cv_bridge
            from geometry_msgs.msg import Pose
            
            bridge = cv_bridge.CvBridge()
            
            # obs is a list of len h_obs. Each has 'image' and 'joint' (or 'ee_pose').
            # We assume 'joint' has positions for joints + gripper
            # Since diffusion policy typically expects [1, C, H, W] for current image or [1, h_obs, C, H, W] depending on the model.
            # In robo_imitate, observation_image is a single image passed to the model which maintains an internal queue.
            # Wait, `self.policy.select_action` from robo_imitate expects:
            # "observation.state": (1, dim), "observation.image": (1, C, H, W)
            # The model internally handles the history (queues).
            
            # So we only need the LATEST observation
            latest_obs = obs[-1]
            cv_img = bridge.imgmsg_to_cv2(latest_obs['image'], desired_encoding='rgb8')
            
            # Prepare image tensor
            img_tensor = torch.from_numpy(cv_img).to(torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            
            # Prepare state tensor
            # The observation_buffer currently stores JointState. 
            # We should extract the positions.
            state_array = np.array(latest_obs['joint'].position, dtype=np.float32)
            state_tensor = torch.from_numpy(state_array).to(torch.float32).to(self.device).unsqueeze(0)
            
            observation = {
                "observation.state": state_tensor,
                "observation.image": img_tensor,
            }
            
            with torch.inference_mode():
                # action = (horizon, action_dim)
                action_seq = self.policy.select_action(observation).squeeze(0).cpu().numpy()
                
            msg = ActionChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            for i in range(action_seq.shape[0]):
                p = Pose()
                p.position.x = float(action_seq[i, 0])
                p.position.y = float(action_seq[i, 1])
                p.position.z = float(action_seq[i, 2])
                
                # If using 6DoF Euler, you might need to convert to Quat here.
                # If using 7DoF Quat natively:
                if action_seq.shape[1] >= 7:
                    p.orientation.x = float(action_seq[i, 3])
                    p.orientation.y = float(action_seq[i, 4])
                    p.orientation.z = float(action_seq[i, 5])
                    p.orientation.w = float(action_seq[i, 6])
                
                msg.poses.append(p)
                
                # Check for Gripper action
                if self.policy.config.use_gripper and action_seq.shape[1] >= 8:
                    msg.gripper_states.append(float(action_seq[i, 7]))
                elif self.policy.config.use_gripper and action_seq.shape[1] == 7: # If 6DoF + gripper
                    msg.gripper_states.append(float(action_seq[i, 6]))
                    
            self.action_pub.publish(msg)
            self.action_executor.execute(msg)
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

