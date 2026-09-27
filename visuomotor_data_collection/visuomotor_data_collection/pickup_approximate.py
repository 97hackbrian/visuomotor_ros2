import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray  # noqa: kept for legacy compatibility
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from tf2_ros import TransformException
import time

from visuomotor_msgs.srv import EpisodeTrigger
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage
from visuomotor_data_collection.utils.pid_controller import CartesianPID

class PickupApproximateNode(Node):
    def __init__(self):
        super().__init__('pickup_approximate_node')
        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()
        
        self.declare_parameter('dataset_path', 'demonstrations.zarr')
        self.declare_parameter('sampling_rate_hz', 20.0)
        self.declare_parameter('object_frame', 'pick_target')
        self.declare_parameter('ee_frame', 'gripper_base_link')
        self.declare_parameter('tcp_offset_z', 0.17)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('max_speed_m_s', 0.15) # Límite de velocidad
        
        # Nuevos parámetros de la secuencia
        self.declare_parameter('init_x', 0.25)        # Posición inicial oficial en X
        self.declare_parameter('init_y', 0.0)         # Posición inicial oficial en Y
        self.declare_parameter('init_z_height', 0.40) # Altura a la que sube para enfocar
        self.declare_parameter('wait_time_s', 2.0)    # Tiempo de espera arriba
        self.declare_parameter('grasp_wait_time_s', 1.0) # Tiempo esperando que cierre
        self.declare_parameter('lift_z_offset', 0.35) # Altura final de levantamiento
        
        # Parámetros avanzados de Control y Tolerancias
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('goal_tolerance_m', 0.03)
        self.declare_parameter('gripper_open_pos', 0.0)
        self.declare_parameter('gripper_closed_pos', -0.01)
        
        self.storage = ZarrStorage(self.get_parameter('dataset_path').value)
        
        self.buffer = []
        self.latest_img = None
        self.latest_state = None
        self.latest_action = None

        self.create_subscription(Image, 'camera/image_raw', self._img_cb, 10, callback_group=self.cbg)
        self.create_subscription(JointState, 'joint_states', self._state_cb, 10, callback_group=self.cbg)
        
        self.action_pub = self.create_publisher(PoseStamped, 'target_frame', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pid = CartesianPID(sampling_rate_hz=self.get_parameter('control_rate_hz').value)
        
        self.fixed_orientation = None
        
        # Máquina de estados: IDLE, INIT_RAISE, WAIT, APPROACH, GRASP, LIFT
        self.state = 'IDLE'
        self.state_start_time = 0.0
        self.last_published_gripper_state = None
        self.is_recording = False

        self.create_service(EpisodeTrigger, '~/trigger_episode', self._srv_trigger, callback_group=self.cbg)

        # Timer para el control a 100 Hz (frecuencia de FZI)
        self.control_timer = self.create_timer(1.0 / self.get_parameter('control_rate_hz').value, self._control_step, callback_group=self.cbg)

        # Timer para recolección de datos a 20 Hz
        rate = self.get_parameter('sampling_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self._sample_step, callback_group=self.cbg)
        
        max_speed = self.get_parameter('max_speed_m_s').value
        self.get_logger().info(f"Pickup & Approximate Sequence Node Inicializado. Velocidad max: {max_speed} m/s")

    def _img_cb(self, msg):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _state_cb(self, msg):
        self.latest_state = np.array(msg.position, dtype=np.float32)

    def _control_step(self):
        obj_frame = self.get_parameter('object_frame').value
        ee_frame = self.get_parameter('ee_frame').value
        
        try:
            t_obj = self.tf_buffer.lookup_transform('link_base', obj_frame, rclpy.time.Time())
            t_ee = self.tf_buffer.lookup_transform('link_base', ee_frame, rclpy.time.Time())
        except TransformException as ex:
            return
            
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        tcp_offset_z = self.get_parameter('tcp_offset_z').value
        
        target_x = t_obj.transform.translation.x
        target_y = t_obj.transform.translation.y
        target_z = t_obj.transform.translation.z
        
        gripper_open_pos = self.get_parameter('gripper_open_pos').value
        gripper_closed_pos = self.get_parameter('gripper_closed_pos').value
        goal_tolerance = self.get_parameter('goal_tolerance_m').value
        
        gripper_state = gripper_open_pos
        if self.state == 'IDLE' or self.state == 'WAIT':
            target_x = self.get_parameter('init_x').value
            target_y = self.get_parameter('init_y').value
            target_z = self.get_parameter('init_z_height').value
        elif self.state == 'APPROACH':
            target_z = t_obj.transform.translation.z + tcp_offset_z
        elif self.state == 'GRASP':
            target_z = t_obj.transform.translation.z + tcp_offset_z
            gripper_state = gripper_closed_pos
        elif self.state == 'LIFT':
            target_z = t_obj.transform.translation.z + self.get_parameter('lift_z_offset').value
            gripper_state = gripper_closed_pos
            
        error = np.array([
            target_x - t_ee.transform.translation.x,
            target_y - t_ee.transform.translation.y,
            target_z - t_ee.transform.translation.z
        ])
        
        dist = np.linalg.norm(error)
        
        now = time.time()
        if self.state == 'WAIT':
            if now - self.state_start_time > self.get_parameter('wait_time_s').value:
                self.state = 'APPROACH'
                self.get_logger().info("Iniciando APPROACH y GRABACIÓN.")
                self.buffer.clear()
                self.is_recording = True
        elif self.state == 'APPROACH':
            if dist < goal_tolerance:
                self.state = 'GRASP'
                self.state_start_time = now
                self.get_logger().info("Objetivo alcanzado. Cerrando gripper...")
        elif self.state == 'GRASP':
            if now - self.state_start_time > self.get_parameter('grasp_wait_time_s').value:
                self.state = 'LIFT'
                self.get_logger().info("Gripper cerrado. Levantando objeto...")
        elif self.state == 'LIFT':
            if dist < goal_tolerance:
                self.get_logger().info("Objeto levantado. Finalizando episodio y guardando...")
                self._save_episode()
                self.state = 'IDLE'

        max_speed_m_s = self.get_parameter('max_speed_m_s').value
        control_rate = self.get_parameter('control_rate_hz').value
        self.pid.sampling_rate_hz = control_rate
        self.pid.update_params(kp, ki, kd, max_speed_m_s)
        
        current_pos = np.array([
            t_ee.transform.translation.x,
            t_ee.transform.translation.y,
            t_ee.transform.translation.z
        ])
        target_pos = np.array([target_x, target_y, target_z])
        
        control_output, _ = self.pid.compute(current_pos, target_pos)
        
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'link_base'
        
        target_pose.pose.position.x = t_ee.transform.translation.x + control_output[0]
        target_pose.pose.position.y = t_ee.transform.translation.y + control_output[1]
        target_pose.pose.position.z = t_ee.transform.translation.z + control_output[2]
        
        if self.fixed_orientation is None:
            self.fixed_orientation = t_ee.transform.rotation
            
        target_pose.pose.orientation = self.fixed_orientation
        
        self.action_pub.publish(target_pose)
        

        if self.last_published_gripper_state != gripper_state:
            gripper_msg = Float64MultiArray()
            gripper_msg.data = [gripper_state]
            self.gripper_pub.publish(gripper_msg)
            self.last_published_gripper_state = gripper_state
            self.get_logger().info(f"Gripper command enviado: {gripper_state}")

        
        self.latest_action = np.array([
            target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z,
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, 
            target_pose.pose.orientation.z, target_pose.pose.orientation.w, 
            gripper_state
        ], dtype=np.float32)

    def _sample_step(self):
        if self.is_recording and self.latest_img is not None and self.latest_state is not None and self.latest_action is not None:
            self.buffer.append({
                'image': self.latest_img.copy(),
                'state': self.latest_state.copy(),
                'action': self.latest_action.copy()
            })

    def _save_episode(self):
        self.is_recording = False
        if len(self.buffer) < 15:
            self.get_logger().warn("Secuencia insuficiente. Episodio descartado.")
        else:
            self.storage.append_episode(self.buffer)
            self.get_logger().info(f"Episodio guardado exitosamente: {len(self.buffer)} pasos.")
        self.buffer.clear()

    def _srv_trigger(self, req, res):
        if req.command == "START":
            self.pid.reset()
            self.fixed_orientation = None
            self.is_recording = False
            self.buffer.clear()
            self.state = 'WAIT'
            self.state_start_time = time.time()
            self.get_logger().info("=== INICIANDO SECUENCIA PICK UP AND APPROXIMATE ===")
            res.success = True
            res.message = "Secuencia iniciada."
        else:
            res.success = False
            res.message = "Comando desconocido. Usa START."
        return res

def main(args=None):
    rclpy.init(args=args)
    node = PickupApproximateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
