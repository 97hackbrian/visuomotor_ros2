import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np

import tf2_ros
from tf2_ros import TransformException

from visuomotor_msgs.srv import EpisodeTrigger
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage
from visuomotor_data_collection.utils.pid_controller import CartesianPID

class SimCollectorNode(Node):
    def __init__(self):
        super().__init__('sim_collector_node')
        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()
        
        # Parámetros generales
        self.declare_parameter('dataset_path', 'demonstrations.zarr')
        self.declare_parameter('sampling_rate_hz', 20.0)
        
        # Parámetros para el PID automático (Demostrador en simulación)
        self.declare_parameter('enable_auto_pid', False)
        self.declare_parameter('object_frame', 'object_link')
        self.declare_parameter('ee_frame', 'link_tcp')
        self.declare_parameter('tcp_offset_z', 0.17) # Offset en metros para la longitud del gripper
        self.declare_parameter('init_x', 0.25)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_z_height', 0.40) # Altura inicial segura
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('max_speed_m_s', 0.15)
        
        # Parámetros avanzados de Control y Tolerancias
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('goal_tolerance_m', 0.03)
        self.declare_parameter('gripper_open_pos', 0.0)
        self.declare_parameter('gripper_closed_pos', -0.01)
        
        storage_path = self.get_parameter('dataset_path').value
        self.storage = ZarrStorage(storage_path)
        
        self.is_recording = False
        self.buffer = []
        self.latest_img = None
        self.latest_state = None
        self.latest_action = None

        # Suscriptores de modalidades sensoriales
        self.create_subscription(Image, 'camera/image_raw', self._img_cb, 10, callback_group=self.cbg)
        self.create_subscription(JointState, 'joint_states', self._state_cb, 10, callback_group=self.cbg)
        
        # Dependiendo si somos auto-demostrador o solo recolector pasivo
        self.enable_auto_pid = self.get_parameter('enable_auto_pid').value
        if self.enable_auto_pid:
            self.action_pub = self.create_publisher(PoseStamped, 'target_frame', 10)
            self.gripper_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
            
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            
            self.pid = CartesianPID(sampling_rate_hz=self.get_parameter('control_rate_hz').value)
            self.fixed_orientation = None
            
            # Timer de control a 100Hz
            self.control_timer = self.create_timer(1.0 / self.get_parameter('control_rate_hz').value, self._control_step, callback_group=self.cbg)
        else:
            # Si el PID está desactivado, solo escuchamos las acciones que otro nodo genere (ej. teleoperación manual)
            self.create_subscription(PoseStamped, 'target_frame', self._action_cb, 10, callback_group=self.cbg)

        # Servicios de control de episodios
        self.create_service(EpisodeTrigger, '~/trigger_episode', self._srv_trigger, callback_group=self.cbg)

        # Bucle principal de muestreo y control
        rate = self.get_parameter('sampling_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self._sample_step, callback_group=self.cbg)
        self.get_logger().info("Nodo recolector inicializado. PID automático: {}".format(self.enable_auto_pid))

    def _img_cb(self, msg):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _state_cb(self, msg):
        self.latest_state = np.array(msg.position, dtype=np.float32)

    def _action_cb(self, msg):
        p = msg.pose
        self.latest_action = np.array([p.position.x, p.position.y, p.position.z,
                                       p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], dtype=np.float32)

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
        max_speed_m_s = self.get_parameter('max_speed_m_s').value
        
        self.pid.sampling_rate_hz = self.get_parameter('control_rate_hz').value
        self.pid.update_params(kp, ki, kd, max_speed_m_s)
        
        tcp_offset_z = self.get_parameter('tcp_offset_z').value
        init_x = self.get_parameter('init_x').value
        init_y = self.get_parameter('init_y').value
        init_z_height = self.get_parameter('init_z_height').value
        
        target_x = t_obj.transform.translation.x if self.is_recording else init_x
        target_y = t_obj.transform.translation.y if self.is_recording else init_y
        target_z = (t_obj.transform.translation.z + tcp_offset_z) if self.is_recording else init_z_height
        
        current_pos = np.array([
            t_ee.transform.translation.x,
            t_ee.transform.translation.y,
            t_ee.transform.translation.z
        ])
        target_pos = np.array([target_x, target_y, target_z])
        
        control_output, error = self.pid.compute(current_pos, target_pos)
        
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
        
        dist = np.linalg.norm(error)
        
        gripper_open_pos = self.get_parameter('gripper_open_pos').value
        gripper_closed_pos = self.get_parameter('gripper_closed_pos').value
        goal_tolerance = self.get_parameter('goal_tolerance_m').value
        
        gripper_state = gripper_closed_pos if (self.is_recording and dist < goal_tolerance) else gripper_open_pos
        
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [gripper_state]
        self.gripper_pub.publish(gripper_msg)
        
        self.latest_action = np.array([
            target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z,
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, 
            target_pose.pose.orientation.z, target_pose.pose.orientation.w, 
            gripper_state
        ], dtype=np.float32)

    def _sample_step(self):
        # Si somos el demostrador automático, generamos la acción
# Grabación del estado si el episodio está activo
        if self.is_recording and self.latest_img is not None and self.latest_state is not None and self.latest_action is not None:
            self.buffer.append({
                'image': self.latest_img.copy(),
                'state': self.latest_state.copy(),
                'action': self.latest_action.copy()
            })

    def _srv_trigger(self, req, res):
        if req.command == "START":
            self.buffer.clear()
            self.is_recording = True
            if self.enable_auto_pid:
                self.pid.reset()
                self.fixed_orientation = None
            res.success = True
            res.message = "Muestreo sincrónico iniciado."
        elif req.command == "SAVE":
            self.is_recording = False
            if len(self.buffer) < 15:
                res.success = False
                res.message = "Secuencia insuficiente. Episodio descartado."
            else:
                self.storage.append_episode(self.buffer)
                res.success = True
                res.message = f"Episodio registrado en Zarr ({len(self.buffer)} iteraciones temporales)."
            self.buffer.clear()
        elif req.command == "DISCARD":
            self.is_recording = False
            self.buffer.clear()
            res.success = True
            res.message = "Búfer volátil liberado."
        else:
            res.success = False
            res.message = "Comando desconocido."
        return res

def main(args=None):
    rclpy.init(args=args)
    node = SimCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
