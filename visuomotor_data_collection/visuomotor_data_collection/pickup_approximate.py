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

def multiply_quaternions(q1, q2):
    # q = [x, y, z, w]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def euler_from_quaternion(q):
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw

def quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

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
        self.declare_parameter('camera_topic', '/rgb')
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
        self.declare_parameter('grasp_wait_time_s', 5.0) # Tiempo esperando que cierre
        self.declare_parameter('settle_time_s', 1.5)  # Tiempo inmóvil antes de cerrar
        self.declare_parameter('approach_timeout_s', 30.0) # Timeout máximo por fase
        self.declare_parameter('hover_z_offset', 0.10) # Altura extra sobre el objeto al alinearse (metros)
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

        camera_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, camera_topic, self._img_cb, 10, callback_group=self.cbg)
        self.action_pub = self.create_publisher(PoseStamped, 'target_frame', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pid = CartesianPID(sampling_rate_hz=self.get_parameter('control_rate_hz').value)
        
        self.fixed_orientation = None
        self.grasp_position = None  # Posición XYZ exacta donde se cierra el gripper
        
        # Máquina de estados: IDLE, WAIT, ALIGN, DESCEND, SETTLE, GRASP, LIFT
        self.state = 'IDLE'
        self.state_start_time = 0.0
        self.last_published_gripper_state = None
        self.is_recording = False
        self.remaining_repetitions = 0
        self.target_orientation_quat = None
        self.initial_ee_quat = None

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

    def _control_step(self):
        obj_frame = self.get_parameter('object_frame').value
        ee_frame = self.get_parameter('ee_frame').value
        
        try:
            t_ee = self.tf_buffer.lookup_transform('link_base', ee_frame, rclpy.time.Time())
        except TransformException:
            return
            
        t_obj = None
        try:
            t_obj = self.tf_buffer.lookup_transform('link_base', obj_frame, rclpy.time.Time())
        except TransformException:
            pass
            
        if t_obj is None and self.state not in ('IDLE', 'WAIT'):
            return  # Requiere objeto para las fases activas
            
        tcp_offset_z = self.get_parameter('tcp_offset_z').value
        hover_z_offset = self.get_parameter('hover_z_offset').value
        
        if t_obj is not None:
            obj_x = t_obj.transform.translation.x
            obj_y = t_obj.transform.translation.y
            obj_z = t_obj.transform.translation.z
        else:
            obj_x = obj_y = obj_z = 0.0
            
        ee_x  = t_ee.transform.translation.x
        ee_y  = t_ee.transform.translation.y
        ee_z  = t_ee.transform.translation.z

        gripper_open_pos   = self.get_parameter('gripper_open_pos').value
        gripper_closed_pos = self.get_parameter('gripper_closed_pos').value
        goal_tolerance     = self.get_parameter('goal_tolerance_m').value

        # ── Determinar target según estado ──────────────────────────────────
        gripper_state = gripper_open_pos

        if self.state in ('IDLE', 'WAIT'):
            # Posición de reposo / inicio
            target_x = self.get_parameter('init_x').value
            target_y = self.get_parameter('init_y').value
            target_z = self.get_parameter('init_z_height').value

        elif self.state == 'ALIGN':
            # Fase 1: moverse en X,Y sobre el objeto, manteniendo la altura actual del EE
            target_x = obj_x
            target_y = obj_y
            target_z = obj_z + tcp_offset_z + hover_z_offset  # altura segura sobre el objeto

        elif self.state == 'DESCEND':
            # Fase 2: bajar verticalmente (X,Y ya están alineados)
            target_x = obj_x
            target_y = obj_y
            target_z = obj_z + tcp_offset_z  # altura de agarre

        elif self.state == 'SETTLE':
            # Brazo inmóvil en el punto de agarre grabado
            if self.grasp_position is not None:
                target_x, target_y, target_z = self.grasp_position
            else:
                target_x, target_y, target_z = ee_x, ee_y, ee_z

        elif self.state == 'GRASP':
            # Brazo completamente inmóvil mientras cierra el gripper
            if self.grasp_position is not None:
                target_x, target_y, target_z = self.grasp_position
            else:
                target_x, target_y, target_z = ee_x, ee_y, ee_z
            gripper_state = gripper_closed_pos

        elif self.state == 'LIFT':
            if self.grasp_position is not None:
                target_x = self.grasp_position[0]
                target_y = self.grasp_position[1]
                target_z = self.grasp_position[2] + self.get_parameter('lift_z_offset').value
            else:
                target_x = ee_x
                target_y = ee_y
                target_z = ee_z + self.get_parameter('lift_z_offset').value
            gripper_state = gripper_closed_pos

        # Orientación target
        if self.target_orientation_quat is None:
            self.target_orientation_quat = [
                t_ee.transform.rotation.x, t_ee.transform.rotation.y,
                t_ee.transform.rotation.z, t_ee.transform.rotation.w
            ]
            self.initial_ee_quat = self.target_orientation_quat.copy()

        if self.state in ('ALIGN', 'DESCEND'):
            # Obtenemos el Yaw del objeto
            q_obj = [
                t_obj.transform.rotation.x, t_obj.transform.rotation.y,
                t_obj.transform.rotation.z, t_obj.transform.rotation.w
            ]
            _, _, obj_yaw = euler_from_quaternion(q_obj)
            
            # Obtenemos el Roll y Pitch INICIALES del gripper (así siempre apunta perfectamente hacia abajo)
            init_roll, init_pitch, _ = euler_from_quaternion(self.initial_ee_quat)
            
            # El Yaw deseado es el del objeto + 90 grados (pi/2) para llegar perpendicular
            target_yaw = obj_yaw + (np.pi / 2.0)
            
            self.target_orientation_quat = quaternion_from_euler(init_roll, init_pitch, target_yaw)

        # ── Error 3D y distancias parciales ────────────────────────────────
        error = np.array([target_x - ee_x, target_y - ee_y, target_z - ee_z])
        dist  = np.linalg.norm(error)
        dist_xy = np.linalg.norm(error[:2])   # solo plano horizontal

        # ── Transiciones de estado ─────────────────────────────────────────
        now     = time.time()
        timeout = self.get_parameter('approach_timeout_s').value

        if self.state == 'WAIT':
            if now - self.state_start_time > self.get_parameter('wait_time_s').value:
                self.state = 'ALIGN'
                self.state_start_time = now
                self.get_logger().info("WAIT terminado → ALIGN: alineando sobre el objeto. [START RECORDING]")
                self.buffer.clear()
                self.is_recording = True

        elif self.state == 'ALIGN':
            dist_z = abs(target_z - ee_z)
            if (dist_xy < goal_tolerance * 1.5 and dist_z < goal_tolerance) or now - self.state_start_time > timeout:
                self.state = 'DESCEND'
                self.state_start_time = now
                self.get_logger().info(
                    f"ALIGN completo (err_xy={dist_xy:.4f}m, err_z={dist_z:.4f}m) → DESCEND: bajando verticalmente.")

        elif self.state == 'DESCEND':
            dist_z = abs(target_z - ee_z)
            if dist_z < goal_tolerance or now - self.state_start_time > timeout:
                self.state = 'SETTLE'
                self.state_start_time = now
                self.grasp_position = (target_x, target_y, target_z)
                self.get_logger().info(
                    f"DESCEND en tolerancia (err_z={dist_z:.4f}m) → SETTLE: asimilando error restante.")

        elif self.state == 'SETTLE':
            if now - self.state_start_time > self.get_parameter('settle_time_s').value:
                self.state = 'GRASP'
                self.state_start_time = now
                self.get_logger().info("SETTLE completo → GRASP: cerrando gripper (brazo inmóvil).")

        elif self.state == 'GRASP':
            elapsed = now - self.state_start_time
            if elapsed > self.get_parameter('grasp_wait_time_s').value:
                self.state = 'LIFT'
                self.get_logger().info("GRASP completo → LIFT: levantando objeto.")
            elif int(elapsed * 100) % 100 == 0:
                self.get_logger().info(f"Brazo inmóvil. Esperando a que el gripper cierre físicamente... ({elapsed:.1f}s)")

        elif self.state == 'LIFT':
            if dist < goal_tolerance:
                self.get_logger().info("LIFT completo. Guardando episodio. [END RECORDING]")
                self._save_episode()
                
                if self.remaining_repetitions > 0:
                    self.remaining_repetitions -= 1
                    self.get_logger().info(f"=== REINICIANDO PARA SIGUIENTE REPETICIÓN ({self.remaining_repetitions} restantes) ===")
                    self.state = 'WAIT'
                    self.state_start_time = time.time()
                    self.grasp_position = None
                    self.target_orientation_quat = None
                    self.initial_ee_quat = None
                else:
                    self.state = 'IDLE'


        max_speed_m_s = self.get_parameter('max_speed_m_s').value
        control_rate  = self.get_parameter('control_rate_hz').value
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
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
        
        current_quat = [
            t_ee.transform.rotation.x, t_ee.transform.rotation.y,
            t_ee.transform.rotation.z, t_ee.transform.rotation.w
        ]
        
        smoothed_quat = self.pid.compute_orientation(current_quat, self.target_orientation_quat)
        
        target_pose.pose.orientation.x = smoothed_quat[0]
        target_pose.pose.orientation.y = smoothed_quat[1]
        target_pose.pose.orientation.z = smoothed_quat[2]
        target_pose.pose.orientation.w = smoothed_quat[3]
        
        self.action_pub.publish(target_pose)
        

        # Publicar el comando del gripper continuamente a 100Hz (requerido por algunos controladores/simuladores)
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [gripper_state]
        self.gripper_pub.publish(gripper_msg)

        if self.last_published_gripper_state != gripper_state:
            self.last_published_gripper_state = gripper_state
            self.get_logger().info(f"Gripper command cambiado a: {gripper_state}")

        
        self.latest_action = np.array([
            target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z,
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, 
            target_pose.pose.orientation.z, target_pose.pose.orientation.w, 
            gripper_state
        ], dtype=np.float32)
        
        self.latest_state = np.array([
            ee_x, ee_y, ee_z,
            current_quat[0], current_quat[1], current_quat[2], current_quat[3],
            gripper_state
        ], dtype=np.float32)

    def _sample_step(self):
        if not self.is_recording:
            return
            
        if self.latest_img is None or self.latest_state is None or self.latest_action is None:
            self.get_logger().warn(f"Faltan datos para grabar! img:{self.latest_img is not None}, state:{self.latest_state is not None}, action:{self.latest_action is not None}", throttle_duration_sec=1.0)
            return
            
        self.buffer.append({
            'image': self.latest_img.copy(),
            'state': self.latest_state.copy(),
            'action': self.latest_action.copy()
        })

    def _save_episode(self):
        self.is_recording = False
        if len(self.buffer) < 15:
            self.get_logger().warn(f"Secuencia insuficiente ({len(self.buffer)} pasos guardados). Episodio descartado.")
        else:
            self.storage.append_episode(self.buffer)
            self.get_logger().info(f"Episodio guardado exitosamente: {len(self.buffer)} pasos.")
        self.buffer.clear()

    def _srv_trigger(self, req, res):
        cmd = req.command.strip().upper()
        if cmd == "START" or cmd.isdigit():
            self.pid.reset()
            self.fixed_orientation = None
            self.target_orientation_quat = None
            self.initial_ee_quat = None
            self.grasp_position = None
            self.is_recording = False
            self.buffer.clear()
            self.state = 'WAIT'
            self.state_start_time = time.time()
            
            reps = int(cmd) if cmd.isdigit() else 1
            self.remaining_repetitions = reps - 1
            
            self.get_logger().info(f"=== INICIANDO SECUENCIA (Repeticiones: {reps}) ===")
            res.success = True
            res.message = f"Secuencia iniciada para {reps} iteración(es)."
        else:
            res.success = False
            res.message = "Comando desconocido. Usa START o un número entero."
        return res

def main(args=None):
    rclpy.init(args=args)
    node = PickupApproximateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
