import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray  # noqa: kept for legacy compatibility
from cv_bridge import CvBridge
import numpy as np
import random
import tf2_ros
import cv2
from tf2_ros import TransformException
import time
from scipy.stats import qmc

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
        
        self.declare_parameter('dataset_path', 'AUTO')
        self.declare_parameter('sampling_rate_hz', 20.0)
        self.declare_parameter('camera_topic', '/rgb')
        self.declare_parameter('object_frame', 'pick_target')
        self.declare_parameter('ee_frame', 'gripper_base_link')
        self.declare_parameter('tcp_offset_z', 0.17)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('max_speed_m_s', 0.15) # Límite de velocidad
        
        self.declare_parameter('kp_rot', 1.5)
        self.declare_parameter('ki_rot', 0.0)
        self.declare_parameter('kd_rot', 0.05)
        self.declare_parameter('max_speed_rad_s', 1.0)
        
        # Nuevos parámetros de la secuencia
        self.declare_parameter('init_x', 0.25)        # Posición inicial oficial en X
        self.declare_parameter('init_y', 0.0)         # Posición inicial oficial en Y
        self.declare_parameter('init_z_height', 0.40) # Altura a la que sube para enfocar
        self.declare_parameter('wait_time_s', 2.0)    # Tiempo de espera arriba
        
        # Parámetros para aleatorización del objeto (respawn)
        self.declare_parameter('random_spawn', True)
        self.declare_parameter('spawn_x_min', 0.25)
        self.declare_parameter('spawn_x_max', 0.32)
        self.declare_parameter('spawn_y_min', -0.65)
        self.declare_parameter('spawn_y_max', -1.13)
        self.declare_parameter('spawn_z', 0.44)
        self.declare_parameter('spawn_yaw_min', -1.5707)
        self.declare_parameter('spawn_yaw_max', 1.5707)
        self.declare_parameter('spawn_safe_margin_x', 0.0) # Margen de seguridad en el eje X
        self.declare_parameter('spawn_safe_margin_y', 0.0555) # Margen de seguridad en el eje Y
        self.declare_parameter('grasp_wait_time_s', 1.0) # Tiempo esperando que cierre
        self.declare_parameter('settle_time_s', 0.5)  # Tiempo inmóvil antes de cerrar
        self.declare_parameter('approach_timeout_s', 15.0) # Timeout máximo por fase
        self.declare_parameter('global_episode_timeout_s', 48.0) # Timeout global de todo el episodio
        self.declare_parameter('hover_z_offset', 0.15) # Altura extra sobre el objeto al alinearse (metros)
        self.declare_parameter('funnel_curvature_multiplier', 1.5) # Curvatura del embudo de aproximación
        self.declare_parameter('lift_z_offset', 0.35) # Altura final de levantamiento
        self.declare_parameter('validate_grasp_success', True) # Validar si el objeto fue levantado
        self.declare_parameter('grasp_success_fraction', 0.5)  # Porcentaje requerido de elevación
        self.declare_parameter('turtle_descent_multiplier', 0.10) # Freno en Z (velocidad tortuga) al rotar
        
        # Parámetros avanzados de Control y Tolerancias
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('goal_tolerance_m', 0.03)
        self.declare_parameter('gripper_open_pos', 0.0)
        self.declare_parameter('gripper_closed_pos', -0.01)
        
        dataset_path = self.get_parameter('dataset_path').value
        if dataset_path == 'AUTO':
            import glob
            import os
            import re
            
            base_dir = "datasets"
            os.makedirs(base_dir, exist_ok=True)
            existing = glob.glob(os.path.join(base_dir, "demonstrations_v*.zarr"))
            max_v = 1
            for path in existing:
                match = re.search(r"demonstrations_v(\d+)\.zarr", path)
                if match:
                    v = int(match.group(1))
                    if v > max_v:
                        max_v = v
            
            # Start at v2 by default since v1 is demonstrations.zarr
            dataset_path = os.path.join(base_dir, f"demonstrations_v{max_v + 1}.zarr")
            self.get_logger().info(f"Modo AUTO: Creado nuevo dataset aislado en {dataset_path}")
        else:
            self.get_logger().info(f"Usando dataset especificado: {dataset_path}")
        
        self.storage = ZarrStorage(dataset_path)
        
        self.buffer = []
        self.latest_img = None
        self.latest_state = None
        self.latest_action = None

        camera_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, camera_topic, self._img_cb, 10, callback_group=self.cbg)
        self.action_pub = self.create_publisher(PoseStamped, 'target_frame', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.respawn_pub = self.create_publisher(Twist, '/respawn', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pid = CartesianPID(sampling_rate_hz=self.get_parameter('control_rate_hz').value)
        
        self.fixed_orientation = None
        self.grasp_position = None  # Posición XYZ exacta donde se cierra el gripper
        self.grasp_orientation_quat = None
        self.wait_reached_time = None
        self.untilt_reached_time = None
        self.episode_start_time = None
        
        # Máquina de estados: IDLE, WAIT, ALIGN, DESCEND, SETTLE, GRASP, LIFT, UNTILT
        self.state = 'IDLE'
        self.state_start_time = 0.0
        self.last_published_gripper_state = None
        self.is_recording = False
        self.remaining_repetitions = 0
        self.target_orientation_quat = None
        self.initial_ee_quat = None
        self.has_respawned_this_wait = False
        
        self.noise_offset = np.zeros(3)
        self.halton_obj = qmc.Halton(d=3, scramble=True)
        self.halton_robot = qmc.Halton(d=2, scramble=True)
        
        self.current_init_x = self.get_parameter('init_x').value
        self.current_init_y = self.get_parameter('init_y').value
        self.current_jitter_roll = 0.0
        self.current_jitter_pitch = 0.0
        self.current_jitter_yaw = 0.0

        self.create_service(EpisodeTrigger, '~/trigger_episode', self._srv_trigger, callback_group=self.cbg)

        # Timer para el control a 100 Hz (frecuencia de FZI)
        self.control_timer = self.create_timer(1.0 / self.get_parameter('control_rate_hz').value, self._control_step, callback_group=self.cbg)

        # Timer para recolección de datos a 20 Hz
        rate = self.get_parameter('sampling_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self._sample_step, callback_group=self.cbg)
        
        max_speed = self.get_parameter('max_speed_m_s').value
        self.get_logger().info(f"Pickup & Approximate Sequence Node Inicializado. Velocidad max: {max_speed} m/s")

    def _img_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # Reducir drásticamente el peso del dataset (de 33GB a ~2GB) redimensionando en origen
        self.latest_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

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

        # ── 1. Orientación target ───────────────────────────────────────────
        if self.initial_ee_quat is None:
            self.initial_ee_quat = [
                t_ee.transform.rotation.x, t_ee.transform.rotation.y,
                t_ee.transform.rotation.z, t_ee.transform.rotation.w
            ]
            
        if self.target_orientation_quat is None:
            self.target_orientation_quat = self.initial_ee_quat.copy()

        if self.state in ('IDLE', 'WAIT', 'APPROACH_XY', 'LIFT', 'UNTILT'):
            # Regresar a la postura inicial pero aplicando el jitter calculado para esta demostración usando cuaterniones directos
            jitter_q = quaternion_from_euler(
                self.current_jitter_roll,
                self.current_jitter_pitch,
                self.current_jitter_yaw
            )
            # multiply_quaternions(rotacion_local, rotacion_base)
            self.target_orientation_quat = multiply_quaternions(jitter_q, self.initial_ee_quat)

        if self.state in ('ALIGN_YAW', 'DESCEND'):
            # Obtenemos la orientación completa del objeto (pick_target)
            q_obj = [
                t_obj.transform.rotation.x, t_obj.transform.rotation.y,
                t_obj.transform.rotation.z, t_obj.transform.rotation.w
            ]
            obj_roll, obj_pitch, obj_yaw = euler_from_quaternion(q_obj)
            
            # Usamos el Roll, Pitch y Yaw directamente del pick_target
            target_roll = obj_roll
            target_pitch = obj_pitch
            target_yaw = obj_yaw
            
            self.target_orientation_quat = quaternion_from_euler(target_roll, target_pitch, target_yaw)


        # ── 2. Determinar posición target según estado ──────────────────────
        gripper_state = gripper_open_pos

        if self.state in ('IDLE', 'WAIT'):
            # Posición de reposo / inicio aleatorizada
            target_x = self.current_init_x
            target_y = self.current_init_y
            target_z = self.get_parameter('init_z_height').value

        elif self.state == 'APPROACH_XY':
            # Fase 1: Alineamiento XY manteniendo altura hover
            qx, qy, qz, qw = self.target_orientation_quat
            
            # Trayectoria de Embudo: Z proporcional al error en XY respecto al objeto
            funnel_mult = self.get_parameter('funnel_curvature_multiplier').value
            dist_to_obj_xy = np.linalg.norm([obj_x - ee_x, obj_y - ee_y])
            funnel_z_offset = hover_z_offset + (dist_to_obj_xy * funnel_mult)
            d = -(tcp_offset_z + funnel_z_offset)
            
            dx = d * 2.0 * (qx*qz + qw*qy)
            dy = d * 2.0 * (qy*qz - qw*qx)
            dz = d * (1.0 - 2.0*(qx*qx + qy*qy))
            
            target_x = obj_x + dx
            target_y = obj_y + dy
            target_z = obj_z + dz

        elif self.state in ('ALIGN_YAW', 'DESCEND'):
            # Fase 2 y 3: Bajar apuntando al objeto. ALIGN_YAW gira mientras baja lento.
            qx, qy, qz, qw = self.target_orientation_quat
            d = -tcp_offset_z
            
            dx = d * 2.0 * (qx*qz + qw*qy)
            dy = d * 2.0 * (qy*qz - qw*qx)
            dz = d * (1.0 - 2.0*(qx*qx + qy*qy))
            
            target_x = obj_x + dx
            target_y = obj_y + dy
            target_z = obj_z + dz

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

        elif self.state in ('LIFT', 'UNTILT'):
            if self.grasp_position is not None:
                target_x = self.grasp_position[0]
                target_y = self.grasp_position[1]
                target_z = self.grasp_position[2] + self.get_parameter('lift_z_offset').value
            else:
                target_x = ee_x
                target_y = ee_y
                target_z = ee_z + self.get_parameter('lift_z_offset').value
            gripper_state = gripper_closed_pos

        # ── Error 3D y distancias parciales ────────────────────────────────
        error = np.array([target_x - ee_x, target_y - ee_y, target_z - ee_z])
        dist  = np.linalg.norm(error)
        dist_xy = np.linalg.norm(error[:2])   # solo plano horizontal

        current_quat = [
            t_ee.transform.rotation.x, t_ee.transform.rotation.y,
            t_ee.transform.rotation.z, t_ee.transform.rotation.w
        ]
        
        if self.target_orientation_quat is not None:
            target_quat = self.target_orientation_quat
        else:
            target_quat = current_quat
            
        # Quaternion dot product for rotational error
        dot_product = np.clip(np.abs(np.sum(np.array(current_quat) * np.array(target_quat))), -1.0, 1.0)
        error_rot_mag = 2.0 * np.arccos(dot_product)

        # ── Transiciones de estado ─────────────────────────────────────────
        now     = time.time()
        timeout = self.get_parameter('approach_timeout_s').value
        global_timeout = self.get_parameter('global_episode_timeout_s').value
        
        # Check global timeout
        if self.episode_start_time is not None:
            if now - self.episode_start_time > global_timeout:
                self.get_logger().warn(f"¡Timeout global superado ({global_timeout}s)! Episodio atascado, descartando y reiniciando.")
                self.is_recording = False
                self.buffer.clear()
                if self.remaining_repetitions > 0:
                    self.remaining_repetitions -= 1
                    self.get_logger().info(f"=== REINICIANDO PARA SIGUIENTE REPETICIÓN ({self.remaining_repetitions} restantes) ===")
                    self.state = 'WAIT'
                    self.state_start_time = time.time()
                    self.has_respawned_this_wait = False
                else:
                    self.state = 'IDLE'
                self.grasp_position = None
                self.grasp_orientation_quat = None
                self.target_orientation_quat = None
                self.wait_reached_time = None
                self.untilt_reached_time = None
                self.episode_start_time = None
                self.pid.reset()
                return


        if self.state == 'WAIT':
            # Local timeout for returning to home to avoid getting permanently stuck
            if now - self.state_start_time > timeout * 3.0:
                self.get_logger().warn(f"¡Timeout en WAIT! Robot atascado intentando volver a home. Reiniciando WAIT.")
                self.state_start_time = now
                self.has_respawned_this_wait = False
                self.wait_reached_time = None
                self.pid.reset()
                return

            # Teletransportar el objeto a mitad del WAIT (da 1 segundo para soltar, y N segundos para asentar)
            if now - self.state_start_time > 1.0 and not self.has_respawned_this_wait:
                self._respawn_object()
                self.has_respawned_this_wait = True
                
            if dist < goal_tolerance * 3.0:  # Ignoramos la rotación para no atascarnos
                if self.wait_reached_time is None:
                    self.wait_reached_time = now
                    
                if now - self.wait_reached_time > self.get_parameter('wait_time_s').value + 1.0:
                    self.state = 'APPROACH_XY'
                    self.state_start_time = now
                    self.wait_reached_time = None
                    self.get_logger().info("WAIT terminado → APPROACH_XY: Centrando sobre el objeto. [START RECORDING]")
                    self.buffer.clear()
                    self.is_recording = True
                    self.episode_start_time = now

        elif self.state == 'APPROACH_XY':
            dist_z = abs(target_z - ee_z)
            # Solo exigimos tolerancia en XY y Z, ignoramos la rotación porque aún no giramos
            if (dist_xy < goal_tolerance * 1.5 and dist_z < goal_tolerance) or now - self.state_start_time > timeout:
                self.state = 'ALIGN_YAW'
                self.state_start_time = now
                self.get_logger().info(
                    f"APPROACH_XY completo (err_xy={dist_xy:.4f}m, err_z={dist_z:.4f}m) → ALIGN_YAW: Rotando para alinear orientación.")

        elif self.state == 'ALIGN_YAW':
            # Ahora exigimos precisión en la rotación antes de descender a tocar el objeto
            if error_rot_mag < 0.05 or now - self.state_start_time > timeout:
                self.state = 'DESCEND'
                self.state_start_time = now
                self.get_logger().info(
                    f"ALIGN_YAW completo (err_rot={error_rot_mag:.4f}rad) → DESCEND: bajando verticalmente.")

        elif self.state == 'DESCEND':
            # Para DESCEND usamos la distancia 3D real porque la aproximación puede ser diagonal
            if (dist < goal_tolerance and error_rot_mag < 0.05) or now - self.state_start_time > 4.0:
                self.state = 'SETTLE'
                self.state_start_time = now
                self.grasp_position = (target_x, target_y, target_z)
                self.grasp_orientation_quat = self.target_orientation_quat.copy()
                self.get_logger().info(
                    f"DESCEND completado/forzado (err_3d={dist:.4f}m, err_rot={error_rot_mag:.4f}rad) → SETTLE: asimilando error restante.")

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
            if dist < goal_tolerance * 3.0 or now - self.state_start_time > 5.0:
                self.state = 'UNTILT'
                self.state_start_time = now
                self.get_logger().info("LIFT completo → UNTILT: Enderezando orientación del objeto.")
                
        elif self.state == 'UNTILT':
            # Esperar a que la orientación vuelva a estar recta y la posición se mantenga (con timeout de 5s para no atascarse)
            if (dist < goal_tolerance * 5.0 and error_rot_mag < 0.25) or now - self.state_start_time > 5.0:
                if self.untilt_reached_time is None:
                    self.untilt_reached_time = now
                
                # Pequeño settling time de 0.5s para asegurar que se grabe la postura recta
                if now - self.untilt_reached_time > 0.5:
                    is_valid = True
                    if self.get_parameter('validate_grasp_success').value and self.grasp_position is not None:
                        # Validación porcentual respecto a la altura objetivo de levantamiento
                        lift_offset = self.get_parameter('lift_z_offset').value
                        success_fraction = self.get_parameter('grasp_success_fraction').value
                        required_z_travel = abs(lift_offset) * success_fraction
                        actual_z_travel = abs(obj_z - self.grasp_position[2])
                        
                        if actual_z_travel < required_z_travel:
                            is_valid = False
                            self.get_logger().warn(
                                f"¡Fallo de agarre detectado! El objeto se movió solo {actual_z_travel:.3f}m en Z. Se requería mínimo {required_z_travel:.3f}m. Episodio descartado."
                            )
                    
                    if is_valid:
                        self.get_logger().info("UNTILT completo y validado. Guardando episodio. [END RECORDING]")
                        self._save_episode()
                    else:
                        self.is_recording = False
                        self.buffer.clear()
                    
                    if self.remaining_repetitions > 0:
                        self.remaining_repetitions -= 1
                        self.get_logger().info(f"=== REINICIANDO PARA SIGUIENTE REPETICIÓN ({self.remaining_repetitions} restantes) ===")
                        self.state = 'WAIT'
                        self.state_start_time = time.time()
                        self.has_respawned_this_wait = False
                        self.grasp_position = None
                        self.grasp_orientation_quat = None
                        self.target_orientation_quat = None
                        self.wait_reached_time = None
                        self.untilt_reached_time = None
                        self.episode_start_time = None
                        self.pid.reset()
                        return
                    else:
                        self.state = 'IDLE'
                        self.target_orientation_quat = None
                        self.grasp_orientation_quat = None
                        self.wait_reached_time = None
                        self.untilt_reached_time = None
                        self.episode_start_time = None
                        self.pid.reset()
                        return


        max_speed_m_s = self.get_parameter('max_speed_m_s').value
        control_rate  = self.get_parameter('control_rate_hz').value
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        
        kp_rot = self.get_parameter('kp_rot').value
        ki_rot = self.get_parameter('ki_rot').value
        kd_rot = self.get_parameter('kd_rot').value
        
        # 1. Rotaciones globales usando el límite estricto del YAML
        max_speed_rad_s = self.get_parameter('max_speed_rad_s').value
        
        # 2. Freno dinámico en Z mientras rota (para evitar desenfoque de cámara y permitir giro de 180 sin chocar)
        if self.state in ['APPROACH_XY', 'ALIGN_YAW', 'DESCEND']:
            if self.state == 'ALIGN_YAW':
                # Freno extremo para inyectar un pequeñísimo impulso de bajada constante
                # mientras completa el giro completo de la muñeca (ej. 180 grados)
                turtle_mult = self.get_parameter('turtle_descent_multiplier').value
                max_speed_m_s = max_speed_m_s * turtle_mult
            else:
                if error_rot_mag > 0.15:
                    max_speed_m_s = max_speed_m_s * 0.30
                elif error_rot_mag > 0.05:
                    max_speed_m_s = max_speed_m_s * 0.60
        
        self.pid.sampling_rate_hz = control_rate
        self.pid.update_params(kp, ki, kd, max_speed_m_s, kp_rot, ki_rot, kd_rot, max_speed_rad_s)
        
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
        
        if self.state in ('SETTLE', 'GRASP') and self.grasp_orientation_quat is not None:
            # Detener el PID de orientación y congelar la postura en el punto de agarre
            smoothed_quat = self.grasp_orientation_quat
        else:
            smoothed_quat = self.pid.compute_orientation(current_quat, target_quat)
        
        target_pose.pose.orientation.x = smoothed_quat[0]
        target_pose.pose.orientation.y = smoothed_quat[1]
        target_pose.pose.orientation.z = smoothed_quat[2]
        target_pose.pose.orientation.w = smoothed_quat[3]
        
        # DAgger / Proprioception Noise Injection
        import copy
        published_pose = copy.deepcopy(target_pose)
        
        if self.state in ('DESCEND', 'ALIGN'):
            # Random walk noise decay
            self.noise_offset *= 0.95
            # Inject new gaussian step
            self.noise_offset += np.random.normal(0, 0.001, 3)
            
            published_pose.pose.position.x += float(self.noise_offset[0])
            published_pose.pose.position.y += float(self.noise_offset[1])
            published_pose.pose.position.z += float(self.noise_offset[2])
        else:
            self.noise_offset = np.zeros(3)
        
        self.action_pub.publish(published_pose)
        

        # Publicar el comando del gripper continuamente a 100Hz (requerido por algunos controladores/simuladores)
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [gripper_state]
        self.gripper_pub.publish(gripper_msg)

        if self.last_published_gripper_state != gripper_state:
            self.last_published_gripper_state = gripper_state
            self.get_logger().info(f"Gripper command cambiado a: {gripper_state}")

        
        # Solución al "Quaternion Double Cover": 
        # Asegurar que el componente W siempre sea positivo para evitar que la red neuronal 
        # colapse al promediar rotaciones de 180 grados (q vs -q son la misma rotación física).
        target_q = [target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        if target_q[3] < 0:
            target_q = [-target_q[0], -target_q[1], -target_q[2], -target_q[3]]
            
        current_q = [current_quat[0], current_quat[1], current_quat[2], current_quat[3]]
        if current_q[3] < 0:
            current_q = [-current_q[0], -current_q[1], -current_q[2], -current_q[3]]
        
        self.latest_action = np.array([
            target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z,
            target_q[0], target_q[1], target_q[2], target_q[3], 
            gripper_state
        ], dtype=np.float32)
        
        self.latest_state = np.array([
            ee_x, ee_y, ee_z,
            current_q[0], current_q[1], current_q[2], current_q[3],
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

    def _respawn_object(self):
        if not self.get_parameter('random_spawn').value:
            return
            
        spawn_x_min = self.get_parameter('spawn_x_min').value
        spawn_x_max = self.get_parameter('spawn_x_max').value
        spawn_y_min = self.get_parameter('spawn_y_min').value
        spawn_y_max = self.get_parameter('spawn_y_max').value
        spawn_z     = self.get_parameter('spawn_z').value
        spawn_yaw_min = self.get_parameter('spawn_yaw_min').value
        spawn_yaw_max = self.get_parameter('spawn_yaw_max').value
        safe_margin_x = self.get_parameter('spawn_safe_margin_x').value
        safe_margin_y = self.get_parameter('spawn_safe_margin_y').value
        
        # Secuencia Quasi-Monte Carlo (Halton) para el objeto (3D: X, Y, Yaw)
        halton_sample = self.halton_obj.random(1)[0]
        
        target_x = spawn_x_min + halton_sample[0] * (spawn_x_max - spawn_x_min)
        target_y = spawn_y_min + halton_sample[1] * (spawn_y_max - spawn_y_min)
        
        # Verificar si el objeto está demasiado cerca del borde de caída
        dist_x = min(abs(target_x - spawn_x_min), abs(target_x - spawn_x_max))
        dist_y = min(abs(target_y - spawn_y_min), abs(target_y - spawn_y_max))
        
        if dist_x < safe_margin_x or dist_y < safe_margin_y:
            target_yaw = 0.0  # Prohibido rotar en el borde para evitar caída
        else:
            target_yaw = spawn_yaw_min + halton_sample[2] * (spawn_yaw_max - spawn_yaw_min)
        
        twist_msg = Twist()
        twist_msg.linear.x = float(target_x)
        twist_msg.linear.y = float(target_y)
        twist_msg.linear.z = float(spawn_z)
        twist_msg.angular.z = float(target_yaw)
        
        self.respawn_pub.publish(twist_msg)
        self.get_logger().info(f"Objeto re-posicionado (Halton): X={target_x:.3f}, Y={target_y:.3f}, Yaw={target_yaw:.2f}")

        # Randomización de la pose inicial del EE (Robot) con Halton (2D: X, Y)
        # Randomizaremos la base alrededor del punto inicial predeterminado
        base_init_x = self.get_parameter('init_x').value
        base_init_y = self.get_parameter('init_y').value
        
        ee_halton = self.halton_robot.random(1)[0]
        self.current_init_x = base_init_x + (ee_halton[0] * 0.1 - 0.05) # +/- 5cm
        self.current_init_y = base_init_y + (ee_halton[1] * 0.2 - 0.1)  # +/- 10cm
        
        self.current_jitter_roll = np.random.uniform(-0.1, 0.1)
        self.current_jitter_pitch = np.random.uniform(-0.1, 0.1)
        self.current_jitter_yaw = np.random.uniform(-0.1, 0.1)
        
        # Orientation down is [0, 1, 0, 0] or something similar? Actually, the initial_ee_quat is captured at runtime.
        # We'll just add this jitter to the PID target in the IDLE state.

    def _srv_trigger(self, req, res):
        cmd = req.command.strip().upper()
        if cmd == "START" or cmd.isdigit():
            self.pid.reset()
            self.fixed_orientation = None
            self.target_orientation_quat = None
            self.grasp_position = None
            self.grasp_orientation_quat = None
            self.wait_reached_time = None
            self.untilt_reached_time = None
            self.episode_start_time = None
            self.is_recording = False
            self.buffer.clear()
            self.state = 'WAIT'
            self.state_start_time = time.time()
            self.has_respawned_this_wait = False
            
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
