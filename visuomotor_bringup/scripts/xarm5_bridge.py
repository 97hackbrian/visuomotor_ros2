#!/usr/bin/env python3
"""Puente seguro entre la política visuomotora y el xArm5 real (driver ``xarm_api`` de UFACTORY).

Flujo de datos::

    policy_node --(target_frame_raw)--> pose_smoother --(target_frame)--> xarm5_bridge
        --> /<hw_ns>/set_servo_cartesian   (modo 1, servo cartesiano)
    policy_node --(/position_controller/commands)--> xarm5_bridge
        --> acción xarm_gripper/gripper_action (gripper xArm)  |  /<hw_ns>/set_vacuum_gripper (ventosa)
    /<hw_ns>/robot_states  --> TF  base_frame -> tcp_frame        (estado para la política)
    /<hw_ns>/joint_states  --> /visuomotor/gripper_joint_states   (apertura en la escala de la política)

Seguridad:
* Arranca DESARMADO: no envía movimiento hasta llamar a ``~/arm`` (std_srvs/Trigger).
* ``dry_run: true`` nunca llama a servicios de movimiento ni de gripper (solo publica ``~/command_preview``).
* Caja de trabajo + límites radiales, yaw acotado sin cruzar ±pi, límite de velocidad lineal/angular,
  rechazo de objetivos lejanos o no finitos, watchdog de objetivo y de respuesta del controlador,
  desarme automático si el controlador reporta error, se pierde su estado o fallan llamadas seguidas.

Gripper xArm: se usa la ACCIÓN ``GripperCommand`` (no el servicio ``set_gripper_position``) porque el driver solo
publica la posición del gripper de forma continua con firmware >= 2.7.101; la acción la publica durante el
movimiento con cualquier firmware (es la interfaz que ya se usó con este brazo en xarm_examples/Nodo_gripper.py).

Todos los valores numéricos, tópicos y nombres se leen de
``visuomotor_bringup/config/xarm5/xarm5_real_params.yaml``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from visuomotor_bringup.xarm5_interface import (
    GRIPPER_TYPES,
    ORIENTATION_MODES,
    GripperBinarizer,
    WorkspaceLimits,
    clamp_yaw,
    from_robot_pose,
    is_finite_pose,
    limit_angle_step,
    limit_step,
    linear_map,
    project_orientation,
    rpy_to_quat,
    to_servo_pose,
)

try:  # xarm_msgs / control_msgs solo existen donde está compilado el driver (imagen Jetson)
    from control_msgs.action import GripperCommand
    from xarm_msgs.msg import RobotMsg
    from xarm_msgs.srv import Call, MoveCartesian, MoveJoint, SetFloat32, SetInt16, SetInt16ById
    from xarm_msgs.srv import VacuumGripperCtrl
    _XARM_MSGS_ERROR = None
except ImportError as exc:  # pragma: no cover - depende del entorno
    _XARM_MSGS_ERROR = exc

XARM_MODE_POSITION = 0
XARM_MODE_SERVO = 1
XARM_STATE_READY = 0
MOTION_ENABLE_ALL_JOINTS = 8


class XArm5Bridge(Node):
    def __init__(self):
        super().__init__("xarm5_bridge")
        if _XARM_MSGS_ERROR is not None:
            raise RuntimeError(
                f"No se pudo importar xarm_msgs/control_msgs ({_XARM_MSGS_ERROR}). "
                "Compila xarm_ros2 (rama humble) y control_msgs en el workspace."
            )
        self._declare_parameters()
        self._read_parameters()

        self._lock = threading.Lock()
        self._armed = False
        self._cmd_xyz: Optional[np.ndarray] = None
        self._cmd_rpy: Optional[list] = None
        self._target: Optional[PoseStamped] = None
        self._target_time = 0.0
        self._measured_xyz: Optional[np.ndarray] = None
        self._measured_rpy: Optional[tuple] = None
        self._robot_state_time = 0.0
        self._robot_err = 0
        self._robot_warn = 0
        self._robot_mode = -1
        self._robot_state = -1
        self._servo_pending = False
        self._servo_sent_time = 0.0
        self._servo_seq = 0
        self._last_cmd_time = 0.0
        self._consecutive_failures = 0
        self._gripper = GripperBinarizer(self.gripper_close_threshold, self.gripper_hysteresis)
        self._gripper_closed_cmd = False
        self._gripper_real_pos: Optional[float] = None
        self._gripper_goal_handle = None
        self._last_gripper_send = 0.0

        # Grupos: el control y la publicación de estado van separados para que una llamada bloqueante
        # (p. ej. al desarmar) no detenga la TF que necesita la política.
        self._cb_ctrl = MutuallyExclusiveCallbackGroup()
        self._cb_state = MutuallyExclusiveCallbackGroup()
        self._cb_io = ReentrantCallbackGroup()
        self._cb_srv = MutuallyExclusiveCallbackGroup()

        ns = "/" + self.hw_ns.strip("/") if self.hw_ns.strip("/") else ""
        mk = lambda srv_type, name: self.create_client(srv_type, f"{ns}/{name}", callback_group=self._cb_io)
        self._cli_motion_enable = mk(SetInt16ById, "motion_enable")
        self._cli_set_mode = mk(SetInt16, "set_mode")
        self._cli_set_state = mk(SetInt16, "set_state")
        self._cli_clean_error = mk(Call, "clean_error")
        self._cli_servo_cartesian = mk(MoveCartesian, "set_servo_cartesian")
        self._cli_servo_angle = mk(MoveJoint, "set_servo_angle")
        self._cli_gripper_enable = mk(SetInt16, "set_gripper_enable")
        self._cli_gripper_mode = mk(SetInt16, "set_gripper_mode")
        self._cli_gripper_speed = mk(SetFloat32, "set_gripper_speed")
        self._cli_vacuum = mk(VacuumGripperCtrl, "set_vacuum_gripper")
        self._gripper_action = ActionClient(self, GripperCommand, self.xarm_gripper_action_name,
                                            callback_group=self._cb_io)

        self.create_subscription(PoseStamped, self.target_topic, self._target_cb, 1, callback_group=self._cb_io)
        self.create_subscription(Float64MultiArray, self.gripper_command_topic, self._gripper_cmd_cb, 10,
                                 callback_group=self._cb_io)
        if not self.simulate_robot_state:
            self.create_subscription(RobotMsg, self.robot_states_topic, self._robot_state_cb, 10,
                                     callback_group=self._cb_io)
            self.create_subscription(JointState, self.robot_joint_states_topic, self._joint_states_cb, 10,
                                     callback_group=self._cb_io)

        self._pub_gripper_state = self.create_publisher(JointState, self.gripper_state_topic, 10)
        self._pub_preview = self.create_publisher(PoseStamped, "~/command_preview", 10)
        self._pub_status = self.create_publisher(String, "~/status", 10)
        self._tf = TransformBroadcaster(self)

        self.create_service(Trigger, "~/arm", self._srv_arm, callback_group=self._cb_srv)
        self.create_service(Trigger, "~/disarm", self._srv_disarm, callback_group=self._cb_srv)
        self.create_service(Trigger, "~/go_home", self._srv_go_home, callback_group=self._cb_srv)

        if self.simulate_robot_state:
            self._measured_xyz = np.array(self.sim_initial_xyz, dtype=np.float64)
            self._measured_rpy = tuple(self.sim_initial_rpy)
            self._robot_state_time = time.time()
            self._robot_mode, self._robot_state = XARM_MODE_POSITION, XARM_STATE_READY

        self.create_timer(1.0 / self.control_rate_hz, self._control_step, callback_group=self._cb_ctrl)
        self.create_timer(1.0 / self.state_publish_rate_hz, self._publish_state, callback_group=self._cb_state)
        self.create_timer(1.0 / self.status_rate_hz, self._publish_status, callback_group=self._cb_state)

        mode = "DRY-RUN (sin comandos al robot)" if self.dry_run else "REAL"
        state = "SIMULADO" if self.simulate_robot_state else f"robot ({self.robot_states_topic})"
        self.get_logger().info(
            f"xarm5_bridge listo | modo={mode} | estado={state} | gripper={self.gripper_type} | "
            f"orientación={self.orientation_mode} | DESARMADO: llama a ~/arm para habilitar movimiento."
        )

    # ------------------------------------------------------------------------------------------
    # Parámetros
    # ------------------------------------------------------------------------------------------
    def _declare_parameters(self):
        d = self.declare_parameter
        d("hw_ns", "xarm")
        d("dry_run", True)
        d("simulate_robot_state", False)
        d("control_rate_hz", 30.0)
        d("state_publish_rate_hz", 30.0)
        d("status_rate_hz", 1.0)
        d("service_timeout_s", 3.0)
        d("servo_response_timeout_s", 0.5)
        d("robot_state_timeout_s", 1.0)
        d("target_topic", "/target_frame")
        d("target_timeout_s", 0.5)
        d("robot_states_topic", "/xarm/robot_states")
        d("robot_joint_states_topic", "/xarm/joint_states")
        d("base_frame", "link_base")
        d("tcp_frame", "xarm_tcp")
        d("publish_tcp_tf", True)
        d("orientation_mode", "top_down_yaw")
        d("top_down_roll", math.pi)
        d("top_down_pitch", 0.0)
        d("fixed_yaw", 0.0)
        d("yaw_offset", 0.0)
        d("yaw_min", -2.6)
        d("yaw_max", 2.6)
        d("workspace_x_min", 0.15)
        d("workspace_x_max", 0.55)
        d("workspace_y_min", -0.30)
        d("workspace_y_max", 0.30)
        d("workspace_z_min", 0.10)
        d("workspace_z_max", 0.50)
        d("workspace_min_radius_xy", 0.15)
        d("workspace_max_radius_xy", 0.60)
        d("max_linear_speed_m_s", 0.05)
        d("max_angular_speed_rad_s", 0.5)
        d("max_target_distance_m", 0.25)
        d("servo_speed", 0.0)
        d("servo_acc", 0.0)
        d("max_consecutive_failures", 5)
        d("gripper_type", "none")
        d("gripper_command_topic", "/position_controller/commands")
        d("gripper_state_topic", "/visuomotor/gripper_joint_states")
        d("gripper_state_joint_name", "gripper")
        d("gripper_close_threshold", -0.001)
        d("gripper_hysteresis", 0.0005)
        d("gripper_policy_open_pos", 0.0)
        d("gripper_policy_closed_pos", -0.01)
        d("gripper_min_command_interval_s", 0.5)
        d("xarm_gripper_action_name", "/xarm_gripper/gripper_action")
        d("xarm_gripper_speed", 1500.0)
        d("xarm_gripper_joint_name", "drive_joint")
        d("xarm_gripper_joint_open", 0.0)
        d("xarm_gripper_joint_closed", 0.85)
        d("home_joints_rad", [0.0, -0.82, -0.28, 1.11, 0.0])
        d("home_speed_rad_s", 0.35)
        d("home_acc_rad_s2", 10.0)
        d("home_timeout_s", 30.0)
        d("sim_initial_xyz", [0.25, 0.0, 0.40])
        d("sim_initial_rpy", [math.pi, 0.0, 0.0])

    def _read_parameters(self):
        g = lambda name: self.get_parameter(name).value
        self.hw_ns = g("hw_ns")
        self.dry_run = bool(g("dry_run"))
        self.simulate_robot_state = bool(g("simulate_robot_state"))
        self.control_rate_hz = float(g("control_rate_hz"))
        self.state_publish_rate_hz = float(g("state_publish_rate_hz"))
        self.status_rate_hz = float(g("status_rate_hz"))
        self.service_timeout_s = float(g("service_timeout_s"))
        self.servo_response_timeout_s = float(g("servo_response_timeout_s"))
        self.robot_state_timeout_s = float(g("robot_state_timeout_s"))
        self.target_topic = g("target_topic")
        self.target_timeout_s = float(g("target_timeout_s"))
        self.robot_states_topic = g("robot_states_topic")
        self.robot_joint_states_topic = g("robot_joint_states_topic")
        self.base_frame = g("base_frame")
        self.tcp_frame = g("tcp_frame")
        self.publish_tcp_tf = bool(g("publish_tcp_tf"))
        self.orientation_mode = g("orientation_mode")
        self.fixed_rpy = [float(g("top_down_roll")), float(g("top_down_pitch")), float(g("fixed_yaw"))]
        self.yaw_offset = float(g("yaw_offset"))
        self.yaw_min = float(g("yaw_min"))
        self.yaw_max = float(g("yaw_max"))
        self.workspace = WorkspaceLimits(
            g("workspace_x_min"), g("workspace_x_max"), g("workspace_y_min"), g("workspace_y_max"),
            g("workspace_z_min"), g("workspace_z_max"), g("workspace_min_radius_xy"), g("workspace_max_radius_xy"))
        self.max_linear_speed = float(g("max_linear_speed_m_s"))
        self.max_angular_speed = float(g("max_angular_speed_rad_s"))
        self.max_target_distance = float(g("max_target_distance_m"))
        self.servo_speed = float(g("servo_speed"))
        self.servo_acc = float(g("servo_acc"))
        self.max_consecutive_failures = int(g("max_consecutive_failures"))
        self.gripper_type = g("gripper_type")
        self.gripper_command_topic = g("gripper_command_topic")
        self.gripper_state_topic = g("gripper_state_topic")
        self.gripper_state_joint_name = g("gripper_state_joint_name")
        self.gripper_close_threshold = float(g("gripper_close_threshold"))
        self.gripper_hysteresis = float(g("gripper_hysteresis"))
        self.gripper_policy_open = float(g("gripper_policy_open_pos"))
        self.gripper_policy_closed = float(g("gripper_policy_closed_pos"))
        self.gripper_min_interval = float(g("gripper_min_command_interval_s"))
        self.xarm_gripper_action_name = g("xarm_gripper_action_name")
        self.xarm_gripper_speed = float(g("xarm_gripper_speed"))
        self.xarm_gripper_joint_name = g("xarm_gripper_joint_name")
        self.xarm_gripper_joint_open = float(g("xarm_gripper_joint_open"))
        self.xarm_gripper_joint_closed = float(g("xarm_gripper_joint_closed"))
        self.home_joints = [float(v) for v in g("home_joints_rad")]
        self.home_speed = float(g("home_speed_rad_s"))
        self.home_acc = float(g("home_acc_rad_s2"))
        self.home_timeout = float(g("home_timeout_s"))
        self.sim_initial_xyz = [float(v) for v in g("sim_initial_xyz")]
        self.sim_initial_rpy = [float(v) for v in g("sim_initial_rpy")]

        if self.orientation_mode not in ORIENTATION_MODES:
            raise ValueError(f"orientation_mode={self.orientation_mode!r} inválido. Opciones: {ORIENTATION_MODES}")
        if self.gripper_type not in GRIPPER_TYPES:
            raise ValueError(f"gripper_type={self.gripper_type!r} inválido. Opciones: {GRIPPER_TYPES}")
        if self.simulate_robot_state and not self.dry_run:
            raise ValueError("simulate_robot_state=true exige dry_run=true (no se puede mover un robot simulado).")
        clamp_yaw(0.0, self.yaw_min, self.yaw_max)  # valida el rango (lanza ValueError si es inválido)
        for name, val in (("control_rate_hz", self.control_rate_hz), ("max_linear_speed_m_s", self.max_linear_speed),
                          ("max_angular_speed_rad_s", self.max_angular_speed),
                          ("servo_response_timeout_s", self.servo_response_timeout_s)):
            if val <= 0.0:
                raise ValueError(f"{name} debe ser > 0 (llegó {val})")

    # ------------------------------------------------------------------------------------------
    # Utilidades de servicios
    # ------------------------------------------------------------------------------------------
    def _call(self, client, request, timeout: Optional[float] = None):
        """Llamada síncrona segura desde un callback (no usa spin_until_future_complete)."""
        timeout = self.service_timeout_s if timeout is None else timeout
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Servicio no disponible: {client.srv_name}")
            return None
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            self.get_logger().error(f"Timeout ({timeout:.1f} s) llamando a {client.srv_name}")
            return None
        return future.result()

    def _call_ok(self, client, request, what: str, timeout: Optional[float] = None) -> bool:
        res = self._call(client, request, timeout)
        if res is None:
            return False
        if res.ret != 0:
            self.get_logger().error(f"{what} devolvió ret={res.ret} ({getattr(res, 'message', '')})")
            return False
        return True

    def _set_mode_state(self, mode: int) -> bool:
        req_e = SetInt16ById.Request()
        req_e.id, req_e.data = MOTION_ENABLE_ALL_JOINTS, 1
        req_m = SetInt16.Request()
        req_m.data = mode
        req_s = SetInt16.Request()
        req_s.data = XARM_STATE_READY
        return (self._call_ok(self._cli_motion_enable, req_e, "motion_enable")
                and self._call_ok(self._cli_set_mode, req_m, f"set_mode({mode})")
                and self._call_ok(self._cli_set_state, req_s, "set_state(0)"))

    # ------------------------------------------------------------------------------------------
    # Servicios propios: arm / disarm / go_home
    # ------------------------------------------------------------------------------------------
    def _robot_state_fresh(self) -> bool:
        if self._measured_xyz is None:
            return False
        if self.simulate_robot_state:
            return True
        return (time.time() - self._robot_state_time) < self.robot_state_timeout_s

    def _gripper_measured_closed(self) -> Optional[bool]:
        if self._gripper_real_pos is None:
            return None
        return abs(self._gripper_real_pos - self.xarm_gripper_joint_closed) < abs(
            self._gripper_real_pos - self.xarm_gripper_joint_open)

    def _srv_arm(self, _req, res):
        if self._armed:
            res.success, res.message = True, "Ya estaba armado."
            return res
        if not self._robot_state_fresh():
            res.success, res.message = False, "Sin estado reciente del robot (¿driver conectado? ¿report_type?)."
            return res
        if not self.dry_run:
            if self._robot_err != 0:
                self.get_logger().warn(f"Controlador con error C{self._robot_err}: intentando clean_error...")
                self._call_ok(self._cli_clean_error, Call.Request(), "clean_error")
            if not self._set_mode_state(XARM_MODE_SERVO):
                res.success, res.message = False, "No se pudo pasar a modo servo (ver log)."
                return res
            if self.gripper_type == "xarm_gripper" and not self._setup_xarm_gripper():
                res.success, res.message = False, "No se pudo configurar el gripper xArm (ver log)."
                return res
        measured_closed = self._gripper_measured_closed() if self.gripper_type == "xarm_gripper" else None
        with self._lock:
            self._cmd_xyz = self._measured_xyz.copy()
            self._cmd_rpy = list(self._measured_rpy)
            self._consecutive_failures = 0
            self._servo_pending = False
            self._last_cmd_time = 0.0
            self._target = None
            if measured_closed is not None:  # sincroniza el binarizador con el gripper real
                self._gripper_closed_cmd = measured_closed
                self._gripper.closed = measured_closed
            self._armed = True
        msg = "ARMADO" + (" (dry-run: no se envían comandos)" if self.dry_run else " (modo servo, enviando comandos)")
        self.get_logger().warn(msg)
        res.success, res.message = True, msg
        return res

    def _srv_disarm(self, _req, res):
        self._disarm("solicitado por servicio")
        res.success, res.message = True, "DESARMADO"
        return res

    def _disarm(self, reason: str, restore_position_mode: bool = True):
        was_armed = self._armed
        with self._lock:
            self._armed = False
            self._target = None
        if was_armed:
            self.get_logger().warn(f"DESARMADO: {reason}")
        if restore_position_mode and not self.dry_run and was_armed and self._robot_err == 0:
            # Volver a modo posición (0) deja el brazo detenido y listo para comandos normales.
            self._set_mode_state(XARM_MODE_POSITION)

    def _srv_go_home(self, _req, res):
        if self._armed:
            res.success, res.message = False, "Desarma primero (~/disarm)."
            return res
        if self.dry_run:
            if self.simulate_robot_state:
                with self._lock:
                    self._measured_xyz = np.array(self.sim_initial_xyz, dtype=np.float64)
                    self._measured_rpy = tuple(self.sim_initial_rpy)
            res.success, res.message = True, f"dry-run: se enviaría home articular {self.home_joints}"
            return res
        if not self._set_mode_state(XARM_MODE_POSITION):
            res.success, res.message = False, "No se pudo pasar a modo posición."
            return res
        req = MoveJoint.Request()
        req.angles = self.home_joints
        req.speed, req.acc, req.wait, req.timeout = self.home_speed, self.home_acc, True, self.home_timeout
        ok = self._call_ok(self._cli_servo_angle, req, "set_servo_angle(home)", timeout=self.home_timeout + 5.0)
        res.success = ok
        res.message = "Home alcanzado." if ok else "Falló el movimiento a home (ver log)."
        return res

    def _setup_xarm_gripper(self) -> bool:
        if not self._gripper_action.wait_for_server(timeout_sec=self.service_timeout_s):
            self.get_logger().error(f"Acción del gripper no disponible: {self.xarm_gripper_action_name} "
                                    "(¿add_gripper=true en el driver?)")
            return False
        en = SetInt16.Request()
        en.data = 1
        mode = SetInt16.Request()
        mode.data = 0
        speed = SetFloat32.Request()
        speed.data = self.xarm_gripper_speed
        return (self._call_ok(self._cli_gripper_enable, en, "set_gripper_enable")
                and self._call_ok(self._cli_gripper_mode, mode, "set_gripper_mode")
                and self._call_ok(self._cli_gripper_speed, speed, "set_gripper_speed"))

    # ------------------------------------------------------------------------------------------
    # Callbacks de entrada
    # ------------------------------------------------------------------------------------------
    def _target_cb(self, msg: PoseStamped):
        frame = msg.header.frame_id.strip("/")
        if frame and frame != self.base_frame:
            self.get_logger().warn(
                f"Objetivo en frame '{frame}' distinto de '{self.base_frame}': ignorado.", throttle_duration_sec=2.0)
            return
        p, o = msg.pose.position, msg.pose.orientation
        if not is_finite_pose([p.x, p.y, p.z], [o.x, o.y, o.z, o.w]):
            self.get_logger().error("Objetivo con NaN/Inf o cuaternión nulo: ignorado.", throttle_duration_sec=2.0)
            return
        with self._lock:
            self._target = msg
            self._target_time = time.time()

    def _robot_state_cb(self, msg):
        try:
            xyz, rpy = from_robot_pose(list(msg.pose))
        except ValueError as exc:
            self.get_logger().error(f"robot_states inválido: {exc}", throttle_duration_sec=5.0)
            return
        if not (np.all(np.isfinite(xyz)) and all(math.isfinite(v) for v in rpy)):
            self.get_logger().error("robot_states con valores no finitos: ignorado.", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._measured_xyz, self._measured_rpy = xyz, rpy
            self._robot_state_time = time.time()
            self._robot_mode, self._robot_state = msg.mode, msg.state
            new_err, self._robot_warn = msg.err, msg.warn
        if new_err != 0 and self._robot_err == 0:
            self.get_logger().error(f"El controlador reporta error C{new_err} (warn={msg.warn}).")
        self._robot_err = new_err
        if new_err != 0 and self._armed:
            # No se llama a servicios: el controlador ya detuvo el brazo.
            with self._lock:
                self._armed = False
                self._target = None
            self.get_logger().error(f"DESARMADO por error del controlador C{new_err}. Revisa y llama a ~/arm.")

    def _joint_states_cb(self, msg: JointState):
        if self.gripper_type != "xarm_gripper":
            return
        for name, pos in zip(msg.name, msg.position):
            if name == self.xarm_gripper_joint_name:
                if math.isfinite(pos):  # el driver inicializa con NaN antes de leer el gripper
                    self._gripper_real_pos = float(pos)
                return

    def _gripper_cmd_cb(self, msg: Float64MultiArray):
        if not msg.data or not math.isfinite(msg.data[0]):
            return
        closed = self._gripper.update(float(msg.data[0]))
        if closed == self._gripper_closed_cmd:
            return
        if not self._armed:
            self.get_logger().info("Comando de gripper ignorado: puente desarmado.", throttle_duration_sec=2.0)
            self._gripper.closed = self._gripper_closed_cmd
            return
        now = time.time()
        if now - self._last_gripper_send < self.gripper_min_interval:
            self._gripper.closed = self._gripper_closed_cmd  # se reintentará con el próximo comando
            return
        self._gripper_closed_cmd = closed
        self._last_gripper_send = now
        action = "CERRAR" if closed else "ABRIR"
        if self.dry_run or self.gripper_type == "none":
            self.get_logger().info(f"Gripper -> {action} ({'dry-run' if self.dry_run else 'gripper_type=none'})")
            return
        if self.gripper_type == "xarm_gripper":
            self._send_gripper_goal(closed, action)
        else:
            req = VacuumGripperCtrl.Request()
            req.on = bool(closed)
            req.wait = False
            future = self._cli_vacuum.call_async(req)
            future.add_done_callback(lambda f, a=action: self._log_async_result(f, f"ventosa {a}"))
        self.get_logger().info(f"Gripper -> {action}")

    def _send_gripper_goal(self, closed: bool, action: str):
        if self._gripper_goal_handle is not None:
            self._gripper_goal_handle.cancel_goal_async()  # el nuevo objetivo reemplaza al anterior
            self._gripper_goal_handle = None
        goal = GripperCommand.Goal()
        goal.command.position = self.xarm_gripper_joint_closed if closed else self.xarm_gripper_joint_open
        goal.command.max_effort = 0.0
        future = self._gripper_action.send_goal_async(goal)

        def _accepted(f):
            try:
                handle = f.result()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"gripper {action}: envío falló ({exc})")
                return
            if not handle.accepted:
                self.get_logger().error(f"gripper {action}: objetivo rechazado por el driver")
                return
            self._gripper_goal_handle = handle

        future.add_done_callback(_accepted)

    def _log_async_result(self, future, what: str):
        try:
            res = future.result()
            if res is not None and res.ret != 0:
                self.get_logger().error(f"{what} devolvió ret={res.ret}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"{what} falló: {exc}")

    # ------------------------------------------------------------------------------------------
    # Bucle de control
    # ------------------------------------------------------------------------------------------
    def _register_failure(self, reason: str):
        self._consecutive_failures += 1
        self.get_logger().error(reason, throttle_duration_sec=1.0)
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._disarm(f"{self._consecutive_failures} fallos seguidos de set_servo_cartesian")

    def _control_step(self):
        if not self._armed:
            return
        if not self.simulate_robot_state and not self._robot_state_fresh():
            self._disarm("se perdió el estado del robot (robot_states)", restore_position_mode=False)
            return
        if self._servo_pending:
            if time.time() - self._servo_sent_time > self.servo_response_timeout_s:
                self._servo_pending = False
                self._register_failure(
                    f"set_servo_cartesian sin respuesta en {self.servo_response_timeout_s:.2f} s")
            return  # no se acumulan comandos mientras el anterior no responde
        with self._lock:
            target, t_target = self._target, self._target_time
            cmd_xyz, cmd_rpy = self._cmd_xyz.copy(), list(self._cmd_rpy)
            measured = self._measured_xyz.copy()
        if target is None or (time.time() - t_target) > self.target_timeout_s:
            self.get_logger().info("Sin objetivo reciente: manteniendo posición.", throttle_duration_sec=2.0)
            return

        p, o = target.pose.position, target.pose.orientation
        q = [o.x, o.y, o.z, o.w]
        goal_xyz, clamped = self.workspace.clamp([p.x, p.y, p.z])
        if clamped:
            self.get_logger().warn(
                f"Objetivo [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] fuera del espacio de trabajo: recortado a "
                f"[{goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f}].", throttle_duration_sec=2.0)
        dist = float(np.linalg.norm(goal_xyz - measured))
        if dist > self.max_target_distance:
            self.get_logger().warn(
                f"Objetivo a {dist:.3f} m del TCP (> {self.max_target_distance:.3f} m): rechazado por seguridad.",
                throttle_duration_sec=2.0)
            return
        goal_rpy = list(project_orientation(q, self.orientation_mode, self.fixed_rpy, self.yaw_offset,
                                            fallback_yaw=cmd_rpy[2]))
        goal_rpy[2] = clamp_yaw(goal_rpy[2], self.yaw_min, self.yaw_max)

        # Paso según el tiempo REAL desde el último comando (acotado al período nominal): si el ejecutor retrasa
        # un ciclo y luego dispara dos seguidos, el segundo paso es más corto y la velocidad nunca supera el límite.
        now = time.time()
        dt_nominal = 1.0 / self.control_rate_hz
        dt = dt_nominal if self._last_cmd_time <= 0.0 else min(max(now - self._last_cmd_time, 0.0), dt_nominal)
        self._last_cmd_time = now
        ang_step = self.max_angular_speed * dt
        new_xyz = limit_step(cmd_xyz, goal_xyz, self.max_linear_speed * dt)
        new_rpy = [limit_angle_step(cmd_rpy[0], goal_rpy[0], ang_step),
                   limit_angle_step(cmd_rpy[1], goal_rpy[1], ang_step),
                   limit_angle_step(cmd_rpy[2], goal_rpy[2], ang_step, wrap=False)]
        # Sin recorte adicional: goal_xyz ya está dentro del volumen y el paso avanza en línea recta hacia él
        # (recortar aquí produciría un salto si el brazo se armó fuera de la caja).

        self._publish_preview(new_xyz, new_rpy)
        if self.dry_run:
            with self._lock:
                self._cmd_xyz, self._cmd_rpy = new_xyz, new_rpy
                if self.simulate_robot_state:
                    self._measured_xyz, self._measured_rpy = new_xyz.copy(), tuple(new_rpy)
                    self._robot_state_time = time.time()
            return

        req = MoveCartesian.Request()
        req.pose = to_servo_pose(new_xyz, new_rpy)
        req.speed, req.acc, req.mvtime = self.servo_speed, self.servo_acc, 0.0
        self._servo_seq += 1
        seq = self._servo_seq
        self._servo_pending = True
        self._servo_sent_time = time.time()
        future = self._cli_servo_cartesian.call_async(req)
        future.add_done_callback(lambda f, x=new_xyz, r=new_rpy, s=seq: self._servo_done(f, x, r, s))

    def _servo_done(self, future, xyz, rpy, seq):
        if seq != self._servo_seq or not self._servo_pending:
            return  # respuesta tardía de una llamada ya dada por fallida (timeout): se descarta
        self._servo_pending = False
        try:
            res = future.result()
        except Exception as exc:  # noqa: BLE001
            self._register_failure(f"set_servo_cartesian falló: {exc}")
            return
        if res is not None and res.ret == 0:
            with self._lock:
                self._cmd_xyz, self._cmd_rpy = xyz, rpy
            self._consecutive_failures = 0
            return
        self._register_failure(f"set_servo_cartesian ret={getattr(res, 'ret', None)} ({getattr(res, 'message', '')})")

    # ------------------------------------------------------------------------------------------
    # Salidas: TF, estado del gripper, vista previa y estado
    # ------------------------------------------------------------------------------------------
    def _publish_preview(self, xyz, rpy):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in xyz)
        q = rpy_to_quat(*rpy)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (
            float(v) for v in q)
        self._pub_preview.publish(msg)

    def _publish_state(self):
        now = self.get_clock().now().to_msg()
        with self._lock:
            xyz, rpy = self._measured_xyz, self._measured_rpy
        if self.publish_tcp_tf and xyz is not None and self._robot_state_fresh():
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.base_frame
            t.child_frame_id = self.tcp_frame
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in xyz)
            q = rpy_to_quat(*rpy)
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = (float(v) for v in q)
            self._tf.sendTransform(t)

        if self.gripper_type == "xarm_gripper" and not self.dry_run and self._gripper_real_pos is not None:
            policy_val = linear_map(self._gripper_real_pos, self.xarm_gripper_joint_open,
                                    self.xarm_gripper_joint_closed, self.gripper_policy_open,
                                    self.gripper_policy_closed)
        else:  # ventosa / sin gripper / dry-run: se informa el último comando aceptado
            policy_val = self.gripper_policy_closed if self._gripper_closed_cmd else self.gripper_policy_open
        js = JointState()
        js.header.stamp = now
        js.name = [self.gripper_state_joint_name]
        js.position = [float(policy_val)]
        self._pub_gripper_state.publish(js)

    def _publish_status(self):
        if self.simulate_robot_state:
            age = 0.0
        else:
            age = time.time() - self._robot_state_time if self._robot_state_time else float("inf")
        with self._lock:
            xyz, rpy = self._measured_xyz, self._measured_rpy
        pose = "pose=desconocida"
        if xyz is not None:
            pose = (f"tcp_xyz_m=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] "
                    f"tcp_rpy_rad=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]")
        text = (f"armed={self._armed} dry_run={self.dry_run} simulated={self.simulate_robot_state} "
                f"robot_state_age_s={age:.2f} err={self._robot_err} warn={self._robot_warn} "
                f"mode={self._robot_mode} state={self._robot_state} gripper_closed={self._gripper_closed_cmd} "
                f"gripper_joint={self._gripper_real_pos} {pose}")
        self._pub_status.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = XArm5Bridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._armed:
            # El contexto ROS puede estar cerrado: no se llama a servicios. En modo servo, dejar de enviar
            # comandos deja el brazo detenido en su última posición.
            node._armed = False
            node.get_logger().warn("Nodo detenido estando ARMADO: el brazo queda detenido en modo servo.")
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
