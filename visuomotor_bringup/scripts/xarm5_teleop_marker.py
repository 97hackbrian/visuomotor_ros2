#!/usr/bin/env python3
"""Teleoperación del xArm5 real desde RViz2 (local o remoto) con un marcador interactivo.

Se ejecuta en la Jetson junto al driver y a xarm5_bridge. RViz2 (en otro equipo, mismo ROS_DOMAIN_ID) muestra el
marcador ``<marker_server_name>`` y lo opera; este nodo publica el objetivo en ``target_topic`` (entrada de
xarm5_bridge), que aplica todas las protecciones (caja, velocidad, distancia, frenos, watchdog).

Seguridad:
* El marcador nace sobre el TCP medido (TF ``base_frame -> tcp_frame``): no hay salto inicial.
* Solo publica objetivos con "Seguir marcador" ACTIVADO y el puente ARMADO. Si el puente se desarma por cualquier
  causa, el seguimiento se desactiva solo.
* Controles limitados a lo que el xArm5 (5 GDL) puede ejecutar: X, Y, Z, giro de la herramienta (anillo azul) y,
  si el puente está en ``orientation_mode: arm_plane``, inclinación en el plano vertical del brazo (anillo verde, eje
  perpendicular a ese plano según el acimut del marcador). Al soltar el ratón la orientación del marcador se ajusta a
  la alcanzable (misma proyección y límites que aplica el puente, leídos de su ``status``).

Menú en inglés (clic derecho): Follow / Stop / Sync / Arm / Disarm / Check brakes / Release brakes / Engage brakes / HOME1 / HOME2 / HOME3 (doble clic).
Parámetros en ``visuomotor_bringup/config/xarm5/xarm5_real_params.yaml`` (sección
``xarm5_teleop_marker``).
"""

from __future__ import annotations

import copy
import math
import re
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers import InteractiveMarkerServer, MenuHandler
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker

from visuomotor_bringup.xarm5_interface import arm_azimuth, arm_plane_quat, clamp_yaw, project_arm_plane

MARKER_NAME = "xarm5_tcp_target"
_H = math.sqrt(0.5)  # cuaterniones unitarios de los ejes de control (convención de interactive_markers)
_NUM = r"(-?[0-9.]+)"
_RE_MODE = re.compile(r"orientation_mode=(\w+)")
_RE_PITCH = re.compile(rf"pitch_limits_rad=\[{_NUM},{_NUM}\]")
_RE_YAW = re.compile(rf"yaw_limits_rad=\[{_NUM},{_NUM}\]")


def tilt_axis_quat(azimuth: float):
    """Orientación (w, x, y, z) de un control cuyo eje X es (-sin phi, cos phi, 0): perpendicular al plano del brazo."""
    a = azimuth + math.pi / 2.0
    return math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)


class XArm5TeleopMarker(Node):
    def __init__(self):
        super().__init__("xarm5_teleop_marker")
        d = self.declare_parameter
        d("base_frame", "link_base")
        d("tcp_frame", "xarm_tcp")
        d("target_topic", "/target_frame")
        d("bridge_namespace", "/xarm5_bridge")
        d("marker_server_name", "xarm5_teleop")
        d("publish_rate_hz", 20.0)
        d("marker_scale", 0.25)
        d("tf_timeout_s", 0.2)
        d("init_retry_s", 1.0)
        d("confirm_window_s", 15.0)
        g = lambda n: self.get_parameter(n).value
        self.base_frame = g("base_frame")
        self.tcp_frame = g("tcp_frame")
        self.marker_scale = float(g("marker_scale"))
        self.tf_timeout = float(g("tf_timeout_s"))
        bridge = g("bridge_namespace").rstrip("/")

        self.following = False
        self.armed = False
        self.dry_run = None  # desconocido hasta recibir el estado del puente
        self.last_msg = "Ready. Right-click the sphere for the menu."
        self.marker_pose: Pose | None = None
        self.confirm_window = float(g("confirm_window_s"))
        self._pending_confirm = None
        self._confirm_until = 0.0
        # Modo de orientación y límites del puente (de su status): el anillo verde solo existe con 'arm_plane'.
        self.orientation_mode = None
        self.pitch_limits = (0.0, 0.0)
        self.yaw_limits = (-math.pi, math.pi)
        self.tilt_pitch = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_target = self.create_publisher(PoseStamped, g("target_topic"), 10)
        self.create_subscription(String, f"{bridge}/status", self._status_cb, 10)
        self.cli = {name: self.create_client(Trigger, f"{bridge}/{name}")
                    for name in ("arm", "disarm", "check_brakes", "brakes_off", "brakes_on", "go_home1", "go_home2", "go_home3")}

        self.server = InteractiveMarkerServer(self, g("marker_server_name"))
        self.menu = MenuHandler()
        self.menu.insert("Follow marker (enable)", callback=self._menu_follow_on)
        self.menu.insert("Stop following", callback=self._menu_follow_off)
        self.menu.insert("Sync marker to TCP", callback=self._menu_sync)
        self.menu.insert("Arm (syncs marker first)", callback=lambda f: self._call_bridge("arm", sync=True))
        self.menu.insert("Disarm", callback=lambda f: self._call_bridge("disarm"))
        self.menu.insert("Check brakes", callback=lambda f: self._call_bridge("check_brakes"))
        self.menu.insert("Release brakes (motors hold the arm)", callback=lambda f: self._call_bridge("brakes_off"))
        self.menu.insert("Engage brakes (disables motors, double click)",
                         callback=lambda f: self._confirm_then_call(
                             "brakes_on", "motors will be DISABLED (brakes engaged); arm must be DISARMED and still"))
        self.menu.insert("Go to HOME1: zero / transport pose (double click)",
                         callback=lambda f: self._confirm_then_call(
                             "go_home1", "the arm will move to HOME1 (all joints 0 deg, UFACTORY transport pose)"))
        self.menu.insert("Go to HOME2: UFACTORY initial pose (double click)",
                         callback=lambda f: self._confirm_then_call(
                             "go_home2", "the arm will move to HOME2 (J4 -90 deg, UFACTORY default initial position)"))
        self.menu.insert("Go to HOME3: work pose (double click)",
                         callback=lambda f: self._confirm_then_call(
                             "go_home3", "the arm will move to HOME3 (work pose high above the base, end effector tilted ~25 deg)"))

        self.create_timer(1.0 / float(g("publish_rate_hz")), self._publish_target)
        self._init_timer = self.create_timer(float(g("init_retry_s")), self._try_init)
        self.get_logger().info(
            f"RViz2 teleop waiting for TF {self.base_frame}->{self.tcp_frame}. Marker server: "
            f"'{g('marker_server_name')}'. Publishes nothing until 'Follow marker' with the bridge ARMED.")

    # --- TF / marcador ---------------------------------------------------------------------------
    def _tcp_pose(self) -> Pose | None:
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.tcp_frame, Time(),
                                                timeout=Duration(seconds=self.tf_timeout))
        except TransformException:
            return None
        p = Pose()
        p.position.x, p.position.y, p.position.z = (t.transform.translation.x, t.transform.translation.y,
                                                    t.transform.translation.z)
        p.orientation = t.transform.rotation
        return p

    def _try_init(self):
        pose = self._tcp_pose()
        if pose is None:
            self.get_logger().info(f"Waiting for TF {self.base_frame}->{self.tcp_frame}...", throttle_duration_sec=5.0)
            return
        self._init_timer.cancel()
        self.marker_pose = pose
        if self.orientation_mode == "arm_plane":
            self._snap_to_reachable()
        self._rebuild_marker()
        self.get_logger().info(
            f"Marker created on the TCP ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}).")

    def _rebuild_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = self.base_frame
        im.name = MARKER_NAME
        im.scale = self.marker_scale
        im.pose = copy.deepcopy(self.marker_pose)
        state = "FOLLOWING" if self.following else "IDLE"
        mode = "" if self.dry_run is False else (
            "\n*** DRY-RUN: the arm will NOT move (relaunch with dry_run:=false) ***" if self.dry_run
            else "\n(waiting for bridge status)")
        tilt = ""
        if self.orientation_mode == "arm_plane":
            tilt = (f"\ntilt {math.degrees(self.tilt_pitch):+.1f} deg (limits {math.degrees(self.pitch_limits[0]):+.0f}"
                    f"..{math.degrees(self.pitch_limits[1]):+.0f})")
        im.description = f"xArm5 [{state} | {'ARMED' if self.armed else 'DISARMED'}]{mode}{tilt}\n{self.last_msg}"

        body = InteractiveMarkerControl()
        body.always_visible = True
        body.interaction_mode = InteractiveMarkerControl.MENU  # clic derecho -> menú
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = sphere.scale.y = sphere.scale.z = self.marker_scale * 0.45
        if self.dry_run:          # amarillo: modo simulación, nada se mueve
            sphere.color.r, sphere.color.g, sphere.color.b = (1.0, 0.85, 0.0)
        elif self.following:      # verde: siguiendo al marcador
            sphere.color.r, sphere.color.g, sphere.color.b = (0.1, 0.9, 0.2)
        else:                     # gris: en espera
            sphere.color.r, sphere.color.g, sphere.color.b = (0.7, 0.7, 0.7)
        sphere.color.a = 0.9
        body.markers.append(sphere)
        im.controls.append(body)

        # Traslación X/Y/Z en el marco fijo (base). Orientación solo en los GDL del xArm5 (5 GDL, sin roll lateral):
        # * top_down_yaw: anillo azul = giro alrededor de la vertical (garra siempre vertical).
        # * arm_plane: anillo azul = giro sobre el eje de la herramienta (J5, sigue a la inclinación) y anillo verde =
        #   inclinación en el plano vertical del brazo (eje perpendicular a ese plano, recalculado al soltar).
        arm_plane = self.orientation_mode == "arm_plane"
        controls = [
            ("mover_x", (_H, _H, 0.0, 0.0), InteractiveMarkerControl.MOVE_AXIS, InteractiveMarkerControl.FIXED),
            ("mover_y", (_H, 0.0, 0.0, _H), InteractiveMarkerControl.MOVE_AXIS, InteractiveMarkerControl.FIXED),
            ("mover_z", (_H, 0.0, _H, 0.0), InteractiveMarkerControl.MOVE_AXIS, InteractiveMarkerControl.FIXED),
            ("girar_yaw", (_H, 0.0, _H, 0.0), InteractiveMarkerControl.ROTATE_AXIS,
             InteractiveMarkerControl.INHERIT if arm_plane else InteractiveMarkerControl.FIXED)]
        if arm_plane:
            p = self.marker_pose.position
            controls.append(("girar_pitch", tilt_axis_quat(arm_azimuth([p.x, p.y, p.z])),
                             InteractiveMarkerControl.ROTATE_AXIS, InteractiveMarkerControl.FIXED))
        for name, (w, x, y, z), mode, frame_mode in controls:
            c = InteractiveMarkerControl()
            c.name = name
            c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z = w, x, y, z
            c.orientation_mode = frame_mode
            c.interaction_mode = mode
            im.controls.append(c)

        self.server.insert(im, feedback_callback=self._feedback)
        self.menu.apply(self.server, MARKER_NAME)
        self.server.applyChanges()

    def _feedback(self, fb: InteractiveMarkerFeedback):
        if fb.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            if fb.header.frame_id and fb.header.frame_id.strip("/") != self.base_frame:
                self.get_logger().warn(f"Feedback in frame '{fb.header.frame_id}': ignored.",
                                       throttle_duration_sec=2.0)
                return
            self.marker_pose = copy.deepcopy(fb.pose)
        elif fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self.orientation_mode == "arm_plane":
            if fb.header.frame_id and fb.header.frame_id.strip("/") != self.base_frame:
                return
            self.marker_pose = copy.deepcopy(fb.pose)
            self._snap_to_reachable()
            self._rebuild_marker()  # eje del anillo verde según el nuevo acimut

    def _snap_to_reachable(self):
        """Ajusta la orientación del marcador a la alcanzable (misma proyección y límites que el puente)."""
        p, o = self.marker_pose.position, self.marker_pose.orientation
        az = arm_azimuth([p.x, p.y, p.z])
        pitch, yaw, _ = project_arm_plane([o.x, o.y, o.z, o.w], az, self.pitch_limits[0], self.pitch_limits[1])
        yaw = clamp_yaw(yaw, self.yaw_limits[0], self.yaw_limits[1])
        q = arm_plane_quat(az, pitch, yaw)
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        self.tilt_pitch = pitch

    # --- menú ------------------------------------------------------------------------------------
    def _set_msg(self, text: str, error: bool = False):
        self.last_msg = text
        # rclpy no permite cambiar la severidad desde una misma línea de log: una llamada por severidad
        if error:
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)
        if self.marker_pose is not None:
            self._rebuild_marker()

    def _menu_follow_on(self, _fb):
        if not self.armed:
            self._set_msg("Not enabled: the bridge is DISARMED (use 'Arm' first).", error=True)
            return
        self.following = True
        if self.dry_run:
            self._set_msg("Following ON in DRY-RUN: the arm will NOT move (relaunch with dry_run:=false).",
                          error=True)
        else:
            self._set_msg("Following ON: the arm follows the marker (bridge limits apply).")

    def _menu_follow_off(self, _fb):
        self.following = False
        self._set_msg("Following stopped: the arm holds its position.")

    def _menu_sync(self, _fb):
        self._sync_to_tcp()

    def _sync_to_tcp(self) -> bool:
        pose = self._tcp_pose()
        if pose is None:
            self._set_msg("Cannot sync: no TCP transform.", error=True)
            return False
        self.marker_pose = pose
        if self.orientation_mode == "arm_plane":
            self._snap_to_reachable()
        self.server.setPose(MARKER_NAME, self.marker_pose)
        self.server.applyChanges()
        self._set_msg("Marker synced to the TCP.")
        return True

    def _confirm_then_call(self, service: str, warning: str):
        """Double confirmation: the first click only warns; a second click within the window calls the service."""
        now = time.time()
        if self._pending_confirm != service or now > self._confirm_until:
            self._pending_confirm, self._confirm_until = service, now + self.confirm_window
            self._set_msg(f"WARNING: {warning}. Select the same option again within "
                          f"{self.confirm_window:.0f} s to confirm.", error=True)
            return
        self._pending_confirm, self._confirm_until = None, 0.0
        self.following = False
        self._call_bridge(service)

    def _call_bridge(self, name: str, sync: bool = False):
        if sync:
            self.following = False
            if not self._sync_to_tcp():
                return
        cli = self.cli[name]
        if not cli.service_is_ready():
            self._set_msg(f"Service {cli.srv_name} not available.", error=True)
            return
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f, n=name: self._bridge_done(n, f))
        self._set_msg(f"Calling {name}...")

    def _bridge_done(self, name, fut):
        try:
            res = fut.result()
            self._set_msg(f"{name}: {res.message}", error=not res.success)
        except Exception as exc:  # noqa: BLE001
            self._set_msg(f"{name} failed: {exc}", error=True)

    # --- estado del puente y publicación -----------------------------------------------------------
    def _status_cb(self, msg: String):
        armed = "armed=True" in msg.data
        dry_run = "dry_run=True" in msg.data
        m_mode, m_pitch, m_yaw = _RE_MODE.search(msg.data), _RE_PITCH.search(msg.data), _RE_YAW.search(msg.data)
        if m_mode and m_pitch and m_yaw:
            mode = m_mode.group(1)
            pitch_limits = (float(m_pitch.group(1)), float(m_pitch.group(2)))
            yaw_limits = (float(m_yaw.group(1)), float(m_yaw.group(2)))
            if (mode, pitch_limits, yaw_limits) != (self.orientation_mode, self.pitch_limits, self.yaw_limits):
                self.orientation_mode, self.pitch_limits, self.yaw_limits = mode, pitch_limits, yaw_limits
                self.get_logger().info(
                    f"Bridge orientation_mode={mode}: "
                    + (f"green ring = tilt in the arm plane, limits [{pitch_limits[0]:.2f}, {pitch_limits[1]:.2f}] rad."
                       if mode == "arm_plane" else "no tilt ring (end effector kept vertical)."))
                if self.marker_pose is not None:
                    if mode == "arm_plane":
                        self._snap_to_reachable()
                    self._rebuild_marker()
        if dry_run != self.dry_run:
            self.dry_run = dry_run
            if dry_run:
                self.get_logger().warn("Bridge in DRY-RUN: commands are computed but NOT sent to the arm.")
            if self.marker_pose is not None:
                self._rebuild_marker()
        if armed != self.armed:
            self.armed = armed
            if not armed and self.following:
                self.following = False
                self._set_msg("Bridge DISARMED: following disabled automatically.", error=True)
            elif self.marker_pose is not None:
                self._rebuild_marker()

    def _publish_target(self):
        if not (self.following and self.armed) or self.marker_pose is None:
            return
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.pose = copy.deepcopy(self.marker_pose)
        self.pub_target.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = XArm5TeleopMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
