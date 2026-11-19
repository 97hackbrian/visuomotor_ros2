#!/usr/bin/env python3
"""Teleoperación del xArm5 real desde RViz2 (local o remoto) con un marcador interactivo.

Se ejecuta en la Jetson junto al driver y a xarm5_bridge. RViz2 (en otro equipo, mismo ROS_DOMAIN_ID) muestra el
marcador ``<marker_server_name>`` y lo opera; este nodo publica el objetivo en ``target_topic`` (entrada de
xarm5_bridge), que aplica todas las protecciones (caja, velocidad, distancia, frenos, watchdog).

Seguridad:
* El marcador nace sobre el TCP medido (TF ``base_frame -> tcp_frame``): no hay salto inicial.
* Solo publica objetivos con "Seguir marcador" ACTIVADO y el puente ARMADO. Si el puente se desarma por cualquier
  causa, el seguimiento se desactiva solo.
* Controles limitados a lo que el xArm5 (5 GDL, garra vertical) puede ejecutar: X, Y, Z y giro (yaw).

Menú (clic derecho sobre el marcador): Seguir / Detener / Sincronizar con TCP / Armar / Desarmar / Ir a home2 /
Comprobar frenos. Parámetros en ``visuomotor_bringup/config/xarm5/xarm5_real_params.yaml`` (sección
``xarm5_teleop_marker``).
"""

from __future__ import annotations

import copy
import math

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

MARKER_NAME = "xarm5_tcp_target"
_H = math.sqrt(0.5)  # cuaterniones unitarios de los ejes de control (convención de interactive_markers)


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
        g = lambda n: self.get_parameter(n).value
        self.base_frame = g("base_frame")
        self.tcp_frame = g("tcp_frame")
        self.marker_scale = float(g("marker_scale"))
        self.tf_timeout = float(g("tf_timeout_s"))
        bridge = g("bridge_namespace").rstrip("/")

        self.following = False
        self.armed = False
        self.dry_run = None  # desconocido hasta recibir el estado del puente
        self.last_msg = "Listo. Clic derecho para el menú."
        self.marker_pose: Pose | None = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_target = self.create_publisher(PoseStamped, g("target_topic"), 10)
        self.create_subscription(String, f"{bridge}/status", self._status_cb, 10)
        self.cli = {name: self.create_client(Trigger, f"{bridge}/{name}")
                    for name in ("arm", "disarm", "go_home2", "check_brakes")}

        self.server = InteractiveMarkerServer(self, g("marker_server_name"))
        self.menu = MenuHandler()
        self.menu.insert("Seguir marcador (activar)", callback=self._menu_follow_on)
        self.menu.insert("Detener seguimiento", callback=self._menu_follow_off)
        self.menu.insert("Sincronizar marcador con TCP", callback=self._menu_sync)
        self.menu.insert("Armar brazo (sincroniza antes)", callback=lambda f: self._call_bridge("arm", sync=True))
        self.menu.insert("Desarmar brazo", callback=lambda f: self._call_bridge("disarm"))
        self.menu.insert("Ir a home2 (solo desarmado)", callback=lambda f: self._call_bridge("go_home2"))
        self.menu.insert("Comprobar frenos", callback=lambda f: self._call_bridge("check_brakes"))

        self.create_timer(1.0 / float(g("publish_rate_hz")), self._publish_target)
        self._init_timer = self.create_timer(float(g("init_retry_s")), self._try_init)
        self.get_logger().info(
            f"Teleop RViz2 en espera del TF {self.base_frame}->{self.tcp_frame}. Marcador: "
            f"'{g('marker_server_name')}'. No publica nada hasta 'Seguir marcador' con el puente ARMADO.")

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
            self.get_logger().info(f"Esperando TF {self.base_frame}->{self.tcp_frame}...", throttle_duration_sec=5.0)
            return
        self._init_timer.cancel()
        self.marker_pose = pose
        self._rebuild_marker()
        self.get_logger().info(
            f"Marcador creado sobre el TCP ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}).")

    def _rebuild_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = self.base_frame
        im.name = MARKER_NAME
        im.scale = self.marker_scale
        im.pose = copy.deepcopy(self.marker_pose)
        state = "SIGUIENDO" if self.following else "EN ESPERA"
        mode = "" if self.dry_run is False else (
            "\n*** DRY-RUN: el brazo NO se moverá (relanzar con dry_run:=false) ***" if self.dry_run
            else "\n(esperando estado del puente)")
        im.description = f"xArm5 [{state} | {'ARMADO' if self.armed else 'DESARMADO'}]{mode}\n{self.last_msg}"

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

        # Ejes en el marco fijo (base): traslación X/Y/Z y giros (yaw/Z paralelo a la base; pitch/Y y roll/X perpendiculares)
        for name, (w, x, y, z), mode in (
                ("mover_x", (_H, _H, 0.0, 0.0), InteractiveMarkerControl.MOVE_AXIS),
                ("mover_y", (_H, 0.0, 0.0, _H), InteractiveMarkerControl.MOVE_AXIS),
                ("mover_z", (_H, 0.0, _H, 0.0), InteractiveMarkerControl.MOVE_AXIS),
                ("girar_yaw", (_H, 0.0, _H, 0.0), InteractiveMarkerControl.ROTATE_AXIS),
                ("girar_pitch", (_H, 0.0, 0.0, _H), InteractiveMarkerControl.ROTATE_AXIS),
                ("girar_roll", (1.0, 0.0, 0.0, 0.0), InteractiveMarkerControl.ROTATE_AXIS)):
            c = InteractiveMarkerControl()
            c.name = name
            c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z = w, x, y, z
            c.orientation_mode = InteractiveMarkerControl.FIXED
            c.interaction_mode = mode
            im.controls.append(c)

        self.server.insert(im, feedback_callback=self._feedback)
        self.menu.apply(self.server, MARKER_NAME)
        self.server.applyChanges()

    def _feedback(self, fb: InteractiveMarkerFeedback):
        if fb.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            if fb.header.frame_id and fb.header.frame_id.strip("/") != self.base_frame:
                self.get_logger().warn(f"Feedback en frame '{fb.header.frame_id}': ignorado.",
                                       throttle_duration_sec=2.0)
                return
            self.marker_pose = copy.deepcopy(fb.pose)

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
            self._set_msg("No se activa: el puente está DESARMADO (usa 'Armar brazo').", error=True)
            return
        self.following = True
        if self.dry_run:
            self._set_msg("Seguimiento ACTIVO en DRY-RUN: el brazo NO se moverá (relanzar con dry_run:=false).",
                          error=True)
        else:
            self._set_msg("Seguimiento ACTIVO: el brazo sigue al marcador (con límites del puente).")

    def _menu_follow_off(self, _fb):
        self.following = False
        self._set_msg("Seguimiento detenido: el brazo mantiene su posición.")

    def _menu_sync(self, _fb):
        self._sync_to_tcp()

    def _sync_to_tcp(self) -> bool:
        pose = self._tcp_pose()
        if pose is None:
            self._set_msg("No se pudo sincronizar: sin TF del TCP.", error=True)
            return False
        self.marker_pose = pose
        self.server.setPose(MARKER_NAME, pose)
        self.server.applyChanges()
        self._set_msg("Marcador sincronizado con el TCP.")
        return True

    def _call_bridge(self, name: str, sync: bool = False):
        if sync:
            self.following = False
            if not self._sync_to_tcp():
                return
        cli = self.cli[name]
        if not cli.service_is_ready():
            self._set_msg(f"Servicio {cli.srv_name} no disponible.", error=True)
            return
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f, n=name: self._bridge_done(n, f))
        self._set_msg(f"Llamando a {name}...")

    def _bridge_done(self, name, fut):
        try:
            res = fut.result()
            self._set_msg(f"{name}: {res.message}", error=not res.success)
        except Exception as exc:  # noqa: BLE001
            self._set_msg(f"{name} falló: {exc}", error=True)

    # --- estado del puente y publicación -----------------------------------------------------------
    def _status_cb(self, msg: String):
        armed = "armed=True" in msg.data
        dry_run = "dry_run=True" in msg.data
        if dry_run != self.dry_run:
            self.dry_run = dry_run
            if dry_run:
                self.get_logger().warn("Puente en DRY-RUN: los comandos se calculan pero NO se envían al brazo.")
            if self.marker_pose is not None:
                self._rebuild_marker()
        if armed != self.armed:
            self.armed = armed
            if not armed and self.following:
                self.following = False
                self._set_msg("Puente DESARMADO: seguimiento desactivado automáticamente.", error=True)
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
