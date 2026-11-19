#!/usr/bin/env python3
"""Comprobaciones del teleoperador RViz2 (xarm5_teleop_marker) contra el driver falso, sin hardware ni RViz.

Simula lo que haría RViz2: clics de menú y arrastres del marcador vía /<servidor>/feedback.
Requiere en el mismo ROS_DOMAIN_ID:
  1) python3 tools/xarm5/fake_xarm_driver.py
  2) ros2 launch visuomotor_bringup xarm5_real.launch.py start_driver:=false start_smoother:=false dry_run:=false
Luego:  python3 tools/xarm5/check_xarm5_teleop.py      (código de salida 0 = todo OK)
"""

from __future__ import annotations

import copy
import json
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import InteractiveMarkerFeedback, InteractiveMarkerUpdate
from visualization_msgs.srv import GetInteractiveMarkers

P = yaml.safe_load(open(f"{get_package_share_directory('visuomotor_bringup')}/config/xarm5/xarm5_real_params.yaml"))
TM = P["xarm5_teleop_marker"]["ros__parameters"]
MARKER = "xarm5_tcp_target"
# Orden de inserción del menú en xarm5_teleop_marker.py (MenuHandler numera desde 1)
FOLLOW_ON, FOLLOW_OFF, SYNC, ARM, DISARM, HOME2, BRAKES, UNWIND, BRAKES_OFF, BRAKES_ON = range(1, 11)


class TeleopChecker(Node):
    def __init__(self):
        super().__init__("xarm5_teleop_checker")
        ns = "/" + TM["marker_server_name"]
        self.pub_fb = self.create_publisher(InteractiveMarkerFeedback, f"{ns}/feedback", 10)
        self.targets = []
        self.status = ""
        self.robot_description = None
        self.marker_pose = None
        self.create_subscription(PoseStamped, TM["target_topic"], lambda m: self.targets.append(m), 10)
        self.create_subscription(String, f"{TM['bridge_namespace']}/status", lambda m: setattr(self, "status", m.data), 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", lambda m: setattr(self, "robot_description", m.data), qos)
        self.create_subscription(InteractiveMarkerUpdate, f"{ns}/update", self._update_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.results = []

    def _update_cb(self, msg):
        for m in msg.markers:
            if m.name == MARKER:
                self.marker_pose = m.pose
        for p in msg.poses:
            if p.name == MARKER:
                self.marker_pose = p.pose

    def spin_for(self, s, every=None):
        end = time.time() + s
        while time.time() < end:
            if every:
                every()
            rclpy.spin_once(self, timeout_sec=0.02)

    def fetch_marker(self):
        """Estado completo del servidor, como hace RViz2 al conectarse (servicio get_interactive_markers)."""
        cli = self.create_client(GetInteractiveMarkers, f"/{TM['marker_server_name']}/get_interactive_markers")
        if not cli.wait_for_service(timeout_sec=5.0):
            return None
        fut = cli.call_async(GetInteractiveMarkers.Request())
        end = time.time() + 5.0
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.destroy_client(cli)
        if not fut.done():
            return None
        for m in fut.result().markers:
            if m.name == MARKER:
                self.marker_pose = m.pose
                return m
        return None

    def stats(self):
        cli = self.create_client(Trigger, "/fake_xarm/get_stats")
        cli.wait_for_service(timeout_sec=5.0)
        fut = cli.call_async(Trigger.Request())
        while not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        self.destroy_client(cli)
        return json.loads(fut.result().message)

    def feedback(self, event, menu_id=0, pose=None):
        fb = InteractiveMarkerFeedback()
        fb.header.frame_id = TM["base_frame"]
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.client_id = "check_xarm5_teleop"
        fb.marker_name = MARKER
        fb.event_type = event
        fb.menu_entry_id = menu_id
        if pose is not None:
            fb.pose = pose
        self.pub_fb.publish(fb)

    def menu(self, entry, wait=1.5):
        self.feedback(InteractiveMarkerFeedback.MENU_SELECT, menu_id=entry)
        self.spin_for(wait)

    def check(self, name, ok, detail=""):
        self.results.append(bool(ok))
        print(f"[{'OK ' if ok else 'FALLA'}] {name}" + (f" -> {detail}" if detail else ""), flush=True)

    def run(self):
        self.spin_for(4.0)
        rd = self.robot_description or ""
        self.check("1. robot_description publicado (URDF xArm5 con joint1..joint5, sin joint6)",
                   all(f'name="joint{i}"' in rd for i in range(1, 6)) and 'name="joint6"' not in rd,
                   f"{len(rd)} caracteres")
        chain = all(self.tf_buffer.can_transform(TM["base_frame"], f, rclpy.time.Time())
                    for f in ("link1", "link5", "link_eef", TM["tcp_frame"]))
        self.check("2. TF link_base -> link1..link5/link_eef (robot_state_publisher) y -> xarm_tcp (puente)", chain)
        im = self.fetch_marker()
        tcp = self.tf_buffer.lookup_transform(TM["base_frame"], TM["tcp_frame"], rclpy.time.Time()).transform.translation
        ok3 = im is not None and abs(im.pose.position.x - tcp.x) < 0.005 and abs(im.pose.position.z - tcp.z) < 0.005
        self.check("3. Marcador interactivo creado sobre el TCP (vía get_interactive_markers, como RViz2)", ok3,
                   f"marcador z={im.pose.position.z:.3f} / TCP z={tcp.z:.3f}" if im else "sin marcador")
        if im is None:
            print("RESUMEN: abortado (sin marcador)")
            return False

        moved = copy.deepcopy(self.marker_pose)
        moved.position.x += 0.03
        n0 = len(self.targets)
        self.spin_for(1.5, every=lambda: self.feedback(InteractiveMarkerFeedback.POSE_UPDATE, pose=moved))
        self.check("4. Arrastrar sin 'Seguir' ni armado NO publica objetivos", len(self.targets) == n0)

        self.menu(FOLLOW_ON)
        self.check("5. 'Seguir' con el puente DESARMADO se rechaza", len(self.targets) == n0 and "armed=False" in self.status)

        self.menu(ARM, wait=3.0)
        p_arm = self.stats()["pose"]
        self.check("6. Menú 'Armar' arma el puente (y sincroniza el marcador con el TCP)", "armed=True" in self.status)
        self.spin_for(1.0)
        self.check("7. Armado sin 'Seguir' no publica objetivos ni mueve", len(self.targets) == n0
                   and self.stats()["pose"][:3] == p_arm[:3])

        self.menu(FOLLOW_ON)
        target = copy.deepcopy(self.marker_pose)
        target.position.x += 0.03
        self.spin_for(2.5, every=lambda: self.feedback(InteractiveMarkerFeedback.POSE_UPDATE, pose=target))
        p_follow = self.stats()["pose"]
        self.check("8. 'Seguir' + arrastre +3 cm en X -> el brazo (falso) sigue al marcador",
                   len(self.targets) > n0 and abs(p_follow[0] - (p_arm[0] + 30.0)) < 1.0,
                   f"x: {p_arm[0]:.1f} -> {p_follow[0]:.1f} mm")

        self.menu(BRAKES)
        self.menu(DISARM, wait=2.5)
        n1 = len(self.targets)
        self.spin_for(1.5, every=lambda: self.feedback(InteractiveMarkerFeedback.POSE_UPDATE, pose=target))
        self.check("9. Menú 'Desarmar' -> desarma y el marcador deja de publicar",
                   "armed=False" in self.status and len(self.targets) == n1)

        self.menu(HOME2, wait=3.0)
        s = self.stats()
        self.check("10. Menú 'Ir a home2' (desarmado) -> set_servo_angle con home2_joints_rad",
                   s["set_servo_angle"][-1:] == [[round(v, 3) for v in
                                                  P["xarm5_bridge"]["ros__parameters"]["home2_joints_rad"]]])

        from xarm_msgs.srv import SetFloat32List
        cli = self.create_client(SetFloat32List, "/fake_xarm/set_joints")
        cli.wait_for_service(timeout_sec=5.0)
        fut = cli.call_async(SetFloat32List.Request(datas=[6.2833, -0.5504, -0.127, 0.6855, 0.4098]))
        while not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        self.spin_for(0.5)
        n_angle = len(self.stats()["set_servo_angle"])
        self.menu(UNWIND, wait=2.0)
        first_only = len(self.stats()["set_servo_angle"]) == n_angle
        self.menu(UNWIND, wait=4.0)
        s = self.stats()
        sent = s["set_servo_angle"][-1] if len(s["set_servo_angle"]) > n_angle else None
        self.check("11. Menú 'Desenrollar J1': 1er clic solo avisa; 2º clic gira solo J1 6.2833 -> ~0",
                   first_only and sent is not None and abs(sent[0] - (6.2833 - 2 * 3.141592653589793)) < 1e-3
                   and sent[1:] == [-0.55, -0.127, 0.686, 0.41], f"enviado={sent}")

        self.menu(BRAKES_OFF, wait=3.0)
        s = self.stats()
        self.check("12. Menú 'Quitar frenos': motores habilitados sin armar", s["mt_able"] == 31 and s["mt_brake"] == 31
                   and "armed=False" in self.status)
        self.spin_for(1.0)
        self.menu(BRAKES_ON, wait=1.5)
        first_only = self.stats()["mt_able"] == 31
        self.menu(BRAKES_ON, wait=3.0)
        s = self.stats()
        self.check("13. Menú 'Poner frenos': 1er clic solo avisa; 2º clic deshabilita motores",
                   first_only and s["mt_able"] == 0 and s["mt_brake"] == 0, f"llamadas={s['motion_enable_calls'][-2:]}")

        print(f"\nRESUMEN: {sum(self.results)}/{len(self.results)} comprobaciones OK", flush=True)
        return all(self.results)


def main():
    rclpy.init()
    node = TeleopChecker()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
