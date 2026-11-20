#!/usr/bin/env python3
"""Comprobaciones de integración de xarm5_bridge en modo REAL contra el driver falso (sin hardware).

Requiere, en el mismo ROS_DOMAIN_ID:
  1) python3 tools/xarm5/fake_xarm_driver.py
  2) ros2 launch visuomotor_bringup xarm5_real.launch.py start_driver:=false start_smoother:=false dry_run:=false

Luego:  python3 tools/xarm5/check_xarm5_bridge.py      (31 comprobaciones; código de salida 0 = todas OK)
Los objetivos se publican directamente en /target_frame (entrada del puente).
"""

from __future__ import annotations

import json
import math
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from xarm_msgs.srv import SetFloat32List, SetInt16

from visuomotor_bringup.xarm5_interface import rpy_to_quat

P = yaml.safe_load(open(f"{get_package_share_directory('visuomotor_bringup')}/config/xarm5/xarm5_real_params.yaml"))
BR = P["xarm5_bridge"]["ros__parameters"]


class Checker(Node):
    def __init__(self):
        super().__init__("xarm5_bridge_checker")
        self.pub_target = self.create_publisher(PoseStamped, BR["target_topic"], 10)
        self.pub_grip = self.create_publisher(Float64MultiArray, BR["gripper_command_topic"], 10)
        self.status = ""
        self.grip_state = None
        self.create_subscription(String, "/xarm5_bridge/status", lambda m: setattr(self, "status", m.data), 10)
        self.create_subscription(JointState, BR["gripper_state_topic"],
                                 lambda m: setattr(self, "grip_state", m.position[0]), 10)
        self.results = []

    # --- utilidades ------------------------------------------------------------------------------
    def spin_for(self, seconds, every=None):
        end = time.time() + seconds
        while time.time() < end:
            if every:
                every()
            rclpy.spin_once(self, timeout_sec=0.02)

    def call(self, srv_type, name, req=None, timeout=15.0):
        cli = self.create_client(srv_type, name)
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"servicio {name} no disponible")
        fut = cli.call_async(req if req is not None else srv_type.Request())
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.destroy_client(cli)
        if not fut.done():
            raise RuntimeError(f"timeout en {name}")
        return fut.result()

    def stats(self):
        return json.loads(self.call(Trigger, "/fake_xarm/get_stats").message)

    def fault(self, name, on):
        self.call(SetBool, f"/fake_xarm/{name}", SetBool.Request(data=on))

    def arm(self):
        return self.call(Trigger, "/xarm5_bridge/arm")

    def disarm(self):
        return self.call(Trigger, "/xarm5_bridge/disarm")

    def armed(self):
        self.spin_for(1.2)  # status a 1 Hz
        return "armed=True" in self.status

    def target(self, x, y, z, yaw=0.0, frame=None, tilt=0.0, nan=False):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = BR["base_frame"] if frame is None else frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float("nan") if nan else x), y, z
        q = rpy_to_quat(math.pi - tilt, 0.0, yaw)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = (float(v) for v in q)
        self.pub_target.publish(m)

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok)))
        print(f"[{'OK ' if ok else 'FALLA'}] {name}" + (f" -> {detail}" if detail else ""), flush=True)

    # --- escenarios ------------------------------------------------------------------------------
    def run(self):
        rate = BR["control_rate_hz"]
        v_max_mm_step = BR["max_linear_speed_m_s"] / rate * 1000.0
        self.spin_for(2.0)

        self.spin_for(1.5, every=lambda: self.target(0.30, 0.0, 0.38))
        s1 = self.stats()
        self.check("1. Desarmado: no envía movimiento", s1["servo_ok"] + s1["servo_rejected"] == 0,
                   f"llamadas servo={s1['servo_ok'] + s1['servo_rejected']}")

        r = self.arm()
        s = self.stats()
        gr = BR["gripper_type"] == "xarm_gripper"
        self.check("2. Armado: motion_enable + modo servo (1) + estado 0" + (" + setup gripper" if gr else ""),
                   r.success and s["mode"] == 1 and s["state"] == 0 and s["motion_enable"] >= 1
                   and (not gr or (s["gripper_enable"] >= 1 and s["gripper_mode"] >= 1
                                   and s["gripper_speed"] == BR["xarm_gripper_speed"])),
                   f"{r.message} | mode={s['mode']} state={s['state']}")

        self.spin_for(4.0, every=lambda: self.target(0.35, 0.05, 0.35, yaw=0.5, tilt=0.3))
        s = self.stats()
        pose = s["pose"]
        self.check("3. Sigue objetivo válido (unidades mm/rad)",
                   abs(pose[0] - 350) < 1.0 and abs(pose[1] - 50) < 1.0 and abs(pose[2] - 350) < 1.0,
                   f"pose_mm={[round(v, 1) for v in pose[:3]]}")
        self.check("4. Límite de velocidad por comando", s["max_step_mm"] <= v_max_mm_step + 1e-3,
                   f"paso_max={s['max_step_mm']:.3f} mm (límite {v_max_mm_step:.3f})")
        self.check("5. Orientación 5 GDL: garra vertical + yaw de la política",
                   abs(abs(pose[3]) - math.pi) < 1e-3 and abs(pose[4]) < 1e-3 and abs(pose[5] - 0.5) < 1e-3,
                   f"rpy={[round(v, 3) for v in pose[3:]]}")

        before = self.stats()["pose"]
        self.spin_for(1.0, every=lambda: self.target(0.30, 0.0, 0.30, frame="world"))
        self.spin_for(1.0, every=lambda: self.target(0.30, 0.0, 0.30, nan=True))
        after = self.stats()["pose"]
        self.check("6. Ignora objetivos en otro frame y con NaN", math.dist(before[:3], after[:3]) < 1e-6)

        self.spin_for(1.0, every=lambda: self.target(0.35, 0.05, 0.35, yaw=0.5))
        before = self.stats()["pose"]
        self.spin_for(1.5, every=lambda: self.target(0.35, 0.05, 0.35 + 0.30))  # z fuera de caja -> 0.50 (15 cm)
        # Esperar a que el watchdog invalide el objetivo anterior: si no, un ciclo residual con ese objetivo
        # movería el brazo después de tomar la referencia y la comprobación 8 fallaría de forma intermitente.
        self.spin_for(BR["target_timeout_s"] + 0.3)
        mid = self.stats()["pose"]
        self.spin_for(1.0, every=lambda: self.target(0.05, -0.29, 0.10))  # >25 cm del TCP
        far = self.stats()["pose"]
        self.check("7. Recorta a la caja de trabajo (z_max)", mid[2] <= BR["workspace_z_max"] * 1000 + 1e-3
                   and mid[2] > before[2], f"z_mm={mid[2]:.1f}")
        self.check("8. Rechaza objetivos lejanos (> max_target_distance_m)", math.dist(mid[:3], far[:3]) < 1e-6)

        n1 = self.stats()["servo_ok"]
        self.spin_for(0.3 + BR["target_timeout_s"])
        n2 = self.stats()["servo_ok"]
        self.spin_for(1.0)
        n3 = self.stats()["servo_ok"]
        self.check("9. Watchdog: sin objetivo reciente deja de enviar", n3 == n2, f"llamadas {n1}->{n2}->{n3}")

        if gr:
            self.spin_for(1.5, every=lambda: (self.target(*[v / 1000 for v in mid[:3]]),
                                              self.pub_grip.publish(Float64MultiArray(data=[BR["gripper_policy_closed_pos"]]))))
            closed_state = self.grip_state
            self.spin_for(1.5, every=lambda: (self.target(*[v / 1000 for v in mid[:3]]),
                                              self.pub_grip.publish(Float64MultiArray(data=[BR["gripper_policy_open_pos"]]))))
            s = self.stats()
            self.check("10. Gripper por acción: cerrar y abrir (estado en escala de la política)",
                       s["gripper_goals"][-2:] == [BR["xarm_gripper_joint_closed"], BR["xarm_gripper_joint_open"]]
                       and abs(closed_state - BR["gripper_policy_closed_pos"]) < 1e-6
                       and abs(self.grip_state - BR["gripper_policy_open_pos"]) < 1e-6,
                       f"goals={s['gripper_goals']} estado_cerrado={closed_state} estado_abierto={self.grip_state}")

        self.fault("inject_error", True)
        self.spin_for(0.5)
        armed_after_err = self.armed()
        n_a = self.stats()["servo_ok"]
        self.spin_for(1.0, every=lambda: self.target(0.34, 0.05, 0.35))
        n_b = self.stats()["servo_ok"]
        self.check("11. Error del controlador (C22) -> desarme y sin comandos", not armed_after_err and n_a == n_b,
                   f"status: {self.status[:60]}")

        r = self.arm()  # err sigue activo: debe hacer clean_error y armar
        s = self.stats()
        self.check("12. Re-armado con error: clean_error + modo servo", r.success and s["clean_error"] >= 1
                   and s["err"] == 0 and s["mode"] == 1, r.message)

        self.fault("fail_servo", True)
        self.spin_for(1.5, every=lambda: self.target(0.33, 0.05, 0.35))
        armed_fail = self.armed()
        self.fault("fail_servo", False)
        self.check(f"13. {BR['max_consecutive_failures']} fallos seguidos de servo -> desarme", not armed_fail)

        self.arm()
        self.fault("slow_servo", True)
        self.spin_for(BR["servo_response_timeout_s"] * (BR["max_consecutive_failures"] + 2) + 2.0,
                      every=lambda: self.target(0.32, 0.05, 0.35))
        armed_slow = self.armed()
        self.fault("slow_servo", False)
        self.spin_for(1.5)
        self.check("14. Controlador sin respuesta (timeout) -> desarme", not armed_slow)

        self.arm()
        self.fault("pause_states", True)
        self.spin_for(BR["robot_state_timeout_s"] + 0.8, every=lambda: self.target(0.32, 0.05, 0.35))
        self.fault("pause_states", False)
        armed_lost = self.armed()
        self.check("15. Pérdida de robot_states -> desarme", not armed_lost)

        self.arm()
        modes_before = len(self.stats()["set_mode"])
        refused = self.call(Trigger, "/xarm5_bridge/go_home1")
        self.disarm()
        s = self.stats()
        self.check("16. go_home1 rechazado estando armado; disarm vuelve a modo posición (0)",
                   not refused.success and s["set_mode"][modes_before:][-1:] == [0], refused.message)
        r = self.call(Trigger, "/xarm5_bridge/go_home1", timeout=BR["home_timeout_s"] + 10)
        s = self.stats()
        self.check("17. go_home1 (HOME1 = cero UFACTORY) desarmado: set_servo_angle con home1_joints_rad",
                   r.success and s["set_servo_angle"][-1] == [round(v, 3) for v in BR["home1_joints_rad"]],
                   r.message)

        # --- Frenos (semántica SDK: bit 1 = freno liberado / motor habilitado) -------------------------
        all_axes = (1 << BR["dof"]) - 1
        r = self.arm()
        s = self.stats()
        cb = self.call(Trigger, "/xarm5_bridge/check_brakes")
        self.check("18. Armado verifica frenos liberados y motores habilitados en todos los ejes",
                   r.success and s["mt_brake"] == all_axes and s["mt_able"] == all_axes and cb.success,
                   cb.message)

        self.fault("stuck_brake", True)          # el freno de J3 se aplica con el brazo armado
        self.spin_for(0.5, every=lambda: self.target(0.32, 0.05, 0.35))
        n_a = self.stats()["servo_ok"]
        armed_stuck = self.armed()
        n_b = self.stats()["servo_ok"]
        self.check("19. Freno aplicado durante el movimiento -> desarme inmediato y sin comandos",
                   not armed_stuck and n_b == n_a and f"mt_brake=0b{all_axes & ~(1 << 2):0{BR['dof']}b}" in self.status,
                   self.status[-40:])

        r = self.arm()                            # el freno de J3 no se libera al habilitar
        s = self.stats()
        self.check("20. Armado con freno sin liberar (J3) -> rechazado, vuelve a modo posición",
                   not r.success and "J3" in r.message and s["mode"] == 0, r.message[:110])

        n_angle = len(self.stats()["set_servo_angle"])
        r = self.call(Trigger, "/xarm5_bridge/go_home2", timeout=BR["home_timeout_s"] + 10)
        self.check("21. go_home2 con freno sin liberar -> cancelado sin mover",
                   not r.success and len(self.stats()["set_servo_angle"]) == n_angle, r.message[:110])
        self.fault("stuck_brake", False)

        r = self.call(Trigger, "/xarm5_bridge/go_home2", timeout=BR["home_timeout_s"] + 10)
        s = self.stats()
        self.check("22. go_home2 (HOME2 = posición inicial UFACTORY, J4 -90): set_servo_angle con home2_joints_rad",
                   r.success and s["set_servo_angle"][-1] == [round(v, 3) for v in BR["home2_joints_rad"]],
                   r.message)

        # --- Límites articulares (caso real: J1 = +6.2822, a 0.001 rad de +2*pi -> C23) ----------------------
        real_home2 = [6.2822, -0.5001, -0.2717, 1.0231, -0.0022]
        self.call(SetFloat32List, "/fake_xarm/set_joints", SetFloat32List.Request(datas=real_home2))
        self.spin_for(0.5)
        r = self.arm()
        self.check("23. Armado con J1 pegada a +2*pi -> rechazado antes de habilitar motores",
                   not r.success and "J1" in r.message, r.message[:100])

        n_angle = len(self.stats()["set_servo_angle"])
        r = self.call(Trigger, "/xarm5_bridge/go_home2", timeout=BR["home_timeout_s"] + 10)
        self.check("24. go_home2 con J1 desfasada 2*pi -> rechazado (giro grande), sugiere unwind_j1",
                   not r.success and "unwind_j1" in r.message and len(self.stats()["set_servo_angle"]) == n_angle,
                   r.message[:100])

        r = self.call(Trigger, "/xarm5_bridge/unwind_j1", timeout=60)
        s = self.stats()
        sent = s["set_servo_angle"][-1]
        self.check("25. unwind_j1: solo J1 gira -2*pi (6.2822 -> -0.0010), resto igual",
                   r.success and abs(sent[0] - (6.2822 - 2 * math.pi)) < 1e-3 and sent[1:] == [round(v, 3) for v in real_home2[1:]],
                   r.message[:100])

        r = self.arm()
        ok_arm = r.success
        self.call(SetFloat32List, "/fake_xarm/set_joints",
                  SetFloat32List.Request(datas=[-0.0010, -0.5001, 0.15, 1.0231, -0.0022]))  # J3 a 0.04 de 0.19198
        armed_near = self.armed()
        s = self.stats()
        self.check("26. Tras unwind arma; junta acercándose a su límite -> desarme y modo posición",
                   ok_arm and not armed_near and s["mode"] == 0, self.status[-70:])

        # --- Freno intermitente de J3 (fallo real): reintento de liberación solo sobre esa junta -------------
        self.call(SetFloat32List, "/fake_xarm/set_joints",
                  SetFloat32List.Request(datas=[-0.0010, -0.5001, -0.2717, 1.0231, -0.0022]))
        self.call(SetInt16, "/fake_xarm/set_stuck_release_after", SetInt16.Request(data=2))
        self.fault("stuck_brake", True)
        n_calls = len(self.stats()["motion_enable_calls"])
        r = self.call(Trigger, "/xarm5_bridge/arm", timeout=30)
        calls = self.stats()["motion_enable_calls"][n_calls:]
        self.check("27. Freno de J3 que se libera al 2º intento -> reintento solo sobre J3 y armado OK",
                   r.success and "retry" in r.message and [3, 1] in calls and [3, 0] not in calls
                   and all(c[0] in (3, 8) for c in calls), f"{r.message[-60:]} | llamadas={calls}")
        self.disarm()

        self.call(SetInt16, "/fake_xarm/set_stuck_release_after", SetInt16.Request(data=-1))
        r = self.call(Trigger, "/xarm5_bridge/arm", timeout=30)
        self.check(f"28. Freno de J3 que nunca se libera -> rechazado tras {BR['brake_release_retries']} reintentos",
                   not r.success and f"{BR['brake_release_retries']} retry" in r.message and "J3" in r.message,
                   r.message[:110])
        self.fault("stuck_brake", False)

        # --- Poner / quitar frenos ------------------------------------------------------------------------
        self.disarm()
        r = self.call(Trigger, "/xarm5_bridge/brakes_off")
        s = self.stats()
        self.check("29. Quitar frenos: motores habilitados, modo posición y SIN armar",
                   r.success and s["mt_able"] == all_axes and s["mt_brake"] == all_axes and s["mode"] == 0
                   and not self.armed(), r.message[:90])
        self.arm()
        r = self.call(Trigger, "/xarm5_bridge/brakes_on")
        self.check("30. Poner frenos estando ARMADO -> rechazado", not r.success and "Disarm" in r.message, r.message)
        self.disarm()
        self.spin_for(1.2)
        r = self.call(Trigger, "/xarm5_bridge/brakes_on", timeout=20)
        s = self.stats()
        self.check("31. Poner frenos desarmado y en reposo: motion_enable(8, 0), frenos aplicados, informa desplazamiento",
                   r.success and s["mt_able"] == 0 and s["mt_brake"] == 0 and s["motion_enable_calls"][-1] == [8, 0]
                   and "mm" in r.message, r.message[:110])

        n_fail = sum(1 for _, ok in self.results if not ok)
        print(f"\nRESUMEN: {len(self.results) - n_fail}/{len(self.results)} comprobaciones OK", flush=True)
        return n_fail == 0


def main():
    rclpy.init()
    node = Checker()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
