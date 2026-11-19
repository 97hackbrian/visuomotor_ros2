#!/usr/bin/env python3
"""Driver xArm5 FALSO para probar xarm5_bridge sin hardware (mismas interfaces que xarm_api).

Imita el contrato del driver oficial (xarm_ros2, rama humble) que usa el puente:
  * Servicios en /<hw_ns>: motion_enable, set_mode, set_state, clean_error, set_servo_cartesian,
    set_servo_angle, set_gripper_enable, set_gripper_mode, set_gripper_speed, set_vacuum_gripper.
  * Acción /xarm_gripper/gripper_action (control_msgs/GripperCommand), posición 0 (abierto) .. 0.85 (cerrado).
  * Tópicos /<hw_ns>/robot_states (pose en mm/rad, mode, state, err) y /<hw_ns>/joint_states (drive_joint).
Como el controlador real, rechaza set_servo_cartesian (ret != 0) si mode != 1, state != 0 o hay error.

Frenos/motores: mt_brake/mt_able = 0 con motores deshabilitados (como el brazo real); motion_enable(8, 1) pone a 1 los
bits de los ejes 1..dof (bit 1 = freno liberado / motor habilitado, semántica del SDK) y motion_enable(8, 0) los borra.
Inyección de fallos (std_srvs/SetBool en /fake_xarm/...): inject_error, fail_servo, pause_states, slow_servo,
stuck_brake (el freno de J3 no se libera al habilitar; si ya estaba habilitado, se aplica de golpe).
Estadísticas: /fake_xarm/get_stats (std_srvs/Trigger, JSON en `message`).
Ángulos articulares: parámetro initial_joints_rad (por defecto home2) y /fake_xarm/set_joints (xarm_msgs/SetFloat32List)
para simular p. ej. J1 en +2*pi; set_servo_angle los actualiza; se publican en RobotMsg.angle y joint_states.

Uso (dentro del contenedor Jetson):  python3 tools/xarm5/fake_xarm_driver.py
"""

from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import Call, MoveCartesian, MoveJoint, SetFloat32, SetFloat32List, SetInt16, SetInt16ById
from xarm_msgs.srv import VacuumGripperCtrl

FAKE_ERROR_CODE = 22  # C22 (autocolisión), visto en los logs reales del brazo


class FakeXArmDriver(Node):
    def __init__(self):
        super().__init__("fake_xarm_driver")
        self.declare_parameter("hw_ns", "xarm")
        self.declare_parameter("initial_pose_mm_rad", [250.0, 0.0, 400.0, math.pi, 0.0, 0.0])
        self.declare_parameter("home_pose_mm_rad", [250.0, 0.0, 400.0, math.pi, 0.0, 0.0])
        self.declare_parameter("state_rate_hz", 100.0)
        self.declare_parameter("dof", 5)
        self.declare_parameter("stuck_brake_joint", 3)
        self.declare_parameter("initial_joints_rad", [-0.0010, -0.5001, -0.2717, 1.0231, -0.0022])
        self.dof = int(self.get_parameter("dof").value)
        self.stuck_joint = int(self.get_parameter("stuck_brake_joint").value)
        ns = "/" + self.get_parameter("hw_ns").value.strip("/")
        self.home_pose = list(self.get_parameter("home_pose_mm_rad").value)

        self.lock = threading.Lock()
        self.pose = list(self.get_parameter("initial_pose_mm_rad").value)
        self.mode, self.state, self.err = 0, 4, 0
        self.mt_brake, self.mt_able = 0, 0   # motores deshabilitados y frenos aplicados (estado real medido)
        self.gripper = 0.0
        self.joints = [float(v) for v in self.get_parameter("initial_joints_rad").value]
        self.vacuum_on = False
        self.stuck_release_after, self.stuck_attempts = -1, 0
        self.faults = {"fail_servo": False, "pause_states": False, "slow_servo": False, "stuck_brake": False}
        self.stats = {"motion_enable": 0, "set_mode": [], "set_state": [], "clean_error": 0,
                      "servo_ok": 0, "servo_rejected": 0, "max_step_mm": 0.0, "last_servo_pose": None,
                      "set_servo_angle": [], "gripper_enable": 0, "gripper_mode": 0, "gripper_speed": None,
                      "gripper_goals": [], "vacuum": [], "motion_enable_calls": []}

        cb = ReentrantCallbackGroup()
        srv = lambda t, name, fn: self.create_service(t, f"{ns}/{name}", fn, callback_group=cb)
        srv(SetInt16ById, "motion_enable", self._motion_enable)
        srv(SetInt16, "set_mode", self._set_mode)
        srv(SetInt16, "set_state", self._set_state)
        srv(Call, "clean_error", self._clean_error)
        srv(MoveCartesian, "set_servo_cartesian", self._servo_cartesian)
        srv(MoveJoint, "set_servo_angle", self._servo_angle)
        srv(SetInt16, "set_gripper_enable", self._gripper_enable)
        srv(SetInt16, "set_gripper_mode", self._gripper_mode)
        srv(SetFloat32, "set_gripper_speed", self._gripper_speed)
        srv(VacuumGripperCtrl, "set_vacuum_gripper", self._vacuum)
        for name in ("inject_error", "fail_servo", "pause_states", "slow_servo", "stuck_brake"):
            self.create_service(SetBool, f"/fake_xarm/{name}", lambda req, res, n=name: self._fault(n, req, res),
                                callback_group=cb)
        self.create_service(Trigger, "/fake_xarm/get_stats", self._get_stats, callback_group=cb)
        self.create_service(SetFloat32List, "/fake_xarm/set_joints", self._set_joints, callback_group=cb)
        self.create_service(SetInt16, "/fake_xarm/set_stuck_release_after", self._set_stuck_release_after,
                            callback_group=cb)
        ActionServer(self, GripperCommand, "/xarm_gripper/gripper_action", self._gripper_execute,
                     goal_callback=lambda _g: GoalResponse.ACCEPT,
                     cancel_callback=lambda _g: CancelResponse.ACCEPT, callback_group=cb)

        self.pub_state = self.create_publisher(RobotMsg, f"{ns}/robot_states", 10)
        self.pub_joints = self.create_publisher(JointState, f"{ns}/joint_states", 10)
        self.create_timer(1.0 / float(self.get_parameter("state_rate_hz").value), self._publish, callback_group=cb)
        self.get_logger().info(f"Driver xArm FALSO listo en {ns} (sin hardware).")

    # --- servicios del "controlador" ---------------------------------------------------------
    def _all_axes(self):
        return (1 << self.dof) - 1

    def _motion_enable(self, req, res):
        """id 8 = todas las juntas; 1..dof = una junta. Con stuck_brake, el freno de stuck_joint no se libera hasta
        el intento número stuck_release_after (-1 = nunca) de habilitar esa junta."""
        self.stats["motion_enable"] += 1
        self.stats["motion_enable_calls"].append([int(req.id), int(req.data)])
        joints = list(range(1, self.dof + 1)) if req.id == 8 else [int(req.id)]
        with self.lock:
            for j in joints:
                bit = 1 << (j - 1)
                if req.data:
                    self.mt_able |= bit
                    self.mt_brake |= bit
                    if self.faults["stuck_brake"] and j == self.stuck_joint:
                        self.stuck_attempts += 1
                        if self.stuck_release_after < 0 or self.stuck_attempts < self.stuck_release_after:
                            self.mt_brake &= ~bit
                else:
                    self.mt_able &= ~bit
                    self.mt_brake &= ~bit
        res.ret = 0
        return res

    def _set_stuck_release_after(self, req, res):
        with self.lock:
            self.stuck_release_after = int(req.data)
            self.stuck_attempts = 0
        res.ret = 0
        return res

    def _set_mode(self, req, res):
        with self.lock:
            self.mode = req.data
        self.stats["set_mode"].append(req.data)
        res.ret = 0
        return res

    def _set_state(self, req, res):
        with self.lock:
            self.state = req.data
        self.stats["set_state"].append(req.data)
        res.ret = 0
        return res

    def _clean_error(self, req, res):
        with self.lock:
            self.err = 0
        self.stats["clean_error"] += 1
        res.ret = 0
        return res

    def _servo_cartesian(self, req, res):
        if self.faults["slow_servo"]:
            time.sleep(1.0)
        with self.lock:
            ok = (self.mode == 1 and self.state == 0 and self.err == 0 and not self.faults["fail_servo"]
                  and self.mt_able == self._all_axes() and self.mt_brake == self._all_axes()
                  and len(req.pose) >= 6 and all(math.isfinite(v) for v in req.pose))
            if ok:
                step = math.dist(self.pose[:3], list(req.pose[:3]))
                self.stats["max_step_mm"] = max(self.stats["max_step_mm"], step)
                self.pose = [float(v) for v in req.pose[:6]]
                self.stats["servo_ok"] += 1
                self.stats["last_servo_pose"] = [round(v, 3) for v in self.pose]
            else:
                self.stats["servo_rejected"] += 1
        res.ret = 0 if ok else 1
        res.message = "ok" if ok else "rechazado (modo/estado/error/fallo inyectado)"
        return res

    def _servo_angle(self, req, res):
        self.stats["set_servo_angle"].append([round(v, 3) for v in req.angles])
        with self.lock:
            ok = (self.mode == 0 and self.err == 0 and len(req.angles) >= self.dof
                  and self.mt_able == self._all_axes() and self.mt_brake == self._all_axes())
            if ok:
                self.pose = list(self.home_pose)
                self.joints = [float(v) for v in req.angles[:self.dof]]
        res.ret = 0 if ok else 1
        return res

    def _set_joints(self, req, res):
        with self.lock:
            self.joints = [float(v) for v in req.datas[:self.dof]]
        res.ret = 0
        return res

    def _gripper_enable(self, req, res):
        self.stats["gripper_enable"] += 1
        res.ret = 0
        return res

    def _gripper_mode(self, req, res):
        self.stats["gripper_mode"] += 1
        res.ret = 0
        return res

    def _gripper_speed(self, req, res):
        self.stats["gripper_speed"] = req.data
        res.ret = 0
        return res

    def _vacuum(self, req, res):
        self.vacuum_on = bool(req.on)
        self.stats["vacuum"].append(bool(req.on))
        res.ret = 0
        return res

    def _gripper_execute(self, goal_handle):
        target = float(goal_handle.request.command.position)
        self.stats["gripper_goals"].append(round(target, 3))
        start = self.gripper
        for i in range(1, 11):  # ~0.3 s de movimiento
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return GripperCommand.Result()
            self.gripper = start + (target - start) * i / 10.0
            time.sleep(0.03)
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = self.gripper
        result.reached_goal = True
        return result

    # --- fallos y estadísticas ---------------------------------------------------------------
    def _fault(self, name, req, res):
        if name == "inject_error":
            with self.lock:
                self.err = FAKE_ERROR_CODE if req.data else 0
                if req.data:
                    self.state = 4  # el controlador real pasa a estado detenido ante un error
        else:
            self.faults[name] = bool(req.data)
            if name == "stuck_brake" and req.data:
                with self.lock:  # el freno se aplica en un eje mientras los motores siguen habilitados
                    self.mt_brake &= ~(1 << (self.stuck_joint - 1))
        res.success, res.message = True, f"{name}={req.data}"
        return res

    def _get_stats(self, _req, res):
        with self.lock:
            data = dict(self.stats, mode=self.mode, state=self.state, err=self.err, pose=self.pose, joints=self.joints,
                        mt_brake=self.mt_brake, mt_able=self.mt_able,
                        gripper=self.gripper, vacuum_on=self.vacuum_on)
        res.success, res.message = True, json.dumps(data)
        return res

    def _publish(self):
        if self.faults["pause_states"]:
            return
        with self.lock:
            msg = RobotMsg()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.state, msg.mode, msg.err = self.state, self.mode, self.err
            msg.mt_brake, msg.mt_able = self.mt_brake, self.mt_able
            msg.angle = [float(v) for v in self.joints]
            msg.pose = [float(v) for v in self.pose]
        self.pub_state.publish(msg)
        js = JointState()
        js.header.stamp = msg.header.stamp
        js.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "drive_joint"]
        js.position = [float(v) for v in self.joints] + [float(self.gripper)]
        self.pub_joints.publish(js)


def main():
    rclpy.init()
    node = FakeXArmDriver()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
