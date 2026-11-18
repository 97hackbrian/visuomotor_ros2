"""Pruebas de la lógica pura del puente xArm5 (sin ROS ni hardware).

Ejecutar:  python3 -m pytest tests/test_xarm5_interface.py -q
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuomotor_bringup"))

from visuomotor_bringup.xarm5_interface import (  # noqa: E402
    GripperBinarizer,
    WorkspaceLimits,
    brake_report,
    clamp_yaw,
    joints_with_bit_cleared,
    from_robot_pose,
    is_finite_pose,
    limit_angle_step,
    limit_step,
    linear_map,
    normalize_quat,
    project_orientation,
    quat_to_matrix,
    quat_to_rpy,
    rpy_to_quat,
    to_servo_pose,
    tool_yaw,
    wrap_angle,
)


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class TestRotations(unittest.TestCase):
    def test_rpy_convention_matches_fixed_axis_xyz(self):
        """La convención xArm es R = Rz(yaw) Ry(pitch) Rx(roll)."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            r, p, y = rng.uniform(-math.pi, math.pi), rng.uniform(-1.4, 1.4), rng.uniform(-math.pi, math.pi)
            expected = _rot("z", y) @ _rot("y", p) @ _rot("x", r)
            np.testing.assert_allclose(quat_to_matrix(rpy_to_quat(r, p, y)), expected, atol=1e-9)

    def test_rpy_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            rpy = (rng.uniform(-math.pi, math.pi), rng.uniform(-1.4, 1.4), rng.uniform(-math.pi, math.pi))
            back = quat_to_rpy(rpy_to_quat(*rpy))
            np.testing.assert_allclose(quat_to_matrix(rpy_to_quat(*back)), quat_to_matrix(rpy_to_quat(*rpy)),
                                       atol=1e-9)

    def test_gimbal_lock_is_finite(self):
        for pitch in (math.pi / 2, -math.pi / 2):
            r, p, y = quat_to_rpy(rpy_to_quat(0.3, pitch, 0.7))
            self.assertTrue(all(math.isfinite(v) for v in (r, p, y)))
            self.assertAlmostEqual(abs(p), math.pi / 2, places=6)

    def test_qw_nonnegative(self):
        self.assertGreaterEqual(rpy_to_quat(math.pi, 0.0, 3.0)[3], 0.0)
        self.assertGreaterEqual(normalize_quat([0, 0, 0, -2])[3], 0.0)

    def test_zero_quat_raises(self):
        with self.assertRaises(ValueError):
            normalize_quat([0, 0, 0, 0])

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-0.5), -0.5)


class TestOrientationProjection(unittest.TestCase):
    TOP_DOWN = (math.pi, 0.0, 0.0)

    def test_top_down_keeps_policy_yaw(self):
        for yaw in (-2.5, -0.4, 0.0, 1.2, 3.0):
            q = rpy_to_quat(math.pi, 0.0, yaw)
            r, p, y = project_orientation(q, "top_down_yaw", self.TOP_DOWN)
            self.assertAlmostEqual(r, math.pi)
            self.assertAlmostEqual(p, 0.0)
            self.assertAlmostEqual(wrap_angle(y - yaw), 0.0, places=9)

    def test_top_down_removes_tilt(self):
        """Una orientación inclinada (inalcanzable con 5 GDL) se proyecta a garra vertical."""
        q = rpy_to_quat(math.pi - 0.3, 0.25, 0.8)
        r, p, y = project_orientation(q, "top_down_yaw", self.TOP_DOWN)
        self.assertAlmostEqual(r, math.pi)
        self.assertAlmostEqual(p, 0.0)
        self.assertTrue(abs(wrap_angle(y - 0.8)) < 0.3)

    def test_top_down_yaw_offset_and_fallback(self):
        q = rpy_to_quat(math.pi, 0.0, 0.5)
        _, _, y = project_orientation(q, "top_down_yaw", self.TOP_DOWN, yaw_offset=0.25)
        self.assertAlmostEqual(y, 0.75)
        q_vertical_x = rpy_to_quat(0.0, -math.pi / 2, 0.0)  # eje X de la herramienta vertical
        self.assertIsNone(tool_yaw(q_vertical_x))
        _, _, y = project_orientation(q_vertical_x, "top_down_yaw", self.TOP_DOWN, fallback_yaw=1.1)
        self.assertAlmostEqual(y, 1.1)

    def test_fixed_and_passthrough(self):
        q = rpy_to_quat(2.0, 0.3, -1.0)
        self.assertEqual(project_orientation(q, "fixed", (math.pi, 0.0, 0.4)), (math.pi, 0.0, 0.4))
        r, p, y = project_orientation(q, "passthrough", self.TOP_DOWN)
        np.testing.assert_allclose([r, p, y], [2.0, 0.3, -1.0], atol=1e-9)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            project_orientation([0, 0, 0, 1], "6dof", self.TOP_DOWN)


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.ws = WorkspaceLimits(0.15, 0.55, -0.30, 0.30, 0.10, 0.50, 0.15, 0.60)

    def test_inside_untouched(self):
        p, clamped = self.ws.clamp([0.3, 0.1, 0.3])
        self.assertFalse(clamped)
        np.testing.assert_allclose(p, [0.3, 0.1, 0.3])

    def test_box_and_radius(self):
        p, clamped = self.ws.clamp([1.0, 0.0, -1.0])
        self.assertTrue(clamped)
        np.testing.assert_allclose(p, [0.55, 0.0, 0.10])
        p, _ = self.ws.clamp([0.5, 0.5, 0.3])  # fuera de radio y de y_max
        self.assertLessEqual(math.hypot(p[0], p[1]), 0.60 + 1e-9)
        self.assertLessEqual(p[1], 0.30 + 1e-12)

    def test_min_radius_near_base_axis(self):
        p, clamped = self.ws.clamp([0.0, 0.0, 0.3])
        self.assertTrue(clamped)
        self.assertGreaterEqual(math.hypot(p[0], p[1]), 0.15 - 1e-9)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            WorkspaceLimits(0.5, 0.1, -0.3, 0.3, 0.1, 0.5, 0.15, 0.6)


class TestStepLimits(unittest.TestCase):
    def test_limit_step(self):
        np.testing.assert_allclose(limit_step([0, 0, 0], [0.001, 0, 0], 0.01), [0.001, 0, 0])
        out = limit_step([0, 0, 0], [3.0, 4.0, 0], 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.5)
        np.testing.assert_allclose(out / 0.5, [0.6, 0.8, 0.0])

    def test_reaches_target_at_speed(self):
        """A 30 Hz y 0.05 m/s, 10 cm se recorren en ~60 pasos sin sobrepasar el objetivo."""
        cur, goal, n = np.zeros(3), np.array([0.1, 0, 0]), 0
        while np.linalg.norm(goal - cur) > 1e-12:
            nxt = limit_step(cur, goal, 0.05 / 30.0)
            self.assertLessEqual(np.linalg.norm(nxt - cur), 0.05 / 30.0 + 1e-12)
            cur, n = nxt, n + 1
        self.assertEqual(n, 60)

    def test_angle_step_shortest_path(self):
        self.assertAlmostEqual(limit_angle_step(3.0, -3.0, 0.1), wrap_angle(3.1))
        self.assertAlmostEqual(limit_angle_step(0.0, 0.05, 0.1), 0.05)
        self.assertAlmostEqual(limit_angle_step(0.0, -1.0, 0.1), -0.1)

    def test_angle_step_without_wrap_never_crosses_pi(self):
        """Con yaw acotado, de 2.5 a -2.5 se va por 0 (no por ±pi)."""
        cur = 2.5
        for _ in range(200):
            cur = limit_angle_step(cur, -2.5, 0.05, wrap=False)
            self.assertLessEqual(abs(cur), 2.5 + 1e-12)
        self.assertAlmostEqual(cur, -2.5)

    def test_clamp_yaw(self):
        self.assertAlmostEqual(clamp_yaw(3.0, -2.6, 2.6), 2.6)
        self.assertAlmostEqual(clamp_yaw(-3.0, -2.6, 2.6), -2.6)
        self.assertAlmostEqual(clamp_yaw(2.0 * math.pi + 0.3, -2.6, 2.6), 0.3)
        with self.assertRaises(ValueError):
            clamp_yaw(0.0, 1.0, -1.0)

    def test_is_finite_pose(self):
        self.assertTrue(is_finite_pose([0.3, 0, 0.3], [0, 0, 0, 1]))
        self.assertFalse(is_finite_pose([float("nan"), 0, 0.3], [0, 0, 0, 1]))
        self.assertFalse(is_finite_pose([0.3, 0, 0.3], [0, float("inf"), 0, 1]))
        self.assertFalse(is_finite_pose([0.3, 0, 0.3], [0, 0, 0, 0]))


class TestBrakes(unittest.TestCase):
    def test_all_released_and_enabled(self):
        ok, stuck, disabled = brake_report(0b11111, 0b11111, 5)
        self.assertTrue(ok)
        self.assertEqual((stuck, disabled), ([], []))

    def test_motors_off_brakes_on(self):
        """Estado medido en el xArm5 real con motores deshabilitados: mt_brake=0, mt_able=0."""
        ok, stuck, disabled = brake_report(0, 0, 5)
        self.assertFalse(ok)
        self.assertEqual(stuck, [1, 2, 3, 4, 5])
        self.assertEqual(disabled, [1, 2, 3, 4, 5])

    def test_single_stuck_brake(self):
        ok, stuck, disabled = brake_report(0b11011, 0b11111, 5)  # J3 con freno aplicado
        self.assertFalse(ok)
        self.assertEqual(stuck, [3])
        self.assertEqual(disabled, [])

    def test_bits_beyond_dof_ignored(self):
        self.assertTrue(brake_report(0b0011111, 0b1111111, 5)[0])
        self.assertEqual(joints_with_bit_cleared(0b10000000 | 0b11111, 5), [])
        with self.assertRaises(ValueError):
            joints_with_bit_cleared(0, 0)


class TestUnitsAndGripper(unittest.TestCase):
    def test_units(self):
        self.assertEqual(to_servo_pose([0.25, -0.1, 0.4], [math.pi, 0.0, 0.2]),
                         [250.0, -100.0, 400.0, math.pi, 0.0, 0.2])
        xyz, rpy = from_robot_pose([250.0, -100.0, 400.0, 3.1, 0.0, 0.2])
        np.testing.assert_allclose(xyz, [0.25, -0.1, 0.4])
        self.assertEqual(rpy, (3.1, 0.0, 0.2))
        with self.assertRaises(ValueError):
            from_robot_pose([1, 2, 3])

    def test_gripper_binarizer_with_policy_values(self):
        g = GripperBinarizer(close_threshold=-0.001, hysteresis=0.0005)
        self.assertFalse(g.update(0.0))          # abierto
        self.assertFalse(g.update(-0.0009))      # aún sobre el umbral
        self.assertTrue(g.update(-0.01))         # cerrado (gripper_closed_pos)
        self.assertTrue(g.update(-0.0008))       # histéresis: sigue cerrado
        self.assertFalse(g.update(0.0))          # vuelve a abrir
        with self.assertRaises(ValueError):
            GripperBinarizer(-0.001, -1.0)

    def test_gripper_state_mapping(self):
        """drive_joint real (0 abierto .. 0.85 cerrado) -> escala de la política (0.0 .. -0.01)."""
        self.assertAlmostEqual(linear_map(0.0, 0.0, 0.85, 0.0, -0.01), 0.0)
        self.assertAlmostEqual(linear_map(0.85, 0.0, 0.85, 0.0, -0.01), -0.01)
        self.assertAlmostEqual(linear_map(0.425, 0.0, 0.85, 0.0, -0.01), -0.005)
        self.assertAlmostEqual(linear_map(2.0, 0.0, 0.85, 0.0, -0.01), -0.01)  # recortado


if __name__ == "__main__":
    unittest.main()
