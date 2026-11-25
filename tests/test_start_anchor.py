"""Anclaje de la pose de inicio con ArUco + IMU (visuomotor_data_collection/start_anchor.py) y su uso en el conversor."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))
from visuomotor_data_collection.start_anchor import (TableMarkers, camera_pose_from_markers,  # noqa: E402
                                                     correct_tilt_with_up, estimate_start_pose, inv_T, pose_error,
                                                     scale_intrinsics, tilt_from_up, triangulate_point)

K = np.array([[606.8, 0, 322.0], [0, 606.5, 254.7], [0, 0, 1.0]])


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def marker(center, side=0.051):
    h = side / 2
    return np.array([center + [-h, h, 0], center + [h, h, 0], center + [h, -h, 0], center + [-h, -h, 0]])


TABLE = TableMarkers("DICT_4X4_50", {12: marker(np.array([0.69, -0.19, 0.02])),
                                     14: marker(np.array([0.69, 0.05, 0.02]))})


def home3_like_camera():
    """Cámara a ~0,75 m sobre la mesa, mirando hacia abajo e inclinada hacia los marcadores."""
    cam, target = np.array([0.25, -0.07, 0.77]), np.array([0.69, -0.07, 0.02])
    z = (target - cam) / np.linalg.norm(target - cam)               # eje óptico hacia los marcadores
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]                         # arriba de la imagen = lejos de la base
    T[:3, 3] = cam
    return T


def project(T_base_cam, pts):
    T_cb = inv_T(T_base_cam)
    pc = (T_cb[:3, :3] @ pts.T).T + T_cb[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:]


class StartAnchorTest(unittest.TestCase):
    def setUp(self):
        self.T = home3_like_camera()
        self.live = {i: project(self.T, c) for i, c in TABLE.corners_base.items()}

    def test_markers_visible(self):
        for uv in self.live.values():
            self.assertTrue(np.all((uv > 0) & (uv < [640, 480])))

    def test_pnp_exact(self):
        est = camera_pose_from_markers(self.live, TABLE, K)
        mm, deg = pose_error(est["T_base_cam"], self.T)
        self.assertLess(mm, 0.01)
        self.assertLess(deg, 1e-4)
        self.assertEqual(est["markers"], [12, 14])

    def test_pnp_noise(self):
        rng = np.random.default_rng(0)
        noisy = {i: uv + rng.normal(0, 0.3, uv.shape) for i, uv in self.live.items()}
        mm, deg = pose_error(camera_pose_from_markers(noisy, TABLE, K)["T_base_cam"], self.T)
        self.assertLess(mm, 20.0)
        self.assertLess(deg, 1.5)

    def test_tilt_correction_keeps_position_and_fixes_up(self):
        bad = self.T.copy()
        bad[:3, :3] = self.T[:3, :3] @ rot("x", 3.0)                 # 3° de inclinación de más
        up_true = self.T[:3, :3].T @ np.array([0, 0, 1.0])           # lo que mediría el acelerómetro
        self.assertGreater(tilt_from_up(bad, up_true), 2.9)
        fixed = correct_tilt_with_up(bad, up_true)
        self.assertLess(tilt_from_up(fixed, up_true), 1e-6)
        np.testing.assert_allclose(fixed[:3, 3], bad[:3, 3])

    def test_triangulation(self):
        X = np.array([0.7, -0.1, 0.02])
        poses = [self.T, self.T @ np.r_[np.c_[rot("y", 8), [0.1, 0, 0]], [[0, 0, 0, 1]]]]
        Ps = [K @ inv_T(T)[:3, :] for T in poses]
        Xe, err = triangulate_point(Ps, [project(T, X[None])[0] for T in poses])
        np.testing.assert_allclose(Xe, X, atol=1e-9)
        self.assertLess(err, 1e-6)

    def test_scale_intrinsics_matches_resize(self):
        K7 = scale_intrinsics(K, (640, 480), (256, 256))
        uv = project(self.T, TABLE.corners_base[12])
        uv7 = (uv + 0.5) * [256 / 640, 256 / 480] - 0.5             # convención de centros de píxel de cv2.resize
        T_cb = inv_T(self.T)
        pc = (T_cb[:3, :3] @ TABLE.corners_base[12].T).T + T_cb[:3, 3]
        p7 = (K7 @ pc.T).T
        np.testing.assert_allclose(p7[:, :2] / p7[:, 2:], uv7, atol=1e-9)

    def test_estimate_start_pose_with_imu(self):
        accel_up_imu = (self.T[:3, :3].T @ np.array([0, 0, 1.0])) * 9.81      # IMU = cámara (T_cam_imu = I)
        accel = np.tile(accel_up_imu, (50, 1))
        res = estimate_start_pose([np.zeros((480, 640), np.uint8)] * 3, TABLE, K, accel=accel, T_cam_imu=np.eye(4),
                                  detect=lambda g, d: self.live)
        self.assertTrue(res["imu_used"])
        self.assertEqual(res["flags"], [])
        mm, deg = pose_error(res["T_base_cam"], self.T)
        self.assertLess(mm, 0.01)
        self.assertLess(deg, 1e-4)

    def test_estimate_start_pose_flags_disagreement(self):
        wrong = (self.T[:3, :3] @ rot("x", 10)).T @ np.array([0, 0, 1.0]) * 9.81   # IMU 10° distinto
        res = estimate_start_pose([np.zeros((480, 640), np.uint8)] * 3, TABLE, K, accel=np.tile(wrong, (50, 1)),
                                  T_cam_imu=np.eye(4), detect=lambda g, d: self.live)
        self.assertFalse(res["imu_used"])
        self.assertTrue(any("discrepan" in f for f in res["flags"]))

    def test_no_markers(self):
        res = estimate_start_pose([np.zeros((480, 640), np.uint8)], TABLE, K, detect=lambda g, d: {})
        self.assertIsNone(res["T_base_cam"])


class ConverterStartPoseTest(unittest.TestCase):
    def test_start_pose_equal_home_is_identical_and_offset_moves_start(self):
        sys.path.insert(0, str(REPO / "tests"))
        from test_convert_vio_dataset import ALIGN, BRIDGE, POLICY, Converter  # noqa: E402
        conv = Converter(BRIDGE, ALIGN, POLICY)
        t = np.linspace(0, 1, 30)
        state = np.zeros((30, 8))
        state[:, 0] = 0.1 * t                                         # odom: avanza 10 cm
        state[:, 6] = 1.0
        action = np.zeros((30, 9))
        action[:, :8] = state
        a, _ = conv.convert_episode(state, action, 20.0)
        b, sb = conv.convert_episode(state, action, 20.0, conv.t_base_home)
        np.testing.assert_allclose(a["state"], b["state"], atol=1e-6)
        self.assertEqual(sb["start_anchor"], "measured")
        moved = conv.t_base_home.copy()
        moved[:3, 3] += [0.0, 0.02, -0.01]
        c, _ = conv.convert_episode(state, action, 20.0, moved)
        np.testing.assert_allclose(c["state"][0, :3] - a["state"][0, :3], [0.0, 0.02, -0.01], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
