"""Pruebas de visuomotor_core.utils.rotations, dataset.trajectory_filters, la recomposición de acciones
``relative_ee`` de DiffusionPolicy y la EMA de pesos.

Ejecutar (contenedor):  python3 -m pytest tests/test_rotations_and_filters.py -q
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_core"))
sys.path.insert(0, str(ROOT / "visuomotor_bringup"))

from visuomotor_core.utils.rotations import (  # noqa: E402
    canonicalize, compose_pose, euler_xyz_to_quat, normalize, quat_multiply, quat_rotate, relative_pose,
    weighted_average)
from visuomotor_core.dataset.trajectory_filters import (  # noqa: E402
    TrajectoryFilterConfig, hampel, lowpass_zero_phase, smooth_pose_trajectory)
from visuomotor_bringup.xarm5_interface import arm_plane_quat, rpy_to_quat  # noqa: E402


def rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class TestRotations(unittest.TestCase):
    def test_euler_matches_xarm_convention(self):
        for rpy in [(3.10176, -0.42961, -0.01136), (0.3, -0.2, 1.1), (math.pi, 0.0, 0.0)]:
            a = euler_xyz_to_quat(*rpy)
            b = rpy_to_quat(*rpy)
            self.assertAlmostEqual(abs(float(np.dot(a, b))), 1.0, places=12)

    def test_home_hemisphere_is_continuous_near_tool_down(self):
        """Con qw >= 0 el cuaternión salta al cruzar yaw - 2·acimut = 0 en HOME3; con 'home' no."""
        q_home = rpy_to_quat(3.10176, -0.42961, -0.01136)
        yaws = np.linspace(-0.05, 0.05, 101)
        qs = np.array([arm_plane_quat(0.0, -0.43, y) for y in yaws])
        flips_qw = int((np.sum(canonicalize(qs, "qw_positive")[1:] * canonicalize(qs, "qw_positive")[:-1], 1) < 0).sum())
        q_home_c = canonicalize(qs, "home", q_home)
        flips_home = int((np.sum(q_home_c[1:] * q_home_c[:-1], 1) < 0).sum())
        self.assertGreater(flips_qw, 0)
        self.assertEqual(flips_home, 0)
        self.assertTrue(np.all(q_home_c @ normalize(q_home) >= 0))

    def test_relative_compose_roundtrip(self):
        rng = np.random.default_rng(0)
        q1, q2 = rand_quat(rng, 50), rand_quat(rng, 50)
        p1, p2 = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
        dp, dq = relative_pose(p1, q1, p2, q2)
        self.assertTrue(np.all(dq[:, 3] >= 0))
        p, q = compose_pose(p1, q1, dp, dq)
        np.testing.assert_allclose(p, p2, atol=1e-10)
        np.testing.assert_allclose(np.abs(np.sum(q * q2, 1)), 1.0, atol=1e-10)
        # la traslación relativa vive en el marco del efector: rotarla con q1 devuelve el desplazamiento en base
        np.testing.assert_allclose(quat_rotate(q1, dp), p2 - p1, atol=1e-10)

    def test_weighted_average_aligns_signs(self):
        q = normalize(np.array([0.977, -0.01, 0.213, 0.02]))
        avg = weighted_average(np.stack([q, -q, q]), np.ones(3), q_ref=q)
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(abs(float(avg @ q)), 1.0, places=9)  # sin alinear, q + (-q) + q tendría norma 1/3
        self.assertIsNone(weighted_average(np.stack([q, quat_multiply(q, [1, 0, 0, 0])]), np.ones(2), q_ref=q,
                                           min_norm=0.9))


class TestTrajectoryFilters(unittest.TestCase):
    def test_hampel_removes_isolated_spike(self):
        x = np.linspace(0, 1, 200)[:, None] * np.ones((1, 3))
        x[100] += 0.08
        y, out = hampel(x, 3, 3.0)
        self.assertTrue(out[100].all())
        self.assertLess(np.abs(y[100] - x[99]).max(), 0.02)

    def test_lowpass_preserves_slow_motion_and_static_ends(self):
        fs = 20.0
        t = np.arange(400) / fs
        sig = np.clip(t - 5, 0, 10)[:, None] * 0.05 * np.array([[1.0, 0.5, -0.3]])  # reposo-rampa-reposo
        sig = np.where(t[:, None] > 15, sig[-1], sig)
        noisy = sig + np.random.default_rng(1).normal(scale=0.008 / np.sqrt(3), size=sig.shape)  # 8 mm RMS 3D (v7)
        f = lowpass_zero_phase(noisy, fs, 1.5, 4)
        self.assertLess(np.sqrt(((f - sig) ** 2).sum(1).mean()), 0.004)  # ruido 8 mm -> < 4 mm
        self.assertLess(np.linalg.norm(f[0] - sig[0]), 0.01)
        self.assertLess(np.linalg.norm(f[-1] - sig[-1]), 0.01)

    def test_smooth_pose_trajectory_keeps_rotation(self):
        from scipy.spatial.transform import Rotation as R
        fs = 20.0
        n = 300
        rv = np.linspace(0, 1.2, n)[:, None] * np.array([[0.0, 0.0, 1.0]])
        q = (R.from_euler("x", 3.1) * R.from_rotvec(rv)).as_quat()
        p = np.linspace(0, 0.5, n)[:, None] * np.ones((1, 3))
        cfg = TrajectoryFilterConfig(enabled=True)
        pf, qf, st = smooth_pose_trajectory(p, q, fs, cfg)
        ang = (R.from_quat(q).inv() * R.from_quat(qf)).magnitude()
        self.assertLess(np.degrees(ang).max(), 1.0)
        self.assertLess(st["position_removed_rms_mm"], 2.0)


class TestPolicyRelativeActions(unittest.TestCase):
    def test_actions_to_absolute_matches_numpy(self):
        import torch
        from visuomotor_core.models.config import DiffusionConfig
        from visuomotor_core.models.diffusion_policy import DiffusionPolicy

        q_home = list(rpy_to_quat(3.10176, -0.42961, -0.01136))
        cfg = DiffusionConfig(action_representation="relative_ee", quaternion_hemisphere="home",
                              quaternion_reference=q_home)
        cfg.input_shapes["observation.state"] = [8]
        cfg.output_shapes["action"] = [9]
        policy = DiffusionPolicy(cfg)
        rng = np.random.default_rng(3)
        state = np.concatenate([rng.normal(scale=0.1, size=(2, 3)), np.stack([q_home, -np.array(q_home)]),
                                np.zeros((2, 1))], axis=1)
        dp = rng.normal(scale=0.05, size=(2, 5, 3))
        dq = canonicalize(normalize(np.concatenate([rng.normal(scale=0.05, size=(2, 5, 3)), np.ones((2, 5, 1))], -1)),
                          "qw_positive")
        acts = np.concatenate([dp, dq, np.zeros((2, 5, 2))], -1)
        out = policy.actions_to_absolute(torch.tensor(acts, dtype=torch.float64),
                                         torch.tensor(state, dtype=torch.float64)).numpy()
        p_ref, q_ref = compose_pose(state[:, None, :3], state[:, None, 3:7], dp, dq)
        np.testing.assert_allclose(out[..., :3], p_ref, atol=1e-9)
        np.testing.assert_allclose(np.abs(np.sum(out[..., 3:7] * q_ref, -1)), 1.0, atol=1e-9)
        self.assertTrue(np.all(out[..., 3:7] @ normalize(q_home) >= 0))  # hemisferio home


class TestEMA(unittest.TestCase):
    def test_ema_decay_schedule_and_update(self):
        import torch
        from visuomotor_core.training.ema import EMAModel
        m = torch.nn.Linear(3, 3)
        ema = EMAModel(m, power=0.75, max_value=0.9999)
        self.assertAlmostEqual(ema.get_decay(0), 0.0)
        self.assertAlmostEqual(ema.get_decay(4001), 1 - 4001 ** -0.75, places=6)
        self.assertAlmostEqual(ema.get_decay(10 ** 7), 0.9999)
        with torch.no_grad():
            m.weight.fill_(1.0)
        ema.step(m)  # decay 0 -> copia
        self.assertTrue(torch.allclose(ema.model.weight, m.weight))


if __name__ == "__main__":
    unittest.main()


class TestReviewFixes(unittest.TestCase):
    def test_unwrap_rotvec_across_pi(self):
        from scipy.spatial.transform import Rotation as R
        from visuomotor_core.dataset.trajectory_filters import unwrap_rotvec
        ang = np.linspace(0.0, 1.5 * math.pi, 200)               # giro continuo de 270° sobre z
        rv = R.from_rotvec(ang[:, None] * np.array([[0.0, 0.0, 1.0]])).as_rotvec()   # scipy envuelve a |r| <= pi
        self.assertGreater(np.abs(np.diff(rv, axis=0)).max(), 1.0)  # salto al cruzar pi
        un = unwrap_rotvec(rv)
        self.assertLess(np.abs(np.diff(un, axis=0)).max(), 0.05)    # continuo
        np.testing.assert_allclose(R.from_rotvec(un).as_matrix(), R.from_rotvec(rv).as_matrix(), atol=1e-9)

    def test_split_requires_training_episodes(self):
        from visuomotor_core.dataset.zarr_dataset import split_episodes
        tr, va = split_episodes([f"episode_{i}" for i in range(10)], 0.2, seed=1)
        self.assertEqual((len(tr), len(va)), (8, 2))
        self.assertFalse(set(tr) & set(va))
        with self.assertRaises(ValueError):
            split_episodes(["episode_0", "episode_1", "episode_2"], 0.9)
