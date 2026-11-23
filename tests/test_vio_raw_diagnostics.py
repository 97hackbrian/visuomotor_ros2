"""Pruebas del diagnóstico de grabaciones crudas (tools/xarm5/analyze_vio_dataset_quality.py --raw-diagnostics)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "xarm5"))

from analyze_vio_dataset_quality import (  # noqa: E402
    fill_masked, frame_speeds, null_pose_mask, step_discontinuities)


def _states(p, q=(0.0, 0.7071, 0.0, 0.7071)):
    s = np.zeros((len(p), 8))
    s[:, :3] = p
    s[:, 3:7] = q
    return s


class NullPoseTest(unittest.TestCase):
    def test_detects_identity_with_minus_initial_position(self):
        ip = np.array([0.3, -0.2, 0.4])
        p = np.cumsum(np.full((10, 3), 0.002), axis=0)
        s = _states(p)
        s[4, :3] = -ip
        s[4, 3:7] = (0.0, 0.0, 0.0, 1.0)
        mask = null_pose_mask(s, ip)
        self.assertEqual(mask.tolist(), [i == 4 for i in range(10)])

    def test_identity_orientation_alone_is_not_null_when_position_differs(self):
        s = _states(np.zeros((3, 3)), q=(0.0, 0.0, 0.0, 1.0))
        self.assertFalse(null_pose_mask(s, [0.3, 0.0, 0.0]).any())
        self.assertTrue(null_pose_mask(s, None).all())        # sin pose inicial: solo el cuaternión

    def test_fill_masked_interpolates(self):
        p = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
        p[2] = 99.0
        out = fill_masked(p, np.array([False, False, True, False, False]))
        np.testing.assert_allclose(out[2], [2.0, 2.0, 2.0])


class DiscontinuityTest(unittest.TestCase):
    def test_step_offset_is_reported_with_its_size(self):
        p = np.zeros((30, 3))
        p[:, 0] = np.arange(30) * 0.002          # 4 cm/s a 20 Hz
        p[15:, 0] += 0.04                        # salto persistente de 4 cm
        jumps = step_discontinuities(p, 0.03, 6)
        self.assertEqual(len(jumps), 1)
        i, step, gap = jumps[0]
        self.assertEqual(i, 14)
        self.assertAlmostEqual(gap, 0.04, places=6)

    def test_fast_continuous_motion_has_no_discontinuity(self):
        p = np.zeros((30, 3))
        p[:, 1] = np.arange(30) * 0.035          # 0.7 m/s continuo: saltos > 3 cm pero sin discontinuidad
        jumps = step_discontinuities(p, 0.03, 6)
        self.assertTrue(len(jumps) > 0)
        self.assertTrue(all(g < 1e-9 for _, _, g in jumps))

    def test_frame_speeds(self):
        p = np.zeros((5, 3))
        p[:, 2] = np.arange(5) * 0.001
        np.testing.assert_allclose(frame_speeds(p, 20.0), 0.02)


if __name__ == "__main__":
    unittest.main()
