"""Pruebas del armado del dataset v8 con poses de DROID (tools/xarm5/build_dataset_from_bags.py), sin ROS."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "xarm5"))
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

from build_dataset_from_bags import build_episode  # noqa: E402


class BuildEpisodeTest(unittest.TestCase):
    def _droid(self, init_noise_mm=0.0, depth=1.0):
        """DROID a 30 Hz: bag desde ARMING (0,35 s), cámara quieta hasta 1,0 s del bag y después se TRASLADA sin girar
        (como la mano en episode_77): el giróscopo diría 'quieta' todo el tiempo."""
        t = np.arange(0.0, 4.0, 1 / 30.0) + 1000.0
        p = np.zeros((len(t), 3))
        move = t - t[0] > 1.0
        p[move, 2] = 0.10 * (t[move] - t[0] - 1.0) * depth                # avanza por el eje óptico a 10 cm/s
        rng = np.random.default_rng(0)
        early = t - t[0] < 0.45
        p[early] += rng.normal(0, init_noise_mm / 1000.0, (early.sum(), 3))  # inicialización de DROID
        q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
        ts = np.arange(t[0] + 0.35, t[-1], 0.05)                          # demo a 20 Hz desde el fin de ARMING
        return t, p, q, ts

    def test_translation_without_rotation_is_kept(self):
        t, p, q, ts = self._droid(init_noise_mm=20.0)
        xyz, quat, info = build_episode(ts, t, p, q, np.eye(4), 1.0, 0.5, 0.4)
        self.assertLess(info["reference_spread_mm"], 1.0)                  # la referencia, en el tramo quieto
        self.assertLessEqual(info["frames_pinned_to_start"], 4)            # solo la inicialización de DROID (<= 0,5 s)
        # A los 3 s del bag la cámara avanzó 20 cm: se conserva (la versión 1 borraba el movimiento "quieto" según el
        # giróscopo)
        i = int(np.argmin(np.abs(ts - (t[0] + 3.0))))
        self.assertAlmostEqual(float(np.linalg.norm(xyz[i])), 0.20, delta=0.005)
        np.testing.assert_allclose(xyz[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(quat[:, 3], 1.0, atol=1e-9)

    def test_depth_scale_and_reference_frame(self):
        t, p, q, ts = self._droid()
        xyz, _q, _info = build_episode(ts, t, p, q, np.eye(4), 0.986, 0.5, 0.4)
        self.assertAlmostEqual(float(np.linalg.norm(xyz[-1])), 0.986 * float(np.linalg.norm(p[-1])), delta=0.003)
        # T_link_opt no trivial: el avance por el eje óptico (z) aparece en el eje de camera_link correspondiente
        T_link_opt = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
        xyz2, _q2, _i2 = build_episode(ts, t, p, q, T_link_opt, 1.0, 0.5, 0.4)
        self.assertGreater(abs(xyz2[-1, 0]), 0.25)                         # x de camera_link = z óptico
        self.assertLess(abs(xyz2[-1, 1]) + abs(xyz2[-1, 2]), 1e-6)


if __name__ == "__main__":
    unittest.main()
