"""Pruebas del monitor de alcance del xArm5 para la recolección a mano (visuomotor_data_collection.reach_monitor)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in ("visuomotor_data_collection", "visuomotor_bringup", "tools/xarm5"):
    sys.path.insert(0, str(ROOT / p))

from scipy.spatial.transform import Rotation  # noqa: E402

from visuomotor_data_collection.reach_monitor import (ReachMonitor, advice, pose_matrix,  # noqa: E402
                                                      relative_transform, summarize)

CFG = (ROOT / "visuomotor_bringup/config/xarm5/xarm5_real_params.yaml",
       ROOT / "visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml",
       ROOT / "deploy/jetson/config/jetson_policy_params.yaml")


class ReachMonitorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ReachMonitor(*CFG)

    def _moved(self, d_base):
        """T_rel de la cámara (en su propio marco) para moverla ``d_base`` (m) en link_base sin girarla."""
        R_bc = (self.m.T_base_home @ self.m.T_tcp_cam)[:3, :3]
        T = np.eye(4)
        T[:3, 3] = R_bc.T @ np.asarray(d_base, dtype=np.float64)
        return T

    def test_home3_is_reachable_and_matches_the_converter(self):
        r = self.m.check(np.eye(4))
        self.assertFalse(r["outside"])
        np.testing.assert_allclose(r["tcp_xyz"], self.m.T_base_home[:3, 3], atol=1e-9)
        self.assertLess(r["ratio"], 0.95)
        from convert_vio_dataset_to_xarm5 import pose_matrix as pm, rpy_to_quat    # misma convención rpy
        import yaml
        al = yaml.safe_load(open(CFG[1]))["xarm5_dataset_alignment"]
        np.testing.assert_allclose(self.m.T_tcp_cam, pm(al["tcp_to_camera_xyz_m"], rpy_to_quat(*al["tcp_to_camera_rpy_rad"])),
                                   atol=1e-9)

    def test_forward_high_is_out_of_reach_but_down_then_forward_is_not(self):
        high = self.m.check(self._moved([0.42, 0.0, -0.05]))           # avanza alto (como episode_48)
        self.assertTrue(high["outside"])
        self.assertIn("brazo demasiado estirado", high["reasons"])
        self.assertEqual(advice(high["reasons"]), "baja primero y después avanza")
        low = self.m.check(self._moved([0.25, 0.0, -0.40]))            # baja primero
        self.assertFalse(low["outside"], low)

    def test_wrist_rotation_beyond_the_bridge_yaw_limit(self):
        """2026-11-26: con las hojas a 180° y la cámara alineada, el giro pedido (~180°) sale de ±2,6 rad del puente."""
        half = np.eye(4)
        half[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()   # media vuelta sobre su eje óptico (x de camera_link)
        # Con el puente a ±177° (2026-11-26) y 5° de tolerancia, alinear la cámara a 180° NO avisa (el robot queda 3°
        # corto); con el rango anterior (±2,6 rad = ±149°) sí
        self.assertNotIn("cámara demasiado girada", self.m.check(half)["reasons"])
        old = ReachMonitor(*CFG)
        old.yaw_limits = (-2.6, 2.6)
        r = old.check(half)
        self.assertIn("cámara demasiado girada", r["reasons"], r)
        self.assertTrue(advice(r["reasons"]).startswith("gira menos la cámara"), advice(r["reasons"]))
        quarter = np.eye(4)
        quarter[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        self.assertNotIn("cámara demasiado girada", self.m.check(quarter)["reasons"])

    def test_relative_transform_and_summary(self):
        q = Rotation.from_euler("z", 30, degrees=True).as_quat()
        T = relative_transform([0, 0, 0], [0, 0, 0, 1], [0.1, 0, 0], q)
        np.testing.assert_allclose(T, pose_matrix([0.1, 0, 0], Rotation.from_quat(q)), atol=1e-12)
        s = summarize({"frames": 10, "outside": 3, "max_ratio": 1.08})
        self.assertEqual((s["frames_outside"], s["fraction_outside"], s["max_reach_ratio"]), (3, 0.3, 1.08))


if __name__ == "__main__":
    unittest.main()
