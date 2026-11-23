"""Pruebas de la calibración cámara–brida (tools/xarm5/calibrate_hand_eye.py) con muestras sintéticas."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "xarm5"))

from calibrate_hand_eye import (  # noqa: E402
    board_in_base, consistency, inv_t, main, rot_angle_deg, solve_hand_eye, to_t)


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


# Verdad: cámara óptica en el TCP (eje óptico ~ a lo largo de la herramienta, desplazada 6 cm) y tablero en la mesa.
X_TRUE = to_t(_rot([0, 0, 1], math.pi / 2) @ _rot([1, 0, 0], 0.05), [0.06, -0.01, 0.035])
BOARD_TRUE = to_t(_rot([0, 0, 1], 0.3), [0.45, -0.05, 0.02])


def _samples(n=20, noise_mm=0.3, noise_deg=0.05, seed=0):
    rng = np.random.default_rng(seed)
    base_tcp, cam_board = [], []
    for _ in range(n):
        # garra hacia abajo con giro de base (J1), inclinación en el plano del brazo y giro de muñeca (J5)
        r = (_rot([0, 0, 1], rng.uniform(-0.6, 0.6)) @ _rot([0, 1, 0], rng.uniform(-0.5, 0.2))
             @ _rot([1, 0, 0], math.pi) @ _rot([0, 0, 1], rng.uniform(-0.8, 0.8)))
        p = np.array([rng.uniform(0.3, 0.5), rng.uniform(-0.15, 0.1), rng.uniform(0.3, 0.5)])
        b = to_t(r, p)
        c = inv_t(X_TRUE) @ inv_t(b) @ BOARD_TRUE
        c = to_t(_rot(rng.normal(size=3), math.radians(noise_deg)) @ c[:3, :3],
                 c[:3, 3] + rng.normal(scale=noise_mm / 1000 / math.sqrt(3), size=3))
        base_tcp.append(b)
        cam_board.append(c)
    return base_tcp, cam_board


class HandEyeTest(unittest.TestCase):
    def test_recovers_extrinsic_and_board(self):
        base_tcp, cam_board = _samples()
        res = solve_hand_eye(base_tcp, cam_board, methods=("PARK", "TSAI"))
        for name, v in res.items():
            x = v["tcp_to_camera"]
            self.assertLess(np.linalg.norm(x[:3, 3] - X_TRUE[:3, 3]) * 1000, 2.0, name)
            self.assertLess(rot_angle_deg(X_TRUE[:3, :3].T @ x[:3, :3]), 0.3, name)
            self.assertLess(v["residual"]["pos_median_mm"], 1.5, name)

    def test_wrong_extrinsic_shows_large_residual(self):
        base_tcp, cam_board = _samples(noise_mm=0.0, noise_deg=0.0)
        x_bad = to_t(X_TRUE[:3, :3], X_TRUE[:3, 3] + [0.0, 0.0, -0.035])      # traslación supuesta nula en z
        self.assertGreater(consistency(board_in_base(base_tcp, x_bad, cam_board))["pos_median_mm"], 5.0)

    def test_solve_command_end_to_end(self):
        base_tcp, cam_board = _samples()
        link_opt = to_t(_rot([1, 0, 0], -math.pi / 2) @ _rot([0, 1, 0], math.pi / 2), [0.0, 0.015, 0.0])
        with tempfile.TemporaryDirectory() as d:
            samples = [{"T_base_tcp": b.tolist(), "T_cam_board": c.tolist(), "home3": False}
                       for b, c in zip(base_tcp, cam_board)]
            Path(d, "samples.json").write_text(json.dumps({"samples": samples,
                                                           "T_cameralink_optical": link_opt.tolist()}))
            self.assertEqual(main(["solve", "--samples", d]), 2)               # tablero sin medir: se niega
            self.assertEqual(main(["solve", "--samples", d, "--allow-nominal"]), 0)
            out = json.loads(Path(d, "hand_eye_result.json").read_text())
            self.assertTrue(out["acceptance"]["pass"])
            x_link = np.asarray(out["tcp_to_camera_link"]["xyz_m"])
            expected = (X_TRUE @ inv_t(link_opt))[:3, 3]
            self.assertLess(np.linalg.norm(x_link - expected) * 1000, 2.0)
            np.testing.assert_allclose(out["base_to_board"]["xyz_m"], BOARD_TRUE[:3, 3], atol=0.002)


if __name__ == "__main__":
    unittest.main()
