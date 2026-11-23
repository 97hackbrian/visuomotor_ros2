"""Pruebas de visuomotor_core.analysis.trajectory_metrics (ATE anclado, Umeyama, RPE, desfase temporal)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_core"))

from visuomotor_core.analysis.trajectory_metrics import (  # noqa: E402
    estimate_time_offset, evaluate_trajectory, interpolate_poses, matrix_to_quat, quat_to_matrix,
    relative_pose_errors, summarize_runs, umeyama)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _reference(n=200, rate=20.0):
    """Trayectoria de referencia tipo aproximación: avanza, baja y gira la muñeca."""
    t = np.arange(n) / rate
    p = np.stack([0.15 + 0.3 * t / t[-1], 0.05 * np.sin(t), 0.75 - 0.4 * t / t[-1]], axis=1)
    r = np.stack([_rz(0.3 * math.sin(0.5 * ti)) @ _rx(math.pi - 0.2 * ti / t[-1]) for ti in t])
    return t, p, matrix_to_quat(r)


class RotationTest(unittest.TestCase):
    def test_quat_matrix_round_trip(self):
        _, _, q = _reference(50)
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)

    def test_interpolation_hits_samples_and_flags_outside(self):
        t, p, q = _reference(50)
        pi, qi, ok = interpolate_poses(np.r_[t[10], t[-1] + 1.0], t, p, q)
        np.testing.assert_allclose(pi[0], p[10], atol=1e-12)
        self.assertEqual(ok.tolist(), [True, False])


class AlignmentTest(unittest.TestCase):
    def test_umeyama_recovers_sim3(self):
        _, p, _ = _reference()
        r_true = _rz(0.7) @ _rx(0.2)
        src = p
        dst = 1.05 * src @ r_true.T + np.array([0.1, -0.2, 0.3])
        s, r, t = umeyama(src, dst, with_scale=True)
        self.assertAlmostEqual(s, 1.05, places=9)
        np.testing.assert_allclose(r, r_true, atol=1e-9)

    def test_rigidly_moved_estimate_has_zero_anchored_error(self):
        t, p, q = _reference()
        g = _rz(1.1) @ _rx(-0.4)                       # marco de odometría arbitrario
        p_est = p @ g.T + np.array([1.0, 2.0, -0.5])
        r_est = np.einsum("ij,njk->nik", g, quat_to_matrix(q))
        res = evaluate_trajectory(t, p_est, matrix_to_quat(r_est), t, p, q)
        self.assertLess(res["ate_anchored_mm"]["max"], 1e-6)
        self.assertLess(res["rot_anchored_deg"]["max"], 1e-4)
        self.assertAlmostEqual(res["scale_error_pct"], 0.0, places=6)
        self.assertLess(res["rpe"]["0.5s"]["trans_mm"]["max"], 1e-6)

    def test_scale_error_is_measured_and_return_point_is_blind_to_it(self):
        t, p, q = _reference()
        p_loop = np.vstack([p, p[::-1]])               # ida y vuelta a la marca
        q_loop = np.vstack([q, q[::-1]])
        t_loop = np.arange(len(p_loop)) / 20.0
        p_est = p_loop[0] + 1.05 * (p_loop - p_loop[0])  # 5 % de error de escala
        res = evaluate_trajectory(t_loop, p_est, q_loop, t_loop, p_loop, q_loop)
        self.assertAlmostEqual(res["scale_error_pct"], 5.0, places=6)
        self.assertGreater(res["ate_anchored_mm"]["max"], 10.0)
        self.assertLess(res["final_error_anchored_mm"], 1e-6)   # el "retorno a la marca" no ve la escala

    def test_drift_shows_in_anchored_ate_and_rpe(self):
        t, p, q = _reference()
        p_est = p + np.outer(t, [0.004, 0.0, 0.0])     # deriva de 4 mm/s
        res = evaluate_trajectory(t, p_est, q, t, p, q)
        self.assertAlmostEqual(res["final_error_anchored_mm"], 4.0 * t[-1], places=6)
        self.assertAlmostEqual(res["rpe"]["1s"]["trans_mm"]["median"], 4.0, places=6)

    def test_lost_frames_are_counted_and_excluded(self):
        t, p, q = _reference()
        p_est = p.copy()
        lost = np.zeros(len(t), bool)
        lost[50:55] = True
        p_est[lost] = 0.0                               # pose nula
        res = evaluate_trajectory(t, p_est, q, t, p, q, lost=lost)
        self.assertAlmostEqual(res["lost_pct"], 2.5)
        self.assertLess(res["ate_anchored_mm"]["max"], 1e-6)


class RpeAndOffsetTest(unittest.TestCase):
    def test_rpe_zero_for_identical(self):
        _, p, q = _reference()
        r = quat_to_matrix(q)
        et, er = relative_pose_errors(p, r, p, r, 10)
        self.assertLess(et.max(), 1e-12)
        self.assertLess(er.max(), 1e-4)

    def test_time_offset_recovered(self):
        def f(t):
            return np.stack([np.sin(0.8 * t) + 0.3 * np.sin(2.1 * t), np.cos(0.5 * t), 0.1 * t], axis=1)
        t = np.arange(0, 20, 0.01)
        t_est = np.arange(0, 20, 1 / 30.0)                 # estimación a 30 Hz, referencia a 100 Hz
        for delay in (0.06, -0.05):
            d = estimate_time_offset(t_est, f(t_est - delay), t, f(t), max_offset_s=0.3)
            self.assertAlmostEqual(d, delay, delta=0.003)

    def test_summarize_runs_reports_distribution_not_best(self):
        runs = [{"ate_anchored_mm": {"rmse": v}} for v in (2.3, 14.4, 16.2, 16.9, 21.6, 28.3, 39.3, 59.0, 59.7)]
        s = summarize_runs(runs, keys=("ate_anchored_mm.rmse",))["ate_anchored_mm.rmse"]
        self.assertEqual(s["n"], 9)
        self.assertAlmostEqual(s["median"], 21.6)
        self.assertAlmostEqual(s["min"], 2.3)


if __name__ == "__main__":
    unittest.main()
