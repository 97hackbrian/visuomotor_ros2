"""Pruebas de la lógica pura del recolector v2 (visuomotor_data_collection/vo_episode_logic.py)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_episode_logic import (  # noqa: E402
    imu_moving_fraction,
    EpisodeSampler, PoseBuffer, PoseSample, StampedBuffer, check_done_segment, continuous_quat, detect_hold_start,
    camera_still, episode_quality, imu_still_since, is_null_odometry)

Q_DOWN = np.array([0.0, 0.7071068, 0.0, 0.7071068])


def _feed(n=90, rate=30.0, jitter=0.002, seed=0, null_at=(), v=np.array([0.05, 0.0, -0.02])):
    """Cámara a 30 Hz con jitter: imágenes y odometría con el MISMO sello (como RTAB-Map)."""
    rng = np.random.default_rng(seed)
    poses, images = PoseBuffer(), StampedBuffer(maxlen=200)
    t0 = 100.0
    stamps = t0 + np.arange(n) / rate + rng.uniform(-jitter, jitter, n)
    for i, s in enumerate(stamps):
        images.add(float(s), np.full((4, 4, 3), i % 256, np.uint8))
        if i in null_at:
            prev = poses.latest()
            poses.add(PoseSample(float(s), prev.p.copy(), prev.q.copy(), lost=True))
        else:
            poses.add(PoseSample(float(s), v * (s - t0), Q_DOWN.copy(), inliers=200))
    return poses, images, stamps


class OdometryTest(unittest.TestCase):
    def test_null_odometry_detection(self):
        self.assertTrue(is_null_odometry([0, 0, 0, 0]))
        self.assertTrue(is_null_odometry([0, 0, 0, 1], covariance_diag=[9999.0] * 6))
        self.assertFalse(is_null_odometry([0, 0, 0, 1], covariance_diag=[1e-4] * 6))

    def test_continuous_quat_keeps_sign(self):
        q = continuous_quat([0.0, -0.7071, 0.0, -0.7071], Q_DOWN)
        self.assertGreater(float(np.dot(q, Q_DOWN)), 0)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0)


class SamplerTest(unittest.TestCase):
    def test_regular_grid_with_paired_images(self):
        poses, images, stamps = _feed()
        s = EpisodeSampler(rate_hz=20.0, max_image_dt_s=0.02, max_pose_gap_s=0.1)
        s.start(float(stamps[0]))
        s.step(poses, images, gripper=0.0, hold=False)
        t = np.array([f.timestamp for f in s.frames])
        self.assertGreater(len(t), 55)
        np.testing.assert_allclose(np.diff(t), 0.05, atol=1e-9)
        img_dt = np.abs(np.array([f.image_stamp for f in s.frames]) - t)
        self.assertLessEqual(img_dt.max(), 0.02)
        q = episode_quality(s.frames, 20.0)
        self.assertEqual(q["duplicate_images"], 0)
        self.assertEqual(q["lost_frames"], 0)
        # posición interpolada = movimiento real en el instante del frame
        for f in s.frames:
            np.testing.assert_allclose(f.p, np.array([0.05, 0.0, -0.02]) * (f.timestamp - 100.0), atol=1e-9)

    def test_null_pose_marks_frames_lost_and_never_becomes_identity(self):
        poses, images, stamps = _feed(null_at=(40, 41))
        s = EpisodeSampler(rate_hz=20.0, max_image_dt_s=0.02, max_pose_gap_s=0.1)
        s.start(float(stamps[0]))
        s.step(poses, images, gripper=0.0, hold=False)
        lost = [f for f in s.frames if f.lost]
        self.assertTrue(1 <= len(lost) <= 4)
        for f in s.frames:
            self.assertGreater(abs(float(np.dot(f.q, Q_DOWN))), 0.99)           # nunca la identidad
            self.assertLess(np.linalg.norm(f.p), 0.2)                          # nunca -pose inicial

    def test_waits_for_late_image(self):
        poses, images, stamps = _feed(n=30)
        last = float(stamps[-1])
        s = EpisodeSampler(rate_hz=20.0, max_image_dt_s=0.02, max_pose_gap_s=0.1)
        s.start(last + 0.025)                    # instante sin imagen a <= 20 ms todavía
        poses.add(PoseSample(last + 0.033, np.zeros(3), Q_DOWN.copy(), inliers=150))
        self.assertEqual(s.step(poses, images, 0.0, False, image_wait_s=0.05), 0)   # espera la imagen
        images.add(last + 0.033, np.zeros((4, 4, 3), np.uint8))                   # llega la imagen de esa pose
        self.assertEqual(s.step(poses, images, 0.0, False, image_wait_s=0.05), 1)
        self.assertFalse(s.frames[-1].lost)

    def test_missing_image_is_marked_lost(self):
        poses, images, stamps = _feed(n=30)
        last = float(stamps[-1])
        s = EpisodeSampler(rate_hz=20.0, max_image_dt_s=0.02, max_pose_gap_s=0.1)
        s.start(last + 0.025)
        poses.add(PoseSample(last + 0.033, np.zeros(3), Q_DOWN.copy(), inliers=150))
        images.add(last + 0.2, np.zeros((4, 4, 3), np.uint8))                     # solo llega una muy posterior
        self.assertEqual(s.step(poses, images, 0.0, False, image_wait_s=0.05), 1)
        self.assertTrue(s.frames[-1].lost)


class DoneTest(unittest.TestCase):
    def _traj(self, moving_s=3.0, hold_s=2.5, rate=20.0, shake=0.0):
        n_m, n_h = int(moving_s * rate), int(hold_s * rate)
        p = np.zeros((n_m + n_h, 3))
        p[:n_m, 0] = np.linspace(0, 0.15, n_m)
        p[n_m:, 0] = 0.15
        if shake:
            p[n_m:, 1] = shake * np.sin(np.arange(n_h) * 2 * math.pi * 2 / rate)
        hold = np.r_[np.zeros(n_m, bool), np.ones(n_h, bool)]
        return p, hold

    def test_correct_hold(self):
        p, hold = self._traj()
        r = check_done_segment(p, hold, 20.0, 2.0, 0.02, 0.5)
        self.assertEqual(r["problems"], [])
        self.assertEqual(int(r["done"].sum()), 50)
        self.assertEqual(detect_hold_start(p, 20.0, 0.02, 0.5) in range(55, 66), True)

    def test_short_or_moving_hold_is_reported(self):
        p, hold = self._traj(hold_s=1.0)
        self.assertTrue(any("HOLD de" in x for x in check_done_segment(p, hold, 20.0, 2.0, 0.02, 0.5)["problems"]))
        p, hold = self._traj(shake=0.02)
        self.assertTrue(any("se mueven" in x for x in check_done_segment(p, hold, 20.0, 2.0, 0.02, 0.5)["problems"]))

    def test_no_hold_is_reported(self):
        p, hold = self._traj()
        self.assertTrue(check_done_segment(p, np.zeros(len(hold), bool), 20.0, 2.0, 0.02, 0.5)["problems"])

    def test_hand_spikes_tolerated_and_consistent_with_detection(self):
        """2026-11-25: 1-3 frames de ~50 sobre 2 cm/s rechazaban demos a mano. Con max_moving_frac 0,2 se toleran
        picos breves, pero no un tramo que de verdad se mueve; y el tramo que detecta detect_hold_start siempre pasa."""
        p, hold = self._traj(hold_s=2.6)
        rng = np.random.default_rng(3)
        p[60:, :] += rng.normal(0, 0.0006, (len(p) - 60, 3))            # ruido de la odometría en reposo
        p[90:93, 1] += 0.015                                          # un tirón de la mano de 1,5 cm y 0,15 s
        hold_det = np.zeros(len(p), bool)
        i0 = detect_hold_start(p, 20.0, 0.02, 0.5)
        self.assertIsNotNone(i0)
        hold_det[i0:] = True
        strict = check_done_segment(p, hold, 20.0, 2.0, 0.02, 0.5, 0.0)
        self.assertTrue(any("se mueven" in x for x in strict["problems"]))
        tol = check_done_segment(p, hold, 20.0, 2.0, 0.02, 0.5, 0.2)
        self.assertEqual(tol["problems"], [], tol)
        p2, hold2 = self._traj()                                       # sigue derivando a 3 cm/s: se rechaza igual
        p2[hold2, 1] = 0.03 * np.arange(hold2.sum()) / 20.0
        self.assertTrue(check_done_segment(p2, hold2, 20.0, 2.0, 0.02, 0.5, 0.2)["problems"])
        p3, _ = self._traj()
        h3 = np.zeros(len(p3), bool)
        h3[detect_hold_start(p3, 20.0, 0.02, 0.5):] = True
        self.assertEqual(check_done_segment(p3, h3, 20.0, 2.0, 0.02, 0.5, 0.0)["moving_frames"], 0)


class ImuStillnessTest(unittest.TestCase):
    """Quietud final por giróscopo: decide el guardado cuando la odometría visual se perdió al final."""

    def test_still_since_after_motion(self):
        t = np.arange(0, 4, 0.005)                                   # 200 Hz
        g = np.zeros((len(t), 3))
        g[t < 2.5, 2] = 0.4                                          # gira hasta 2,5 s, después quieta
        rng = np.random.default_rng(0)
        g += rng.normal(0, 0.01, g.shape)                            # ruido / temblor leve
        s = imu_still_since(t, g, 0.08)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, 2.5, delta=0.15)

    def test_moving_at_the_end_is_none(self):
        t = np.arange(0, 2, 0.005)
        g = np.zeros((len(t), 3))
        g[t > 1.5, 0] = 0.3
        self.assertIsNone(imu_still_since(t, g, 0.08))

    def test_always_still(self):
        t = np.arange(0, 1, 0.005)
        self.assertEqual(imu_still_since(t, np.zeros((len(t), 3)), 0.08), t[0])



class ImuMovingFractionTest(unittest.TestCase):
    def test_fraction_of_turning_frames(self):
        t = np.arange(0.0, 3.0, 0.005)                                # IMU a 200 Hz
        w = np.zeros((len(t), 3))
        w[(t > 1.0) & (t < 1.5), 2] = 0.6                             # gira medio segundo
        frames = np.arange(0.5, 2.5, 0.05)                            # 2 s de frames a 20 Hz
        frac = imu_moving_fraction(frames, t, w, 0.08, 0.15)
        self.assertAlmostEqual(frac, (0.5 + 0.15) / 2.0, delta=0.03)          # el suavizado alarga el giro 0,15 s
        self.assertEqual(imu_moving_fraction(frames, t, np.zeros_like(w), 0.08), 0.0)
        self.assertIsNone(imu_moving_fraction(frames + 5.0, t, w, 0.08))       # el IMU no cubre esos frames


class CameraStillTest(unittest.TestCase):
    """2026-11-26: el aviso de alcance se calla con la cámara quieta (quietud final), con el mismo criterio que el
    guardado automático: desplazamiento en la ventana / duración < velocidad estática."""

    def test_still_moving_and_short(self):
        rate, w, v = 20.0, 0.5, 0.02
        still = [np.array([0.40, 0.0, 0.30]) + 0.0005 * np.sin(i) for i in range(20)]   # ruido de 0,5 mm
        moving = [np.array([0.40 + 0.003 * i, 0.0, 0.30]) for i in range(20)]           # 6 cm/s
        self.assertTrue(camera_still(still, rate, w, v))
        self.assertFalse(camera_still(moving, rate, w, v))
        self.assertFalse(camera_still(still[:5], rate, w, v))                           # menos que la ventana
        self.assertTrue(camera_still(moving + [moving[-1]] * 11, rate, w, v))            # llegó y se quedó quieta


if __name__ == "__main__":
    unittest.main()
