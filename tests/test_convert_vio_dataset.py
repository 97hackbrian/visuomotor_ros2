"""Prueba de la conversión VIO -> xArm5 (tools/xarm5/convert_vio_dataset_to_xarm5.py) sin zarr ni ROS.

Genera una trayectoria del TCP conocida y alcanzable en link_base, simula lo que grabaría vo_collector_node (pose de la
cámara en un marco 'odom' arbitrario, posición relativa al inicio, qw >= 0) y verifica que la conversión la recupera.

Ejecutar:  python3 -m pytest tests/test_convert_vio_dataset.py -q
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_bringup"))
sys.path.insert(0, str(ROOT / "tools" / "xarm5"))

from convert_vio_dataset_to_xarm5 import Converter, inv, pose_matrix  # noqa: E402
from visuomotor_bringup.xarm5_interface import (  # noqa: E402
    arm_azimuth, arm_plane_quat, matrix_to_quat, normalize_quat, rotation_angle, rpy_to_quat)

BRIDGE = {
    "orientation_mode": "arm_plane", "top_down_roll": math.pi, "top_down_pitch": 0.0, "fixed_yaw": 0.0,
    "arm_plane_pitch_min": -0.5, "arm_plane_pitch_max": 0.5, "yaw_min": -2.6, "yaw_max": 2.6,
    "workspace_x_min": 0.15, "workspace_x_max": 0.65, "workspace_y_min": -0.30, "workspace_y_max": 0.30,
    "workspace_z_min": 0.10, "workspace_z_max": 0.75, "workspace_min_radius_xy": 0.15, "workspace_max_radius_xy": 0.65,
    "workspace_shoulder_xyz_m": [0.0, 0.0, 0.267], "workspace_wrist_offset_m": [-0.076, 0.0, -0.097],
    "workspace_max_reach_m": 0.6086, "workspace_min_reach_m": 0.12,
    "max_linear_speed_m_s": 0.05, "max_angular_speed_rad_s": 0.5,
}
POLICY = {"home_x": 0.2221, "home_y": -0.004, "home_z": 0.116, "home_roll": math.pi, "home_pitch": 0.0,
          "home_yaw": 0.0, "relative_coordinates": True}
ALIGN = {"tcp_to_camera_xyz_m": [0.05, -0.02, 0.03], "tcp_to_camera_rpy_rad": [0.1, -1.4, 0.2],
         "extrinsic_calibrated": True, "max_orientation_deviation_rad": 0.15, "max_workspace_violation_frames": 0,
         "speed_tolerance_ratio": 1.0, "speed_window_steps": 4}


def tcp_trajectory(n=120, tilt=0.3, out_of_plane=0.0):
    """TCP alcanzable: desde HOME avanza en X/Y/Z, se inclina en el plano del brazo y gira la herramienta."""
    home = np.array([POLICY["home_x"], POLICY["home_y"], POLICY["home_z"]])
    poses = []
    for i in range(n):
        s = i / (n - 1)
        xyz = home + s * np.array([0.10, 0.06, 0.08])
        q = arm_plane_quat(arm_azimuth(xyz), tilt * s, 0.4 * s)
        if out_of_plane:  # roll lateral (inalcanzable) sobre el eje radial del brazo
            a = arm_azimuth(xyz)
            axis = np.array([math.cos(a), math.sin(a), 0.0])
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            r = np.eye(3) + math.sin(out_of_plane * s) * k + (1 - math.cos(out_of_plane * s)) * k @ k
            q = matrix_to_quat(r @ pose_matrix([0, 0, 0], q)[:3, :3])
        poses.append(pose_matrix(xyz, q))
    return poses


def record_like_vo_collector(tcp_poses, t_tcp_cam, t_odom_base):
    """Lo que guarda vo_collector_node: posición de la cámara relativa al inicio (en odom) + cuaternión absoluto."""
    cams = [t_odom_base @ t @ t_tcp_cam for t in tcp_poses]
    p0 = cams[0][:3, 3]
    rows = []
    for c in cams:
        q = normalize_quat(matrix_to_quat(c[:3, :3]))
        rows.append([*(c[:3, 3] - p0), *q, 0.0])
    state = np.array(rows, dtype=np.float32)
    action = np.concatenate([state, np.zeros((len(state), 1), np.float32)], axis=1)
    action[-40:, 8] = 1.0
    return state, action


class TestConvertVioDataset(unittest.TestCase):
    def setUp(self):
        self.conv = Converter(BRIDGE, ALIGN, POLICY)
        self.t_tcp_cam = pose_matrix(ALIGN["tcp_to_camera_xyz_m"], rpy_to_quat(*ALIGN["tcp_to_camera_rpy_rad"]))
        self.t_odom_base = pose_matrix([1.3, -0.7, 0.4], rpy_to_quat(0.5, -0.3, 2.1))  # marco odom arbitrario

    def test_recovers_reachable_tcp_trajectory(self):
        tcp = tcp_trajectory()
        state, action = record_like_vo_collector(tcp, self.t_tcp_cam, self.t_odom_base)
        out, stats = self.conv.convert_episode(state, action, 20.0)
        home = np.array([POLICY["home_x"], POLICY["home_y"], POLICY["home_z"]])
        for i, t in enumerate(tcp):
            np.testing.assert_allclose(out["state"][i, :3] + home, t[:3, 3], atol=2e-5)
            self.assertLess(rotation_angle(out["state"][i, 3:7], matrix_to_quat(t[:3, :3])), 1e-4)
        np.testing.assert_allclose(out["state"][0, :3], 0.0, atol=1e-6)    # el episodio empieza en HOME (Delta-Home)
        self.assertTrue(np.all(out["state"][:, 6] >= 0.0))                 # qw >= 0
        np.testing.assert_array_equal(out["action"][:, 7:], action[:, 7:])  # gripper y done intactos
        self.assertLess(stats["max_orientation_deviation_rad"], 1e-4)
        self.assertEqual(stats["flags"], [])

    def test_unreachable_tilt_is_projected_and_flagged(self):
        tcp = tcp_trajectory(out_of_plane=0.4)
        state, action = record_like_vo_collector(tcp, self.t_tcp_cam, self.t_odom_base)
        out, stats = self.conv.convert_episode(state, action, 20.0)
        # roll lateral de 0.4 rad: la orientación alcanzable más cercana queda a <= 0.4 rad (la proyección puede
        # compensar parte del roll con inclinación/giro), pero claramente sobre el umbral de 0.15 rad
        self.assertTrue(0.3 < stats["max_orientation_deviation_rad"] <= 0.4 + 1e-6, stats["max_orientation_deviation_rad"])
        self.assertTrue(any("alcanzable" in f for f in stats["flags"]))
        # la acción convertida es alcanzable: su eje de herramienta está en el plano del brazo
        home = np.array([POLICY["home_x"], POLICY["home_y"], POLICY["home_z"]])
        for row in out["action"]:
            a = arm_azimuth(row[:3] + home)
            z_tool = pose_matrix([0, 0, 0], row[3:7])[:3, 2]
            self.assertAlmostEqual(float(np.dot(z_tool, [-math.sin(a), math.cos(a), 0.0])), 0.0, places=5)

    def test_fast_demonstration_is_flagged(self):
        tcp = tcp_trajectory(n=20)  # 0.14 m en 1 s >> 0.05 m/s del puente
        state, action = record_like_vo_collector(tcp, self.t_tcp_cam, self.t_odom_base)
        _, stats = self.conv.convert_episode(state, action, 20.0)
        self.assertTrue(any("velocidad lineal" in f for f in stats["flags"]))

    def test_out_of_reach_is_flagged(self):
        # sube a z = 0.70 con x = 0.55: dentro de la caja pero fuera del alcance hombro-muñeca
        home = np.array([POLICY["home_x"], POLICY["home_y"], POLICY["home_z"]])
        tcp = [pose_matrix(home + s * (np.array([0.55, 0.0, 0.70]) - home), arm_plane_quat(0.0, 0.0, 0.0))
               for s in np.linspace(0, 1, 400)]
        state, action = record_like_vo_collector(tcp, self.t_tcp_cam, self.t_odom_base)
        _, stats = self.conv.convert_episode(state, action, 20.0)
        self.assertGreater(stats["workspace_violation_frames"], 0)
        self.assertTrue(any("alcance" in f for f in stats["flags"]))

    def test_inverse(self):
        t = pose_matrix([0.1, 0.2, 0.3], rpy_to_quat(0.3, 0.2, 0.1))
        np.testing.assert_allclose(inv(t) @ t, np.eye(4), atol=1e-12)



class YawNearPiTest(unittest.TestCase):
    """2026-11-26: con las hojas a 180° y la cámara alineada, el giro pedido oscila alrededor de ±180°. Proyectando
    cada frame por separado, el yaw saltaba de un límite al otro (velocidad angular 5,4 rad/s marcada en 10 demos).
    Recordando el yaw del frame anterior (como el puente con su yaw comandado) queda de un solo lado."""

    def test_jitter_around_pi_does_not_flip(self):
        conv = Converter(dict(BRIDGE, yaw_min=-3.0892, yaw_max=3.0892), ALIGN, POLICY)
        xyz = np.array([0.40, 0.0, 0.30])
        az = arm_azimuth(xyz)
        reqs = [arm_plane_quat(az, 0.0, math.radians(s * 179.5)) for s in (1, -1) * 10]
        prev, yaws, quats = None, [], []
        for q in reqs:
            qp, prev = conv.project(xyz, q, prev)
            yaws.append(prev)
            quats.append(qp)
        self.assertEqual(len({round(y, 6) for y in yaws}), 1, yaws)                  # siempre el mismo límite
        self.assertLess(max(rotation_angle(quats[i], quats[i + 1]) for i in range(len(quats) - 1)), 1e-6)
        self.assertLess(rotation_angle(reqs[0], quats[0]), math.radians(3.1))       # a <= 3° de lo pedido
        old = [conv.project(xyz, q)[1] for q in reqs]                                  # sin memoria: salta
        self.assertEqual(len({round(y, 6) for y in old}), 2)


class DiscardedEpisodeTest(unittest.TestCase):
    """Las demos marcadas ``discarded`` al grabar ([U] del recolector o a pedido del usuario, 2026-11-25) no se
    convierten y el informe lo dice; las demás sí."""

    def test_discarded_episode_is_skipped(self):
        import json
        import tempfile
        import zarr
        from convert_vio_dataset_to_xarm5 import main
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "src.zarr"
        root = zarr.open_group(str(src), mode="w")
        n = 60
        for k, discarded in ((0, False), (1, True)):
            ep = root.create_group(f"episode_{k}")
            state = np.zeros((n, 8), np.float32)
            state[:, 2] = np.linspace(0.0, 0.10, n)                 # la cámara avanza 10 cm por su eje óptico
            state[:, 6] = 1.0
            action = np.zeros((n, 9), np.float32)
            action[:, :8] = state
            action[-20:, 8] = 1.0
            ep.create_dataset("state", data=state)
            ep.create_dataset("action", data=action)
            ep.create_dataset("image", data=np.zeros((n, 32, 32, 3), np.uint8))
            ep.attrs.update({"sampling_rate_hz": 20.0, "relative_coordinates": True})
            if discarded:
                ep.attrs.update({"discarded": True, "discard_reason": "prueba"})
        dst = tmp / "dst.zarr"
        self.assertEqual(main(["--src", str(src), "--dst", str(dst)]), 0)
        out = zarr.open_group(str(dst), mode="r")
        self.assertEqual(sorted(out.group_keys()), ["episode_0"])
        rep = json.loads((tmp / "dst_conversion_report.json").read_text())
        self.assertEqual(rep["episodes"]["episode_1"], {"skipped": "discarded", "reason": "prueba"})


if __name__ == "__main__":
    unittest.main()
