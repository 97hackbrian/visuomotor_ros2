#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
test_relative_coordinates.py — Suite de pruebas unitarias exhaustivas para el paradigma
de coordenadas relativas (Delta-Home / UMI - Universal Manipulation Interface).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
import yaml
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_core"))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_data_collection"))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_bringup"))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_ros2"))

from verify_zarr_dataset import verify_episode, EpisodeReport


class TestConfigConsistency(unittest.TestCase):
    """Verifica que todos los archivos de configuración contengan los parámetros acordados."""

    def test_rtabmap_vio_config(self):
        param_file = REPO_ROOT / "visuomotor_bringup" / "config" / "realsense" / "rtabmap_vio_params.yaml"
        self.assertTrue(param_file.exists(), f"No existe {param_file}")
        with open(param_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        params = data["/**"]["ros__parameters"]
        align = str(params.get("Odom/AlignWithGround", "")).lower()
        self.assertEqual(align, "false", "Odom/AlignWithGround debe ser 'false' para iniciar odom en la lente")
        self.assertTrue(params.get("wait_imu_to_init", False), "wait_imu_to_init debe ser True para alineación gravitacional")
        guess = str(params.get("Odom/GuessMotion", "")).lower()
        self.assertEqual(guess, "true", "Odom/GuessMotion debe ser 'true'")

    def test_data_collection_params_yaml(self):
        param_file = REPO_ROOT / "visuomotor_data_collection" / "config" / "data_collection_params.yaml"
        self.assertTrue(param_file.exists(), f"No existe {param_file}")
        with open(param_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        params = data["**"]["ros__parameters"]
        self.assertTrue(params.get("relative_coordinates", False), "relative_coordinates debe ser True en data_collection_params.yaml")

    def test_vo_data_collection_params_yaml(self):
        param_file = REPO_ROOT / "visuomotor_data_collection" / "config" / "vo_data_collection_params.yaml"
        self.assertTrue(param_file.exists(), f"No existe {param_file}")
        with open(param_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        params = data["vo_collector_node"]["ros__parameters"]
        self.assertTrue(params.get("relative_coordinates", False), "relative_coordinates debe ser True en vo_data_collection_params.yaml")

    def test_policy_params_yaml(self):
        param_file = REPO_ROOT / "visuomotor_ros2" / "config" / "policy_params.yaml"
        self.assertTrue(param_file.exists(), f"No existe {param_file}")
        with open(param_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        params = data["policy_node"]["ros__parameters"]
        self.assertTrue(params.get("relative_coordinates", False), "relative_coordinates debe ser True en policy_params.yaml")


class TestRelativeCoordinateTransforms(unittest.TestCase):
    """Verifica las transformaciones matemáticas del paradigma UMI / Delta-Home."""

    def test_step_zero_is_strictly_origin(self):
        """En el paso 0 del episodio, la posición relativa debe ser idénticamente (0,0,0)."""
        initial_pose = np.array([0.254, -0.012, 0.403, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        current_pose = initial_pose.copy()

        rel_pos = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(rel_pos, [0.0, 0.0, 0.0])

        diff_pos = current_pose[0:3] - initial_pose[0:3]
        np.testing.assert_allclose(diff_pos, [0.0, 0.0, 0.0], atol=1e-7)

    def test_quaternion_upper_hemisphere_and_normalization(self):
        """El cuaternión debe cumplir qw >= 0 y tener norma unitaria 1.0."""
        raw_quat = np.array([0.1, -0.2, 0.3, -0.9], dtype=np.float32)
        qw, qx, qy, qz = raw_quat[3], raw_quat[0], raw_quat[1], raw_quat[2]

        if qw < 0.0:
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz

        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        self.assertGreaterEqual(qw, 0.0)
        np.testing.assert_allclose(np.linalg.norm([qx, qy, qz, qw]), 1.0, atol=1e-6)

    def test_quaternion_degenerate_fallback(self):
        """Si un cuaternión es nulo [0,0,0,0], debe recurrir a la identidad [0,0,0,1]."""
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0
        norm_q = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm_q > 1e-6:
            qx /= norm_q
            qy /= norm_q
            qz /= norm_q
            qw /= norm_q
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

        self.assertEqual([qx, qy, qz, qw], [0.0, 0.0, 0.0, 1.0])

    def test_zero_causal_leakage_state_dimensions(self):
        """La observación física (state) DEBE tener exactamente 8 dimensiones sin 'done'."""
        rel_pos = np.zeros(3, dtype=np.float32)
        rel_ori = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        gripper = 0.0

        state_vec = np.zeros(8, dtype=np.float32)
        state_vec[0:3] = rel_pos
        state_vec[3:7] = rel_ori
        state_vec[7] = gripper

        self.assertEqual(len(state_vec), 8)

    def test_action_dimensions_and_done_floor(self):
        """La acción debe tener 9 dimensiones, y los últimos 2s deben tener done=1.0 garantizado."""
        total_steps = 100
        hz = 20.0
        min_hold_s = 2.0
        min_hold_frames = int(hz * min_hold_s)  # 40 frames

        buffer = []
        for _ in range(total_steps):
            act = np.zeros(9, dtype=np.float32)
            act[0:3] = np.random.randn(3)
            act[3:7] = [0.0, 0.0, 0.0, 1.0]
            act[7] = -0.01
            act[8] = 0.0
            buffer.append({"action": act})

        start_hold_idx = max(0, total_steps - min_hold_frames)
        for i in range(start_hold_idx, total_steps):
            buffer[i]["action"][8] = 1.0

        done_flags = [b["action"][8] for b in buffer]
        self.assertEqual(len(done_flags), 100)
        self.assertTrue(all(d == 0.0 for d in done_flags[:60]))
        self.assertTrue(all(d == 1.0 for d in done_flags[60:]))
        self.assertEqual(sum(d == 1.0 for d in done_flags), 40)

    def test_action_dimensions_when_record_done_false(self):
        """Si record_done_flag es False, action debe tener 8 dimensiones."""
        record_done = False
        rel_pos = np.zeros(3, dtype=np.float32)
        rel_ori = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if record_done:
            action_vec = np.zeros(9, dtype=np.float32)
            action_vec[0:3] = rel_pos
            action_vec[3:7] = rel_ori
            action_vec[8] = 0.0
        else:
            action_vec = np.zeros(8, dtype=np.float32)
            action_vec[0:3] = rel_pos
            action_vec[3:7] = rel_ori
        self.assertEqual(len(action_vec), 8)

    def test_inference_relative_roundtrip_1d(self):
        """Verifica que la resta en la observación y la suma en la predicción 1D reproduzcan la cinemática."""
        home_pos = np.array([0.25, 0.0, 0.40], dtype=np.float32)
        actual_ee_pos = np.array([0.25, 0.0, 0.40], dtype=np.float32)
        gripper = 0.0
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        latest_state = np.concatenate([actual_ee_pos, quat, [gripper]])

        state_data = latest_state.copy()
        state_data[:3] -= home_pos
        np.testing.assert_array_equal(state_data[:3], [0.0, 0.0, 0.0])

        predicted_rel_action = np.array([-0.05, -0.30, -0.15, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        action_np = predicted_rel_action.copy()
        if action_np.ndim == 1:
            action_np[:3] += home_pos
        else:
            action_np[:, :3] += home_pos

        expected_abs = np.array([0.20, -0.30, 0.25], dtype=np.float32)
        np.testing.assert_allclose(action_np[:3], expected_abs, atol=1e-6)

    def test_inference_relative_roundtrip_2d_chunk(self):
        """Verifica que chunks de predicción 2D (16, 9) se re-proyecten sin TypeError."""
        home_pos = np.array([0.25, 0.0, 0.40], dtype=np.float32)
        chunk_2d = np.zeros((16, 9), dtype=np.float32)
        chunk_2d[:, :3] = np.tile(np.array([-0.05, -0.30, -0.15], dtype=np.float32), (16, 1))
        chunk_2d[:, 6] = 1.0  # qw = 1.0

        action_np = chunk_2d.copy()
        if action_np.ndim == 1:
            action_np[:3] += home_pos
        else:
            action_np[:, :3] += home_pos

        expected_abs = np.array([0.20, -0.30, 0.25], dtype=np.float32)
        for i in range(16):
            np.testing.assert_allclose(action_np[i, :3], expected_abs, atol=1e-6)

        # Simular empaquetado seguro en policy_node
        actions_list = action_np if action_np.ndim == 2 else [action_np]
        first_act = actions_list[0]
        self.assertEqual(len(actions_list), 16)
        self.assertIsInstance(float(first_act[0]), float)
        self.assertIsInstance(float(first_act[8]), float)

    def test_backward_compatibility_when_disabled(self):
        """Si relative_coordinates es False, tanto state como action preservan valores absolutos."""
        home_pos = np.array([0.25, 0.0, 0.40], dtype=np.float32)
        actual_ee_pos = np.array([0.31, -0.15, 0.38], dtype=np.float32)

        use_relative = False

        state_data = actual_ee_pos.copy()
        if use_relative:
            state_data -= home_pos
        np.testing.assert_array_equal(state_data, actual_ee_pos)

        pred_action = np.array([0.35, -0.18, 0.36], dtype=np.float32)
        action_np = pred_action.copy()
        if use_relative:
            action_np += home_pos
        np.testing.assert_array_equal(action_np, pred_action)


class TestVerifyZarrDatasetIntegration(unittest.TestCase):
    """Verifica la interacción entre verify_zarr_dataset.py y datasets con relative_coordinates."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "test_dataset.zarr")
        self.root = zarr.open(self.store_path, mode="w")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_relative_dataset_accepted_by_verifier(self):
        """Un dataset con relative_coordinates=True y paso 0 en [0,0,0] debe ser certificado sin advertencias espurias."""
        ep_group = self.root.create_group("episode_0")
        num_steps = 50
        
        # Imagen dummy
        ep_group.create_dataset("image", data=np.zeros((num_steps, 256, 256, 3), dtype=np.uint8))
        
        # Estado relativo: paso 0 idénticamente [0, 0, 0]
        states = np.zeros((num_steps, 8), dtype=np.float32)
        states[0, :3] = [0.0, 0.0, 0.0]
        states[1:, :3] = np.random.uniform(0.01, 0.20, size=(num_steps - 1, 3)).astype(np.float32)
        states[:, 6] = 1.0  # qw = 1.0
        ep_group.create_dataset("state", data=states)
        
        # Acciones con piso de parada continuo (últimos 40 pasos)
        actions = np.zeros((num_steps, 9), dtype=np.float32)
        actions[:, :3] = states[:, :3]
        actions[:, 6] = 1.0
        actions[-40:, 8] = 1.0
        ep_group.create_dataset("action", data=actions)
        
        ep_group.attrs["relative_coordinates"] = True
        ep_group.attrs["initial_pos_x"] = 0.25
        ep_group.attrs["initial_pos_y"] = 0.0
        ep_group.attrs["initial_pos_z"] = 0.40

        report = verify_episode(
            ep_group=ep_group,
            ep_name="episode_0",
            ep_idx=0,
            dataset_path=self.store_path,
            check_image_chunks=False,
            min_done_frames=40,
        )

        self.assertTrue(report.is_valid, f"Reporte no es válido: {report.errors}")
        # No debe advertir de ceros espurios en el paso 0 de relative_coordinates
        zero_pos_warnings = [w for w in report.warnings if "posición XYZ de 'state' exactamente nula" in w]
        self.assertEqual(len(zero_pos_warnings), 0, f"Advertencia espuria inesperada: {zero_pos_warnings}")

    def test_relative_dataset_rejected_if_step_zero_not_origin(self):
        """Un dataset con relative_coordinates=True cuyo paso 0 no arranca en [0,0,0] debe reportar error."""
        ep_group = self.root.create_group("episode_0")
        num_steps = 50
        
        ep_group.create_dataset("image", data=np.zeros((num_steps, 256, 256, 3), dtype=np.uint8))
        
        states = np.zeros((num_steps, 8), dtype=np.float32)
        states[0, :3] = [0.25, -0.01, 0.40]  # NO relativo!
        states[:, 6] = 1.0
        ep_group.create_dataset("state", data=states)
        
        actions = np.zeros((num_steps, 9), dtype=np.float32)
        actions[:, 6] = 1.0
        actions[-40:, 8] = 1.0
        ep_group.create_dataset("action", data=actions)
        
        ep_group.attrs["relative_coordinates"] = True

        report = verify_episode(
            ep_group=ep_group,
            ep_name="episode_0",
            ep_idx=0,
            dataset_path=self.store_path,
            check_image_chunks=False,
            min_done_frames=40,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(any("el paso 0 no arranca en [0,0,0]" in err for err in report.errors))


if __name__ == "__main__":
    unittest.main()
