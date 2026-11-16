#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
test_verify_zarr_dataset.py — Suite de Pruebas Unitarias para verify_zarr_dataset.py
====================================================================================
Verifica minuciosamente todas las capacidades de detección de errores de verify_zarr_dataset:
1. Validación de dataset íntegro, métricas, compatibilidad DataLoader y DiffusionPolicy.
2. Detección de discontinuidades en la secuencia de episodios.
3. Detección de la regla arquitectónica crítica: Fuga Causal (state con 9 dimensiones).
4. Detección de tipos de datos incorrectos (ej. image float32 en vez de uint8).
5. Detección de formas anómalas (ej. state con 7 dimensiones, action con 8 dimensiones).
6. Detección de NaNs e Infs en datos de estado y acción.
7. Detección de ceros espurios y cuaterniones nulos [0,0,0,0] en state y action.
8. Detección de anomalías en timestamps (ceros espurios 0.0, dtypes incorrectos, no-monotónicos).
9. Validación de metadatos Zarr (.zgroup, .zarray corruptos o vacíos).
10. Detección de nodo Grupo en lugar de Array numérico.
11. Detección de ausencia de piso de parada y señales de parada no-contiguas/intermitentes.
12. Detección de desajuste de longitud temporal T entre modalidades.
13. Detección de chunks corruptos o truncados en disco.
14. Verificación de modo rápido (--skip-images) con validación obligatoria de chunk 0.
15. Ejecución CLI de verify_zarr_dataset.py vía subproceso con flags (--strict, --out-json).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import zarr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TOOLS_DIR = os.path.join(REPO_DIR, "tools")
CORE_DIR = os.path.join(REPO_DIR, "visuomotor_core")
for d in (TOOLS_DIR, CORE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from verify_zarr_dataset import (
    verify_dataset,
    verify_episode,
    verify_zarr_structure,
    verify_torch_dataloader,
)


def create_mock_episode(
    ep_group,
    num_steps: int = 25,
    img_shape=(256, 256, 3),
    corrupt_mode: str = None,
    done_frames_count: int = 5,
    include_timestamp: bool = False,
):
    """Genera un episodio sintético válido o inyecta anomalías controladas."""
    images = np.random.randint(0, 255, size=(num_steps, *img_shape), dtype=np.uint8)
    # Posiciones nominales en [-1.0, 1.0] metros
    states = np.zeros((num_steps, 8), dtype=np.float32)
    states[:, :3] = np.random.uniform(-1.0, 1.0, size=(num_steps, 3)).astype(np.float32)
    # Cuaterniones unitarios variados (norma 1.0, qw >= 0)
    quats = np.random.randn(num_steps, 4).astype(np.float32)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-6)
    quats[quats[:, 3] < 0] *= -1.0
    states[:, 3:7] = quats
    states[:, 7] = np.linspace(0.0, -0.01, num_steps, dtype=np.float32)

    actions = np.zeros((num_steps, 9), dtype=np.float32)
    actions[:, :3] = states[:, :3] + np.random.uniform(-0.05, 0.05, size=(num_steps, 3)).astype(np.float32)
    actions[:, 3:7] = quats
    actions[:, 7] = states[:, 7]
    actions[:, 8] = 0.0
    if done_frames_count > 0:
        actions[-done_frames_count:, 8] = 1.0

    if corrupt_mode == "causal_leakage":
        states = np.zeros((num_steps, 9), dtype=np.float32)
        states[:, 3:7] = quats
    elif corrupt_mode == "wrong_dtype_image":
        images = images.astype(np.float32)
    elif corrupt_mode == "wrong_dtype_state":
        states = states.astype(np.float64)
    elif corrupt_mode == "wrong_dim_state":
        states = states[:, :7]
    elif corrupt_mode == "wrong_dim_action":
        actions = actions[:, :8]
    elif corrupt_mode == "nan_state":
        states[5, 0] = np.nan
    elif corrupt_mode == "inf_action":
        actions[10, 2] = np.inf
    elif corrupt_mode == "mismatch_len":
        states = states[: num_steps // 2]
    elif corrupt_mode == "no_done":
        actions[:, 8] = 0.0
    elif corrupt_mode == "done_out_of_range":
        actions[0, 8] = 2.5
    elif corrupt_mode == "short_episode":
        images = images[:10]
        states = states[:10]
        actions = actions[:10]
    elif corrupt_mode == "zero_state_row":
        states[3, :] = 0.0
    elif corrupt_mode == "null_state_quat":
        states[4, 3:7] = 0.0
    elif corrupt_mode == "zero_action_row":
        actions[2, :] = 0.0
    elif corrupt_mode == "null_action_quat":
        actions[5, 3:7] = 0.0
    elif corrupt_mode == "intermittent_done":
        # Activar done en un fotograma intermedio antes del piso terminal
        actions[2, 8] = 1.0

    ep_group.create_dataset("image", data=images, chunks=(1, *images.shape[1:]), dtype=images.dtype)
    ep_group.create_dataset("state", data=states, chunks=(1, *states.shape[1:]), dtype=states.dtype)
    ep_group.create_dataset("action", data=actions, chunks=(1, *actions.shape[1:]), dtype=actions.dtype)

    if include_timestamp:
        t0 = 1700000000.0
        timestamps = t0 + np.arange(num_steps, dtype=np.float64) * 0.050
        if corrupt_mode == "zero_timestamp":
            timestamps[5] = 0.0
        elif corrupt_mode == "non_monotonic_timestamp":
            timestamps[5] = timestamps[4]  # repetido
        elif corrupt_mode == "wrong_dtype_timestamp":
            timestamps = timestamps.astype(np.int32)
        elif corrupt_mode == "wrong_dim_timestamp":
            timestamps = timestamps.reshape(-1, 1)

        ep_group.create_dataset("timestamp", data=timestamps, chunks=(1,), dtype=timestamps.dtype)


class TestVerifyZarrDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_verify_zarr_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_valid_dataset(self):
        """Verifica que un dataset sintético válido sea certificado exitosamente."""
        ds_path = os.path.join(self.test_dir, "valid.zarr")
        root = zarr.open_group(ds_path, mode="w")

        for i in range(3):
            ep_grp = root.create_group(f"episode_{i}")
            create_mock_episode(ep_grp, num_steps=30, done_frames_count=10)

        report = verify_dataset(
            ds_path,
            check_all_images=True,
            min_done_frames=5,
            test_dataloader=True,
            test_policy_step=True,
            batch_size=2,
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_episodes, 3)
        self.assertEqual(report.episodes_passed, 3)
        self.assertEqual(report.episodes_failed, 0)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(report.metrics["total_steps"], 90)
        self.assertTrue(report.dataloader_compatible)
        self.assertTrue(report.policy_step_verified)
        self.assertIsNotNone(report.initial_policy_loss)

    def test_missing_episode_in_sequence(self):
        """Verifica la detección de discontinuidad en la secuencia de episodios."""
        ds_path = os.path.join(self.test_dir, "discontinuous.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"))
        create_mock_episode(root.create_group("episode_2"))

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Discontinuidad" in err for err in report.errors))

    def test_causal_leakage_detection(self):
        """Verifica la detección estricta de Fuga Causal (state con 9 dimensiones)."""
        ds_path = os.path.join(self.test_dir, "causal_leak.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), corrupt_mode="causal_leakage")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 1)
        self.assertTrue(any("CAUSAL LEAKAGE" in err for err in report.errors))

    def test_dtype_and_shape_mismatches(self):
        """Verifica la detección de tipos de datos y formas incorrectas."""
        ds_path = os.path.join(self.test_dir, "mismatches.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), corrupt_mode="wrong_dtype_image")
        create_mock_episode(root.create_group("episode_1"), corrupt_mode="wrong_dim_state")
        create_mock_episode(root.create_group("episode_2"), corrupt_mode="wrong_dim_action")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 3)
        self.assertTrue(any("uint8" in err for err in report.errors))
        self.assertTrue(any("exactamente 8 dimensiones" in err for err in report.errors))
        self.assertTrue(any("exactamente 9 dimensiones" in err for err in report.errors))

    def test_nan_and_inf_detection(self):
        """Verifica la detección de NaNs e Infs en state y action."""
        ds_path = os.path.join(self.test_dir, "nan_inf.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), corrupt_mode="nan_state")
        create_mock_episode(root.create_group("episode_1"), corrupt_mode="inf_action")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 2)
        self.assertTrue(any("NaN" in err for err in report.errors))
        self.assertTrue(any("Inf" in err for err in report.errors))

    def test_spurious_zeros_and_null_quaternions(self):
        """Verifica la detección rigurosa de ceros espurios y cuaterniones nulos [0,0,0,0]."""
        ds_path = os.path.join(self.test_dir, "zeros_check.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), corrupt_mode="zero_state_row")
        create_mock_episode(root.create_group("episode_1"), corrupt_mode="null_state_quat")
        create_mock_episode(root.create_group("episode_2"), corrupt_mode="zero_action_row")
        create_mock_episode(root.create_group("episode_3"), corrupt_mode="null_action_quat")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 4)
        self.assertTrue(any("vector de 'state' completamente nulo" in err for err in report.errors))
        self.assertTrue(any("cuaternión de 'state' nulo" in err for err in report.errors))
        self.assertTrue(any("vector de 'action' completamente nulo" in err for err in report.errors))
        self.assertTrue(any("cuaternión de 'action' nulo" in err for err in report.errors))

    def test_timestamp_validations(self):
        """Verifica validación de timestamps: válido, ceros espurios, no-monotonía y dtypes."""
        ds_path = os.path.join(self.test_dir, "ts_check.zarr")
        root = zarr.open_group(ds_path, mode="w")

        # ep_0 válido con timestamp
        create_mock_episode(root.create_group("episode_0"), include_timestamp=True)
        # ep_1 con timestamp 0.0 espurio
        create_mock_episode(root.create_group("episode_1"), include_timestamp=True, corrupt_mode="zero_timestamp")
        # ep_2 con timestamps no monotónicos
        create_mock_episode(root.create_group("episode_2"), include_timestamp=True, corrupt_mode="non_monotonic_timestamp")
        # ep_3 con dtype timestamp erróneo
        create_mock_episode(root.create_group("episode_3"), include_timestamp=True, corrupt_mode="wrong_dtype_timestamp")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        # ep_1 y ep_3 son errores fatales
        self.assertTrue(any("ceros espurios" in err for err in report.errors))
        self.assertTrue(any("float64 o float32" in err for err in report.errors))
        # ep_2 es advertencia de no-monotonía
        self.assertTrue(any("no son estrictamente crecientes" in warn for warn in report.warnings))

    def test_metadata_corruptions(self):
        """Verifica la detección de metadatos corruptos (.zgroup, .zarray con JSON inválido o vacíos)."""
        ds_path = os.path.join(self.test_dir, "meta_corrupt.zarr")
        root = zarr.open_group(ds_path, mode="w")
        create_mock_episode(root.create_group("episode_0"))

        # Corromper el archivo .zarray de image
        zarray_path = os.path.join(ds_path, "episode_0", "image", ".zarray")
        self.assertTrue(os.path.isfile(zarray_path))
        with open(zarray_path, "w", encoding="utf-8") as f:
            f.write("INVALID_JSON_CONTENT_CORRUPTED{{{")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 1)
        self.assertTrue(any("Metadatos '.zarray' de 'image' corruptos" in err for err in report.errors))

    def test_group_instead_of_array_detection(self):
        """Verifica la detección cuando un nodo esperado como Array es un Grupo."""
        ds_path = os.path.join(self.test_dir, "group_anomaly.zarr")
        root = zarr.open_group(ds_path, mode="w")
        ep0 = root.create_group("episode_0")
        create_mock_episode(ep0)

        # Eliminar el array 'state' y crear un subgrupo llamado 'state'
        del ep0["state"]
        ep0.create_group("state")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 1)
        self.assertTrue(any("no es un Array de Zarr" in err for err in report.errors))

    def test_done_flag_validations(self):
        """Verifica la validación del rango del done_flag, ausencia y no-contigüidad."""
        ds_path = os.path.join(self.test_dir, "done_check.zarr")
        root = zarr.open_group(ds_path, mode="w")

        # Episodio sin señal de parada
        create_mock_episode(root.create_group("episode_0"), corrupt_mode="no_done")
        # Episodio con done_flag fuera de rango
        create_mock_episode(root.create_group("episode_1"), corrupt_mode="done_out_of_range")
        # Episodio con done_flag no contiguo
        create_mock_episode(root.create_group("episode_2"), corrupt_mode="intermittent_done")

        report = verify_dataset(ds_path, min_done_frames=3, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("no contiene ninguna señal de término" in err for err in report.errors))
        self.assertTrue(any("fuera del rango" in err for err in report.errors))
        self.assertTrue(any("Señal de parada no contigua" in warn for warn in report.warnings))

    def test_temporal_mismatch_and_short_episodes(self):
        """Verifica la detección de pasos temporales desfasados y episodios menores a H_act."""
        ds_path = os.path.join(self.test_dir, "temporal.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), corrupt_mode="mismatch_len")
        create_mock_episode(root.create_group("episode_1"), corrupt_mode="short_episode")

        report = verify_dataset(ds_path, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 2)
        self.assertTrue(any("Desajuste de pasos temporales" in err for err in report.errors))
        self.assertTrue(any("demasiado corto" in err for err in report.errors))

    def test_corrupt_chunk_detection(self):
        """Verifica que un archivo de chunk truncado en disco sea detectado durante la descompresión."""
        ds_path = os.path.join(self.test_dir, "corrupt_chunk.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), num_steps=20)

        # Corromper deliberadamente uno de los chunks de imagen en disco
        image_chunk_dir = os.path.join(ds_path, "episode_0", "image")
        chunk_files = [
            f for f in os.listdir(image_chunk_dir) if f not in (".zarray", ".zattrs")
        ]
        self.assertTrue(len(chunk_files) > 0)
        target_chunk = os.path.join(image_chunk_dir, chunk_files[0])
        with open(target_chunk, "wb") as f:
            f.write(b"CORRUPTED_DATA_TRUNCATED")

        report = verify_dataset(ds_path, check_all_images=True, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 1)
        self.assertTrue(any("Fallo de descompresión" in err for err in report.errors))

    def test_skip_images_mode_checks_chunk_zero(self):
        """Verifica que incluso con check_all_images=False, se compruebe el chunk 0."""
        ds_path = os.path.join(self.test_dir, "skip_img.zarr")
        root = zarr.open_group(ds_path, mode="w")

        create_mock_episode(root.create_group("episode_0"), num_steps=20)

        # Corromper chunk 0
        target_chunk = os.path.join(ds_path, "episode_0", "image", "0.0.0.0")
        if not os.path.exists(target_chunk):
            # En zarr, chunk index 0 puede ser '0.0.0.0' o '0' dependiendo de chunks
            chunk_candidates = [
                os.path.join(ds_path, "episode_0", "image", f)
                for f in os.listdir(os.path.join(ds_path, "episode_0", "image"))
                if f not in (".zarray", ".zattrs")
            ]
            target_chunk = sorted(chunk_candidates)[0]

        with open(target_chunk, "wb") as f:
            f.write(b"CORRUPTED_CHUNK_ZERO")

        report = verify_dataset(ds_path, check_all_images=False, test_dataloader=False)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.episodes_failed, 1)
        self.assertTrue(any("Fallo de descompresión" in err for err in report.errors))

    def test_cli_execution_and_json_report(self):
        """Verifica la ejecución de verify_zarr_dataset.py desde línea de comandos con flags."""
        ds_path = os.path.join(self.test_dir, "cli_valid.zarr")
        json_out = os.path.join(self.test_dir, "report.json")
        root = zarr.open_group(ds_path, mode="w")

        for i in range(2):
            create_mock_episode(root.create_group(f"episode_{i}"), num_steps=20, done_frames_count=5)

        cmd = [
            sys.executable,
            os.path.join(TOOLS_DIR, "verify_zarr_dataset.py"),
            "--dataset", ds_path,
            "--min-done-frames", "3",
            "--out-json", json_out,
            "--batch-size", "2",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, f"CLI falló con: {result.stderr}")
        self.assertTrue(os.path.isfile(json_out))

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["total_episodes"], 2)
        self.assertTrue(data["policy_step_verified"])


if __name__ == "__main__":
    unittest.main()
