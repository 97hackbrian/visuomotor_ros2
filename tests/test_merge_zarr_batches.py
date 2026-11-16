#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
test_merge_zarr_batches.py — Suite de Pruebas Unitarias para merge_zarr_batches.py
==================================================================================
Verificación exhaustiva de:
1. Ordenamiento natural de lotes y episodios.
2. Detección y continuación determinista de índices (ej. episode_178 -> episode_179).
3. Validación estricta de tensores (dtypes, shapes y concordancia de pasos).
4. Fusión limpia en dataset nuevo y en dataset preexistente.
5. Opción --remove-sources y resiliencia ante batches vacíos o intentos de auto-fusión.
6. Ejecución CLI de herramientas vía subproceso.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import zarr

# Asegurar importación de tools
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TOOLS_DIR = os.path.join(REPO_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from merge_zarr_batches import (
    natural_sort_key,
    get_sorted_episode_groups,
    get_next_episode_index,
    validate_episode,
    find_input_paths,
    merge_batches,
)


def create_dummy_episode(ep_group, num_steps=20, img_shape=(256, 256, 3), corrupt_field=None, wrong_dtype=None):
    """Crea un episodio sintético válido o corrupto para pruebas."""
    images = np.random.randint(0, 255, size=(num_steps, *img_shape), dtype=np.uint8)
    states = np.random.randn(num_steps, 8).astype(np.float32)
    actions = np.random.randn(num_steps, 9).astype(np.float32)
    actions[:, 8] = 0.0
    actions[-5:, 8] = 1.0  # done_flag

    if wrong_dtype == "image":
        images = images.astype(np.float32)
    elif wrong_dtype == "state":
        states = states.astype(np.float64)

    if corrupt_field == "mismatch_len":
        states = states[: num_steps // 2]
    elif corrupt_field == "wrong_dim_state":
        states = np.random.randn(num_steps, 7).astype(np.float32)

    ep_group.create_dataset("image", data=images, chunks=(1, *images.shape[1:]), dtype=images.dtype)
    if corrupt_field != "missing_state":
        ep_group.create_dataset("state", data=states, chunks=(1, *states.shape[1:]), dtype=states.dtype)
    if corrupt_field != "missing_action":
        ep_group.create_dataset("action", data=actions, chunks=(1, *actions.shape[1:]), dtype=actions.dtype)


class TestMergeZarrBatchesUnit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_zarr_merge_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_natural_sort_key(self):
        raw = ["batch_10.zarr", "batch_1.zarr", "batch_2.zarr", "batch_20.zarr"]
        sorted_list = sorted(raw, key=natural_sort_key)
        expected = ["batch_1.zarr", "batch_2.zarr", "batch_10.zarr", "batch_20.zarr"]
        self.assertEqual(sorted_list, expected)

    def test_get_sorted_episode_groups(self):
        path = os.path.join(self.test_dir, "test_order.zarr")
        root = zarr.open_group(path, mode="w")
        root.create_group("episode_10")
        root.create_group("episode_0")
        root.create_group("episode_2")
        root.create_group("unrelated_meta")

        eps = get_sorted_episode_groups(root)
        self.assertEqual(len(eps), 3)
        self.assertEqual([idx for idx, _ in eps], [0, 2, 10])
        self.assertEqual([name for _, name in eps], ["episode_0", "episode_2", "episode_10"])

    def test_get_next_episode_index(self):
        path = os.path.join(self.test_dir, "test_next_idx.zarr")
        root = zarr.open_group(path, mode="w")
        self.assertEqual(get_next_episode_index(root), 0)

        # Crear episode_0 a episode_178 (simulando demonstrations_v8)
        for i in range(179):
            root.create_group(f"episode_{i}")
        self.assertEqual(get_next_episode_index(root), 179)

    def test_validate_episode_valid(self):
        path = os.path.join(self.test_dir, "test_valid.zarr")
        root = zarr.open_group(path, mode="w")
        ep = root.create_group("episode_0")
        create_dummy_episode(ep, num_steps=15)

        valid, reason = validate_episode(ep, "episode_0")
        self.assertTrue(valid, f"Falló validación de episodio válido: {reason}")
        self.assertEqual(reason, "OK")

    def test_validate_episode_corruptions(self):
        path = os.path.join(self.test_dir, "test_corrupt.zarr")
        root = zarr.open_group(path, mode="w")

        # 1. Falta 'action'
        ep1 = root.create_group("ep_missing_act")
        create_dummy_episode(ep1, corrupt_field="missing_action")
        valid, reason = validate_episode(ep1, "ep_missing_act")
        self.assertFalse(valid)
        self.assertIn("Falta el dataset requerido 'action'", reason)

        # 2. Desajuste de longitud
        ep2 = root.create_group("ep_mismatch")
        create_dummy_episode(ep2, corrupt_field="mismatch_len")
        valid, reason = validate_episode(ep2, "ep_mismatch")
        self.assertFalse(valid)
        self.assertIn("Desajuste de pasos temporales", reason)

        # 3. Dtype incorrecto en imagen
        ep3 = root.create_group("ep_wrong_img_dtype")
        create_dummy_episode(ep3, wrong_dtype="image")
        valid, reason = validate_episode(ep3, "ep_wrong_img_dtype")
        self.assertFalse(valid)
        self.assertIn("'image' debe ser uint8", reason)

        # 4. Dimensión incorrecta en state
        ep4 = root.create_group("ep_wrong_state_dim")
        create_dummy_episode(ep4, corrupt_field="wrong_dim_state")
        valid, reason = validate_episode(ep4, "ep_wrong_state_dim")
        self.assertFalse(valid)
        self.assertIn("'state' debe tener dimensiones (T, 8)", reason)

    def test_find_input_paths_excludes_output(self):
        out_path = os.path.join(self.test_dir, "demo.zarr")
        b1 = os.path.join(self.test_dir, "demo_batch_1.zarr")
        b2 = os.path.join(self.test_dir, "demo_batch_2.zarr")
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(b1, exist_ok=True)
        os.makedirs(b2, exist_ok=True)

        inputs = [os.path.join(self.test_dir, "demo*.zarr")]
        resolved = find_input_paths(inputs, out_path)
        # Debe incluir b1 y b2 pero excluir out_path
        self.assertIn(os.path.realpath(b1), resolved)
        self.assertIn(os.path.realpath(b2), resolved)
        self.assertNotIn(os.path.realpath(out_path), resolved)

    def test_merge_batches_into_new_destination(self):
        b1_path = os.path.join(self.test_dir, "batch_1.zarr")
        b2_path = os.path.join(self.test_dir, "batch_2.zarr")
        out_path = os.path.join(self.test_dir, "merged.zarr")

        # Crear lote 1 con 3 episodios
        b1_root = zarr.open_group(b1_path, mode="w")
        for i in range(3):
            ep = b1_root.create_group(f"episode_{i}")
            create_dummy_episode(ep, num_steps=10 + i)

        # Crear lote 2 con 2 episodios
        b2_root = zarr.open_group(b2_path, mode="w")
        for i in range(2):
            ep = b2_root.create_group(f"episode_{i}")
            create_dummy_episode(ep, num_steps=20 + i)

        success = merge_batches(
            input_paths=[b1_path, b2_path],
            output_path=out_path,
            remove_sources=False,
        )
        self.assertTrue(success)

        # Verificar destino
        out_root = zarr.open_group(out_path, mode="r")
        eps = get_sorted_episode_groups(out_root)
        self.assertEqual(len(eps), 5)
        self.assertEqual([name for _, name in eps], [f"episode_{i}" for i in range(5)])

        # Comprobar longitudes originales
        self.assertEqual(out_root["episode_0/image"].shape[0], 10)
        self.assertEqual(out_root["episode_1/image"].shape[0], 11)
        self.assertEqual(out_root["episode_2/image"].shape[0], 12)
        self.assertEqual(out_root["episode_3/image"].shape[0], 20)
        self.assertEqual(out_root["episode_4/image"].shape[0], 21)

    def test_merge_batches_into_existing_destination(self):
        out_path = os.path.join(self.test_dir, "preexisting.zarr")
        b1_path = os.path.join(self.test_dir, "batch_add.zarr")

        # Destino ya tiene 3 episodios (episode_0, episode_1, episode_2)
        out_root = zarr.open_group(out_path, mode="w")
        for i in range(3):
            ep = out_root.create_group(f"episode_{i}")
            create_dummy_episode(ep, num_steps=15)

        # Lote fuente tiene 2 episodios (episode_0, episode_1)
        b1_root = zarr.open_group(b1_path, mode="w")
        for i in range(2):
            ep = b1_root.create_group(f"episode_{i}")
            create_dummy_episode(ep, num_steps=25)

        success = merge_batches(
            input_paths=[b1_path],
            output_path=out_path,
            remove_sources=True,
        )
        self.assertTrue(success)

        # Verificar que el lote fuente fue eliminado
        self.assertFalse(os.path.exists(b1_path))

        # Verificar que el destino ahora tiene 5 episodios (episode_0..episode_4)
        out_root = zarr.open_group(out_path, mode="r")
        eps = get_sorted_episode_groups(out_root)
        self.assertEqual(len(eps), 5)
        self.assertEqual([name for _, name in eps], [f"episode_{i}" for i in range(5)])
        self.assertEqual(out_root["episode_3/image"].shape[0], 25)
        self.assertEqual(out_root["episode_4/image"].shape[0], 25)

    def test_merge_cli_invocation(self):
        b1_path = os.path.join(self.test_dir, "cli_b1.zarr")
        out_path = os.path.join(self.test_dir, "cli_out.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        ep = b1_root.create_group("episode_0")
        create_dummy_episode(ep, num_steps=12)

        script_path = os.path.join(TOOLS_DIR, "merge_zarr_batches.py")
        cmd = [
            sys.executable,
            script_path,
            "--inputs",
            b1_path,
            "--output",
            out_path,
            "--remove-sources",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error en CLI: {res.stderr}")
        self.assertFalse(os.path.exists(b1_path))
        self.assertTrue(os.path.exists(out_path))

        out_root = zarr.open_group(out_path, mode="r")
        self.assertEqual(len(get_sorted_episode_groups(out_root)), 1)

    def test_merge_with_empty_batch(self):
        b1_path = os.path.join(self.test_dir, "nonempty_b1.zarr")
        b2_path = os.path.join(self.test_dir, "empty_b2.zarr")
        out_path = os.path.join(self.test_dir, "out_with_empty.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        ep = b1_root.create_group("episode_0")
        create_dummy_episode(ep, num_steps=10)

        # Batch 2 vacío
        zarr.open_group(b2_path, mode="w")

        success = merge_batches(
            input_paths=[b1_path, b2_path],
            output_path=out_path,
            remove_sources=True,
        )
        self.assertTrue(success)
        out_root = zarr.open_group(out_path, mode="r")
        self.assertEqual(len(get_sorted_episode_groups(out_root)), 1)
        self.assertFalse(os.path.exists(b1_path))
        self.assertFalse(os.path.exists(b2_path))

    def test_merge_preserves_attributes(self):
        b1_path = os.path.join(self.test_dir, "attrs_b1.zarr")
        out_path = os.path.join(self.test_dir, "out_attrs.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        ep = b1_root.create_group("episode_0")
        create_dummy_episode(ep, num_steps=10)
        ep.attrs["custom_tag"] = "test_tag_42"
        ep["action"].attrs["action_units"] = "meters_and_quaternions"

        success = merge_batches(
            input_paths=[b1_path],
            output_path=out_path,
            remove_sources=False,
        )
        self.assertTrue(success)
        out_root = zarr.open_group(out_path, mode="r")
        self.assertEqual(out_root["episode_0"].attrs.get("custom_tag"), "test_tag_42")
        self.assertEqual(out_root["episode_0/action"].attrs.get("action_units"), "meters_and_quaternions")

    def test_merge_skips_corrupt_episode_non_strict(self):
        b1_path = os.path.join(self.test_dir, "partially_corrupt_b1.zarr")
        out_path = os.path.join(self.test_dir, "out_corrupt_skip.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        ep0 = b1_root.create_group("episode_0")
        create_dummy_episode(ep0, num_steps=10)
        ep1 = b1_root.create_group("episode_1")
        # Simular corte abortado por Ctrl+C
        create_dummy_episode(ep1, corrupt_field="mismatch_len")
        ep2 = b1_root.create_group("episode_2")
        create_dummy_episode(ep2, num_steps=15)

        success = merge_batches(
            input_paths=[b1_path],
            output_path=out_path,
            remove_sources=False,
            strict=False,
        )
        self.assertTrue(success, "La fusión debe tener éxito omitiendo el episodio corrupto en modo non-strict")

        out_root = zarr.open_group(out_path, mode="r")
        eps = get_sorted_episode_groups(out_root)
        self.assertEqual(len(eps), 2)
        self.assertEqual([name for _, name in eps], ["episode_0", "episode_1"])
        self.assertEqual(out_root["episode_0/image"].shape[0], 10)
        self.assertEqual(out_root["episode_1/image"].shape[0], 15)

    def test_merge_corrupt_episode_strict_aborts_without_modifying_destination(self):
        b1_path = os.path.join(self.test_dir, "strict_b1.zarr")
        out_path = os.path.join(self.test_dir, "out_strict.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        ep0 = b1_root.create_group("episode_0")
        create_dummy_episode(ep0, num_steps=10)
        ep1 = b1_root.create_group("episode_1")
        create_dummy_episode(ep1, corrupt_field="mismatch_len")

        success = merge_batches(
            input_paths=[b1_path],
            output_path=out_path,
            remove_sources=False,
            strict=True,
        )
        self.assertFalse(success, "Modo estricto debe retornar False ante episodios corruptos")
        # El destino no debe haber sido creado o modificado
        self.assertFalse(os.path.exists(out_path))

    def test_find_input_paths_excludes_ancestor_directories(self):
        parent_dir = os.path.join(self.test_dir, "datasets_dir")
        os.makedirs(parent_dir, exist_ok=True)
        dest_path = os.path.join(parent_dir, "demonstrations_v8.zarr")
        batch_path = os.path.join(parent_dir, "demonstrations_v8_batch_1.zarr")
        os.makedirs(dest_path, exist_ok=True)
        os.makedirs(batch_path, exist_ok=True)

        # Si el usuario pasa erróneamente el directorio padre
        resolved = find_input_paths([parent_dir, batch_path], dest_path)
        # Debe excluir parent_dir (por ser ancestro del destino) y dest_path, manteniendo solo batch_path
        self.assertNotIn(os.path.realpath(parent_dir), resolved)
        self.assertNotIn(os.path.realpath(dest_path), resolved)
        self.assertIn(os.path.realpath(batch_path), resolved)

    def test_cli_strict_flag(self):
        b1_path = os.path.join(self.test_dir, "cli_corrupt.zarr")
        out_path = os.path.join(self.test_dir, "cli_strict_out.zarr")

        b1_root = zarr.open_group(b1_path, mode="w")
        create_dummy_episode(b1_root.create_group("episode_0"), corrupt_field="missing_action")

        script_path = os.path.join(TOOLS_DIR, "merge_zarr_batches.py")
        cmd = [
            sys.executable,
            script_path,
            "--inputs",
            b1_path,
            "--output",
            out_path,
            "--strict",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0, "CLI con --strict debe fallar ante corrupción")
        self.assertFalse(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
