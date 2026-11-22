#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
test_collection_telemetry.py — Pruebas unitarias para persistencia y parsing de telemetría
========================================================================================
Valida:
1. Almacenamiento en ZarrStorage con atributos enriquecidos por episodio.
2. Persistencia fail-safe y lectura de dataset_episodes en SQLite experiments.db.
3. Extracción de telemetría desde logs de pickup_approximate a CSV.
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr

# Asegurar importación de módulos
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_core"))
sys.path.insert(0, str(REPO_ROOT / "visuomotor_data_collection"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from visuomotor_core.utils.tracker import init_db, record_dataset_episode
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage
from tools.experiment_tracker import ExperimentTracker
from tools.parse_collection_telemetry import parse_single_log, export_to_csv


class TestCollectionTelemetry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_telem_")
        self.db_path = os.path.join(self.test_dir, "test_experiments.db")
        self._prev_db_path = os.environ.get("VISUOMOTOR_DB_PATH")
        os.environ["VISUOMOTOR_DB_PATH"] = self.db_path

    def tearDown(self):
        # restaurar (no borrar): borrar la variable devolvía a las pruebas siguientes a la base real experiments.db
        if self._prev_db_path is None:
            os.environ.pop("VISUOMOTOR_DB_PATH", None)
        else:
            os.environ["VISUOMOTOR_DB_PATH"] = self._prev_db_path
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_zarr_storage_with_attrs(self):
        """Verifica que ZarrStorage almacene los atributos del episodio en episode_group.attrs."""
        zarr_path = os.path.join(self.test_dir, "test.zarr")
        storage = ZarrStorage(zarr_path)

        buffer = []
        for i in range(10):
            buffer.append({
                "image": np.zeros((256, 256, 3), dtype=np.uint8),
                "state": np.zeros(8, dtype=np.float32),
                "action": np.zeros(9, dtype=np.float32),
            })

        attrs = {
            "obj_spawn_x": -0.35,
            "obj_spawn_y": -0.85,
            "obj_spawn_z": 0.44,
            "obj_spawn_yaw_rad": 1.57,
            "ee_jitter_dx_m": 0.02,
            "retry_count": 1,
            "duration_approach_s": 4.5,
        }

        storage.append_episode(buffer, attrs=attrs)

        # Verificar lectura desde Zarr
        root = zarr.open_group(zarr_path, mode="r")
        self.assertIn("episode_0", root)
        ep = root["episode_0"]
        self.assertEqual(ep.attrs.get("obj_spawn_x"), -0.35)
        self.assertEqual(ep.attrs.get("retry_count"), 1)
        self.assertEqual(ep.attrs.get("duration_approach_s"), 4.5)
        self.assertEqual(ep["image"].shape, (10, 256, 256, 3))

    def test_sqlite_record_and_query_episodes(self):
        """Verifica que record_dataset_episode persista y ExperimentTracker recupere los datos."""
        init_db(Path(self.db_path))

        ok = record_dataset_episode(
            dataset_name="test_dataset.zarr",
            episode_idx=0,
            obj_spawn={"x": 0.28, "y": -0.92, "z": 0.44, "yaw": -0.5},
            ee_jitter={"dx": 0.01, "dy": -0.02, "droll": 0.05, "dpitch": 0.0, "dyaw": 0.01},
            steps=240,
            sim_duration_s=12.0,
            wall_duration_s=48.0,
            cycle_wall_s=60.0,
            retry_count=0,
            phase_durations={"duration_approach_s": 5.2, "duration_grasp_s": 2.0},
            phase_errors={"err_approach_xy_m": 0.0025},
        )
        self.assertTrue(ok)

        tracker = ExperimentTracker(db_path=self.db_path)
        episodes = tracker.get_dataset_episodes(dataset_name="test_dataset.zarr")
        self.assertEqual(len(episodes), 1)
        ep = episodes[0]
        self.assertAlmostEqual(ep["obj_spawn_x"], 0.28, places=3)
        self.assertAlmostEqual(ep["obj_spawn_y"], -0.92, places=3)
        self.assertEqual(ep["steps"], 240)
        self.assertEqual(ep["retry_count"], 0)

    def test_parse_log_and_export_csv(self):
        """Verifica que parse_single_log extraiga datos sintéticos de log y los exporte a CSV."""
        sample_log_path = os.path.join(self.test_dir, "sample_pickup.log")
        log_content = """
[INFO] [1793601825.175972466] [pickup_approximate_node]: Objeto re-posicionado (Halton): X=-0.357, Y=-0.859, Yaw=-2.56
[INFO] [1793601825.178398417] [pickup_approximate_node]: Pose aleatoria EE inyectada: Traslación(dx=0.043m, dy=-0.032m) | Rotación(dRoll=-0.061, dPitch=0.054, dYaw=0.053 rad)
[INFO] [1793601838.038469083] [pickup_approximate_node]: WAIT terminado → APPROACH_XY: Centrando sobre el objeto. [START RECORDING]
[INFO] [1793601843.487019087] [pickup_approximate_node]: APPROACH_XY completo (err_xy=0.0033m, err_z=0.0078m) → ALIGN_ROTATION: Rotando para alinear orientación.
[INFO] [1793601855.467757272] [pickup_approximate_node]: ALIGN_ROTATION completo (err_rot=0.0400rad) → DESCEND: bajando verticalmente.
[INFO] [1793601860.308068309] [pickup_approximate_node]: DESCEND completado/forzado (err_3d=0.0091m, err_rot=0.0002rad) → SETTLE: asimilando error restante.
[INFO] [1793601860.696493246] [pickup_approximate_node]: SETTLE completo → GRASP: cerrando gripper (brazo inmóvil).
[INFO] [1793601862.884579922] [pickup_approximate_node]: GRASP completo → LIFT: levantando objeto.
[INFO] [1793601873.493882671] [pickup_approximate_node]: LIFT completo → UNTILT: Enderezando orientación del objeto.
[INFO] [1793601876.021337649] [pickup_approximate_node]: UNTILT completo y validado → MOVE_TO_TARGET: Llevando objeto a posición destino.
[INFO] [1793601876.087807614] [pickup_approximate_node]: MOVE_TO_TARGET completo (err_3d=0.0359m, err_rot=0.0133rad) → HOLD_TARGET: Manteniendo posición final.
[INFO] [1793601884.598819054] [pickup_approximate_node]: HOLD_TARGET completo. Tarea finalizada. [STOP RECORDING]
[INFO] [1793601884.838122278] [pickup_approximate_node]: Episodio guardado exitosamente: 230 pasos. Total en dataset: 1
[INFO] [1793601884.838677716] [pickup_approximate_node]: ⏱️ TELEMETRÍA EPISODIO 1: Pasos=230 | Duración Sim=11.50s | Grabación Wall=46.80s (0.25x) | Ciclo Total Wall=60.95s
"""
        with open(sample_log_path, "w", encoding="utf-8") as f:
            f.write(log_content)

        episodes = parse_single_log(sample_log_path)
        self.assertEqual(len(episodes), 1)
        ep = episodes[0]
        self.assertEqual(ep["obj_spawn_x"], -0.357)
        self.assertEqual(ep["obj_spawn_y"], -0.859)
        self.assertEqual(ep["obj_spawn_yaw_rad"], -2.56)
        self.assertEqual(ep["ee_jitter_dx_m"], 0.043)
        self.assertEqual(ep["steps"], 230)
        self.assertEqual(ep["duration_sim_s"], 11.5)
        self.assertEqual(ep["err_approach_xy_m"], 0.0033)

        out_csv = os.path.join(self.test_dir, "test_out.csv")
        export_to_csv(episodes, out_csv)
        self.assertTrue(os.path.exists(out_csv))

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(float(reader[0]["obj_spawn_x"]), -0.357)
            self.assertEqual(int(reader[0]["steps"]), 230)

    def test_record_dataset_episode_none_safety(self):
        """Verifica que record_dataset_episode no lance TypeError cuando los campos contengan None."""
        init_db(Path(self.db_path))
        ok = record_dataset_episode(
            dataset_name="test_none.zarr",
            episode_idx=0,
            obj_spawn={"x": None, "y": None, "z": None, "yaw": None},
            ee_jitter={"dx": None, "dy": None, "droll": None, "dpitch": None, "dyaw": None},
            steps=100,
            sim_duration_s=5.0,
            wall_duration_s=20.0,
            cycle_wall_s=25.0,
            retry_count=0,
        )
        self.assertTrue(ok)

        tracker = ExperimentTracker(db_path=self.db_path)
        episodes = tracker.get_dataset_episodes(dataset_name="test_none.zarr")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["obj_spawn_x"], 0.0)

        # Verificar que el formateador del tracker maneje None sin lanzar excepción
        tracker.print_summary_table()

    def test_zarr_storage_safe_episode_index(self):
        """Verifica que ZarrStorage compute max(idx)+1 para evitar colisiones ante grupos no contiguos."""
        zarr_path = os.path.join(self.test_dir, "test_gap.zarr")
        storage = ZarrStorage(zarr_path)

        dummy_step = {
            "image": np.zeros((10, 10, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
            "action": np.zeros(9, dtype=np.float32),
        }
        # Crear episode_0
        storage.append_episode([dummy_step])
        # Crear un hueco manualmente (episode_4)
        storage.root.create_group("episode_4")

        # El próximo debe ser episode_5, no episode_2
        storage.append_episode([dummy_step])
        self.assertIn("episode_5", storage.root)

    def test_parse_all_logs_zarr_alignment(self):
        """Verifica que parse_all_logs alinee con 100% de exactitud contra la secuencia Zarr aun con mtime invertido."""
        from tools.parse_collection_telemetry import parse_all_logs, read_zarr_step_counts

        # 1. Crear dataset Zarr sintético con 2 episodios: ep_0=15 pasos, ep_1=25 pasos
        zarr_path = os.path.join(self.test_dir, "mock_align.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        for ep_idx, steps in [(0, 15), (1, 25)]:
            g = root.create_group(f"episode_{ep_idx}")
            g.create_dataset("action", data=np.zeros((steps, 9), dtype=np.float32), chunks=(1, 9))

        zarr_steps = read_zarr_step_counts(zarr_path)
        self.assertEqual(zarr_steps, [15, 25])

        # 2. Crear dos logs donde el archivo con 25 pasos tiene fecha de modificación más antigua
        log_25_path = os.path.join(self.test_dir, "log_b2_batch_2.log")
        with open(log_25_path, "w", encoding="utf-8") as f:
            f.write("[INFO] [100.0] Usando dataset especificado: mock_batch_2.zarr\n")
            f.write("[INFO] [105.0] ⏱️ TELEMETRÍA EPISODIO 1: Pasos=25 | Duración Sim=1.25s | Grabación Wall=5.0s (0.25x) | Ciclo Total Wall=6.0s\n")

        log_15_path = os.path.join(self.test_dir, "log_b1_batch_1.log")
        with open(log_15_path, "w", encoding="utf-8") as f:
            f.write("[INFO] [200.0] Usando dataset especificado: mock_batch_1.zarr\n")
            f.write("[INFO] [205.0] ⏱️ TELEMETRÍA EPISODIO 1: Pasos=15 | Duración Sim=0.75s | Grabación Wall=3.0s (0.25x) | Ciclo Total Wall=4.0s\n")

        # Invertir mtimes explícitamente: log_25 es más reciente que log_15
        os.utime(log_15_path, (1000, 1000))
        os.utime(log_25_path, (2000, 2000))

        # Al parsear con align_zarr, DEBE respetar [15, 25] de Zarr, no el mtime
        episodes = parse_all_logs([log_25_path, log_15_path], align_zarr=zarr_path)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["steps"], 15)
        self.assertEqual(episodes[0]["global_episode_idx"], 0)
        self.assertEqual(episodes[1]["steps"], 25)
        self.assertEqual(episodes[1]["global_episode_idx"], 1)


if __name__ == "__main__":
    unittest.main()
