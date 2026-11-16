#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Base de datos relacional (SQLite) para registro, consulta y comparación sistemática
de todos los experimentos realizados en el framework visuomotor.
Muestra el estado consolidado de:
1. Entrenamientos Supervisados (Diffusion Policy)
2. Fine-Tuning por Refuerzo (DPPO RL)
3. Evaluaciones y Benchmarking
4. Datasets Recolectados
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Añadir visuomotor_core al path si está disponible
sys.path.append(str(Path(__file__).parent.parent / "visuomotor_core"))
from visuomotor_core.utils.tracker import (
    find_db_path,
    init_db,
    record_dataset,
    record_dataset_episode,
    record_eval_run,
    record_rl_run,
    record_train_run,
)


class ExperimentTracker:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else find_db_path()
        init_db(self.db_path)

    def _query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[WARN] Error al consultar base de datos: {e}")
            return []

    def get_train_runs(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM train_runs ORDER BY timestamp DESC")

    def get_rl_runs(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM rl_runs ORDER BY timestamp DESC")

    def get_eval_runs(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM eval_runs ORDER BY timestamp DESC")

    def get_datasets(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM datasets ORDER BY timestamp DESC")

    def get_dataset_episodes(self, dataset_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if dataset_name:
            return self._query(
                "SELECT * FROM dataset_episodes WHERE dataset_name = ? ORDER BY episode_idx ASC LIMIT ?",
                (dataset_name, limit),
            )
        return self._query("SELECT * FROM dataset_episodes ORDER BY timestamp DESC LIMIT ?", (limit,))

    def print_summary_table(self) -> None:
        print("\n" + "=" * 90)
        print("📊 REPOSITORIO DE EXPERIMENTOS CIENTÍFICOS — VISUOMOTOR ROS 2")
        print(f"📁 Base de Datos: {self.db_path}")
        print("=" * 90)

        # 1. ENTRENAMIENTOS SUPERVISADOS (DP)
        train_runs = self.get_train_runs()
        print(f"\n🧠 1. ENTRENAMIENTOS SUPERVISADOS — DIFFUSION POLICY ({len(train_runs)} registrados)")
        if train_runs:
            header = f"{'ID':<28} | {'Fecha':<16} | {'Épocas':<8} | {'Batch':<6} | {'Loss':<8} | {'Loss EMA':<9} | {'VRAM(MB)':<9} | {'Duración':<10}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for r in train_runs:
                date_str = r['timestamp'][:16].replace('T', ' ')
                dur_str = f"{r['duration_s']:.1f}s"
                print(f"{r['id'][:28]:<28} | {date_str:<16} | {r['epochs_completed']}/{r['epochs_requested']:<5} | {r['batch_size']:<6} | {r['final_loss']:<8.4f} | {r['final_loss_ema']:<9.4f} | {r['vram_peak_mb']:<9.0f} | {dur_str:<10}")
        else:
            print("   (No hay entrenamientos supervisados registrados aún)")

        # 2. ENTRENAMIENTOS POR REFUERZO (RL DPPO)
        rl_runs = self.get_rl_runs()
        print(f"\n🎯 2. FINE-TUNING POR REFUERZO — DPPO RL ({len(rl_runs)} registrados)")
        if rl_runs:
            header = f"{'ID':<28} | {'Fecha':<16} | {'Iters':<6} | {'Mean Reward':<12} | {'SR (%)':<8} | {'P-Loss':<8} | {'C-Loss':<8} | {'KL':<8}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for r in rl_runs:
                date_str = r['timestamp'][:16].replace('T', ' ')
                print(f"{r['id'][:28]:<28} | {date_str:<16} | {r['iterations_completed']:<6} | {r['mean_reward']:<12.2f} | {r['task_success_rate']:<8.1f} | {r['policy_loss']:<8.4f} | {r['critic_loss']:<8.4f} | {r['approx_kl']:<8.4f}")
        else:
            print("   (No hay corridas de RL registradas aún)")

        # 3. EVALUACIONES FORMALES (BENCHMARKING)
        eval_runs = self.get_eval_runs()
        print(f"\n📈 3. EVALUACIONES Y BENCHMARKING ({len(eval_runs)} registrados)")
        if eval_runs:
            header = f"{'ID':<26} | {'Modelo':<12} | {'Demostrador':<14} | {'Eps':<5} | {'SR (%)':<8} | {'Dur(s)':<7} | {'Track Err(mm)':<14} | {'Term Err(mm)':<12}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for r in eval_runs:
                track_mm = r['mean_tracking_error_m'] * 1000.0
                print(f"{r['id'][:26]:<26} | {r['model_type'][:12]:<12} | {r['demonstrator_type'][:14]:<14} | {r['total_episodes']:<5} | {r['success_rate']:<8.1f} | {r['mean_duration_s']:<7.1f} | {track_mm:<14.2f} | {r['terminal_error_mm']:<12.2f}")
        else:
            print("   (No hay evaluaciones formales registradas aún)")

        # 4. DATASETS RECOLECTADOS
        datasets = self.get_datasets()
        print(f"\n📦 4. DATASETS DE DEMOSTRACIONES RECOLECTADOS ({len(datasets)} registrados)")
        if datasets:
            header = f"{'ID':<28} | {'Fecha':<16} | {'Demostrador':<14} | {'Episodios':<10} | {'Done Flag':<10}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for d in datasets:
                date_str = d['timestamp'][:16].replace('T', ' ')
                has_done = "Sí (Dim 9)" if d['has_done_flag'] else "No"
                print(f"{d['id'][:28]:<28} | {date_str:<16} | {d['demonstrator_type'][:14]:<14} | {d['total_episodes']:<10} | {has_done:<10}")
        else:
            print("   (No hay datasets registrados aún)")

        # 5. TELEMETRÍA DE EPISODIOS INDIVIDUALES
        eps_meta = self._query("SELECT COUNT(*) as c, COUNT(DISTINCT dataset_name) as d FROM dataset_episodes")
        ep_count = eps_meta[0]['c'] if eps_meta else 0
        ds_ep_count = eps_meta[0]['d'] if eps_meta else 0
        print(f"\n🔬 5. TELEMETRÍA DE EPISODIOS INDIVIDUALES ({ep_count} episodios en {ds_ep_count} datasets)")
        if ep_count > 0:
            sample_eps = self.get_dataset_episodes(limit=5)
            header = f"{'Dataset':<24} | {'Ep':<4} | {'Spawn (X, Y, Z)':<24} | {'Yaw(rad)':<9} | {'Steps':<6} | {'Sim(s)':<7} | {'Wall(s)':<7}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for e in sample_eps:
                sx = f"{e['obj_spawn_x']:.3f}" if e.get('obj_spawn_x') is not None else "N/A"
                sy = f"{e['obj_spawn_y']:.3f}" if e.get('obj_spawn_y') is not None else "N/A"
                sz = f"{e['obj_spawn_z']:.2f}" if e.get('obj_spawn_z') is not None else "N/A"
                spawn_str = f"({sx}, {sy}, {sz})"
                yaw_str = f"{e['obj_spawn_yaw']:<9.2f}" if e.get('obj_spawn_yaw') is not None else f"{'N/A':<9}"
                steps_val = e.get('steps', 0) or 0
                sim_dur = e.get('sim_duration_s', 0.0) or 0.0
                wall_dur = e.get('wall_duration_s', 0.0) or 0.0
                print(f"{e['dataset_name'][:24]:<24} | {e['episode_idx']:<4} | {spawn_str:<24} | {yaw_str} | {steps_val:<6} | {sim_dur:<7.2f} | {wall_dur:<7.2f}")
        else:
            print("   (No hay episodios individuales registrados aún)")

        print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestor y Visor de Experimentos Científicos")
    parser.add_argument("action", choices=["summary", "list", "episodes"], default="summary", nargs="?")
    parser.add_argument("--dataset", type=str, default=None, help="Filtrar por dataset")
    parser.add_argument("--limit", type=int, default=20, help="Límite de episodios a mostrar")
    parser.add_argument("--db", type=str, default=None, help="Ruta alternativa a experiments.db")
    args = parser.parse_args()

    tracker = ExperimentTracker(db_path=args.db)
    if args.action == "episodes":
        eps = tracker.get_dataset_episodes(dataset_name=args.dataset, limit=args.limit)
        print(f"\n🔬 TELEMETRÍA DE EPISODIOS ({len(eps)} mostrados):")
        header = f"{'Dataset':<24} | {'Ep':<4} | {'Spawn (X, Y, Z)':<24} | {'Yaw(rad)':<9} | {'Steps':<6} | {'Sim(s)':<7} | {'Wall(s)':<7}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for e in eps:
            sx = f"{e['obj_spawn_x']:.3f}" if e.get('obj_spawn_x') is not None else "N/A"
            sy = f"{e['obj_spawn_y']:.3f}" if e.get('obj_spawn_y') is not None else "N/A"
            sz = f"{e['obj_spawn_z']:.2f}" if e.get('obj_spawn_z') is not None else "N/A"
            spawn_str = f"({sx}, {sy}, {sz})"
            yaw_str = f"{e['obj_spawn_yaw']:<9.2f}" if e.get('obj_spawn_yaw') is not None else f"{'N/A':<9}"
            steps_val = e.get('steps', 0) or 0
            sim_dur = e.get('sim_duration_s', 0.0) or 0.0
            wall_dur = e.get('wall_duration_s', 0.0) or 0.0
            print(f"{e['dataset_name'][:24]:<24} | {e['episode_idx']:<4} | {spawn_str:<24} | {yaw_str} | {steps_val:<6} | {sim_dur:<7.2f} | {wall_dur:<7.2f}")
    else:
        tracker.print_summary_table()

