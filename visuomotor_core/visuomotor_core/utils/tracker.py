# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Módulo de registro relacional SQLite Fail-Safe y No Intrusivo.
Garantiza que ningún fallo en la base de datos interrumpa el entrenamiento o ejecución.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def find_db_path() -> Path:
    """Busca o crea la ruta para experiments.db en el root del repositorio."""
    if "VISUOMOTOR_DB_PATH" in os.environ:
        return Path(os.environ["VISUOMOTOR_DB_PATH"])

    # Intentar resolver desde cwd o desde la ubicación de este archivo
    candidate = Path("experiments/experiments.db")
    if candidate.parent.exists() or Path("visuomotor_core").exists():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate.resolve()

    # Fallback relativo a este archivo (visuomotor_ros2/visuomotor_core/...)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    exp_dir = repo_root / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return (exp_dir / "experiments.db").resolve()


def init_db(db_path: Optional[Path] = None) -> None:
    """Crea las tablas si no existen, dentro de un bloque seguro."""
    try:
        path = db_path or find_db_path()
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            # 1. Tabla de entrenamientos supervisados DP
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS train_runs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    dataset_path TEXT,
                    epochs_requested INTEGER,
                    epochs_completed INTEGER,
                    batch_size INTEGER,
                    lr REAL,
                    final_loss REAL,
                    final_loss_ema REAL,
                    vram_peak_mb REAL,
                    duration_s REAL,
                    checkpoint_dir TEXT,
                    metadata_json TEXT
                )
            """)

            # 2. Tabla de entrenamientos por refuerzo (DPPO RL)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_runs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    base_checkpoint TEXT,
                    iterations_completed INTEGER,
                    mean_reward REAL,
                    policy_loss REAL,
                    critic_loss REAL,
                    approx_kl REAL,
                    task_success_rate REAL,
                    duration_s REAL,
                    checkpoint_dir TEXT,
                    metadata_json TEXT
                )
            """)

            # 3. Tabla de evaluaciones y benchmarking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    demonstrator_type TEXT NOT NULL,
                    checkpoint_path TEXT,
                    total_episodes INTEGER,
                    success_rate REAL,
                    mean_duration_s REAL,
                    mean_tracking_error_m REAL,
                    terminal_error_mm REAL,
                    slip_rate REAL,
                    metadata_json TEXT
                )
            """)

            # 4. Tabla de datasets recolectados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    path TEXT NOT NULL,
                    demonstrator_type TEXT NOT NULL,
                    total_episodes INTEGER,
                    duration_s REAL,
                    has_done_flag INTEGER,
                    metadata_json TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        log.warning(f"[DB Fail-Safe] No se pudo inicializar SQLite: {e}")


def record_train_run(
    run_id: str,
    dataset_path: str,
    epochs_requested: int,
    epochs_completed: int,
    batch_size: int,
    lr: float,
    final_loss: float,
    final_loss_ema: float,
    vram_peak_mb: float,
    duration_s: float,
    checkpoint_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Registra una corrida de entrenamiento supervisado de forma 100% fail-safe."""
    try:
        path = find_db_path()
        init_db(path)
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO train_runs (
                    id, timestamp, dataset_path, epochs_requested, epochs_completed,
                    batch_size, lr, final_loss, final_loss_ema, vram_peak_mb,
                    duration_s, checkpoint_dir, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    str(dataset_path),
                    epochs_requested,
                    epochs_completed,
                    batch_size,
                    lr,
                    final_loss,
                    final_loss_ema,
                    vram_peak_mb,
                    duration_s,
                    str(checkpoint_dir),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        log.info(f"💾 [DB Tracker] Entrenamiento registrado exitosamente en {path} (ID: {run_id})")
        return True
    except Exception as e:
        log.warning(f"[DB Fail-Safe] Error silencioso al registrar train_run (no interrumpe): {e}")
        return False


def record_rl_run(
    run_id: str,
    base_checkpoint: str,
    iterations_completed: int,
    mean_reward: float,
    policy_loss: float,
    critic_loss: float,
    approx_kl: float,
    task_success_rate: float,
    duration_s: float,
    checkpoint_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Registra una corrida de fine-tuning RL de forma 100% fail-safe."""
    try:
        path = find_db_path()
        init_db(path)
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO rl_runs (
                    id, timestamp, base_checkpoint, iterations_completed,
                    mean_reward, policy_loss, critic_loss, approx_kl,
                    task_success_rate, duration_s, checkpoint_dir, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    str(base_checkpoint),
                    iterations_completed,
                    mean_reward,
                    policy_loss,
                    critic_loss,
                    approx_kl,
                    task_success_rate,
                    duration_s,
                    str(checkpoint_dir),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        log.info(f"💾 [DB Tracker] Corrida RL registrada exitosamente en {path} (ID: {run_id})")
        return True
    except Exception as e:
        log.warning(f"[DB Fail-Safe] Error silencioso al registrar rl_run (no interrumpe): {e}")
        return False


def record_eval_run(
    run_id: str,
    model_type: str,
    demonstrator_type: str,
    checkpoint_path: str,
    total_episodes: int,
    success_rate: float,
    mean_duration_s: float,
    mean_tracking_error_m: float,
    terminal_error_mm: float,
    slip_rate: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Registra una evaluación de inferencia formal de forma 100% fail-safe."""
    try:
        path = find_db_path()
        init_db(path)
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO eval_runs (
                    id, timestamp, model_type, demonstrator_type, checkpoint_path,
                    total_episodes, success_rate, mean_duration_s,
                    mean_tracking_error_m, terminal_error_mm, slip_rate, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    model_type,
                    demonstrator_type,
                    str(checkpoint_path),
                    total_episodes,
                    success_rate,
                    mean_duration_s,
                    mean_tracking_error_m,
                    terminal_error_mm,
                    slip_rate,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        log.info(f"💾 [DB Tracker] Evaluación registrada exitosamente en {path} (ID: {run_id})")
        return True
    except Exception as e:
        log.warning(f"[DB Fail-Safe] Error silencioso al registrar eval_run (no interrumpe): {e}")
        return False


def record_dataset(
    dataset_id: str,
    path_str: str,
    demonstrator_type: str,
    total_episodes: int,
    duration_s: float,
    has_done_flag: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Registra un dataset creado de forma 100% fail-safe."""
    try:
        path = find_db_path()
        init_db(path)
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO datasets (
                    id, timestamp, path, demonstrator_type, total_episodes,
                    duration_s, has_done_flag, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    dataset_id,
                    datetime.now().isoformat(),
                    str(path_str),
                    demonstrator_type,
                    total_episodes,
                    duration_s,
                    int(has_done_flag),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        log.info(f"💾 [DB Tracker] Dataset registrado exitosamente en {path} (ID: {dataset_id})")
        return True
    except Exception as e:
        log.warning(f"[DB Fail-Safe] Error silencioso al registrar dataset: {e}")
        return False
