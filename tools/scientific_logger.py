#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Módulo de registro científico de telemetría y métricas para experimentos de robótica visuomotora.
Exporta series temporales a CSV con metadatos JSON completos para papers de investigación.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ScientificLogger:
    """
    Logger científico para registrar series temporales a nivel de paso (20 Hz)
    y resúmenes consolidados a nivel de episodio y experimento.
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: str = "experiments",
        dataset_demonstrator: str = "PID_PILOT",
        learning_paradigm: str = "SOLO_DP",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_id = experiment_id
        self.dataset_demonstrator = dataset_demonstrator
        self.learning_paradigm = learning_paradigm

        # Directorio específico del experimento
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(output_dir) / f"{timestamp_str}_{experiment_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Archivo de metadatos
        self.metadata = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "dataset_demonstrator": dataset_demonstrator,
            "learning_paradigm": learning_paradigm,
            **(metadata or {}),
        }
        self._save_metadata()

        # Configurar CSV de pasos (step_metrics.csv)
        self.step_csv_path = self.exp_dir / "step_metrics.csv"
        self.step_fields = [
            "timestamp",
            "experiment_id",
            "dataset_demonstrator",
            "learning_paradigm",
            "episode_idx",
            "step_idx",
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "cmd_x",
            "cmd_y",
            "cmd_z",
            "tracking_error_m",
            "obj_x",
            "obj_y",
            "obj_z",
            "is_clamped",
            "done_flag_pred",
            "contact_force_est",
            "reward_instant",
        ]
        self._init_csv(self.step_csv_path, self.step_fields)

        # Configurar CSV de episodios (episode_summary.csv)
        self.episode_csv_path = self.exp_dir / "episode_summary.csv"
        self.episode_fields = [
            "timestamp",
            "experiment_id",
            "episode_idx",
            "task_completed",
            "duration_s",
            "total_steps",
            "total_reward",
            "mean_tracking_error_m",
            "terminal_error_mm",
            "grasp_successful",
            "slip_occurred",
            "done_flag_final",
            "termination_reason",
        ]
        self._init_csv(self.episode_csv_path, self.episode_fields)

    def _init_csv(self, path: Path, fields: List[str]) -> None:
        if not path.exists():
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fields)

    def _save_metadata(self) -> None:
        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def log_step(
        self,
        episode_idx: int,
        step_idx: int,
        tcp_pos: List[float],
        cmd_pos: List[float],
        tracking_error_m: float,
        obj_pos: List[float],
        is_clamped: bool,
        done_flag_pred: float,
        contact_force_est: float = 0.0,
        reward_instant: float = 0.0,
    ) -> None:
        """Registra un paso individual a alta frecuencia."""
        row = [
            datetime.now().isoformat(),
            self.experiment_id,
            self.dataset_demonstrator,
            self.learning_paradigm,
            episode_idx,
            step_idx,
            round(tcp_pos[0], 5),
            round(tcp_pos[1], 5),
            round(tcp_pos[2], 5),
            round(cmd_pos[0], 5),
            round(cmd_pos[1], 5),
            round(cmd_pos[2], 5),
            round(tracking_error_m, 6),
            round(obj_pos[0], 5),
            round(obj_pos[1], 5),
            round(obj_pos[2], 5),
            int(is_clamped),
            round(done_flag_pred, 5),
            round(contact_force_est, 6),
            round(reward_instant, 4),
        ]
        with open(self.step_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_episode(
        self,
        episode_idx: int,
        task_completed: bool,
        duration_s: float,
        total_steps: int,
        total_reward: float,
        mean_tracking_error_m: float,
        terminal_error_mm: float,
        grasp_successful: bool,
        slip_occurred: bool,
        done_flag_final: float,
        termination_reason: str = "NORMAL",
    ) -> None:
        """Registra el resumen de un episodio completo."""
        row = [
            datetime.now().isoformat(),
            self.experiment_id,
            episode_idx,
            int(task_completed),
            round(duration_s, 2),
            total_steps,
            round(total_reward, 3),
            round(mean_tracking_error_m, 6),
            round(terminal_error_mm, 2),
            int(grasp_successful),
            int(slip_occurred),
            round(done_flag_final, 5),
            termination_reason,
        ]
        with open(self.episode_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
