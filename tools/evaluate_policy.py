#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Evaluador estandarizado de políticas visuomotoras (Solo DP y DP + RL).
Orquesta N episodios automatizados de evaluación con semillas fijas, registrando
telemetría en CSV y actualizando la base de datos de experimentos SQLite para papers.
"""

import argparse
import sys
import time
from pathlib import Path

# Añadir tools y visuomotor_core al path para imports locales
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "visuomotor_core"))
from scientific_logger import ScientificLogger
from experiment_tracker import ExperimentTracker
from visuomotor_core.utils.tracker import record_eval_run

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float32, Bool
    import tf2_ros
except ImportError:
    # Permitir que el módulo cargue en entornos de análisis offline
    rclpy = None


class PolicyEvaluatorNode:
    """
    Nodo de evaluación en vivo para ROS 2 Jazzy.
    """
    def __init__(
        self,
        exp_id: str,
        model_type: str,
        demonstrator_type: str,
        num_episodes: int = 100,
        max_episode_duration_s: float = 60.0,
    ):
        self.exp_id = exp_id
        self.model_type = model_type
        self.demonstrator_type = demonstrator_type
        self.num_episodes = num_episodes
        self.max_episode_duration_s = max_episode_duration_s

        self.logger = ScientificLogger(
            experiment_id=exp_id,
            dataset_demonstrator=demonstrator_type,
            learning_paradigm=model_type,
        )
        self.tracker = ExperimentTracker()
        self.tracker.register_experiment(
            exp_id=exp_id,
            name=f"Evaluación Formal {model_type} ({num_episodes} eps)",
            model_type=model_type,
            demonstrator_type=demonstrator_type,
        )

    def run_benchmark(self) -> None:
        """
        Ejecuta el protocolo de evaluación sistemático.
        """
        print(f"🚀 Iniciando Campaña de Evaluación: {self.exp_id}")
        print(f"   • Paradigma: {self.model_type} | Demostrador Base: {self.demonstrator_type}")
        print(f"   • Episodios Planificados: {self.num_episodes}")
        print(f"   • Directorio de Telemetría: {self.logger.exp_dir}")

        # Hook inicial a SQLite (Fail-Safe)
        try:
            record_eval_run(
                run_id=self.exp_id,
                model_type=self.model_type,
                demonstrator_type=self.demonstrator_type,
                checkpoint_path=str(self.logger.exp_dir),
                total_episodes=self.num_episodes,
                success_rate=0.0,
                mean_duration_s=0.0,
                mean_tracking_error_m=0.0,
                terminal_error_mm=0.0,
            )
        except Exception as e:
            print(f"[DB Fail-Safe] Error no crítico al registrar inicio de evaluación: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluador Automatizado de Políticas Visuomotoras")
    parser.add_argument("--exp_id", type=str, required=True, help="Identificador único del experimento (ej. exp_eval_v8_dp_baseline)")
    parser.add_argument("--model_type", type=str, choices=["SOLO_DP", "DP_PLUS_RL"], default="SOLO_DP")
    parser.add_argument("--demonstrator_type", type=str, choices=["PID_PILOT", "MPC_CUROBO", "CARTESIAN_IMPEDANCE"], default="PID_PILOT")
    parser.add_argument("--num_episodes", type=int, default=100, help="Número de episodios de evaluación")
    args = parser.parse_args()

    evaluator = PolicyEvaluatorNode(
        exp_id=args.exp_id,
        model_type=args.model_type,
        demonstrator_type=args.demonstrator_type,
        num_episodes=args.num_episodes,
    )
    evaluator.run_benchmark()


if __name__ == "__main__":
    main()
