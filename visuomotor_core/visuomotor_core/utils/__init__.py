"""
Utilidades científicas y de profiling para visuomotor_core (Off-ROS).
"""

from visuomotor_core.utils.logger import benchmark, configure_logging
from visuomotor_core.utils.tracker import (
    record_dataset,
    record_eval_run,
    record_rl_run,
    record_train_run,
)

__all__ = [
    "benchmark",
    "configure_logging",
    "record_train_run",
    "record_rl_run",
    "record_eval_run",
    "record_dataset",
]
