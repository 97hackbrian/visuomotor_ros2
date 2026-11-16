# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Módulo de análisis de convergencia y criterios de parada para políticas visuomotoras.
"""

from visuomotor_core.analysis.stopping_criteria import (
    ConvergenceAnalyzer,
    KneedleDetector,
    GradientFlatnessDetector,
    VarianceStabilizationDetector,
    RelativeImprovementDetector,
    PrecheltProgressDetector,
    EMAGapDetector,
)

__all__ = [
    "ConvergenceAnalyzer",
    "KneedleDetector",
    "GradientFlatnessDetector",
    "VarianceStabilizationDetector",
    "RelativeImprovementDetector",
    "PrecheltProgressDetector",
    "EMAGapDetector",
]
