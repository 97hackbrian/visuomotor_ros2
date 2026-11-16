# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Módulo Científico de Análisis de Convergencia y Criterios de Parada (Stopping Criteria).
Implementa métodos del Estado del Arte (SOTA) para determinar la convergencia óptima
de políticas generativas visuomotoras (Diffusion Policy) en espacio dual:
Épocas (Macro) y Pasos de Optimización (Micro / Steps).

Métodos implementados:
1. Curvatura Máxima / Codo (Kneedle Algorithm / Maximum Curvature Point)
2. Gradiente Aprox. Cero / Tasa de Variación Nula (Smoothed Gradient Flatness)
3. Estabilización de Varianza en Ventana Móvil (Rolling Loss Relative Standard Deviation - RSD)
4. Tasa de Mejora Relativa con Paciencia (Relative Improvement Rate / Early Stopping)
5. Cociente de Progreso de Prechelt (Prechelt Progress Criterion - PQ)
6. Estabilización de Gap de Media Móvil Exponencial (EMA Loss Gap)
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configurar backend headless para matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log = logging.getLogger(__name__)


# Paleta científica accesible (Colorblind-friendly / Alta Visibilidad)
METHOD_COLORS = {
    "kneedle_elbow": "#d62728",           # Rojo carmesí
    "gradient_flatness": "#2ca02c",       # Verde bosque
    "variance_stabilization": "#9467bd",  # Púrpura sobrio
    "relative_improvement": "#ff7f0e",    # Naranja ámbar
    "prechelt_progress": "#17becf",       # Cian / Turquesa
    "ema_gap": "#8c564b",                 # Marrón oscuro
    "consensus": "#e377c2",               # Rosa magenta
}

METHOD_LABELS = {
    "kneedle_elbow": "Codo / Máx. Curvatura (Kneedle)",
    "gradient_flatness": "Gradiente Aprox. 0 (|dL/dt| < ε)",
    "variance_stabilization": "Estabilización de Varianza (RSD < τ)",
    "relative_improvement": "Mejora Relativa Acotada (ΔL/L < τ)",
    "prechelt_progress": "Cociente de Progreso (Prechelt PQ)",
    "ema_gap": "Estabilización de Gap EMA",
    "consensus": "Consenso Pareto Óptimo",
}


@dataclass
class StoppingRecommendation:
    method_id: str
    method_name: str
    recommended_step: int
    recommended_epoch: float
    loss_at_stop: float
    metric_value: float
    metric_name: str
    threshold: float
    patience: int
    confidence: float
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def smooth_curve_ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Suavizado por Media Móvil Exponencial (EMA)."""
    if len(values) == 0:
        return np.array([])
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def moving_average_edge_padded(arr: np.ndarray, window: int) -> np.ndarray:
    """Calcula media móvil con relleno en bordes para evitar artefactos de decaimiento a cero."""
    if len(arr) == 0 or window <= 1:
        return arr.copy()
    w = min(window, len(arr))
    half = w // 2
    padded = np.pad(arr, (half, w - 1 - half), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(arr)]


def step_to_epoch(step: Union[int, float, np.ndarray], steps_per_epoch: int) -> Union[float, np.ndarray]:
    """
    Conversión exacta y sincrónica de Paso de Optimización a Época continua:
    epoch(t) = step / steps_per_epoch.
    Paso 0 -> Época 0.0 (inicio de entrenamiento).
    Paso N -> Época 1.0 (exactamente 1 época de datos procesados).
    """
    if steps_per_epoch <= 0:
        return 0.0
    res = step / float(steps_per_epoch)
    if isinstance(res, np.ndarray):
        return np.maximum(0.0, res)
    return max(0.0, float(res))


def epoch_to_step(epoch: Union[int, float, np.ndarray], steps_per_epoch: int) -> Union[int, np.ndarray]:
    """
    Conversión exacta y sincrónica de Época a Paso de Optimización:
    step(e) = epoch * steps_per_epoch.
    """
    if steps_per_epoch <= 0:
        return 0
    res = epoch * float(steps_per_epoch)
    if isinstance(res, np.ndarray):
        return np.maximum(0, np.round(res).astype(int))
    return max(0, int(round(res)))


# ==============================================================================
# 1. MÉTODO DEL CODO / CURVATURA MÁXIMA (KNEEDLE ALGORITHM)
# ==============================================================================
class KneedleDetector:
    """
    Detecta el 'codo' (elbow) o punto de inflexión de máxima curvatura en la curva de pérdida.
    Basado en Satopaa et al. (2011), 'Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior'.
    """

    def __init__(self, sensitivity: float = 1.0, ema_alpha: float = 0.08):
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < 5 or len(losses) < 5:
            return None

        y_smooth = smooth_curve_ema(losses, alpha=self.ema_alpha)
        x = steps.astype(np.float64)

        # Normalización min-max al cuadrado unitario [0, 1]
        x_min, x_max = x[0], x[-1]
        y_min, y_max = np.min(y_smooth), np.max(y_smooth)

        if x_max <= x_min or y_max <= y_min:
            return None

        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y_smooth - y_min) / (y_max - y_min)

        # Cuerda diagonal de referencia entre el primer y último punto normalizado
        # Para pérdida decreciente convexa, el codo maximiza la distancia a la cuerda
        y_chord = y_norm[0] + x_norm * (y_norm[-1] - y_norm[0])
        diff = y_chord - y_norm

        knee_idx = int(np.argmax(diff))
        max_diff = float(diff[knee_idx])

        if max_diff <= 0.0:
            return None

        rec_step = int(steps[knee_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(y_smooth[knee_idx])

        confidence = float(np.clip(max_diff * 2.5, 0.4, 0.99))

        return StoppingRecommendation(
            method_id="kneedle_elbow",
            method_name=METHOD_LABELS["kneedle_elbow"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(max_diff, 6),
            metric_name="Distancia Máxima a Cuerda Diagonal (Normalizada)",
            threshold=0.0,
            patience=0,
            confidence=round(confidence, 3),
            justification=(
                f"Punto de máxima curvatura / codo macro (Kneedle). "
                f"A partir del paso {rec_step} (época {rec_epoch:.2f}), la tasa de descenso "
                f"de la pérdida experimenta su transición más aguda hacia la fase de rendimientos decrecientes."
            ),
        )


# ==============================================================================
# 2. GRADIENTE APROXIMADAMENTE CERO (SMOOTHED GRADIENT FLATNESS)
# ==============================================================================
class GradientFlatnessDetector:
    """
    Detecta cuándo la primera derivada dL/dt cae por debajo de un umbral ε
    durante una ventana de paciencia P, indicando entrada en un plateau horizontal.
    Utiliza media móvil con relleno en bordes para evitar artefactos espurios de convolución.
    """

    def __init__(
        self,
        threshold: float = 3.0e-5,
        window_size: int = 15,
        patience: int = 10,
        ema_alpha: float = 0.05,
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.patience = patience
        self.ema_alpha = ema_alpha

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < self.window_size + self.patience:
            return None

        y_smooth = smooth_curve_ema(losses, alpha=self.ema_alpha)
        dy = np.gradient(y_smooth, steps.astype(np.float64))
        abs_dy_smooth = moving_average_edge_padded(np.abs(dy), self.window_size)

        consecutive = 0
        trigger_idx = None

        for idx in range(self.window_size, len(steps)):
            if abs_dy_smooth[idx] <= self.threshold:
                consecutive += 1
                if consecutive >= self.patience:
                    trigger_idx = idx
                    break
            else:
                consecutive = 0

        # Si no se cumple con el umbral estricto, buscar el mínimo de gradiente sostenido
        if trigger_idx is None:
            min_grad_idx = int(np.argmin(abs_dy_smooth[self.window_size:])) + self.window_size
            trigger_idx = min_grad_idx
            is_fallback = True
        else:
            is_fallback = False

        rec_step = int(steps[trigger_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(y_smooth[trigger_idx])
        metric_val = float(abs_dy_smooth[trigger_idx])

        confidence = 0.92 if not is_fallback else 0.65

        if is_fallback:
            justification = (
                f"[Aviso Fallback: Gradiente mínimo local] Pendiente mínima alcanzada de {metric_val:.2e} "
                f"(umbral estricto ε={self.threshold:.2e} no sostenido por {self.patience} pasos)."
            )
        else:
            justification = (
                f"El gradiente temporal de la pérdida se aplanó a {metric_val:.2e} "
                f"(umbral ε={self.threshold:.2e}) sostenido por {self.patience} pasos consecutivos. "
                f"No hay avance direccional significativo adicional."
            )

        return StoppingRecommendation(
            method_id="gradient_flatness",
            method_name=METHOD_LABELS["gradient_flatness"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(metric_val, 8),
            metric_name="|dL/dt| Suavizado (Pendiente Media)",
            threshold=self.threshold,
            patience=self.patience,
            confidence=round(confidence, 3),
            justification=justification,
        )


# ==============================================================================
# 3. ESTABILIZACIÓN DE VARIANZA EN VENTANA MÓVIL (RELATIVE STD DEV - RSD)
# ==============================================================================
class VarianceStabilizationDetector:
    """
    Calcula el Coeficiente de Variación (RSD = σ / μ) sobre una ventana móvil.
    Detecta cuándo las fluctuaciones estocásticas de la política se estabilizan alrededor
    de la cuenca de convergencia con variabilidad relativa menor a un umbral τ_rsd.
    Aplica suavizado EMA previo para desacoplar el ruido estocástico del mini-batch de la
    dinámica de aprendizaje real de la red.
    """

    def __init__(
        self,
        threshold_rsd: float = 0.025,
        window_size: int = 30,
        patience: int = 15,
        ema_alpha: float = 0.05,
        min_loss_ratio: float = 0.5,
    ):
        self.threshold_rsd = threshold_rsd
        self.window_size = window_size
        self.patience = patience
        self.ema_alpha = ema_alpha
        self.min_loss_ratio = min_loss_ratio

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < self.window_size + self.patience:
            return None

        y_smooth = smooth_curve_ema(losses, alpha=self.ema_alpha)
        initial_loss = float(y_smooth[0])
        rsds = np.full(len(steps), np.nan)
        for i in range(self.window_size, len(steps)):
            sub = y_smooth[i - self.window_size : i]
            mean_val = np.mean(sub)
            std_val = np.std(sub)
            if mean_val > 1e-12:
                rsds[i] = std_val / mean_val
            else:
                rsds[i] = 0.0

        consecutive = 0
        trigger_idx = None
        for i in range(self.window_size, len(steps)):
            # Asegurar que se haya superado la fase inicial de pérdida máxima
            has_progressed = (y_smooth[i] <= self.min_loss_ratio * initial_loss) or (i >= len(steps) // 2)
            if not np.isnan(rsds[i]) and rsds[i] <= self.threshold_rsd and has_progressed:
                consecutive += 1
                if consecutive >= self.patience:
                    trigger_idx = i
                    break
            else:
                consecutive = 0

        if trigger_idx is None:
            valid_rsds = np.where(np.isnan(rsds), np.inf, rsds)
            trigger_idx = int(np.argmin(valid_rsds))
            is_fallback = True
        else:
            is_fallback = False

        rec_step = int(steps[trigger_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(y_smooth[trigger_idx])
        rsd_val = float(rsds[trigger_idx]) if not np.isnan(rsds[trigger_idx]) else 0.0

        confidence = 0.88 if not is_fallback else 0.60

        if is_fallback:
            justification = (
                f"[Aviso Fallback: RSD mínima alcanzada] La variabilidad relativa de la curva cayó a "
                f"{rsd_val * 100:.2f}% (umbral estricto τ={self.threshold_rsd * 100:.1f}% no sostenido por {self.patience} pasos)."
            )
        else:
            justification = (
                f"La variabilidad relativa de la pérdida se redujo a {rsd_val * 100:.2f}% "
                f"(umbral τ={self.threshold_rsd * 100:.1f}%) en ventana de {self.window_size} muestras sostenida por {self.patience} pasos. "
                f"Las oscilaciones estocásticas se mantienen estables en el atractor."
            )

        return StoppingRecommendation(
            method_id="variance_stabilization",
            method_name=METHOD_LABELS["variance_stabilization"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(rsd_val, 6),
            metric_name="Desviación Estándar Relativa (RSD = σ/μ)",
            threshold=self.threshold_rsd,
            patience=self.patience,
            confidence=round(confidence, 3),
            justification=justification,
        )


# ==============================================================================
# 4. TASA DE MEJORA RELATIVA CON PACIENCIA (RELATIVE IMPROVEMENT / EARLY STOP)
# ==============================================================================
class RelativeImprovementDetector:
    """
    Evalúa la tasa de mejora relativa frente al mejor valor histórico previo:
    Δ_rel = (L_best_prev - L_curr) / L_best_prev.
    Detiene si la mejora es inferior al umbral τ_improv durante una ventana de paciencia P.
    """

    def __init__(
        self,
        threshold: float = 0.005,  # 0.5% de mejora
        patience: int = 25,
        ema_alpha: float = 0.05,
    ):
        self.threshold = threshold
        self.patience = patience
        self.ema_alpha = ema_alpha

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < self.patience * 2:
            return None

        y_smooth = smooth_curve_ema(losses, alpha=self.ema_alpha)
        best_so_far = np.minimum.accumulate(y_smooth)

        consecutive_stagnant = 0
        trigger_idx = None
        last_improv = 0.0

        for i in range(self.patience, len(steps)):
            prev_ref = best_so_far[i - self.patience]
            curr_best = best_so_far[i]
            rel_improv = (prev_ref - curr_best) / max(prev_ref, 1e-12)

            if rel_improv < self.threshold:
                consecutive_stagnant += 1
                if consecutive_stagnant >= self.patience:
                    trigger_idx = i
                    last_improv = float(rel_improv)
                    break
            else:
                consecutive_stagnant = 0

        if trigger_idx is None:
            trigger_idx = len(steps) - 1
            last_improv = float((best_so_far[max(0, trigger_idx - self.patience)] - best_so_far[trigger_idx]) / max(best_so_far[max(0, trigger_idx - self.patience)], 1e-12))
            is_fallback = True
        else:
            is_fallback = False

        rec_step = int(steps[trigger_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(y_smooth[trigger_idx])

        confidence = 0.90 if not is_fallback else 0.70

        if is_fallback:
            justification = (
                f"[Aviso Fallback: Fin de datos] Mejora relativa final de {last_improv * 100:.3f}% "
                f"(umbral τ={self.threshold * 100:.2f}% evaluado al final del registro)."
            )
        else:
            justification = (
                f"Mejora relativa de sólo {last_improv * 100:.3f}% inferior al umbral de "
                f"{self.threshold * 100:.2f}% sostenida durante {self.patience} pasos de paciencia. "
                f"El modelo ha entrado en la zona de rendimientos decrecientes."
            )

        return StoppingRecommendation(
            method_id="relative_improvement",
            method_name=METHOD_LABELS["relative_improvement"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(last_improv, 6),
            metric_name="Mejora Relativa Acumulada (ΔL / L_best)",
            threshold=self.threshold,
            patience=self.patience,
            confidence=round(confidence, 3),
            justification=justification,
        )


# ==============================================================================
# 5. COCIENTE DE PROGRESO DE PRECHELT (PRECHELT PROGRESS CRITERION - PQ)
# ==============================================================================
class PrecheltProgressDetector:
    """
    Implementa el criterio de progreso formal de Lutz Prechelt (1998):
    P_k(t) = 1000 * ( (sum_{i in U_k} L(i)) / (k * min_{i in U_k} L(i)) - 1 )
    Detiene cuando el progreso medio en la tira de longitud k no justifica continuar el cómputo.
    Opera sobre la pérdida filtrada por EMA para desacoplar el ruido estocástico del mini-batch.
    """

    def __init__(
        self,
        strip_length: int = 20,
        threshold_pq: float = 25.0,
        patience: int = 10,
        ema_alpha: float = 0.05,
    ):
        self.strip_length = strip_length
        self.threshold_pq = threshold_pq
        self.patience = patience
        self.ema_alpha = ema_alpha

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < self.strip_length + self.patience:
            return None

        y_smooth = smooth_curve_ema(losses, alpha=self.ema_alpha)
        pqs = np.full(len(steps), np.nan)

        for i in range(self.strip_length, len(steps)):
            strip = y_smooth[i - self.strip_length : i]
            mean_strip = np.mean(strip)
            min_strip = np.min(strip)
            if min_strip > 1e-12:
                pqs[i] = 1000.0 * ((mean_strip / min_strip) - 1.0)
            else:
                pqs[i] = 0.0

        consecutive = 0
        trigger_idx = None
        for i in range(self.strip_length, len(steps)):
            if not np.isnan(pqs[i]) and pqs[i] <= self.threshold_pq:
                consecutive += 1
                if consecutive >= self.patience:
                    trigger_idx = i
                    break
            else:
                consecutive = 0

        if trigger_idx is None:
            valid_pqs = np.where(np.isnan(pqs), np.inf, pqs)
            trigger_idx = int(np.argmin(valid_pqs))
            is_fallback = True
        else:
            is_fallback = False

        rec_step = int(steps[trigger_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(y_smooth[trigger_idx])
        pq_val = float(pqs[trigger_idx]) if not np.isnan(pqs[trigger_idx]) else 0.0

        confidence = 0.88 if not is_fallback else 0.60

        if is_fallback:
            justification = (
                f"[Aviso Fallback: Progreso mínimo local] Progreso mínimo P_k = {pq_val:.2f} alcanzado "
                f"(umbral estricto {self.threshold_pq:.1f} no sostenido por {self.patience} pasos)."
            )
        else:
            justification = (
                f"El progreso en la tira de {self.strip_length} muestras cayó a P_k = {pq_val:.2f} "
                f"(umbral={self.threshold_pq:.1f}) sostenido por {self.patience} pasos. "
                f"El gasto computacional adicional produce ganancias de pérdida insignificantes."
            )

        return StoppingRecommendation(
            method_id="prechelt_progress",
            method_name=METHOD_LABELS["prechelt_progress"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(pq_val, 4),
            metric_name="Cociente de Progreso P_k (Prechelt)",
            threshold=self.threshold_pq,
            patience=self.patience,
            confidence=round(confidence, 3),
            justification=justification,
        )


# ==============================================================================
# 6. ESTABILIZACIÓN DE GAP DE MEDIA MÓVIL EXPONENCIAL (EMA GAP DETECTOR - SOTA)
# ==============================================================================
class EMAGapDetector:
    """
    Detecta la convergencia asintótica monitoreando la divergencia relativa entre dos
    medias móviles exponenciales con diferentes escalas temporales (EMA rápida vs EMA lenta):
    Gap(t) = |EMA_fast(t) - EMA_slow(t)| / EMA_slow(t).
    Durante el descenso rápido, la EMA rápida aventaja significativamente a la lenta (Gap > 10%).
    En convergencia asintótica, ambas convergen y el Gap relativo cae por debajo de τ_gap (ej. 3.0%).
    """

    def __init__(
        self,
        fast_alpha: float = 0.10,
        slow_alpha: float = 0.02,
        threshold_gap: float = 0.03,  # 3% de gap
        patience: int = 15,
        min_steps: int = 30,
    ):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.threshold_gap = threshold_gap
        self.patience = patience
        self.min_steps = min_steps

    def detect(
        self,
        steps: np.ndarray,
        losses: np.ndarray,
        steps_per_epoch: int,
    ) -> Optional[StoppingRecommendation]:
        if len(steps) < self.min_steps + self.patience:
            return None

        ema_fast = smooth_curve_ema(losses, alpha=self.fast_alpha)
        ema_slow = smooth_curve_ema(losses, alpha=self.slow_alpha)
        gap = np.abs(ema_fast - ema_slow) / np.maximum(ema_slow, 1e-8)

        consecutive = 0
        trigger_idx = None
        for i in range(self.min_steps, len(steps)):
            if gap[i] <= self.threshold_gap:
                consecutive += 1
                if consecutive >= self.patience:
                    trigger_idx = i
                    break
            else:
                consecutive = 0

        if trigger_idx is None:
            trigger_idx = int(np.argmin(gap[self.min_steps:])) + self.min_steps
            is_fallback = True
        else:
            is_fallback = False

        rec_step = int(steps[trigger_idx])
        rec_epoch = float(step_to_epoch(rec_step, steps_per_epoch))
        loss_val = float(ema_fast[trigger_idx])
        gap_val = float(gap[trigger_idx])

        confidence = 0.90 if not is_fallback else 0.65

        if is_fallback:
            justification = (
                f"[Aviso Fallback: Gap EMA mínimo] El gap relativo mínimo alcanzado fue {gap_val * 100:.2f}% "
                f"(umbral estricto τ={self.threshold_gap * 100:.1f}% no sostenido por {self.patience} pasos)."
            )
        else:
            justification = (
                f"El gap relativo entre EMA rápida (α={self.fast_alpha}) y EMA lenta (α={self.slow_alpha}) "
                f"se redujo a {gap_val * 100:.2f}% (umbral τ={self.threshold_gap * 100:.1f}%) "
                f"sostenido por {self.patience} pasos. La dinámica de optimización alcanzó régimen estacionario."
            )

        return StoppingRecommendation(
            method_id="ema_gap",
            method_name=METHOD_LABELS["ema_gap"],
            recommended_step=rec_step,
            recommended_epoch=round(rec_epoch, 2),
            loss_at_stop=round(loss_val, 6),
            metric_value=round(gap_val, 6),
            metric_name="Gap Relativo EMA (|EMA_rápida - EMA_lenta| / EMA_lenta)",
            threshold=self.threshold_gap,
            patience=self.patience,
            confidence=round(confidence, 3),
            justification=justification,
        )


# ==============================================================================
# ORQUESTADOR INTEGRAL: CONVERGENCE ANALYZER
# ==============================================================================
class ConvergenceAnalyzer:
    """
    Orquestador maestro de evaluación de convergencia y criterios de parada.
    Integra y contrasta todos los métodos, calcula métricas en Épocas y Pasos de Optimización,
    genera tablas comparativas formales y renderiza figuras dual-axis de calidad de publicación.
    """

    def __init__(
        self,
        steps_per_epoch: int = 1,
        step_history: Optional[List[Dict[str, Any]]] = None,
        epoch_history: Optional[List[Dict[str, Any]]] = None,
        dataset_name: Optional[str] = None,
    ):
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.step_history = step_history or []
        self.epoch_history = epoch_history or []
        self.dataset_name = dataset_name or "demonstrations.zarr"

        self.kneedle = KneedleDetector()
        self.gradient = GradientFlatnessDetector()
        self.variance = VarianceStabilizationDetector()
        self.rel_improv = RelativeImprovementDetector()
        self.prechelt = PrecheltProgressDetector()
        self.ema_gap = EMAGapDetector()

        self.recommendations: Dict[str, StoppingRecommendation] = {}
        self.consensus_recommendation: Optional[StoppingRecommendation] = None

    @classmethod
    def from_csv_files(
        cls,
        step_csv: Path,
        epoch_csv: Optional[Path] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> "ConvergenceAnalyzer":
        """Instancia el analizador cargando archivos CSV generados por train.py."""
        step_hist = []
        if step_csv.exists():
            import csv
            with open(step_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    step_hist.append({
                        "step": int(r["step"]),
                        "epoch": int(r.get("epoch", 1)),
                        "loss": float(r["loss"]),
                        "loss_ema": float(r.get("loss_ema", r["loss"])),
                        "lr": float(r.get("lr", 1e-4)),
                        "vram_peak_mb": float(r.get("vram_peak_mb", 0.0)),
                    })

        epoch_hist = []
        if epoch_csv and epoch_csv.exists():
            import csv
            with open(epoch_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    epoch_hist.append({
                        "epoch": int(r["epoch"]),
                        "avg_loss": float(r["avg_loss"]),
                        "final_lr": float(r.get("final_lr", 1e-4)),
                        "duration_s": float(r.get("duration_s", 0.0)),
                    })

        # Inferir steps_per_epoch si no se proporciona
        if steps_per_epoch is None:
            # Primero intentar leer training_meta.json si existe en el mismo directorio
            meta_json = step_csv.parent / "training_meta.json"
            if meta_json.exists():
                try:
                    with open(meta_json, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if "steps_per_epoch" in meta and int(meta["steps_per_epoch"]) > 0:
                            steps_per_epoch = int(meta["steps_per_epoch"])
                except Exception:
                    pass

        if steps_per_epoch is None:
            if epoch_hist and step_hist and len(epoch_hist) > 0:
                # Estimar pasos por época: (max_step + 1) / total_epochs con redondeo
                max_step = step_hist[-1]["step"] + 1
                steps_per_epoch = max(1, int(round(max_step / float(len(epoch_hist)))))
            elif step_hist:
                max_ep = max(r.get("epoch", 1) for r in step_hist)
                max_step = step_hist[-1]["step"] + 1
                steps_per_epoch = max(1, int(round(max_step / float(max(1, max_ep)))))
            else:
                steps_per_epoch = 1

        return cls(
            steps_per_epoch=steps_per_epoch,
            step_history=step_hist,
            epoch_history=epoch_hist,
        )

    def analyze(self) -> Dict[str, StoppingRecommendation]:
        """Ejecuta todos los detectores y calcula el consenso asintótico."""
        if not self.step_history:
            log.warning("No hay datos de pasos (step_history) para evaluar convergencia.")
            return {}

        steps = np.array([r["step"] for r in self.step_history])
        losses = np.array([r["loss"] for r in self.step_history])

        self.recommendations = {}

        # 1. Kneedle
        k_res = self.kneedle.detect(steps, losses, self.steps_per_epoch)
        if k_res:
            self.recommendations["kneedle_elbow"] = k_res

        # 2. Gradient Flatness
        g_res = self.gradient.detect(steps, losses, self.steps_per_epoch)
        if g_res:
            self.recommendations["gradient_flatness"] = g_res

        # 3. Variance Stabilization
        v_res = self.variance.detect(steps, losses, self.steps_per_epoch)
        if v_res:
            self.recommendations["variance_stabilization"] = v_res

        # 4. Relative Improvement
        r_res = self.rel_improv.detect(steps, losses, self.steps_per_epoch)
        if r_res:
            self.recommendations["relative_improvement"] = r_res

        # 5. Prechelt Progress
        p_res = self.prechelt.detect(steps, losses, self.steps_per_epoch)
        if p_res:
            self.recommendations["prechelt_progress"] = p_res

        # 6. EMA Gap
        e_res = self.ema_gap.detect(steps, losses, self.steps_per_epoch)
        if e_res:
            self.recommendations["ema_gap"] = e_res

        # Consenso Asintótico Pareto (Mediana ponderada de los métodos de estacionariedad)
        asymptotic_methods = [
            "gradient_flatness",
            "variance_stabilization",
            "relative_improvement",
            "prechelt_progress",
            "ema_gap",
        ]
        asymptotic_recs = [self.recommendations[m] for m in asymptotic_methods if m in self.recommendations]

        if asymptotic_recs:
            rec_steps = [r.recommended_step for r in asymptotic_recs]
            rec_weights = [r.confidence for r in asymptotic_recs]

            sorted_indices = np.argsort(rec_steps)
            sorted_steps = np.array(rec_steps)[sorted_indices]
            sorted_weights = np.array(rec_weights)[sorted_indices]
            cum_weights = np.cumsum(sorted_weights)
            cutoff = cum_weights[-1] / 2.0
            median_idx = int(np.searchsorted(cum_weights, cutoff))
            consensus_step = int(sorted_steps[median_idx])
            consensus_epoch = float(step_to_epoch(consensus_step, self.steps_per_epoch))

            step_diffs = np.abs(steps - consensus_step)
            nearest_idx = int(np.argmin(step_diffs))
            y_smooth_eval = smooth_curve_ema(losses, alpha=0.05)
            loss_at_consensus = float(y_smooth_eval[nearest_idx])

            self.consensus_recommendation = StoppingRecommendation(
                method_id="consensus",
                method_name=METHOD_LABELS["consensus"],
                recommended_step=consensus_step,
                recommended_epoch=round(consensus_epoch, 2),
                loss_at_stop=round(loss_at_consensus, 6),
                metric_value=float(np.mean(rec_steps)),
                metric_name="Mediana Ponderada de Métodos Asintóticos",
                threshold=0.0,
                patience=0,
                confidence=round(float(np.mean(rec_weights)), 3),
                justification=(
                    f"Consenso asintótico Pareto entre {len(asymptotic_recs)} métodos de convergencia fina. "
                    f"Representa la cota óptima donde el aprendizaje de la política visuomotora alcanza estabilidad "
                    f"y mínima varianza direccional (Pérdida ~{loss_at_consensus:.4f})."
                ),
            )
            self.recommendations["consensus"] = self.consensus_recommendation

        return self.recommendations

    def generate_summary_table(self) -> str:
        """Genera una tabla ASCII formal y reproducible con los resultados comparativos."""
        lines = []
        lines.append("=" * 105)
        lines.append("🔬 EVALUACIÓN COMPARATIVA DE CRITERIOS DE PARADA (STOPPING CRITERIA) — VISUOMOTOR DP")
        lines.append(f"⚙️ Configuración: Pasos por Época = {self.steps_per_epoch} | Muestras Analizadas = {len(self.step_history)}")
        lines.append("=" * 105)
        header = f"{'Método Evaluado':<36} | {'Época Rec.':<10} | {'Paso (Step)':<11} | {'Loss (MSE)':<10} | {'Confianza':<9} | {'Criterio / Métrica'}"
        lines.append(header)
        lines.append("-" * 105)

        for method_id, rec in self.recommendations.items():
            ep_str = f"{rec.recommended_epoch:.2f}"
            st_str = f"{rec.recommended_step:,}"
            loss_str = f"{rec.loss_at_stop:.5f}"
            conf_str = f"{rec.confidence * 100:.1f}%"
            val_str = f"{rec.metric_name}: {rec.metric_value}"
            lines.append(f"{rec.method_name:<36} | {ep_str:<10} | {st_str:<11} | {loss_str:<10} | {conf_str:<9} | {val_str}")

        lines.append("=" * 105)
        return "\n".join(lines)

    def save_report(self, output_dir: Path) -> Path:
        """Exporta el reporte analítico en JSON y CSV tabular."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_data = {
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps_analyzed": len(self.step_history),
            "total_epochs_recorded": len(self.epoch_history),
            "recommendations": {k: v.to_dict() for k, v in self.recommendations.items()},
        }

        json_path = output_dir / "convergence_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # CSV Tabular
        import csv
        csv_path = output_dir / "convergence_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method_id",
                "method_name",
                "recommended_epoch",
                "recommended_step",
                "loss_at_stop",
                "metric_name",
                "metric_value",
                "threshold",
                "patience",
                "confidence",
                "justification",
            ])
            for rec in self.recommendations.values():
                writer.writerow([
                    rec.method_id,
                    rec.method_name,
                    rec.recommended_epoch,
                    rec.recommended_step,
                    rec.loss_at_stop,
                    rec.metric_name,
                    rec.metric_value,
                    rec.threshold,
                    rec.patience,
                    rec.confidence,
                    rec.justification,
                ])

        log.info(f"[Convergencia] Reporte guardado en {json_path} y {csv_path}")
        return json_path

    def plot_dual_axis(
        self,
        output_dir: Path,
        dpi: int = 300,
        show_diagnostics: bool = True,
        dataset_name: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Genera la figura de publicación con Doble Eje X Sincrónico:
        Eje Inferior: Pasos de Optimización (Steps)
        Eje Superior: Épocas de Entrenamiento (Epochs)
        Incluye marcadores verticales explícitos y callouts para cada criterio evaluado.
        Exporta versiones PNG y PDF vectorial sin dependencias de display X11.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.step_history:
            log.warning("No hay datos para graficar convergencia.")
            return Path(), Path()

        steps = np.array([r["step"] for r in self.step_history])
        losses = np.array([r["loss"] for r in self.step_history])
        if self.step_history and "loss_ema" in self.step_history[0]:
            ema_losses = np.array([r.get("loss_ema", losses[i]) for i, r in enumerate(self.step_history)])
        else:
            ema_losses = smooth_curve_ema(losses)

        # ----------------------------------------------------------------------
        # FIGURA 1: FIGURA CANÓNICA DE PUBLICACIÓN DUAL-AXIS (SOLO CURVA + STOPPING)
        # ----------------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(13, 7))

        # Eje Inferior (Steps)
        p1 = ax1.plot(steps, losses, color="#1f77b4", alpha=0.25, linewidth=1.2, label="Pérdida Cruda (Batch MSE)")
        p2 = ax1.plot(steps, ema_losses, color="#0b3c5d", linewidth=2.4, label="Pérdida Filtrada EMA (Suavizado α=0.05)")

        ax1.set_xlabel("Paso de Optimización (Optimization Step)", fontsize=12, fontweight="bold", labelpad=8)
        ax1.set_ylabel("Pérdida de Difusión (MSE Loss)", fontsize=12, fontweight="bold", labelpad=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Configurar Eje Superior (Epochs) mediante transformación sincrónica
        f_step_to_epoch = lambda s: step_to_epoch(s, self.steps_per_epoch)
        f_epoch_to_step = lambda e: epoch_to_step(e, self.steps_per_epoch)
        secax = ax1.secondary_xaxis("top", functions=(f_step_to_epoch, f_epoch_to_step))
        secax.set_xlabel(f"Época de Entrenamiento (1 Época = {self.steps_per_epoch:,} Pasos)", fontsize=12, fontweight="bold", labelpad=8)

        # Marcadores verticales de cada criterio evaluado
        marker_handles = []
        y_min_plot, y_max_plot = np.min(losses), np.max(losses)
        span_y = y_max_plot - y_min_plot

        # Ordenar recomendaciones por paso para acomodar anotaciones sin colisiones
        sorted_recs = sorted(
            [r for r in self.recommendations.values() if r.method_id != "consensus"],
            key=lambda r: r.recommended_step,
        )

        # Añadir consenso al final para dibujarlo con mayor realce
        if self.consensus_recommendation:
            sorted_recs.append(self.consensus_recommendation)

        y_text_offsets = np.linspace(y_max_plot - span_y * 0.15, y_max_plot - span_y * 0.55, len(sorted_recs))

        for idx, rec in enumerate(sorted_recs):
            color = METHOD_COLORS.get(rec.method_id, "#333333")
            is_cons = (rec.method_id == "consensus")
            lw = 2.8 if is_cons else 1.8
            ls = "-" if is_cons else "--"
            alpha = 0.95 if is_cons else 0.85

            line = ax1.axvline(
                x=rec.recommended_step,
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                label=f"{rec.method_name}: Ép. {rec.recommended_epoch:.2f} (Paso {rec.recommended_step:,})",
            )
            marker_handles.append(line)

            # Punto de detención exacto
            ax1.scatter(
                [rec.recommended_step],
                [rec.loss_at_stop],
                color=color,
                edgecolors="black",
                s=90 if is_cons else 65,
                zorder=6,
            )

            # Anotación callout inteligente
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.92)
            txt = f"Época {rec.recommended_epoch:.1f} (Paso {rec.recommended_step:,})\nL={rec.loss_at_stop:.4f}"
            
            # Posicionamiento alternado para evitar solapamientos
            x_offset = 15 if idx % 2 == 0 else -110
            y_pos = y_text_offsets[idx] if len(sorted_recs) > 1 else y_max_plot - span_y * 0.25

            ax1.annotate(
                txt,
                xy=(rec.recommended_step, rec.loss_at_stop),
                xytext=(rec.recommended_step + x_offset * max(1, len(steps)//200), y_pos),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8),
                fontsize=8.5,
                fontweight="bold" if is_cons else "normal",
                bbox=bbox_props,
                zorder=7,
            )

        ds_label = dataset_name or self.dataset_name or "demonstrations.zarr"

        ax1.set_title(
            "Determinación de Convergencia y Evaluación de Criterios de Parada\n"
            f"Diffusion Policy (DDIM Schedule + ResNet-18) sobre {ds_label}",
            fontsize=14,
            fontweight="bold",
            pad=18,
        )

        ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95, edgecolor="gray")
        plt.tight_layout()

        canonical_png = output_dir / "convergence_stopping_criteria_dual_axis.png"
        canonical_pdf = output_dir / "convergence_stopping_criteria_dual_axis.pdf"
        plt.savefig(canonical_png, dpi=dpi, bbox_inches="tight")
        plt.savefig(canonical_pdf, bbox_inches="tight")
        plt.close(fig)

        # ----------------------------------------------------------------------
        # FIGURA 2: PANEL DIAGNÓSTICO MULTI-CANAL (2x2) DE CRITERIOS MATEMÁTICOS
        # ----------------------------------------------------------------------
        if show_diagnostics:
            fig_diag, axs = plt.subplots(2, 2, figsize=(15, 11))
            fig_diag.suptitle(
                f"Dinámica de Métricas Matemáticas de Convergencia y Regímenes de Parada\n"
                f"Diffusion Policy (DDIM + ResNet-18) sobre {ds_label}",
                fontsize=15,
                fontweight="bold",
            )

            # Subplot 1: Curva de Pérdida con Dual Axis y Bandas de Parada
            axs[0, 0].plot(steps, losses, color="#1f77b4", alpha=0.25, label="Loss Crudo")
            axs[0, 0].plot(steps, ema_losses, color="#d62728", linewidth=2.0, label="Loss EMA")
            for rec in self.recommendations.values():
                c = METHOD_COLORS.get(rec.method_id, "#333")
                axs[0, 0].axvline(rec.recommended_step, color=c, linestyle="--", alpha=0.7)
            axs[0, 0].set_title("A) Evolución de Pérdida y Puntos de Detención", fontweight="bold")
            axs[0, 0].set_xlabel("Paso de Optimización (Step)")
            axs[0, 0].set_ylabel("Loss (MSE)")
            axs[0, 0].grid(True, linestyle="--", alpha=0.5)
            axs[0, 0].legend(loc="upper right", fontsize=8.5)

            if self.steps_per_epoch > 0:
                secax_diag = axs[0, 0].secondary_xaxis(
                    "top",
                    functions=(
                        lambda s: step_to_epoch(s, self.steps_per_epoch),
                        lambda e: epoch_to_step(e, self.steps_per_epoch),
                    ),
                )
                secax_diag.set_xlabel(f"Época (1 Época = {self.steps_per_epoch:,} Pasos)", fontsize=9, fontweight="bold")

            # Subplot 2: Magnitud de Gradiente Suavizado |dL/dt|
            dy = np.gradient(smooth_curve_ema(losses, 0.05), steps.astype(np.float64))
            abs_dy = moving_average_edge_padded(np.abs(dy), 15)
            axs[0, 1].plot(steps, abs_dy, color="#2ca02c", linewidth=1.8, label="|dL/dt| Suavizado")
            if "gradient_flatness" in self.recommendations:
                g_rec = self.recommendations["gradient_flatness"]
                axs[0, 1].axhline(g_rec.threshold, color="red", linestyle=":", label=f"Umbral ε={g_rec.threshold:.2e}")
                axs[0, 1].axvline(g_rec.recommended_step, color="#2ca02c", linestyle="--", label=f"Stop Paso {g_rec.recommended_step}")
            axs[0, 1].set_yscale("log")
            axs[0, 1].set_title("B) Tasa de Variación / Gradiente Temporal (|dL/dt|)", fontweight="bold")
            axs[0, 1].set_xlabel("Paso de Optimización (Step)")
            axs[0, 1].set_ylabel("Pendiente Absoluta (Escala Log)")
            axs[0, 1].grid(True, linestyle="--", alpha=0.5)
            axs[0, 1].legend(loc="upper right", fontsize=8.5)

            # Subplot 3: Desviación Estándar Relativa (RSD = σ/μ)
            win_size = 30
            y_s_diag = smooth_curve_ema(losses, 0.05)
            rsds = np.full(len(steps), np.nan)
            for i in range(win_size, len(steps)):
                sub = y_s_diag[i - win_size : i]
                m = np.mean(sub)
                if m > 1e-12:
                    rsds[i] = np.std(sub) / m
            axs[1, 0].plot(steps, rsds * 100.0, color="#9467bd", linewidth=1.8, label="RSD Móvil (%)")
            if "variance_stabilization" in self.recommendations:
                v_rec = self.recommendations["variance_stabilization"]
                axs[1, 0].axhline(v_rec.threshold * 100.0, color="red", linestyle=":", label=f"Umbral τ={v_rec.threshold*100:.1f}%")
                axs[1, 0].axvline(v_rec.recommended_step, color="#9467bd", linestyle="--", label=f"Stop Paso {v_rec.recommended_step}")
            axs[1, 0].set_title("C) Estabilización de Varianza Estocástica (RSD)", fontweight="bold")
            axs[1, 0].set_xlabel("Paso de Optimización (Step)")
            axs[1, 0].set_ylabel("RSD Relativo (%)")
            axs[1, 0].grid(True, linestyle="--", alpha=0.5)
            axs[1, 0].legend(loc="upper right", fontsize=8.5)

            # Subplot 4: Dinámica Macro por Época
            if self.epoch_history:
                eps = [r["epoch"] for r in self.epoch_history]
                ep_losses = [r["avg_loss"] for r in self.epoch_history]
                axs[1, 1].plot(eps, ep_losses, color="#ff7f0e", marker="o", markersize=4, linewidth=2.0, label="Pérdida Media / Época")
                for rec in self.recommendations.values():
                    c = METHOD_COLORS.get(rec.method_id, "#333")
                    axs[1, 1].axvline(rec.recommended_epoch, color=c, linestyle="--", alpha=0.7)
                axs[1, 1].set_title("D) Macro Convergencia por Época Completa", fontweight="bold")
                axs[1, 1].set_xlabel("Época")
                axs[1, 1].set_ylabel("Pérdida Promedio (MSE)")
                axs[1, 1].grid(True, linestyle="--", alpha=0.5)
                axs[1, 1].legend(loc="upper right", fontsize=8.5)
            else:
                axs[1, 1].text(0.5, 0.5, "Historial de épocas no disponible", ha="center", va="center")
                axs[1, 1].set_title("D) Macro Convergencia por Época", fontweight="bold")

            plt.tight_layout()
            diag_png = output_dir / "convergence_diagnostics_4panel.png"
            diag_pdf = output_dir / "convergence_diagnostics_4panel.pdf"
            plt.savefig(diag_png, dpi=dpi, bbox_inches="tight")
            plt.savefig(diag_pdf, bbox_inches="tight")
            plt.close(fig_diag)

        log.info(f"[Gráficas] Figuras de convergencia exportadas exitosamente en {output_dir}")
        return canonical_png, canonical_pdf
