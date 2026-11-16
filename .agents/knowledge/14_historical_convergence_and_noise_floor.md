# 14. Dinámica Histórica de Convergencia, Piso de Ruido Asintótico y Graficado Parametrizado

Este módulo documenta el estudio analítico de convergencia retrospectiva sobre el modelo maestro `train_500_dataV7_fixed`, la metodología de digitalización sub-pixel de series temporales, el comportamiento físico del piso irreducible de ruido (Noise Floor) en Diffusion Policy, y la directiva obligatoria de graficado agnóstico con doble eje temporal sincrónico.

---

## 1. El Modelo Maestro `train_500_dataV7_fixed`

Entrenado el 13 de octubre de 2026 sobre el dataset `demonstrations_v7_done_fixed.zarr` (667 episodios autónomos balanceados, $415,908$ fotogramas con horizonte $H_{\text{act}} = 16$).
* **Volumen:** 100 épocas completas, 700,000 pasos de optimización con batch size 64 ($N_{\text{steps\_per\_epoch}} = 7,000$).
* **Firma Criptográfica Inviolable:** `sha256 = b25e4ceb88039f97226ff4646c3fea673ca16d5ca2246728f8232f2e27314cb3`.
* **Registro Gráfico Original:** `losses/loss_2026_10_13_05_21_57.png`.

---

## 2. Metodología de Digitalización Sub-Pixel

Dado que las versiones tempranas de `train.py` no registraban archivos tabulares CSV, se implementó una reconstrucción analítica precisa a partir del renderizado original:
1. **Detección Geométrica de Espinas (Spines):**
   - Marco del gráfico identificado en píxeles: $X \in [182, 1670]$, $Y \in [92, 1201]$.
2. **Mapeo Lineal Calibrado:**
   - **Eje Y (Loss MSE):** Ticks detectados en $Y \in [94, 270, 446, 623, 799, 975, 1151]$ ($176\text{ px}$ por cada $0.20$ de pérdida $\implies 880\text{ px/unidad}$).
     $$\mathcal{L} = \frac{1151.0 - Y}{880.0}$$
   - **Eje X (Pasos de Optimización):** Ticks detectados cada $194\text{ px}$ por cada $100,000$ pasos entre $X=250$ (paso 0) y $X=1608$ (paso 700,000):
     $$\text{step} = (X - 250) \times \frac{700000.0}{1358.0}$$
3. **Muestreo:** Se extrajeron 1,356 puntos de alta resolución libres del cuadro de leyenda, alimentando formalmente `training_metrics.csv` y `epoch_metrics.csv`.

---

## 3. Dinámica Numérica de Convergencia

Aplicando la suite multidetector de `visuomotor_core.analysis.stopping_criteria`:

| Hito / Criterio | Época | Paso | Loss MSE | Interpretación Científica |
| :--- | :---: | :---: | :---: | :--- |
| **Inicio de Entrenamiento** | 0.00 | 0 | 1.148 | Inicio de difusión con ruido Gaussiano puro $\mathcal{N}(\mathbf{0}, \mathbf{I})$. |
| **Codo Kneedle** | **4.49** | 31,443 | 0.02187 | Punto de máxima curvatura matemática. Fin de aproximación macro. |
| **Zona de Rendimientos Decrecientes** | **17.01** | 119,072 | 0.01692 | Mejora relativa $\Delta\mathcal{L}/\mathcal{L} < 0.5\%$ en ventana sostenida. |
| **Consenso Asintótico Pareto** | **37.33** | 261,340 | 0.00828 | Estabilización de varianza $RSD = 2.02\% < 2.5\%$. **Punto óptimo de parada.** |
| **Meseta de Saturación** | 50.00 | 350,000 | 0.00536 | Ganancia residual $\le 1.4 \times 10^{-3}$ en las siguientes 50 épocas. |
| **Pérdida Mínima Promedio** | **100.00** | — | **0.00398** | Menor pérdida media por época completa en toda la historia. |
| **Pérdida Mínima Absoluta** | **99.78** | **698,453** | **0.00341** | Mínimo puntual absoluto alcanzado en el entrenamiento. |

---

## 4. El Fenómeno del Piso Irreducible de Ruido (Noise Floor)

Entre las épocas 75 y 100, la pérdida se mantuvo confinada en la banda $[0.00398, 0.00539]$ MSE:
* **Naturaleza Estocástica:** En Diffusion Policy con DDIM, la red predice el ruido $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t, \mathbf{c})$. Dado que el ruido inyectado $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ y el timestep $k \sim \mathcal{U}(0, K-1)$ varían aleatoriamente por mini-batch, existe una varianza mínima irreductible por debajo de la cual el optimizador no puede descender.
* **Micro-Rebotes:** En la época 97 se registró un pico transitorio a $0.00662$ (con un paso a $0.00966$), producto puramente de la estocasticidad del muestreo de timesteps difíciles, retornando de inmediato a $0.00398$ en la época 100. Estos rebotes no denotan inestabilidad ni divergencia.

---

## 5. Reglas Inviolables de Graficado y Visualización

1. **Parámetro Dinámico `dataset_name`:**
   - Prohibición estricta de cadenas de caracteres fijadas (hardcoded) como `demonstrations_v8.zarr` en scripts de análisis o graficado (`stopping_criteria.py`, `train.py`).
   - Toda función receptora debe exponer `dataset_name: Optional[str] = None` y reflejar el dataset verificado en el título y subtítulo de la figura.
2. **Doble Eje X Sincrónico:**
   - Eje Inferior: Pasos de Optimización (Steps).
   - Eje Superior: Épocas de Entrenamiento ($\text{epoch} = \text{step} / N_{\text{steps\_per\_epoch}}$).
3. **Inviolabilidad de Checkpoints:**
   - Todas las tareas de digitalización, re-ploteo y benchmarking operan en modo de solo lectura sobre `model.safetensors` y los directorios `checkpoints/`.
