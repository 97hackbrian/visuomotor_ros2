# Módulo de Conocimiento: Determinación de Convergencia y Criterios de Parada SOTA (Stopping Criteria)

Este documento sintetiza la teoría formal, formulaciones matemáticas, trampas de implementación y directrices experimentales para determinar la convergencia óptima en el entrenamiento supervisado de **Políticas Visuomotoras Generativas (Diffusion Policy, DDIM Schedule)**.

---

## 1. El Dilema Fundamental: Ruido de Mini-Batch vs. Dinámica de Aprendizaje

En políticas de difusión supervisadas con red DDIM + ResNet-18 U-Net:
1. En cada paso de optimización, el generador estocástico muestrea un paso temporal discreto $k \sim \mathcal{U}(0, K-1)$ y ruido gaussiano $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. Esto introduce una varianza intrínseca inter-batch de $\sigma / \mu \approx 10\% - 25\%$, incluso cuando la red ha alcanzado convergencia total en sus pesos sinápticos.
3. **Regla de Oro:** Evaluar criterios de parada como Prechelt ($P_k$) o Varianza Relativa ($RSD$) directamente sobre la pérdida cruda de cada mini-batch (`batch MSE`) causará que los detectores **NUNCA alcancen sus umbrales** y caigan invariablemente en ramas de *fallback*, arrojando paradas falsas y prematuras en pérdidas elevadas ($0.08 - 0.12$).
4. **Solución:** Los detectores asintóticos deben operar obligatoriamente sobre la trayectoria suavizada por Media Móvil Exponencial ($\text{EMA}, \alpha \approx 0.05$) o sobre la pérdida agregada por época.

---

## 2. Los Dos Regímenes de Convergencia en Robótica Manipuladora

El entrenamiento supervisado de Diffusion Policy no posee un único punto de parada, sino dos regímenes con significados físicos y operativos distintos:

### Régimen 1: Inflexión de Máxima Curvatura / Codo (Kneedle Elbow)
* **Definición Matemática:** Punto que maximiza la distancia perpendicular entre la curva normalizada $\bar{\mathcal{L}}(t)$ y la cuerda secante unitaria (Satopaa et al., 2011).
* **Física Robótica:** Marca la finalización de la fase de descenso abrupto (aprendizaje de macro-trayectorias cartesianas y aproximación global hacia el objeto).
* **Ubicación Típica:** Época $\sim 2.60$ (Paso 390 en $E=20, S=150$), con Pérdida $\sim 0.134$.
* **Aplicación:** Ideal para exploración rápida de hiperparámetros (*early prototyping*) o cuando el presupuesto computacional es severamente restringido.

### Régimen 2: Estacionariedad Asintótica / Convergencia Fina (Fine Manipulation Basin)
* **Definición Matemática:** Cuenca donde la derivada temporal $|d\mathcal{L}/dt| \le 3 \times 10^{-5}$, la varianza relativa $RSD \le 2.5\%$, el progreso Prechelt $P_k \le 25$, la mejora relativa $\Delta \mathcal{L} / \mathcal{L} \le 0.5\%$, y el gap entre EMA rápida y lenta cae a $< 0.1\%$.
* **Física Robótica:** Adquisición de la cinemática de precisión milimétrica, sincronización del gripper durante el contacto fino (`HOLD_TARGET`) y supresión de jitter.
* **Fenómeno de Rebote en Piso Asintótico ($\mathcal{L} \sim 0.001 - 0.005$):**
  - Evidenciado en el modelo maestro `train_500_dataV7_fixed` (700,000 pasos de optimización a lo largo de 100 épocas de dataset completo):
    La red desciende hasta un piso plano donde oscila continuamente entre $\approx 0.0010$ y $\approx 0.0050$.
  - Intentar forzar descensos por debajo de $10^{-3}$ resulta fútil debido al ruido intrínseco del schedule de difusión ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$) y denota sobreentrenamiento en modos cinemáticos rígidos.
* **Ubicación Típica:**
  - En entrenamientos completos de gran escala (1000 eps, 4,279 batches/época), la meseta se alcanza de forma consolidada entre los pasos 50,000 y 200,000.
* **Aplicación:** Checkpoints de producción obligatorios para despliegue físico o simulación cerrada (*closed-loop rollouts*).

---

## 3. Formulación de los 6 Métodos SOTA Implementados

| Método | ID | Formulación Matemática | Umbral Canónico | Paciencia |
| :--- | :--- | :--- | :--- | :--- |
| **Codo Kneedle** | `kneedle_elbow` | $\arg\max_t (y_{\text{chord}}(t) - y_{\text{norm}}(t))$ | $\Delta y_{\text{chord}} > 0$ | $P = 0$ |
| **Gradiente Aprox. 0** | `gradient_flatness` | $\|d\mathcal{L}_{\text{EMA}}/dt\| \le \epsilon$ con *edge padding* | $\epsilon = 3.0 \times 10^{-5}$ | $P = 10$ |
| **Varianza Móvil (RSD)** | `variance_stabilization` | $\sigma_{\text{EMA}}(W) / \mu_{\text{EMA}}(W) \le \tau_{\text{rsd}}$ | $\tau_{\text{rsd}} = 0.025$ ($2.5\%$) | $P = 15$ |
| **Mejora Relativa** | `relative_improvement` | $(\mathcal{L}_{\min, t-P} - \mathcal{L}_{\min, t}) / \mathcal{L}_{\min, t-P} < \tau$ | $\tau = 0.005$ ($0.5\%$) | $P = 25$ |
| **Progreso Prechelt ($P_k$)** | `prechelt_progress` | $1000 \cdot (\bar{\mathcal{L}}_k / \min_k \mathcal{L} - 1) \le \tau_{\text{pq}}$ | $\tau_{\text{pq}} = 25.0$ ($2.5\%$) | $P = 10$ |
| **Gap EMA (SOTA)** | `ema_gap` | $\|\text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}\| / \text{EMA}_{\text{slow}} \le \tau_{\text{gap}}$ | $\tau_{\text{gap}} = 0.03$ ($3.0\%$) | $P = 15$ |
| **Consenso Asintótico** | `consensus` | Mediana Ponderada de métodos asintóticos | N/A | N/A |

---

## 4. Rigor Dimensional: Sincronización Biyectiva Steps $\leftrightarrow$ Épocas

Para evitar discrepancias donde un entrenamiento de 20 épocas reporte erróneamente paradas en la "Época 20.93":
* **Definición continua estricta:**
  $$\text{epoch}(s) = \frac{s}{N_{\text{steps\_per\_epoch}}}$$
* **Propiedades:**
  - En $s = 0$ (inicio): $\text{epoch} = 0.00$.
  - En $s = N_{\text{steps\_per\_epoch}}$: $\text{epoch} = 1.00$.
  - En $s = 20 \times N_{\text{steps\_per\_epoch}}$: $\text{epoch} = 20.00$.
* En matplotlib, el eje secundario superior se proyecta con:
  ```python
  secax = ax.secondary_xaxis('top', functions=(lambda s: s / steps_per_epoch, lambda e: e * steps_per_epoch))
  ```

---

## 5. Herramientas Desacopladas y Uso CLI

```bash
# Analizar automáticamente un entrenamiento previo desde sus CSVs:
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/analyze_convergence.py \
    --run_dir visuomotor_core/outputs/train_v8_convergence_study \
    --out_dir visuomotor_core/outputs/train_v8_convergence_study/convergence \
    --dpi 300
```
Genera:
* `convergence_stopping_criteria_dual_axis.png` y `.pdf` (Canónica dual-axis con anotaciones de callouts y líneas de método).
* `convergence_diagnostics_4panel.png` y `.pdf` (Diagnóstico 2x2: Loss + Dual Axis, Log-Gradient, RSD%, Macro Épocas).
* `convergence_report.json` y `convergence_summary.csv` estructurados.
