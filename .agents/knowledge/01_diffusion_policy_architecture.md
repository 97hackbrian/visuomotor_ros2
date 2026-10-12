# Módulo 01: Arquitectura de Diffusion Policy y Modelado Neuronal

Este documento compila el conocimiento técnico, formal y matemático extraído de la auditoría línea por línea del paquete `visuomotor_core` (`models/diffusion_policy.py`, `models/config.py`, `models/inference.py`, `models/utils.py`, `train.py`).

---

## 1. El Paradigma Generativo en Robótica Visuomotora

En aprendizaje por imitación para robótica, los modelos deterministas (como MLPs o RNNs clásicos con MSE directo sobre acciones) colapsan ante **distribuciones multimodales** (ej. evitar un obstáculo por la izquierda o por la derecha, o alcanzar un objeto desde diferentes trayectorias). En presencia de multimodalidad, un modelo MSE promedia las trayectorias posibles, resultando en acciones intermedias no válidas.

**Diffusion Policy** resuelve este colapso formulando el problema como el modelado de la distribución de probabilidad condicional de trayectorias completas:

$$p(A_{t:t+H} \mid O_{t-T_{obs}+1:t})$$

Donde:
* $O$: Observación sensorial multivariada que combina visión RGB ($256 \times 256 \times 3$) y propiocepción física ($8$ dimensiones).
* $T_{obs}$: Horizonte de observación temporal ($n\_obs\_steps = 2$).
* $H$: Horizonte temporal de acciones predichas ($horizon = 16$).
* $A_{t:t+H}$: Trayectoria generada para los siguientes 16 pasos ($16 \times 9$).

---

## 2. Componentes de la Arquitectura de Red (`visuomotor_core/models/`)

### A. Encoder Visual (`ResNet-18` modificado + `SpatialSoftmax`)
Definido en `visuomotor_core/models/diffusion_policy.py`:
1. **Backbone ResNet-18 con GroupNorm:**
   - La red carga un ResNet-18 estándar pero sustituye **todas** las capas `BatchNorm2d` por `GroupNorm(num_groups=C // 16, num_channels=C)` mediante la función recursiva `_replace_submodules`.
   - *Justificación matemática:* En entrenamiento por difusión con lotes de tamaño moderado ($B=64$), `BatchNorm` calcula medias y varianzas de lote ruidosas que desestabilizan las estadísticas del gradiente y el decaimiento móvil exponencial (EMA). `GroupNorm` calcula estadísticas por canales dentro de cada muestra individual de forma invariante al tamaño de batch.
   - Las capas `conv1` inicial y los bloques residuales se preservan; la capa `fc` final y el `avgpool` original son eliminados.
2. **Spatial Softmax (64 Dimensiones Latentes):**
   - El tensor de características convolucional resultante $[B, 512, H_f, W_f]$ (donde $H_f = W_f = 8$ tras los strides) entra a una capa `SpatialSoftmax` con $K = 32$ keypoints.
   - Para cada uno de los $K$ canales de mapas de activación $a_k(x, y)$, se aplica un softmax espacial bidimensional con temperatura learnable/fija:
     $$\alpha_k(x, y) = \frac{\exp(a_k(x, y) / \tau)}{\sum_{x'} \sum_{y'} \exp(a_k(x', y') / \tau)}$$
   - Se calcula el centroide esperado $(x_k, y_k)$ normalizado en el intervalo $[-1, 1]$:
     $$x_k = \sum_{x, y} x \cdot \alpha_k(x, y), \quad y_k = \sum_{x, y} y \cdot \alpha_k(x, y)$$
   - Los $K=32$ pares de coordenadas $(x_k, y_k)$ se aplanan en un vector latente de **64 dimensiones**, preservando la geometría espacial métrica sin requerir capas densas gigantescas.

### B. Condicionamiento Global y FiLM (*Feature-wise Linear Modulation*)
1. **Vector de Condicionamiento Global ($\mathbf{cond}$):**
   - El vector de entrada condicional concatena los keypoints visuales y el estado propioceptivo para los $T_{obs} = 2$ pasos temporales:
     $$\mathbf{cond} = [\text{Visual}_{t-1}, \text{Visual}_t, \text{Propio}_{t-1}, \text{Propio}_t]$$
     $$\text{Dimensión: } (64 \times 2) + (8 \times 2) = 128 + 16 = 144 \text{ dimensiones}$$
   - Si $T_{obs} = 1$: $\mathbf{cond} \in \mathbb{R}^{64 + 8} = \mathbb{R}^{72}$.
2. **Inyección FiLM en la 1D Temporal U-Net:**
   - La U-Net 1D está compuesta por bloques `ConditionalResidualBlock1D`.
   - Cada bloque convolucional recibe la activación intermedia $h \in \mathbb{R}^{B, C, L}$ y el embedding conjunto de tiempo difusivo y observación $e \in \mathbb{R}^{B, D_{emb}}$.
   - Una capa lineal proyecta $e$ hacia parámetros de escala $\gamma$ y traslación $\beta$:
     $$\text{FiLM}(h) = \gamma(e) \cdot \text{GroupNorm}(h) + \beta(e)$$
   - Esto modula dinámicamente los filtros temporales en función directa de lo que el robot ve y siente.

### C. 1D Temporal U-Net (`ConditionalUnet1D`)
* **Estructura Jerárquica:**
  - Dimensiones de canal descendentes: `down_dims = (128, 256, 512)`.
  - Tamaño del kernel convolucional: `kernel_size = 5`.
  - Normalización: `GroupNorm` con 8 grupos por defecto (`n_groups = 8`).
  - Activación: `Mish(x) = x \cdot \tanh(\ln(1 + e^x))`.
* **Ruta de Downsampling:** Convoluciones 1D con stride 2 que reducen la dimensión temporal de 16 a 8 y luego a 4.
* **Bloque Central (Mid):** Dos bloques residuales condicionales en la dimensión máxima (512).
* **Ruta de Upsampling:** Convoluciones transpuestas 1D (`ConvTranspose1d`) con conexiones residuales (*skip connections*) concatenadas desde la ruta descendente.
* **Capa de Salida:** `Conv1d(128, action_dim=9, kernel_size=1)` produciendo el tensor predicho de ruido $\hat{\epsilon} \in \mathbb{R}^{B, 16, 9}$.

---

## 3. Dinámica del Proceso de Difusión y Sampling DDIM

### A. Schedule de Ruido en Entrenamiento
* Se utiliza un schedule **Squared Cosine Beta Schedule** (`betas = "squaredcos_cap_v2"`) con $T_{train} = 100$ pasos.
* La formulación de cosenos acumulados $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ evita la degradación abrupta de la señal al inicio del proceso y garantiza que en $t = T$ la distribución sea prácticamente ruido Gaussiano puro $\mathcal{N}(0, I)$.
* **Tipo de Predicción:** $\epsilon$-prediction (`prediction_type: "epsilon"`). La red predice el vector de ruido inyectado, no el valor limpio directo $x_0$, lo que suaviza la topología del gradiente.

### B. Muestreo Acelerado Determinista (DDIM) en Inferencia
* En entrenamiento se usan 100 pasos de difusión.
* En inferencia en vivo (`inference_hz = 20.0 Hz`), 100 pasos requerirían ~400 ms de GPU, violando el ciclo de control de 50 ms.
* Se implementa **DDIM (*Denoising Diffusion Implicit Models*)** con un subconjunto de pasos de inferencia (`num_inference_steps = 10`).
* La trayectoria se reconstruye iterativamente en 10 pasos deterministas:
  $$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \hat{\epsilon}_\theta(x_{\tau_i}, \tau_i)}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \hat{\epsilon}_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i} \epsilon$$
  Para DDIM determinista, $\sigma = 0$, lo que produce una trayectoria limpia y reproducible sin varianza espuria.
* Tiempo total de denoising: **~25-45 ms en GPU NVIDIA**, permitiendo inferencia fluida a 20 Hz.

---

## 4. Temporal Ensemble (Suavizado Temporal Recedente)

En `visuomotor_core/models/diffusion_policy.py` y `visuomotor_ros2/policy_node.py`:
* **Concepto:** En cada invocación (cada $n\_action\_steps$ ciclos, ej. cada 3 ciclos), la red genera un bloque futuro de 16 acciones. A medida que avanza el tiempo, múltiples predicciones generadas en instantes $t-2, t-1, t$ se superponen sobre el mismo instante futuro.
* **Fórmula de Promedio Ponderado con Decaimiento Exponencial:**
  Para cada paso temporal $t$, la acción ejecutada $\bar{A}_t$ es la combinación convexa de las predicciones $A_{t}^{(j)}$ de edad $j$:
  $$w_j = \exp(-k \cdot j)$$
  $$\bar{A}_t = \frac{\sum_{j} w_j A_{t}^{(j)}}{\sum_{j} w_j}$$
  - $k = \text{temporal\_ensemble\_k}$:
    - Si $k = 0.0$: Promedio simple entre todas las trayectorias solapadas (máximo suavizado cinemático).
    - Si $k = 0.01$: Ponderación con ligero decaimiento (SOTA empírico).
    - Si $k \gg 0.1$: Da prioridad casi exclusiva a la predicción más reciente (más reactivo, menos suave).

---

## 5. Entrenamiento y Convergencia (`train.py`)

* **Optimizador:** `AdamW` con `weight_decay = 1e-6` y tasa de aprendizaje ajustada por raíz cuadrada del batch:
  $$\text{lr} = 1.0 \times 10^{-4} \cdot \sqrt{\frac{\text{batch\_size}}{64}}$$
* **Scheduler:** `CosineAnnealingLR` con decaimiento suave hasta `eta_min = 1e-6`.
* **Precisión Mixta:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` para maximizar el throughput sin subdesbordamiento de gradiente.
* **Manejo de Pérdida en Padding (`do_mask_loss_for_padding`):**
  - El dataloader marca los pasos rellenados con `action_is_pad`.
  - La función de pérdida calcula MSE únicamente sobre los pasos válidos o aplica la máscara booleana para evitar que el relleno distorsione los gradientes del inicio de la trayectoria.
* **Fenómeno de *Flat Minima*:**
  - El loss MSE en robótica por difusión oscila sanamente entre `0.003` y `0.008`.
  - Detener el entrenamiento al estabilizarse el loss en meseta evita el sobreajuste a modos cerrados que degradan la recuperación ante perturbaciones en el robot real o simulado.
