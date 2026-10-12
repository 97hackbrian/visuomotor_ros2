---
name: diffusion-policy-expert
description: Advanced knowledge base and best practices for tuning, debugging, and improving Diffusion Policy (DP) models in robotic visuomotor control. Includes architectural pipelines, mathematical formulations, and troubleshooting guides.
---

# Diffusion Policy Expert Knowledge Base

## 1. La Jerarquía Visuomotora (The Robotics AI Stack)
El ecosistema visuomotor opera en 6 capas rigurosamente desacopladas:

1. **El Dominio $\rightarrow$ Visuomotor Control:** Píxeles crudos directos a comandos de motores (End-to-End).
2. **La Metodología $\rightarrow$ Behavior Cloning (Imitation Learning):** Enseñar al robot copiando demostraciones de un maestro experto, sin funciones de recompensa manuales.
3. **El Paradigma Generativo $\rightarrow$ Diffusion Policy:** Usar modelos de difusión para mapear imágenes a distribuciones multimodales de trayectorias completas, evitando el promedio catastrófico de acciones.
4. **La Arquitectura (El Cerebro) $\rightarrow$ 1D Temporal U-Net + ResNet-18:**
   - **ResNet-18 + Spatial Softmax:** Extrae keypoints visuales métricos ($K=32 \rightarrow 64$ dimensiones latentes). Sustituye **siempre** `BatchNorm2d` por `GroupNorm(num_groups=C // 16)` para estabilizar el cálculo de medias/varianzas con batch sizes pequeños y evitar derivas en el decaimiento móvil exponencial (EMA).
   - **FiLM (Feature-wise Linear Modulation):** El vector de condicionamiento global ($\mathbf{cond} \in \mathbb{R}^{144}$ para $T_{obs}=2$) modula las activaciones intermedias temporales de la U-Net aplicando $\gamma(e) \cdot \text{GroupNorm}(h) + \beta(e)$ en cada bloque residual.
   - **1D Temporal U-Net vs DiT:** Para tareas continuas de *Pick&Place* con datasets de escala moderada (< 2,000 episodios), la CNN 1D es inmensamente superior: entrena en minutos/horas, converge a *Flat Minima* estables y no sufre el sobreajuste severo típico de los *Diffusion Transformers* (DiT).
5. **El Algoritmo (Motor Matemático) $\rightarrow$ DDIM (Denoising Diffusion Implicit Models):**
   - **Squared-Cosine Beta Schedule:** Curva no lineal que preserva la estructura de la trayectoria en los primeros niveles de ruido.
   - **Predicción $\epsilon$ (Epsilon):** La U-Net predice el ruido Gaussiano introducido, estabilizando la topología del gradiente frente a la predicción directa de $x_0$.
   - **Inferencia DDIM:** Comprime 100 pasos de difusión en **10 saltos deterministas** (~30 ms en GPU), logrando ciclos de control a 20 Hz.
6. **Estrategia de Ejecución $\rightarrow$ Receding Horizon Control (RHC) & ROS 2 Filtering:**
   - Predice un horizonte futuro de $H=16$ pasos.
   - **Temporal Ensemble:** Promedia bloques continuos superpuestos con decaimiento exponencial $w_j = \exp(-k \cdot j)$ para suavizado de alta fidelidad.
   - **Filtro 1-Euro Adaptativo:** Elimina micro-vibraciones de alta frecuencia en reposo sin añadir desfase temporal (*lag*) en tránsitos rápidos.

---

## 2. Pipeline Completo del Sistema

```mermaid
flowchart TD
    %% --- RECOLECCIÓN DE DATOS ---
    subgraph DataGen [FASE 1: Generación de Datos Sintéticos Expertos]
        direction TB
        H[Halton QMC 3D: XY Yaw]:::math --> FSM[Expert FSM: 12 Estados]:::control
        FSM --> PID[Cartesian PID 100Hz: Virtual Setpoint Anti-Runaway]:::control
        PID --> RobotState["Propiocepción 8D (Qw >= 0)"]:::hardware
        Cam[Cámara RGB Muñeca 256x256]:::hardware --> ImgData[Imágenes RGB]:::database
        RobotState --> ZARR[("Zarr Dataset: demonstrations_vX.zarr")]:::database
        ImgData --> ZARR
    end

    %% --- DATALOADER & REGULARIZACIÓN ---
    subgraph DataloaderPipe [FASE 2: Ingesta, Copy-Padding y Regularización]
        direction TB
        ZARR --> CP[Copy-Padding Terminal: Repite Último Frame H=16]:::math
        CP --> DropProp[Proprioception Dropout: 15% Ceros]:::math
        DropProp --> Norm[Normalización Min-Max: -1 a 1 con eps=1e-4]:::math
        Norm --> Batch[Batch Tensorizado 64]:::database
    end

    %% --- ENTRENAMIENTO ---
    subgraph TrainPipe [FASE 3: Entrenamiento - Forward Diffusion]
        direction TB
        Batch --> GroundTruth[Acción Experta 16x9]:::database
        RealNoise[Ruido Gaussiano N:0,I]:::database --> NoisyTraj
        GroundTruth -->|Inyección Ruido Coseno t| NoisyTraj[Trayectoria Ruidosa 16x9]:::database
        NoisyTraj --> UNET_T[1D Temporal U-Net: GroupNorm + Mish]:::neuralnet
        Batch -->|Visión 64D + Propio 8D x Obs=2| FiLMMod[Modulación FiLM 144D]:::math
        FiLMMod -.-> UNET_T
        UNET_T --> PredNoise[Epsilon Predicho]:::database
        PredNoise --> Loss{Masked MSE Loss}:::math
        RealNoise --> Loss
        Loss --> Opt[AdamW + CosineAnnealingLR]:::math
    end

    %% --- INFERENCIA ---
    subgraph InferPipe [FASE 4: Inferencia en Vivo ROS 2 - 20 Hz]
        direction TB
        CamLive[Cámara /rgb]:::hardware --> PrepImg[Resize 256x256 + Norm]:::math
        RobLive[TF link_base + /joint_states]:::hardware --> PrepState["Estado 8D (Qw >= 0)"]:::math
        PureNoise[Ruido Gaussiano Inicial 16x9]:::database --> DDIM[DDIM Solver: 10 Pasos]:::math
        DDIM --> UNET_I[1D U-Net Congelada bfloat16]:::neuralnet
        PrepImg & PrepState --> FiLMModLive[FiLM Conditioning]:::math
        FiLMModLive -.-> UNET_I
        UNET_I --> RawChunk[Acción 16x9 Desnormalizada]:::database
        RawChunk --> TempEns[Temporal Ensemble: Exp Decay w = exp -k*age]:::math
        TempEns --> CheckStop{Auto-Stop Neuronal: Dim 9 >= Umbral?}:::control
        CheckStop -- Sí --> Freeze[State.DONE: Congelar Brazo y Gripper]:::control
        CheckStop -- No --> Filter[Filtro 1-Euro: PoseSmoother]:::control
        Filter --> ROS2Cmd[/target_frame: Cartesian Motion Controller/]:::hardware
    end
```

---

## 3. Diagnóstico de Entrenamiento y Convergencia (*Flat Minima*)

### El Fenómeno del "Flat Minima" y la Oscilación de Pérdida
En políticas visuomotoras continuas basadas en difusión:
* **El MSE Loss NUNCA desciende a 0.0:** Debido a la naturaleza estocástica del ruido forward inyectado, el loss MSE converge típicamente a una meseta oscilante entre `0.003` y `0.008`.
* **Criterio de Parada Temprana:** Al estabilizarse en esta meseta durante varias decenas de épocas, el entrenamiento **debe detenerse**. Extender el entrenamiento con el objetivo de reducir el loss a mínimos extremos provoca sobreajuste a *Sharp Minima*, destruyendo la capacidad del modelo para generalizar ante variaciones visuales y de postura.
* **Tasa de Aprendizaje y Escala de Lote:**
  $$\text{lr} = 1.0 \times 10^{-4} \cdot \sqrt{\frac{\text{batch\_size}}{64}}$$
  El uso de `AdamW` (`weight_decay = 1e-6`) con `CosineAnnealingLR` garantiza una navegación suave en la meseta de mínimos planos.

---

## 4. Técnicas Clave de Robustez y Anti-Overfitting

### A. Proprioception Dropout (15%)
* **Problema:** Un modelo que recibe la propiocepción $[X, Y, Z, q_x, q_y, q_z, q_w]$ en todo momento tiende a aprender trayectorias en lazo abierto (*blind dead-reckoning*), ignorando las imágenes de la cámara.
* **Técnica:** En `zarr_dataset.py`, aplicar aleatoriamente con un 15% de probabilidad:
  ```python
  if torch.rand(1).item() < 0.15:
      state = torch.zeros_like(state)
  ```
  Esto fuerza al encoder visual (ResNet-18 + SpatialSoftmax) a guiar activamente el efector final basándose en la retroalimentación visual en vivo.

### B. Corrección de Semiesfera y Doble Cubierta de Cuaterniones (*Double Cover*)
* **Propiedad de $SO(3)$:** Los cuaterniones $q$ y $-q$ representan exactamente la misma orientación física tridimensional. Sin embargo, para una red neuronal convolucional o una función de pérdida L2, predecir $-q$ cuando la meta está en $q$ produce un gradiente enorme y destructivo que atraviesa el origen $\|q\| \rightarrow 0$.
* **Regla Inviolable:** Forzar siempre que la componente escalar de los cuaterniones en la observación y acciones pertenezca a la semiesfera superior:
  ```python
  if q[3] < 0.0:
      q = -q
  ```
  En interpolaciones SLERP del generador sintético:
  ```python
  dot = np.sum(q1 * q2)
  if dot < 0.0:
      q1 = -q1
      dot = -dot
  ```

### C. Freno Dinámico en Z (Prevención de Desenfoque Óptico)
* Movimientos rápidos de rotación del gripper combinados con descensos bruscos provocan *Motion Blur* severo en la cámara de muñeca, degradando las activaciones del ResNet.
* **Solución:** Incluir un multiplicador de "velocidad tortuga" (`turtle_descent_multiplier: 0.15`) y modular la velocidad de descenso en $Z$ de forma inversamente proporcional al error angular pendiente.

---

## 5. Modelado de Fin de Tarea (Auto-Stop) y Reglas de Tensores

### A. Especificación Formal de Tensores
* **Observación (`observation.state`) — 8 Dimensiones:**
  Modela magnitudes físicas medidas por hardware en el instante $t$:
  `[X, Y, Z, Qx, Qy, Qz, Qw, Gripper_Actual]` (con $Q_w \ge 0$).
* **Acción (`action`) — 9 Dimensiones:**
  Modela la trayectoria futura predicha para los siguientes 16 timesteps:
  `[Target_X, Target_Y, Target_Z, Target_Qx, Target_Qy, Target_Qz, Target_Qw, Target_Gripper, Done_Flag]`
  - Dimensión 8 (`Done_Flag`): Probabilidad continua $[0, 1]$ de finalización de tarea.

### B. Regla Fundamental: Prohibición de Fuga Causal (*Causal Leakage*)
> [!CRITICAL]
> **La señal de parada (`done`) JAMÁS debe formar parte de la observación `observation.state`.**
* La señal de parada es una **decisión cognitiva de salida**, no un sensor físico de entrada. Si entra a la observación, se crea una dependencia circular en inferencia donde el nodo requeriría un oráculo externo para saber qué alimentar a la red, y la red colapsaría aprendiendo simplemente a copiar la entrada a la salida.

### C. El Bug del Recorte del Horizonte (*Horizon Truncation Bug*) y Solución Copy-Padding
* **El Problema:** Al predecir $H=16$ pasos futuros, un dataloader ingenuo descarta los últimos 16 frames de cada demostración para evitar indexar fuera del array. Si el fin de la tarea se grabó en los últimos frames, el modelo entrena con un $100\%$ de muestras con `done = 0.0`.
* **Solución Mandatoria (*Copy-Padding*):** Replicar el último vector de acción terminal (que contiene la pose final y `done = 1.0`) hasta completar los 16 pasos:
  ```python
  pad_len = self.h_act - actual_len
  pad_actions = np.repeat(actual_actions[-1:], pad_len, axis=0)
  actions = np.concatenate([actual_actions, pad_actions], axis=0)
  ```
  Esto enseña a la red a detenerse y permanecer estática en destino cuando la imagen confirma el éxito.

### D. Doble Capa de Auto-Stop Desacoplado en Inferencia
1. **Auto-Stop Neuronal (Primario):**
   - Parámetros: `enable_neural_auto_stop: true`, `neural_auto_stop_threshold: 0.05` a `0.50`.
   - Se activa cuando $\text{action}[0, 8] \ge \text{umbral}$.
   - Pasa inmediatamente a `State.DONE`, congela el brazo y retiene el gripper.
2. **Auto-Stop Heurístico (Respaldo):**
   - Parámetros: `enable_heuristic_auto_stop: false`, `auto_stop_variance_threshold: 1.0e-7`, `auto_stop_patience_steps: 20`.
   - Monitorea la varianza cartesiana 3D en ventana deslizante; si el robot permanece quieto durante la ventana de paciencia, asume finalización asintótica.

---

## 6. Pipeline de Suavizado y Ejecución Cinemática en Tiempo Real

Para mitigar el jitter sin retrasar la respuesta:
1. **Capa 1: Temporal Ensemble (Receding Horizon):**
   Combina predicciones solapadas mediante decaimiento exponencial:
   $$w_j = \exp(-k \cdot j)$$
   $$\bar{A}_t = \frac{\sum_j w_j A_t^{(j)}}{\sum_j w_j}$$
   Con $k = 0.0$ (promedio plano, máxima suavidad) o $k = 0.01$ (SOTA).
2. **Capa 2: Filtro 1-Euro Adaptativo (`pose_smoother.py`):**
   Modula la frecuencia de corte según la velocidad del efector:
   $$f_c = f_{c,\min} + \beta \|\hat{\dot{x}}\|$$
   - Aplica desenvolvimiento angular de Euler (`np.unwrap`) en cuaterniones para evitar discontinuidades en $\pm\pi$.
   - Timeout de 0.5 s para resetear memorias si la inferencia se suspende.
3. **Capa 3: Publicación Unidimensional del Gripper:**
   El controlador en simulación espera un `Float64MultiArray` con **estrictamente 1 elemento** `[pos]`. Publicar arreglos bidimensionales produce el rechazo silencioso de la orden.
