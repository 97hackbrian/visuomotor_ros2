# AGENTS.md — Master Autonomous Agent Operational Contract

Este documento es el contrato maestro de operación y guía arquitectónica para **todos los agentes de Inteligencia Artificial** (Google Antigravity, OpenAI Codex/Operator, Anthropic Claude Code, Cursor, Windsurf, Copilot, etc.) que colaboren en el repositorio `visuomotor_ros2`.

Cualquier agente que inicie una sesión en este proyecto **DEBE leer este documento primero** y adherirse a sus directivas de manera estricta.

---

## 1. Misión del Proyecto y Topología Arquitectónica

El proyecto implementa un framework robótico de aprendizaje por imitación para **Políticas Visuomotoras Generativas (Diffusion Policy, DDIM Schedule)** sobre **ROS 2 Jazzy**.

El repositorio se divide en 5 paquetes desacoplados:
1. **`visuomotor_core`** (Python Puro / Off-ROS):
   - Redes neuronales (1D U-Net + ResNet-18 GroupNorm + Spatial Softmax + FiLM).
   - Dataloaders Zarr y script de entrenamiento (`train.py`).
   - *Regla inviolable:* Cero dependencias de ROS (`rclpy`, `ament`). Debe poder ejecutarse en cualquier cluster de GPU sin ROS instalado.
2. **`visuomotor_msgs`** (ament_cmake):
   - Definiciones de mensajes (`ActionChunk.msg`, `ObservationData.msg`) y servicios (`EpisodeTrigger.srv`).
3. **`visuomotor_data_collection`** (ament_python):
   - Recolección autónoma de datos (`pickup_approximate.py`) mediante generador experto FSM + PID Cartesiano + Respawn estocástico (Halton Quasi-Monte Carlo).
   - Generación de datasets en formato Zarr (`demonstrations_vX.zarr`).
4. **`visuomotor_ros2`** (ament_python):
   - Nodo de inferencia en tiempo real (`policy_node.py`).
   - Sincronización sensorial, normalización, inferencia por lotes y ejecutores de acción (`ActionExecutor`).
5. **`visuomotor_bringup`** (ament_cmake):
   - Launch files y descripciones URDF/hardware específicas (único silo donde se permiten particularidades del brazo o simulador).

---

## 2. Reglas Operativas del Entorno (No Negociables)

1. **Entorno Docker Obligatorio (Usuario `hackbrian` y Consentimiento Expreso):**
   - Todo lo relacionado con Python, ROS 2, scripts, entrenamiento, datasets y compilación opera dentro del contenedor Docker `visuomotor_ros2` (montado en `/ros`).
   - NUNCA ejecutes comandos de instalación o compilación de ROS directamente en el host Linux del usuario si alteran el entorno.
   - **Permisos de Usuario Obligatorios (`hackbrian@robot_dev`):** Los comandos dentro del contenedor DEBEN ejecutarse siempre bajo el usuario `hackbrian` (`-u hackbrian`) para preservar los permisos y evitar que los archivos generados queden como propiedad de `root`:
     ```bash
     docker exec -u hackbrian -w /ros visuomotor_ros2 bash -c "<comando>"
     ```
   - **Condición de Consentimiento:** La IA ÚNICAMENTE ejecutará comandos en la terminal cuando el usuario lo exprese directamente en su instrucción. NUNCA ejecutes comandos proactivamente sin orden previa.
2. **Gestión de Git Exclusiva en el Host Local:**
   - **NUNCA ejecutes comandos de Git (`git status`, `git add`, `git commit`, `git config`, etc.) dentro del contenedor Docker.**
   - El control de versiones Git lo gestiona exclusivamente el usuario desde su entorno local en el Host.
3. **Protección Absoluta de Datasets:**
   - NUNCA sobrescribas, borres ni alteres los datasets existentes en `datasets/` (ej. `demonstrations_v7.zarr`).
   - Cualquier enriquecimiento o corrección debe generar una copia explícita (ej. `demonstrations_v7_done_fixed.zarr`).
   - Los nuevos datasets usan modo `"AUTO"` generando automáticamente versiones incrementales (`demonstrations_v8.zarr`, etc.).
4. **Prohibición Total de Hardcoding:**
   - Ningún tópico de ROS, umbral numérico, duración de reset, tamaño de imagen o geometría de gripper debe estar quemado en código Python.
   - Todo debe declararse como parámetro ROS 2 y leerse de `visuomotor_ros2/config/policy_params.yaml` o `visuomotor_data_collection/config/data_collection_params.yaml`.
5. **Idioma de Comunicación:**
   - La comunicación con el usuario debe ser en **Español**, clara, técnica y concisa.

---

## 3. Especificación de Tensores y Dimensiones

### A. Observación Propioceptiva (`observation.state`) — 8 Dimensiones
Modela exclusivamente lo que los sensores físicos miden en el instante $t$:
* `state[0:3]`: Posición cartesiana actual del TCP $[X, Y, Z]$ en metros (referencia `link_base`).
* `state[3:7]`: Orientación del TCP como cuaternión $[q_x, q_y, q_z, q_w]$. Signo según `quaternion_hemisphere`: `qw_positive` ($q_w \ge 0$, histórico/simulación) u `home` ($\mathbf{q}\cdot\mathbf{q}_{\text{home}} \ge 0$, obligatorio con la garra del xArm5 hacia abajo, donde $q_w \approx 0$ y $q_w \ge 0$ es discontinuo). El checkpoint guarda la convención en `config.json`.
* `state[7]`: Estado medido de apertura del gripper (leído de `/joint_states`).

> [!CRITICAL]
> **Prohibición de Fuga Causal (Causal Leakage):**
> La 9ª dimensión (`done`) **JAMÁS debe formar parte de `observation.state`**. La observación es un sensor físico; la señal de parada es una decisión. Si se agrega a la entrada, la red aprende una dependencia circular y en inferencia requeriría un oráculo externo.

### B. Espacio de Acciones (`action`) — 9 Dimensiones
Modela la trayectoria futura predicha por la U-Net ($16 \times 9$):
* `action[0:3]`: Coordenadas objetivo $[X, Y, Z]$.
* `action[3:7]`: Cuaternión objetivo $[q_x, q_y, q_z, q_w]$ (normalizado en inferencia).
* `action[7]`: Comando de gripper ($0.0$ abierto, $\le -0.005$ cerrado).
* `action[8]`: **`done_flag` / Auto-Stop Neuronal** ($0.0$ = tarea en curso, $1.0$ = tarea completada).
* `action_representation` (en `config.json`): `absolute` (pose en el marco del estado) o `relative_ee` (pose relativa al efector actual, estilo UMI; `DiffusionPolicy` la recompone a absoluta con el estado observado, así que `policy_node` y `ActionChunk` no cambian).

### C. Observación Visual (`observation.image`)
* Formato: $[3, 256, 256]$ RGB normalizado en $[0, 1]$.
* ResNet-18 con `GroupNorm` + `SpatialSoftmax` (reduce a 64 keypoints latentes).

---

## 4. Mecanismo de Auto-Stop Desacoplado

El nodo de inferencia implementa dos capas independientes de detención:
1. **Auto-Stop Neuronal (Primario):**
   - Parámetros: `enable_neural_auto_stop: true`, `neural_auto_stop_threshold: 0.50`.
   - Se activa cuando la red predice `action[8] >= neural_auto_stop_threshold`.
   - Congela el brazo en su posición final y retiene el gripper en su estado.
2. **Auto-Stop Heurístico (Respaldo):**
   - Parámetros: `enable_heuristic_auto_stop: false`, `auto_stop_variance_threshold: 1.0e-7`, `auto_stop_patience_steps: 20`.
   - Se activa si la varianza del movimiento del brazo cae por debajo del umbral durante la ventana de paciencia.
3. **Monitoreo en Logs:**
   - Durante inferencia regular: Log a 1 Hz reportando acción XYZ, estado del gripper y valor de la 9ª dimensión vs umbral.
   - Al activarse: Log explícito con el valor exacto de la 9ª dimensión que disparó la parada.
   - En estado congelado (`DONE`): Log periódico cada 2 s reteniendo el valor y motivo exacto que detonó la parada.

4. **Telemetría y Métricas en Tiempo Real (rqt_plot) y Offline (CSV + Matplotlib):**
   - **Tópicos de Inferencia (`/policy/metrics/`):** `done_flag`, `done_threshold`, `gripper_cmd`, `gripper_state`, `tracking_error_xyz`, `tcp_speed_m_s`, `action_variance`, `dist_to_home`, `tcp_cmd_xyz`, `tcp_actual_xyz`.
   - **Tópicos de RL (`/rl/metrics/`):** `instant_reward`, `episode_reward`, `critic_value`, `done_flag`, `pseudo_force`, `dist_hover`, `dist_target`, `r_approach`, `r_orient`, `r_grasp`, `r_transport`, `iter_mean_reward`, `iter_policy_loss`, `iter_critic_loss`, `iter_approx_kl`.
   - **Guardado Offline:** Ambas modalidades generan archivos `.csv` tabulares y curvas en `.png` y `.pdf` renderizadas con backend headless (`Agg`) de `matplotlib` para evitar conflictos con entornos sin display X11.

---

## 5. Dataloaders e Ingesta de Datos

1. **Horizon Truncation Bug & Solución Copy-Padding (`zarr_dataset.py`):**
   - Debido a que la red predice $H_{act} = 16$ pasos hacia el futuro, un slicing directo descartaría los últimos 16 frames de cada demostración.
   - En datasets con señal de término al final, esto borraría el 100% de las muestras de `done = 1.0`.
   - *Solución obligatoria:* Implementar siempre **Copy-Padding** replicando el último frame hasta completar los 16 pasos cuando la ventana supere el fin del episodio.
2. **Recolección en `pickup_approximate.py`:**
   - Mantener `record_done_flag: true`.
   - En la fase `HOLD_TARGET` (3.5 segundos en destino), el flag se inyecta en `1.0`.
   - Al guardar el episodio, asegurar un piso garantizado de mínimo 2.0 segundos (40 frames a 20 Hz) con `done = 1.0`.

---

## 6. Metodología Científica, Benchmarking y Logging Off-ROS

1. **Los Dos Ejes Fundamentales de Investigación (Papers):**
   - **Eje 1 (Demostrador Experto de Datasets):** Comparación sistemática de datasets generados por el PID cartesiano (piloto actual) frente a demostradores SOTA (Model Predictive Control / cuRobo GPU MPC, y Cartesian Impedance Control / CRISP).
   - **Eje 2 (Paradigma de Aprendizaje):** Comparación empírica formal de **Solo Diffusion Policy (Behavior Cloning)** frente a **Diffusion Policy + RL Fine-Tuning (DPPO)**.
   - *Matriz Factorial:* Toda serie experimental cruzará $(D_{\text{experto}}) \times (\text{Algoritmo})$ evaluada sobre $N \ge 100$ episodios bajo semillas idénticas Halton QMC.
2. **Estándar de Logging y Benchmarking Off-ROS (`visuomotor_core`):**
   - Todo módulo Off-ROS implementa obligatoriamente `visuomotor_core.utils.logger`:
     - `configure_logging(level=logging.DEBUG, log_file="run.log")`: `coloredlogs` enriquecido con milisegundos, número de línea e hilo.
     - `@contextmanager def benchmark(name, log, track_gpu=True)`: timing nanosegundo (`time.perf_counter_ns()`), formato legible (`humanize.precisedelta`) y monitoreo de VRAM pico GPU (`torch.cuda.max_memory_allocated()`).
     - Prohibición estricta de `rclpy` en `visuomotor_core` y de `print()` informales en bucles de producción.
3. **Calibración Propioceptiva y Respawn Determinista:**
   - Umbral de agarre calibrado: `pseudo_force_clamped_threshold: 0.000700` ($0.7\text{ mm}$).
   - Certificación en vuelo: $\Delta z \ge 8\text{ cm}$ con garra cerrada valida `is_clamped = True` incondicionalmente.
   - Respawn bifásico: pulso con elevación ($Z+4\text{ cm}$, $\pm 0.5\text{ mm}$ jitter) para invalidar el filtro `last_lin_processed` de Isaac Sim, seguido de 2.0s de estabilización gravitacional en Home y purgado de semáforo de cámara.
4. **Base de Datos Relacional SQLite Fail-Safe (`experiments/experiments.db`):**
   - Conexión automática no intrusiva en `train.py`, `rl_train_node.py`, `pickup_approximate.py` y `evaluate_policy.py`.
   - Protección integral con `try/except` para garantizar que fallos en la base de datos nunca detengan el entrenamiento ni el guardado de checkpoints.
   - Consulta instantánea en terminal: `python3 tools/experiment_tracker.py summary`.
5. **Modelo Cinemático TCP y Offset de Pinza ($79.0\text{ mm}$):**
   - En `GymROS2Env`, la proximidad se calcula desde el Tool Center Point proyectando la orientación $\mathbf{q}$:
     $$\mathbf{v}_z = [2(q_x q_z + q_w q_y), 2(q_y q_z - q_w q_x), 1 - 2(q_x^2 + q_y^2)], \quad \mathbf{p}_{\text{tcp}} = \mathbf{p}_{\text{ee}} + \mathbf{v}_z \cdot 0.079$$
   - Esto evita mediciones artificiales de $\sim 80\text{ mm}$ desde la brida y reporta el contacto físico real ($0.88 - 7.14\text{ mm}$ en agarres exitosos).
6. **Protocolo de Benchmarking Pareado y Tratamiento de Outliers:**
   - Comparación determinista pareada (Solo DP vs. DP + RL) bajo semillas Halton QMC compartidas (`tools/compare_models.py`).
   - Generación de panel gráfico $2 \times 3$ en alta resolución (PNG 300 DPI y PDF) con escala logarítmica en distancia mínima de contacto, zoom inteligente en rango nominal para error terminal y duración, y badges explícitos para outliers.
   - Herramienta desacoplada sin simulación: `tools/replot_benchmark.py`.
   - Habilidad operativa documentada en `.agents/skills/visuomotor-benchmark-evaluator/SKILL.md`.
7. **Estándar Editorial Científico, Diagramación TikZ y Estadística No Paramétrica:**
   - Erradicación de sesgos y metalenguaje de IA (*"según la rúbrica"*, *"entrega final"*, hipérboles) en favor de prosa académica humana, sobria y directa en tercera persona o pasiva refleja.
   - Consistencia aritmética y dimensional en esquemas vectoriales de redes neuronales ($2 \times 64\text{D} + 16\text{D} = 144\text{D}$ para $T_{\text{obs}}=2$) y resolución geométrica de colisiones en aristas (`[midway, right=4pt]`).
   - Fundamentación estadística formal del test de rangos con signo de Wilcoxon para distribuciones asimétricas y *outliers* cinemáticos en robótica manipuladora.
   - Reconciliación empírica con telemetría real: justificación de dinámicas de divergencia KL discretas acotadas por recorte PPO ($\epsilon = 0{,}2$) y pérdida de imitación ($\lambda_{\text{BC}}$), diferenciación entre retornos por lote y episodios individuales, y cita de DPPO fundacional (Allen Z. Ren et al., ICLR 2025).
   - Habilidad operativa documentada en `.agents/skills/scientific-latex-reporting/SKILL.md` y memoria modular en `.agents/knowledge/06_scientific_latex_reporting_and_tikz.md`.
8. **Evaluación Avanzada SOTA y Cuantificación de Incertidumbre en Políticas Generativas:**
   - Superación del dilema multimodalidad vs. incertidumbre epistémica (Diff-DAgger / consistencia de score y desacuerdo en campo de velocidades).
   - Predicción Conforme (PlanCP) para tubos de trayectoria con garantía matemática de cobertura libre de distribución ($1 - \alpha = 95\%$).
   - Escalado en tiempo de inferencia (*Test-Time Scaling / Best-of-N*) con el Crítico $V_\phi$ como verificador de trayectorias generadas en paralelo.
   - Diagnóstico interpretable de modos de fallo mediante AdaBoost con tocones de decisión sobre telemetría tabular.
   - Memoria modular documentada en `.agents/knowledge/07_sota_visuomotor_evaluation_and_uncertainty.md`.
9. **Visión 3D (DP3) y Rigor Metodológico en Machine Learning:**
   - Transición hacia 3D Diffusion Policy (DP3) con nubes de puntos dispersas ($512/1024$ pts por FPS), `DP3Encoder` sin T-Net ni BatchNorm (con LayerNorm), recorte espacial de volumen de trabajo y predicción de muestra ($x_0$).
   - Rigor en datos secuenciales: prohibición de partición por fotogramas (*data leakage* temporal), partición estricta por episodios completos (split 80/20), test de sanidad de sobreajuste (Ley de Kolter) y normalizaciones independientes de batch size (`LayerNorm` / `GroupNorm`).
   - Memoria modular documentada en `.agents/knowledge/08_3d_diffusion_policy_and_ml_rigor.md`.
10. **Transferencia Sim-to-Real: Few-Shot Fine-Tuning y Políticas Residuales (ResiP/TRANSIC):**
   - Espacio de acción cartesiano de alto nivel ($16 \times 9$) absorbe el $90\%$ de discrepancias inerciales y de torque motor, focalizando la brecha en visión y contacto mecánico.
   - SOTA en Fine-Tuning: prevención de olvido catastrófico mediante Co-Fine-Tuning ($20\%$ real, $80\%$ sim) y LoRA en capas visuales.
   - SOTA en Residual Policy Learning: desacoplamiento en macro-planificador congelado $\pi_{\text{sim}}$ y residuo reactivo $\pi_{\text{res}}$ acotado con $\pm \delta_{\max} \tanh(\cdot)$ para micro-contacto y cumplimiento fino (ResiP / TRANSIC).
   - Memoria modular documentada en `.agents/knowledge/10_sim_to_real_finetuning_and_residual_policies.md`.
11. **Aceleración Headless con Real-Time 2.0 (RTPT) + NVIDIA DLSS Performance (x2) y Eco Mode:**
   - En Isaac Sim 6.0, el modo "Real-Time 2.0" de la GUI mapea al backend nativo `RealTimePathTracing` (`RTPT`).
   - Parámetros canónicos inyectados en Kit CLI y `carb.settings`:
     - Eco Mode: `/rtx/ecoMode/enabled = true`
     - Super Resolution: `/rtx/post/aa/op = 3` (`DLSS`)
     - Modo DLSS: `/rtx/post/dlss/execMode = 0` (`Performance`)
     - Frame Generation (x2): `/rtx/dlss/frameGeneration = true`, `/rtx-transient/dlssg/enabled = true`, `/rtx-transient/internal/dlssg/rtpt/interpolatedFrameCount = 1`
   - *Protección crítica de Hydra (Zero-Crash):* Prohibición absoluta de invocar `CreateAttribute` sobre prims de texturas de render activo en tiempo de carga (`_MarkSprimDirty` crash en `changeTracker.cpp`); actualización en memoria RAM restringida a prims y atributos existentes.
12. **Calibración Cinemática Óptima ($0.20\text{ m/s}$), Tiempos FSM y Realidad de `frameSkipCount`:**
   - *Velocidad cartesiana calibrada:* `max_speed_m_s: 0.20` representa la frontera Pareto óptima entre dinamismo y estabilidad (suprime oscilaciones inerciales en `ALIGN_ROTATION` y amortigua el contacto fino durante `SETTLE`, logrando un ciclo de ~56 s de pared).
   - *Tiempos de FSM y Piso de Término:* `wait_time_s: 0.8`, `reset_joints_duration_s: 1.5`, `settle_time_s: 0.1` y `hold_target_time_s: 2.0` (garantizando exactamente 40 frames de `done = 1.0` a 20 Hz).
   - *Realidad de `frameSkipCount`:* En Isaac Sim 6.0, `frameSkipCount` en `OgnROS2CameraHelper` está deprecado y solo descarta la publicación en DDS post-render, sin reducir pasadas de Hydra en GPU (el RTF físico se mantiene en $0.33\times$). La aceleración real se deriva de headless, DLSS Performance, reajuste en RAM a 256x256 y cinemática optimizada.
   - Habilidad operativa estandarizada en `.agents/skills/headless-experimentation-workflow/SKILL.md`.
13. **Recolección Concurrente Paralela Dual (2 Instancias), Watchdog Auto-Heal y Fusión Atómica Zarr:**
   - Para saturar de forma eficiente la GPU (RTX 4060 Laptop 8GB VRAM al ~60%) y duplicar el rendimiento de recolección, `run_headless_system.sh collect --instances 2` despliega 2 instancias de Isaac Sim Headless + ROS 2 desacopladas (`ROS_DOMAIN_ID=42` / Kit 8211 y `ROS_DOMAIN_ID=43` / Kit 8212).
   - *Aislamiento por Procesos:* Las rutinas de limpieza y reinicio filtran `/proc/$pid/environ` por `ROS_DOMAIN_ID` para no afectar instancias hermanas.
   - *Watchdog Auto-Heal (140 s):* Reinicia en caliente instancias desfasadas o divergentes sin detener la recolección paralela.
   - *Fusión Atómica Zarr (`tools/merge_zarr_batches.py`):* Cada instancia graba en shards temporales (`_batch_1.zarr`, `_batch_2.zarr`). Al finalizar, un merge indexa y aplana los episodios consecutivamente (`episode_0` .. `episode_N-1`), garantizando compatibilidad 100% con los dataloaders estándar sin partición jerárquica.
14. **Determinación de Convergencia y Criterios de Parada SOTA (Stopping Criteria):**
   - En entrenamiento supervisado de políticas de difusión, el ruido intrínseco del mini-batch ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $k \sim \mathcal{U}(0, K-1)$) genera fluctuaciones estocásticas ($\sigma/\mu \sim 15\%$). Los criterios de parada deben evaluar la dinámica sobre la curva suavizada ($\text{EMA}, \alpha=0.05$) o sobre pérdidas por época.
   - *Desacoplamiento de Dos Regímenes:*
     - *Régimen 1 (Codo Kneedle / Curvatura Máxima):* Marca la adquisición de macro-trayectorias (Época $\sim 2.60$, Paso 390, Loss $\sim 0.134$).
     - *Régimen 2 (Consenso Asintótico Pareto):* Convergencia fina y destreza de agarre sostenida por Gradiente Plano ($|d\mathcal{L}/dt| < 3\times 10^{-5}$), Varianza $RSD < 2.5\%$, Prechelt $P_k < 25$, Mejora Relativa $\Delta \mathcal{L}/\mathcal{L} < 0.5\%$ y Gap EMA $< 3\%$ (Época $\sim 19.57$, Paso 2,935, Loss $\sim 0.0336$).
   - *Sincronización Dual Sincronizada:* $\text{epoch} = \text{step} / N_{\text{steps\_per\_epoch}}$. Prohibición de offsets de índice ordinal que reporten épocas ficticias fuera del rango total ejecutado.
   - *Módulo y Memoria:* `visuomotor_core/visuomotor_core/analysis/stopping_criteria.py` y `.agents/knowledge/12_training_convergence_and_stopping_criteria.md`.
15. **Evaluación Multi-Checkpoint, Corrección Geométrica de Auto-Stop y Frente de Pareto Físico:**
   - *Calibración de Evaluación:* Umbral de auto-stop ajustado a $\le 5\text{ cm}$ para objetos sujetos (`arrival_thresh`), descuento formal de los $79\text{ mm}$ del efector final ($\mathbf{p}_{\text{obj\_target}} = \mathbf{p}_{\text{home}} - [0, 0, 0.079]$) y criterio ampliado para sujeción física firme (`is_clamped = True`).
   - *Fenomenología y Frente de Pareto (`train_v8_300data`):* Identificación empírica de 3 fases: sub-ajuste (época 10, $0\%$), meseta de destreza y mínima latencia (épocas 50 a 200, $33.3\%$ éxito, $E^* = 50$, época 150 con tiempo récord de $99.5\text{ s}$), y sobreajuste motor severo (épocas 250 a 300, $0\%$ éxito por desincronización de garra).
   - *Memoria Modular:* `.agents/knowledge/13_multicheckpoint_evaluation_and_pareto_convergence.md`.
16. **Prohibición de Datasets Hardcodeados, Doble Eje X Obligatorio y Protección de Pesos:**
   - *Dataset Dinámico Obligatorio:* Ninguna herramienta analítica o de visualización puede asumir o tener quemado en strings el nombre del dataset (ej. `demonstrations_v8.zarr`). Toda función debe parametrizar `dataset_name` y reflejar el dataset real evaluado en títulos y leyendas.
   - *Doble Eje X Sincrónico:* Todo gráfico de pérdida temporal debe mostrar de forma explícita el eje X inferior en **Pasos de Optimización** y el eje X superior en **Épocas de Entrenamiento** ($1\text{ Época} = N\text{ Pasos}$).
   - *Inviolabilidad de Pesos:* Las herramientas de análisis, graficado y re-evaluación operan estrictamente en modo de solo lectura sobre los checkpoints y `model.safetensors`.
   - *Memoria Modular:* `.agents/knowledge/14_historical_convergence_and_noise_floor.md`.
17. **Odometría Visual e Inercial con RTAB-Map, RealSense D415 (VO) y D435i (VIO):**
   - *Arquitectura Desacoplada:* Captura manual de aproximación (*approaching*) en hardware real. Soporta dos modalidades independientes y coexistentes:
     - **D415 (VO puro):** [d415_camera_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/d415_camera_params.yaml) + [realsense_d415_vo.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d415_vo.launch.py). Medido en reposo: $30.63\text{ Hz}$, jitter $0.394\text{ mm}$, $300.5\text{ inliers}$. Que los puntos se anclen al patrón IR no está verificado; si ocurriera, sería un defecto, porque el patrón se mueve con la cámara.
     - **D435i (odometría RGB-D con el marco inicial alineado a la gravedad por el IMU BMI085):** [d435i_camera_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/d435i_camera_params.yaml) + [rtabmap_vio_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml) + `imu_filter_madgwick` + [realsense_d435i_vio.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d435i_vio.launch.py). `rgbd_odometry` F2M (GFTT/ORB sobre la imagen RGB, PnP + RANSAC, subpíxel 5×5/15, Kalman), con `GuessMotion: false` y `BundleAdjustment: 0`. El IMU solo alinea el marco inicial con la gravedad (`wait_imu_to_init`): no es un VIO acoplado. Origen en la cámara (`Odom/AlignWithGround: false`). **Precisión sin medir contra referencia:** las cifras de jitter y 0 % de pérdidas se tomaron en reposo y no representan la toma real (ver §20 y `context.md` lección 43).
   - *Recolector FSM a 20 Hz:* `vo_collector_node.py` v3 (2026-11-24; v2 del 2026-11-23):
     - rejilla regular de 20 Hz con sellos de tiempo, pose interpolada y la imagen de sello más cercano;
     - la odometría nula de RTAB-Map se marca como perdida (nunca se guarda como pose);
     - `reset_odom` al empezar cada episodio;
     - done verificado (quieto y ≥ `min_hold_duration_s`, sin forzarlo en frames en movimiento);
     - control de calidad que bloquea el guardado: [D] descartar o [F] forzar;
     - metadatos de trazabilidad;
     - **v3:** guía de inicio a la vista de HOME3. Compara los ArUco de la mesa con `home3_view_reference.yaml`,
       generado desde los intentos reales. Muestra la imagen `~/start_guide` y no empieza fuera de tolerancia;
       [I] empieza igual y queda registrado.
     - **v3, opcionales:** inicio automático (vista alineada y quieta), guardado automático (quieta al final), aviso de
       velocidad > 0,15 m/s y programa de celdas × orientaciones (`schedule_file`).
     - **Logs en rclpy:** nunca `(warn if x else info)(msg)` en una misma línea. Si la severidad cambia entre
       llamadas, rclpy lanza `ValueError` y se cae el ejecutor; al recolector v2 le pasaba con el primer frame perdido.
     - **Recompilar tras cambiar el código o el YAML:** `install/visuomotor_data_collection` es una copia.
     - **Medición de la odometría** (sin robot): tablero ChArUco A3 9×7 de 36 mm y bags MCAP.
       - El tablero se usa solo en las corridas de medición, en el centro de la mesa.
       - Las demos v8 se graban sin tablero, con la misma escena que v7 y que los ensayos.
       - `check_board_view.py` comprueba que se vean las 48 esquinas desde HOME3.
       - `record_vio_bag.sh` graba las corridas.
       - `evaluate_pose_estimates.py` compara contra el tablero.
       - `replay_odometry_candidates.py` reprocesa los mismos bags con cada candidato.
       - `make_synthetic_vio_bag.py` genera un bag con verdad conocida para probar la cadena.
       - Plan en `docs/plans/plan_d435i_herramienta_demostraciones.md`, sección "Trabajo sin el robot".
     Imagen 256×256 RGB; estado 8D sin fuga causal; acción 9D. Lanzadores de recolección dedicados: [realsense_d415_vo_collection.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d415_vo_collection.launch.py) y [realsense_d435i_vio_collection.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d435i_vio_collection.launch.py).
   - *Lanzador y Permisos USB/Sysfs:* `tools/hardware/run_vo_container.sh` con passthrough dinámico de `/dev`, `/sys` y `/dev/bus/usb`, y preservación estricta de propiedad de archivos (`hackbrian:hackbrian`, UID 1000). Script de configuración udev: `tools/hardware/setup_realsense_udev.sh` (con desactivación de `iio-sensor-proxy` para evitar contención del sensor inercial).
   - *Memoria Modular:* `.agents/knowledge/15_visual_odometry_rtabmap_realsense.md`.
18. **Manipulador Real xArm5 (UFACTORY, 5 GDL) en Jetson Orin Nano:**
   - *Cadena:* `policy_node` → `target_frame_raw` → `pose_smoother` → `target_frame` → `xarm5_bridge` → `/xarm/set_servo_cartesian` (modo 1). TF `link_base→xarm_tcp` y `/visuomotor/gripper_joint_states` (escala de la política) los publica el puente.
   - *Seguridad obligatoria:* el puente arranca **desarmado** y con `dry_run: true`; solo `~/arm` habilita movimiento. Caja de trabajo, límites de velocidad, rechazo de objetivos lejanos, watchdog y desarme ante error del controlador. Nunca desactivar estas protecciones en código; ajustarlas en `visuomotor_bringup/config/xarm5/xarm5_real_params.yaml`.
   - *5 GDL:* orientación 6D no alcanzable. `orientation_mode: arm_plane` (defecto) = inclinación solo en el plano vertical del brazo (±`arm_plane_pitch_*`, 1.48 rad ≈ 85°; tope < π/2 por bloqueo de cardán del rpy del controlador) + giro J5, interpolado en (pitch, yaw); con pitch 0 equivale a `top_down_yaw`. RViz2: anillo verde (inclinación, eje según acimut, ajuste al soltar) y azul (giro de la herramienta).
   - *Dataset real → xArm5:* entrenar SOLO con la salida de `tools/xarm5/convert_vio_dataset_to_xarm5.py`, que lleva la cámara en `odom` al TCP en `link_base` (home = HOME3 de `jetson_policy_params.yaml`, extrínseca en `xarm5_dataset_alignment.yaml`), proyecta con el modo del puente y audita. Escribe una copia nueva y nunca sobrescribe.
   - *Teleop RViz2 remota:* Jetson corre `xarm5_real.launch.py` (driver + puente + `robot_state_publisher` + `xarm5_teleop_marker`); el otro equipo solo `xarm5_rviz.launch.py`. `ROS_DOMAIN_ID=42` + `rmw_cyclonedds_cpp` para el robot real (mismo dominio que la simulación: con el brazo conectado, simulación y políticas solo con `ROS_LOCALHOST_ONLY=1`); DDS por `wlan0` (`deploy/jetson/config/cyclonedds.xml`).
   - *Frenos:* tras habilitar motores, todos los ejes deben reportar freno liberado (bit 1 en `mt_brake`) y motor habilitado (bit 1 en `mt_able`) o no se envía movimiento; `~/check_brakes` solo lee. No usar `motion_enable` data=0 con el brazo en el aire (caída de ~5 cm medida).
   - *Base móvil:* el xArm5 va montado sobre una base móvil tipo Husky A200, con una Kinect 360 en el frente (este proyecto no la usa). En HOME3, la cámara de la brida ve abajo el frente amarillo de la base, la Kinect y los salientes de las ruedas (~18 % de la imagen); también salen en el primer frame de las demos v7, grabadas con la base frente a la mesa. Grabar las demos con la base en su lugar (`docs/plans/plan_d435i_herramienta_demostraciones.md`, "Riesgos nuevos").
   - *HOME1/HOME2 (fabricante, modelo XF1305):* HOME1 = Zero Position [0,0,0,0,0] (pose de transporte UFACTORY), HOME2 = Initial Position por defecto del xArm 5 [0,0,0,-90°,0]; HOME3 = pose de trabajo del usuario (redefinida 2026-11-20: TCP (0.150, 0.005, 0.750) m, garra inclinada ~−24.6°; juntas leídas del brazo) y home de la política; servicios `~/go_home1`/`~/go_home2`/`~/go_home3` y menú RViz2 en inglés (doble clic). `unwind_j1` solo por terminal. El marcador ofrece X/Y/Z, giro y (con `arm_plane`) inclinación en el plano del brazo; volumen = caja (radio ≤ 0.65 m, z ≤ 0.75 m) ∩ alcance hombro–muñeca (`workspace_max_reach_m` 0.6086 = 95% del codo recto, `workspace_min_reach_m` 0.12), válido también con la garra inclinada; la altura máxima baja con el radio (0.75 m cerca de la base, 0.37 m a 0.65 m).
   - *Límites articulares:* el puente no arma con una junta a < `joint_limit_margin_rad` de su límite y se detiene antes del C23; J1 en ±2π se corrige solo con `~/unwind_j1` (manual, giro de base de 360°). Rangos del Hardware Manual v2.7.1 para XF1305 (`bio/xarm5_ufactory_XF1305/`): J1 ±360°, J2 -117°..116°, J3 -219°..10°, J4 -97°..180°, J5 ±360° (más estrechos que la URDF/manual V2.3.0 usados hasta el 2026-11-22).
   - *Alcance vs. caja:* alcance nominal 700 mm (rango cartesiano X/Y ±700 mm) = radio máximo alrededor del eje de J1 a la altura del hombro (J2, 0.267 m); a la altura de aproximación de las demostraciones (brida ~0.28 m, garra vertical) el alcance lateral cinemático es ±0.47 m a la profundidad de los objetos. La caja `|y| <= 0.30 m` del puente es una decisión de seguridad de la celda (conservadora, ampliable tras medirla), no un límite del brazo. Herramienta: `tools/xarm5/analyze_xarm5_workspace.py`.
   - *Driver:* `xarm_ros2` rama `humble` fijado por commit y compilado en `/opt/xarm_ws` dentro de `deploy/jetson/Dockerfile.jetson`. `xarm_msgs` no es dependencia rosdep (no se declara en `package.xml`).
   - *Cámara e inferencia en la Jetson (2026-11-23):*
     - driver D435i en `Dockerfile.realsense_base` (librealsense 2.59.1, realsense-ros 4.59.1, RSUSB), activado con `docker-compose.realsense.yml`; cámara con `d435i_camera.launch.py`;
     - inferencia con CUDA Graph y 5 pasos DDIM: 69 ms medidos (205 ms con 10 pasos sin grafo) y 20 Hz de comandos;
     - `benchmark_jetson.py` y `check_policy_closed_loop_sim.py` miden lo que hace `policy_node`;
     - detalle en `context.md`, lección 44.
   - *Memoria Modular:* `.agents/knowledge/16_xarm5_real_hardware_bringup.md`.
19. **Unificación de Coordenadas Relativas (Modo SOTA UMI / Delta-Home):**
   - *Invariante Arquitectónica:* Para eliminar la dependencia de marcos globales arbitrarios o de la cota del piso en RTAB-Map (`Odom/AlignWithGround: false`), todo episodio arranca en el origen estricto: $\mathbf{p}_{\text{rel}} = (0.0, 0.0, 0.0)$ en el paso $t=0$.
   - *Recolección Real e Isaac Sim:* Activado mediante `relative_coordinates: true` en `vo_data_collection_params.yaml` y `data_collection_params.yaml`. En simulación resta la pose inicial Home ($\mathbf{p}_t - \mathbf{p}_{\text{home}}$); en recolección manual resta la pose inicial del lente ($\mathbf{p}_t - \mathbf{p}_{\text{start}}$). Certificado en metadatos Zarr (`ep.attrs['relative_coordinates'] = True`) y auditado por `tools/verify_zarr_dataset.py`.
   - *Inferencia en Robot:* `policy_node.py` resta `home_pos` a `observation.state[0:3]` y suma `home_pos` a las trayectorias predichas en `action[0:3]` (soporta vectores 1D y chunks 2D) antes de publicarlas hacia `ActionChunk` / `xarm5_bridge`, manteniendo consistencia sim-to-real idéntica sin requerir tags en la mesa. Estabilización estricta de cuaternión en semiesfera superior ($q_w \ge 0$) y norma unitaria. Test suite: `tests/test_relative_coordinates.py` (15/15 passed).

20. **Odometría de la D435i (sin certificar), Recolección con Auto-Incremento y Topología de Red Dual (Local vs. Jetson Orin):**
    - *Odometría SIN certificar (revisión 2026-11-23):* 2.307 mm fue el mejor de 9 retornos a la marca de la conf 4: mediana 21.6 mm, máximo 59.7 mm, solo 1 de 9 ≤ 3 mm. El jitter en reposo tiene mediana 3.0 mm (1.6–6.1). El error de retorno no mide ni la trayectoria ni la escala. La grabación v7 tiene poses nulas guardadas como poses (ep. 23 y 24), 784 discontinuidades > 2 cm en 71/83 episodios y ningún sello de tiempo (`analyze_vio_dataset_quality.py --raw-diagnostics`). Toda cifra de precisión se reporta como distribución medida contra una referencia (xArm5 o tablero ChArUco), nunca como el mejor intento. Plan: `docs/plans/plan_d435i_herramienta_demostraciones.md`.
    - *Topología de Red Aislada (Zero-Latency Local):* Para recolección de demostraciones expertas a mano alzada en la estación de trabajo, el stream de video de $46\text{ MB/s}$ de la RealSense y la odometría se confinan a Memoria Compartida (SHM) / loopback local (`lo`) bajo `ROS_DOMAIN_ID=0` con aislamiento estricto `ROS_LOCALHOST_ONLY=1` y `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`. Esto blinda la estación de trabajo para que ningún otro equipo con ROS 2 en la misma red Wi-Fi o laboratorio pueda ver o interferir con los tópicos, erradicando al 100% la saturación de red y garantizando frecuencia canónica a $20.0\text{ Hz}$ sin latencia interna.
    - *Topología de Red Remota (Jetson Orin xArm5):* El despliegue de inferencia y teleoperación remota con el manipulador xArm5 en la Jetson Orin opera bajo `ROS_DOMAIN_ID=42` vía CycloneDDS Unicast (`DDS_PEER=<ip_jetson>` o `deploy/jetson/config/cyclonedds.xml`), transmitiendo únicamente telemetría ligera de control (`ActionChunk`, `/target_frame`, `/policy/metrics` a $< 50\text{ KB/s}$) sin interferir con el bus de video local.
    - *Flexibilidad de Recolección Zarr:* `vo_collector_node` y `run_data_collection.sh` aceptan `dataset_path`:
      - Modo `AUTO`: Genera automáticamente `demonstrations_real_vX.zarr` incremental escaneando versiones existentes.
      - Modo `AUMENTO`: Normaliza la ruta hacia `datasets/` (ej. `demonstrations_real_v5`), detecta episodios previos y añade episodios nuevos consecutivamente (`episode_N`).
    - *Desacoplamiento de Gripper:* Vector de estado 8D y acción 9D se preservan con `0.0` para pinza (sin sesgo en entrenamiento, compatible sim-to-real con `demonstrations_v8.zarr`), tecla interactiva `[G]` para alternar abierto/cerrado, y lectura automática opcional de `/visuomotor/gripper_joint_states`.
    - *Memoria Modular:* `.agents/knowledge/15_visual_odometry_rtabmap_realsense.md` y `.agents/knowledge/16_xarm5_real_hardware_bringup.md`.
21. **Calibración, Conversión SOTA y Certificación del Dataset Real v7 para xArm5:**
    - *Dataset Compilado Obligatorio:* `datasets/demonstrations_real_v7_xarm5.zarr` (81 episodios limpios, $26,868$ pasos a 20.0 Hz). Los 2 episodios anómalos de pérdida de tracking ($v > 2.3\text{ m/s}$) fueron excluidos con `--drop-flagged`.
    - *Parámetros de Calibración en `xarm5_dataset_alignment.yaml`:*
      - `extrinsic_calibrated: true`, `tcp_to_camera_xyz_m: [0.0, 0.0, 0.0]`, `tcp_to_camera_rpy_rad: [0.0, -1.5707963, 0.0]` (montaje en brida hacia $+Z_{\text{tcp}}$ con pitch $-90^\circ$).
      - `speed_tolerance_ratio: 12.0` (hasta $0.60\text{ m/s}$ y $6.0\text{ rad/s}$ para absorber movimientos manuales sin sesgo).
      - `max_orientation_deviation_rad: 1.50` (proyección matemática al plano vertical alcanzable `arm_plane`).
      - `max_workspace_violation_frames: 550` (despegue continuo desde HOME3: $X=0.15009\text{ m}, Z=0.75000\text{ m}$).
    - *Certificación de Tensores:* Compatible al 100% con PyTorch DataLoader (batch de 16: imágenes $[16, 2, 3, 256, 256]$, estado $[16, 2, 8]$, acción $[16, 16, 9]$) y paso forward+backward de `DiffusionPolicy` verificado en CUDA (Loss DDIM: $1.1839$).
    - *Suite de Evaluación y Registro SQLite:* `tools/evaluate_real_dataset.py` genera paneles multicanal 2x3 a 300 DPI y PDF 100% en inglés académico (`demonstrations_real_v7_xarm5_sota_panel.png/.pdf` y `demonstrations_real_v7_sota_panel.png/.pdf`). Subfigura D (Eficiencia Geométrica: $D = 0.60\text{ m}$ vs $L_{\text{eff}} = 1.12\text{ m}$, tortuosidad $1.88 \pm 0.40$), Subfigura E (Espacio de Fases: $\mathbf{v}(d_{\text{target}})$ con desaceleración en zona de contacto $d \le 5\text{ cm}$), y registro relacional en `experiments/experiments.db`.
    - *Cláusula de Estado Operativo:* **Hasta esta fase, el dataset está 100% certificado y listo, pero el modelo aún NO ha sido evaluado en el robot físico ni se tienen resultados de despliegue en hardware real.**
    - *Memoria Modular:* `.agents/knowledge/16_xarm5_real_hardware_bringup.md` §5r.
    - **SUPERADO por el punto 22:** la auditoría física de la Fase 22 encontró la extrínseca invertida 180°, saltos de signo del cuaternión, ruido VIO de ~8.8 mm RMS y umbrales de auditoría relajados. No entrenar con `demonstrations_real_v7_xarm5.zarr`.

22. **Curación Física del Dataset Real v7, Acciones Relativas (UMI) y Entrenamiento para el xArm5 (Fase 22):**
    - *Dataset obligatorio:* `datasets/demonstrations_real_v7_xarm5_v2.zarr` (76 episodios, 23,731 frames, conversor v2). Extrínseca `tcp_to_camera_rpy_rad: [pi, -pi/2, 0]` determinada con la gravedad del VIO y los ArUco de la mesa (arriba de la imagen = lejos de la base en HOME3), filtro de trayectoria (Hampel + Butterworth 1.5 Hz de fase cero), `quaternion_hemisphere: home`, 7 episodios descartados (giros de muñeca ~180° y pérdidas de tracking).
    - *Reglas de datos reales:* una "certificación" de integridad/tensores no basta: auditar la física (gravedad, continuidad del cuaternión, PSD/ruido, velocidad real, volumen del puente). La odometría de `vo_collector` no se reinicia entre episodios: usar solo movimiento relativo dentro del episodio. Nunca relajar umbrales de auditoría para que un dataset "pase".
    - *Entrenamiento:* `--action_representation relative_ee` (piloto en validación: ADE 12.6 mm vs 38.9 mm con `absolute`), EMA de pesos, warmup, `--val_ratio` (split por episodio) con evaluación offline por muestreo completo (ADE/FDE/rotación/F1 done/progreso vs línea base "quedarse quieto"), `--color_jitter` (GPU), `--resume`, `--max_rss_gb`.
    - *Recursos del host (31 GB RAM):* UN entrenamiento pesado a la vez (máx. dos con `--num_workers 6 --prefetch_factor 2`). Tres simultáneos provocaron un OOM global que cerró la sesión gráfica del usuario.
    - *Despliegue:* `policy_node` con `quaternion_hemisphere: auto|home`, `image_resize_method: cv2_area`, `neural_auto_stop_patience_steps`; `xarm5_real.launch.py` con `xarm5_pose_smoother_params.yaml` (1.5 Hz). Verificación sin hardware: `tools/xarm5/replay_policy_offline.py` y `tools/xarm5/check_policy_closed_loop_sim.py`. Recompilar `install/` con `--symlink-install` y reconstruir la imagen de la Jetson.
    - *Modelo entregado:* `deploy/jetson/model/` = época 200 de `visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300` (300 épocas). Validación (8 episodios): ADE 11.6 mm, ADE ejecutado 7.6 mm (línea base 11.0), FDE 18.5 mm, 1.74°, F1 done 0.70, progreso 1.00. Auto-stop calibrado 0.60 x 5 pasos (100% paradas, 0 prematuras, ~8 mm). Lazo cerrado ROS 2 simulado: coseno 0.99, 0 rechazos.
    - *Selección de checkpoints:* los criterios de parada basados en la pérdida (punto 14) recomendaban la época ~8; las métricas de comportamiento mejoraron hasta ~200. En DP se elige con métricas de muestreo completo y reproducción de la ruta de despliegue, no con la pérdida.
    - *Líneas base exigentes (obligatorio):* "quedarse quieto" es una vara baja. `evaluate_sampling` reporta además "velocidad constante" (`cv_ade_mm`, `cv_ade_exec_mm`, `cv_fde_mm`; desfase acción/estado estimado del dataset). Época 200 en validación: la política pierde con la velocidad constante en los 3 pasos ejecutados (3.6 vs 1.3 mm) y gana desde ~0.53 s (FDE 18.5 vs 21.3 mm). Ablación visual (`tools/diagnose_policy_baselines.py`): con imágenes de otro episodio el FDE sube a 29.2 mm (la política sí usa la imagen); desde el reposo acierta la dirección en el 64 % frente al 55 % de "ir siempre hacia la dirección media" (n = 22): los datos v7 apenas obligan a usar la visión.
    - *Estado (2026-11-23):* **evaluado en el xArm5 físico (14 intentos, 5 cm/s).**
      - 9/14 llegaron y se detuvieron solos sobre la herramienta (IC de Wilson al 95 %: 39–84 %).
      - 7/14 terminaron con la vista de las demos (IC 27–73 %).
      - Funciona con las tijeras rectas en las zonas que cubren las demos v7.
      - Falla con las tijeras giradas (intentos 9 y 10) y junto a la base (11).
      - El intento 12 (centro, tijeras rectas) reconoció el final una sola inferencia (done 0,66) y siguió bajando.
      - El intento 14 tenía otra escena (mantel corrido y sin ArUco): la señal de término no pasó de 0,11.
      - **Cuando la política no reconoce el final, sigue bajando.** El piso del vigilante (brida a z ≥ 0,15 m) la
        detuvo 4 veces: es obligatorio en todo ensayo.
      - No usar la altura de las demos vecinas como criterio: la altura de parada humana sobre un mismo lugar varía
        ±7 cm.
      - Resultados en `experiments/xarm5_real_trials/2026-11-23/`, herramienta `tools/xarm5/real_trials/run_trial.sh`; detalle en `context.md`, lección 45.
    - *Memoria Modular:* `.agents/knowledge/17_real_vio_dataset_curation_and_relative_actions.md`.

---

## 7. Comandos de Referencia Rápida

```bash
# 1. Compilación del workspace dentro del contenedor
docker exec -u hackbrian -w /ros visuomotor_ros2 bash -c "colcon build --symlink-install"

# 2. Lanzar recolección autónoma de datos (genera automáticamente demonstrations_vX.zarr)
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_data_collection pickup_approximate.launch.py"

# 3. Entrenar Diffusion Policy con el dataset parchado o nuevo
docker exec -it -u hackbrian visuomotor_ros2 bash -c "python3 /ros/visuomotor_core/train.py --root /ros/visuomotor_core/datasets/demonstrations_v7_done_fixed.zarr --epoch 100 --batch_size 64 --out_dir /ros/visuomotor_core/outputs/train_500_dataV7_fixed"

# 4. Lanzar inferencia en vivo con el nodo de política
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_ros2 inference.launch.py"

# 5. Lanzar fine-tuning por refuerzo (DPPO RL)
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_ros2 rl_training.launch.py"

# 6. Monitoreo en tiempo real con rqt_plot en el Host o contenedor
ros2 run rqt_plot rqt_plot /policy/metrics/done_flag/data /policy/metrics/done_threshold/data /policy/metrics/gripper_cmd/data /policy/metrics/gripper_state/data
ros2 run rqt_plot rqt_plot /rl/metrics/instant_reward/data /rl/metrics/episode_reward/data /rl/metrics/critic_value/data

# 7. Ejecutar campaña de evaluación comparativa pareada (Solo DP vs. DP + RL)
docker exec -it -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && source /ros/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export ROS_LOCALHOST_ONLY=1 && export ROS_DOMAIN_ID=42 && python3 tools/compare_models.py --episodes 10 --max_steps 1500 --seed 123"

# 8. Regenerar figuras comparativas 2x3 de publicación desde el CSV (sin simulación)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/replot_benchmark.py

# 9. Compilación limpia y auditoría de informe científico en LaTeX
pdflatex -interaction=nonstopmode informe_avance_visuomotor_cd.tex && biber informe_avance_visuomotor_cd && pdflatex -interaction=nonstopmode informe_avance_visuomotor_cd.tex

# 10. Orquestador Maestro Headless (Isaac Sim acelerado 3x-7x + ROS 2 Jazzy)
./tools/sim/run_headless_system.sh sim-only     # Simulación headless + control cartesiano
./tools/sim/run_headless_system.sh collect      # Recolección experta (--episodes N --dataset RUTA)
./tools/sim/run_headless_system.sh rl           # Fine-tuning por refuerzo DPPO (--checkpoint RUTA)
./tools/sim/run_headless_system.sh benchmark    # Benchmarking pareado (--episodes N --dp_ckpt RUTA --rl_ckpt RUTA)
./tools/sim/run_headless_system.sh inference    # Inferencia en vivo de política (--checkpoint RUTA)
./tools/sim/run_headless_system.sh stop         # Detener simulación, nodos y liberar VRAM de inmediato

# 11. Analizar convergencia y criterios de parada SOTA sobre curvas de entrenamiento
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/analyze_convergence.py --run_dir visuomotor_core/outputs/train_v8_continuous_convergence

# 12. Entrenamiento Continuo Dinámico (Épocas ∞, Checkpoints cada 10 épocas, Render en vivo cada 500 steps)
docker exec -it -u 1000 -w /ros visuomotor_ros2 bash -c "python3 /ros/visuomotor_core/train.py --root /ros/datasets/demonstrations_v8.zarr --epoch 0 --batch_size 64 --save_every 10 --log_freq 10 --plot_freq 500 --out_dir /ros/visuomotor_core/outputs/train_v8_continuous_convergence"

# 13. Parada Limpia a Demanda del Entrenamiento (sin pérdida de datos ni corrupción)
touch visuomotor_core/outputs/train_v8_continuous_convergence/STOP

# 14. Recolección y Visualización de Demostraciones Reales (RealSense D415 + RTAB-Map VO)
./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d415_vo.launch.py gui:=true
docker exec -it -u 1000 visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=0 && ros2 run visuomotor_data_collection vo_collector"
docker exec -u 1000 visuomotor_ros2 python3 tools/visualize_dataset.py --dataset datasets/demonstrations_real_v2.zarr --out-dir datasets/visualizations_real_v2

# 15. Recolección con la D435i + RTAB-Map (odometría RGB-D; la prueba de retorno es solo un chequeo rápido, no mide precisión)
./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false
./tools/hardware/run_benchmark.sh "conf 4"                          # Chequeo rápido: jitter en reposo y retorno a la marca
./tools/hardware/run_data_collection.sh                              # Recolección AUTO (crea demonstrations_real_vX.zarr)
./tools/hardware/run_data_collection.sh demonstrations_real_v5       # Continuar aumentando dataset existente

# 16. Dataset real -> xArm5 (conversor v2: auditoría, filtro, hemisferio) y diagnóstico físico con figuras
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_v7.zarr --dst datasets/demonstrations_real_v7_xarm5_v2.zarr --drop-flagged
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/analyze_vio_dataset_quality.py --raw datasets/demonstrations_real_v7.zarr --old datasets/demonstrations_real_v7_xarm5.zarr --new datasets/demonstrations_real_v7_xarm5_v2.zarr --out-dir datasets/evaluations_real_v7_xarm5_v2

# 17. Entrenamiento real xArm5 (relative_ee, EMA, validación por episodio, guardia de RAM; reanudable con --resume)
docker exec -d -u 1000 -w /ros visuomotor_ros2 bash -c "python3 visuomotor_core/train.py --root /ros/datasets/demonstrations_real_v7_xarm5_v2.zarr --epoch 300 --batch_size 64 --num_workers 6 --prefetch_factor 2 --max_rss_gb 12 --save_every 10 --eval_every 10 --val_ratio 0.1 --action_representation relative_ee --color_jitter --out_dir /ros/visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300 > /ros/visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300.stdout 2>&1"

# 18. Verificación de despliegue sin hardware (reproducción de la ruta de la Jetson y lazo cerrado ROS 2 con puente simulado)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/replay_policy_offline.py --checkpoint <ckpt> --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr --episodes val --out-dir <dir>
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && source /tmp/xarm_ws/install/setup.bash && source install/setup.bash && python3 tools/xarm5/check_policy_closed_loop_sim.py --checkpoint <ckpt> --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr --episode episode_45 --out-dir <dir>"

# 19. Política vs. líneas base (quedarse quieto, velocidad constante) y ablación visual, sobre varios checkpoints
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/diagnose_policy_baselines.py --run-dir visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300 --epochs 100,150,200,250,300 --highlight 200 --out-dir visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300/baseline_diagnostics

# 20. Alcance del xArm5 (manual + modelo del puente) vs. caja de seguridad vs. demostraciones (con caja propuesta opcional)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/analyze_xarm5_workspace.py --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr --out-dir datasets/evaluations_real_v7_xarm5_v2 --propose-y-max 0.45

# 21. Referencia visual de HOME3 (primer frame de las demos) para comparar con la cámara montada antes de armar el brazo
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/export_home3_reference.py --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr

# 22. Diagnóstico de una grabación cruda de la D435i (poses nulas, duplicados, discontinuidades, done en movimiento)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/analyze_vio_dataset_quality.py --raw datasets/demonstrations_real_v7.zarr --raw-diagnostics --out-dir datasets/evaluations_real_v7_xarm5_v2

# 23. Tablero ChArUco A3 9x7 de 36 mm (PDF escrito sin matplotlib, a escala 1:1; imprimir al 100 %, medir con calibre y poner measured: true en charuco_board.yaml)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/hardware/make_charuco_board.py

# 25. Ensayo en el xArm5 real desde el PC (brazo en HOME3, puente lanzado por el usuario, usuario en la parada de emergencia)
tools/xarm5/real_trials/run_trial.sh --dry-run                # etapa B (puente en dry-run; no mueve)
tools/xarm5/real_trials/run_trial.sh 9 "derecha-lejos"        # etapa C, intento 9 (modo real), con métricas y gráfica
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/plot_real_trial.py experiments/xarm5_real_trials/2026-11-23/trial_008   # gráfica de control

# 24. Calibración cámara-brida con el xArm5 y el tablero (capture: ROS 2 con cámara + TF del robot; solve: sin ROS, no edita YAML)
python3 tools/xarm5/calibrate_hand_eye.py capture --out-dir datasets/hand_eye/<fecha>
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/calibrate_hand_eye.py solve --samples datasets/hand_eye/<fecha>

# 26. Recolección v8 con guía de inicio a HOME3, inicio/guardado automáticos, programa y bags crudos (ver la guía: rqt_image_view /vo_collector_node/start_guide)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/hardware/make_collection_schedule.py --out experiments/collection_schedules/v8_schedule.csv
COLLECTOR_ARGS="-p auto_start:=true -p auto_save:=true -p schedule_file:=experiments/collection_schedules/v8_schedule.csv -p raw_bag_enabled:=true" ./tools/hardware/run_data_collection.sh demonstrations_real_v8

# 27. Exposición/ganancia/balance automáticos de la D435i -> valores manuales (contenedor de la cámara corriendo)
docker exec -it -u 1000 -w /ros visuomotor_vo_active bash -c "source /opt/ros/jazzy/setup.bash && python3 tools/hardware/read_camera_auto_settings.py"

# 28. Medición de la odometría contra el tablero ChArUco: grabar corridas, evaluar la odometría en vivo y reprocesar con cada candidato
./tools/hardware/record_vio_bag.sh reposo_home3 15
./tools/hardware/record_vio_bag.sh aprox_01
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && python3 tools/hardware/evaluate_pose_estimates.py --run datasets/vio_bags/<bag> --out-dir experiments/vio_measurement/<fecha>"
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && python3 tools/hardware/replay_odometry_candidates.py --run datasets/vio_bags/<bag1> --run datasets/vio_bags/<bag2> --out-dir experiments/vio_replay/<fecha>"

# 29. Probar la cadena de medición sin cámara (bag sintético con verdad conocida)
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && python3 tools/hardware/make_synthetic_vio_bag.py --out /tmp/syn && python3 tools/hardware/replay_odometry_candidates.py --run /tmp/syn --candidates baseline --allow-nominal --ground-truth-topic /ground_truth/optical_pose --out-dir /tmp/syn_replay"

# 31. Colocar el tablero para medir: desde la vista de HOME3, guía de inicio + esquinas vistas, distancia y ángulo (imagen en /board_check/image)
docker exec -it -u 1000 -w /ros visuomotor_vo_active bash -c "source /opt/ros/jazzy/setup.bash && python3 tools/hardware/check_board_view.py"

# 30. Regenerar la referencia de la vista de HOME3 (si cambian la cámara, su montaje, HOME3 o los ArUco de la mesa)
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/hardware/make_home3_view_reference.py --images "experiments/xarm5_real_trials/<fecha>/trial_*/pre.png"
```

---

## 8. Contrato de Mantenimiento Continuo (Agent Handoff)

Al concluir cada sesión de desarrollo donde se hayan tomado decisiones arquitectónicas, solucionado bugs o actualizado datasets/pesos:
1. El agente activo DEBE auditar el código fuente línea por línea para certificar su estado funcional real.
2. El agente activo DEBE actualizar `context.md` y este archivo `AGENTS.md`.
3. El agente activo DEBE nutrir de forma aditiva la base de conocimiento modular en `.agents/knowledge/` (consultando los diffs reales de los commits).
4. Se debe invocar el protocolo automatizado definido en `UPDATE_KNOWLEDGE.md` (`/update-knowledge`).
