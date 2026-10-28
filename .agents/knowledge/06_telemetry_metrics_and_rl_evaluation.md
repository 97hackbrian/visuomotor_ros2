# Telemetría en Tiempo Real, Métricas y Evaluación Offline (DP e Inferencia / DPPO RL)

Este documento detalla la infraestructura de observabilidad, registro de telemetría en tiempo real para `rqt_plot` y el sistema de exportación offline de métricas (`.csv`, figuras vectoriales `.pdf` y `.png` con `matplotlib`) implementado en `visuomotor_ros2` y `visuomotor_core`.

---

## 1. Justificación y Arquitectura de Telemetría

En robótica visuomotora generativa e interactiva (Diffusion Policy y RL Fine-Tuning), el log textual en consola a altas frecuencias (20 Hz) genera saturación del buffer, pérdida de contexto temporal y dificultad para diagnosticar:
1. Discrepancias de seguimiento cinemático entre predicción y ejecución real.
2. Momentos exactos de binarización del gripper vs comando continuo predicho.
3. Transición del flag neuronal de detención (`action[8]`).
4. Recompensas intermedias y potencial de transporte durante el aprendizaje por refuerzo.

Para solucionar esto de manera desacoplada:
* Se publican tópicos escalares independientes en `std_msgs/msg/Float32` y puntos espaciales en `geometry_msgs/msg/PointStamped`.
* Cada métrica puede ser graficada instantáneamente en `rqt_plot` sin necesidad de plugins o deserializadores propietarios.
* Ambas modalidades (Supervisada y RL) exportan métricas estructuradas a `.csv` y renderizan automáticamente gráficos de calidad de publicación mediante `matplotlib` en backend headless (`Agg`).

---

## 2. Métricas en Inferencia de Diffusion Policy (`policy_node.py`)

### Tópicos Publicados bajo `/policy/metrics/`
| Tópico | Tipo de Mensaje | Frecuencia | Descripción |
| :--- | :--- | :--- | :--- |
| `/policy/metrics/done_flag` | `std_msgs/Float32` | 20 Hz | Valor predicho de la 9ª dimensión (`action[8]`). |
| `/policy/metrics/done_threshold` | `std_msgs/Float32` | 20 Hz | Umbral configurado para auto-stop neuronal (default `0.50`). |
| `/policy/metrics/gripper_cmd` | `std_msgs/Float32` | 20 Hz | Comando continuo predicho para el gripper (`action[7]`). |
| `/policy/metrics/gripper_state` | `std_msgs/Float32` | 20 Hz | Posición articular real del gripper leída de `/joint_states`. |
| `/policy/metrics/tracking_error_xyz` | `std_msgs/Float32` | 20 Hz | Error euclidiano $||P_{\text{cmd}} - P_{\text{actual}}||_2$ en metros. |
| `/policy/metrics/tcp_speed_m_s` | `std_msgs/Float32` | 20 Hz | Velocidad tangencial instantánea estimada del efector final. |
| `/policy/metrics/action_variance` | `std_msgs/Float32` | 20 Hz | Varianza temporal del chunk de acciones predicho (estabilidad). |
| `/policy/metrics/dist_to_home` | `std_msgs/Float32` | 20 Hz | Distancia euclidiana respecto a la pose de reposo (`home_x, home_y, home_z`). |
| `/policy/metrics/tcp_cmd_xyz` | `geometry_msgs/PointStamped` | 20 Hz | Coordenadas cartesianas 3D del comando actual. |
| `/policy/metrics/tcp_actual_xyz` | `geometry_msgs/PointStamped` | 20 Hz | Coordenadas cartesianas 3D medidas del efector final. |

### Comandos de Visualización en Vivo (`rqt_plot`)
```bash
# Panel 1: Auto-Stop Neuronal y Estado de Gripper
ros2 run rqt_plot rqt_plot \
  /policy/metrics/done_flag/data \
  /policy/metrics/done_threshold/data \
  /policy/metrics/gripper_cmd/data \
  /policy/metrics/gripper_state/data

# Panel 2: Precisión Cinemática y Dinámica del Brazo
ros2 run rqt_plot rqt_plot \
  /policy/metrics/tracking_error_xyz/data \
  /policy/metrics/tcp_speed_m_s/data \
  /policy/metrics/dist_to_home/data
```

---

## 3. Métricas en Aprendizaje por Refuerzo DPPO (`rl_train_node.py`)

### Tópicos Publicados bajo `/rl/metrics/`
| Tópico | Tipo de Mensaje | Nivel | Descripción |
| :--- | :--- | :--- | :--- |
| `/rl/metrics/instant_reward` | `std_msgs/Float32` | Paso (PPO Substep) | Recompensa instantánea en el sub-paso actual. |
| `/rl/metrics/episode_reward` | `std_msgs/Float32` | Paso | Recompensa acumulada a lo largo del episodio actual. |
| `/rl/metrics/critic_value` | `std_msgs/Float32` | Paso | Estimación del valor de estado $V(s)$ calculada por la red Crítica. |
| `/rl/metrics/done_flag` | `std_msgs/Float32` | Paso | Señal de término del episodio ($0.0$ o $1.0$). |
| `/rl/metrics/pseudo_force` | `std_msgs/Float32` | Paso | Estimador cinemático de fuerza de agarre sobre el cubo. |
| `/rl/metrics/dist_hover` | `std_msgs/Float32` | Paso | Distancia al punto de aproximación superior (Hover). |
| `/rl/metrics/dist_target` | `std_msgs/Float32` | Paso | Distancia al objetivo de transporte final (Home). |
| `/rl/metrics/r_approach` | `std_msgs/Float32` | Paso | Componente de recompensa de aproximación exponencial. |
| `/rl/metrics/r_orient` | `std_msgs/Float32` | Paso | Componente de alineación angular cuaterniónica. |
| `/rl/metrics/r_grasp` | `std_msgs/Float32` | Paso | Bonus de cierre exitoso con agarre efectivo. |
| `/rl/metrics/r_transport` | `std_msgs/Float32` | Paso | Recompensa potencial de transporte hacia destino. |
| `/rl/metrics/iter_mean_reward` | `std_msgs/Float32` | Iteración | Retorno medio por episodio en la iteración RL. |
| `/rl/metrics/iter_policy_loss` | `std_msgs/Float32` | Iteración | Función de pérdida del Actor PPO (con término BC). |
| `/rl/metrics/iter_critic_loss` | `std_msgs/Float32` | Iteración | Pérdida MSE del Crítico $V(s)$. |
| `/rl/metrics/iter_approx_kl` | `std_msgs/Float32` | Iteración | Divergencia KL estimada para monitorear desplazamiento de política. |

### Comandos de Visualización en Vivo (`rqt_plot`)
```bash
# Monitoreo de Recompensa y Crítico en Tiempo Real
ros2 run rqt_plot rqt_plot \
  /rl/metrics/instant_reward/data \
  /rl/metrics/episode_reward/data \
  /rl/metrics/critic_value/data

# Monitoreo de Descomposición de Recompensas
ros2 run rqt_plot rqt_plot \
  /rl/metrics/r_approach/data \
  /rl/metrics/r_orient/data \
  /rl/metrics/r_grasp/data \
  /rl/metrics/r_transport/data

# Monitoreo de Pérdidas de Optimización por Iteración
ros2 run rqt_plot rqt_plot \
  /rl/metrics/iter_mean_reward/data \
  /rl/metrics/iter_policy_loss/data \
  /rl/metrics/iter_critic_loss/data \
  /rl/metrics/iter_approx_kl/data
```

---

## 4. Registro Offline y Generación de Gráficos

### A. Entrenamiento Supervisado (`visuomotor_core/train.py`)
* **Archivos Generados en `outputs/<run>/`:**
  * `training_metrics.csv`: `[step, epoch, loss, loss_ema, lr, vram_peak_mb, timestamp]` (registrado cada `log_freq=20` pasos).
  * `epoch_metrics.csv`: `[epoch, avg_loss, final_lr, duration_s, timestamp]` (registrado al fin de cada época).
  * `losses/training_curves.png` y `losses/training_curves.pdf`: Panel 2x2:
    1. Loss Crudo vs Loss EMA (Paso a paso).
    2. Pérdida media por época (Macro convergencia).
    3. Cronograma Cosine Annealing de Learning Rate.
    4. Consumo Peak VRAM de la GPU.

### B. Entrenamiento RL DPPO (`visuomotor_ros2/rl_train_node.py`)
* **Archivos Generados en `output_dir/`:**
  * `rl_training_metrics.csv`: `[iteration, elapsed_s, mean_reward, task_completed, policy_loss, critic_loss, approx_kl, is_warmup, timestamp]`.
  * `rl_training_curves.png` y `rl_training_curves.pdf`: Panel 2x2:
    1. Retorno episódico con media móvil de 5 iteraciones.
    2. Pérdida MSE del Crítico $V(s)$.
    3. Pérdida de la Política (Actor PPO + BC).
    4. Divergencia KL estimada vs Target KL.

---

## 5. Parámetros Críticos de Exploración y Alineación de Tarea en RL

1. **Horizonte de Episodio (`max_episode_steps: 1200` a 20 Hz = 60.0s):**
   Garantiza que el manipulador tenga tiempo suficiente para realizar el ciclo completo de aproximación, agarre, elevación y transporte de regreso a Home más el reposo de 3.5s.
2. **Criterio de Validación Terminal con Offset de TCP (`expected_obj_target`):**
   Al considerar la longitud de pinza `tcp_offset_z = 0.079` m, la llegada a Home se valida cuando `dist_target < 0.08` o `dist_ee_home < 0.06` con el cubo debidamente elevado y aprisionado.
3. **Escala de Exploración Multimodal en DDIM:**
   * $\eta = 0.4$ y `min_sampling_denoising_std = 0.04` para exploración activa en el espacio 3D ($\sim 2-3$ cm de dispersión).
   * Factor de ruido en Gripper (`std_eff[:, :, 7] *= 1.8`) para permitir explorar cierre/apertura.
   * Amortiguación de rotación (`std_eff[:, :, 3:7] *= 0.05`) para evitar cabeceo o torsión de muñeca.
4. **Cuadrícula de Respawn Halton y Curriculum en RL (`respawn_mode`):**
   * Publicación de mensajes `geometry_msgs/Twist` en `/respawn` con muestreo Quasi-Monte Carlo de Halton (`scipy.stats.qmc`).
   * Cuadrícula configurable en `rl_training_params.yaml`: `spawn_x: [-0.35, -0.52]`, `spawn_y: [-0.56, -1.08]`, `spawn_z: 0.44`, `spawn_yaw: [-3.1415, 3.1415]`.
   * Modo `on_success`: Retiene la posición del objeto si el episodio previo no tuvo éxito para permitir que el agente explore y corrija el error en esa misma pose, y teletransporta a una nueva pose aleatoria únicamente tras lograr `task_success: True`.
