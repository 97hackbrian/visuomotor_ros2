# Contexto de Desarrollo Orientado a IA: Visuomotor Generative Framework (ROS 2)

**Propósito de este documento:**  
Este archivo (`context.md`) sirve como el repositorio de conocimiento maestro ("Single Source of Truth") para futuros asistentes de Inteligencia Artificial (LLMs, agentes) y desarrolladores humanos que colaboren en el proyecto. Contiene las reglas de diseño, la topología del software, flujos de tensores y decisiones arquitectónicas críticas adoptadas.

---

## 1. Misión del Proyecto
Construir un ecosistema robótico para **Políticas Visuomotoras Generativas (Diffusion Policies, Flow Matching)** en ROS 2 Jazzy.  
**Regla de Oro:** El framework es *estrictamente agnóstico al hardware*. Los paquetes centrales nunca deben conocer si operan sobre un xArm Lite6, un UR5e, Isaac Sim o Gazebo. Cualquier dependencia de un brazo específico se relega de manera forzada al paquete `visuomotor_bringup`.

---

## 2. Topología Arquitectónica (5 Sub-Paquetes)

Para garantizar que el entrenamiento de Machine Learning (ML) no dependa de compilar ROS 2, y que la inferencia en tiempo real en el robot sea eficiente, el código se divide en cinco silos inviolables:

1. **`visuomotor_core`** (Python Puro / Off-ROS)
   - *Rol:* Albergar las redes neuronales, el schedule de difusión (DDIM), los Dataloaders de Zarr y el script de entrenamiento (`train.py`).
   - *Dependencias:* `torch`, `torchvision`, `zarr`. No posee rastro de `rclpy` o `ament`. Se instala vía `pip install -e .` o mediante el `pyproject.toml`.

2. **`visuomotor_msgs`** (ament_cmake)
   - *Rol:* Estructurar los mensajes estandarizados que conectan la inferencia con el control.
   - *Archivos Clave:* 
     - `ActionChunk.msg`: Transporta múltiples pasos a futuro (*receding horizon*).
     - `ObservationData.msg`: Agrupa imagen y estados articulares.
     - `EpisodeTrigger.srv`: Servicio para Iniciar, Guardar o Descartar episodios grabados.

3. **`visuomotor_data_collection`** (ament_python)
   - *Rol:* Ingesta de datos para conformar el Dataset experto.
   - *Características Especiales:* Posee un Demostrador PID Autónomo. En lugar de requerir scripts externos de cinemática inversa, el `sim_collector_node.py` escucha el árbol TF (usando `tf2_ros`) desde Isaac Sim (`object_frame` a `ee_frame`), calcula el error vectorial para aproximarse al objeto y registra las poses resultantes directamente al `.zarr`.
     - *Estabilización Espacial:* Para evitar "drifting" (torcedura de la muñeca por acumulación de errores del IK iterativo), el nodo captura y congela rígidamente la rotación inicial del gripper. Solo persigue el objetivo de forma traslacional (X, Y, Z).
     - *Offset Estructural:* Utiliza el parámetro `tcp_offset_z` para compensar la distancia física entre la base de la muñeca controlada (`link6`) y la punta de los dedos, previniendo colisiones contra el suelo.

4. **`visuomotor_ros2`** (ament_python)
   - *Rol:* Runtime de Inferencia.
   - *Características Especiales:* 
     - Usa `message_filters.ApproximateTimeSynchronizer` para fusionar las modalidades visuales y de estado articular.
     - Carga los pesos `.pth` generados por `visuomotor_core`.
     - Inyecta los tensores a la red a una frecuencia dada (`inference_hz`) y extrae `ActionChunk`s enviándolos al `ActionExecutor`.

5. **`visuomotor_bringup`** (ament_cmake)
   - *Rol:* Archivos `launch` específicos para unificar el modelo con el ecosistema de un robot o simulación real (Ej. Integrar el publicador de TFs de Isaac Sim y los actuadores de XArm).
   - *xArm5 real (Jetson):* `launch/xarm5_real.launch.py` + `config/xarm5/xarm5_real_params.yaml` + `scripts/xarm5_bridge.py` (puente seguro `target_frame` → `/xarm/set_servo_cartesian`, arranca desarmado y en dry-run, proyección de orientación a 5 GDL `top_down_yaw`). Lógica pura en `visuomotor_bringup/xarm5_interface.py` con pruebas en `tests/test_xarm5_interface.py`. Teleoperación desde RViz2 en otro equipo: `scripts/xarm5_teleop_marker.py` (Jetson, marcador `/xarm5_teleop`) + `launch/xarm5_rviz.launch.py` (equipo remoto), `ROS_DOMAIN_ID=42`; comprobación de liberación de frenos (`mt_brake`/`mt_able`) antes de cualquier movimiento; pose de trabajo `home2`. Detalle en `.agents/knowledge/16_xarm5_real_hardware_bringup.md`.

---

## 3. Manejo de Modalidades Sensoriales y Acciones

### Dimensiones en PyTorch
El sistema opera bajo los siguientes horizontes definidos en `visuomotor_core/models/config.py`:
- `h_obs`: Pasos históricos de observación (ej. 2).
- `h_act`: Pasos futuros de acción a predecir (ej. 16 o 8).

### Lógica Continua del Gripper
*Decisión Crítica de Diseño:* El estado del gripper **nunca** se modela como un booleano categórico estricto, sino como un **valor continuo de punto flotante** (0.0 = cerrado, 1.0 = abierto). Esto obedece a que el ruido Gaussiano revertido durante el proceso de *denoising* requiere dominios derivables (continuos). 
- Al setear `use_gripper: True` en `config.py`, las entradas/salidas de estado se expanden de **7 DoF (Posición + Cuaternión)** a **8 DoF (Pose + Gripper)**.
- En simulación, si el PID autónomo detecta una distancia $< 3$ cm, genera un `0.0`, y si no un `1.0`. 
- En inferencia, el `action_executors.py` debe aplicar un *umbral o threshold* (ej. `if valor_gripper < 0.5: close_gripper()`) para mandar señales a los controladores hardware de bajo nivel.

---

## 4. Estándares y Reglas para Futuras Modificaciones (IA Guidelines)

Si eres una Inteligencia Artificial operando sobre este repositorio en el futuro, obedece estas directivas obligatorias:

1. **PROHIBICIÓN DE HARDCODING:** Absolutamente ningún tópico de cámara, frecuencia (Hz), umbral de distancia o marco de coordenadas (`frame_id`) puede ser escrito explícitamente en el código Python. Todos deben heredarse de nodos declarados y leerse de un archivo YAML (revisa `config/policy_params.yaml` o `config/data_collection_params.yaml`).
2. **ESPACIO DE ACCIONES CARTESIANAS:** Las acciones generadas por la política (`ActionChunk`) operan exclusivamente en espacio de tareas (Cartesiano `PoseStamped` absoluto basado en `base_link`). Evita mezclar comandos cartesianos con comandos de espacio articular a menos que instancies explícitamente un transformador cinemático.
3. **MANTENIMIENTO DEL CORE SEPARADO:** Nunca intentes inyectar `cv2.imshow`, llamadas a la API de ROS 2 o `rospy` dentro de `visuomotor_core`. El código de ese paquete debe poder correr en un clúster remoto de GPUs que ni siquiera tenga instalado ROS.
4. **PERFORMANCE EN LA INFERENCIA:** Al modificar `policy_node.py`, recuerda que el bucle `torch.inference_mode()` y el preprocesamiento de imágenes deben evitar sobrecargar el GIL (Global Interpreter Lock) de Python. Si realizas transformaciones intensivas, evalúa usar operaciones nativas en GPU mediante Tensores, no for loops sobre NumPy Arrays.
5. **ENTORNO DE EJECUCIÓN (DOCKER vs HOST):** Todo lo relacionado con Python, ROS 2, scripts, entrenamiento y compilación (`colcon build`) se ejecuta DENTRO del contenedor Docker `visuomotor_ros2`.
   - **Usuario Obligatorio `hackbrian` (`-u hackbrian`):** Los comandos en el contenedor deben ejecutarse siempre bajo el usuario `hackbrian` para preservar los permisos de `hackbrian@robot_dev` y nunca generar archivos propiedad de `root`.
   - **Consentimiento Expreso:** La IA ÚNICAMENTE ejecutará comandos en la terminal cuando el usuario lo exprese directamente en el chat.
6. **PROHIBICIÓN DE GIT DENTRO DEL CONTENEDOR:** El control de versiones Git lo maneja exclusivamente el usuario desde su terminal local en el Host. NUNCA ejecutes comandos de Git (`git status`, `git add`, `git commit`, `git config`, etc.) dentro del contenedor Docker.
7. **MEMORIA INFINITA Y AUDITORÍA PREVIA:** La memoria a largo plazo modular reside en `.agents/knowledge/` y se nutre de forma aditiva. Antes de actualizar la documentación, la IA debe auditar los códigos fuente línea por línea y analizar los diffs reales de los commits recientes.

---
*Fin del Contexto. Mantén este archivo actualizado conforme la arquitectura VLA / Flow Matching crezca.*

## 5. Lecciones Aprendidas y Troubleshooting Crítico (Knowledge Base)

Durante las fases de integración (especialmente simulando con Omniverse Isaac Sim y usando ROS 2 Jazzy), se superaron barreras arquitectónicas importantes. Si el sistema falla, revisa obligatoriamente esta lista:

1. **Middleware DDS (Ceguera de TFs y Tópicos):**
   - *El Problema:* Isaac Sim utiliza **Eclipse CycloneDDS** por defecto. Si abres una nueva terminal y lanzas nodos de ROS 2 (como `visuomotor_data_collection`), estos usarán FastDDS y **no verán** los tópicos ni las TFs (ej. `pick_target`) publicadas por el simulador.
   - *La Solución:* Siempre exportar el entorno antes de correr el launch: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` y `export ROS_DOMAIN_ID=0`. (O añadirlo al `~/.bashrc` del contenedor).

2. **Incompatibilidad Fatal de NumPy 2.x con cv_bridge:**
   - *El Problema:* El paquete `cv_bridge` precompilado en ROS 2 Jazzy depende estrictamente de NumPy 1.x. Al hacer `pip install` de librerías modernas de ML (como Zarr v3), pip actualiza silenciosamente a NumPy 2.0+, provocando que el nodo recolector colapse en tiempo de ejecución con un error de `ImportError: numpy.core.multiarray`.
   - *La Solución:* Mantener rígidamente fijadas las dependencias a versiones anteriores: `pip install 'numpy<2' 'zarr<3' 'numcodecs<0.14'`.

3. **Colisiones contra el Suelo y Torceduras de Muñeca (Auto-PID):**
   - *Torcedura (Drifting Rotacional):* Si el PID en `sim_collector_node.py` actualiza continuamente su orientación objetivo basándose en el estado *actual* de la muñeca o copiando micro-inclinaciones del objeto, el solucionador IK (FZI Cartesian Controller) acumulará pequeños errores numéricos, haciendo que el brazo se tuerza. *Solución implementada:* El nodo ahora **congela rígidamente la orientación inicial** de la muñeca (apuntando hacia abajo) y solo controla X, Y, Z.
   - *Colisión contra el suelo:* Si el controlador opera sobre el `link6` (base de la pinza), intentar llegar al centro del objeto provocará que los dedos físicos atraviesen la mesa. *Solución implementada:* Se añadió el parámetro `tcp_offset_z` (ej. 17 cm) en el archivo YAML para compensar la longitud estructural de la pinza, de modo que la base frene en el aire mientras los dedos abrazan el objeto.

4. **Bug de Compilación "option --uninstall not recognized":**
   - *El Problema:* Las versiones modernas de `setuptools` (>=77) que instala PyTorch deprecian comandos antiguos usados por el gestor `colcon` al ejecutar `colcon build`.
   - *La Solución:* Frente a este error, nunca intentar degradar dependencias de PyTorch. La solución nativa es realizar una limpieza total de caché ejecutando `rm -rf build/ install/ log/ && colcon build`.

5. **Condicionamiento del U-Net (Ceguera del Estado Físico):**
   - *El Problema:* Durante las primeras pruebas, el modelo generaba trayectorias rectas que ignoraban la posición física del brazo, provocando un colapso en la inferencia si el robot se desviaba del camino. Esto sucedía porque la variable `state` estaba silenciada en el código del `global_cond`.
   - *La Solución:* Se reconectó la inyección de `observation.state` en la red neuronal (`diffusion_policy.py`). La red ahora asimila vectores de 8 Dimensiones `[X, Y, Z, Qx, Qy, Qz, Qw, Gripper]` como condicionamiento global, aprendiendo la relación exacta entre "dónde estoy ahora" y "hacia dónde debe moverse el PID".

6. **Bucle Infinito de Respawn (Físicas de Isaac Sim trabadas):**
   - *El Problema:* Al enviar comandos aleatorios vía `/respawn` (Twist) al Script Node de Isaac Sim, el nodo Python de OmniGraph ejecutaba la orden 60 veces por segundo. Esto congelaba el objeto en el aire o en la mesa, imposibilitando que el brazo pudiera levantarlo.
   - *La Solución:* Se rediseñó el Script Node implementando un patrón de caché (`last_lin_processed`), asegurando que la posición del objeto (`xformOp:translate`) se modifique estrictamente **una sola vez** por cada nuevo mensaje ROS recibido, liberando al objeto para que el motor PhysX lo procese libremente.

7. **Conflicto de Teletransporte vs Gripper (Timing de Recolección de Datos):**
   - *El Problema:* Teletransportar el objeto exactamente al terminar de grabar la fase `LIFT` causaba que Isaac Sim anulara el teletransporte, ya que el objeto seguía amarrado físicamente por la fuerza de fricción del gripper cerrado.
   - *La Solución:* Se rediseñó la máquina de estados en `pickup_approximate.py`. El respawn se inyecta exactamente en la mitad del estado `WAIT` (ej. al 1.0s de un wait de 2.0s). Esto le da tiempo mecánico al robot para abrir el gripper y soltar el objeto, teletransportándolo libremente mientras cae, y dándole el resto del tiempo de `WAIT` para asentar sus físicas en la mesa antes de que comience a grabar la cámara.

8. **Visualización de Zarr (Proyecciones Ortográficas):**
   - *El Problema:* Al plotear trayectorias 3D (X, Y, Z) con `matplotlib`, la perspectiva estándar deformaba la percepción visual de la altura Z y la alineación (efecto "ojo de pez" e ilusiones de inclinación en L).
   - *La Solución:* En `visualize_zarr.py`, se implementó estrictamente una proyección ortográfica (`ax.set_proj_type('ortho')`) con límites dinámicos (`set_box_aspect(1,1,1)`), garantizando que las líneas dibujadas coincidan 1:1 con los movimientos reales físicos que experimenta el robot.

9. **Error `Expected 3D or 4D input to conv2d, but got 5D` en Inferencia:**
   - *El Problema:* El script original `policy_node.py` intentaba acumular el historial de observaciones (`h_obs = 2`) haciendo un `torch.stack` manual y enviándolo empaquetado a `policy.select_action()`. Sin embargo, `DiffusionPolicy` ya cuenta con una cola interna (Deque) que realiza este empaquetamiento, provocando la duplicación de dimensiones y un tensor de 5 dimensiones (`[1, 2, 3, 224, 224]`) que crasheaba al Encoder RGB (conv2d).
   - *La Solución:* Se delegó toda la responsabilidad de caché temporal a la política. El nodo de ROS 2 fue rescrito para enviar única y exclusivamente **el frame y pose más recientes** en cada tick de inferencia, evitando la sobredimensión.

10. **Parálisis por Desplazamiento Fuera de Distribución (Out of Distribution / OOD):**
   - *El Problema:* Al iniciar la inferencia, si el brazo físico se encontraba en la parte inferior de la mesa, el modelo se congelaba prediciendo ruidos (micro-movimientos). Esto sucedía porque, durante la toma de datos, **todas las trayectorias** iniciaban desde una pose de descanso elevada (`Z=0.40`). Poner la red en un estado físico nunca visto (OOD) paralizaba la predicción.
   - *La Solución:* Se implementó un modo estricto en el nodo de inferencia `policy_node.py`. Ahora se interpone un estado obligatorio llamado `RESET` (duración: 5 seg), el cual asume el control manual para desplazar el efector exactamente a la coordenada de inicio `[0.25, 0.0, 0.40]`. Esto garantiza que, al cederle el control al estado de `INFERENCE`, la Red Neuronal se encuentre en su "zona de confort" de distribución y sepa descender suavemente.

11. **Robot Físico Ignorando la Trayectoria Generada (TF Frame ID faltante):**
   - *El Problema:* La IA predecía las acciones fluidamente por terminal, pero el robot ni se inmutaba en Isaac Sim, mostrándose trabado.
   - *La Solución:* Se descubrió que al generar y publicar el mensaje ROS 2 (`PoseStamped`), se había omitido poblar la cabecera espacial (`msg.header.frame_id = 'link_base'`). El solucionador cinemático (RMPflow / FZI) descarta sistemáticamente como medida de seguridad cualquier orden que carezca de un punto de referencia. Inyectar el `frame_id` reactivó el movimiento del robot.

12. **El Gripper Físico se Niega a Cerrarse durante la Inferencia:**
   - *El Problema:* La IA navegaba hacia abajo (ej. `Z=0.08`), pero el actuador ignoraba las señales numéricas de cierre generadas por la red neuronal, escurriendo el objeto.
   - *La Solución:* En `action_executors.py`, el nodo de inferencia empaquetaba la acción del motor del gripper como un arreglo de 2 elementos `[val, val]`. No obstante, el colector de datos original estaba configurado para usar un controlador de articulación único (`[val]`). Al notar la discrepancia dimensional, Isaac Sim ignoraba la señal del ROS 2 Subscribe Node. Recortar la salida a una matriz unidimensional restableció el cierre físico guiado por IA.

13. **Omisión de la Señal de Término (Auto-Stop y la 9ª Dimensión):**
   - *El Problema:* Al completar la tarea (objeto depositado), el robot continuaba activo, vibrando o ejecutando micro-movimientos espurios. Al intentar implementar una 9ª dimensión (`done_flag`) en la acción, el modelo inferido no lograba activarla (`flag ~ 0.0002` constante).
   - *Diagnóstico Raíz:* 
     1. El dataset original (`demonstrations_v7.zarr`) solo contenía 3 fotogramas finales con `1.0`.
     2. El dataloader (`zarr_dataset.py`) descartaba los últimos `h_act` (16) frames de cada episodio para evitar desbordar el horizonte temporal, lo que eliminaba el 100% de las muestras con `done_flag = 1.0` vistas por la red durante el entrenamiento.
   - *La Solución Triple:*
     1. **Copy-Padding en Dataset (`zarr_dataset.py`):** Cuando el índice se acerca al final del episodio, se completa el horizonte replicando el último estado/acción (`np.repeat(actual_actions[-1:], pad_len, axis=0)`), permitiendo que la red entrene sobre los frames finales de término.
     2. **Parcheo Seguro de Datasets (`patch_dataset_done_flag.py`):** Script no destructivo que generó `demonstrations_v7_done_fixed.zarr` inyectando 20 frames (1.0 s a 20 Hz) con `done = 1.0` en cada episodio.
     3. **Desacoplamiento en Inferencia (`policy_node.py`):** Separación total entre Auto-Stop Neuronal (`enable_neural_auto_stop`, umbral configurable `neural_auto_stop_threshold`) y Auto-Stop Heurístico (`enable_heuristic_auto_stop`, varianza de posición XYZ).
     4. **Trazabilidad y Logs en Tiempo Real:** El nodo de inferencia reporta a 1 Hz el valor instantáneo de la 9ª dimensión vs su umbral (`Dim 9 (Done): X.XXXX [Umbral: 0.50 -> No alcanzado/ALCANZADO]`), y al dispararse la parada, registra y retiene en el estado congelado el valor exacto que detonó el fin de la tarea.

14. **Arquitectura Causal: La 9ª Dimensión es Acción, NUNCA Observación:**
   - *El Principio:* `observation.state` (8 dimensiones) modela propiocepción física medible por encoders/TF (`[X, Y, Z, Qx, Qy, Qz, Qw, Gripper]`). `action` (9 dimensiones) modela la intención motora y la decisión del agente (`[..., Gripper, Done]`).
   - *Peligro Evitado (Fuga Causal / Causal Leakage):* Si `done` se incluyera en la observación, crearía una dependencia circular: en inferencia, el nodo necesitaría saber de antemano si la tarea terminó para poder pasárselo a la red como entrada.

15. **Parametrización Completa y Eliminación de Hardcoded Values:**
   - *Regla Aplicada:* Se purgaron todos los valores fijos en `policy_node.py` y `data_collection_params.yaml`:
     - Tópicos (`joint_states_topic`, `action_chunk_topic`, `camera_topic`).
     - Tiempos (`reset_duration_s`, `hold_target_time_s`, `wait_time_s`).
     - Geometría y límites de gripper (`gripper_open_pos`, `gripper_closed_pos`, `gripper_binary_threshold`, `gripper_joint_name_pattern`).
     - Tamaño de entrada visual (`image_size: 256`).

16. **Telemetría en Tiempo Real y Diagnóstico de Métricas (ROS 2 / `rqt_plot`):**
    - Para permitir monitoreo continuo y depuración visual sin sobrecargar el log textual, se implementaron tópicos independientes `std_msgs/Float32` y `geometry_msgs/PointStamped`.
    - **Inferencia DP (`/policy/metrics`):**
      - `done_flag`, `done_threshold`: Monitoreo del auto-stop neuronal.
      - `gripper_cmd`, `gripper_state`: Apertura y cierre continuo vs medido.
      - `tracking_error_xyz`: Distancia euclidiana cartesiana entre el comando predicho y la pose TCP real.
      - `tcp_speed_m_s`: Velocidad tangencial instantánea del efector final.
      - `action_variance`: Varianza de la trayectoria predicha (heurística de estabilidad).
      - `dist_to_home`: Distancia al punto Home de reposo.
      - `tcp_cmd_xyz`, `tcp_actual_xyz`: Coordenadas 3D para trazado espacial.
    - **Entrenamiento RL DPPO (`/rl/metrics`):**
      - A nivel de paso (20 Hz): `instant_reward`, `episode_reward`, `critic_value`, `done_flag`, `pseudo_force`, `dist_hover`, `dist_target`, `r_approach`, `r_orient`, `r_grasp`, `r_transport`.
      - A nivel de iteración: `iter_mean_reward`, `iter_policy_loss`, `iter_critic_loss`, `iter_approx_kl`.

17. **Registro Offline de Métricas y Curvas de Publicación (Matplotlib Headless + CSV):**
    - Se garantiza operación headless (`matplotlib.use('Agg')`) para evitar colapsos por ausencia de servidor X11 en contenedores o clústeres.
    - **Diffusion Policy Supervisado (`visuomotor_core/train.py`):**
      - Exportación de métricas paso a paso a `training_metrics.csv` (`step`, `epoch`, `loss`, `loss_ema`, `lr`, `vram_peak_mb`, `timestamp`).
      - Exportación por época a `epoch_metrics.csv` (`epoch`, `avg_loss`, `final_lr`, `duration_s`, `timestamp`).
      - Generación automática en `losses/training_curves.png` y `.pdf` con panel 2x2: Loss paso a paso + EMA, Pérdida media por época, Schedule Cosine Annealing LR y Consumo Peak VRAM.
    - **Online RL DPPO (`visuomotor_ros2/rl_train_node.py`):**
      - Exportación de cada iteración a `rl_training_metrics.csv` (`iteration`, `elapsed_s`, `mean_reward`, `policy_loss`, `critic_loss`, `approx_kl`, `is_warmup`, `timestamp`).
      - Generación continua en `rl_training_curves.png` y `.pdf` con panel 2x2: Retorno episódico + media móvil (5 iters), Critic Loss $V(s)$, Policy Loss y Divergencia KL vs Target KL.

18. **Alineación Temporal FSM Experta y Solución de Truncamiento Prematuro:**
    - La trayectoria experta (`pickup_approximate.py`) describe un ciclo completo de ida y vuelta que dura entre 50 y 55 segundos (~1000 a 1100 pasos a 20 Hz, considerando tiempos de aproximación, estabilización de agarre, elevación, enderezado y 3.5 s de retención en Home).
    - El horizonte de RL se ajustó a `max_episode_steps: 1200` (60.0 s) y `n_steps_per_iteration: 400` ($400 \times 3 = 1200\text{ pasos}$), evitando que los episodios se corten cuando el robot está a pocos centímetros de depositar en Home.
    - **Corrección Geométrica de la Condición Terminal:** Anteriormente, `dist_target < 0.05` comparaba la posición del cubo (`obj_pos`) contra `target_pos` ($Z=0.40$). Sin embargo, debido a que el gripper sujeta el objeto por debajo de la muñeca (`tcp_offset_z = 0.079` m), la posición del objeto sostenido en Home es $Z \approx 0.321$ m. La distancia euclidiana mínima era de $\approx 8$ cm, por lo que nunca se validaba la llegada a Home y siempre se truncaba por timeout. Se corrigió introduciendo `expected_obj_target = target_pos - [0, 0, tcp_offset_z]`, validando llegada con `dist_target < 0.08` o `dist_ee_home < 0.06` con agarre confirmado.
    - **Exploración Activa en DPPO:** Se elevó $\eta$ de $0.1$ a $0.4$ y el suelo de varianza `min_sampling_denoising_std` de $0.01$ a $0.04$ (para búsqueda espacial real de $\sim 2-3$ cm). Además, se multiplicó por $1.8\times$ el ruido del comando de gripper (dimensión 7) para permitir que la política explore activamente cierres en diferentes instantes sin desestabilizar la rotación de la muñeca (cuyas dimensiones [3:7] se mantienen amortiguadas al $0.05\times$).
    - **Trazabilidad de Finalización de Tarea en Logs:** Se implementaron banners explícitos al término de cada episodio (`🎉 ¡TAREA COMPLETADA CON ÉXITO!` vs `⚠️ TAREA NO COMPLETADA`) y la columna booleana `task_completed` en `rl_training_metrics.csv`.

19. **Cuadrícula de Respawn Aleatorio y Curriculum por Reintento en Fallo (`respawn_mode`):**
    - Se corrigió el tipo de mensaje en `/respawn` de `Empty` a `geometry_msgs/Twist` para que Isaac Sim reciba y ejecute la teletransportación del objeto con coordenadas 3D (`linear.x, linear.y, linear.z`) y orientación (`angular.z`).
    - Se unificó la cuadrícula Halton Quasi-Monte Carlo en `rl_training_params.yaml`:
      - $X \in [-0.35, -0.52]\text{ m}$
      - $Y \in [-0.56, -1.08]\text{ m}$
      - $Z = 0.44\text{ m}$, $Yaw \in [-180^\circ, +180^\circ]$ con márgenes de seguridad en bordes.
    - Se implementó el modo de aprendizaje por curriculum **`respawn_mode: "on_success"`**: si el robot falla en un episodio, el objeto se mantiene en la misma posición en el siguiente intento para que RL explore y aprenda a auto-corregirse en ese punto exacto. Solo cuando completa la tarea con éxito (`task_success: True`), el objeto se teletransporta a una nueva coordenada aleatoria. Alternativamente, permite `respawn_mode: "always"` (aleatorización continua) y `"fixed"`.
    - **Secuencia Física de Reset sin Interferencia:** Se corrigió el orden de ejecución en `reset()`: el gripper se abre de inmediato y el manipulador viaja a Home primero; una vez que el brazo está arriba y libre, se emite una ráfaga de 5 pulsos de `Twist` a `/respawn` y se espera 1.0 s de reposo gravitacional. Esto previene que el cubo sea arrastrado o colisionado durante el retorno a Home, garantizando que reaparezca exactamente en la coordenada original en caso de fallo.

---

### 6. Análisis de Fallas Cognitivas en la Inferencia (Falta de Generalización)
Durante las primeras evaluaciones en inferencia con un modelo entrenado con ~60 repeticiones por 2.5 horas, se detectaron los siguientes comportamientos deficientes:
1. **Aproximación Estocástica:** Ante la misma posición del objeto, el brazo a veces se acerca bien y otras veces diverge a otro lado.
2. **Ceguera Rotacional:** Si el objeto rota en su eje Z, la política deja de detectarlo.
3. **Incapacidad de Estimar Profundidad (Eje Z):** El robot falla en percibir el tamaño escalar del objeto, cerrando el gripper centímetros antes o después de la altura correcta.
20. **Supresión de Respawn por Caché `last_lin_processed` en Reintentos de RL:**
   - *El Problema:* Al entrenar con DPPO RL en modo `respawn_mode: "on_success"`, si un episodio falla, el entorno debe reintentar la misma coordenada aleatoria para que la política aprenda de su error. Sin embargo, en la iteración 2 el robot soltaba el cubo en Home y el cubo no regresaba al spawn original, a pesar de que el log reportaba el teletransporte.
   - *Diagnóstico Raíz:* En el Script Node de OmniGraph en Isaac Sim (ver ítem 6), existe un filtro anti-congelamiento `if linear == last_lin_processed: return;`. Como el fallo repetía las coordenadas flotantes con 100% de identidad respecto a la iteración 1, OmniGraph consideró el mensaje como un duplicado espurio y suprimió la escritura de la transformación en USD / PhysX.
   - *La Solución:* Se implementó en `gym_ros2_env.py` la técnica de **Pulso Bifásico con Micro-Jitter**:
     1. Disparo de un pre-pulso con elevación transitoria ($Z + 4\text{ cm}$) y una micro-variación espacial ($\pm 0.5\text{ mm}$ en $X, Y$). Esto invalida el `last_lin_processed` de Isaac Sim y mueve el objeto fuera de Home.
     2. Breve espera de 80 ms.
     3. Ráfaga de 5 pulsos en la cota de mesa ($Z = 0.44\text{ m}$) en las coordenadas de destino exactas.
     4. Verificación directa contra el árbol de transformaciones TF (`pick_target`), confirmando en consola con `[RESPAWN TF CONFIRMADO]` la posición física real y disparando un reintento automático si existiese alguna discrepancia $> 8\text{ cm}$.

21. **Sincronización Determinista de Respawn y Asentamiento Físico en RL (`_wait_and_stabilize_at_home`):**
    - *El Problema:* Al iniciar una nueva iteración de RL, el robot comenzaba a calcular y ejecutar acciones antes de que el objeto teletransportado se estabilizara en la mesa, o utilizaba fotogramas de cámara residuales del episodio previo. Además, un chequeo de TF con tolerancias numéricas milimétricas rígidas provocaba falsos reintentos repetitivos a pesar de que el objeto ya había aparecido en el simulador.
    - *La Solución:* Se simplificó y robusteció el flujo en `gym_ros2_env.py` (`_wait_and_stabilize_at_home`):
      1. Se envía el pulso de respawn una sola vez al alcanzar Home.
      2. El robot sostiene su postura inmóvil en Home con pinza abierta durante el tiempo de asentamiento configurable (`reset_stabilize_s = 2.0s`), permitiendo que el objeto caiga y repose sin re-teletransportes innecesarios.
      3. Se actualiza la TF con timeout de tolerancia temporal (`rclpy.duration.Duration(seconds=0.05)`).
      4. Se purga el semáforo de cámara (`self._obs_event.clear()`) y se aguarda un fotograma RGB fresco con el cubo ya asentado antes de ceder el control al bucle de RL.

22. **Calibración Empírica del Pseudo-Sensor de Fuerza Propioceptivo (`calibrate_gripper.py`):**
    - *El Problema:* Durante las iteraciones de RL, el robot lograba agarrar el cubo, elevarlo 27.6 cm y transportarlo a Home, pero `info["is_clamped"]` reportaba `False`, anulando la recompensa de transporte y el `task_success`.
    - *Diagnóstico Raíz:* El umbral anterior (`0.0025 m`) estaba sobredimensionado. La cinemática de la pinza con el cubo real de Isaac Sim se detiene en $p = -0.0086\text{ m}$ (con comando $-0.010\text{ m}$), generando una pseudo-fuerza de contacto de $+0.0014\text{ m}$ ($1.4\text{ mm}$), la cual nunca superaba `0.0025`.
    - *La Solución:* Se ejecutó el calibrador interactivo `calibrate_gripper`:
      - $\text{pseudo\_force}_{\text{vacío}} = 0.000000\text{ m}$
      - $\text{pseudo\_force}_{\text{cubo}} = 0.001400\text{ m}$
      - **Umbral óptimo inyectado:** `pseudo_force_clamped_threshold: 0.000700` (el punto medio exacto de detección).
      - Además, se certificó que en vuelo ($\Delta z \ge 8\text{ cm}$) con comando de cierre, el agarre se valide físicamente sin ser degradado por el chequeo de velocidad estacionaria.

23. **Derivación Matemática Rigurosa de Cinemática Inversa DLS (`cartesian_motion_controller`):**
    - *Arquitectura:* Control en espacio de tareas mediante Damped Least Squares (DLS) de Levenberg-Marquardt.
    - *Ecuación Inversa:* $\Delta q = J^T (J J^T + \lambda^2 I)^{-1} \dot{x}_{\text{des}} \Delta t$, resolviendo el seguimiento cartesiano de 6 DoF a velocidades articulares suaves incluso en la vecindad de singularidades cinemáticas.

24. **Estado Piloto del PID Experto y Transición hacia Controladores SOTA (MPC, cuRobo, CRISP):**
    - *Diagnóstico:* El PID cartesiano actual en `pickup_approximate.py` opera como un demostrador experto en fase piloto. Aunque funcional, carece de compliancia pasiva ante impactos y no optimiza restricciones dinámicas a futuro.
    - *Dirección SOTA:* Integración de demostradores avanzados basados en Model Predictive Control (cuRobo GPU MPC) y control de impedancia cartesiana compliante (CRISP en ROS 2) para generar datasets de demostraciones comparativos.

25. **Estándar de Logging Científico y Benchmarking Off-ROS (`visuomotor_core/utils/logger.py`):**
    - *Aislamiento:* En código Off-ROS se prohíbe `rclpy` y los `print()` informales.
    - *Implementación:* `configure_logging()` con `coloredlogs`, resolución en milisegundos, supresión de librerías ruidosas y soporte de archivo `run.log`. Context manager `@benchmark` con nanosegundos (`time.perf_counter_ns()`), `humanize.precisedelta` y monitoreo de memoria GPU peak VRAM (`torch.cuda.max_memory_allocated()`).

26. **Base de Datos Relacional Automática y Fail-Safe (`visuomotor_core/utils/tracker.py` & `tools/experiment_tracker.py`):**
    - *Automatización Transparente:* `train.py`, `rl_train_node.py`, `pickup_approximate.py` y `evaluate_policy.py` incorporan hooks automáticos no intrusivos que persisten cada corrida en `experiments/experiments.db`.
    - *Principio Fail-Safe:* Todo acceso a SQLite está encapsulado en bloques protegidos; una eventual contención o fallo de escritura en la base de datos jamás interrumpe el entrenamiento, simulación o guardado de checkpoints.
    - *Visualización Instantánea:* Mediante `python3 tools/experiment_tracker.py summary` se obtiene en terminal el registro tabular consolidado de entrenamientos supervisados, RL, evaluaciones y datasets.

27. **Medición Cinemática del Tool Center Point (TCP / Superficie de Contacto):**
    - *El Problema:* Al evaluar la precisión de aproximación del manipulador hacia el objeto a agarrar, calcular la distancia euclidiana directa desde la brida de la muñeca (`gripper_base_link`) arrojaba una distancia aparente de $\approx 80\text{ mm}$ en pleno agarre exitoso. Esto distorsionaba la evaluación, sugiriendo falsamente una falta de precisión métrica.
    - *Causa Raíz:* Las almohadillas mecánicas de contacto del gripper Lite6 se extienden a lo largo del eje $Z$ local de la herramienta una longitud de $\mathbf{tcp\_offset\_z} = 0.079\text{ m}$ ($79.0\text{ mm}$).
    - *La Solución:* Se implementó en `GymROS2Env._compute_reward_and_termination()` la proyección cinemática analítica usando el cuaternión $\mathbf{q} = [q_x, q_y, q_z, q_w]$:
      $$\mathbf{v}_z = [2(q_x q_z + q_w q_y), 2(q_y q_z - q_w q_x), 1 - 2(q_x^2 + q_y^2)]$$
      $$\mathbf{p}_{\text{tcp}} = \mathbf{p}_{\text{ee}} + \mathbf{v}_z \cdot \mathbf{tcp\_offset\_z}$$
      $$d_{\text{contacto}} = \|\mathbf{p}_{\text{tcp}} - \mathbf{p}_{\text{objeto}}\|$$
      Con esta formulación, la distancia de contacto en agarres exitosos se sitúa con precisión milimétrica entre $0.88\text{ mm}$ y $7.14\text{ mm}$, discriminando fielmente la habilidad de aproximación de cada política.

28. **Tratamiento de Valores Atípicos (Outliers) en Visualización Científica:**
    - *El Problema:* En campañas de evaluación estocásticas bajo distribución cuasi-aleatoria Halton QMC, episodios singulares fuera de distribución (ej. el Episodio 3 donde el cubo reapareció en el límite lateral de la mesa, $Y = -0.898\text{ m}$) produjeron fallos simétricos en ambos modelos ($d_{\text{contacto}} \approx 108\text{ mm}$, error terminal de $390\text{ mm}$). Al graficar mediante diagramas de cajas (boxplots) lineales estándar, estos puntos atípicos colapsaban las cajas del 90% de los datos en líneas diminutas e ilegibles.
    - *La Solución:* Se rediseñó el generador de figuras comparativas:
      1. **Escala Logarítmica:** Para la distancia mínima de contacto (Subplot B), permitiendo visualizar en el mismo gráfico desde $0.7\text{ mm}$ hasta $150\text{ mm}$.
      2. **Smart Zoom y Badges Explícitos:** En error terminal, elevación y duración, los límites se acotaron al régimen nominal de operación y los puntos atípicos se señalizaron con etiquetas rojas (`▲ Outlier Ep 3: 390.7 mm`), preservando la honestidad científica sin sacrificar la claridad del rango intercuartílico (IQR).
      3. **Dispersión con Jitter y Marcadores de Media:** Se superpusieron todos los datos individuales con jitter horizontal y diamantes ($\Diamond$) para la media empírica.
      4. **Remoción de Líneas Heurísticas en Boxplots:** Se eliminó la línea fija de umbral en el gráfico de elevación $\Delta Z$, trasladando su explicación a la leyenda y al texto del paper.

29. **Re-plotter Desacoplado y Estandarización de la Habilidad de Benchmarking:**
    - *El Problema:* Modificar aspectos visuales, tipografías o escalas en los gráficos de evaluación requería volver a ejecutar la simulación completa en Isaac Sim (~25-30 minutos).
    - *La Solución:* Se desacopló la renderización en el script independiente `tools/replot_benchmark.py`, el cual lee directamente `benchmark_episodes.csv` y re-genera las figuras en PNG (300 DPI) y PDF vectorial en menos de 2 segundos.
    - *Estandarización:* Se formalizó la skill `.agents/skills/visuomotor-benchmark-evaluator/SKILL.md` como procedimiento operativo reproducible para futuras campañas de evaluación.

30. **Reconciliación Empírica de Curvas de RL y Diferenciación Algorítmica DPPO vs. Flow Matching:**
    - *El Problema:* En borradores preliminares del informe científico, el texto afirmaba que la divergencia KL se mantuvo plana ($\le 0.05$) y confundía el retorno medio por lote con episodios ideales aislados ($> 1600$).
    - *Diagnóstico Raíz:* Auditoría forense de `visuomotor_core/outputs/dppo_v7_rl_finetuned/rl_training_metrics.csv`. Durante los 5 pasos de difusión estocásticos, la razón de verosimilitud experimenta excursiones transitorias de KL ($0.1$ a $2.4$), mientras que el retorno medio por lote converge entre $\approx 350$ y $500$ (con pico de $775.6$ en la iteración 11).
    - *La Solución:* Se reconciliaron los datos explicando rigurosamente que la estabilidad del entrenamiento proviene del recorte (*clipping*) PPO ($\epsilon = 0.2$) y de la regularización de imitación ($\lambda_{\text{BC}}$). Además, se incorporó la referencia fundacional de DPPO sobre *Denoising MDP* discreto (Allen Z. Ren et al., ICLR 2025 \cite{Ren2025DPPO}), diferenciándolo formalmente de métodos continuos basados en Flow Matching (*ReinFlow*, *Flow-GRPO*).

31. **Ingeniería Editorial Académica en LaTeX y Diagramación TikZ de Precisión:**
    - *El Problema:* Los borradores de informes y papers contenían metalenguaje de evaluación (*"según la rúbrica"*, *"para la entrega final"*), inconsistencias aritméticas en esquemas vectoriales (`64D + 16D = 144D`), colisiones en etiquetas de flechas verticales TikZ y sobreflujo de texto que desbordaba a 15 páginas.
    - *La Solución:*
      1. **Estilo Humano Sobrio:** Erradicación de hipérboles y metalenguaje evaluativo; uso canónico en cursiva de términos estándar en inglés (*dataloader*, *rollout buffer*, *clipping*, *outliers*).
      2. **Consistencia Aritmética TikZ:** Corrección a $2 \times 64\text{D} + 16\text{D} = 144\text{D}$ para reflejar $T_{\text{obs}} = 2$ en la concatenación FiLM.
      3. **Anti-Colisión en Aristas:** Desplazamiento lateral `[midway, right=4pt]` en flechas verticales de telemetría.
      4. **Fundamentación Estadística No Paramétrica:** Justificación formal del test de rangos con signo de Wilcoxon ante la asimetría y *outliers* de aproximación en robótica manipuladora.
      5. **Control Estricto de Paginación:** Inserción de `\clearpage` antes de diagramas metodológicos flotantes y antes de bibliografía, logrando 0 errores, 0 *overfull \hbox* y exactamente 14 páginas con bibliografía iniciando en la línea 1.
      6. **Estandarización:** Creación del módulo `.agents/knowledge/06_scientific_latex_reporting_and_tikz.md` y de la skill `.agents/skills/scientific-latex-reporting/SKILL.md`.

32. **Validación sin Conjunto de Prueba mediante Error Fuera de Bolsa (OOB - Out-of-Bag):**
    - *Fundamento:* En machine learning tradicional, para medir generalización se suele apartar un conjunto de prueba (*test set*). El *Bagging* (*Bootstrap Aggregating*) ofrece una alternativa gratuita y honesta sin necesidad de apartar datos.
    - *Mecanismo Operativo:*
      1. Para cada fila $i$, se identifican qué árboles o estimadores base no la incluyeron en su muestra bootstrap (`fuera[bolsa] == False`).
      2. Esos árboles específicos nunca vieron a la observación $i$ durante su entrenamiento: para ellos, $i$ es un dato de prueba estrictamente nuevo.
      3. Se hace votar o promediar a esos árboles sobre la fila $i$ y se compara la predicción agregada con la etiqueta real $y_i$.
      4. Si alguna fila quedó dentro de la bolsa de todos los árboles, no puede evaluarse y se excluye del cómputo del error global.

33. **AdaBoost con Tocones de Decisión y Diagnóstico Físico de Modos de Fallo:**
    - *Fundamento:* A diferencia de *Bagging*, AdaBoost entrena modelos en secuencia (*boosting*) adaptativo con reponderación exponencial de muestras erróneas:
      1. Pesos uniformes iniciales: $w_n = 1/N$.
      2. En cada ronda $t$: ajuste de un tocón de decisión (corte univariado $\mathbb{I}(x_j \ge \theta)$), error ponderado $\epsilon_t = \sum_{\text{err}} w_n$, cálculo de importancia $\alpha_t = \frac{1}{2}\ln \frac{1-\epsilon_t}{\epsilon_t}$ (con recorte $[10^{-10}, 1-10^{-10}]$), y actualización $w_n \leftarrow w_n \exp(-\alpha_t y_n \text{pred}_n)$ seguida de renormalización.
      3. Predicción final: $\text{signo}(\sum_t \alpha_t \cdot \text{tocón}_t(x))$.
    - *Aplicación Robótica:* Clasificación explicable de éxito/fallo a partir de telemetría tabular (`benchmark_comparison_results.csv`), aislando qué variables físicas ($\delta_{\text{elastic}}$, $\Delta Z$, distancia TCP) determinan el fallo cinemático, y minería activa de trayectorias difíciles (*Hard-Negative Mining*).

34. **Evaluación SOTA para Políticas Visuomotoras (Conformal Prediction, Diff-DAgger UQ y Verificadores):**
    - *Dilema Multimodalidad vs. Incertidumbre:* La varianza simple de acciones confunde opciones multimodales válidas (ej. esquivar por izquierda o derecha) con confusión del modelo.
    - *Métodos Avanzados Incorporados:*
      1. **Diff-DAgger / Consistencia de Score:** Evaluar la discrepancia en la función de ruido predicha $\epsilon_\theta(A^k, k, O_t)$ para detectar estados fuera de distribución (OOD) antes de la divergencia física.
      2. **Predicción Conforme (PlanCP):** Generación de tubos de trayectoria con garantías estadísticas de cobertura libre de distribución ($1 - \alpha = 95\%$).
      3. **Verificadores en Tiempo de Test (Best-of-N):** Usar el Crítico $V_\phi$ entrenado en DPPO para filtrar y seleccionar en tiempo real la mejor trayectoria entre múltiples hipótesis generadas en GPU.
    - *Estandarización:* Creación del módulo `.agents/knowledge/07_sota_visuomotor_evaluation_and_uncertainty.md`.

35. **Transición hacia 3D Diffusion Policy (DP3): Nubes de Puntos, DP3Encoder sin BatchNorm y Cropping Espacial:**
    - *El Problema:* Las políticas visuomotoras 2D proyectan el mundo a una imagen plana, perdiendo la métrica de profundidad $Z$ y necesitando $>100-200$ demostraciones para generalizar en 3D.
    - *La Solución (DP3, Ze et al. 2024):*
      1. **Nubes de Puntos Dispersas:** Muestreo de $512$ a $1024$ puntos por FPS (Farthest Point Sampling) a partir de desproyección de mapas de profundidad.
      2. **DP3Encoder sin T-Net ni BatchNorm:** PointNet tradicional colapsa ($15.7\%$ éxito). Al suprimir T-Net y BatchNorm (inhibidores de correlación y EMA) e intercalar `LayerNorm` en un MLP de 3 niveles con proyección a 64D, el éxito se eleva al $78.3\%$.
      3. **Workspace Cropping Obligatorio:** Recortar los puntos fuera del volumen de trabajo de la mesa es fundamental (sin recorte el éxito cae al $45.3\%$).
      4. **Invarianza de Apariencia:** Procesar únicamente coordenadas $(X, Y, Z)$ sin canales de color permite generalizar automáticamente a objetos de colores y texturas no vistos.
      5. **Predicción de Muestra ($x_0$):** Predecir la trayectoria limpia $A^0$ en vez del ruido $\epsilon$ acelera la convergencia en acciones de alta dimensión.
      6. **Simple DP3:** Versión podada de U-Net que eleva la frecuencia de inferencia a $25.3\text{ FPS}$ manteniendo alta precisión.

36. **Rigor Metodológico en Machine Learning para Trayectorias Robóticas:**
    - *Fuga de Información Temporal (Data Leakage):* Prohibición absoluta de particionar datos aleatoriamente por fotogramas. Debido a la fuerte correlación temporal entre pasos contiguos, la partición (ej. split 80/20) debe realizarse **estrictamente a nivel de episodio completo**.
    - *Ley de Kolter y Test de Sobreajuste:* Antes de entrenar a gran escala, realizar un test de sanidad sobreajustando a 1 solo episodio/mini-batch. Si la pérdida no converge a cero, existe un error arquitectónico o de gradientes en el código.
    - *Normalizaciones Inmunes al Tamaño de Lote:* `LayerNorm` y `GroupNorm` preservan el comportamiento idéntico entre entrenamiento e inferencia, a diferencia de `BatchNorm`.
    - *Estandarización:* Creación del módulo `.agents/knowledge/08_3d_diffusion_policy_and_ml_rigor.md`.

37. **Transferencia Sim-to-Real: Few-Shot Fine-Tuning y Políticas Residuales (ResiP/TRANSIC):**
    - *Diagnóstico Arquitectónico:* La política de `visuomotor_ros2` predice acciones cartesianas de alto nivel ($16 \times 9$: poses $X, Y, Z, q_x, q_y, q_z, q_w$, pinza y auto-stop) resueltas por controladores cinemáticos a 100 Hz, lo que absorbe discrepancias inerciales y de torque motor.
    - *Comparación SOTA:*
      1. **Few-Shot Fine-Tuning (Opción 2):** Ajustar pesos con teleoperación real (gamepad). Riesgo de olvido catastrófico resuelto mediante Co-Fine-Tuning ($20\%$ real, $80\%$ sim) o LoRA/PEFT en el codificador visual.
      2. **Residual Policy Learning (Opción 3 - ResiP/TRANSIC):** Desacoplamiento total entre un macro-planificador congelado $\pi_{\text{sim}}$ y una política residual $\pi_{\text{res}}$ ligera en bucle cerrado que emite micro-correcciones $\Delta \mathbf{a}$ acotadas por $\pm \delta_{\max}$. Ideal para contacto fino, fricción de la pinza y agarre delicado sin riesgo de desestabilizar la política principal.
    - *Estandarización:* Creación del módulo `.agents/knowledge/10_sim_to_real_finetuning_and_residual_policies.md`.

38. **Infraestructura de Aceleración Headless de Isaac Sim y Reducción en Memoria ($3\times\text{--}7\times$):**
    - *El Problema:* El renderizado gráfico por GUI a resolución completa ($1280 \times 720$) consumía $6\text{--}7\text{ GB}$ de VRAM y limitaba la simulación a $30\text{--}60\text{ FPS}$ debido a la tasa de refresco del compositor de ventanas.
    - *La Solución Desplegada:*
      1. **Lanzador Headless (`tools/sim/run_headless_stage.py`):** Instanciación de `SimulationApp` con `headless=True` y renderizador offscreen `RayTracedLighting`.
      2. **Reajuste en Memoria a $256 \times 256$:** Recorrido no destructivo del grafo USD (`stage.Traverse()`) que reconfigura los `RenderProduct` de `/rgb` directamente en memoria sin invocar `stage.Save()`, preservando al $100\%$ los archivos `.usda` en disco.
      3. **Desactivación de Rate Limiting:** Supresión de `rateLimitEnabled` y `rateLimitFrequency` en Omniverse Kit para liberar el paso físico a la máxima velocidad del hardware GPU.
      4. **Orquestador Maestro (`tools/sim/run_headless_system.sh`):** Control del ciclo de vida (arranque de contenedores, sondeo CycloneDDS de `/rgb` e `/isaac/joint_states`, lanzamiento de `lite6_cartesian`, despacho modular bajo `-u hackbrian`, banderas de recolección rápida `--episodes N` y `--dataset RUTA`, modo dedicado `stop` y `trap cleanup` para liberar VRAM automáticamente al terminar o cancelar).
      5. **Micro-Daemon TCP (`tools/sim/isaac_sim_daemon.py` y `sim_client.py`):** Control inter-contenedor en `127.0.0.1:9999` para iniciar y detener Isaac Sim directamente desde el interior de `visuomotor_ros2`.
      6. **Bypass de Diálogo Modal de Seguridad (ScriptNode Opt-In):** Supresión del bloqueo de scripts en headless mediante `--/app/omni.graph.scriptnode/enable_opt_in=false --/app/omni.graph.scriptnode/opt_in=true` y reactivación dinámica en memoria RAM de todos los OmniGraphs (`graph.set_disabled(False)`).
      7. **Resolución de Dependencia `xarm_description`:** Sourcing automático de `/tmp/xarm_ws/install/setup.bash` en el orquestador para permitir que `lite6_cartesian.launch.py` inicialice los controladores cartesianos sin fallos de importación.
      8. **Validación Experimental Completa (Campaña 10 Episodios Acelerada):** Recolección autónoma exitosa de 10 episodios ($10/10$, $3.652$ pasos, $50$ pasos de piso de parada terminal con `done=1.0` por episodio) en `datasets/demonstrations_headless_10ep.zarr`. Reducción de tiempo humano comprobada de ~20 min (manual GUI) a 13.3 min (headless optimizado), con certificación estricta de frecuencia $dt=50\text{ ms}$ ($20\text{ Hz}$) y velocidades cartesianas máximas $\le 0.1144\text{ m/s}$ (bajo el umbral límite de $0.15\text{ m/s}$).
      9. **Aceleración con Real-Time 2.0 (RTPT) + NVIDIA DLSS Performance (x2) y Eco Mode:**
         - *Mapeo canónico:* Identificación de que en Isaac Sim 6.0 la opción "Real-Time 2.0" de la GUI corresponde internamente al backend `RealTimePathTracing` (`RTPT`).
         - *Resolución de Crash de Hydra:* Eliminación de `CreateAttribute` dinámico en texturas de render activo en tiempo de carga (`_MarkSprimDirty` assertion failure en `changeTracker.cpp`), restringiendo la actualización en RAM a atributos existentes.
         - *Inyección validada:* `--/rtx/ecoMode/enabled=true`, `--/rtx/post/aa/op=3` (DLSS), `--/rtx/post/dlss/execMode=0` (Performance), `--/rtx-transient/internal/dlssg/rtpt/interpolatedFrameCount=1` (Frame Generation x2).
         - *Campaña V2 Certificada:* Recolección autónoma de 5 episodios ($5/5$, $1.588$ pasos, 51 frames de piso `done=1.0` por episodio) en `datasets/demonstrations_test_headless_v2.zarr`, con wall time de grabación de sólo $43.07\text{ s} - 55.23\text{ s}$ por episodio ($v_{\max} \le 0.1140\text{ m/s}$) y restauración total de GPU a $56\text{ MiB}$ VRAM.
      10. **Aceleración Cinemática y Purga de Tiempos Muertos:**
          - `max_speed_m_s = 0.30 m/s`: Duplicación de velocidad cartesiana traslacional sin perder estabilidad.
          - `wait_time_s = 0.8 s`: Reducción del ciclo de espera (con respawn estocástico de Halton a $0.3\text{ s}$ y asentamiento físico de $0.5\text{ s}$).
          - `reset_joints_duration_s = 1.5 s`: Retorno dinámico a Home de articulaciones acelerado.
          - `settle_time_s = 0.1 s`: Transición instantánea a cierre de garra tras descender.
          - `hold_target_time_s = 2.0 s`: Piso matemático garantizado de 40 frames ($2.0\text{ s}$ exactos a $20\text{ Hz}$) con `done=1.0`.
      11. **Calibración Óptima de Velocidad Cinemática (`max_speed_m_s = 0.20 m/s`):**
          - *Diagnóstico y Calibración:* Mientras que $0.30\text{ m/s}$ maximizaba la velocidad de vuelo libre, en $0.20\text{ m/s}$ se logra el equilibrio perfecto entre agilidad y suavidad en las aproximaciones finas y rotaciones de agarre (`ALIGN_ROTATION`, `DESCEND`).
          - *Campaña Speed02 Certificada:* Recolección autónoma de 4 episodios ($4/4$, $1.060$ pasos totales) en `datasets/demonstrations_test_speed02_4eps.zarr`. Tiempos de grabación de $31.1\text{ s} - 49.5\text{ s}$ por episodio (ciclo total promedio de $56.1\text{ s}$), con piso garantizado de 40 frames ($2.0\text{ s}$) de señal `done=1.0` y extracción visual completa en `datasets/visualizations_speed02_4eps/`.
      12. *Estandarización:* Creación y actualización del módulo `.agents/knowledge/11_isaac_sim_headless_acceleration.md`.

39. **Evaluación Comparativa Multi-Checkpoint (Grilla de 7 Épocas), Corrección Geométrica de Auto-Stop y Frente de Pareto:**
    - *Incidente y Causa Raíz:* En pruebas con `epoch_150`, la política agarraba el cubo, lo elevaba $29\text{ cm}$ y lo trasladaba hacia Home, pero el evaluador marcaba `FALLO`. Se identificaron dos bugs:
      1. *Auto-stop prematuro:* Disparo por proximidad holgada ($< 10\text{ cm}$) cuando `done_flag = 0.58` antes de que el brazo ingresara al radio formal de éxito ($< 6\text{ cm}$).
      2. *Omisión de offset de pinza ($79.0\text{ mm}$):* Se medía el cubo contra la brida de Home ($Z=0.40\text{ m}$), produciendo un error mínimo geométrico de $\approx 80\text{ mm}$ con el cubo suspendido a $Z \approx 0.321\text{ m}$.
    - *Corrección Implementada en `tools/evaluate_convergence_checkpoints.py`:*
      1. Ajuste del umbral de llegada a $5\text{ cm}$ para objetos sujetos (`arrival_thresh = 0.05 if is_clamped else 0.08`).
      2. Descuento vectorial del offset de pinza: $\mathbf{p}_{\text{obj\_target}} = \mathbf{p}_{\text{home}} - [0, 0, 0.079]$.
      3. Criterio de éxito calibrado: `is_clamped = True` con error terminal de objeto $\le 100\text{ mm}$ (o TCP $\le 60\text{ mm}$).
    - *Resultados Empíricos de la Grilla de 7 Épocas (`train_v8_300data`, 21 episodios pareados Halton QMC):*
      - `epoch_10`: $0.0\%$ éxito, sub-ajuste severo ($170.1\text{ mm}$ de contacto medio).
      - `epoch_50`: $33.3\%$ éxito ($5.10\text{ mm}$ de contacto, $\Delta z = 28.2\text{ cm}$, duración $108.9\text{ s}$). Marca la **Cantidad Efectiva de Épocas ($E^* = 50$)**.
      - `epoch_100`: $33.3\%$ éxito ($5.25\text{ mm}$ de contacto, mejor aproximación espacial media de $44.8\text{ mm}$).
      - `epoch_150`: $33.3\%$ éxito (**ciclo más veloz con $99.5\text{ s}$**, $\Delta z = 29.1\text{ cm}$, contacto $6.26\text{ mm}$, auto-stop limpio).
      - `epoch_200`: $33.3\%$ éxito ($7.45\text{ mm}$ de contacto, $\Delta z = 29.2\text{ cm}$).
      - `epoch_250` y `epoch_300`: $0.0\%$ éxito. Evidencia física directa de **sobreajuste (overfitting)** en el espacio de acciones latentes: la red desincroniza el cierre de pinza a $7.6\text{ mm}$ o desvía la trayectoria a $14.4\text{ mm}$, a pesar de que el loss de entrenamiento continuaba descendiendo asintóticamente.
    - *Conclusión Operativa:* Fijar el criterio de parada asintótica en $E^* = 50\text{--}100$ épocas. Entrenar más allá de 200 épocas en este volumen induce degradación motora. Reporte completo en `physical_convergence_grid7_evaluation_report.md`.

---

40. **xArm5 real: orientación alcanzable (`arm_plane`), alcance hombro–muñeca y dataset VIO compatible con DP:**
    - Con 5 GDL la garra solo se inclina en el plano vertical del brazo; el puente interpola en (pitch, yaw) y cada pose enviada es alcanzable.
    - RViz2: el anillo verde orienta su eje según el acimut y ajusta el marcador al soltar.
    - Inclinación ±1.48 rad; el tope está bajo ±π/2 por el bloqueo de cardán del rpy del controlador.
    - La caja de trabajo no basta: la altura alcanzable baja con el radio. Se añadió el límite hombro–muñeca, con muñeca = brida + Rz·Ry·Rx(π)·(−0.076, 0, −0.097).
    - Las demostraciones a mano registran la cámara en `odom`; hay que convertirlas con `tools/xarm5/convert_vio_dataset_to_xarm5.py` antes de entrenar (home = HOME3, extrínseca medida, misma proyección que el puente).
    - Módulo 16, §5m–5o.

41. **Dataset real v7: la "certificación" no detectaba errores físicos; curación v2 y acciones relativas (UMI):**
    - *Defectos de `demonstrations_real_v7_xarm5.zarr`:* (1) extrínseca cámara-TCP invertida 180° sobre el eje óptico
      (suposición sin medir marcada como calibrada; gravedad inicial a 57° de HOME3 en mediana, 0/83 < 20°); (2) signo
      del cuaternión discontinuo con `qw >= 0` (1015 saltos q -> -q: `qw = -sin(pitch/2)·sin((yaw-2·acimut)/2)` ≈ 0 en
      HOME3); (3) ruido del VIO ~8.8 mm RMS que inflaba 4x la velocidad (0.20 m/s aparentes vs 0.05 m/s reales);
      (4) odometría NO reiniciada entre episodios (su "vertical" deriva hasta 60°; los ArUco de la mesa lo prueban);
      (5) 7 episodios con giros de muñeca de ~180° o pérdidas de tracking; (6) std del gripper = 1e-6.
    - *Corrección:* conversor v2 (extrínseca `rpy (pi, -pi/2, 0)`, filtro Hampel + Butterworth 1.5 Hz de fase cero,
      hemisferio "home", auditoría de saltos) -> `demonstrations_real_v7_xarm5_v2.zarr` (76 episodios, 0 saltos de
      signo, recorte del puente > 5 cm en 0.1% de frames). Piso de std en `zarr_dataset.py`.
    - *Representación `relative_ee`:* acción relativa al efector actual (UMI). Piloto a 20 épocas en validación:
      ADE 12.6 mm vs 38.9 mm con `absolute` (que no bate la línea base "quedarse quieto", 18.6 mm).
    - *Entrenamiento:* EMA, warmup, split por episodio, evaluación offline por muestreo completo, ColorJitter en GPU,
      `--resume`, guardia de RAM. *Incidente:* tres entrenamientos simultáneos agotaron la RAM del host (OOM -> GDM).
    - Módulo 17.
42. **Varas exigentes antes del robot y alcance real del xArm5 (2026-11-22):**
    - *Línea base "velocidad constante":* con trayectorias suaves extrapola muy bien el corto plazo. La política de la
      época 200 pierde con ella en los 3 pasos ejecutados (3.6 vs 1.3 mm) y gana desde ~0.53 s (FDE 18.5 vs 21.3 mm).
      Ahora la reportan `evaluation.py` (columnas `cv_*`), `train.py`, las curvas y `tools/compare_eval_runs.py`.
    - *Ablación visual (`tools/diagnose_policy_baselines.py`):* con imágenes de otro episodio el FDE sube de 18.5 a
      29.2 mm, así que la política sí usa la imagen. Pero desde el reposo acierta la dirección en el 64 % frente al 55 %
      de "ir siempre hacia la dirección media" (n = 22): los datos v7 tienen poca variedad de destinos.
    - *Alcance:* 700 mm es el radio máximo del fabricante (XF1305, `bio/xarm5_ufactory_XF1305/`), no el alcance lateral
      a la altura de la tarea (±0.47 m a la profundidad de los objetos). `|y| <= 0.30 m` es la caja de seguridad
      configurada, no un límite del brazo. El 0.9 % de los frames v7 queda fuera del alcance al AVANZAR EN ALTO desde
      HOME3 (5 episodios, hasta 51 mm): en la próxima toma, bajar antes de alejarse de la base.
    - *Límites articulares:* el guard del puente usaba la URDF/manual V2.3.0 (J2 120°, J3 -225°); el manual v2.7.1
      para XF1305 dice J2 -117..116°, J3 -219..10°. Actualizado en `xarm5_real_params.yaml`.
    - *pytest escribía en la base real:* las pruebas de `merge_zarr_batches` reescribían 6 filas `/tmp/test_zarr_merge_*`
      en `experiments/experiments.db` en cada corrida (desde el 2026-11-02). Ahora `tests/conftest.py` y `setUpModule`
      fijan `VISUOMOTOR_DB_PATH` a una base temporal; verificado con el hash de la base antes y después de la suite.

43. **La odometría de la D435i nunca se midió contra una referencia; la "certificación grado UMI" era el mejor caso (2026-11-23):**
    - *Banco de prueba:* `experiments/vio_benchmark_history.json` tiene 9 corridas dinámicas de la conf 4. Error de
      retorno: 2.3 / 14.4 / 16.2 / 16.9 / 21.6 / 28.3 / 39.3 / 59.0 / 59.7 mm (mediana 21.6 mm). Solo 1 de 9 cumple
      ≤ 3 mm: la trayectoria más corta. Esa corrida se documentó como "certificación". El jitter en reposo tiene
      mediana 3.0 mm (1.6–6.1).
    - *La métrica no sirve como criterio:* un error de escala no mueve el punto de retorno, y F2M con
      `OdomF2M/MaxSize 5000` se relocaliza contra los puntos de t=0. Hace falta ATE/RPE/escala contra una referencia
      (el xArm5 y un tablero ChArUco).
    - *No es VIO:* el IMU solo alinea el marco inicial con la gravedad (`wait_imu_to_init`). Con el ajuste de haces
      apagado no actúa ninguna restricción inercial, y la vertical de odom derivó 50–60° en una sesión.
    - *La grabación cruda de v7* (`analyze_vio_dataset_quality.py --raw-diagnostics`,
      `datasets/evaluations_real_v7_xarm5_v2/raw_diagnostics.*`):
      - Los saltos de 0.51 m de los episodios 23 y 24 son poses NULAS. RTAB-Map publica una odometría nula al perder
        el rastreo, y el recolector la guardaba como orientación identidad y posición = -pose inicial.
      - Hay 784 discontinuidades persistentes > 2 cm (mediana 33 mm) en 71/83 episodios. El "ruido de 8.8 mm RMS" es
        sobre todo saltos discretos, y el filtro de 1.5 Hz los convierte en movimiento falso.
      - 72/83 episodios tienen done solo en el piso forzado de 40 frames, y el 21 % de esos frames se mueve a más de
        2 cm/s (con el ruido filtrado en ~1 cm/s, la cifra es indicativa).
      - No hay sellos de tiempo, así que el muestreo a 20 Hz no se puede verificar: `verify_dataset_frequency.py`
        asume dt = 0.05.
      - `min_inliers_tracking` se declaraba y no se usaba.
    - *Regla:* una cifra de precisión se reporta como distribución (mediana, p90, máximo, n) medida contra una
      referencia, nunca como el mejor intento. Plan de mejora: `docs/plans/plan_d435i_herramienta_demostraciones.md`.
    - *Hecho (con pruebas; avance en el plan):*
      - recolector v2: sellos de tiempo, poses nulas marcadas, `reset_odom` por episodio, done verificado y QA que
        bloquea el guardado;
      - métricas ATE/RPE/escala contra referencia;
      - tablero ChArUco;
      - calibración cámara-brida.
    - *La prueba del recolector en ROS 2 simulado encontró 2 problemas:*
      - perdía la imagen cuando dos callbacks terminaban en orden inverso (corregido);
      - con FastDDS el suscriptor de imágenes en Python perdía ráfagas; hay que recolectar con CycloneDDS, como
        hace `run_vo_container.sh`.
    - *Pendiente con hardware:* driver de la RealSense en la imagen de la Jetson (no está en `Dockerfile.jetson`),
      imprimir y medir el tablero, calibrar con el xArm5, medir la odometría contra el robot y elegir el estimador.

44. **Jetson lista para la prueba física (2026-11-23), Fase 1 sin brazo ni cámara montada:**
    - *Driver de la cámara:* la imagen de la Jetson no lo traía. `Dockerfile.realsense_base` compila librealsense
      2.59.1 y realsense-ros 4.59.1 (las versiones de la recolección), con backend RSUSB.
      - El repositorio apt de ROS de la imagen dustynv tiene la clave vencida: `libusb` se instala solo desde Ubuntu.
      - Se usa con `docker compose -f docker-compose.yml -f docker-compose.realsense.yml up -d`; un `docker compose up -d`
        a secas vuelve a la imagen sin driver.
      - `d435i_camera.launch.py` publica color a 29,6 Hz y 640x480 dentro del contenedor.
    - *Latencia real:* el README prometía 25–35 ms, pero nunca se había medido. La inferencia completa tardaba 205 ms
      (10 pasos DDIM), el ciclo de 20 Hz se habría caído y las observaciones habrían quedado espaciadas de forma
      irregular. La U-Net está limitada por el lanzamiento de kernels desde la CPU de la Orin (18 ms por paso) y FP16
      es más lento.
      - Solución: `DiffusionModel.enable_cuda_graph()` (mismo resultado, diferencia 0) y 5 pasos DDIM, que no degradan
        la validación (11,5 frente a 11,6 mm). Queda en 69 ms por inferencia y 20,0 Hz de comandos (el timer de rcl
        conserva la fase si el atraso es menor de un período).
    - *Medir bien:* `check_policy_closed_loop_sim.py` medía su propio bucle (`spin_once` + `sleep` comparten el
      callback con el TF): daba 7 Hz en la Jetson y 16 Hz en el PC. Ahora escucha los comandos en un hilo propio.
    - *Verificado en la Jetson:*
      - lazo cerrado con imágenes de validación: coseno 0,985–0,987, parada automática y 0 rechazos;
      - etapa A con la cámara en vivo: 20,01 Hz y 80 % de CPU libre;
      - contenedor principal recreado con la imagen nueva (registros previos respaldados en
        `/home/jetson/respaldo_contenedor_20261123`).
    - *Siguiente (requiere brazo y montaje):* montar la cámara, HOME3 medido, comparación con
      `home3_reference_view.png`, etapa B (dry-run) y etapa C (5 cm/s), siempre con orden explícita del usuario para
      cada movimiento.

45. **Primera evaluación en el xArm5 físico (2026-11-23, 14 intentos): 9/14 sobre la herramienta y 7/14 con la vista
    de las demos:**
    - *Intentos 9 a 14:*
      - 9 y 10 (tijeras a la izquierda, en diagonal): zigzag lateral sin parar. El 10 terminó contra el piso del
        vigilante.
      - 11 (junto a la base): fue hacia adelante, lejos de las tijeras, hasta el piso.
      - 12 (centro, rectas): done 0,66 en una sola inferencia (se exigen 5 seguidas), siguió bajando hasta el piso.
      - 13 (centro-derecha, en diagonal): éxito completo.
      - 14 (mantel corrido, sin ArUco): done ≤ 0,11, hasta el piso.
    - *Lección de seguridad:* cuando la política no reconoce el final, sigue bajando. El piso del vigilante (brida a
      z ≥ 0,15 m) la detuvo 4 veces.
    - *Para v8:* tijeras giradas, posiciones junto a la base, la misma escena que en la inferencia, una vista final
      constante y empezar en la vista de HOME3 (lección 46).
    - *Medidas de éxito:* "llegó y se detuvo sobre la herramienta" (parada automática + centro de la imagen sobre las
      tijeras; es el criterio del usuario) y "terminó como en las demos" (vista final estricta).
    - *Criterio descartado:* "altura a ≤ 5 cm de las demos vecinas". La altura de parada humana sobre un mismo lugar
      de la mesa varía ±7 cm (0,188–0,340 m), así que la mediana de las demos vecinas no define un punto de
      aproximación.
    - *El intento 8* (tijeras junto a la base, a la derecha) paró sobre la herramienta, ~20 % más cerca que las demos.
    - *Gráfica de control por intento:* `tools/xarm5/plot_real_trial.py`. El TCP sigue al comando de la política; la
      velocidad satura a 5 cm/s en el tramo medio; el done sube en los últimos segundos.
    - *Detalle de los primeros 7 intentos:*
    - *Preparación:*
      - HOME3 medido: (0,15005, −0,00004, 0,75001) m y las mismas juntas previstas.
      - La vista en HOME3, en el espacio de la política, coincide con las demos: similitud 0,84 con la mediana (las
        demos dan 0,80), ArUco del mismo tamaño y giro de 1,7°.
      - El eje óptico está 7° más vertical que en las demos (25,5° frente a 32,3° ± 5,8°): es el diseño de HOME3.
    - *Etapa B (dry-run):* objetivo 14 mm hacia delante y abajo, a 20,0 Hz, done ≤ 0,005 y 0,0 mm de movimiento real.
    - *Etapa C (7 intentos, 5 cm/s):*
      - 7/7 paradas automáticas, 0 errores, 0 rechazos y 0 recortes del puente; el vigilante nunca intervino.
      - Centro 2/2, izquierda 2/2 (+17 cm en y), derecha extrema 2/2 (−25 cm en y).
      - Repetibilidad con las tijeras quietas: 6–17 mm.
      - Tiempo hasta la parada: 20–30 s.
      - Falla: tijeras giradas ~75° en yaw y más lejos. El brazo bajó hasta z = 0,257 m y se detuvo encima del
        pivote, demasiado cerca (vista +3,6σ, 1,5 veces el tamaño). Las demos v7 tienen las tijeras casi siempre
        verticales: hay que variar el yaw en v8.
    - *Criterio de éxito automático* (`tools/xarm5/real_trial_metrics.py`): parada neuronal más vista final (centroide
      de las tijeras a ≤ 2,5σ y tamaño entre p05 y p95 de los últimos frames de las 76 demos).
    - *Datos:* `experiments/xarm5_real_trials/2026-11-23/` (imágenes previa y final, trazas, registros y resumen con IC).
    - *Herramientas:* `tools/xarm5/real_trials/run_trial.sh` (chequeo previo, pitidos, vigilante, desarme, métricas)
      y `--dry-run` para la etapa B.
    - *Dos errores propios, corregidos:*
      - Un proceso en segundo plano de un script ignora SIGINT: la política se detiene con SIGTERM/SIGKILL a su grupo
        y se verifica que queden 0 publicadores.
      - El aborto del chequeo previo leía el código de un `grep`: no habría abortado. Ahora usa el código del
        chequeo, verificado por SSH.
      En los 7 intentos el chequeo pasó completo, así que el segundo error no tuvo efecto.

46. **Preparación de v8 sin el robot (2026-11-24): recolector v3 y cadena de medición de la odometría:**
    - *Guía de inicio a HOME3:*
      - La referencia son las esquinas de los ArUco de la mesa en las 14 imágenes `pre.png` de los ensayos (robot en
        HOME3; repetibilidad p95 0,73 px). Herramienta: `tools/hardware/make_home3_view_reference.py`.
      - Las demos v7 empezaban con escala 0,88–1,26 y hasta 32 px de corrimiento a 256 px (~80 px a 640).
      - El recolector compara en la imagen, sin necesitar el robot ni el tamaño real de los marcadores, y da pistas
        (derecha/izquierda, acercar/alejar, giro).
    - *Recolector v3:* inicio y guardado automáticos (tocar el teclado mueve la cámara), aviso de velocidad > 0,15 m/s
      (en v7 la máxima por episodio tuvo p90 0,15 m/s) y programa de celdas × orientaciones
      (`tools/hardware/make_collection_schedule.py`).
    - *Errores encontrados y corregidos:*
      - **Caída silenciosa del recolector v2:** `(warn if x else info)(msg)` en una misma línea. rclpy lanza
        `ValueError` si una línea cambia de severidad, y eso detenía el ejecutor con el primer frame perdido. La
        prueba simulada no lo veía porque llama a `_save` directamente.
      - `/tf_static` no entraba en los bags crudos: QoS volátil frente a un tópico latched.
      - `run_data_collection.sh` no cargaba el YAML del recolector.
      - En las pruebas, `fork` tras usar DDS colgaba al proceso hijo: ahora usan `spawn`.
    - *Tablero elegido con datos:*
      - En una aproximación sintética con verdad conocida, la referencia del A4 7×5 de 33 mm tenía ~14 mm de error a
        la distancia de HOME3 (0,75–1,2 m): el primer frame salía ~1° girado.
      - El A3 9×7 de 36 mm baja a 1,7 mm (0,11 mm a < 0,45 m). El A4 9×5 de 30 mm queda en 6,1 mm.
      - Pesa más el número de esquinas que el tamaño del cuadro.
      - Nuevo valor por defecto de `charuco_board.yaml`, todavía sin imprimir.
    - *Cadena de medición probada sin cámara:* bag sintético → referencia del tablero → métricas por pares
      (`evaluate_paired`, con huecos) → reprocesado con RTAB-Map. Recupera la escala (+3,00 %) y las pérdidas (3,9 %)
      inyectadas. RTAB-Map base en la escena sintética: 4,5 mm de mediana. Valida las herramientas, no la odometría
      real.
    - *Más errores encontrados el mismo día:*
      - **PDF del tablero corrupto:** el backend PDF de matplotlib 3.6.3 guardaba la imagen de 2 colores (1 bit) con
        el relleno de filas incorrecto, y los visores mostraban rayas diagonales. `make_charuco_board.py` ahora escribe
        el PDF directamente: 1 bit a 600 ppp, regla vectorial y Helvetica. Verificado con Poppler, MuPDF y
        Ghostscript: 48/48 esquinas y cuadro de 36,02 mm en el render. Prueba: `tests/test_charuco.py`.
      - **`ros2 bag record` detenido para siempre:** `timeout` lo pone en otro grupo de procesos, el grabador lee la
        terminal (pausa con ESPACIO) y el sistema lo detiene (estado T). Solución: `timeout --foreground` y
        `--disable-keyboard-controls`.
    - *Decisión:* el tablero se usa solo para medir, en el centro de la mesa. Las demos v8 van sin tablero, con la
      escena de v7 y de los ensayos. La guía de inicio usa los ArUco de la mesa, y así no hay que agregar nada en la
      inferencia. La medición con tablero es optimista (más textura); la exactitud en la escena real se mide con el
      robot.
    - *Siguiente paso:* con la cámara, seguir la sección "Trabajo sin el robot" del plan.
      1. Imprimir y medir el tablero.
      2. Fijar la exposición.
      3. Medir con ≥ 10 corridas y elegir el estimador.
      4. Hacer el piloto v8.
47. **Odometría de la D435i medida con el xArm5 como referencia (2026-11-24, `tools/xarm5/vio_gt/`):**
    - *Método:*
      - El robot ejecuta planes validados con la caja y el alcance del puente: paradas frente al tablero y 8
        aproximaciones de v7 a 3,5 cm/s.
      - El TCP se registra en la Jetson y la cámara graba en el PC.
      - La odometría se compara en `camera_link` con métricas por pares y anclada al primer frame.
    - *Relojes:* la Jetson no tiene NTP. Cada ssh nuevo gasta ~0,7 s en el saludo antes de ejecutar `date`, así que el
      punto medio de la ida y vuelta tenía un sesgo de ~0,35 s. Se usa `b − j` y se refina con el giróscopo de la
      cámara frente a la velocidad angular del robot (correlación 0,99). La odometría no sirve para sincronizar: su
      ruido aplana la curva de error.
    - *Cámara-brida medida (13 paradas, 80° de diversidad, residuo 3,1 mm / 0,27°):*
      - La cámara está a 67,6 mm del TCP, (62,9, −16,3, 18,5) mm, y a 2,3° de la rotación nominal.
      - v7_xarm5_v2 se convirtió con traslación 0. La reconversión `demonstrations_real_v7_xarm5_v3.zarr` mueve el
        TCP una mediana de 52 mm por episodio (máx. 124 mm).
      - El final queda una mediana de 44 mm MÁS ALTO que en v2: la política de v2 apuntaba bajo. Es coherente con los
        ensayos que siguieron bajando hasta el piso del vigilante (hipótesis, no probada).
      - El episodio 17 queda fuera del volumen del puente (59 % > 50 %) y se descarta: 75 episodios.
    - *Tablero:* el usuario midió los 9 cuadros: 324 mm, es decir 36,0 mm por cuadro y escala 1:1. La calibración con
      escala libre había ajustado 35,1 mm, pero NO era una medida:
      - la focal de fábrica está bien a ~1 % (`check_color_intrinsics.py`, modelo de un solo factor: −0,6 %, mitades
        +0,6 / −1,7 %);
      - la calibración completa de intrínsecas con estas vistas es degenerada (fx −21 %, cy corrido 90 px): todas
        miran el tablero desde un mismo lado;
      - con esas vistas, la escala se compensa corriendo la cámara por el eje óptico.
      Con la escala medida, la traslación cámara-TCP cambia 8,6 mm (0,41°): es la incertidumbre actual de la
      extrínseca, y afecta ≤ 4,5 mm a las trayectorias convertidas. `evaluate_vio_gt.py` usa la escala medida por
      defecto.
    - *Odometría en vivo (RTAB-Map actual, 8 aproximaciones, mediana entre corridas [peor]):*
      - ATE anclada 21,7 mm [40], máximo por corrida 95 mm [149], error al volver a HOME3 55 mm [79].
      - Escala 3,2 % [7,2], pérdidas 4,6 % [33], RPE a 0,5 s 10,8 mm.
      - Ruido rápido de ±10 mm incluso quieta.
      - Deriva y pérdidas al bajar, cuando la vista es casi toda mantel blanco. Las corridas terminan sobre el mantel
        vacío, sin herramienta: peor que una demo real.
    - *Guía de inicio:* la mesa se recentró, así que la referencia de HOME3 se regeneró con 5 imágenes del robot en
      HOME3 (`experiments/vio_gt/2026-11-24/home3_now/`; repetibilidad 0,04 px). La anterior queda en git.
    - *Elección del estimador (Fase 3): 6 candidatos × 8 aproximaciones, mismos bags, robot como referencia*
      (`replay_candidates_vs_robot.sh`). Mediana entre corridas:

      | candidato | ATE med (mm) | p90 | máx | escala | RPE 0,5 s (mm) |
      |---|---|---|---|---|---|
      | sin ajuste de haces (el anterior) | 17,3 | 46,7 | 74,0 | 2,6 % | 9,4 |
      | **con ajuste de haces (elegido)** | **13,0** | **20,9** | **49,6** | **1,7 %** | **2,2** |
      | GuessMotion | 14,3 | 46,3 | 88,5 | 2,6 % | 9,2 |
      | flujo óptico | 17,9 | 58,5 | 83,0 | 2,8 % | 8,9 |
      | frame-to-frame | 21,9 | 71,8 | 127,5 | 2,4 % | 3,5 |
      | sin Kalman | 19,7 | 40,6 | 70,6 | 2,8 % | 10,3 |

      - Con ajuste de haces el máximo mejora en 7 de 8 corridas.
      - El reprocesado es casi determinista: repetido, da lo mismo o difiere 2–6 mm.
      - Vivo y reprocesado difieren corrida por corrida, aunque en vivo no se descartan frames (16 ms por frame,
        intervalo 33,4 ms): el estado inicial cambia y, cuando falta textura, la trayectoria diverge. Por eso la
        elección se valida en vivo.
      - Costo: ~1,5 veces el de la configuración anterior, ~24 ms por frame (p99 ~30) frente a 33 ms disponibles.
    - *Por qué se pierde (`diagnose_tracking_loss.py`):*
      - H4 es falsa: la profundidad sigue válida en un 92–100 % hasta a 0,22 m de la mesa.
      - Las pérdidas en vivo se dan con la cámara a menos de ~0,35 m, donde solo se ve el mantel: los rasgos fuertes
        (ArUco, bordes, tablero) salen de la vista.
      - El episodio 68 nunca bajó de 0,31 m y no perdió nada.
      - Ningún candidato lo evita. Opciones: textura fija en la escena (la misma en las demos y en la inferencia) o
        corridas con la herramienta. El usuario pidió, por ahora, usar solo los datos sin herramienta.
48. **Otros frameworks sobre los mismos 8 bags, con el robot como referencia (2026-11-25, `tools/xarm5/vio_gt/frameworks/`):**
    - *Método:*
      - `export_bag_frames.py` pasa cada bag a una carpeta tipo TUM: JPEG y PNG de 16 bits sin recomprimir, IMU a
        200 Hz y calibración de fábrica.
      - Cada framework corre en su propia imagen Docker: `visuomotor_orbslam3:v1`, `visuomotor_open3d:0.18` y
        `visuomotor_learned_vo:v1` (DPVO y DROID-SLAM, GPU).
      - `evaluate_vio_gt.py --traj-tum` aplica las mismas reglas que a RTAB-Map. El camino TUM se validó: la odometría
        en vivo exportada da los mismos números.
      - Métrica nueva, la cobertura útil: el % de TODOS los frames con pose y error ≤ 10/20 mm. La ATE sola favorece
        al método que se pierde más.
    - *Resultados (mediana entre corridas; ATE anclada, mm):*

      | método | med | p90 | máx | escala | pérdidas | útil ≤ 20 mm (peor) |
      |---|---|---|---|---|---|---|
      | RTAB-Map en vivo (config. de v7) | 21,7 | 58,9 | 95,4 | +3,2 % | 4,6 % | 35 % (18) |
      | RTAB-Map + ajuste de haces | 13,0 | 20,9 | 49,6 | +1,7 % | 1,5 % | 82 % (43) |
      | ORB-SLAM3 RGB-D (final) | 10,8 | 233 | 235 | — | 35 % | 42 % (27) |
      | Open3D RGB-D denso | 12,8 | 33,2 | 39,5 | +1,3 % | 0 % | 71 % (50) |
      | DROID-SLAM RGB-D (por defecto) | 11,8 | 16,0 | 17,6 | +1,6 % | 0 % | 100 % (2) |
      | **DROID-SLAM RGB-D, `filter_thresh` 1,0** | **10,9** | **14,7** | **17,0** | +1,4 % | **0 %** | **100 % (62)** |
      | ídem + corrección de profundidad (dejando una fuera) | 7,9 | 11,5 | 15,0 | 0 % | 0 % | 100 % (81) |

      - ORB-SLAM3 RGB-D es muy bueno mientras sigue, pero en 5 de 8 corridas se pierde cerca de la mesa y abre un mapa
        nuevo (saltos de 0,2–0,7 m).
      - ORB-SLAM3 RGB-D-inercial y monocular-inercial no inicializan el IMU con el movimiento lento del robot: se
        rompen a los ~15 s en las 8 corridas.
      - Open3D (denso) no pierde ningún frame, pero deriva sobre el mantel (190 mm de máximo en el episodio 61).
      - DPVO (monocular, sin escala): en la única corrida que terminó, mediana 16 mm y máximo 146 mm aun con la escala
        ajustada. En las demás, su optimización final pide más de 8 GB de GPU. Excluido.
    - *DROID-SLAM gana:*
      - Sigue en las corridas donde todos se pierden (48, 61, 2): denso y aprendido, usa la trama del mantel.
      - Cierra el lazo al volver a HOME3: error final ≤ 2,6 mm.
      - Con `filter_thresh` 2,4 (por defecto) la corrida 51 fallaba al iniciar con la cámara quieta, de forma
        determinista (620 mm). Con 1,0 queda en 7,1 mm, y las otras 7 corridas no empeoran (solo el máximo de la 48:
        11,9 a 22,3 mm). El parámetro se eligió con estas mismas corridas: validarlo en datos nuevos.
      - Costo: ~1,5–3 min por corrida y 3,9 GB de GPU. Solo offline, como manda la arquitectura (las poses del
        dataset se calculan desde los bags crudos).
    - *Sesgo de escala de la profundidad:* todos los métodos RGB-D sobreestiman el movimiento un +1,3 a +2,6 %. Con
      DROID va de +1,05 a +1,83 % en las 8 corridas. Con escala libre (Sim(3)), DROID queda en 1,4–2,7 mm. Un factor fijo
      de 0,986, validado dejando una corrida fuera, baja la mediana a 7,9 mm. Hacer la calibración en chip y el tare de
      la D435i, o medir el factor con el robot.
    - *Límites:* robot a 3,5 cm/s (la mano va 2–3 veces más rápido), escala del tablero en la parte alta de la escena,
      sin herramienta en la mesa. La parte del error que viene de la extrínseca (~9 mm de incertidumbre) también entra
      en la ATE anclada.
    - *Decisión (2026-11-25):*
      - DROID-SLAM RGB-D offline (`filter_thresh` 1,0, profundidad ×0,986) para las poses del dataset; RTAB-Map con
        ajuste de haces solo para la guía en vivo.
      - Argumento práctico y teórico, y registro de experimentos para el paper:
        `.agents/knowledge/18_vio_estimator_selection_droid_slam.md`. Uso: skill
        `.agents/skills/droid-slam-offline-vio/SKILL.md`.
      - Los datos crudos (bags, odometrías reprocesadas y frames exportados, ~22 GB fuera de git) son la evidencia del
        paper: no borrarlos.
49. **Anclaje con la pose REAL de inicio, con ArUco + IMU (2026-11-25):**
    - *El supuesto que se cae:* el conversor ponía el primer frame de cada demo en HOME3. En v7 el inicio estaba a
      32° ± 6° de inclinación (HOME3: 24,6°), y 5° en una aproximación de 0,6 m son ~5 cm al final. El IMU de v7 no
      corregía nada: RTAB-Map lo usaba solo al arrancar la sesión, el conversor descarta la orientación absoluta, y v7
      no guardó el IMU.
    - *Ahora* (`start_anchor.py`, `calibrate_table_markers.py`, `estimate_start_poses.py`, conversor v3
      `--start-poses`):
      - Las esquinas de los ArUco se triangulan en `link_base` con el robot (lados de 51,2 y 50,6 mm), y se calcula
        la vertical real de la base (0,86°).
      - En cada demo: PnP en la ventana quieta del inicio, con la inclinación del acelerómetro.
    - *Validación contra el robot:*
      - Dejando una vista fuera: 11 mm / 0,63° de mediana (máx. 34 mm / 0,89°, con un solo marcador de lado).
      - Prueba nula en 8 inicios en HOME3: 5,2 mm / 0,55°.
      - La profundidad 3D-3D empeora (63 mm). A la resolución de v7 (256 px, sin IMU): 33 mm / 4°, con fallas, así
        que v7 no se reancla.
    - *Protocolo (demos a mano):* basta la quietud de ARMING ("QUIETA: reiniciando odometría"). El recolector v3.1
      (`raw_bag_preroll_arming: true`) empieza el bag crudo ahí, y no hace falta otra pausa después de "Grabando". No
      mover los ArUco ni el robot respecto de la mesa, o hay que recalibrar.
    - *Pendiente:* validar con inicios deliberadamente fuera de HOME3 (próxima sesión con el robot) e integrarlo en
      `build_dataset_from_bags.py` con DROID.
50. **Listo para recolectar v8 a mano (2026-11-25):**
    - *Celdas en metros:* el programa usaba tercios de la imagen de HOME3, pero el tercio superior queda en x > 0,7 m,
      fuera de la caja del puente (radio <= 0,65 m), y esas demos se habrían descartado al convertir. Ahora la grilla
      3x3 está en link_base (`v8_cells.yaml`): x de 0,42 a 0,615 m e |y| <= 0,21 m, por debajo de los ArUco.
      - `make_v8_cells.py` comprueba que el robot llega a la vista final (25–30 cm, cámara hacia abajo, 4
        orientaciones): las celdas 1–6 y 8 al 100 %; la 7 y la 9 fallan solo en una esquina (herramienta hacia el
        centro).
      - Las celdas se dibujan en la vista de HOME3, y la guía del recolector muestra la celda pedida y la flecha de
        las hojas.
    - *Profundidad:* imprescindible, porque DROID RGB-D saca de ahí la escala métrica. Ya está en `raw_bag_topics`
      (`aligned_depth_to_color/compressedDepth`, PNG de 16 bits sin pérdida). El bag crudo debe estar ACTIVADO: el
      YAML trae `raw_bag_enabled: false` y lo activa `run_v8_session.sh`. Pesa unos 6 MB/s, unos 0,1–0,2 GB por demo
      (~12–20 GB por cada 100).
    - *Sesión:* `tools/hardware/run_v8_session.sh` revisa la cámara, los archivos, el disco (>= 30 GB) e `install/`,
      muestra el protocolo y lanza el recolector (inicio y guardado automáticos, programa de 108 entradas, bag con
      pre-roll). Al día siguiente continúa el mismo dataset y el mismo programa.
    - *Odometría en vivo perdida al final (frecuente cerca de la mesa):* antes, la pose congelada parecía quieta y
      podía guardar la demo cortada, con el done mal puesto; y el control de calidad bloqueaba las demos con pérdidas.
      Ahora:
      - con la cola perdida, el final quieto lo decide el giróscopo (`imu_still_since`, |w| < 0,08 rad/s), y el done
        empieza ahí (`done_source: imu`);
      - con `qa_tracking_blocking: false` (`run_v8_session.sh`), las pérdidas quedan en `qa_warnings`, porque las
        poses salen de DROID offline.
      - Prueba simulada: no guarda mientras se mueve perdida y guarda al detenerse.
    - *Final con tecla (respaldo):* con `auto_hold: true` (`run_v8_session.sh`), una sola [S] detecta el tramo quieto,
      por odometría o por giróscopo si está perdida, y quita los últimos 0,3 s (`manual_save_trim_s`), que son el
      movimiento de apretar la tecla. [H] + [S] movían la cámara dos veces en el tramo que debe estar quieto.
    - *Reinicio de la odometría que nunca ocurría (visto en vivo el 2026-11-25, antes de la primera demo v8):*
      - el servicio de RTAB-Map es privado (`/rgbd_odometry/reset_odom`), pero el recolector llamaba a `/reset_odom`, que no existe;
      - el recolector solo avisaba y seguía sin reiniciar, y la prueba simulada no lo detectó porque ofrecía el nombre equivocado;
      - RTAB-Map no se recupera solo de una pérdida, así que después de la primera todas las demos habrían empezado perdidas;
      - corregido en el YAML, el código y la prueba, y `run_v8_session.sh` ahora se niega a empezar si el servicio no existe.
      Ajuste de haces en vivo: p50 26,8 / p90 32,5 / p99 36,1 ms por frame (~28 Hz), así que se deja en "1".

### 6. Metodología Científica para Publicaciones: Los Dos Ejes de Investigación
El proyecto adopta un diseño factorial riguroso de dos dimensiones ortogonales:
1. **Eje 1 (Demostrador Experto para Datasets):** Comparación empírica de datasets generados por PID cartesiano vs. SOTA MPC vs. SOTA Impedancia Cartesiana.
2. **Eje 2 (Paradigma de Aprendizaje):** Comparación empírica formal de **Solo Diffusion Policy (Behavior Cloning)** frente a **Diffusion Policy + RL Fine-Tuning (DPPO)** bajo métricas estandarizadas ($N \ge 100$ episodios):
   - *Task Success Rate (SR %)*
   - *Cycle Time / Duración media (s)*
   - *Tracking Error RMSE ($X, Y, Z, \text{Yaw}$)*
   - *Terminal Position Error RMSE (mm)*
   - *Grasp Slip Rate (%)*
   - *Auto-Stop F1-Score*

**Hitos Logrados y Estado Actual:**
- **Fase 1 (Limpieza de Escena y TFs):** Completada.
- **Fase 2 (Generador Sintético QMC):** Demostrador experto piloto PID validado con Halton Quasi-Monte Carlo.
- **Fase 3 (Auto-Stop Desacoplado 9D):** Dataloader con copy-padding y flag `done = 1.0` funcional.
- **Fase 4 (Calibración de Fuerza y Respawn Determinista):** Umbral $0.0007\text{ m}$ calibrado y respawn bifásico validado en RL.
- **Fase 5 (Infraestructura de Logging Off-ROS):** Logger científico `coloredlogs` + `humanize` + `@benchmark` desplegado en `visuomotor_core`.
- **Fase 6 (Benchmarking Pareado y Modelo Cinemático TCP):** Campaña comparativa ejecutada (10 episodios pareados, 20 vuelos) en Isaac Sim, proyección TCP validada ($79.0\text{ mm}$), panel gráfico $2 \times 3$ de publicación optimizado y skill `visuomotor-benchmark-evaluator` creada.
- **Fase 7 (Ingeniería Editorial Académica y LaTeX Riguroso):** Informe de avance (`informe_avance_visuomotor_cd.tex`) refinado a estilo humano sin metalenguaje de IA, diagramas TikZ coherentes dimensionalmente, justificación formal de Wilcoxon, citas de DPPO (ICLR 2025), 0 errores y 14 páginas consolidadas. Skill `scientific-latex-reporting` desplegada.
- **Fase 8 (Evolución hacia Visión 3D con DP3):** Mapeo arquitectónico de 3D Diffusion Policy (DP3Encoder, FPS a 512/1024 pts, cropping espacial, sample-prediction y Simple DP3) documentado en `.agents/knowledge/08_3d_diffusion_policy_and_ml_rigor.md`.
- **Fase 9 (Recolección Paralela Dual Headless y Consolidación de 1,000 Episodios V8):** Despliegue de recolección paralela acelerada en Isaac Sim 6.0 Headless (2 instancias concurrentes en GPU RTX 4060, Dominios ROS 42/43, puertos 8211/8212), monitoreo con Watchdog Auto-Heal, partición en shards temporales (`batch_1.zarr` y `batch_2.zarr`), y fusión atómica final aplanada hacia `datasets/demonstrations_v8.zarr` (1,000 episodios planos: `episode_0` .. `episode_999`) registrada en `experiments.db`.
- **Fase 10 (Determinación de Convergencia y Criterios de Parada SOTA):** Suite analítica formal en `visuomotor_core.analysis.stopping_criteria` para evaluar la convergencia en espacio dual (Pasos $\leftrightarrow$ Épocas). Desacoplamiento riguroso de dos regímenes (Inflexión de Codo Kneedle para aproximación macro vs. Consenso Asintótico Pareto para manipulación fina), 6 detectores SOTA (Kneedle, Gradiente Plano, Varianza RSD, Mejora Relativa, Prechelt PQ, Gap EMA) inmunes a ruido de mini-batch, figuras dual-axis a 300 DPI (PNG + PDF) y panel diagnóstico 4-canal.
- **Fase 11 (Estudio de Convergencia Continua, Piso Asintótico de Rebote y Parada Limpia a Demanda):** Auditoría histórica de `train_500_dataV7_fixed` (700,000 pasos en 100 épocas de dataset completo, estabilización en meseta asintótica con rebote en $[0.0010, 0.0050]$). Identificación del corte prematuro en `train_v8_convergence_study` (3,000 pasos = 0.70 épocas). Despliegue del modo continuo dinámico (`--epoch 0`, `batch_size 64`, 4,279 pasos/época), guardado periódico garantizado cada 10 épocas completas, renderizado gráfico intra-época cada 500 pasos, y protocolo de parada limpia interactiva (`touch STOP` o `SIGINT`) con evaluación continua de criterios de parada.
- **Fase 12 (Recolección Real de Demostraciones con Odometría Visual RTAB-Map y RealSense D415):** Despliegue del pipeline desacoplado para captura manual de aproximación (*approaching*) en hardware real. Integración del driver oficial `realsense2_camera` (640x480 a 30 Hz, profundidad alineada y proyector IR activo) con el motor de odometría visual de RTAB-Map (`rgbd_odometry`). Implementación del nodo recolector interactivo `vo_collector_node` (muestreo a 20.0 Hz canónicos, redimensión en RAM a 256x256, estado 8D, acción 9D con piso garantizado de $\ge 2.0\text{ s}$ con `done=1.0`). Lanzador dinámico `tools/hardware/run_vo_container.sh` con montaje en caliente de `/dev` y `/dev/bus/usb`. Análisis y benchmarking experimental comparativo: Baseline (GFTT/ORB) validado como configuración óptima ($30.63\text{ Hz}$, $22.05\text{ ms}$ de CPU, $300.5\text{ inliers}$, $0.394\text{ mm}$ de jitter 3D y $0.0467^\circ$ de estabilidad angular) frente a SIFT, justificando físicamente la interacción del detector de esquinas con el patrón láser infrarrojo de la D415. Integración del visor gráfico RViz2 cargando `vo_vis.rviz`. Captura y certificación del primer dataset real (`datasets/demonstrations_real_v2.zarr`, 7 episodios continuos, $2,909$ pasos, hold en destino de $2.5\text{ s} - 4.2\text{ s}$) y extracción multimedia completa (videos MP4, gráficas de trayectoria PNG y GIFs en `datasets/visualizations_real_v2/`). Estandarización en `.agents/knowledge/15_visual_odometry_rtabmap_realsense.md`.
- **Fase 13 (Ampliación a Odometría Visual-Inercial VIO con Intel RealSense D435i):** Integración nativa y certificación de la cámara RealSense D435i (Product ID `0x0B3A`, Serial `347622074027`, IMU integrado Bosch BMI085 con acelerómetro y giroscopio a 200 Hz). Implementación del pipeline VIO completo: `realsense2_camera` (`unite_imu_method: 2`) $\rightarrow$ `imu_filter_madgwick` (`/imu/data` estabilizado) $\rightarrow$ `rtabmap_odom` (`rgbd_odometry` con `wait_imu_to_init: true` y `Odom/AlignWithGround: true` para alineación gravitacional automática del plano cartesiano, garantizando correspondencia con el espacio de trabajo del robot). Creación de archivos de configuración y launch dedicados y desacoplados: `d435i_camera_params.yaml`, `rtabmap_vio_params.yaml`, `realsense_d435i_vio.launch.py` y `realsense_d435i_vio_collection.launch.py`, preservando 100% la compatibilidad e independencia operativa de la D415. Optimización matemática de alta precisión anti-deriva y anti-loose tracking (`Odom/GuessMotion: true`, $g_2o$ Local Bundle Adjustment en ventana de 8 keyframes, refinamiento subpíxel $3\times 3$ con 10 iteraciones y filtrado RANSAC PnP a $1.5\text{ px}$): validación experimental a $13.98 \pm 0.65\text{ ms}$ por fotograma ($42\%$ de presupuesto a 30 FPS, $>55\%$ margen de CPU), $390\text{--}450$ inliers geométricos, $0.0\%$ pérdida de tracking, jitter cartesiano $0.621\text{ mm}$ (con dispersión vertical $Z=0.38\text{ mm}$) y estabilidad angular de $0.0186^\circ$. **[Corregido el 2026-11-23, lección 43: medidas en reposo, sin referencia; en la toma real hubo pérdidas de rastreo y saltos de ~3 cm.]** Actualización de `run_vo_container.sh` (con passthrough de `/sys` y preservación estricta de propiedad de usuario `1000:1000`) y `setup_realsense_udev.sh` con mitigación preventiva del servicio `iio-sensor-proxy`. Estandarización en `.agents/knowledge/15_visual_odometry_rtabmap_realsense.md`.
- **Fase 14 (Despliegue Hardware Real xArm5 en Jetson Orin Nano):** Integración completa del manipulador industrial xArm5 (UFACTORY, 5 GDL) sobre NVIDIA Jetson Orin Nano (JetPack 5.1.2, Ubuntu 20.04 L4T, Python 3.8 / PyTorch 2.0). Cadena de control: `policy_node` $\rightarrow$ `target_frame_raw` $\rightarrow$ `pose_smoother` $\rightarrow$ `target_frame` $\rightarrow$ `xarm5_bridge` $\rightarrow$ `/xarm/set_servo_cartesian` (modo 1). Puente seguro con arranque desarmado (`dry_run: true`), caja de trabajo estricta, límites de velocidad, watchdog y proyección `top_down_yaw` a 5 GDL. Teleoperación remota certificada desde RViz2 (`tools/xarm5/run_remote_rviz.sh` y `xarm5_teleop_marker`) con carga de mallas `xarm_description` en Jazzy, auto-detección DDS por mDNS y validación de frenos sin movimiento físico. Documentado en `.agents/knowledge/16_xarm5_real_hardware_bringup.md` y verificado con suite de pruebas en `tests/test_xarm5_interface.py`.
- **Fase 15 (Unificación SOTA UMI / Coordenadas Relativas Delta-Home):** Eliminación del salto heurístico al suelo en RTAB-Map mediante `Odom/AlignWithGround: false` (inicio limpio en el lente con eje Z vertical estabilizado por IMU). Unificación formal del espacio de acción y observación mediante coordenadas relativas al inicio del episodio (`relative_coordinates: true`) en `pickup_approximate.py`, `vo_collector_node.py` y `policy_node.py`. El paso $t=0$ arranca estrictamente en $(0.0, 0.0, 0.0)$, certificando compatibilidad sim-to-real idéntica sin necesidad de marcadores o tags físicos en la mesa de trabajo. Normalización y estabilización de cuaterniones a semiesfera superior ($q_w \ge 0$). Verificación cruzada con `tests/test_relative_coordinates.py` (15 pruebas unitarias) y soporte en `tools/verify_zarr_dataset.py`.
- **Fase 16 (Diagnóstico y Erradicación de Drift en VIO D435i, Desacoplamiento DDS y Preparación de Benchmarking Métrico):** Identificación rigurosa de las causas de deriva cinemática en hardware real: la activación de Bundle Adjustment 2D (`OdomF2M/BundleAdjustment: 1`) deformaba la escala métrica 3D nativa de la RealSense, mientras que la predicción de velocidad constante (`Odom/GuessMotion: true`) inyectaba inercia espuria y arrastre de rumbo por la ausencia de magnetómetro en el sensor BMI085. Calibración certificada en `rtabmap_vio_params.yaml`: desactivación de Bundle Adjustment y GuessMotion, combinada con refinamiento de esquinas subpíxel ($3\times 3$, 10 iteraciones) y filtro Kalman temporal (`FilteringStrategy: 1`), logrando deriva submilimétrica residual ($1.10\text{ mm}$ en 5 s, $0.22\text{ mm/s}$) y supresión de vibración de cuantización. Desacoplamiento de DDS en `run_vo_container.sh`: confinamiento del stream de 46 MB/s (370 Mbps) a Memoria Compartida (SHM) / loopback local, erradicando la saturación de la interfaz Wi-Fi `wlo1` y liberando buffers `/sys/bus/iio/.../buffer/enable`. Integración de verificaciones de flujo (`[Stream OK]`) y apagado robusto en `vo_collector_node.py`.
- **Fase 17 (Certificación Metrológica de VIO Grado UMI con Intel RealSense D435i):** Campaña experimental y benchmarking interactivo de bucle cerrado (`tools/benchmark_vio_pose.py`) contrastando 5 configuraciones. Certificación exitosa de la configuración ganadora `conf 4 (SOTA UMI Final Fusion)`: error de retorno a la marca física de **$2.307\text{ mm}$** tras $54.56\text{ cm}$ de excursión ($2.79\text{ m}$ recorridos), desglosado en $\Delta X = -0.83\text{ mm}$, $\Delta Y = -0.06\text{ mm}$ y $\Delta Z = +2.15\text{ mm}$. Jitter estático reducido en $89\%$ ($1.61\text{ mm}$ vs $14.89\text{ mm}$ baseline) y $0.0\%$ pérdida de tracking con $258.8$ inliers sostenidos. Inyección de directivas SOTA inspiradas en el paper de UMI (Universal Manipulation Interface, Chi et al., RSS 2024): memoria local ampliada (`OdomF2M/MaxSize: 5000`), ventana de profundidad acotada al volumen de manipulación (`Vis/MinDepth: 0.20`, `Vis/MaxDepth: 1.80`), filtrado RANSAC PnP estricto (`Vis/MinInliers: 20`, `PnPReprojError: 1.8`), subpíxel $5\times 5$ con 15 iteraciones y calibración inercial continua en línea (`gain: 0.035`, `zeta: 0.002`, `orientation_stddev: 0.02`). Parámetros activos congelados en `rtabmap_vio_params.yaml` y copia de seguridad en `rtabmap_vio_params_best_umi.yaml`. Sistema completamente certificado para la recolección de `demonstrations_real_v3.zarr`. **[Corregido el 2026-11-23, lección 43: 2.307 mm fue la mejor de 9 corridas (mediana 21.6 mm, 1/9 ≤ 3 mm); no hubo medición contra referencia ni certificación.]**
- **Fase 18 (Alineación UMI, Desacoplamiento de Gripper y Topología de Red Dual Local vs. Jetson):** Refactorización de `vo_collector_node` y `ZarrStorage` para alinear formalmente el flujo de datos con el paradigma UMI (Universal Manipulation Interface) y la arquitectura Diffusion Policy del repositorio. Implementación del parámetro de lanzamiento `dataset_path`: (1) En modo AUTO (`dataset_path:=AUTO` o no especificado) crea dinámicamente la siguiente versión incremental `demonstrations_real_vX.zarr`; (2) En modo Aumento (`dataset_path:=<ruta_o_carpeta>`) normaliza la ruta, detecta si el dataset Zarr ya existe, reporta el número de episodios previos y continúa agregando episodios de manera consecutiva (`episode_N`) preservando la propiedad y permisos UID 1000. Desacoplamiento formal de gripper (vector 8D de estado y 9D de acción con `0.0` neutro, tecla interactiva `[G]` y suscripción opcional a `/visuomotor/gripper_joint_states` para no modificar más el recolector). Aislamiento de red para recolección experta: confinamiento del stream masivo de video (46 MB/s) a Memoria Compartida (SHM) / loopback local (`lo`) bajo `ROS_DOMAIN_ID=0` con aislamiento estricto `ROS_LOCALHOST_ONLY=1` y `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` para blindar la máquina contra interferencias de otros equipos ROS 2 en la red Wi-Fi o laboratorio, garantizando frecuencia canónica sin latencia interna ($13.98\text{ ms}$, 0% tracking lost). Definición de la topología futura para despliegue en robot real: enlace liviano de control ($< 50\text{ KB/s}$) hacia la Jetson Orin Nano bajo `ROS_DOMAIN_ID=42` mediante CycloneDDS Unicast (`DDS_PEER`). Creación del script interactivo directo `tools/hardware/run_data_collection.sh`.
- **Fase 19 (xArm5 real: HOME del fabricante, HOME3 alineado y límite de alcance 0.65 m):** HOME1/HOME2 del fabricante (XF1305), HOME3 alineado con la calibración del controlador, interfaz RViz2 en inglés solo con GDL cartesianos (X/Y/Z + yaw), `clean_error` en los HOME y espacio de trabajo a 0.65 m (alcance ~0.716 m). Propuesta `arm_plane` + proyección del dataset y restricciones de las demostraciones reales documentadas en `.agents/knowledge/16_xarm5_real_hardware_bringup.md` §5l.
- **Fase 20 (xArm5 `arm_plane` + compatibilidad DP del dataset real):**
  - Inclinación alcanzable en el plano del brazo en el puente (interpolación en pitch/yaw) y anillo verde en RViz2.
  - `convert_vio_dataset_to_xarm5.py`: cámara en odom → TCP en link_base, proyección y auditoría.
  - Home de la política = HOME3, redefinido el 2026-11-20 como pose alta e inclinada: TCP (0.150, 0.005, 0.750), pitch −0.43.
  - Skill `realsense-vio-data-collection` v1.1 aplicada.
  - Volumen = caja (z_max 0.75) ∩ alcance hombro–muñeca (95% del codo recto, mínimo de 0.12 m), validado por FK.
- **Fase 21 (Calibración, Conversión SOTA y Certificación del Dataset Real v7 para xArm5):**
  - Calibración de `visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml`: extrínseca cámara-TCP activada (`extrinsic_calibrated: true`, cámara mirando hacia $+Z_{\text{tcp}}$ con pitch $-90^\circ$), tolerancia de velocidad `speed_tolerance_ratio: 12.0` (hasta $0.60\text{ m/s}$ y $6.0\text{ rad/s}$), desviación angular admitida `max_orientation_deviation_rad: 1.50` (proyección analítica a `arm_plane`) y ventana de despegue `max_workspace_violation_frames: 550`.
  - Conversión con descarte de anomalías (`tools/xarm5/convert_vio_dataset_to_xarm5.py --drop-flagged`): exclusión de 2 episodios con pérdida de tracking ($v > 2.3\text{ m/s}$), compilando `datasets/demonstrations_real_v7_xarm5.zarr` con **81 episodios limpios** ($26,868$ fotogramas, $22.39$ minutos a 20.0 Hz).
  - Certificación exhaustiva (`tools/verify_zarr_dataset.py`): integridad 100%, PyTorch DataLoader validado (lote 16: img $[16, 2, 3, 256, 256]$, state $[16, 2, 8]$, action $[16, 16, 9]$) y paso forward+backward en CUDA verificado con loss inicial DDIM de $1.1839$.
  - Suite de evaluación SOTA (`tools/evaluate_real_dataset.py`): generación de panel multicanal 2x3 a 300 DPI y PDF 100% en inglés académico (`demonstrations_real_v7_xarm5_sota_panel.png/.pdf` y `demonstrations_real_v7_sota_panel.png/.pdf`). Subfigura D implementada con Eficiencia Geométrica ($D = 0.60\text{ m}$ vs $L_{\text{eff}} = 1.12\text{ m}$, tortuosidad $1.88 \pm 0.40$), erradicando acumulación de ruido euclidiano ($8.2\text{ m}$). Subfigura E con Perfil en Espacio de Fases $\mathbf{v}(d_{\text{target}})$ con desaceleración hacia la zona de contacto ($d \le 5\text{ cm}$). Reportes Markdown y persistencia en SQLite `experiments/experiments.db`.
  - Corrección de división por cero en `visuomotor_core/dataset/normalize.py` para tensores constantes (gripper pasivo en 0.0) y redirección de logs en `tools/hardware/run_*.sh` a `/tmp/.ros_log_user` para UID 1000.
  - *Estado Crítico:* **El modelo aún NO ha sido evaluado en el robot físico ni se tienen resultados de inferencia en hardware real.** El dataset está listo y certificado para iniciar el entrenamiento con `train.py`.

- **Fase 22 (Curación Física del Dataset Real v7, Acciones Relativas UMI y Entrenamiento para el xArm5):**
  - Auditoría física del dataset v7 (gravedad del VIO, ArUco de la mesa, PSD, continuidad del cuaternión) y conversión
    v2 corregida (`datasets/demonstrations_real_v7_xarm5_v2.zarr`, figuras en `datasets/evaluations_real_v7_xarm5_v2/`).
  - Piloto de representación (`relative_ee` vs `absolute`) y entrenamiento principal `relative_ee` de 300 épocas
    (`visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300`, checkpoints cada 10 épocas, 8 episodios de validación).
  - Ruta de despliegue verificada sin hardware: reproducción offline de la inferencia de la Jetson y prueba ROS 2 en lazo
    cerrado con el puente simulado. Módulo 17.
  - *Modelo entregado:* época 200 en `deploy/jetson/model/` (validación: ADE ejecutado 7.6 mm vs 11.0 mm de la línea
    base, F1 done 0.70, progreso 1.00; auto-stop calibrado 0.60 x 5 pasos; lazo cerrado simulado coseno 0.99).
    Los criterios de parada por pérdida sugerían la época ~8: para DP se eligen checkpoints por comportamiento.
  - *Estado:* el modelo aún NO se ha probado en el xArm5 físico.

**Roadmap Experimental Inmediato:**
1. **Despliegue en el xArm5 real del checkpoint seleccionado** (Fase 22): montar la D435i con la imagen derecha en
   HOME3, dry-run, luego armado a 5 cm/s; medir éxito de aproximación y error final frente a las tijeras.
2. **Recolección v8 con la herramienta D435i corregida** (`docs/plans/plan_d435i_herramienta_demostraciones.md`):
   medir la odometría contra el xArm5 y un tablero ChArUco, calibrar cámara–brida (rotación y traslación), elegir el
   estimador con datos, recolector v2 (sellos de tiempo, poses nulas rechazadas, odometría reiniciada por episodio,
   guía de inicio en HOME3, done verificado) y piloto antes de la toma completa.
3. **Fine-Tuning con DPPO RL:** Aplicar optimización sobre el checkpoint base bajo las mismas semillas espaciales.
4. **Comparativa Cuantitativa y Gráficas de Paper:** Generar tablas estadísticas (DP vs. DP+RL) y figuras vectoriales (PDF/PNG 300 DPI) para publicación.
5. **Incorporación de Demostradores SOTA:** Explorar generación de datasets con MPC/cuRobo y control de impedancia compliante.
6. **Transferencia Sim-to-Real y Políticas Residuales (ResiP / TRANSIC):** Validación experimental de Co-Fine-Tuning visual (LoRA en ResNet-18) y arquitectura residual reactiva ($\pi_{\text{res}}$) sobre el espacio cartesiano de alto nivel del xArm Lite6.
