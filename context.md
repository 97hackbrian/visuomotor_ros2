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

---

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

**Roadmap Experimental Inmediato:**
1. **Toma de Dataset Fresco V8:** Generar `demonstrations_v8.zarr` con el demostrador experto PID actualizado (`hold_target_time_s: 3.5s`, `record_done_flag: true`).
2. **Entrenamiento y Evaluación de Solo DP (Baseline):** Entrenar con `train.py` instrumentado y evaluar 100 episodios automatizados generando telemetría en `metrics.csv`.
3. **Fine-Tuning con DPPO RL:** Aplicar optimización sobre el checkpoint base bajo las mismas semillas espaciales.
4. **Comparativa Cuantitativa y Gráficas de Paper:** Generar tablas estadísticas (DP vs. DP+RL) y figuras vectoriales (PDF/PNG 300 DPI) para publicación.
5. **Incorporación de Demostradores SOTA:** Explorar generación de datasets con MPC/cuRobo y control de impedancia compliante.
