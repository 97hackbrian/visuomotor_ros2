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
