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
* `state[3:7]`: Orientación del TCP como cuaternión $[q_x, q_y, q_z, q_w]$ (con $q_w \ge 0$).
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

## 6. Comandos de Referencia Rápida

```bash
# 1. Compilación del workspace dentro del contenedor
docker exec -u hackbrian -w /ros visuomotor_ros2 bash -c "colcon build --symlink-install"

# 2. Lanzar recolección autónoma de datos (genera automáticamente demonstrations_vX.zarr)
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_data_collection pickup_approximate.launch.py"

# 3. Entrenar Diffusion Policy con el dataset parchado o nuevo
docker exec -it -u hackbrian visuomotor_ros2 bash -c "python3 /ros/visuomotor_core/train.py --root /ros/visuomotor_core/datasets/demonstrations_v7_done_fixed.zarr --epoch 100 --batch_size 64 --out_dir /ros/visuomotor_core/outputs/train_500_dataV7_fixed"

# 4. Lanzar inferencia en vivo con el nodo de política
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_ros2 inference.launch.py"
```

---

## 7. Contrato de Mantenimiento Continuo (Agent Handoff)

Al concluir cada sesión de desarrollo donde se hayan tomado decisiones arquitectónicas, solucionado bugs o actualizado datasets/pesos:
1. El agente activo DEBE actualizar `context.md` y este archivo `AGENTS.md`.
2. El agente activo DEBE registrar los nuevos hitos en la sección de progreso.
3. Se puede invocar el protocolo automatizado definido en `UPDATE_KNOWLEDGE.md`.
