# Visuomotor Generative Framework (ROS 2 Jazzy)

Este meta-repositorio (`visuomotor_ros2`) consolida la arquitectura completa para la recolección autónoma de datos, entrenamiento e inferencia en tiempo real de **Políticas Visuomotoras Generativas (Diffusion Policy, DDIM Schedule)** para manipuladores robóticos sobre **ROS 2 Jazzy**.

---

## 1. Topología Modular (5 Silos Desacoplados)

El repositorio mantiene una separación estricta entre Machine Learning puro (Off-ROS), la capa de mensajes ROS 2, la recolección autónoma, el nodo de inferencia y la abstracción de hardware:

1. **`visuomotor_core`** *(Python Puro / Off-ROS)*:
   - Redes neuronales en PyTorch (1D Temporal U-Net con GroupNorm y Mish + ResNet-18 con SpatialSoftmax y FiLM).
   - Solvers generativos (Squared-Cosine Beta Schedule con muestreador DDIM acelerado a 10 pasos).
   - Ingesta de datos Zarr con **Copy-Padding** y **15% Proprioception Dropout**.
   - Script de entrenamiento (`train.py`) con AdamW, CosineAnnealingLR y precisión mixta `bfloat16`.
2. **`visuomotor_msgs`** *(ament_cmake)*:
   - Interfaces estándar agnósticas al hardware (`ActionChunk.msg`, `ObservationData.msg`, `EpisodeTrigger.srv`).
3. **`visuomotor_data_collection`** *(ament_python)*:
   - Recolección autónoma experta (`pickup_approximate.py`) mediante FSM de 12 estados + PID Cartesiano a 100 Hz con setpoint virtual anti-escape.
   - Respawn estocástico de baja discrepancia (Halton Quasi-Monte Carlo 3D).
   - Empaquetado de episodios en formato Zarr (`demonstrations_vX.zarr`) con inyección garantizada de la señal de término.
4. **`visuomotor_ros2`** *(ament_python)*:
   - Runtime de inferencia en tiempo real (`policy_node.py`) operando a 20 Hz.
   - Sincronización sensorial mediante `ApproximateTimeSynchronizer`.
   - **Temporal Ensemble** con decaimiento exponencial ($w_j = \exp(-k \cdot j)$).
   - **Doble Capa de Auto-Stop Desacoplado** (Neuronal en la 9ª dimensión + Heurístico por varianza espacial).
5. **`visuomotor_bringup`** *(ament_cmake)*:
   - Capa de hardware y simulación (Isaac Sim / xArm Lite6).
   - Filtro adaptativo **1-Euro** (`pose_smoother.py`) con desenvolvimiento de Euler para eliminación de jitter sin retardo cinemático.
   - Lanzadores completos con ROS 2 Control (`cartesian_motion_controller` + `position_controller`).

---

## 2. Especificación de Tensores

* **Observación Propioceptiva (`observation.state`) — 8 Dimensiones:**
  $$\mathbf{s}_t = [X, Y, Z, q_x, q_y, q_z, q_w, \text{gripper}]$$
  Modela magnitudes físicas medidas en el instante $t$ (con garantía estricta de semiesfera superior $q_w \ge 0$).
* **Espacio de Acciones (`action`) — 9 Dimensiones $\times 16$ Pasos:**
  $$\mathbf{a}_{t:t+16} = [\text{Target}_X, \text{Target}_Y, \text{Target}_Z, \text{Target}_{qx}, \text{Target}_{qy}, \text{Target}_{qz}, \text{Target}_{qw}, \text{Target}_{\text{gripper}}, \text{done\_flag}]$$
  - *La 9ª dimensión (`done_flag`)* modela la intención neuronal de finalización de tarea.
* **Prohibición de Fuga Causal (*Causal Leakage*):**  
  La señal de parada reside estrictamente en el espacio de acciones y **nunca** en la observación propioceptiva, preservando la causalidad física.

---

## 3. Protocolo de Ejecución en Entorno Docker

Todo el desarrollo de Python, compilación ROS 2, recolección y entrenamiento se realiza dentro del contenedor Docker `visuomotor_ros2` bajo el usuario `hackbrian` (`hackbrian@robot_dev`) para preservar los permisos de archivos en los volúmenes montados:

```bash
# 1. Compilación del workspace
docker exec -u hackbrian -w /ros visuomotor_ros2 bash -c "colcon build --symlink-install"

# 2. Recolección autónoma de demostraciones (Genera automáticamente demonstrations_vX.zarr)
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_data_collection pickup_approximate.launch.py"

# 3. Entrenamiento de Diffusion Policy
docker exec -it -u hackbrian visuomotor_ros2 bash -c "python3 /ros/visuomotor_core/train.py --root /ros/visuomotor_core/datasets/demonstrations_v7_done_fixed.zarr --epoch 100 --batch_size 64 --out_dir /ros/visuomotor_core/outputs/train_500_dataV7_fixed"

# 4. Inferencia en vivo con Auto-Stop Neuronal
docker exec -it -u hackbrian visuomotor_ros2 bash -c "ros2 launch visuomotor_ros2 inference.launch.py"
```

> [!NOTE]
> La gestión del control de versiones Git se realiza **exclusivamente desde la máquina local (Host)** del usuario, nunca dentro del contenedor Docker.

---

## 4. Base de Conocimiento Persistente y Documentación

El repositorio cuenta con una memoria a largo plazo modular y aditiva en [`.agents/knowledge/`](./.agents/knowledge/):
* **[`01_diffusion_policy_architecture.md`](./.agents/knowledge/01_diffusion_policy_architecture.md)**: Arquitectura 1D U-Net, ResNet-18 GroupNorm, SpatialSoftmax, FiLM y DDIM.
* **[`02_tensors_and_autostop.md`](./.agents/knowledge/02_tensors_and_autostop.md)**: Formato de tensores 8D/9D, prevención de fuga causal y auto-stop neuronal.
* **[`03_data_collection_and_fsm.md`](./.agents/knowledge/03_data_collection_and_fsm.md)**: FSM experta, PID cartesiano con virtual setpoint, Halton QMC y freno Z dinámico.
* **[`04_dataloaders_and_datasets.md`](./.agents/knowledge/04_dataloaders_and_datasets.md)**: Estructura Zarr, solución Copy-Padding al corte de horizonte y 15% propioception dropout.
* **[`05_troubleshooting_and_environment.md`](./.agents/knowledge/05_troubleshooting_and_environment.md)**: CycloneDDS, ABI NumPy 2.x, filtro 1-Euro adaptativo y parámetros de ejecución.
* **[`AGENTS.md`](./AGENTS.md)**: Contrato maestro operacional para agentes autónomos de IA.
* **[`context.md`](./context.md)**: Repositorio histórico de decisiones arquitectónicas y estado del proyecto.
* **[`UPDATE_KNOWLEDGE.md`](./UPDATE_KNOWLEDGE.md)**: Protocolo universal automatizado de auditoría y sincronización (`/update-knowledge`).