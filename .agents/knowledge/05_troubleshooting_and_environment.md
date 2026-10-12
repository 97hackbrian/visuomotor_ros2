# Módulo 05: Entorno Docker, Filtro 1-Euro y Troubleshooting Crítico

Este documento compila el protocolo operacional del entorno de desarrollo, la arquitectura de los launch files, la formulación del suavizado cinemático (`pose_smoother.py`) y la bitácora exhaustiva de problemas resueltos, auditado línea por línea en `visuomotor_bringup`, `visuomotor_ros2` y `visuomotor_data_collection`.

---

## 1. Contrato Operativo del Entorno de Desarrollo

### A. Ejecución Obligatoria bajo el Usuario `hackbrian`
* **Directorio de Trabajo:** El repositorio en el host Linux está montado dentro del contenedor Docker `visuomotor_ros2` en la ruta `/ros`.
* **Riesgo de Hijacking de Permisos:** Si se ejecutan comandos como `root` (el usuario por defecto de `docker exec`), cualquier archivo o carpeta generada (`build/`, `install/`, `log/`, `.zarr`, `.pt`) queda con permisos exclusivos de `root:root`, bloqueando la edición, guardado y commit por parte del usuario en el host local.
* **Comando Obligatorio:** Todo comando dentro del contenedor debe invocar explícitamente `-u hackbrian`:
  ```bash
  docker exec -u hackbrian -w /ros visuomotor_ros2 bash -c "<comando>"
  ```

### B. Principio de Consentimiento Expreso para Terminal
* Los agentes de IA tienen **estrictamente prohibido ejecutar comandos proactivamente** en la terminal.
* Solo se ejecutan comandos cuando el usuario lo exprese de forma clara y directa en su mensaje de chat.

### C. Exclusividad de Git en el Host Local
* **NUNCA ejecutar comandos de Git (`git status`, `git add`, `git commit`, `git push`, `git config`, etc.) dentro de Docker.**
* Todo el control de versiones de Git lo opera el usuario directamente en su máquina anfitriona (host).

---

## 2. Arquitectura de Despliegue y Launch Files

### A. Inferencia en Vivo (`visuomotor_ros2/launch/inference.launch.py`)
* Carga los parámetros desde `visuomotor_ros2/config/policy_params.yaml`.
* Instancia el nodo ejecutor `policy_node` que orquesta la cámara `/rgb`, el búfer propioceptivo `/joint_states`, la sincronización sensorial y el ejecutor de acciones hacia `/target_frame_raw` y `/position_controller/commands`.

### B. Recolección Autónoma (`visuomotor_data_collection/launch/pickup_approximate.launch.py`)
* Acepta el argumento de lanzamiento `dataset_path` (por defecto `"AUTO"`).
* Si es `"AUTO"`, inspecciona `datasets/` mediante regex para crear incrementalmente `demonstrations_v{X+1}.zarr`.
* Carga la configuración completa de la FSM y cinemática desde `data_collection_params.yaml`.

### C. Stack Completo del Robot Lite6 (`visuomotor_bringup/launch/lite6_cartesian.launch.py`)
* Orquesta de forma integrada:
  1. `ros2_control_node`: Carga el plugin de hardware (`topic_based_ros2_control/TopicBasedSystem` en simulación o `uf_robot_hardware/UFRobotSystemHardware` en hardware real).
  2. `cartesian_motion_controller`: Controlador cinemático que recibe `/target_frame` y comanda las articulaciones.
  3. `joint_state_broadcaster`: Publica `/joint_states` a 100 Hz.
  4. `position_controller`: Maneja el actuador del gripper.
  5. `pose_smoother`: Filtro 1-Euro que transforma `/target_frame_raw` en `/target_frame`.
  6. `image_transport/republish`: Comprime la imagen de muñeca `/rgb` a `/rgb/compressed` para visualización remota.
  7. `foxglove_bridge`: Puente WebSocket en puerto 8765 (`0.0.0.0`) para telemetría visual.

---

## 3. Filtrado Adaptativo 1-Euro (`pose_smoother.py`)

Auditado en `visuomotor_bringup/scripts/pose_smoother.py`:

Para eliminar el temblor o micro-vibración (*jitter*) inherente a los modelos de difusión generativa sin introducir retraso (*lag*) perceptible durante movimientos rápidos, se implementa el filtro adaptativo **1-Euro** (Géry Casiez et al., 2012):

### A. Formulación Matemática
1. **Frecuencia de Corte Adaptativa:**
   La frecuencia de corte $f_c$ varía dinámicamente con la magnitud de la velocidad estimada $\hat{\dot{x}}$:
   $$f_c = f_{c,\min} + \beta \cdot \|\hat{\dot{x}}\|$$
   - A baja velocidad (brazo casi quieto): $f_c \approx f_{c,\min}$, filtrando agresivamente el ruido de alta frecuencia.
   - A alta velocidad (brazo en tránsito): $f_c$ se eleva en función de $\beta$, permitiendo el paso de la señal sin desfase temporal.
2. **Constante de Tiempo y Factor de Suavizado:**
   $$\tau = \frac{1}{2\pi f_c}, \quad \alpha = \frac{1}{1 + \frac{\tau}{dt}}$$
   $$\hat{x}_t = \alpha x_t + (1 - \alpha) \hat{x}_{t-1}$$

### B. Parámetros Calibrados (`pose_smoother_params.yaml`)
* `min_cutoff: 0.1`: Frecuencia mínima en reposo (elimina cualquier micro-oscilación al sostener el objeto).
* `beta: 0.00125`: Sensibilidad a la velocidad (garantiza seguimiento ágil en traslados rápidos).
* `d_cutoff: 0.95`: Frecuencia de corte para la estimación de la derivada de velocidad.

### C. Desenvoltorio Angular de Euler (*Unwrapping*) y Reset
* **Problema de Filtrado en Cuaterniones:** Los cuaterniones no forman un espacio euclidiano lineal. Filtrar sus componentes directamente con filtros pasabajos lineales altera la norma y distorsiona la rotación.
* **Solución Implementada:**
  1. Se convierte el cuaternión recibido a ángulos de Euler (Roll, Pitch, Yaw).
  2. Se aplica `np.unwrap` con respecto al estado previo para neutralizar discontinuidades en la frontera $[-\pi, +\pi]$.
  3. Se filtra cada ángulo con el filtro 1-Euro.
  4. Se reconvierte a cuaternión unitario $[q_x, q_y, q_z, q_w]$.
* **Timeout de Inactividad (0.5 s):** Si transcurren más de 0.5 s sin mensajes en `/target_frame_raw`, el filtro resetea sus memorias internas (`t_prev = None`) para evitar interpolar contra una pose obsoleta al reanudar la inferencia.

---

## 4. Bitácora Exhaustiva de Errores Críticos y Soluciones

### 1. Incompatibilidad de Middleware DDS (Ceguera Total de Tópicos con Isaac Sim)
* **Síntoma:** Isaac Sim está ejecutándose y publicando `/rgb` y `/tf`, pero los nodos de ROS 2 en el contenedor no reciben ningún mensaje (tópicos mudos).
* **Causa Raíz:** Isaac Sim viene precompilado con **CycloneDDS**, mientras que la instalación por defecto de ROS 2 Jazzy utiliza **FastDDS**. La disparidad de protocolo DDS impide el descubrimiento dinámico de nodos (*Discovery Phase*).
* **Solución Definitiva:** Forzar CycloneDDS y el ID de dominio en el entorno:
  ```bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export ROS_DOMAIN_ID=0
  ```

### 2. Conflicto de Versiones NumPy 2.x con `cv_bridge`
* **Síntoma:** `ImportError: numpy.core.multiarray failed to import` o fallos de segmentación al invocar `CvBridge().imgmsg_to_cv2()`.
* **Causa Raíz:** ROS 2 Jazzy `cv_bridge` está enlazado a la ABI C de NumPy 1.x. Instalar paquetes modernos como Zarr v3 o dependencias recientes actualiza silenciosamente a NumPy 2.x, rompiendo la compatibilidad binaria.
* **Solución Definitiva:** Mantener el entorno fijado a las versiones compatibles:
  ```bash
  pip install 'numpy<2' 'zarr<3' 'numcodecs<0.14'
  ```

### 3. Descarte Silencioso de Poses por Falta de `header.frame_id`
* **Síntoma:** El nodo de inferencia publica poses continuas en `/target_frame_raw`, pero el brazo Lite6 no ejecuta ningún movimiento en el simulador.
* **Causa Raíz:** El nodo `cartesian_motion_controller` de ROS 2 Control verifica la cabecera del mensaje. Si `header.frame_id` está vacío (`""`), el mensaje se descarta silenciosamente en el hilo de tiempo real por carecer de sistema de referencia.
* **Solución Definitiva:** Asignar siempre explícitamente el marco base:
  ```python
  msg.header.frame_id = 'link_base'
  msg.header.stamp = self.get_clock().now().to_msg()
  ```

### 4. Rechazo de Mensajes del Gripper por Discrepancia Dimensional
* **Síntoma:** El brazo alcanza la posición del objeto pero las mordazas del gripper no se mueven ni cierran.
* **Causa Raíz:** El controlador de posición de articulaciones de Isaac Sim (`/position_controller/commands`) está configurado para un único canal escalar `[pos]`. Si el ejecutor enviaba un vector con dos valores `[pos, pos]` (asumiendo dos dedos independientes), el controlador descartaba el paquete por discrepancia de tamaño.
* **Solución Definitiva:** Enviar estrictamente un array de longitud 1:
  ```python
  gripper_msg = Float64MultiArray()
  gripper_msg.data = [float(gripper_val)]
  self.gripper_pub.publish(gripper_msg)
  ```
