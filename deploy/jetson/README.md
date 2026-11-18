# Guía de Despliegue de Inferencia Visuomotora en NVIDIA Jetson Orin Nano (8 GB)

Este directorio contiene todo lo necesario para compilar y ejecutar el nodo de inferencia reactiva (`policy_node.py`) en una **NVIDIA Jetson Orin Nano (8 GB)** utilizando exclusivamente **Docker** y **Docker Compose**.

---

## 1. Arquitectura y Compatibilidad de Versiones

### Estado de tu Jetson (Detectado en `fastfetch`)
* **Host OS:** Ubuntu 20.04.6 LTS (`aarch64`)
* **Kernel:** `Linux 5.10.216-tegra` $\implies$ **JetPack 5.1.4 / 5.1.5 (L4T R35.6.x)**
* **Memoria:** 7.28 GiB LPDDR5 unificada

### Selección de Imagen Base Docker
En JetPack 5.x, el runtime de NVIDIA (`nvidia-container-cli`) monta los drivers CUDA 11.4 del host. Para garantizar compatibilidad binaria absoluta con la GPU Tegra sin errores de `libcuda.so` ni dependencias rotas de glibc:
* **JetPack 5.1.x (Por defecto):** Utiliza `dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1` (Ubuntu 20.04 Focal en contenedor, ROS 2 Humble Desktop compilado desde fuente en `/opt/ros/humble/install`, Python 3.8, PyTorch 2.0.0+nv23.05 con CUDA 11.4). La imagen solo añade dependencias de Python (`torchvision==0.15.1`, `diffusers==0.30.3`, etc.) y no modifica el sistema base. `visuomotor_core` incluye `from __future__ import annotations` para ser compatible con Python 3.8.
* **JetPack 6.x (Si en el futuro actualizas la Jetson):** Puedes cambiar `BASE_IMAGE` a `dustynv/ros:jazzy-pytorch-l4t-r36.2.0` en `docker-compose.yml`.

> [!NOTE]
> **Compatibilidad ROS 2 sobre DDS:**
> El nodo de inferencia corriendo en ROS 2 Humble dentro del contenedor de la Jetson se comunica de forma transparente sobre la red DDS (`rmw_cyclonedds_cpp`, Domain ID 42) con cualquier nodo externo (cámaras, robots o computadoras corriendo ROS 2 Humble, Iron o Jazzy).

---

## 2. Preparación del Host (Se ejecuta una sola vez)

Antes de levantar Docker, optimiza la memoria y los relojes de la Jetson:

```bash
cd deploy/jetson/scripts
chmod +x setup_host.sh
./setup_host.sh
```

Este script automatiza:
1. Fijar el modo de energía a **15W Máximo Rendimiento** (`nvpmodel -m 0`).
2. Bloquear relojes de CPU, GPU y memoria a máxima frecuencia (`jetson_clocks`).
3. Activar una memoria **SWAP de 8 GB** para proteger la memoria unificada contra el *OOM Killer*.
4. Configurar `"default-runtime": "nvidia"` en `/etc/docker/daemon.json`.
5. Reducir la latencia de puertos USB serie a **1 ms** (`/sys/bus/usb-serial/devices/ttyUSB*/latency_timer`).

---

## 3. Despliegue en 3 Pasos

### Paso 1: Copiar los Pesos del Modelo
Copia los pesos y la configuración del modelo entrenado a la carpeta `deploy/jetson/model/`:

```bash
# Ejemplo usando el modelo maestro train_500_dataV7_fixed:
cp visuomotor_core/outputs/train_500_dataV7_fixed/model.safetensors deploy/jetson/model/
cp visuomotor_core/outputs/train_500_dataV7_fixed/config.json deploy/jetson/model/
```

### Paso 2: Ajustar Parámetros de la Cámara (Opcional)
Edita `deploy/jetson/config/jetson_policy_params.yaml` según los tópicos de tu cámara física:
```yaml
camera_topic: "/camera/color/image_raw"  # Tópico de tu cámara RealSense o USB
joint_states_topic: "/joint_states"
inference_hz: 20.0
num_inference_steps: 10
```

### Paso 3: Construir y Levantar el Contenedor
Desde la carpeta `deploy/jetson/`:

```bash
cd deploy/jetson
docker compose build
docker compose up -d
```

`docker compose up -d` deja el contenedor **en espera**: NO inicia la inferencia (el brazo no se mueve al encender la Jetson ni al reiniciar el contenedor). La inferencia se lanza manualmente cuando tú decidas:

```bash
# Opción A: lanzarla directamente
docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh \
  ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model

# Opción B: entrar con ROS ya cargado y lanzarla desde dentro (Ctrl+C la detiene sin tumbar el contenedor)
docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh bash
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model
```

Para inferencia automática al arrancar (modo producción), reemplazar `command: sleep infinity` en `docker-compose.yml` por el `ros2 launch` anterior.

Para ver los logs del contenedor:
```bash
docker compose logs -f
```

Para detener el servicio:
```bash
docker compose down
```

---

## 3b. Manipulador real xArm5 (UFACTORY)

La imagen incluye el driver oficial `xarm_ros2` (rama `humble`, fijado por commit) y el bringup
`visuomotor_bringup/launch/xarm5_real.launch.py`. Toda la configuración (IP, gripper, límites, velocidades,
home) está en `visuomotor_bringup/config/xarm5/xarm5_real_params.yaml`, montado en el contenedor (se edita
en el host sin reconstruir).

```
policy_node --target_frame_raw--> pose_smoother --target_frame--> xarm5_bridge --> /xarm/set_servo_cartesian
policy_node --/position_controller/commands--> xarm5_bridge --> gripper xArm / ventosa
/xarm/robot_states --> xarm5_bridge --> TF link_base->xarm_tcp + /visuomotor/gripper_joint_states --> policy_node
```

**Seguridad del puente (`xarm5_bridge`):** arranca **desarmado** y con `dry_run: true`; caja de trabajo y radio
XY, 5 cm/s y 0.5 rad/s por defecto, rechazo de objetivos a más de 25 cm del TCP, *watchdog* de objetivo,
y desarme automático si el controlador reporta error, se pierde `robot_states` o fallan 5 envíos seguidos.

**Orientación con 5 GDL:** el xArm5 no puede seguir una orientación 6D arbitraria. Por defecto
(`orientation_mode: top_down_yaw`) la garra se mantiene vertical y solo se usa el giro (yaw) que pide la política.

### Red
El controlador está en `192.168.1.228` (valor histórico). La Jetson necesita una interfaz **cableada** en esa
subred, por ejemplo `192.168.1.10/24` (sin puerta de enlace, para no romper el Wi-Fi).

### Puesta en marcha por etapas (dentro del contenedor: `docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh bash`)

```bash
# 0) Sin hardware: robot simulado (valida la cadena completa con la política)
ros2 launch visuomotor_bringup xarm5_real.launch.py simulate_robot_state:=true

# 1) Conexión (brazo energizado, paro de emergencia liberado): driver + puente en DRY-RUN. No mueve nada.
ping -c 3 192.168.1.228
ros2 launch visuomotor_bringup xarm5_real.launch.py
ros2 topic echo /xarm5_bridge/status          # err=0, robot_state_age_s pequeño
ros2 run tf2_ros tf2_echo link_base xarm_tcp  # debe coincidir con la pose del panel UFACTORY

# 2) Home articular (desarmado, modo posición)
ros2 service call /xarm5_bridge/go_home std_srvs/srv/Trigger

# 3) Movimiento real: relanzar con dry_run:=false, armar y luego lanzar la política
ros2 launch visuomotor_bringup xarm5_real.launch.py dry_run:=false
ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model

# Detener en cualquier momento (además del paro de emergencia físico)
ros2 service call /xarm5_bridge/disarm std_srvs/srv/Trigger
```

En dry-run, `ros2 topic echo /xarm5_bridge/command_preview` muestra la pose que se enviaría al brazo.
`/xarm5_bridge/status` incluye la pose medida del TCP: tras `go_home`, úsala para ajustar `home_x/y/z` de
`jetson_policy_params.yaml` (la fase RESET de la política envía esa pose y el puente rechaza objetivos a más de
`max_target_distance_m` del TCP). En la primera conexión, verificar también `ros2 topic hz /xarm/robot_states`.

### Comprobaciones sin hardware (driver falso)
```bash
python3 tools/xarm5/fake_xarm_driver.py &
ros2 launch visuomotor_bringup xarm5_real.launch.py start_driver:=false start_smoother:=false dry_run:=false &
python3 tools/xarm5/check_xarm5_bridge.py      # 17 comprobaciones; código de salida 0 = OK
```

---

## 4. Benchmark de Latencia y VRAM en Borde

Para verificar cuántos milisegundos toma la inferencia y cuánta memoria consume en la Jetson Orin Nano antes de conectar los motores físicos del robot:

```bash
# Abrir una terminal interactiva dentro del contenedor en ejecución
docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh bash

# Ejecutar el benchmark automatizado (evalúa FP32 vs FP16)
python3 /ros_ws/src/visuomotor_ros2/deploy/jetson/scripts/benchmark_jetson.py --model_path /ros_ws/model
```

### Resultados Esperados en Jetson Orin Nano (15W):
* **DDIM (10 pasos, FP16 con Tensor Cores):** $\approx 25 - 35\text{ ms}$ ($\mathbf{\approx 30\text{ Hz}}$, superando con holgura los 20 Hz requeridos).
* **Consumo VRAM pico:** $\approx 1.8\text{ GB}$ (dejando más de 5 GB libres en la Jetson).

---

## 5. Edición de Código en Vivo con VS Code (Modo Desarrollo Dual)

Si en algún momento deseas modificar `policy_node.py` o los módulos de `visuomotor_core` usando **VS Code** sin tener que reconstruir la imagen Docker (`docker compose build`), dispones de dos métodos directos:

### Método 1: Montaje en Vivo con Docker Compose Dev (Recomendado)
Levanta el contenedor utilizando el archivo de extensión `docker-compose.dev.yml`:

```bash
cd deploy/jetson
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

* **Cómo opera:** Monta en tiempo real las carpetas de código de tu host (`visuomotor_ros2/visuomotor_ros2` y `visuomotor_core/visuomotor_core`) directamente sobre el workspace del contenedor.
* **Flujo de trabajo:** Abres el repositorio en VS Code en la Jetson (o vía Remote SSH desde tu laptop), editas cualquier archivo `.py` y guardas (`Ctrl+S`). El cambio se ejecuta **de inmediato** en el robot sin recompilar.

### Método 2: VS Code "Attach to Running Container"
Con el contenedor estándar de Producción corriendo (`docker compose up -d`):
1. En VS Code, instala la extensión oficial **Dev Containers** (o **Docker**).
2. Presiona `F1` (o `Ctrl+Shift+P`) y selecciona:
   `Dev Containers: Attach to Running Container...` $\to$ elige `visuomotor_orin_inference`.
3. Abre la carpeta `/ros_ws/src/`.
4. Podrás editar código con resaltado de sintaxis, autocompletado y terminal interactiva con acceso a GPU y ROS 2 directamente dentro del contenedor.

