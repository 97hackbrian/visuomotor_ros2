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
El controlador está en `192.168.1.228` (verificado). La Jetson usa `eth0` con `192.168.1.5/24` para el robot y
`wlan0` para la red del laboratorio. **DDS (ROS 2) sale por `wlan0`** (`config/cyclonedds.xml`, variable `DDS_IFACE`):
el driver habla con el controlador por TCP, no por DDS. Si la red bloquea multicast, definir `DDS_PEER=<IP del otro
equipo>` antes de `docker compose up -d`.

**Dominio DDS del robot real: `ROS_DOMAIN_ID=42`** con `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (compose; todo el
contenedor lo hereda). **Precaución:** la simulación del otro equipo también usa 42: mientras el brazo real esté
conectado, no correr simulación ni políticas en el dominio 42 de la misma red (o usarlas con `ROS_LOCALHOST_ONLY=1`),
porque sus objetivos (`/target_frame_raw`) llegarían al puente del brazo real.

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
ros2 service call /xarm5_bridge/go_home1 std_srvs/srv/Trigger   # HOME1 = cero UFACTORY

# 3) Movimiento real: relanzar con dry_run:=false, armar y luego lanzar la política
ros2 launch visuomotor_bringup xarm5_real.launch.py dry_run:=false
ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model

# Detener en cualquier momento (además del paro de emergencia físico)
ros2 service call /xarm5_bridge/disarm std_srvs/srv/Trigger
```

En dry-run, `ros2 topic echo /xarm5_bridge/command_preview` muestra la pose que se enviaría al brazo.
`/xarm5_bridge/status` incluye la pose medida del TCP: tras `go_home1`/`go_home2`, úsala para ajustar `home_x/y/z` de
`jetson_policy_params.yaml` (la fase RESET de la política envía esa pose y el puente rechaza objetivos a más de
`max_target_distance_m` del TCP). En la primera conexión, verificar también `ros2 topic hz /xarm/robot_states`.

### Volumen de trabajo: caja + alcance del brazo
La caja (`workspace_*`, z hasta 0.75 m) no basta: la altura alcanzable baja con el radio. El puente además limita la
distancia hombro (J2) – muñeca (J4) a `workspace_max_reach_m` (95% del codo recto, singularidad/C24) y a
`workspace_min_reach_m` (brazo plegado junto a la base). La muñeca se calcula desde la pose de la brida y la
inclinación, así que vale también con la garra inclinada. Un objetivo fuera de alcance se recorta hacia el hombro
(mismo acimut). Altura máxima de la brida con la garra vertical (FK URDF):

| x (m) | 0.15 | 0.25 | 0.35 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 |
|---|---|---|---|---|---|---|---|---|
| z máx (m) | 0.75 | 0.75 | 0.71 | 0.65 | 0.61 | 0.55 | 0.48 | 0.37 |

Validado por FK: todo el volumen es alcanzable con la garra vertical y con inclinación ±0.5 rad. Con |pitch| ≈ 1 rad
quedan bordes (muy alto cerca de la base, o muy bajo junto a ella) que detiene el guard de límites articulares.

### Teleoperación desde RViz2 en otro equipo
En la Jetson, `xarm5_real.launch.py` lanza además `robot_state_publisher` (URDF del xArm5 desde `xarm_description`)
y `xarm5_teleop_marker` (marcador interactivo `/xarm5_teleop`). En el otro equipo (mismo `ROS_DOMAIN_ID=42`, RMW
CycloneDDS y `xarm_description` instalado para las mallas):
```bash
ros2 launch visuomotor_bringup xarm5_rviz.launch.py
```
Clic derecho sobre la esfera del marcador: *Arm* (sincroniza antes) → *Follow marker* → arrastrar flechas
(X/Y/Z), el anillo **azul** (giro de la herramienta, J5) o el anillo **verde** (inclinación). *Stop following* /
*Disarm* en cualquier momento. El marcador no publica nada si no está en *Follow* con el puente ARMADO, y el puente
sigue aplicando caja, velocidad, distancia máxima y frenos.

**Anillo verde (inclinación, `orientation_mode: arm_plane`).** El xArm5 (5 GDL) solo puede inclinar la garra dentro del
plano vertical que contiene al brazo (acimut `atan2(y, x)`). El anillo verde gira alrededor del eje perpendicular a ese
plano y se re-orienta al soltar el ratón tras mover el marcador. Al soltar, la orientación del marcador se ajusta a la
alcanzable (inclinación acotada a `arm_plane_pitch_min/max`, ±1.48 rad ≈ ±85°; en ±90° el rpy del controlador queda en bloqueo de cardán), que es exactamente lo que ejecutará el puente;
la descripción del marcador muestra `tilt` en grados. Con `orientation_mode: top_down_yaw` no hay anillo verde y la
garra se mantiene vertical (el primer seguimiento la endereza si no lo está).
No usar el marcador y la política a la vez (ambos publican en `/target_frame`).

### Frenos
Al armar y antes de `~/go_home1`/`~/go_home2`, el puente habilita motores y exige que **todos** los ejes reporten freno
liberado (bit 1 en `mt_brake`) y motor habilitado (bit 1 en `mt_able`) en `brake_check_timeout_s`; si no, no envía
movimiento e indica la articulación (p. ej. `freno APLICADO en J3`). Armado, si un eje pierde un bit, se desarma.
`ros2 service call /xarm5_bridge/check_brakes std_srvs/srv/Trigger` informa sin actuar.
**Freno intermitente (J3 del brazo real):** si un eje no libera, el puente re-habilita SOLO ese eje hasta
`brake_release_retries` veces (3); el ciclo bloquear/desbloquear de ese eje (`brake_retry_disable_cycle`) está
desactivado hasta validarlo. Solo cuentan reportes de estado posteriores a la habilitación (`brake_state_settle_s`). No deshabilitar motores
(`motion_enable` data=0) con el brazo en el aire: hizo caer el brazo ~5 cm.

### Límites articulares y desenrollado de J1
J1 y J5 del xArm5 tienen rango ±2π (±360°). El 2026-11-18 J1 estaba en +6.2822/+6.2832 rad (en el tope): al teleoperar,
el controlador cortó con **C23 (Joint Angle Exceed Limit)**. Ahora el puente **no arma** si una junta está a menos de
`joint_limit_margin_rad` de su límite, **se detiene antes del C23** si una junta se acerca durante el movimiento, y
`go_home1`/`go_home2` se rechazan si alguna junta debe moverse más de `max_home_joint_delta_rad` (por junta; J1 0.5 rad: evita giros de base
inesperados). Para devolver J1 a [-π, π] (misma postura física) con el puente lanzado con `dry_run:=false`:
```bash
ros2 service call /xarm5_bridge/unwind_j1 std_srvs/srv/Trigger   # gira SOLO la base -360° a 0.3 rad/s (~21 s)
```
Solo a mano (no está en el menú de RViz2), desarmado, con la zona alrededor del brazo despejada y alguien en la parada
de emergencia. Con C2 activo (seta pulsada) se cancela: liberar la seta y reintentar.

### Poner y quitar frenos (menú de RViz2 o servicios)
* **Quitar frenos** (`~/brakes_off`): habilita motores en modo posición, el brazo queda sostenido (no arma ni mueve);
  pasa por la comprobación de frenos con reintento de J3.
* **Poner frenos** (`~/brakes_on`, doble clic en RViz2): deshabilita motores. Solo desarmado y con el brazo quieto
  `brakes_on_min_idle_s` (1 s): la caída de ~5 cm ocurrió al deshabilitar justo tras un movimiento; en reposo se midió
  0 mm. Informa el desplazamiento del TCP al frenar.
* `unwind_j1` solo por terminal (se retiró del menú de RViz2).

### Posiciones de reposo HOME1 / HOME2 (fabricante)
Modelo: **UFACTORY xArm 5, código XF1305** ("XF13" en la caja; el controlador reporta `DETAIL: 5,5,XF1305`), 5 GDL,
3 kg, 700 mm. Menú de RViz2 (doble clic, solo desarmado) o servicios `~/go_home1` y `~/go_home2`:
| | Articulaciones (°) | Fuente | Brida (FK URDF) |
|---|---|---|---|
| **HOME1** | [0, 0, 0, 0, 0] | "Zero Position"; manual V2.3.0 §1.5: mover a cero antes de embalar/transportar | (0.207, 0, 0.112) m, hacia abajo |
| **HOME2** | [0, 0, 0, -90, 0] | "Initial Position" por defecto del xArm 5 (xArm Studio V1.0.0); "a relatively safe position" (V2.3.0) | (0.228, 0, 0.285) m, horizontal |
| **HOME3** | [-1.96, -41.99, -92.46, 109.92, -0.47] | Pose de trabajo del usuario (redefinida 2026-11-20), leída del brazo real y centrada en y = 0 (J1/J5 -2.01°); home de la política | TCP previsto (0.150, 0.000, 0.750) m, garra inclinada ~-24.6° |

Todos los HOME limpian un error C2 pendiente (seta recién liberada) antes de moverse.

Aviso del fabricante (V2.3.0 §3.1.3): con gripper montado, en cero la herramienta sobrepasa la superficie de montaje.

### Comprobaciones sin hardware (driver falso)
```bash
python3 tools/xarm5/fake_xarm_driver.py &
ros2 launch visuomotor_bringup xarm5_real.launch.py start_driver:=false start_smoother:=false dry_run:=false &
python3 tools/xarm5/check_xarm5_bridge.py      # 35 comprobaciones; código de salida 0 = OK
python3 tools/xarm5/check_xarm5_teleop.py      # 16 comprobaciones (incluye anillo verde 10b-10d)
python3 -m pytest tests/test_xarm5_interface.py tests/test_convert_vio_dataset.py -q   # lógica pura (50)
```

### Demostraciones reales → dataset entrenable para el xArm5
Las demostraciones a mano (`vo_collector`, RTAB-Map VIO) guardan la pose de la **cámara** en el marco `odom`; la
política en el robot observa el **TCP** en `link_base` menos `home_x/y/z`. Antes de entrenar, convertir (copia nueva,
el origen no se toca):
```bash
python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_vN.zarr --report-only   # auditoría
python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_vN.zarr [--drop-flagged]
# -> datasets/demonstrations_real_vN_xarm5.zarr + _conversion_report.json   (entrenar con ESTE dataset)
```
Fuentes (sin duplicar valores): home = `config/jetson_policy_params.yaml` (HOME3), modo/límites/volumen/velocidades =
`xarm5_real_params.yaml`, extrínseca cámara→TCP = `xarm5_dataset_alignment.yaml` (**medirla**; con
`extrinsic_calibrated: false` la herramienta se niega a convertir). Marca episodios con orientación inalcanzable
(> 0.15 rad), TCP fuera del volumen o más rápidos que el puente (5 cm/s, 0.5 rad/s).
Reglas para grabar: empezar cada episodio con la cámara en la pose equivalente a HOME3 (misma disposición de la escena),
garra vertical o inclinada solo en el plano del brazo, moverse despacio (≤ 5 cm/s, o subir el límite del puente y
reconvertir), y usar el MISMO soporte de cámara en la mano y en la muñeca del robot.

---

### Modelo real v7 (Fase 22): aproximación a las tijeras desde HOME3

Modelo en `deploy/jetson/model/` (checkpoint época 200 de `train_real_v7x5v2_relative_ee_e300`; ficha:
`MODEL_CARD.md`, figura: `final_model_report.png`). Entrenado con `demonstrations_real_v7_xarm5_v2` (acciones
`relative_ee`, cuaternión `home`). **Verificado offline y en ROS 2 con el puente simulado; aún no en el brazo físico.**

Requisitos previos:
0. **Pesos (no están en git: 109 MB > límite de GitHub).** Copiar desde el PC los pesos y la referencia visual a la
   misma carpeta del repo en la Jetson; `config.json`, `training_meta.json` y `MODEL_CARD.md` llegan con `git pull`:
   ```bash
   scp deploy/jetson/model/model.safetensors deploy/jetson/model/home3_reference_view.png \
       <usuario>@<ip_jetson>:<repo>/deploy/jetson/model/
   sha256sum deploy/jetson/model/model.safetensors   # en la Jetson: debe dar
   # 5c5c7ed532e9e3fdac1873519783e3c93f814493864888a935a743a3243545fa
   ```
   La referencia se regenera con `tools/xarm5/export_home3_reference.py`.
1. **Código nuevo en la Jetson:** reconstruir la imagen (`docker compose build`) o usar `docker-compose.dev.yml`
   (monta `visuomotor_core` y `visuomotor_ros2`): los modelos `relative_ee` necesitan el `DiffusionPolicy` actual, y
   el puente usa los límites articulares del Hardware Manual v2.7.1 (XF1305) desde el 2026-11-22. El código de la
   Jetson (Python 3.8, torch 2.0) pasó un chequeo estático de compatibilidad; la prueba real es la etapa A en la Jetson.
1b. **Driver de la cámara (2026-11-23).** `Dockerfile.jetson` no trae el driver de la RealSense. La imagen base
   dustynv tiene ROS compilado desde fuente, así que no se puede usar `apt install ros-humble-realsense2-camera`.
   El driver se compila una vez en una imagen base aparte:
   - librealsense 2.59.1 y realsense-ros 4.59.1, las mismas versiones que en la recolección;
   - backend RSUSB, sin tocar el kernel ni el host.
   ```bash
   docker tag visuomotor_jetson:latest visuomotor_jetson:pre-realsense-$(date +%Y%m%d)   # respaldo
   docker build -f deploy/jetson/Dockerfile.realsense_base -t visuomotor_jetson_base:realsense-2.59.1 deploy/jetson
   cd deploy/jetson
   docker compose -f docker-compose.yml -f docker-compose.realsense.yml build
   docker compose -f docker-compose.yml -f docker-compose.realsense.yml up -d
   # cámara (dentro del contenedor): mismos parámetros que la recolección
   docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh ros2 launch visuomotor_bringup d435i_camera.launch.py
   ```
   Para volver atrás: `docker compose up -d`, que usa la imagen sin driver.
2. **Cámara:** la misma D435i a 640x480 color, montada en la brida con el eje óptico a lo largo de la herramienta y la
   **imagen derecha en HOME3** (arriba de la imagen = lejos de la base, como se sostuvo al grabar). Verificar que
   `camera_topic` de `jetson_policy_params.yaml` existe (`ros2 topic list | grep color`).
   **Comprobación visual obligatoria antes de armar:** con el brazo en HOME3, abrir
   `ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw` y compararla con
   `deploy/jetson/model/home3_reference_view.png`. Deben coincidir la orientación y el encuadre de la mesa, y no deben
   verse los dedos de la garra (en las demos no aparecen). Si no coincide, la política verá imágenes fuera de
   distribución y fallará de forma sistemática: corregir el montaje antes de seguir.
3. **Escena:** tijeras de podar sobre la misma mesa (mantel blanco y los dos ArUco), dentro de |y| <= 0.30 m (caja de
   seguridad configurada del puente; el brazo alcanza ±0.47 m a esa profundidad) y entre ~0.35 y 0.60 m delante de la
   base, como en los datos.
4. **Latencia en la Orin (medida el 2026-11-23, sección 4):** la inferencia completa tarda 69 ms con 5 pasos DDIM y
   CUDA Graph (`num_inference_steps: 5`, `use_cuda_graph: true`); con 10 pasos sin grafo tardaba 205 ms.
   `policy_node` infiere cada 3 ticks, y el timer de 20 Hz conserva la fase. Resultado: 20,0 Hz de comandos, con un
   hueco de unos 120 ms por inferencia que absorbe el filtro de 1,5 Hz.
   Verificado en la Jetson sin brazo (2026-11-23):
   - lazo cerrado con imágenes de validación (`check_policy_closed_loop_sim.py`, episodios 45 y 7): coseno 0,985–0,987,
     parada automática, 0 rechazos y 20,07 Hz;
   - etapa A con la cámara en vivo: 20,01 Hz y 80 % de CPU libre.

Puesta en marcha (cada etapa solo si la anterior se comporta como se describe):
```bash
# 0) Contenedor con el driver de la cámara (desde deploy/jetson) y la cámara en una terminal propia
docker compose -f docker-compose.yml -f docker-compose.realsense.yml up -d
docker exec -it visuomotor_orin_inference /ros_ws/entrypoint.sh ros2 launch visuomotor_bringup d435i_camera.launch.py
# A) Sin brazo: robot simulado + política (con la cámara montada y en HOME3: debe bajar hacia delante y auto-detenerse)
ros2 launch visuomotor_bringup xarm5_real.launch.py simulate_robot_state:=true
ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model

# B) Brazo real en DRY-RUN: ir a HOME3 y observar los comandos sin mover nada
ros2 launch visuomotor_bringup xarm5_real.launch.py
ros2 service call /xarm5_bridge/go_home3 std_srvs/srv/Trigger
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model
ros2 topic echo /xarm5_bridge/command_preview      # avance hacia delante/abajo, hacia las tijeras
ros2 topic echo /policy/metrics/done_flag          # sube a ~1 solo cerca del objeto

# C) Movimiento real a 5 cm/s (límite del puente); paro de emergencia a mano
ros2 launch visuomotor_bringup xarm5_real.launch.py dry_run:=false
ros2 service call /xarm5_bridge/go_home3 std_srvs/srv/Trigger
ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger
ros2 launch visuomotor_ros2 inference.launch.py checkpoint_path:=/ros_ws/model
```
Parámetros relevantes ya fijados en `config/jetson_policy_params.yaml`: `quaternion_hemisphere: auto`,
`image_resize_method: cv2_area`, `n_action_steps: 3` + ensemble temporal, auto-stop `0.60` durante `5` pasos
(calibrado offline: 100% de paradas, 0 prematuras, ~8 mm de la meta) y `pose_smoother` a 1.5 Hz
(`config/xarm5/xarm5_pose_smoother_params.yaml`). Si el brazo va claramente rezagado frente a la política, subir
`max_linear_speed_m_s` a 0.10 m/s en `xarm5_real_params.yaml` tras validar la etapa C (las demostraciones tienen media
0.05 m/s y p99 0.14 m/s). Registrar por intento: éxito (se detiene sobre las tijeras), distancia final y tiempo.

**Protocolo de ensayo (etapa C).**
- **Ensayos:** 20 intentos en una grilla de 6 posiciones de las tijeras (izquierda, centro y derecha × cerca y lejos),
  rotando el orden.
- **Éxito:** desde HOME3, el brazo se detiene solo (auto-stop) a ≤ 5 cm del punto de aproximación, sin intervención y
  sin rechazos del puente.
- **Registrar:** distancia final, tiempo, paradas prematuras, recortes del puente y un rosbag de `/policy/metrics`.
- **Estimación previa:** ~40 % (30-55 %) si la vista en HOME3 coincide con la referencia. Con 20 intentos, el
  intervalo de confianza al 95 % es de unos ±20 puntos (Wilson: 10/20 → 30-70 %); con 10, de unos ±30.
- **Resultados del 2026-11-23:** 6/7 éxitos (86 %, IC 49–97 %).
  - 7/7 paradas automáticas, 0 errores del puente.
  - Falla con las tijeras giradas ~75° en yaw.
  - Detalle en `experiments/xarm5_real_trials/2026-11-23/summary.md`.
- **Orquestador desde el PC:** `tools/xarm5/real_trials/run_trial.sh N "posición"`.
  - Hace chequeo previo de solo lectura (si falla, aborta sin armar), suena un pitido al empezar y otro al terminar,
    y corre un vigilante que desarma ante z < 0,15 m, errores o 45 s.
  - Desarma sin poner frenos y verifica que no quede ninguna política publicando.
  - Las métricas las calcula `tools/xarm5/real_trial_metrics.py`.
- **Diagnóstico por síntoma:**
  - Dirección equivocada desde el arranque: montaje o vista.
  - Arranca bien y se desvía: error que se acumula en lazo cerrado; hacen falta más datos o correcciones grabadas
    con la cámara del robot.
  - No se detiene, o se detiene antes: umbral y paciencia del auto-stop.

## 4. Benchmark de Latencia y VRAM en Borde

Para verificar cuántos milisegundos toma la inferencia y cuánta memoria consume en la Jetson Orin Nano antes de conectar los motores físicos del robot:

```bash
# El script no está dentro de la imagen: correrlo en un contenedor temporal con el repo montado (desde la raíz del repo)
docker run --rm -v "$PWD":/repo:ro -v "$PWD"/deploy/jetson/model:/ros_ws/model:ro visuomotor_jetson:realsense \
    python3 /repo/deploy/jetson/scripts/benchmark_jetson.py --model_path /ros_ws/model --steps 5
# Compara la ejecución normal con el CUDA Graph usando predict_chunk, la misma llamada que hace policy_node
```

### Resultados medidos en la Jetson Orin Nano (15 W, `jetson_clocks`, 2026-11-23; `predict_chunk` completo)
| Pasos DDIM | Normal | CUDA Graph |
|---|---|---|
| 10 | 205 ms | 119 ms |
| **5 (configuración actual)** | 111 ms | **69 ms** |

* La U-Net cuesta unos 18 ms por paso en modo normal y 9 ms en grafo: está limitada por el lanzamiento de kernels
  desde la CPU. FP16 (autocast) es MÁS lento y no se usa.
* VRAM pico: 150–265 MB. Las cifras anteriores (25–35 ms con FP16) nunca se habían medido.
* Con 5 pasos la calidad no cambia: en validación (`replay_policy_offline.py`, 8 episodios), el error del comando es
  11,5 mm frente a 11,6 mm con 10 pasos.
* Después de un reinicio hay que volver a fijar los relojes con `sudo jetson_clocks` (también lo hace
  `scripts/setup_host.sh`).

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

