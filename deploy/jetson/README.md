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
| **HOME3** | [-1.04, -0.11, -2.54, 2.66, -1.04] (°) | Pose de trabajo del usuario (2026-11-20), efector alineado por FK: vertical (J4 = -(J2+J3)) y yaw 0 (J5 = J1) | (0.222, -0.004, 0.116) m, hacia abajo |

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

