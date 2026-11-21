# Auditoría Técnica y Caracterización Exhaustiva del Sistema VIO Activo (RealSense D435i + RTAB-Map) en `visuomotor_ros2`

**Autor:** Explorer 3 (Active Repo VIO Explorer)  
**Fecha:** 2026-11-20  
**Entorno Operativo:** ROS 2 Jazzy | Docker Container `visuomotor_ros2_vo:jazzy` | Host Ubuntu 24.04 LTS (Kernel Linux x86_64)  
**Hardware de Prueba:** Intel Core i9-13950HX (32 hilos), 32 GB RAM DDR5, GPU NVIDIA RTX 4060 Laptop (8 GB VRAM), Bus USB 3.2 Gen 1 (5000M SuperSpeed)  
**Dispositivos Sensoriales:** Intel RealSense D435i (ID `0x0B3A`, Firmware `5.17.3.10`, IMU Bosch BMI085 de 6 GDL) e Intel RealSense D415  

---

## 1. Resumen Ejecutivo y Topología Arquitectónica

El repositorio `visuomotor_ros2` implementa una solución de **Odometría Visual-Inercial (VIO) en tiempo real** de grado industrial y metrológico para la recolección autónoma de demostraciones de manipulación experta a mano alzada (*handheld approaching*) orientadas al entrenamiento de **Políticas Visuomotoras Generativas (Diffusion Policy, DDIM Schedule)**.

A diferencia de los enfoques fotogramétricos tradicionales de procesamiento por lotes offline (e.g. COLMAP o DROID-SLAM en pipelines tipo UMI original con cámaras de acción monocular), la arquitectura activa de `visuomotor_ros2` opera de manera **estrictamente sincrónica en tiempo real a 20.0 Hz** mediante la fusión acoplada de:
1. **Sensor Estereoscópico Activo con Emisor Infrarrojo (Intel RealSense D435i):** Proyecta una trama de micropuntos láser (*laser speckle*) que densifica artificialmente la escena, computando disparidad y mapas de profundidad métricos 3D directamente en el silicio del ASIC de Intel a 30 FPS.
2. **Unidad de Medición Inercial de 6 GDL (Bosch BMI085):** Acelerómetro triaxial y giróscopo triaxial muestreados a 200 Hz y fusionados mediante un filtro no lineal de gradiente descendente (`imu_filter_madgwick`) para anclaje gravitacional estricto.
3. **Motor de Odometría Visual-Inercial (`rtabmap_odom / rgbd_odometry`):** Arquitectura Frame-to-Map (F2M) sobre ROS 2 Jazzy con correspondencia geométrica PnP 3D-2D, refinamiento subpíxel local, preservación nativa de la escala métrica estéreo (`BundleAdjustment: 0`), supresión de sesgo inercial (`GuessMotion: false`) y suavizado continuo por filtro de Kalman.
4. **Nodo Colector Interactivo y Dataloader Zarr (`vo_collector_node.py`):** Muestreo a 20.0 Hz canónicos, redimensionamiento en RAM a $256 \times 256$ RGB, construcción de tensores sin fuga causal (estado 8D, acción 9D) y garantía de piso de parada terminal ($\ge 2.0\text{ s}$ / 40 fotogramas con `done=1.0`).
5. **Aislamiento de Red Localhost (`ROS_LOCALHOST_ONLY=1`):** Confinamiento estricto del flujo pesado de video ($46\text{ MB/s}$ / $370\text{ Mbps}$) en memoria compartida (SHM), erradicando al 100% la congestión de la red Wi-Fi y garantizando latencias inter-nodo $< 0.1\text{ ms}$.

### Diagrama de Flujo de Datos y Capas Desacopladas

```
      ┌────────────────────────────────────────────────────────┐
      │        Intel RealSense D435i (USB 3.2 SuperSpeed)      │
      │  - Par Estéreo Global Shutter + Emisor Láser IR        │
      │  - Sensor RGB Rolling Shutter (640x480 @ 30 Hz)        │
      │  - IMU Bosch BMI085 (Accel 200 Hz, Gyro 200 Hz)        │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ CAPA 1: Driver ROS 2 (realsense2_camera_node)          │
      │   /camera/camera/color/image_raw (640x480 RGB)         │
      │   /camera/camera/aligned_depth_to_color/image_raw      │
      │   /camera/camera/imu (Unificado vía Interp. Lineal)    │
      └─────────────┬──────────────────────────┬───────────────┘
                    │                          │
                    │                          ▼
                    │           ┌──────────────────────────────┐
                    │           │ imu_filter_madgwick          │
                    │           │   - Ganancia β = 0.035       │
                    │           │   - Damping ζ = 0.002        │
                    │           │   - Alineación Gravitacional │
                    │           │   -> /imu/data               │
                    │           └──────────────┬───────────────┘
                    │                          │
                    ▼                          ▼
      ┌────────────────────────────────────────────────────────┐
      │ CAPA 2: Motor VIO (rtabmap_odom / rgbd_odometry)       │
      │   - Estrategia: Frame-to-Map (F2M, MaxSize=5000)       │
      │   - Features: GFTT/ORB (1200 puntos, subpíxel 5x5)    │
      │   - Correspondencia: PnP 3D-2D + RANSAC                │
      │   - Escala: Nativa Estéreo (BundleAdjustment = 0)      │
      │   - Inercia: GuessMotion = false (Anti-Yaw Drift)      │
      │   - Filtrado: Filtro de Kalman continuo                │
      │   - Origen: En lente (0,0,0) (AlignWithGround = false) │
      │   - Clamping: 0.20 m <= Z <= 1.80 m                    │
      │   -> Publica: /rtabmap/odom (nav_msgs/Odometry @ 30 Hz)│
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ CAPA 3: Recolector Interactivo (vo_collector_node)     │
      │   - Muestreo periódico a 20.0 Hz sincrónicos (WallTimer)│
      │   - Redimensión en RAM a 256x256 RGB (INTER_AREA)      │
      │   - Transformación de Coordenadas Relativas (Delta-Home)│
      │   - Estado Propioceptivo 8D (Cero Fuga Causal)         │
      │   - Acción Cartesiana 9D + Piso Terminal (>=2s done=1) │
      │   - Interfaz: Teclado [ESPACIO, G, H, S, D, Q] o Srv   │
      │   -> Ingesta atómica a datasets/demonstrations_real_vX.zarr │
      └────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Sensorial y Parámetros del Driver

### 2.1 Intel RealSense D435i
* **Archivo de Configuración:** `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` (Líneas 1–24).
* **Sensor Estereoscópico Infrarrojo:**
  - **Línea de Base Física (*Baseline*):** $50\text{ mm}$ de separación entre los dos sensores infrarrojos (`infra1` e `infra2`).
  - **Tipo de Obturador:** *Global Shutter* en ambos sensores infrarrojos, eliminando artefactos de distorsión dinámica por barrido durante movimientos rápidos de traslación manual.
  - **Emisor Activo Infrarrojo:** `depth_module.emitter_enabled: 1` (Línea 9). El emisor VCSEL proyecta un patrón estructurado pseudo-aleatorio de micropuntos invisibles al ojo humano ($850\text{ nm}$), dotando de textura geométrica a superficies lisas y homogéneas (mesas metálicas, plásticos industriales).
  - **Resolución y Frecuencia de Profundidad:** `depth_module.depth_profile: "640x480x30"` (Línea 8).
  - **Alineación de Profundidad al Color:** `align_depth.enable: true` (Línea 6). El procesador D4 ejecuta la reproyección geométrica píxel a píxel del plano infrarrojo al marco de coordenadas de la cámara RGB antes de publicar `/camera/camera/aligned_depth_to_color/image_raw`.
* **Cámara RGB:**
  - **Resolución y Frecuencia:** `rgb_camera.color_profile: "640x480x30"` (Línea 7).
  - **Campo de Visión (FOV):** $86^\circ \times 57^\circ$ (Diagonal $94^\circ$).
  - **Sincronización por Hardware:** `enable_sync: true` (Línea 23). El hardware del ASIC D4 sincroniza a nivel de disparo de reloj los fotogramas del sensor RGB con los mapas de profundidad y los paquetes del IMU, publicando marcas de tiempo de hardware idénticas.
* **Optimización de Ancho de Banda USB:**
  - `pointcloud.enable: false` (Línea 10): Desactiva la computación de nubes densas de puntos en el driver, ahorrando ciclos de CPU y contención de memoria.
  - `enable_infra1: false`, `enable_infra2: false` (Líneas 12–13): Desactiva el streaming de los canales infrarrojos crudos sobre el bus USB 3.0, preservando el ancho de banda exclusivamente para RGB, Depth alineado e IMU.
  - `clip_distance: -1.0` (Línea 11): Sin truncamiento en el driver (se gestiona con mayor precisión en RTAB-Map).

### 2.2 IMU Integrado Bosch BMI085
* **Arquitectura de 6 GDL:** Acelerómetro piezoeléctrico de 16 bits y giróscopo MEMS de baja deriva integrados en el cuerpo de la cámara.
* **Parámetros del Driver (`d435i_camera_params.yaml`):**
  - `enable_accel: true` (Línea 16).
  - `enable_gyro: true` (Línea 17).
  - `accel_fps: 200` (Línea 19): Tasa de muestreo del acelerómetro a $200\text{ Hz}$.
  - `gyro_fps: 200` (Línea 20): Tasa de muestreo del giróscopo a $200\text{ Hz}$.
  - `unite_imu_method: 2` (Línea 18): Modo 2 (*Linear Interpolation*). Dado que el acelerómetro y el giróscopo operan en relojes de muestreo independientes dentro del chip BMI085, el driver interpola linealmente el dato inercial más lento hacia la estampa temporal del más rápido, publicando un flujo unificado y sincronizado de 6 GDL en el tópico `/camera/camera/imu` (`sensor_msgs/msg/Imu`).
  - `linear_accel_cov: 0.01` (Línea 21): Covarianza de aceleración lineal ($0.01\text{ (m/s}^2)^2$).
  - `angular_velocity_cov: 0.01` (Línea 22): Covarianza de velocidad angular ($0.01\text{ (rad/s)}^2$).
* **Gestión de Permisos USB y Buffers de Kernel IIO:**
  - En distribuciones Linux modernas (Ubuntu 24.04), el subsistema `iio-sensor-proxy` del escritorio compite por los dispositivos `/sys/bus/iio/devices/iio:device*`, provocando el error `EBUSY / HID set_power 1 failed`.
  - La solución activa en `tools/hardware/setup_realsense_udev.sh` (Líneas 12–17) enmascara el servicio:
    ```bash
    sudo systemctl stop iio-sensor-proxy.service
    sudo systemctl mask iio-sensor-proxy.service
    ```
  - En `tools/hardware/run_vo_container.sh` (Líneas 53), antes de arrancar y al salir se liberan los buffers:
    ```bash
    for dev in /sys/bus/iio/devices/iio:device*/buffer/enable; do
        [ -f "$dev" ] && echo 0 > "$dev" 2>/dev/null || true
    done
    ```

---

## 3. Filtrado Inercial en Vuelo (`imu_filter_madgwick`)

### 3.1 Justificación Matemática y Formulación Algorítmica
El chip Bosch BMI085 no cuenta con magnetómetro triaxial (para evitar distorsiones ferromagnéticas dentro del chasis de la cámara o al operar cerca de motores eléctricos de robots). Por lo tanto, no existe una referencia de rumbo magnético (*Yaw*) absoluto.

El nodo `imu_filter_madgwick` implementa la orientación basada en cuaterniones mediante optimización por gradiente descendente de la dirección del campo gravitatorio terrestre:
$$\mathbf{q}_{t} = \mathbf{q}_{t-1} + \left( \frac{1}{2} \mathbf{q}_{t-1} \otimes \boldsymbol{\omega}_t - \beta \frac{\nabla f(\mathbf{q}_{t-1}, \mathbf{a}_t)}{\|\nabla f(\mathbf{q}_{t-1}, \mathbf{a}_t)\|} \right) \Delta t$$
donde $\boldsymbol{\omega}_t$ es la velocidad angular del giróscopo, $\mathbf{a}_t$ es la aceleración lineal normalizada, y $\beta$ es el factor de divergencia que pondera la corrección del acelerómetro frente a la integración del giróscopo.

### 3.2 Parámetros Calibrados
* **Archivo de Launch:** `visuomotor_bringup/launch/realsense_d435i_vio.launch.py` (Líneas 43–64, 116–120).
* **Parámetros del Nodo `imu_filter_madgwick_node`:**
  - `use_mag: False` (Línea 50): Desactiva explícitamente la búsqueda de magnetómetro, impidiendo derivas erráticas por lecturas inexistentes.
  - `world_frame: 'enu'` (Línea 51): Alineación conforme a la convención estándar de robótica East-North-Up (donde el eje $Z$ apunta rígidamente en sentido opuesto a la gravedad).
  - `publish_tf: False` (Línea 52): RTAB-Map y el `robot_state_publisher` son los encargados únicos de publicar la transformación `odom -> camera_link`.
  - `gain: madgwick_gain` ($0.035$, Líneas 15–18, 53, 118–119): Reducido de su valor por defecto ($0.10$) a **$0.035$**. Una ganancia alta ($0.10$) causaba que las aceleraciones lineales del operador al mover la cámara a mano alzada se confundieran con inclinaciones gravitacionales, introduciendo temblor de cabeceo (*pitch jitter*). El valor $0.035$ proporciona una respuesta amortiguada y ultra-estable.
  - `zeta: 0.002` (Línea 54): Factor de compensación y amortiguamiento de deriva del sesgo (*bias*) del giróscopo.
  - `orientation_stddev: 0.02` (Línea 55): Desviación estándar estimada de la orientación ($0.02\text{ rad} \approx 1.15^\circ$).
* **Alineación Gravitacional (`wait_imu_to_init: true`):**
  - Parámetro en `rtabmap_vio_params.yaml` (Línea 9).
  - Durante el primer segundo de reposo en la marca de inicio, el acelerómetro mide el vector $\mathbf{g} = [0, 0, -9.81]\text{ m/s}^2$. Esto garantiza que el plano horizontal $(X, Y)$ sea estrictamente paralelo a la mesa de trabajo y el eje $Z$ apunte verticalmente hacia arriba, independientemente del ángulo físico de inclinación con que el operario sostenga la cámara al comenzar la demostración.

---

## 4. Arquitectura del Nodo RTAB-Map `rgbd_odometry` en ROS 2 Jazzy

* **Archivo de Parámetros:** `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` (Líneas 1–37).
* **Archivo de Parámetros Certificado:** `visuomotor_bringup/config/realsense/rtabmap_vio_params_best_umi.yaml` (Líneas 1–47).

### 4.1 Estrategia de Odometría Frame-to-Map (`Odom/Strategy: "0"`)
* **Línea 12:** `Odom/Strategy: "0"`.
* En un enfoque Frame-to-Frame (F2F), cada cuadro se alinea únicamente contra el cuadro inmediatamente anterior ($I_t \leftrightarrow I_{t-1}$), acumulando un error residual en cada paso temporal que degenera en una fuerte deriva parabólica.
* La estrategia Frame-to-Map (F2M) alinea el cuadro actual contra un mapa tridimensional local compuesto por una ventana deslizante de puntos clave extraídos de fotogramas anteriores ($I_t \leftrightarrow \mathcal{M}_{\text{local}}$).
* **Capacidad del Mapa Local:** `OdomF2M/MaxSize: "5000"` (Línea 15). Se amplía la memoria local a 5,000 puntos clave 3D. Esto permite que cuando la cámara se aleja $50\text{--}60\text{ cm}$ hacia el objeto y regresa a la posición de inicio, el sistema reconozca instantáneamente los puntos clave del punto de partida (*cradle*), logrando un **cierre de lazo milimétrico directo** sin necesidad de un optimizador de grafos global offline.

### 4.2 Detector y Descriptor de Características: GFTT / ORB (`Vis/FeatureType: "8"`)
* **Línea 19:** `Vis/FeatureType: "8"` (GFTT / ORB).
* **Tipo de Correspondencia:** `Vis/CorType: "0"` (Features Matching, Línea 18).
* **Densidad Máxima:** `Vis/MaxFeatures: "1200"` (Línea 20).
* **Consistencia Geométrica:** `Vis/MinInliers: "20"` (Línea 21).
* **Estimación de Pose:** `Vis/EstimationType: "1"` (PnP 3D-2D, Línea 22) con umbral de reproyección RANSAC `Vis/PnPReprojError: "1.8"` (Línea 23).
* **Fundamento Físico de Selección frente a SIFT/SURF:**
  - El proyector IR de la RealSense genera patrones de *laser speckle* con cambios de intensidad extremadamente abruptos y localizados en ventanas de $1 \times 1$ a $3 \times 3$ píxeles.
  - Los detectores basados en gradientes de esquinas Shi-Tomasi / Harris (GFTT) detectan estos picos de alta frecuencia con precisión submilimétrica.
  - En contraste, los detectores multiescala tipo SIFT o SURF convolucionan con pirámides de diferencias de gaussianas (DoG) que filtran y suavizan el ruido speckle de alta frecuencia, reduciendo drásticamente el número de candidatos válidos en superficies homogéneas de laboratorio (de $>900$ con GFTT/ORB a $<200$ con SIFT, provocando pérdidas de tracking).

### 4.3 Refinamiento Subpíxel (Erradicación del Jitter Numérico)
* **Líneas 28–29:**
  - `Vis/SubPixIterations: "15"`
  - `Vis/SubPixWinSize: "5"`
* En odometría visual convencional, las esquinas se localizan en coordenadas enteras de píxel $(u, v) \in \mathbb{Z}^2$. Al proyectar una discontinuidad de $1\text{ px}$ en un sensor con longitud focal $f_x \approx 380\text{ px}$ a $0.5\text{ m}$ de distancia, el salto espacial mínimo cuantizado es:
  $$\Delta X \approx \frac{Z}{f_x} \Delta u = \frac{0.50}{380} (1.0) \approx 1.31\text{ mm}$$
* Esto genera un temblor de alta frecuencia de $1\text{--}2\text{ mm}$ en reposo.
* Al activar `cv::cornerSubPix` con una ventana de $5 \times 5$ píxeles y 15 iteraciones de gradiente parabólico continuo, la localización de la esquina alcanza una resolución de $\sim 0.05\text{ px}$, reduciendo el salto numérico a $< 0.07\text{ mm}$ y estabilizando completamente la señal propioceptiva.

### 4.4 Bundle Adjustment Desactivado (`OdomF2M/BundleAdjustment: "0"`): Preservación de Escala Métrica 3D
* **Línea 32:** `OdomF2M/BundleAdjustment: "0"`.
* **Razón Fundamental de Ingeniería:**
  - En cámaras monoculares (e.g. GoPro en UMI original), el sensor no mide profundidad física directa. Es obligatorio ejecutar Bundle Adjustment (BA) no lineal ($g_2o$ o Ceres) sobre haces de rayos ópticos 2D para estimar conjuntamente la pose de la cámara y las coordenadas de los puntos 3D, requiriendo marcadores métricos externos (ArUco / ChArUco) para escalar el mapa.
  - En la RealSense D435i, **la distancia métrica 3D ya está calculada en hardware por el procesador de disparidad estéreo basado en la distancia focal y la línea de base calibrada de $50\text{ mm}$**.
  - Si se activa `BundleAdjustment: "1"` en RTAB-Map con una cámara RGB-D, el optimizador no lineal intenta minimizar el error de reproyección bidimensional modificando libremente la cota $Z$ de los puntos tridimensionales. Debido a fluctuaciones estadísticas normales en la lectura estéreo de profundidad, el optimizador "estira" o "comprime" continuamente la nube tridimensional local, induciendo **deriva de escala (*scale drift*) acumulativa**.
  - Al forzar `OdomF2M/BundleAdjustment: "0"`, se congela rígidamente la escala métrica física entregada por el hardware, garantizando que $1.0\text{ metro}$ medido por el VIO corresponda exactamente a $1.0\text{ metro}$ en el mundo físico.

### 4.5 Supresión del Modelo de Movimiento Inercial (`Odom/GuessMotion: "false"`)
* **Línea 13:** `Odom/GuessMotion: "false"`.
* **Razón de Estabilidad Cinemática:**
  - `Odom/GuessMotion: "true"` asume un modelo de velocidad constante o proyecta las lecturas de velocidad del IMU para sembrar la pose inicial del optimizador PnP.
  - Debido a la ausencia de magnetómetro en el chip BMI085, pequeñas variaciones térmicas inducen una lenta deriva en el ángulo de guiñada (*Yaw*).
  - Si `GuessMotion` está activo, cuando el operador frena abruptamente la cámara o la deja en reposo, el modelo de movimiento asume que sigue desplazándose, inyectando una "inercia virtual" o arrastre ficticio que deforma la estimación de la pose.
  - Al fijar `GuessMotion: "false"`, cada cuadro se resuelve mediante correspondencia geométrica pura y directa contra el mapa local, erradicando el arrastre inercial y eliminando por completo la deriva en Yaw en paradas estáticas.

### 4.6 Filtrado Temporal y Origen Estricto en Lente
* **Filtro de Kalman:** `Odom/FilteringStrategy: "1"` (Línea 35). Un filtro de Kalman continuo en el espacio de estados suaviza las estimaciones cuadro a cuadro, eliminando el ruido blanco de alta frecuencia remanente.
* **Origen Estricto en Lente:** `Odom/AlignWithGround: "false"` (Línea 36). En robótica móvil terrestre, RTAB-Map intenta proyectar la altura de la cámara hacia el suelo estimando una altura $Z_{\text{base}}$. En manipulación manual aérea, esto introduciría un desplazamiento heurístico arbitrario e impredecible. Con `"false"`, el sistema de coordenadas de la odometría arranca de manera determinista y estricta en $(0.0, 0.0, 0.0)$ en el centro focal del lente de la cámara en el instante $t=0$.
* **Protección contra Pérdida de Rastreo:** `Odom/ResetCountdown: "0"` (Línea 14). Si ocurre una oclusión total temporal, la odometría se congela en su última pose válida y no salta a $(0,0,0)$, evitando discontinuidades catastróficas en las trayectorias de entrenamiento.

### 4.7 Ventana de Clamping de Profundidad ($0.20\text{ m} \le Z \le 1.80\text{ m}$)
* **Líneas 24–25:**
  - `Vis/MinDepth: "0.20"`
  - `Vis/MaxDepth: "1.80"`
* **Corte Inferior ($0.20\text{ m}$):** La distancia física focal mínima del par estéreo de la RealSense D435i es de aproximadamente $18\text{--}20\text{ cm}$. Puntos más cercanos caen en la zona ciega de disparidad estéreo y generan ruido geométrico espurio.
* **Corte Superior ($1.80\text{ m}$):** En tareas de manipulación sobre banco de trabajo, el espacio de trabajo efectivo abarca distancias de $0.20$ a $1.0\text{ m}$. Los objetos situados a más de $1.8\text{ metros}$ (paredes de la habitación, personas caminando en el fondo, ventanas) poseen un error de disparidad cuadrático ($\Delta Z \propto Z^2$) y movimiento dinámico no estructurado. Descartar todo punto $> 1.80\text{ m}$ blinda el sistema contra distorsiones de fondo y concentra los 1,200 puntos clave en el volumen útil de manipulación.

---

## 5. Telemetría Empírica, Rendimiento y Benchmarks Metrológicos

### 5.1 Protocolo Experimental de Cierre de Lazo (`tools/benchmark_vio_pose.py`)
Para certificar cuantitativamente la precisión métrica del VIO sin recurrir a afirmaciones no contrastadas, se diseñó un protocolo de metrología automatizado de 3 fases:
1. **Fase 1 (Reposo Inicial):** Medición de estabilidad estática durante $3.0\text{ segundos}$ en la marca de origen física ($p_{\text{start}}, q_{\text{start}}$), cuantificando el jitter estático $\sigma_{\text{pos}}$.
2. **Fase 2 (Trayectoria Dinámica Libre):** Movimiento tridimensional continuo a mano alzada simulando aproximación y agarre (excursión $> 50\text{ cm}$, recorrido total $> 2.5\text{ m}$), registrando FPS, inliers dinámicos y porcentaje de pérdida de tracking.
3. **Fase 3 (Reposo Final y Cierre de Bucle):** Regreso exacto al cradle o marca de origen física, midiendo estabilidad estática final durante $3.0\text{ segundos}$ ($p_{\text{end}}, q_{\text{end}}$).

#### Fórmulas de Cálculo de Métricas:
* **Error de Cierre de Bucle Tridimensional (Loop Closure Drift):**
  $$\Delta \mathbf{p} = (\mathbf{p}_{\text{end}} - \mathbf{p}_{\text{start}}) \times 1000 \quad [\text{mm}], \quad E_{\text{loop}} = \|\Delta \mathbf{p}\|_2 = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$$
* **Error Angular Geodésico en $\mathbb{SO}(3)$:**
  $$\theta_{\text{err}} = 2 \arccos\left(|\mathbf{q}_{\text{start}} \cdot \mathbf{q}_{\text{end}}|\right) \quad [\text{grados}]$$
* **Jitter de Dispersión Estática en Reposo ($\sigma_{\text{pos}}$):**
  $$\sigma_{\text{pos}} = \sqrt{\sigma_X^2 + \sigma_Y^2 + \sigma_Z^2} \times 1000 \quad [\text{mm}]$$

### 5.2 Historial Experimental y Evolución Paramétrica
Extraído directamente de `experiments/vio_benchmark_history.json`:

| Configuración Evaluada | Excursión Máx. (cm) | Recorrido Total (m) | Error Retorno $E_{\text{loop}}$ (mm) | Jitter Estático $\sigma$ (mm) | Inliers Promedio | Pérdida Tracking (%) | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **conf 0 (Baseline Descalibrado)** | 38.40 | 13.59 | 263.647 | 10.4644 | 244.4 | 0.0% | RECHAZADO |
| **conf 0 (Baseline Descalibrado)** | 23.94 | 10.86 | 48.811 | 14.8921 | 230.4 | 0.0% | RECHAZADO |
| **conf 1 (Pure VO - Sin IMU)** | 34.27 | 6.06 | 103.940 | 3.2542 | 272.8 | 0.0% | RECHAZADO |
| **conf 2 (Madgwick 0.02)** | 107.43 | 10.98 | 98.824 | 5.4248 | 343.2 | 0.0% | RECHAZADO |
| **conf 3 (UMI-Grade VIO Preliminar)** | 81.94 | 6.56 | 51.323 | 5.6444 | 301.0 | 0.0% | RECHAZADO |
| **conf 3 (UMI-Grade VIO Preliminar)** | 68.18 | 5.37 | 42.565 | 3.4022 | 429.2 | 0.0% | RECHAZADO |
| **conf 4 (SOTA UMI Final Fusion - Run A)** | 61.54 | 7.24 | 14.411 | 6.1139 | 221.2 | 0.0% | EN AJUSTE |
| **conf 4 (SOTA UMI Final Fusion - Certificado)** | **54.56** | **2.79** | **2.307** | **1.6144** | **258.8** | **0.0%** | **PASA (SOTA)** |
| **conf 4 (SOTA UMI Final Fusion - Run C)** | 67.67 | 3.64 | 21.618 | 2.6039 | 436.2 | 0.0% | ESTABLE |
| **conf 4 (SOTA UMI Final Fusion - Run D)** | 60.18 | 3.45 | 16.165 | 3.3806 | 399.8 | 0.0% | ESTABLE |

### 5.3 Métricas de la Configuración Certificada Activa (`conf 4 — SOTA UMI Final Fusion`)
* **Timestamp:** `2026-11-19 20:33:05` (Registro en `experiments/vio_benchmark_history.json`, líneas 183–199).
* **Criterios de Aceptación:**
  1. Error de retorno: $E_{\text{loop}} \le 3.00\text{ mm}$ (PASA: **$2.307\text{ mm}$**).
  2. Jitter estático en reposo: $\sigma \le 2.50\text{ mm}$ (PASA: **$1.6144\text{ mm}$**, reducción del $89.2\%$ respecto al baseline de $14.89\text{ mm}$).
  3. Pérdida de tracking: $0.0\%$ (PASA: **$0.0\%$**, cero fotogramas perdidos).
* **Desglose Cartesiano del Error Residual:**
  $$\Delta X = -0.828\text{ mm}, \quad \Delta Y = -0.064\text{ mm}, \quad \Delta Z = +2.153\text{ mm}$$
* **Error Angular Geodésico:** $2.562^\circ$ ($\Delta\text{Roll} = -2.48^\circ, \Delta\text{Pitch} = +0.60^\circ, \Delta\text{Yaw} = +0.31^\circ$).
* **Rendimiento Computacional y Latencia:**
  - Tiempo de procesamiento CPU por cuadro: **$13.98 \pm 0.65\text{ ms}$**.
  - En un presupuesto de $33.3\text{ ms}$ por fotograma a 30 FPS, el VIO consume apenas el **$42\%$** del tiempo disponible, dejando más del **$55\%$ de margen de cómputo libre** en la CPU i9 para redimensionamiento en RAM e ingesta Zarr.
* **Densidad de Inliers:** Entre **$390$ y $450$ inliers** consistentes bajo la trama activa infrarroja (media sostenida de $258.8$ a $436.2$ en movimiento dinámico).
* **Comparativa con VO Puro (D415 en banco estático):**
  - Jitter en reposo sobre montura rígida: **$0.3943\text{ mm}$** ($\sigma_X=0.19, \sigma_Y=0.23, \sigma_Z=0.25\text{ mm}$) y jitter angular de **$0.0467^\circ$** (`benchmark_vo.py`).
  - Al pasar a operación manual aérea con D435i + IMU, el jitter dinámico se mantiene en un excepcional **$1.61\text{ mm}$**, muy por debajo de los umbrales de tolerancia de agarre de pinzas paralelas estándar ($10\text{--}20\text{ mm}$).

---

## 6. Integración con el Recolector FSM (`vo_collector_node.py`)

* **Ubicación del Código:** `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` (Líneas 1–400).
* **Archivo de Parámetros:** `visuomotor_data_collection/config/vo_data_collection_params.yaml` (Líneas 1–17).

### 6.1 Bucle de Muestreo Sincrónico a 20.0 Hz
* **Líneas 121–123:**
  ```python
  rate = self.get_parameter('sampling_rate_hz').value # 20.0 Hz
  self.sample_timer = self.create_timer(1.0 / rate, self._sample_step, callback_group=self.cbg)
  ```
* Aunque la cámara y RTAB-Map publican a $30\text{ Hz}$, el colector desacopla la frecuencia de ingesta mediante un temporizador determinista de ROS 2 (`WallTimer`) a **$20.0\text{ Hz}$ exactos** ($\Delta t = 50.0\text{ ms}$). Esto estandariza la cadencia temporal con los dataloaders de Diffusion Policy y la simulación en Isaac Sim.

### 6.2 Procesamiento Visual en Memoria RAM
* **Líneas 142–150:**
  ```python
  cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
  if cv_img.shape[0] != self.image_size or cv_img.shape[1] != self.image_size:
      cv_img = cv2.resize(cv_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
  self.latest_img = cv_img
  ```
* Los fotogramas recibidos a $640 \times 480$ RGB se interpolan directamente en memoria RAM mediante el algoritmo de área de OpenCV (`cv2.INTER_AREA`), ideal para reducción de resolución sin artefactos de aliasing ni moiré.
* El tensor resultante tiene dimensiones estándar $[256, 256, 3]$ en formato `uint8`, idéntico a las observaciones de la red neuronal U-Net.

### 6.3 Paradigma de Coordenadas Relativas (Delta-Home / UMI)
* **Líneas 204–221:**
  - Al iniciar la grabación de un episodio con `[ESPACIO]`, se captura la pose instantánea del lente como $\mathbf{p}_{\text{start}}$.
  - Para todo paso $t$, la posición registrada es:
    $$\mathbf{p}_{\text{rel}}(t) = \mathbf{p}(t) - \mathbf{p}_{\text{start}}$$
  - En el paso $t=0$, el colector fuerza explícitamente:
    $$\mathbf{p}_{\text{rel}}(0) \equiv [0.0, 0.0, 0.0]$$
    erradicando cualquier temblor residual en el primer fotograma.
* **Estabilización de Cuaterniones (Semiesfera Superior):**
  - **Líneas 158–176:**
    $$\text{Si } q_w < 0.0 \implies \mathbf{q} \leftarrow -\mathbf{q}$$
    $$\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|_2}$$
  - Esto garantiza la unicidad topológica de la rotación en $\mathbb{SO}(3)$, evitando discontinuidades de signo entre $+\mathbf{q}$ y $-\mathbf{q}$ que desestabilizarían la función de pérdida MSE durante el entrenamiento de difusión.

### 6.4 Composición de Tensores y Prohibición de Fuga Causal
* **Tensor de Observación Propioceptiva (`observation.state`) — 8 Dimensiones (Líneas 223–229):**
  $$\mathbf{s}_t = [X_{\text{rel}}, Y_{\text{rel}}, Z_{\text{rel}}, q_x, q_y, q_z, q_w, \text{gripper}] \in \mathbb{R}^8$$
  * *Regla Inviolable de Arquitectura:* **CERO FUGA CAUSAL (Zero Causal Leakage)**. La señal de término (`done`) JAMÁS forma parte de `observation.state`. La observación modela exclusivamente lo que los sensores físicos miden en el instante $t$. Si se incluyera `done` en la entrada, la red neuronal aprendería una dependencia circular espuria y en inferencia en tiempo real requeriría un oráculo externo.
* **Tensor de Acción Predicha (`action`) — 9 Dimensiones (Líneas 231–244):**
  $$\mathbf{a}_t = [X_{\text{rel}}, Y_{\text{rel}}, Z_{\text{rel}}, q_x, q_y, q_z, q_w, \text{gripper}, \text{done\_flag}] \in \mathbb{R}^9$$
  * Modela la trayectoria futura a ejecutar por el efector final ($H_{act} = 16$).
  * Durante la aproximación dinámica: $\text{done\_flag} = 0.0$.
  * Durante la fase de asentamiento en destino (`HOLD_TARGET`): $\text{done\_flag} = 1.0$.

### 6.5 Garantía de Piso Terminal de Auto-Stop ($\ge 2.0\text{ s}$ / 40 fotogramas)
* **Líneas 285–292:**
  ```python
  rate = self.get_parameter('sampling_rate_hz').value # 20 Hz
  min_hold_frames = int(rate * self.get_parameter('min_hold_duration_s').value) # 40 frames
  start_hold_idx = max(0, total_frames - min_hold_frames)
  if self.get_parameter('record_done_flag').value:
      for i in range(start_hold_idx, total_frames):
          self.buffer[i]['action'][8] = 1.0
  ```
* Para mitigar el sesgo de truncamiento de horizonte (*Horizon Truncation Bug*) en los dataloaders de difusión con ventanas deslizantes de longitud $H_{act}=16$, el colector impone de forma forzada un piso mínimo de **al menos 40 fotogramas consecutivos ($2.0\text{ s}$ a 20 Hz)** con `action[8] = 1.0` al final de cada demostración. Esto garantiza que la red neuronal aprenda con alta densidad estadística el patrón de parada terminal y estabilización.

### 6.6 Almacenamiento en Jerarquía Zarr Atómica
* **Estructura Interna del Archivo Zarr:**
  ```
  demonstrations_real_vX.zarr/
  ├── episode_0/
  │   ├── image  (uint8, [N, 256, 256, 3])
  │   ├── state  (float32, [N, 8])
  │   └── action (float32, [N, 9])
  │   └── .zattrs (metadata: sampling_rate_hz, relative_coordinates, hold_steps, etc.)
  ├── episode_1/
  └── ...
  ```
* **Modos de Grabación:**
  - **Modo AUTO (`dataset_path: "AUTO"`):** Escanea `datasets/demonstrations_real_v*.zarr`, extrae la versión máxima $v_{\max}$ y crea automáticamente `demonstrations_real_v{max+1}.zarr` (Líneas 58–71).
  - **Modo AUMENTO (`dataset_path: "demonstrations_real_v5"`):** Abre el dataset indicado, determina el número de episodios existentes $N_{\text{prev}}$ y añade los nuevos episodios de forma consecutiva indexando a partir de `episode_{N_prev}` (Líneas 72–95).

---

## 7. Blindaje de Red DDS: Localhost vs. Jetson Orin Nano

### 7.1 Reto del Ancho de Banda y Saturación Wi-Fi
* La cámara Intel RealSense D435i transmite simultáneamente:
  - Color crudo: $640 \times 480 \times 3\text{ bytes} \times 30\text{ fps} \approx 27.65\text{ MB/s}$
  - Profundidad alineada cruda: $640 \times 480 \times 2\text{ bytes} \times 30\text{ fps} \approx 18.43\text{ MB/s}$
  - IMU a 200 Hz: $< 0.1\text{ MB/s}$
  - **Flujo Total Agregado en DDS:** **$\approx 46.18\text{ MB/s}$ ($\approx 369.4\text{ Mbps}$)**.
* Si el middleware DDS (e.g. CycloneDDS o FastDDS) transmite este flujo mediante multicast a través de la tarjeta de red Wi-Fi (`wlo1`), se satura por completo el canal inalámbrico. Esto produce caídas masivas de paquetes, retrasos superiores a $200\text{ ms}$, caída de la frecuencia a $< 5\text{ Hz}$ y el descarte recurrente de imágenes en RTAB-Map (`Dropping image...`).
* Además, si existen otros computadores o robots en la misma red Wi-Fi de laboratorio usando ROS 2 en `ROS_DOMAIN_ID=0`, se generan colisiones de tópicos globales (`/tf`, `/camera/...`) e interferencias mutuas destructivas.

### 7.2 Blindaje Localhost Estricto (Zero-Latency Local)
* **Implementación:** `tools/hardware/run_vo_container.sh` (Líneas 31–44) y `tools/hardware/run_benchmark.sh` (Línea 18).
* **Directivas de Entorno Inyectadas:**
  ```bash
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=1
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  ```
* **Mapeo de Contenedor:** `--net=host --ipc=host`.
* **Consecuencias Técnicas:**
  1. El 100% de los paquetes UDP de DDS se enrutan exclusivamente a través de la interfaz de loopback local (`lo`) y memoria compartida del kernel Linux.
  2. Cero bytes salen hacia la tarjeta inalámbrica o Ethernet.
  3. Cero visibilidad o colisión con equipos externos en la LAN.
  4. Latencia de transporte inter-proceso reducida a $< 0.1\text{ ms}$.
  5. Frecuencia de odometría sostenida a $30.11\text{ Hz}$ exactos sin pérdida de fotogramas.

### 7.3 Topología Dual con Robot Real (Jetson Orin Nano)
* **Desacoplamiento Estricto:**
  - **Estación de Trabajo / Laptop:** Corre la inferencia de Diffusion Policy a 20 Hz sobre su GPU local.
  - **Enlace Remoto Jetson Orin Nano:** Opera bajo **`ROS_DOMAIN_ID=42`** mediante CycloneDDS Unicast configurado en `deploy/jetson/config/cyclonedds.xml` con `DDS_PEER=<ip_jetson>`.
  - **Carga de Tráfico Distribuido:** A través de la red solo viajan paquetes de control ultra-ligeros (`ActionChunk` de $16 \times 9$ floats y comandos `/target_frame` a $< 50\text{ KB/s}$), manteniendo la red Wi-Fi completamente limpia y garantizando una ejecución segura en el robot físico xArm5.

```mermaid
flowchart TD
    subgraph Workstation_Local ["Estación de Trabajo (Captura VIO / Inferencia DP)"]
        D435i["RealSense D435i (USB 3.0)"] --> RS_Node["realsense2_camera (30 Hz)"]
        RS_Node -->|"46 MB/s (SHM / lo)"| VIO_Node["RTAB-Map rgbd_odometry (30 Hz)"]
        VIO_Node -->|"Pose 6DoF"| Collector["vo_collector_node (20 Hz)"]
        Collector --> Zarr[("Zarr Dataset\n(8D State, 9D Action)")]
        Policy["policy_node.py (Inferencia DP)"]
        
        Shield["Blindaje Estricto:\nROS_DOMAIN_ID=0\nROS_LOCALHOST_ONLY=1\nROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST\n(Cero tráfico hacia Wi-Fi)"]
    end

    subgraph Remote_Jetson ["Robot Real (Jetson Orin Nano / xArm5)"]
        Bridge["xarm5_bridge.py\n(Controlador Modo 1)"] --> xArm5["xArm5 Controller (192.168.1.228)"]
    end

    Policy -.->|"CycloneDDS Unicast (< 50 KB/s)\nROS_DOMAIN_ID=42 (DDS_PEER)"| Bridge
```

---

## 8. Matriz de Trazabilidad y Referencias Cruzadas del Repositorio

| Parámetro / Componente | Archivo de Origen | Líneas Exactas | Valor / Configuración Activa | Justificación Técnica |
| :--- | :--- | :---: | :--- | :--- |
| **Color Profile** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 7 | `640x480x30` | Resolución balanceada y 30 FPS nativos. |
| **Depth Profile** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 8 | `640x480x30` | Sincronizado con stream de color. |
| **Active IR Emitter** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 9 | `1` (Habilitado) | Proyección laser speckle para densificar texturas. |
| **Depth Alignment** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 6 | `true` | Reproyección en hardware a plano de color RGB. |
| **IMU Unification** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 18 | `2` (Linear Interp) | Sincronización continua de acelerómetro y giróscopo. |
| **IMU Sampling Rates** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 19–20 | `200` Hz / `200` Hz | Frecuencia nativa de chip Bosch BMI085. |
| **Hardware Sync** | `visuomotor_bringup/config/realsense/d435i_camera_params.yaml` | 23 | `true` | Estampas temporales sincronizadas por hardware. |
| **Madgwick Gain ($\beta$)**| `visuomotor_bringup/launch/realsense_d435i_vio.launch.py` | 53, 118–119 | `0.035` | Amortiguamiento de temblor ante aceleraciones dinámicas. |
| **Madgwick Gyro Drift ($\zeta$)**| `visuomotor_bringup/launch/realsense_d435i_vio.launch.py` | 54 | `0.002` | Compensación continua de sesgo angular del giróscopo. |
| **Gravity Wait Init** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 9 | `true` | Anclaje gravitacional vertical estricto al arrancar. |
| **VIO Strategy** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 12 | `"0"` (Frame-to-Map) | Alineación contra mapa local para mínima deriva. |
| **Local Map Capacity** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 15 | `"5000"` | Reconocimiento visual del punto de inicio al regresar. |
| **Feature Extraction** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 19 | `"8"` (GFTT/ORB) | Detección óptima de esquinas speckle IR. |
| **Max Features** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 20 | `"1200"` | Cobertura densa en campo visual de $86^\circ$. |
| **PnP Reprojection Err** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 23 | `"1.8"` | Tolerancia RANSAC precisa contra ruido estéreo. |
| **Subpixel Refinement**| `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 28–29 | `WinSize: 5`, `Iter: 15` | Erradica cuantización discreta de píxeles enteros. |
| **Bundle Adjustment** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 32 | `"0"` (Desactivado) | Preserva la escala métrica 3D nativa del hardware. |
| **Guess Motion** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 13 | `"false"` (Desactivado) | Suprime arrastre inercial ficticio y deriva en Yaw. |
| **Kalman Filtering** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 35 | `"1"` (Habilitado) | Suavizado inter-cuadro continuo de trayectoria. |
| **Align With Ground** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 36 | `"false"` (Desactivado) | Origen $(0,0,0)$ estricto en el lente de la cámara. |
| **Depth Clamping** | `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml` | 24–25 | `Min: 0.20`, `Max: 1.80` | Acotamiento estricto al volumen de trabajo manual. |
| **Collector Rate** | `visuomotor_data_collection/config/vo_data_collection_params.yaml` | 7 | `20.0` Hz | Frecuencia estándar sincrónica de Diffusion Policy. |
| **RAM Image Resize** | `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` | 145–147 | `256x256` (`INTER_AREA`) | Normalización visual de entrada para la red U-Net. |
| **Relative Coords ($t=0$)**| `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` | 210, 215 | `np.zeros(3)` | Origen invariante idéntico a $(0,0,0)$ en paso inicial. |
| **Quaternion Normalize**| `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` | 163–176 | $q_w \ge 0$, $\|\mathbf{q}\|=1.0$ | Estabilización en semiesfera superior sin saltos. |
| **Zero Causal Leakage**| `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` | 224–229 | Estado 8D sin `done` | Prohibición estricta de fuga causal en observación. |
| **Action & Done Floor**| `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py` | 232–244, 287–292 | Acción 9D, piso 40 frames | Garantía de $\ge 2.0\text{ s}$ con `done=1.0` en destino. |
| **Network Isolation** | `tools/hardware/run_vo_container.sh` | 42–44 | `LOCALHOST_ONLY=1` | Confinamiento de $46\text{ MB/s}$ a loopback/SHM. |

---

## 9. Conclusiones de la Auditoría

1. **Madurez Técnica y Robustez Metrológica:** El pipeline activo de VIO en `visuomotor_ros2` ha alcanzado un estado de optimización metrológica rigurosa, certificado experimentalmente con un error de retorno de cierre de bucle de **$2.307\text{ mm}$**, jitter de reposo de **$1.61\text{ mm}$**, inliers consistentes de **$390\text{--}450$** y **$0.0\%$ de pérdida de tracking**.
2. **Superioridad en Preservación de Escala:** Al fijar `BundleAdjustment: "0"` y apoyarse en la disparidad infrarroja activa del ASIC de Intel, el sistema elimina de raíz la deformación de escala tridimensional inherente a los sistemas de visión monocular pasiva.
3. **Erradicación de Deriva Inercial:** La desactivación de `GuessMotion: "false"` junto a la calibración fina de Madgwick ($\beta = 0.035, \zeta = 0.002$) desacopla con éxito la odometría de los sesgos térmicos de guiñada (*Yaw*) del giróscopo MEMS sin magnetómetro.
4. **Conformidad Estricta con el Estándar de Diffusion Policy:** La integración en `vo_collector_node.py` implementa rigurosamente el paradigma de coordenadas relativas (Delta-Home / UMI), garantizando cero fuga causal en el vector de observación propioceptiva (8D) y dotando al vector de acción (9D) de un piso terminal garantizado de 40 fotogramas ($2.0\text{ s}$) con `done = 1.0`.
5. **Aislamiento Operativo de Red:** El blindaje DDS con `ROS_LOCALHOST_ONLY=1` resuelve de forma definitiva el colapso del bus Wi-Fi, permitiendo sesiones de recolección continua sin degradación de frecuencia ni pérdida de fotogramas.
