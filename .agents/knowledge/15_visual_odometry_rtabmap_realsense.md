# 15. Odometría Visual con RTAB-Map y RealSense D415 para Demostraciones Reales de Approaching

> [!WARNING]
> **Revisión crítica del 2026-11-23 (ver §0 al final del módulo y `context.md`, lección 43).** Las cifras de "precisión
> certificada" de este módulo (§6, §9, §10.D y §F) se midieron en reposo o son el mejor de varios intentos, y nunca
> contra una referencia. 2.307 mm fue el mejor de 9 retornos (mediana 21.6 mm). Además, la configuración D435i no es
> un VIO: el IMU solo alinea el marco inicial con la gravedad. No usar estas cifras como precisión de las
> demostraciones. Plan de mejora: `docs/plans/plan_d435i_herramienta_demostraciones.md`.

Este módulo documenta la arquitectura técnica, integración de hardware y software, parámetros algorítmicos, validación experimental rigurosa y protocolo de captura de demostraciones expertas reales de aproximación (*approaching*) mediante **Odometría Visual (VO)** sobre **ROS 2 Jazzy** en un contenedor Docker con aceleración de hardware.

---

## 1. Misión y Topología del Sistema Físico

El objetivo es capturar demostraciones cinemáticas continuas 6DoF a mano alzada para entrenar modelos de **Políticas Visuomotoras Generativas (Diffusion Policy)** sin requerir un robot físico manipulador encendido durante la fase de recolección de aproximación.

### Hardware Base
* **Estación de Cómputo:** Portátil MSI Raider GE68HX (Intel Core i9-13950HX, 32 hilos, 32 GB RAM DDR5, GPU NVIDIA RTX 4060 Laptop 8GB VRAM).
* **Sensor Óptico:** Intel RealSense D415 conectada a puerto USB 3.2 Gen 1 (5000M SuperSpeed).
  * *Características físicas:* Sensor estereoscópico activo con emisor infrarrojo (IR), resolución nativa 1920x1080 (RGB) / 1280x720 (Depth), obturador rodante (*rolling shutter*), sin IMU interna. Distancia mínima medible $Z_{\min} \approx 20\text{ cm}$.
* **Entorno Gráfico y Contenedor:** Ubuntu 24.04 LTS (Noble) con compositor Hyprland (Wayland / XWayland). El procesamiento se ejecuta dentro del contenedor Docker `visuomotor_ros2` (imagen snapshot `visuomotor_ros2_vo:jazzy`).

---

## 2. Flujo de Datos y Arquitectura de Software

Para evitar contención en el bus USB y asegurar sincronización milimétrica sin fugas causales, el flujo se organiza en tres capas desacopladas conectadas por el middleware ROS 2:

```
[Intel RealSense D415 (USB 3.2)]
              │
              ▼
[Capa 1: Driver ROS 2 (realsense2_camera_node)]
  ├─► /camera/camera/color/image_raw (640x480 @ 30 Hz)
  └─► /camera/camera/aligned_depth_to_color/image_raw (640x480 @ 30 Hz)
              │
              ▼
[Capa 2: Motor VO (rtabmap_odom / rgbd_odometry)]
  └─► Frame-to-Map (F2M) + PnP 3D-2D + RANSAC + Filtro Kalman
  └─► Publica: /rtabmap/odom (nav_msgs/msg/Odometry a ~30–57 Hz)
              │
              ▼
[Capa 3: Recolector Interactivo (vo_collector_node)]
  ├─ Sincronización temporal a 20.0 Hz canónicos
  ├─ Redimensión en RAM a 256x256 RGB
  ├─ Construcción de tensores:
  │    ├─ observation.state (8D): [X, Y, Z, Qx, Qy, Qz, Qw, Gripper]
  │    └─ action (9D): [X_tgt, Y_tgt, Z_tgt, Qx, Qy, Qz, Qw, Gripper, Done]
  └─ Ingesta atómica a datasets/demonstrations_real_vX.zarr
```

---

## 3. Desacoplamiento y Organización de Archivos

1. **`visuomotor_bringup` (Configuraciones de Hardware, RViz y Launch):**
   - `config/realsense/d415_camera_params.yaml`: Configura perfiles a `640x480x30`, activa alineación de profundidad (`align_depth.enable: true`) y enciende el proyector IR (`depth_module.emitter_enabled: 1`).
   - `config/realsense/rtabmap_vo_params.yaml`: Parámetros Baseline de odometría (GFTT/ORB) calibrados para mínima deriva y supresión de auto-reinicio.
   - `config/realsense/rtabmap_vo_ultra_precision.yaml`: Parámetros Ultra-Precisión (SIFT 128D, subpíxel $3\times 3$, $g_2o$ BA).
   - `rviz/vo_vis.rviz`: Configuración de RViz2 para visualización de mapa local 3D (`/odom_local_map`), TF de cámara y odometría.
   - `launch/realsense_d415_vo.launch.py`: Driver + Odometría (+ Visor RViz2 con `gui:=true` y selector dinámico `ultra_precision:=true/false` vía `OpaqueFunction`).
   - `launch/realsense_d415_vo_collection.launch.py`: Orquestador maestro que levanta cámara, VO y nodo recolector.
2. **`visuomotor_data_collection` (Captura y Almacenamiento):**
   - `config/vo_data_collection_params.yaml`: Parámetros de tasa de muestreo (20 Hz), tópicos y prefijo `demonstrations_real`.
   - `visuomotor_data_collection/vo_collector_node.py`: Nodo de recolección interactiva por teclado con FSM y garantía de piso de parada.
3. **`tools/hardware/` (Automatización en Host):**
   - `run_vo_container.sh`: Lanzador Docker con paso directo dinámico de `/dev` y `/dev/bus/usb`, permisos de grupo `--group-add 44 --group-add 46`, variables XWayland y CycloneDDS.
   - `setup_realsense_udev.sh`: Reglas udev de permisos en el host para dispositivos RealSense.
   - `benchmark_vo.py`: Script de metrología que evalúa frecuencia, tiempo de CPU, inliers y jitter RMS en posición (mm) y rotación (grados).
   - `run_comparison_experiment.sh`: Test harness automatizado para contrastar configuraciones en condiciones de laboratorio estáticas.

---

## 4. Análisis Comparativo: RTAB-Map RGB-D vs. NVIDIA cuVSLAM

| Dimensión Técnica | RTAB-Map (`rgbd_odometry`) | NVIDIA cuVSLAM (`isaac_ros_visual_slam`) |
| :--- | :--- | :--- |
| **Backend de Cómputo** | CPU Multihilo optimizada (C++) | GPU Masiva (CUDA / Tensor Cores) |
| **Entrada de Datos** | **RGB + Mapa de Profundidad ASIC** | **Par Estéreo Infrarrojo Crudo + IMU** |
| **Dependencia de IMU** | **Nula** (opera con alta precisión sólo con RGB-D) | **Crítica** (requiere fusión VIO para estabilidad) |
| **Sensibilidad a Rolling Shutter** | Baja/Media (mitigada por matching 3D denso) | **Muy Alta** (diseñado para Global Shutter) |
| **Compatibilidad con D415** | **Nativa y Oficial** | **No soportada oficialmente por NVIDIA** |
| **Frecuencia en i9 / RTX 4060** | ~30–57 Hz (actualización en 13–22 ms) | 60–120+ Hz (actualización en < 5 ms) |
| **Idoneidad para Approaching** | **Excelente** (alta densidad y menor deriva a 20–50 cm) | Pobre con D415 (pérdida de escala y drift sin IMU) |

**Conclusión:** Para la cámara Intel RealSense D415, RTAB-Map RGB-D es superior técnicamente debido a que aprovecha el proyector de textura IR y el cómputo de disparidad en silicio del ASIC de Intel, mientras que cuVSLAM sufre inestabilidades severas al carecer de IMU física y estar sujeto a distorsiones de rolling shutter.

---

## 5. Estrategias Algorítmicas de Odometría en RTAB-Map

### Configuración Equilibrada (Baseline — Ganadora Indiscutible)
* `Odom/Strategy = "0"`: **Frame-to-Map (F2M)**. El cuadro actual se alinea contra un mapa tridimensional local de keyframes recientes, suprimiendo la deriva acumulativa de los enfoques Frame-to-Frame.
* `Vis/FeatureType = "8"`: **GFTT/ORB (Good Features to Track + ORB)**. Extracción rápida de esquinas Shi-Tomasi con hasta 1000 puntos clave.
* `Vis/EstimationType = "1"`: **PnP 3D-2D**. Minimiza el error de reproyección en el plano de la imagen mediante RANSAC (`Vis/PnPReprojError: "2.0"`).
* `Odom/FilteringStrategy = "1"`: **Filtro de Kalman**. Suavizado continuo de coordenadas y orientaciones.
* `Odom/ResetCountdown = "0"`: **Protección contra Pérdida de Origen**. Si el tracking se pierde, congela la odometría y no resetea las coordenadas a $(0,0,0)$.

### Configuración Ultra-Precisión (SIFT Calibrado)
* `Vis/FeatureType = "1"`: SIFT continuo de 128 dimensiones con `Vis/MaxFeatures: "400"`.
* `Vis/SubPixIterations = "10"`, `Vis/SubPixWinSize = "3"`: Refinamiento de esquinas mediante `cv::cornerSubPix`.
* `OdomF2M/BundleAdjustment = "1"`, `MaxFrames = "8"`: Optimización no lineal continua con $g_2o$.

---

## 6. Evidencia Experimental Comparativa (Medición en Laboratorio)

En ensayos controlados sobre la D415 en reposo sobre la mesa de trabajo a $\sim 40\text{ cm}$:

| Métrica Experimental | Baseline (GFTT / ORB) | Ultra-Precisión (SIFT Calibrado) | Veredicto Científico |
| :--- | :---: | :---: | :--- |
| **Frecuencia Odometría (Hz)** | **30.63 $\pm$ 11.22 Hz** | **29.62 $\pm$ 1.52 Hz** | Ambos operan a los $\sim 30\text{ Hz}$ óptimos. |
| **Tiempo de Cómputo CPU** | **22.05 $\pm$ 0.80 ms** | 29.64 $\pm$ 0.89 ms | GFTT/ORB es un 25% más rápido. |
| **Puntos Clave Detectados** | **908.6 features** | 200.3 features | ORB detecta **4.5x más puntos**. |
| **Inliers Geométricos 3D** | **300.5 inliers** | 0.0 inliers *(rechazado por PnP)* | **GFTT/ORB se ancla firmemente a la trama IR**. |
| **Pérdida de Tracking** | **0.0 % (Rastreo Perfecto)** | 100.0 % *(en esa pose)* | **GFTT/ORB es inmune a pérdida en superficies lisas**. |
| **Jitter Posición 3D ($\sigma_{\text{pos}}$)** | **0.3943 mm** ($< 0.4\text{ mm}$) | $N/A$ | **Precisión submilimétrica real** ($X=0.19, Y=0.23, Z=0.25\text{ mm}$). |
| **Jitter Angular 3D ($\sigma_{\text{rot}}$)** | **0.0467°** ($< 0.05^\circ$) | $N/A$ | **Orientación ultra-estable** sin vibraciones. |

### Justificación Física del Resultado (hipótesis NO verificada; ver §0)
La RealSense D415 proyecta un patrón de micropuntos láser infrarrojos (laser speckle) de alta frecuencia espacial. Los detectores de esquinas locales (GFTT/FAST) se anclan directamente a estos puntos singulares ($1\text{--}2\text{ px}$), generando cientos de inliers 3D. En cambio, SIFT aplica pirámides de diferencias de gaussianas (DoG) que atenúan los detalles microscópicos, resultando en menos candidatos sobre mesas homogéneas.

> **Corrección (2026-11-23):** nadie comprobó que el patrón IR se vea en la imagen RGB, que es la que usa el rastreo.
> Si los puntos se anclaran a él, sería un defecto y no una ventaja: el proyector va fijo a la cámara, así que el
> patrón se mueve con ella y sesga el movimiento estimado hacia cero. Además, el jitter en reposo mide estabilidad,
> no exactitud.

---

## 7. Integración de RViz2 (`vo_vis.rviz`)

Para evitar visores pesados no conformes con Wayland y aprovechar las capacidades de RViz2:
* Archivo de configuración: [visuomotor_bringup/rviz/vo_vis.rviz](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/rviz/vo_vis.rviz) (sincronizado con `.rviz2/vo_vis.rviz`).
* Al ejecutar `realsense_d415_vo.launch.py gui:=true`, se lanza directamente `rviz2 -d ...` bajo aceleración OpenGL 4.6 en XWayland.
* Permite observar en tiempo real:
  - La nube de puntos 3D del mapa local (`/odom_local_map`).
  - La trayectoria cartesiana estimada (`/rtabmap/odom`).
  - El árbol de transformaciones espaciales (`tf`).

---

## 8. Certificación del Primer Dataset Real: `demonstrations_real_v2.zarr`

El 17 de noviembre de 2026 se completó la primera campaña de captura experta manual con la D415:
* **Ubicación:** `datasets/demonstrations_real_v2.zarr`.
* **Volumen:** 7 episodios continuos, $2,909$ pasos totales a $20.0\text{ Hz}$ ($\approx 145.4\text{ s}$ de datos de entrenamiento).
* **Cinemática de Approaching:**
  - `episode_2`: Desplazamiento final $+27.0\text{ cm}$ en $Z$ ($380$ pasos, $3.9\text{ s}$ de hold con `done=1.0`).
  - `episode_3`: Desplazamiento final $+23.6\text{ cm}$ en $Z$ ($356$ pasos, $3.6\text{ s}$ de hold con `done=1.0`).
  - `episode_4`: Desplazamiento final $+16.8\text{ cm}$ en $Z$ ($364$ pasos, $2.7\text{ s}$ de hold con `done=1.0`).
  - `episode_5`: Desplazamiento final $+36.3\text{ cm}$ en $Z$ ($287$ pasos, $2.5\text{ s}$ de hold con `done=1.0`).
  - `episode_6`: Desplazamiento final $+30.6\text{ cm}$ en $Z$ ($432$ pasos, $4.2\text{ s}$ de hold con `done=1.0`).
* **Piso de Término Garantizado:** Todos los episodios superan holgadamente el requisito de $\ge 2.0\text{ s}$ (40 fotogramas) con `done = 1.0` en destino.
* **Archivos Multimedia:** Extraídos con `tools/visualize_dataset.py` a [datasets/visualizations_real_v2/](file:///home/hackbrian/Documents/gits/visuomotor_ros2/datasets/visualizations_real_v2/) (videos `.mp4`, gráficas 4-panel de trayectoria `.png` y `.gif`).

---

## 9. Expansión a Odometría Visual-Inercial (VIO) con Intel RealSense D435i

El 18 de noviembre de 2026 se integró el soporte nativo para la **Intel RealSense D435i** (Product ID `0x0B3A`, Serial `347622074027`, Firmware `5.17.3.10`):

### Características Físicas y de Fusión Inercial
* **IMU Integrado:** Chip Bosch BMI085 (Acelerómetro a 200 Hz y Giroscopio a 200 Hz).
* **Obturador Estéreo:** Global Shutter en los sensores infrarrojos (elimina distorsión por escaneo rápido de líneas).
* **Filtro Madgwick:** Nodo `imu_filter_madgwick` (`use_mag: false`, `world_frame: 'enu'`) calcula el cuaternión de orientación estabilizado en `/imu/data`.
* **Alineación con la Gravedad (Gravity Alignment):** el acelerómetro mide la gravedad y `wait_imu_to_init: true` alinea con ella el marco de odometría **solo al inicializar**. La configuración activa usa `Odom/AlignWithGround: "false"`: una versión anterior de este párrafo decía `true`, lo que contradecía el punto 6 de abajo. Con `OdomF2M/BundleAdjustment: 0` no actúa ninguna restricción de gravedad después del inicio, y en v7 la vertical de odom derivó 50–60° dentro de una sesión (módulo 17 §1). **No es un VIO acoplado:** la pose sale de la imagen y la profundidad.

### Arquitectura de Archivos D435i (Desacoplada y Coexistente con D415)
* **Parámetros de Cámara:** [visuomotor_bringup/config/realsense/d435i_camera_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/d435i_camera_params.yaml) (`unite_imu_method: 2`, `accel_fps: 200`, `gyro_fps: 200`).
* **Parámetros VIO:** [visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml).
* **Launch Files Dedicados:**
  - Inferencia / Visualización: [visuomotor_bringup/launch/realsense_d435i_vio.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d435i_vio.launch.py)
  - Recolección Zarr: [visuomotor_bringup/launch/realsense_d435i_vio_collection.launch.py](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/launch/realsense_d435i_vio_collection.launch.py)
* **Rendimiento Experimental Certificado (Optimizado para Cero Deriva y Máxima Precisión):**
  - **Tiempo de Cómputo CPU:** $13.98 \pm 0.65\text{ ms}$ (apenas el $42\%$ del presupuesto de $33.3\text{ ms}$ a 30 FPS, con $>55\%$ de margen de CPU).
  - **Jitter Posición 3D:** **$0.62\text{ mm}$** con refinamiento subpíxel y filtro Kalman temporal.
  - **Jitter Angular 3D:** **$0.0186^\circ$** ($< 1/50$ de grado).
  - **Pérdida de Tracking:** **$0.0\%$** (seguimiento perfecto e ininterrumpido).
  - **Inliers Geométricos 3D:** $\approx 390\text{--}450$ inliers consistentes.
* **Directivas Algorítmicas Calibradas en `rtabmap_vio_params.yaml`:**
  1. `Odom/GuessMotion: "false"`: Desactivado. Previene el arrastre inercial ficticio al detener la cámara y neutraliza la deriva en Yaw acumulada por el filtro Madgwick (debido a la ausencia física de magnetómetro en el chip BMI085).
  2. `OdomF2M/BundleAdjustment: "0"`: Desactivado. En cámaras RGB-D la profundidad 3D ya está calculada por hardware estéreo; optimizar haces en 2D sin fijar escala causaba deformación continua del mapa local y deriva métrica de escala (*scale drift*).
  3. `Vis/SubPixIterations: "10"` y `Vis/SubPixWinSize: "3"`: Refinamiento de esquinas subpíxel mediante interpolación de gradiente parabólico local que erradica el salto de cuantización de píxeles enteros (suprime el temblor / jitter de alta frecuencia).
  4. `Vis/PnPReprojError: "2.0"` y `Vis/MaxFeatures: "1200"`: Densidad ampliada sobre el FOV gran angular ($86^\circ$) con tolerancia RANSAC balanceada contra el ruido estéreo.
  5. `Odom/FilteringStrategy: "1"`: Filtro Kalman de odometría activo para suavizado inter-cuadro continuo.
  6. `Odom/AlignWithGround: "false"`: Inicia estrictamente en el centro focal del lente de la cámara $(0, 0, 0)$, eliminando traslaciones heurísticas hacia el piso.
* **Correcciones (2026-11-23) a la lista anterior:**
  - El subpíxel activo es 5×5 con 15 iteraciones (§10.D), no 3×3 con 10.
  - `GuessMotion` es un modelo de velocidad constante: no tiene relación con la deriva de yaw del filtro Madgwick.
  - Que el ajuste de haces "deforme la escala" en RGB-D es una hipótesis que se apoya en 1 o 2 corridas por
    configuración, con trayectorias distintas. Se decidirá midiendo contra referencia.
  - Las cifras de cómputo, jitter y 0 % de pérdidas se tomaron con la cámara en reposo.
* **Compatibilidad Inversa:** La cámara D415 conserva sus launch files dedicados (`realsense_d415_vo.launch.py` y `realsense_d415_vo_collection.launch.py`), permitiendo alternar entre ambos sensores sin conflicto.

---

## 10. Diagnóstico de Drift, Prevención de Congestión DDS y Protocolo de Evaluación Métrica

### A. Diagnóstico de la Congestión DDS y Aislamiento de Red
1. **Saturación de la Tarjeta Wi-Fi (`wlo1`):**
   - Cuando se introdujo `cyclonedds.xml` forzando `<NetworkInterface name="wlo1"/>` para comunicar la laptop con la Jetson, CycloneDDS redirigió todo el tráfico a través del adaptador inalámbrico.
   - La RealSense publica Color ($640\times 480\times 3$) + Profundidad ($640\times 480\times 2$) a 30 Hz = **$\approx 46\text{ MB/s}$ ($\approx 370\text{ Mbps}$)** de vídeo crudo.
   - Saturar la antena Wi-Fi con 370 Mbps de paquetes UDP multicast provocaba pérdidas masivas, desincronización de marcas de tiempo y el descarte de imágenes en RTAB-Map (`Dropping image...`).
2. **Solución por Memoria Compartida:**
   - En [run_vo_container.sh](file:///home/hackbrian/Documents/gits/visuomotor_ros2/tools/hardware/run_vo_container.sh) se desacopló `CYCLONEDDS_URI` del Wi-Fi por defecto. Los contenedores operan en `--net=host --ipc=host`, confinando el streaming de 46 MB/s a la memoria RAM / loopback del kernel, con latencia $< 0.1\text{ ms}$ y tasa inter-contenedor certificada de **30.11 Hz exactos**.
3. **Liberación de Buffers IIO:**
   - Script de arranque ejecuta preventivamente `echo 0 > /sys/bus/iio/devices/iio:device*/buffer/enable` tanto al inicio como en el trap de salida, erradicando el error de contención de kernel `EBUSY / HID set_power 1 failed`.

### B. Análisis Empírico del Drift y Jitter en la D435i
* **Causa Raíz del Drift:** Activación simultánea de `BundleAdjustment: 1` y `GuessMotion: true`.
  - El Bundle Adjustment 2D deformaba la escala 3D ante el ruido de profundidad.
  - `GuessMotion` asumía velocidad constante e inyectaba el desvío de rumbo del giroscopio.
* **Causa Raíz del Jitter:** Detección de esquinas a nivel de píxel entero sin refinamiento subpíxel (`SubPixIterations: 0`) o desactivación del filtro de Kalman (`FilteringStrategy: 0`).
* **Estado Calibrado y Verificado:** Con SubPix (10 iter, ventana 3x3), Kalman activado (1), Bundle Adjustment desactivado (0), GuessMotion desactivado (false) y AlignWithGround desactivado (false), la deriva estática en reposo es de apenas **$1.10\text{ mm}$ en 5 segundos** ($0.22\text{ mm/s}$), iniciando rígidamente en $(0,0,0)$ en el lente.
* **Corrección (2026-11-23):** las "causas raíz" de arriba son hipótesis. Se compararon configuraciones con 1 o 2
  corridas y trayectorias distintas, sin referencia, y la deriva en reposo no mide la deriva en movimiento.

### C. Hoja de Ruta Inmediata para Evaluación y Benchmarking de Poses VIO
Para certificar cuantitativamente la fidelidad cinemática de las demostraciones manuales antes de la recolección masiva:
1. **Herramienta de Evaluación Métrica Pareada de VIO (`tools/benchmark_vio_pose.py`):**
   - Script interactivo de benchmarking que graba una trayectoria de bucle cerrado (retorno exacto a una marca física fija o cradle de inicio).
   - Medición automatizada de:
     * **Error de Cierre de Bucle (Loop Closure Drift):** $\|\mathbf{p}_{\text{retorno}} - \mathbf{p}_{\text{inicio}}\|_2$ en milímetros.
     * **Dispersión Estática (Jitter $\sigma_{\text{pos}}$ y $\sigma_{\text{rot}}$):** En posición y cuaternión durante la fase estática inicial y final.
     * **Consistencia de Inliers y FPS:** Tasa sostenida de cálculo y rechazo de falsas correspondencias.

### D. Campaña de retorno a la marca (`conf 4`): mejor caso, NO certificación
El 19 de noviembre de 2026 se compararon 5 configuraciones con 1 o 2 corridas cada una (la conf 4 con 9). Las trayectorias no fueron iguales entre corridas.

> **Corrección (2026-11-23):** los 9 retornos dinámicos de la conf 4 en `experiments/vio_benchmark_history.json`
> fueron 2.3 / 14.4 / 16.2 / 16.9 / 21.6 / 28.3 / 39.3 / 59.0 / 59.7 mm: mediana 21.6 mm, y solo 1 de 9 cumple el
> criterio ≤ 3 mm (la trayectoria más corta, 279 cm). El jitter en reposo tiene mediana 3.0 mm (1.6–6.1). El error de
> retorno tampoco mide la trayectoria ni la escala: un factor de escala deja el punto de retorno donde está, y F2M con
> `OdomF2M/MaxSize: 5000` se relocaliza contra los puntos de t=0. Lo que sigue describe el mejor intento.

* **Mejor intento (no representativo):** la corrida de `conf 4` que cumplió el criterio de cierre de lazo ($\le 3.00\text{ mm}$):
  - **Error de Retorno (Loop Closure):** **$2.307\text{ mm}$** tras una excursión máxima de **$54.56\text{ cm}$** y un recorrido total de **$2.79\text{ metros}$**.
  - **Desglose Cartesiano Milimétrico:** $\Delta X = -0.828\text{ mm}$, $\Delta Y = -0.064\text{ mm}$, $\Delta Z = +2.153\text{ mm}$.
  - **Jitter Estático en Reposo:** **$1.6144\text{ mm}$** (reducción del $89\%$ respecto al baseline descalibrado de $14.89\text{ mm}$).
  - **Error Angular de Retorno:** $2.562^\circ$ ($\Delta\text{Pitch} = +0.60^\circ$).
  - **Inliers y Fiabilidad:** $258.8$ inliers sostenidos con **$0.0\%$ de pérdida de tracking**.
* **Directivas Algorítmicas SOTA Clave:**
  1. `Vis/MinDepth: 0.20` y `Vis/MaxDepth: 1.80`: Acotamiento estricto al volumen de trabajo de manipulación, eliminando el ruido de fondo distante ($> 1.8\text{ m}$) que deformaba la escala métrica.
  2. `OdomF2M/MaxSize: 5000`: Expansión de la memoria de mapa local para garantizar el reconocimiento visual y cierre de bucle exacto contra el cradle / marca de origen.
  3. `Vis/MinInliers: 20` y `Vis/PnPReprojError: 1.8`: Criterio estricto de rechazo de correspondencias periféricas dudosas.
  4. `Vis/SubPixWinSize: 5` y `Vis/SubPixIterations: 15`: Ventana ampliada de gradiente continuo que erradica el salto de píxeles discretos.
  5. `madgwick_gain: 0.035`, `zeta: 0.002` y `orientation_stddev: 0.02`: Supresión del ruido del acelerómetro, compensación continua en línea del sesgo del giróscopo y ponderación Bayesiana consistente de covarianza inercial.
* **Archivos Congelados:** Configuración activa en [visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml) y copia de seguridad certificada en [visuomotor_bringup/config/realsense/rtabmap_vio_params_best_umi.yaml](file:///home/hackbrian/Documents/gits/visuomotor_ros2/visuomotor_bringup/config/realsense/rtabmap_vio_params_best_umi.yaml).

---

## 11. Recolección Masiva de Demostraciones Expertas Reales (`demonstrations_real_v3.zarr`)
~~Con el sistema VIO formalmente certificado por debajo de $2.5\text{ mm}$ de precisión de retorno y anclado gravitacionalmente mediante la IMU, el pipeline está listo para la captura interactiva a 20.0 Hz.~~ Corregido el 2026-11-23: el sistema no estaba certificado (§10.D y §0).

---

## E. Protocolo de Recolección de Demostraciones UMI y Parámetro `dataset_path`

### 1. Alineación Matemática con UMI (Universal Manipulation Interface) y Diffusion Policy
El recolector `vo_collector_node` reproduce fielmente el estándar UMI (Chi et al., RSS 2024 / Stanford / TRI) y es 100% compatible con la arquitectura de Diffusion Policy del repositorio:
1. **Frecuencia Canónica:** 20.0 Hz sincronizados con timer de ROS 2.
2. **Observación Propioceptiva (`observation.state`) — 8D:**
   - $[X, Y, Z]$ traslación cartesiana relativa al punto de inicio del episodio ($t=0 \to (0, 0, 0)$).
   - $[q_x, q_y, q_z, q_w]$ cuaternión absoluto de orientación con estabilización de semiesfera superior ($q_w \ge 0$) y norma unitaria.
   - Gripper físico en $0.0$.
   - **Regla Inviolable:** CERO fuga causal: la señal `done` nunca forma parte de `observation.state`.
3. **Espacio de Acciones (`action`) — 9D:**
   - $[X, Y, Z]$ traslaciones relativas objetivo a predecir ($H_{act} = 16$).
   - $[q_x, q_y, q_z, q_w]$ orientaciones relativas objetivo.
   - Gripper objetivo.
   - `done_flag` (9ª dimensión): $0.0$ durante trayectoria, $1.0$ durante la fase de llegada/asentamiento (`HOLD_TARGET`), garantizando un piso de $\ge 2.0\text{ s}$ (40 frames a 20 Hz) con `done=1.0`.
4. **Almacenamiento Zarr:** Grupos indexados `episode_0`, `episode_1`, etc., con datasets `image` (uint8, $256 \times 256 \times 3$), `state` (float32, $N \times 8$) y `action` (float32, $N \times 9$).

### 2. Parámetro de Lanzamiento Flexible: Auto-Incremento vs Aumento de Dataset
El nodo y los archivos launch aceptan el parámetro `dataset_path`:
* **Modo AUTO (`dataset_path:=AUTO` o sin definir):**
  - Escanea `datasets/` buscando patrones `demonstrations_real_v*.zarr`.
  - Determina automáticamente la versión máxima existente ($v_{\max}$) y crea una nueva carpeta incremental: `datasets/demonstrations_real_v{max+1}.zarr`.
* **Modo AUMENTO / CONTINUACIÓN (`dataset_path:=<nombre_o_ruta>`):**
  - Si se proporciona un nombre (ej. `demonstrations_real_v5` o `demonstrations_real_v5.zarr` o `datasets/demonstrations_real_v5.zarr`), normaliza la ruta hacia `datasets/`.
  - Si el dataset ya existe, detecta el número de episodios previos existentes (ej. 8 episodios), emite un log informativo y anexa los nuevos episodios consecutivamente a partir de `episode_8`.

### 3. Ejecución Rápida
```bash
# Modo 1: Crear automáticamente un nuevo dataset incremental demonstrations_real_vX.zarr
./tools/hardware/run_data_collection.sh

# Modo 2: Continuar aumentando un dataset específico existente
./tools/hardware/run_data_collection.sh demonstrations_real_v5
```

---

## F. Topología de Red DDS: Aislamiento Local (Zero-Latency) vs. Conexión Remota con Jetson Orin Nano

### 1. Régimen de Recolección Local de Demostraciones (Laptop / Workstation)
* **Reto de Ancho de Banda:** El pipeline de la cámara Intel RealSense D435i (640x480 a 30 FPS RGB + Depth alineado + IMU a 200 Hz) genera un flujo continuo de aproximadamente **$46\text{ MB/s}$ ($370\text{ Mbps}$)** en el bus DDS de ROS 2.
* **Aislamiento en Memoria Compartida (SHM) y Localhost Estricto:**
  - Si CycloneDDS hace multicast en la interfaz Wi-Fi (`wlo1`), el canal inalámbrico colapsa, provocando pérdidas de paquetes, caídas de frecuencia a $< 5\text{ Hz}$ y jitter visual.
  - Además, si otros equipos en la misma red de laboratorio tienen instancias de ROS 2 corriendo (por defecto en `ROS_DOMAIN_ID=0`), se produciría descubrimiento cruzado, colisión de tópicos (`/tf`, `/camera/...`) y tormentas de tráfico.
  - Solución obligatoria inyectada en `run_vo_container.sh`, `run_benchmark.sh` y `run_data_collection.sh`:
    `ROS_LOCALHOST_ONLY=1` y `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` junto a `ROS_DOMAIN_ID=0`.
  - Esto blinda la máquina al 100%: ningún paquete sale hacia la Wi-Fi/LAN y ningún equipo externo puede interferir.
* **Resultado:** sin saturación de Wi-Fi e inmune a otros nodos ROS 2 de la red, con $13.98 \pm 0.65\text{ ms}$ de cómputo por cuadro. El "0 % de pérdida de rastreo y 20.0 Hz sostenidos" no se cumplió en v7: hubo poses nulas, y sin sellos de tiempo el muestreo no se puede verificar (§0).

### 2. Régimen de Conexión Remota con Robot Real (Jetson Orin Nano)
* **Desacoplamiento de Carga:** Cuando se ejecute la inferencia o teleoperación remota con el xArm5 físico conectado a la Jetson Orin Nano en la misma red de laboratorio:
  - **Dominio Dedicado:** El hardware del robot y sus controladores operan bajo **`ROS_DOMAIN_ID=42`**.
  - **Canal de Control Ligero:** La estación de trabajo que corre la red neuronal predice trayectorias y transmite únicamente comandos de control (`ActionChunk`, `/target_frame`, `/policy/metrics`). Este tráfico es extremadamente ligero ($< 50\text{ KB/s}$ a 20 Hz).
  - **Enrutamiento CycloneDDS Unicast:** La comunicación inter-máquinas se gestiona mediante unicast peer-to-peer (`deploy/jetson/config/cyclonedds.xml` con `DDS_PEER=<ip_jetson>` o `DDS_IFACE=wlan0`), evitando saturar la red local.
* **Separación de Responsabilidades:**
  - Scripts de recolección local (`run_data_collection.sh`, `vo_collector`): operan autónomos y aislados con la cámara conectada directamente por USB.
  - Scripts de puente e inferencia (`run_remote_rviz.sh`, `xarm5_bridge.py`, `policy_node.py`): operan sobre la red distribuida hacia la Jetson Orin sin interferencias de streams de video pesado.

---

## 0. Revisión crítica y evidencia (2026-11-23)

Herramienta: `tools/xarm5/analyze_vio_dataset_quality.py --raw-diagnostics` (salida en
`datasets/evaluations_real_v7_xarm5_v2/raw_diagnostics.{json,md}` y `fig4_raw_recording_diagnostics.{png,pdf}`).
Pruebas: `tests/test_vio_raw_diagnostics.py`.

| Afirmación anterior | Evidencia | Estado |
|---|---|---|
| "Certificación grado UMI: retorno 2.307 mm" | 9 corridas dinámicas de la conf 4: mediana 21.6 mm, máx. 59.7 mm, 1/9 ≤ 3 mm (la más corta) | Mejor caso, no certificación |
| El retorno a la marca mide precisión | No es sensible a la escala, y F2M se relocaliza contra los puntos de t=0 | Criterio inválido; hace falta ATE/RPE/escala contra referencia |
| "VIO; Z estrictamente vertical" | El IMU solo inicializa la gravedad; no hay restricción inercial con BA = 0; la vertical derivó 50–60° | Odometría RGB-D con marco inicial alineado a la gravedad |
| "0 % de pérdida de rastreo" | v7: 4 frames con pose NULA (ep. 23 y 24) guardados como identidad y -pose inicial (salto de 0.51 m) | Falso en la toma real |
| "Ruido VIO ~8.8 mm RMS" (módulo 17) | 926 saltos > 3 cm entre frames; 784 discontinuidades persistentes > 2 cm (mediana 33 mm, p90 55 mm) en 71/83 episodios | Saltos discretos, no ruido gaussiano; el filtro de 1.5 Hz los suaviza como movimiento falso |
| "20 Hz canónicos sincrónicos" | Sin sellos de tiempo por frame; `verify_dataset_frequency.py` asume dt = 0.05; el timer toma la última imagen y la última pose por separado | No verificable (4 estados y 0 imágenes consecutivas duplicadas) |
| "Piso de done garantizado" | 72/83 episodios tienen done solo en el piso forzado de 40 frames; 21 % de esos frames > 2 cm/s (filtrado; el ruido filtrado ronda 1 cm/s) | Etiquetas con ruido |
| `min_inliers_tracking: 10` | Se declara y no se usa en `vo_collector_node.py` | Sin control de rastreo al grabar |
| "GFTT/ORB se ancla a la trama IR" | Sin verificar; el rastreo usa la imagen RGB (rolling shutter) | Hipótesis; si fuera cierta, sería un defecto |

### Herramientas de la mejora (2026-11-23)
* Diagnóstico de grabaciones crudas: `analyze_vio_dataset_quality.py --raw-diagnostics` (`tests/test_vio_raw_diagnostics.py`).
* Métricas contra referencia: `visuomotor_core/analysis/trajectory_metrics.py`
  (`tests/test_trajectory_metrics.py`). Calcula:
  - ATE anclado al primer frame, ATE SE(3)/Sim(3) con el error de escala, RPE a 0.25/0.5/1 s y el desfase temporal;
  - `summarize_runs`, que da mediana, p90, máximo y n, nunca el mejor intento.
  Una prueba demuestra que el retorno a la marca no ve un 5 % de error de escala.
* Tablero ChArUco: `visuomotor_bringup/charuco.py` + `config/realsense/charuco_board.yaml` +
  `tools/hardware/make_charuco_board.py` (`tests/test_charuco.py`, con una vista sintética de pose conocida).
* Calibración cámara-brida: `tools/xarm5/calibrate_hand_eye.py` (`tests/test_hand_eye.py`). Mide la rotación y la
  traslación; hasta ahora la traslación se suponía [0, 0, 0].
* Recolector v2: `vo_collector_node.py` + `vo_episode_logic.py` + `vo_raw_recorder.py` (`tests/test_vo_episode_logic.py`,
  `tests/test_vo_collector_node_sim.py` con cámara y odometría simuladas en ROS 2).

Regla: una cifra de precisión de la herramienta se reporta como distribución (mediana, p90, máximo, n) medida contra
una referencia (xArm5 o tablero ChArUco), nunca como el mejor intento. Plan: `docs/plans/plan_d435i_herramienta_demostraciones.md`.

### Medición sin el robot y recolector v3 (2026-11-24)
* **Tamaño del tablero, decidido con datos:** aproximación sintética con verdad conocida
  (`tools/hardware/make_synthetic_vio_bag.py`: plano texturado y tablero, profundidad exacta, IMU y extrínseca
  típica). Error de posición de la referencia, mediana en mm:

  | Tablero | < 0,45 m | 0,45–0,6 m | 0,6–0,75 m | 0,75–1,2 m (HOME3) |
  |---|---|---|---|---|
  | A4 7×5, 33 mm | 0,35 | 1,2 | 2,3 | 14,0 |
  | A4 9×5, 30 mm | 0,33 | 0,9 | 2,3 | 6,1 |
  | A3 9×7, 36 mm | 0,11 | 0,4 | 0,7 | 1,7 |

  - Lejos, la orientación del tablero chico sale ~1° errada con PnP iterativo e IPPE por igual. Al anclar en ese
    frame, el error se propaga (~9 mm a 0,5 m de recorrido).
  - Más esquinas reducen el error más que cuadros más grandes.
* **Herramientas:**
  - `vio_bag_utils.py`: lectura de MCAP, imágenes (incluida la compressedDepth), TF estáticas y odometría nula.
  - `evaluate_pose_estimates.py`: referencia del tablero, pares por sello y `evaluate_paired`, con huecos y RPE por
    tiempo.
  - `replay_odometry_candidates.py`: reproducción con reloj simulado a 0,5×, `republish`, Madgwick y
    `rgbd_odometry` con los parámetros del candidato, en el dominio DDS 88.
  - `record_vio_bag.sh` y `read_camera_auto_settings.py`.
  - Prueba de punta a punta: `tests/test_vio_measurement_pipeline.py`. Recupera la escala (+3,00 %) y las pérdidas
    inyectadas. RTAB-Map base en la escena sintética: 4,5 mm de mediana.
* **Recolector v3:** guía de inicio a la vista de HOME3 (ArUco de la mesa), inicio y guardado automáticos, aviso de
  velocidad y programa de celdas × orientaciones (skill `realsense-vio-data-collection` §4).
* **Corregido:**
  - Caída del ejecutor del recolector v2 por un log que cambia de severidad en la misma línea (`ValueError` de
    rclpy).
  - `/tf_static` ausente en los bags crudos (QoS volátil).
  - `run_data_collection.sh` no cargaba el YAML.
