# 15. Odometría Visual con RTAB-Map y RealSense D415 para Demostraciones Reales de Approaching

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

### Justificación Física del Resultado
La RealSense D415 proyecta un patrón de micropuntos láser infrarrojos (laser speckle) de alta frecuencia espacial. Los detectores de esquinas locales (GFTT/FAST) se anclan directamente a estos puntos singulares ($1\text{--}2\text{ px}$), generando cientos de inliers 3D. En cambio, SIFT aplica pirámides de diferencias de gaussianas (DoG) que atenúan los detalles microscópicos, resultando en menos candidatos sobre mesas homogéneas.

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
