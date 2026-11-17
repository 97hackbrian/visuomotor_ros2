# 15. Odometría Visual con RTAB-Map y RealSense D415 para Demostraciones Reales de Approaching

Este módulo documenta la arquitectura técnica, integración de hardware y software, parámetros algorítmicos y protocolo experimental para la captura de demostraciones expertas reales de aproximación (*approaching*) mediante **Odometría Visual (VO)** sobre **ROS 2 Jazzy** en un contenedor Docker con aceleración de hardware.

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
  └─► Publica: /rtabmap/odom (nav_msgs/msg/Odometry a ~57 Hz)
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

1. **`visuomotor_bringup` (Configuraciones de Hardware y Launch):**
   - `config/realsense/d415_camera_params.yaml`: Configura perfiles a `640x480x30`, activa alineación de profundidad (`align_depth.enable: true`) y enciende el proyector IR (`depth_module.emitter_enabled: 1`).
   - `config/realsense/rtabmap_vo_params.yaml`: Parámetros de odometría calibrados para mínima deriva y supresión de auto-reinicio.
   - `launch/realsense_d415_vo.launch.py`: Driver + Odometría (+ Visor opcional `gui:=true`).
   - `launch/realsense_d415_vo_collection.launch.py`: Orquestador maestro que levanta cámara, VO y nodo recolector.
2. **`visuomotor_data_collection` (Captura y Almacenamiento):**
   - `config/vo_data_collection_params.yaml`: Parámetros de tasa de muestreo (20 Hz), tópicos y prefijo `demonstrations_real`.
   - `visuomotor_data_collection/vo_collector_node.py`: Nodo de recolección interactiva por teclado con FSM y garantía de piso de parada.
3. **`tools/hardware/` (Automatización en Host):**
   - `run_vo_container.sh`: Lanzador Docker con paso directo dinámico de `/dev` y `/dev/bus/usb`, variables de render XWayland y CycloneDDS.
   - `setup_realsense_udev.sh`: Reglas udev de permisos en el host para dispositivos RealSense.

---

## 4. Análisis Comparativo: RTAB-Map RGB-D vs. NVIDIA cuVSLAM

| Dimensión Técnica | RTAB-Map (`rgbd_odometry`) | NVIDIA cuVSLAM (`isaac_ros_visual_slam`) |
| :--- | :--- | :--- |
| **Backend de Cómputo** | CPU Multihilo optimizada (C++) | GPU Masiva (CUDA / Tensor Cores) |
| **Entrada de Datos** | **RGB + Mapa de Profundidad ASIC** | **Par Estéreo Infrarrojo Crudo + IMU** |
| **Dependencia de IMU** | **Nula** (opera con alta precisión sólo con RGB-D) | **Crítica** (requiere fusión VIO para estabilidad) |
| **Sensibilidad a Rolling Shutter** | Baja/Media (mitigada por matching 3D denso) | **Muy Alta** (diseñado para Global Shutter) |
| **Compatibilidad con D415** | **Nativa y Oficial** | **No soportada oficialmente por NVIDIA** |
| **Frecuencia en i9 / RTX 4060** | ~57 Hz (actualización en 13–22 ms) | 60–120+ Hz (actualización en < 5 ms) |
| **Idoneidad para Approaching** | **Excelente** (alta densidad y menor deriva a 20–50 cm) | Pobre con D415 (pérdida de escala y drift sin IMU) |

**Conclusión:** Para la cámara Intel RealSense D415, RTAB-Map RGB-D es superior técnicamente debido a que aprovecha el proyector de textura IR y el cómputo de disparidad en silicio del ASIC de Intel, mientras que cuVSLAM sufre inestabilidades severas al carecer de IMU física y estar sujeto a distorsiones de rolling shutter.

---

## 5. Estrategias Algorítmicas de Odometría en RTAB-Map

### Configuración Equilibrada (Baseline)
* `Odom/Strategy = "0"`: **Frame-to-Map (F2M)**. El cuadro actual se alinea contra un mapa tridimensional local de keyframes recientes, suprimiendo la deriva acumulativa de los enfoques Frame-to-Frame.
* `Vis/FeatureType = "8"`: **ORB (Oriented FAST and Rotated BRIEF)**. Extracción rápida y binaria de hasta 1000 puntos clave.
* `Vis/EstimationType = "1"`: **PnP 3D-2D**. Minimiza el error de reproyección en el plano de la imagen mediante RANSAC.
* `Odom/FilteringStrategy = "1"`: **Filtro de Kalman**. Suavizado continuo de coordenadas y orientaciones.
* `Odom/ResetCountdown = "0"`: **Protección contra Pérdida de Origen**. Si el tracking se pierde, congela la odometría y no resetea las coordenadas a $(0,0,0)$.

### Configuración Ultra-Precisión (High-Precision Tuning para i9)
Dado que el procesador i9-13950HX posee 32 hilos y una latencia ociosa significativa, es factible elevar la exactitud métrica al máximo:
1. **SIFT (`Vis/FeatureType: 1`):** Invarianza analítica a cambios continuos de escala (zoom óptico durante el acercamiento al objeto).
2. **Refinamiento Subpíxel (`Vis/SubPixIterations: 10`, `Vis/SubPixWinSize: 3`):** Interpola matemáticamente las esquinas con precisión de $\sim 0.05$ píxeles, reduciendo el ruido angular a $\pm 0.2^\circ$.
3. **PnP Estricto (`Vis/PnPReprojError: 1.2`, `Vis/MinInliers: 20`):** Exige mayor consistencia geométrica antes de aceptar la transformación.
4. **Local Bundle Adjustment (`OdomF2M/BundleAdjustment: 1`, `MaxFrames: 15`):** Optimización continua con g2o sobre las últimas 15 poses.

---

## 6. Protocolo Operativo de Captura de Demostraciones

Para garantizar que los datos recolectados cumplan con los estándares de **Diffusion Policy**:

1. **Inicio de Episodio (`[ESPACIO]`):**
   - Posicionar la cámara en la pose de partida (hover/aproximación a 40–50 cm).
   - Verificar inliers verdes en el visor. Al pulsar `[ESPACIO]`, la pose actual se fija como $(0,0,0, q_{\text{start}})$.
2. **Trayectoria de Aproximación:**
   - Desplazar la cámara de manera suave y continua hacia el objetivo. Evitar movimientos balísticos rápidos.
3. **Llegada y Retención de Destino (`[H]`):**
   - Al alcanzar la distancia final (respetando $Z_{\min} \ge 20\text{ cm}$), presionar **`[H]`**.
   - **Regla Inviolable:** Mantener la cámara completamente inmóvil durante **2 a 3 segundos**. Esto inyecta $40$ frames a 20 Hz con `done = 1.0` y velocidad cero, enseñando a la red a desacelerar y detenerse.
4. **Consolidación (`[S]`) o Descarte (`[D]`):**
   - Presionar **`[S]`** para guardar atómicamente en el dataset Zarr.
   - Si hubo vibraciones excesivas o advertencias de tracking, presionar **`[D]`** para limpiar el buffer.
