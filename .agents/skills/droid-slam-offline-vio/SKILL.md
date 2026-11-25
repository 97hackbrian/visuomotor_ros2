---
name: droid-slam-offline-vio
description: Calcular OFFLINE las poses de la cámara D435i con DROID-SLAM RGB-D (el estimador elegido contra el xArm5 el 2026-11-25) a partir de bags MCAP de demostraciones o de corridas de medición, y evaluarlas contra el robot. Usar al construir datasets reales (v8 en adelante), al reprocesar bags, al medir la odometría con el robot como referencia o al comparar un estimador nuevo.
---

# Skill: poses offline con DROID-SLAM RGB-D (v1.0, 2026-11-25)

DROID-SLAM RGB-D (Teed y Deng, NeurIPS 2021) es el estimador elegido para las **poses del dataset**. Medido contra el
xArm5 en 8 aproximaciones de v7:
- mediana de 10,9 mm, p90 14,7 y máximo 17,0;
- 0 % de pérdidas y 100 % de frames con error ≤ 20 mm;
- con la profundidad corregida ×0,986: 7,9 / 11,5 / 15,0 mm.

La configuración de v7 daba 21,7 / 58,9 / 95,4 mm. Decisión, argumento práctico y teórico, y todos los números:
`.agents/knowledge/18_vio_estimator_selection_droid_slam.md` y `context.md`, lecciones 47 y 48.

**Reparto de roles:**
- **En vivo:** RTAB-Map (`OdomF2M/BundleAdjustment: "1"`) solo guía al operador y alimenta el control de calidad.
- **Poses del dataset:** se calculan aquí, offline, desde los bags crudos.

---

## 1. Cuándo usarla
- Construir o reconstruir un dataset real a partir de los bags crudos del recolector v3 (`raw_bag_enabled: true`).
- Reprocesar bags antiguos con un estimador mejor.
- Medir la odometría contra el robot (`tools/xarm5/vio_gt/`) o comparar un framework nuevo con las mismas reglas.

## 2. Requisitos
- **Imagen Docker** `visuomotor_learned_vo:v1`, independiente del resto:
  `docker build -t visuomotor_learned_vo:v1 tools/xarm5/vio_gt/frameworks/learned`.
  - La compilación de lietorch y droid_backends tarda ~20 min.
  - Los pesos se descargan de Google Drive (`droid.pth`) y Dropbox (`dpvo.pth`) con los enlaces del Dockerfile.
- **GPU NVIDIA con ≥ 4 GB libres:** DROID usa 3,9 GB a 640×480 reescalado a ~384×512. **No correrlo junto a un
  entrenamiento** (regla de un trabajo pesado a la vez).
- **Bag MCAP** con color comprimido (JPEG), `aligned_depth_to_color/compressedDepth`, `color/camera_info`,
  `/camera/camera/imu` y `/tf_static`. Son los tópicos de `vio_measurement.yaml` y del recolector v3.
- **Todo dentro de Docker,** como UID 1000.

## 3. Procedimiento
```bash
# 1) Bag -> carpeta tipo TUM (JPEG/PNG sin recomprimir, IMU, calibración de fábrica). Se niega a sobrescribir.
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && \
  python3 tools/xarm5/vio_gt/export_bag_frames.py --bag datasets/vio_bags/<bag> --out datasets/vio_frames/<nombre>"

# 2) DROID-SLAM RGB-D (filter_thresh 1.0 OBLIGATORIO: con el 2.4 por defecto falla si el episodio empieza quieto)
docker run --rm --gpus all -u 1000 -e HOME=/tmp -v "$PWD":/ros -w /ros visuomotor_learned_vo:v1 \
  python3 tools/xarm5/vio_gt/frameworks/learned/run_droid.py datasets/vio_frames/<nombre> <salida>.tum --filter-thresh 1.0

# 3) (Con referencia del robot) evaluación con las reglas comunes; --scale-factor aplica la corrección de profundidad
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && \
  python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --run <corrida> --hand-eye <static>/hand_eye.json \
  --traj-tum <salida>.tum --traj-frame optical --frame-times datasets/vio_frames/<nombre>/rgb.txt \
  --scale-factor 0.986 --label droid_rgbd_ft1_depthcorr"
```
**Salida:** TUM (`t x y z qx qy qz qw`), la pose de la cámara de COLOR (marco óptico) en el mundo del primer
keyframe, una pose por frame. El tiempo es el sello del bag. Para `camera_link`:
`T_world_link = T_world_opt · inv(T_link_opt)`, con `T_link_opt` de `calib.json`.

### 3b. Anclaje con la pose REAL de inicio (ArUco + IMU)
DROID da el movimiento RELATIVO dentro del episodio. Para anclarlo en `link_base` no hay que suponer que la demo
empezó exactamente en HOME3: con la pose medida al inicio, la inclinación baja de ~7° ± 6° (v7) a ~0,6°, y la
posición queda en ~1 cm. Validado contra el robot.
```bash
# Pose de inicio de cada episodio (exporta los bags crudos si hace falta) -> JSON para el conversor
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && \
  python3 tools/xarm5/estimate_start_poses.py --zarr datasets/demonstrations_real_v8.zarr --out <start_poses.json>"
python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src <dataset> --dst <nuevo> --drop-flagged --start-poses <start_poses.json>
```
- **Requisitos:**
  - `visuomotor_bringup/config/realsense/table_markers.yaml`: esquinas de los ArUco en la base y vertical real,
    calibradas con el robot (`tools/xarm5/vio_gt/calibrate_table_markers.py`). Recalibrar si se mueven los
    marcadores o el robot respecto de la mesa.
  - Tras el pitido de inicio, quedarse quieto ~0,5 s: la ventana quieta da los frames y el acelerómetro.
- **Fallback:** episodios sin marcadores, o con PnP e IMU discrepando > 3°, quedan con `usable: false` y se anclan en
  HOME (con aviso).

## 4. Parámetros
| Parámetro | Valor | Motivo |
|---|---|---|
| `--filter-thresh` | **1,0** (por defecto 2,4) | Con 2,4, la corrida 51 (arranque quieto en HOME3) falla de forma determinista: 620 mm. Con 1,0 queda en 7,1 mm, y las otras 7 no empeoran |
| `--warmup` | 8 | Valor de `demo.py` de DROID-SLAM |
| `--buffer` | 512 | Suficiente para demos de ≤ 4000 frames con movimiento lento |
| Factor de profundidad | **×0,986** (escala de la trayectoria) | La D435i sobreestima el movimiento un +1,05 a +1,83 % (todos los métodos RGB-D lo muestran). Validado dejando una corrida fuera. **Medirlo de nuevo si se recalibra la cámara** |
| Imagen | ~384×512, múltiplos de 8 | Preprocesado de `demo.py`; intrínsecas escaladas y profundidad con vecino más cercano |

## 5. Errores conocidos y comprobaciones
- **Arranque quieto:** sin `--filter-thresh 1.0` los primeros segundos pueden salir como basura y todo el episodio
  queda corrido.
  - Comprobación barata: los primeros ~2 s (cámara quieta en HOME3) deben moverse < 5 mm.
- **Sesgo de escala:** sin la corrección queda un error de escala de ~+1,4 %, unos 8 mm en 0,6 m.
- **Solo offline:** ~1–1,7 veces el tiempo real en la RTX 4060. Para el operador en vivo se usa RTAB-Map.
- **DPVO no sirve como alternativa:** es monocular (sin escala) y pide > 8 GB de GPU en secuencias de > 2000 frames.
- **ORB-SLAM3 no sirve como respaldo en esta escena:** se pierde cerca de la mesa y abre otro mapa. Sus modos
  inerciales no inicializan con movimiento lento.
- **Pendiente:** control de calidad automático que compare la rotación de DROID con el giróscopo integrado de cada
  episodio, y `build_dataset_from_bags.py`, que integre esta skill en la construcción del dataset.

## 6. Validación pendiente (antes de declararlo definitivo)
- Datos nuevos (el parámetro se eligió con las mismas corridas).
- Velocidad de mano: 2–3 veces la del robot.
- Herramienta en la mesa.
- Factor de profundidad después de la calibración en chip y el *tare* de la D435i.
- Próxima sesión con el robot: grabar también el IR estéreo para OpenVINS, Basalt y ORB-SLAM3 estéreo-inercial.
