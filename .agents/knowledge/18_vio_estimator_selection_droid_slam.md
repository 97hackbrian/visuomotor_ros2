# 18. Elección del estimador de pose de la cámara: DROID-SLAM RGB-D offline (2026-11-24/25)

Registro de la decisión y de todos los experimentos que la sostienen, con el nivel de detalle necesario para el
paper. Complementa `15_visual_odometry_rtabmap_realsense.md` (RTAB-Map en vivo) y `17_real_vio_dataset_curation_and_
relative_actions.md` (curación de v7). Resumen operativo: skill `.agents/skills/droid-slam-offline-vio/SKILL.md`.
Lecciones 47 y 48 de `context.md`.

---

## 1. Decisión

| Uso | Estimador | Por qué |
|---|---|---|
| **Poses del dataset (offline)** | **DROID-SLAM RGB-D** (Teed y Deng, 2021), pesos públicos `droid.pth`, `filter_thresh` = 1,0, profundidad corregida ×0,986 | Menor error y sin pérdidas en las 8 aproximaciones medidas contra el robot (§3) |
| Guía y control de calidad en vivo | RTAB-Map `rgbd_odometry` con `OdomF2M/BundleAdjustment: "1"` | Corre en tiempo real en el portátil; mejor configuración de RTAB-Map (lección 47) |

**Coherencia con la arquitectura del plan:** la odometría en vivo solo guía al operador y alimenta el control de
calidad. Las poses del dataset se calculan offline desde la grabación cruda de cada episodio (bags MCAP), así que un
método con GPU que no corre en tiempo real es admisible. Además, se puede reprocesar si mañana aparece uno mejor.

---

## 2. Protocolo experimental (para el paper)

### 2.1 Referencia (ground truth)
- **Robot:** el xArm5 (UFACTORY, 5 GDL) con la D435i en la brida ejecuta planes validados con la caja y el alcance
  del puente (`tools/xarm5/vio_gt/plan_vio_gt.py`).
  - Velocidad ≤ 3,5 cm/s y ≤ 0,3 rad/s.
  - Paradas de 3 s al inicio y al final, y regreso a HOME3.
- **Registro del TCP:** `link_base → xarm_tcp` en la Jetson, ~100 Hz (`jetson/vio_gt_motion.py`), con vigilante
  (piso z ≥ 0,15 m, seguimiento ≤ 60 mm, errores del puente) y desarme sin frenos al final.
- **Relojes:**
  - Desfase PC–Jetson tomado por ssh con `b − j`. El punto medio de la ida y vuelta tenía un sesgo de ~0,35 s,
    porque el saludo SSH dura ~0,7 s.
  - Refinado por correlación de la rapidez angular del giróscopo de la cámara con la del robot (r = 0,63–0,99).
- **Cámara-brida** (`evaluate_vio_gt.py`, `cv2.calibrateHandEye`):
  - 13 paradas frente a un ChArUco A3 9×7 de 36 mm medido (324 mm los 9 cuadros), con 80° de diversidad de giro.
  - Residuo de 3,1 mm / 0,27°.
  - `camera_link` queda a (62,9, −16,3, 18,5) mm del TCP.
  - Incertidumbre de la traslación ~9 mm: con la escala del tablero fija, la solución se mueve 8,6 mm y 0,41°.
- **Corridas** (`experiments/vio_gt/2026-11-24/runs/`):
  - 1 de paradas (`static_board`).
  - 8 aproximaciones de demos v7 escaladas en el tiempo: episodios 2, 3, 18, 48, 51, 61, 68 y 73, de 0,77 a 1,57 m
    de recorrido, 57–113 s, con destinos repartidos lateralmente (y de −0,30 a +0,26 m).
  - Mesa con mantel blanco y dos ArUco. Sin herramienta. El tablero queda en el fondo (fuera de la vista final).

### 2.2 Datos
- **Bags MCAP:** `datasets/vio_bags/` (5,5 GB; git los ignora).
  - Color JPEG 640×480 a 30 Hz y profundidad alineada en PNG de 16 bits.
  - IMU unificado a 200 Hz, giróscopo y acelerómetro crudos.
  - Odometría en vivo, `odom_info` y `tf_static`.
- **Export TUM para los otros frameworks:** `datasets/vio_frames/<corrida>/` (`tools/xarm5/vio_gt/export_bag_frames.py`;
  sin recomprimir).

### 2.3 Métricas (`visuomotor_core/analysis/trajectory_metrics.py::evaluate_paired`)
- **Pares:** cada pose estimada se compara con la del robot interpolada al mismo instante. La pose de la cámara según
  el robot es `T_base_tcp · T_tcp_camera_link`.
- **ATE anclada al primer frame:** es como el pipeline usa las poses (movimiento relativo dentro del episodio). Se
  reporta en mm: mediana, p90 y máximo por corrida.
- **ATE con escala libre (Sim(3)):** de ahí sale el error de escala.
- **RPE a 0,25, 0,5 y 1 s, y porcentaje de pérdidas.**
- **Cobertura útil:** % de TODOS los frames, perdidos incluidos, con pose y error ≤ 10/20 mm. La ATE sola favorece al
  método que se pierde más.
- **Agregado:** mediana entre corridas, nunca el mejor caso, más la peor corrida.

### 2.4 Candidatos y ejecución
Cada framework corre en su propia imagen Docker y sobre los mismos datos:
- **RTAB-Map en vivo:** configuración de la grabación de v7.
- **RTAB-Map reprocesado** (`replay_odometry_candidates.py`, ×0,5): base, ajuste de haces, GuessMotion, flujo óptico,
  frame-to-frame y sin Kalman.
- **ORB-SLAM3 v1.0** (`frameworks/orbslam3`; RGB-D, RGB-D-inercial y monocular-inercial): ruidos del IMU del ejemplo
  oficial de la D435i y extrínseca cámara-IMU de fábrica.
- **Open3D 0.18,** `compute_rgbd_odometry` con término híbrido, contra keyframe.
- **DROID-SLAM RGB-D** y **DPVO** (`frameworks/learned`, PyTorch 2.3 + CUDA 12.1, RTX 4060 de 8 GB).
- **Validaciones:**
  - El camino TUM se validó: la odometría en vivo exportada a TUM da métricas idénticas a la evaluación directa.
  - El reprocesado de RTAB-Map es casi determinista: repetido, da lo mismo o difiere 0–6 mm.
  - DROID-SLAM es determinista: repetido, da lo mismo.

---

## 3. Resultados

### 3.1 Comparación (mediana entre las 8 corridas; ATE anclada en mm)

| Método | med | p90 | máx | escala | RPE 0,5 s | pérdidas | útil ≤ 10 mm | útil ≤ 20 mm (peor) |
|---|---|---|---|---|---|---|---|---|
| RTAB-Map en vivo (config. de v7) | 21,7 | 58,9 | 95,4 | +3,2 % | 10,8 | 4,6 % | 10 % | 35 % (18) |
| RTAB-Map reprocesado, sin ajuste de haces | 17,3 | 46,7 | 74,0 | +2,6 % | 9,4 | 1,1 % | 15 % | 36 % (29) |
| RTAB-Map + ajuste de haces | 13,0 | 20,9 | 49,6 | +1,7 % | 2,2 | 1,5 % | 30 % | 82 % (43) |
| RTAB-Map + GuessMotion | 14,3 | 46,3 | 88,5 | +2,6 % | 9,2 | 17,7 % | 19 % | 64 % (23) |
| RTAB-Map + flujo óptico | 17,9 | 58,5 | 83,0 | +2,8 % | 8,9 | 1,1 % | 17 % | 48 % (29) |
| RTAB-Map frame-to-frame | 21,9 | 71,8 | 127,5 | +2,4 % | 3,5 | 0 % | 21 % | 41 % (23) |
| RTAB-Map sin Kalman | 19,7 | 40,6 | 70,6 | +2,8 % | 10,3 | 7,8 % | 14 % | 35 % (21) |
| ORB-SLAM3 RGB-D (en vivo) | 11,4 | 24,1 | 471 | — | 1,6 | 49 % | 29 % | 42 % (27) |
| ORB-SLAM3 RGB-D (final, con ajuste global) | 10,8 | 233 | 235 | — | 1,6 | 35 % | 31 % | 42 % (27) |
| ORB-SLAM3 RGB-D-inercial | no inicializa: se rompe a los ~15 s en 8/8 | | | | | 97 % | 3 % | 3 % |
| ORB-SLAM3 monocular-inercial | 81,8 | 144 | 236 | — | 39 | 97 % | 0 % | 1 % |
| Open3D RGB-D denso | 12,8 | 33,2 | 39,5 | +1,3 % | 0,9 | 0 % | 37 % | 71 % (50) |
| DROID-SLAM RGB-D (`filter_thresh` 2,4, por defecto) | 11,8 | 16,0 | 17,6 | +1,6 % | 0,9 | 0 % | 40 % | 100 % (2) |
| **DROID-SLAM RGB-D (`filter_thresh` 1,0)** | **10,9** | **14,7** | **17,0** | +1,4 % | **0,9** | **0 %** | **45 %** | **100 % (62)** |
| **ídem, profundidad ×0,986 (dejando una corrida fuera)** | **7,9** | **11,5** | **15,0** | 0 % | 0,9 | 0 % | 70 % | 100 % (81) |

DPVO (monocular) quedó excluido. En la corrida 3, aun ajustando su escala, dio una mediana de 16,1 y un máximo de
145,6 mm. En las demás, su optimización final pide más de 8 GB de GPU, incluso a 10 Hz y con un búfer de 1024.

### 3.2 DROID-SLAM (`filter_thresh` 1,0) por corrida

| Episodio v7 | med | p90 | máx | final | Sim(3) med | escala |
|---|---|---|---|---|---|---|
| 2 | 12,0 | 13,7 | 15,3 | 0,7 | 2,7 | +1,30 % |
| 51 | 7,1 | 11,1 | 11,7 | 0,8 | 1,8 | +1,23 % |
| 18 | 11,8 | 15,8 | 16,8 | 0,4 | 2,3 | +1,46 % |
| 48 | 8,4 | 10,3 | 22,3 | 0,3 | 2,0 | +1,05 % |
| 61 | 11,0 | 15,7 | 17,2 | 2,6 | 2,4 | +1,27 % |
| 3 | 8,4 | 11,1 | 12,9 | 0,2 | 1,8 | +1,80 % |
| 68 | 10,7 | 17,6 | 19,8 | 0,6 | 1,9 | +1,68 % |
| 73 | 15,9 | 24,2 | 24,8 | 0,8 | 1,4 | +1,83 % |

Costo: 96–183 s por corrida (57–113 s de video, 2032–3696 frames), con 3,88 GB de GPU como máximo.

### 3.3 Diagnóstico de las pérdidas (`diagnose_tracking_loss.py`)
- **La profundidad no es la causa:** sigue válida en un 92–100 % hasta a 0,22 m de la mesa. H4 del plan es falsa.
- **Las pérdidas son por falta de textura.**
  - Ocurren con la cámara a menos de ~0,35 m, cuando los ArUco, los bordes y el tablero salen de la vista y solo
    queda mantel blanco.
  - Las esquinas GFTT bajan de ~220 a ~20. El umbral relativo se satura en 1000 respuestas débiles de la trama.
  - El episodio 68 nunca bajó de 0,31 m y no perdió nada.

---

## 4. Argumento práctico
1. **Exactitud.** Mediana 10,9 mm, frente a 13,0 de la mejor configuración de RTAB-Map y 21,7 de la grabación de v7.
   - El p90 y el máximo bajan a la mitad o a un tercio, y es la deriva grande (p90 y máximo) la que separa las demos.
   - Sim(3): 1,4–2,7 mm.
2. **Robustez.** 0 % de pérdidas en las 8 corridas. Las tres en que RTAB-Map y ORB-SLAM3 se pierden (48, 61, 2)
   quedan en un máximo de 15–22 mm.
   - Cobertura útil a ≤ 20 mm: 100 % (peor corrida, 62 %), frente a 82 % (43) de RTAB-Map con ajuste de haces.
   - Un episodio con pérdidas se descarta en el control de calidad, así que la cobertura es rendimiento de la
     recolección.
3. **Consistencia global.** Al volver a HOME3, el error final es ≤ 2,6 mm (RTAB-Map en vivo: 55 mm de mediana). Es
   lo que pedía el problema original: "mucho drift entre demostraciones".
4. **Reproducibilidad.** Es determinista, y como corre offline sobre los bags crudos se puede reprocesar.
5. **Costo aceptable.** ~2 min por demo y 3,9 GB de GPU en la RTX 4060. Es compatible con "un entrenamiento pesado a
   la vez": no correrlo junto a un entrenamiento.

## 5. Argumento teórico
**Características del problema:**
- La escena tiene poca textura (mantel blanco) y la cámara se acerca a 0,22–0,35 m, donde los rasgos fuertes salen
  de la vista.
- El color es de obturador rodante.
- La profundidad (estéreo activo) es buena, pero con un sesgo de escala de ~1,4 %.
- El IMU es MEMS de bajo costo y el movimiento es suave.
- Se permite cómputo offline.

1. **Por qué fallan los métodos de rasgos dispersos (RTAB-Map F2M, ORB-SLAM3).**
   - Detectan esquinas GFTT/ORB repetibles, las asocian por descriptor y resuelven la pose con PnP + RANSAC.
   - Sin textura, el detector responde al ruido, las asociaciones no superan el mínimo de inliers, y el sistema entra
     en un estado discreto de pérdida.
   - RTAB-Map entonces congela la pose; ORB-SLAM3 abre un mapa nuevo con otro origen, lo que da saltos de 0,2–0,7 m.
   - La falla es binaria, no gradual. El ajuste de haces local de RTAB-Map reduce el ruido y la escala (lección 47),
     pero no crea rasgos donde no los hay.
2. **Por qué un denso directo clásico ayuda, pero no basta (Open3D; Steinbrücker et al., 2011; Park et al., 2017).**
   - Minimiza residuos fotométricos y geométricos de TODOS los píxeles: no hay detección discreta y no se pierde
     (0 %).
   - Pero el término geométrico es degenerado sobre una mesa plana: no fija la traslación paralela al plano. El
     fotométrico depende de gradientes débiles, con una cuenca de convergencia chica.
   - Al optimizar solo contra el keyframe (sin ajuste global), la deriva se acumula: 190 mm en el episodio 61.
3. **Por qué DROID-SLAM** (Teed y Deng, NeurIPS 2021):
   - *Correspondencia densa aprendida:* rasgos aprendidos y un volumen de correlación 4D sobre todos los pares de
     píxeles (como RAFT; Teed y Deng, ECCV 2020), refinado por un operador recurrente. Aprovecha la trama tenue del
     mantel donde un detector de esquinas no encuentra nada repetible.
   - *Ajuste de haces denso (DBA) diferenciable:* cada iteración resuelve por Gauss-Newton las poses y la profundidad
     inversa por píxel. Usa pesos de confianza por píxel que predice la red, así que lo poco fiable se pondera hacia
     abajo en vez de provocar una falla discreta. No tiene estado de "perdido".
   - *RGB-D:* la profundidad del sensor entra en el DBA como observación de la profundidad inversa. Eso fija la
     escala métrica, y también explica que herede el sesgo de ~1,4 % de la D435i (todos los métodos RGB-D lo
     muestran), corregible con un factor.
   - *Grafo de frames con aristas de proximidad y ajuste global en el backend:* al volver cerca de vistas previas (a
     HOME3), las aristas de largo alcance corrigen la deriva acumulada. De ahí los ≤ 2,6 mm de error final.
   - *Validez fuera de dominio:* entrenado en TartanAir (sintético), generaliza a TUM RGB-D, ETH3D y EuRoC, según su
     paper. Aquí se verificó contra el robot y no se asume.
4. **Relación con UMI** (Chi et al., RSS 2024).
   - UMI usa ORB-SLAM3 monocular-inercial con un mapa previo y escenas con textura.
   - Aquí la escena es pobre en textura, el movimiento del robot no excita lo bastante al IMU para inicializar ORB-SLAM3
     inercial, y está la profundidad activa. Eso favorece un método denso con RGB-D.
5. **Límites y amenazas a la validez:**
   - *Velocidad:* el robot fue a 3,5 cm/s y la mano va 2–3 veces más rápido.
   - *Escena:* sin herramienta, y el tablero solo al fondo.
   - *Selección del parámetro:* `filter_thresh` 1,0 se eligió viendo la falla determinista de la corrida 51, con
     arranque quieto. Las otras 7 no empeoran, pero hay que validarlo en datos nuevos.
   - *Corrección de profundidad:* el factor 0,986 se validó dejando una corrida fuera en la misma sesión. Medirlo de
     nuevo tras calibrar la cámara.
   - *Referencia:* la ATE anclada incluye ~9 mm de incertidumbre de la extrínseca cámara-brida.
   - *Determinismo:* es determinista en el mismo hardware y software. No se probó en otras GPU.

## 6. Pendientes
1. `build_dataset_from_bags.py`: poses del dataset v8 con DROID-SLAM (`filter_thresh` 1,0, profundidad ×0,986) desde
   los bags crudos del recolector v3.
2. Calibración en chip y *tare* de la D435i, o factor de escala medido con el robot tras calibrarla.
3. Validación con datos nuevos: velocidad de mano (contra el tablero o con el robot más rápido) y con herramienta en
   la mesa. **Las demos se graban A MANO** (la cámara en la mano, acercándose al objeto). Protocolo sin robot:
   - El tablero va donde termina la aproximación, para tener referencia de ~0,1–0,4 mm en la vista final.
   - La pose de inicio se ancla con los ArUco + IMU (~1 cm / 0,6°).
   - Varias aproximaciones a velocidad real de mano (p99 ~0,12 m/s; 2–3 veces la del robot) y con temblor de mano.
   - Medir el error de DROID al final de la aproximación, la parte donde más deriva.
   - El tablero agrega textura (resultado algo optimista): repetir también con la herramienta sola.
   - Revisar el umbral de quietud del giróscopo (0,05 rad/s) frente al temblor de mano.
4. Próxima sesión con el robot: grabar el IR estéreo para OpenVINS, Basalt y ORB-SLAM3 estéreo-inercial (pedido del
   usuario).
5. Control de calidad automático de DROID: comparar su rotación con la del giróscopo integrado por episodio (no
   implementado).

## 7. Archivos (todo en git salvo los datos crudos)
- **Herramientas:** `tools/xarm5/vio_gt/` (planes, movimiento, orquestación, evaluación, resumen, diagnóstico,
  export) y `tools/xarm5/vio_gt/frameworks/` (Dockerfiles y ejecutores).
- **Resultados:**
  - `experiments/vio_gt/2026-11-24/`: runs/*/vio_vs_robot_<método>.json/png, hand_eye*.json,
    color_intrinsics_check.json, tracking_loss.*, summary_*.md/json.
  - `frameworks/` (trayectorias TUM y bitácoras).
  - `plans/`.
- **Datos crudos (en disco, fuera de git):**
  - `datasets/vio_bags/` (5,5 GB);
  - `experiments/vio_gt/2026-11-24/replay*/` (13,5 GB de odometrías reprocesadas);
  - `datasets/vio_frames/` (3 GB).
  - **No borrarlos: son la evidencia del paper.**

## 8. Anclaje con la pose REAL de inicio: ArUco + IMU (2026-11-25)

**Problema:**
- El conversor suponía que cada demo empieza exactamente en HOME3 y le pegaba el movimiento relativo de la
  odometría. Un error de inclinación θ en el inicio rota todo el episodio: unos 0,6 m · θ al final (5° son ~5 cm).
- En v7 el inicio estaba a 32° ± 6° de inclinación frente a los 24,6° de HOME3, y el encuadre variaba hasta ~80 px.
- Probablemente es un error mayor que el de la odometría, y nunca se corrigió: el conversor descarta la orientación
  absoluta, y v7 no guardó el IMU.

**Método** (`visuomotor_data_collection/start_anchor.py`):
- *Calibración* (`tools/xarm5/vio_gt/calibrate_table_markers.py`):
  - Las 8 esquinas de los dos ArUco de la mesa se triangulan en `link_base` con 9 vistas del robot en poses
    conocidas. No hace falta medir los marcadores: el lado sale de 51,2 y 50,6 mm.
  - La vertical real de la base (0,86° de z: desnivel de la base + sesgo del IMU) sale del acelerómetro en esas vistas.
  - Se guarda en `visuomotor_bringup/config/realsense/table_markers.yaml`.
- *Estimación por episodio* (`tools/xarm5/estimate_start_poses.py`):
  - Ventana quieta del inicio, detectada con el giróscopo.
  - PnP (SQPnP + LM) con las esquinas medianas de varios frames.
  - Inclinación reemplazada por la del acelerómetro. Si discrepan > 3°, el episodio queda en HOME y se marca.
  - Sale `T_base_camera_link` → `convert_vio_dataset_to_xarm5.py --start-poses` (conversor v3). Sin la opción, la
    salida es idéntica a v3 (verificado en los 83 episodios).

**Validación contra el robot** (ventanas quietas de la corrida `static_board`, dejando una fuera; error de
posición / rotación):

| Variante | Mediana | Máximo |
|---|---|---|
| PnP solo | 11,2 mm / 1,20° | 33,6 mm / 5,84° |
| **PnP + IMU** | **11,2 mm / 0,63°** | **33,6 mm / 0,89°** |
| PnP + IMU, con los dos marcadores (caso HOME3) | 10,8 mm / 0,61° | — |
| 3D-3D con la profundidad | 63 mm / 4,5° | — |
| Híbrido (rotación de PnP + IMU, posición de la profundidad) | 16,8 mm / 0,63° | 18,2 mm |
| Resolución de v7 (256×256, un frame, sin IMU) | 33 mm / 4,0° | con fallas de 543 mm / 50° |

- El 3D-3D se descarta: muestra un desfase de ~17 mm entre la profundidad y las esquinas trianguladas (extrínseca o
  profundidad).
- Prueba nula en las 8 aproximaciones de la sesión, que empezaron en HOME3 con los dos marcadores visibles:
  - error contra el robot de 5,2 mm / 0,55° de mediana (máx. 7,0 mm / 0,63°);
  - 8/8 utilizables y el IMU de acuerdo en todas.
- Conclusión: la inclinación de inicio baja de ~7° ± 6° (v7) a ~0,6°, y la posición queda en ~1 cm.
- v7 **no** se reancla: con 256 px, un solo frame y sin IMU, el anclaje no mejora el error original.

**Límites:**
- La validación es de la misma sesión de la calibración. Con un solo marcador visible de lado, el error de posición
  llega a ~3 cm.
- La calibración vale mientras no se muevan los ArUco ni el robot respecto de la mesa.
- Falta validarlo con inicios deliberadamente fuera de HOME3: en la próxima sesión, ±3 cm y ±5°, con el robot como
  referencia.
- Protocolo (demos a mano): el recolector v3.1 empieza el bag crudo en ARMING, cuando ya pide la cámara QUIETA para
  reiniciar la odometría. Esa ventana da los frames y el acelerómetro, así que no hace falta otra pausa después del
  pitido de "Grabando". Si la pausa de ARMING es corta (< 0,2 s), se usa la quietud que detecta el giróscopo.

**Referencias:**
- Teed y Deng, "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras", NeurIPS 2021.
- Teed y Deng, "RAFT", ECCV 2020.
- Teed, Lipson y Deng, "Deep Patch Visual Odometry", NeurIPS 2023.
- Campos et al., "ORB-SLAM3", IEEE T-RO 2021.
- Labbé y Michaud, "RTAB-Map as an open-source lidar and visual SLAM library", J. Field Robotics 2019.
- Steinbrücker, Sturm y Cremers, "Real-time visual odometry from dense RGB-D images", ICCV Workshops 2011.
- Park, Zhou y Koltun, "Colored point cloud registration revisited", ICCV 2017.
- Chi et al., "Universal Manipulation Interface", RSS 2024.
