# 17. Curación del Dataset Real VIO v7, Acciones Relativas (UMI) y Entrenamiento para el xArm5 (Fase 22)

Fecha: 2026-11-22. Dataset de origen: `datasets/demonstrations_real_v7.zarr` (83 episodios a mano, D435i + RTAB-Map
VIO, tarea: SOLO aproximarse a unas tijeras de podar sobre la mesa desde HOME3; sin agarre, gripper pasivo en 0.0).
Resultado: `datasets/demonstrations_real_v7_xarm5_v2.zarr` (76 episodios, 23.731 frames, 19.8 min).

---

## 1. Defectos encontrados en `demonstrations_real_v7_xarm5.zarr` (conversión v1, "certificada")

La certificación anterior comprobaba integridad y formas de tensores, no la física. Auditoría del 2026-11-22:

| # | Defecto | Evidencia | Consecuencia |
|---|---|---|---|
| 1 | **Extrínseca cámara-TCP invertida 180° sobre el eje óptico** (`rpy (0,-pi/2,0)` era la suposición sin medir marcada `extrinsic_calibrated: true`) | Gravedad del VIO al inicio vs HOME3: 57° de mediana (0/83 < 20°); con `(pi,-pi/2,0)`: 13° (59/83 < 20°). ArUco (independiente): eje óptico a 32° ± 6° de la vertical (HOME3: 24.6°) e imagen derecha (giro 0.1°) | Desplazamientos laterales reflejados: TCP "detrás de la base" (x hasta -0.04 m), 49% de frames recortados por el puente |
| 2 | **Signo del cuaternión discontinuo** (`qw >= 0` con la garra hacia abajo) | En `arm_plane`: `qw = -sin(pitch/2)·sin((yaw-2·acimut)/2)` ~ ±0.001 en HOME3; 1015 saltos q -> -q en 77/81 episodios; `qx` salta entre +0.98 y -0.98 | Objetivo discontinuo; el ensemble temporal promedia q y -q -> norma ~0 -> identidad (garra hacia ARRIBA) en `policy_node` |
| 3 | **Ruido del VIO en movimiento ~8.8 mm RMS** (isótropo, no artefacto de muestreo) | PSD: movimiento < 0.8 Hz, piso plano desde 0.9 Hz; saltos aislados de 3-9 cm/frame | Velocidad aparente 0.20 m/s vs real 0.05 m/s (p99 0.14 m/s); la política aprendería el ruido |
| 4 | **Odometría NO reiniciada entre episodios** | `initial_pos_*` deriva dentro de cada sesión; inclinación inicial según VIO hasta 86° vs 34° según ArUco (ep. 35) | La "vertical" de odom no es fiable: solo usar el movimiento RELATIVO dentro del episodio (la conversión ya lo hace) |
| 5 | **Giros de muñeca de ~180°** (ep. 23, 24, 25, 30, 31, 32, 35) | Yaw > ±149° (límite del puente) -> salto de 15 rad/s al proyectar a 5 GDL; 23/24 además con pérdida de tracking (0.45-0.68 m) | No ejecutables por el xArm5 -> descartados |
| 6 | **Std del gripper = 1e-6** (constante 0.0) | `zarr_dataset.py` normalizaba (x - 0)/1e-6 | Una lectura de -1e-4 en el robot = z-score -99 |

Umbrales de auditoría relajados en v1 (`speed_tolerance_ratio 12`, `max_orientation_deviation_rad 1.50`,
`max_workspace_violation_frames 550`) ocultaban los defectos 1, 3 y 5. La métrica "0.199 m/s coincide con la
calibración 0.20 m/s" era casual: 0.20 m/s es el PID de Isaac Sim y la cifra estaba inflada por el ruido.

## 2. Correcciones (todas parametrizadas, sin valores fijos en código)

* `xarm5_dataset_alignment.yaml`: `tcp_to_camera_rpy_rad: [pi, -pi/2, 0]` (arriba de la imagen = lejos de la base en
  HOME3), `trajectory_filter` (Hampel 7 muestras/3σ + Butterworth orden 4 a 1.5 Hz, `filtfilt` con relleno constante),
  `max_raw_step_m 0.15`, `workspace_tolerance_m 0.03`, `max_workspace_violation_fraction 0.5`,
  `speed_tolerance_ratio 6.0`, `max_orientation_deviation_rad 0.70`.
* `xarm5_dataset_alignment.yaml`: `quaternion_hemisphere: "home"` (signo con dot(q, q_home) >= 0, q_home = rpy de HOME
  de `jetson_policy_params.yaml`). `train.py --quaternion_hemisphere auto` la hereda del dataset (un valor explícito
  contradictorio es un error) y la guarda en `config.json`; `jetson_policy_params.yaml` usa `"auto"` (la del
  checkpoint) y un valor forzado distinto del checkpoint detiene `policy_node`.
* Conversor v2 (`tools/xarm5/convert_vio_dataset_to_xarm5.py`, `CONVERTER_VERSION = 2`): filtro opcional, hemisferio,
  auditoría de saltos VIO, fracción de volumen con tolerancia y `workspace_max_clamp_m`. Sin las claves nuevas se
  comporta como v1 (pruebas existentes intactas).
* Módulos nuevos: `visuomotor_core/utils/rotations.py` (solo NumPy, usable en la Jetson),
  `visuomotor_core/dataset/trajectory_filters.py`.
* `zarr_dataset.py`: piso de std (dimensiones constantes -> std 1), subconjunto de episodios, split por episodio
  estratificado (`split_episodes`), `action_representation`, `image_size`/`proprio_dropout_prob` parametrizados.

Resultado v2: 0 saltos de signo, recorte del puente > 5 cm en 0.1% de frames (antes 16%), fin de la aproximación en
x = 0.47 ± 0.06, y = -0.10 ± 0.14, z = 0.28 ± 0.07 m (delante del robot, sobre la mesa). Figuras:
`datasets/evaluations_real_v7_xarm5_v2/fig1..fig3` (herramienta `tools/xarm5/analyze_vio_dataset_quality.py`).

## 3. Implicaciones de despliegue (hardware)

* **Montaje de la cámara**: D435i en la brida con el eje óptico a lo largo de +Z_tcp y la imagen DERECHA en HOME3
  (arriba de la imagen hacia fuera de la base, base abajo). Si se monta girada 180°, las imágenes quedan invertidas
  respecto al dataset. Traslación cámara-TCP sin medir (0): medir y reconvertir afinará posiciones al girar.
* Velocidad real de las demostraciones ~0.05 m/s (p99 0.14): el límite del puente (0.05 m/s) es compatible; subirlo a
  0.10-0.15 m/s tras validar en dry-run reproduce mejor los tramos rápidos.
* Objetos dentro de |y| <= 0.30 m (caja conservadora del puente); algunos episodios terminan en y ≈ -0.35.

## 4. Representación de acción `relative_ee` (UMI) y piloto comparativo

* `DiffusionConfig.action_representation` ∈ {`absolute`, `relative_ee`} (se guarda en `config.json`). Con
  `relative_ee` la acción [0:7] es la pose objetivo expresada en el marco del efector ACTUAL
  (`T_actual^-1 · T_objetivo`; Chi et al., UMI RSS 2024: 18-20/20 frente a 5/20 con acción absoluta por errores de
  calibración SLAM-robot). `ZarrDataset` la calcula al vuelo (ancla = estado real antes del dropout propioceptivo) y
  `DiffusionPolicy.actions_to_absolute` la recompone con el estado observado dentro de `select_action`/`predict_chunk`:
  `policy_node` sigue recibiendo poses absolutas (base - home) sin cambios de interfaz.
* Los cuaterniones relativos están cerca de la identidad: la discontinuidad de signo desaparece en la acción.
* Piloto (mismo split por episodios, semilla 42, 20 épocas, EMA, ColorJitter), episodios de validación
  `[7, 16, 26, 35, 45, 54, 64, 73]` de v2:

| Época 20 (val) | ADE | ADE ejecutado (8 pasos) | FDE | Rotación | Progreso | F1 done |
|---|---|---|---|---|---|---|
| `relative_ee` | **12.6 mm** | **8.4 mm** | **20.2 mm** | **1.89°** | 1.07 | **0.52** |
| `absolute` | 38.9 mm | 37.0 mm | 43.5 mm | 4.24° | 1.62 | 0.42 |
| Línea base "quedarse quieto" | 18.6 mm | 11.0 mm | — | — | 0 | — |

  `absolute` no bate la línea base trivial y sobrepasa (progreso 1.62). Figura:
  `visuomotor_core/outputs/pilot_comparison/pilot_action_representation.png` (`tools/compare_eval_runs.py`).

## 5. Cambios en el entrenamiento (`visuomotor_core/train.py` y `visuomotor_core/training/`)

* EMA de pesos (`training/ema.py`, power 0.75, máx. 0.9999; se guardan y evalúan los pesos EMA), warmup lineal (500)
  + coseno a 1e-6, ColorJitter UMI en GPU (`training/augment.py`; en CPU dividía por 10 el DataLoader).
* Split estricto por episodio (`--val_ratio`, `split_episodes` estratificado) y evaluación offline por muestreo
  completo DDIM (`training/evaluation.py`): ADE/FDE (mm), ADE de los pasos ejecutados, error geodésico, F1 del done,
  razón de progreso y línea base "quedarse quieto" (horizonte completo y ejecutado). La pérdida de validación NO
  predice el éxito (Mandlekar et al. 2021; Chi et al. 2023): solo se usa como diagnóstico.
* Robustez: `--resume` (estado completo en `training_state.pt`, escritura atómica, CSV reescritos desde historiales),
  guardia de RAM `--max_rss_gb`, `--prefetch_factor` (defecto 2), figuras intermedias a 120 DPI solo en épocas de
  evaluación (finales a 300 DPI), registro de "RAM proceso" en el log.
* Figuras en inglés con el nombre del dataset (`training/plots.py`), sin doble eje Y.
* Rendimiento medido (RTX 4060 Laptop 8 GB): ~5.6 it/s a batch 64 (GPU al 99-100%, cómputo del ResNet-18 + U-Net;
  `channels_last` no aporta), ~2 GB de VRAM.

### Incidente OOM (2026-11-22)
Tres entrenamientos simultáneos (10-12 workers, prefetch 4, ~100 MB por lote en memoria compartida + copia fijada)
agotaron los 31 GB del host: el OOM-killer cerró procesos Python y servicios de la sesión gráfica (vuelta a GDM).
Regla: un entrenamiento pesado a la vez (dos como máximo con ≤ 6 workers y prefetch 2), comprobar `free -m` antes de
lanzar y usar `--max_rss_gb`. La caché de memoria fijada de PyTorch no se libera: los DataLoaders de evaluación ya no
usan `pin_memory`.

## 6. Ruta de despliegue verificada sin hardware

* `policy_node`: `quaternion_hemisphere` ('auto' = la del checkpoint), cuaternión de comando con la convención del
  checkpoint y "mantener la última orientación válida" (nunca la identidad), `image_resize_method: cv2_area`
  (= recolector), `neural_auto_stop_patience_steps` (pasos consecutivos sobre el umbral).
* `DiffusionPolicy.select_action`: ensemble temporal con signos alineados y renormalización; `reset()` limpia los
  chunks activos (antes se mezclaban entre episodios).
* `xarm5_real.launch.py` usa `config/xarm5/xarm5_pose_smoother_params.yaml` (min_cutoff 1.5 Hz; el compartido de
  simulación usa 0.1 Hz ≈ 1.6 s de retardo).
* Herramientas: `tools/xarm5/replay_policy_offline.py` (ruta exacta de inferencia de la Jetson con observaciones de la
  demostración + barrido umbral x paciencia del auto-stop) y `tools/xarm5/check_policy_closed_loop_sim.py` (cadena ROS 2
  completa con el puente en `simulate_robot_state`, dominio aislado + `ROS_LOCALHOST_ONLY`; requiere
  `source /tmp/xarm_ws/install/setup.bash` para `xarm_msgs`).
* `install/` de colcon contenía copias antiguas de `visuomotor_core`/`visuomotor_ros2`: recompilar con
  `colcon build --symlink-install --packages-select visuomotor_core visuomotor_ros2 visuomotor_bringup`.
* Jetson: Python 3.8 + torch 2.0 → código verificado con `ast.parse(feature_version=(3, 8))` y sin
  `torch.where` escalar-escalar. La imagen de producción debe reconstruirse (o usar `docker-compose.dev.yml`) para
  incluir el código nuevo.

## 7. Revisión de código (2026-11-22) y correcciones
* Convención del cuaternión: fuente única en la alineación del dataset -> checkpoint -> `policy_node` ('auto').
* `unwrap_rotvec` en el filtro de orientación (continuo al cruzar pi; v2 no estaba afectado: rotación máx. 108°).
* `--resume` guarda solo épocas completas (una época interrumpida se repite) y verifica más claves de configuración.
* `split_episodes` exige al menos un episodio de entrenamiento; el piso de std solo aplica a dimensiones CONSTANTES
  (max - min <= 1e-6), conservando la normalización histórica de dimensiones con variación pequeña.
* Verificación: reconvertir v7 con el código final reproduce `demonstrations_real_v7_xarm5_v2.zarr` bit a bit.

## 8. Resultado del entrenamiento principal y selección del modelo (2026-11-22)

Corrida `visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300`: 300 épocas x 329 pasos (98.700), batch 64,
`relative_ee`, EMA, ColorJitter, 68 episodios de entrenamiento / 8 de validación (`[7, 16, 26, 35, 45, 54, 64, 73]`),
~5 h a 5.6 it/s, RAM del proceso estable en 5.1 GB.

| Época | ADE | ADE ejec. | FDE | Rot. | Progreso | F1 done |
|---|---|---|---|---|---|---|
| 10 | 14.6 | 10.0 | 23.4 | 2.37° | 1.08 | 0.38 |
| 100 | 11.8 | 7.64 | 19.1 | 1.72° | 1.04 | 0.66 |
| 180 | 11.5 | **7.55** | 18.5 | 1.71° | 1.01 | 0.70 |
| **200 (seleccionada)** | 11.6 | 7.61 | 18.5 | 1.74° | **1.00** | 0.70 |
| 220 | 11.6 | 7.61 | 18.5 | 1.75° | 1.00 | **0.71** |
| 300 | 11.6 | 7.67 | 18.4 | 1.76° | 1.00 | 0.67 |
| Línea base "quedarse quieto" | 18.6 | 11.0 | — | — | 0 | — |

* El ADE de entrenamiento sigue bajando (memorización esperable en DP) mientras el de validación se estabiliza desde
  la época ~150; el F1 del done cae tras la 220 (ligero sobreajuste de la señal de término).
* **Los criterios de parada basados en la pérdida (módulo 12) recomiendan la época ~8 (consenso, paso 2.560)**, cuando
  las métricas de comportamiento mejoran hasta ~200 (F1 done 0.38 -> 0.70). Para DP la pérdida no sirve para elegir
  checkpoints: usar métricas de muestreo completo + reproducción de la ruta de despliegue.
* Reproducción de la ruta de la Jetson (8 episodios de validación): error del comando 11.2-11.7 mm y 1.6-1.75° en las
  épocas 100-300; la época 200 maximiza la robustez del auto-stop (9 configuraciones umbral x paciencia con 100% de
  paradas y 0 prematuras). Elegido: **umbral 0.60, paciencia 5** (robusto en las épocas 180/200/220): 100% de paradas,
  0 prematuras, 8 mm de la meta, 0.28 s antes que la persona (con 0.50/1: 2 paradas > 1 s adelantadas).
* Lazo cerrado ROS 2 con puente simulado (episodios 45 y 7): coseno de dirección 0.992 / 0.993, 0.56 / 0.61 m
  recorridos, auto-stop correcto, 0 rechazos del puente, 0 errores.
* Paquete: `deploy/jetson/model/` (`model.safetensors` sha256 `5c5c7ed5…`, `config.json`, `training_meta.json`,
  `MODEL_CARD.md`, `final_model_report.png/.pdf`). Despliegue por etapas en `deploy/jetson/README.md`.
* **El modelo aún NO se ha probado en el xArm5 físico.**

## 9. Diagnóstico frente a líneas base exigentes y ablación visual (2026-11-22)

`tools/diagnose_policy_baselines.py` mide sobre las 2.582 ventanas de validación, con `n_action_steps` = 3 como la
Jetson. La tabla de §8 usaba los 8 pasos de `config.json`.
* **Velocidad constante:** extrapola la velocidad de los dos últimos estados y no usa la imagen.
* **Imágenes cruzadas:** cada ventana recibe la imagen de otra ventana del lote; el 88 % viene de otro episodio.
* **Dirección:** coseno > 0.9 entre el desplazamiento predicho y el demostrado a 0.75 s.

| Época 200 (validación) | Política | Imágenes cruzadas | Velocidad constante | Quedarse quieto |
|---|---|---|---|---|
| Error en los 3 pasos ejecutados (mm) | 3.63 | 4.80 | **1.27** | 5.04 |
| ADE del horizonte (mm) | 11.6 | 16.9 | **10.6** | 18.6 |
| FDE a 0.75 s (mm) | **18.5** | 29.2 | 21.3 | 33.4 |
| Dirección correcta, ya en movimiento (n = 2010) | **63 %** | 36 % | — | — (dirección media: 27 %) |
| Dirección correcta, desde el reposo (n = 22) | **64 %** | 50 % | — | — (dirección media: 55 %) |

* **La política usa la imagen.** Con imágenes cruzadas el FDE casi iguala a quedarse quieto, y la dirección en
  movimiento cae de 63 % a 36 %.
* **Pierde con la velocidad constante a corto plazo y gana a largo.** El perfil de error se cruza a ~0.53 s. La
  política aporta la intención (hacia dónde y cuándo frenar), pero sus primeros pasos son ruidosos; el ensamble
  temporal y el suavizador del robot tienen que absorberlo.
* **Por episodio:** la política bate a la velocidad constante en FDE en 7/8 episodios y en los 3 pasos ejecutados en 0/8.
* **Desde el reposo el estado es idéntico en todos los episodios.** Ahí la ventaja sobre "ir siempre hacia la
  dirección media" es pequeña (64 % vs 55 %, n = 22): v7 tiene poca variedad de destinos (desviación estándar lateral
  de 14 cm; todos van hacia delante y abajo).
* **Entre las épocas 100 y 300 nada cambia:** FDE 19.1 → 18.4 mm, error en 3 pasos 3.71 → 3.64 mm. Validación
  estancada y entrenamiento en 4.3 mm: el límite son los datos, no las épocas.
* **Integración:** `evaluate_sampling` calcula la línea base en cada evaluación de `train.py` (`cv_ade_mm`,
  `cv_ade_exec_mm`, `cv_fde_mm`, `hold_fde_mm`). El desfase acción/estado se estima del dataset con
  `estimate_action_offset`: en v7, `action[t] = state[t]` (desfase 0). Las curvas de evaluación y
  `tools/compare_eval_runs.py` dibujan la línea base. Pruebas: `tests/test_evaluation_baselines.py`.
* **Criterio para futuros checkpoints:** batir a quedarse quieto no basta. Hay que batir a la velocidad constante en
  FDE y que el FDE suba claramente con imágenes cruzadas. El objetivo de la próxima toma es reducir el error en los
  pasos ejecutados hacia el de la velocidad constante.

## 10. Alcance del xArm5 frente a la caja del puente (2026-11-22)

Detalle en el módulo 16 §5s y en `datasets/evaluations_real_v7_xarm5_v2/fig_xarm5_workspace_vs_demos.png`.
* **`|y| <= 0.30 m` no es un límite físico.** Es la caja de seguridad configurada; el brazo llega a ±0.47 m a la
  profundidad y altura de los objetos de v7.
* **El alcance físico limita el inicio.** 209 frames (0.9 %) de 5 episodios avanzan en alto desde HOME3 más allá del
  alcance del puente.
* **Para la próxima toma:** ampliar la caja si la celda lo permite (decisión del usuario; con ±0.45 m ningún frame v7
  queda fuera), repartir los objetos a ambos lados, y desde HOME3 bajar antes de alejarse de la base.

## 11. Preparación para la prueba física (2026-11-22)

* **Python 3.8 en la Jetson** (L4T r35.4.1, torch 2.0.0, diffusers 0.30.3). Chequeo estático del código que corre ahí
  (`visuomotor_core`, `visuomotor_ros2`, `visuomotor_bringup`, 46 archivos): sintaxis con `ast`
  (`feature_version=(3, 8)`), anotaciones `X | Y` / `list[...]` evaluadas sin `from __future__ import annotations` y
  APIs de la stdlib posteriores a 3.8.
  - Solo fallaba `dataset/dataset.py`, un archivo antiguo que nadie importa; se corrigió con la importación de
    `__future__`.
  - La prueba real sigue siendo la etapa A (robot simulado) ejecutada en la Jetson.
* **Pesos fuera de git** (`deploy/jetson/model/*.safetensors`): se copian con `scp` y se verifica el sha256
  `5c5c7ed532e9e3fd…`.
* **Referencia visual** (`tools/xarm5/export_home3_reference.py` → `deploy/jetson/model/home3_reference_view.png`):
  primer frame de 12 demos más la mediana de las 76. Muestra la mesa con mantel blanco, los dos ArUco, las tijeras de
  podar y la base amarilla y negra abajo, y ninguna garra en el encuadre. Hay que compararla con la cámara montada en
  HOME3 antes de armar.
* **Protocolo** (`deploy/jetson/README.md`, Fase 22): 20 intentos en grilla de 6 posiciones; éxito = parada
  automática a ≤ 5 cm sin intervención; estimación previa ~40 % (30-55 %).

