# 19. Dataset real v8: demostraciones a mano con la D435i, poses de DROID y conversión al xArm5

Registro técnico (2026-11-25/26) para el paper y para retomar. Decisión del estimador: `18_vio_estimator_selection_droid_slam.md`.

## 1. Recolección (sesión del 2026-11-25)

**Herramientas:**
- `run_v8_session.sh` / `v8_tmux.sh`: cámara en su propia sesión `cam_v8`, recolector, guía y voz.
- `vo_collector_node` v3.1:
  - bag crudo desde ARMING;
  - inicio automático por la vista de HOME3, con tolerancia de 20 px a mano;
  - guardado automático, cuya quietud final se juzga con el giróscopo (`qa_hold_motion_source: imu`);
  - [S] guarda siempre (los problemas quedan como avisos), y [U] descarta la última demo marcándola, sin borrarla.

**Protocolo:**
- Programa v2 (`v8_schedule_v2.csv`): 9 celdas × 6 orientaciones de las hojas (−90, −45, 0, 45, 90, 180, con +90 = derecha) × 2. Va en rondas balanceadas: si se corta, las celdas quedan parejas.
- Programa v3 (`v8_schedule_v3.csv`, 2026-11-26), para seguir sumando. v2 quedó con 104 de 108 hechas. v3 empieza por sus 4 pendientes (las demos descartadas, con la misma celda y orientación) y sigue con 4 ciclos nuevos: 220 en total (`make_collection_schedule.py --continue-from`, semilla 3, elegida para que la primera ronda no repita las recuperadas). `run_v8_session.sh` lo usa por defecto. Sin él, al terminar v2 el recolector habría seguido grabando sin celda ni orientación asignadas.
- La guía dibuja la grilla numerada SOBRE LA MESA, con la homografía de los ArUco.
- Al final, la cámara queda alineada con la herramienta (hojas hacia arriba en la imagen), por decisión del usuario.

**Resultado:**
- 126 episodios grabados en ~2 h 20 min, a ~0,25 GB por bag (profundidad PNG de 16 bits, color JPEG e IMU a 200 Hz).
- El 2026-11-26 había 14 descartados (los descartados con [U] más 5 de la validación) y 114 válidos: 105 del programa v2, 8 de la primera tanda y 2 sin programa (126 y 127).

## 2. Validación offline (`validate_v8_batch.sh` + `validate_v8_episodes.py report`)

Por demo se corre DROID-SLAM RGB-D sobre el bag crudo (~40-60 s de GPU) y se revisa:
- que DROID entregue todas las poses;
- saltos no físicos: pasos > 20 mm entre poses a 30 Hz que el acelerómetro no respalda;
- la coincidencia con la odometría en vivo;
- el final quieto;
- el recorrido de inicio a fin;
- la pose real de inicio (ArUco + IMU, `estimate_start_poses.py`).

Resultado sobre 119 demos: 106 OK, 8 REVISAR y 5 REPETIR (descartadas a pedido del usuario).

**Modos de falla encontrados:**
- **Salto no físico de DROID** al girar la cámara sobre el mantel blanco, sin textura (episode_71: 66 mm; episode_117: 245 mm). El IMU lo desmiente: no hay aceleración.
  - Tres de los cuatro casos son con las hojas a 180°, que exige girar la cámara media vuelta.
  - Con esa orientación, la odometría en vivo se pierde en el 40-68 % de los frames.
- **Inicialización de DROID:** en ~0,5 s puede oscilar hasta 0,7 m (episode_80) con la cámara quieta. Se mide aparte (`init_spread_mm`).
- **La demo termina donde empezó** (episode_5): bajó a la herramienta, VOLVIÓ a HOME3 y el guardado automático se disparó ahí. Regla nueva: recorrido de inicio a fin ≥ 0,25 m. Las demás miden ≥ 0,50 m (mediana 0,61 m).
- **Pose de inicio:** está a 44 mm / 3,9° de HOME3 (mediana; p90 93 mm), mucho más que en las primeras pruebas (6-15 mm). Anclar con la pose medida no es opcional.

## 3. Armado del dataset con poses de DROID (`build_dataset_from_bags.py`, versión 2)

1. **Marco:** las poses ópticas de DROID se llevan a `camera_link` con `T_link_opt`, la calibración exportada del bag.
2. **Referencia:** la mediana de las poses entre `t_bag + 0,5 s` (pasada la inicialización) y el primer frame + 0,4 s.
   - Medido en 114 demos: ARMING dura ~0,35 s; la mano se mueve al menos 0,53 s después del primer frame (mediana 1,07 s); a los 0,5 s DROID ya está estable (p95 2,5 mm).
   - Es el mismo instante en que se mide la pose real de inicio (ARMING).
   - Solo se fijan a la referencia los frames de la inicialización de DROID (≤ 4 por demo).
   - **Error de la versión 1:** la ventana "quieta" salía del giróscopo, pero la mano puede trasladarse sin girar. Borraba hasta 17 cm de movimiento real al inicio (episode_77) y fijaba hasta 150 frames.
3. **Escala:** ×0,986 (sesgo de profundidad de la D435i medido con el robot).
4. **Interpolación:** posición lineal y SLERP a los sellos de 20 Hz.
5. **Done:** el tramo final quieto según DROID, si dura ≥ 2 s. Pasó en 104 de 114; en las otras 10 queda el del recolector.

## 4. Conversión al xArm5 y verificación

**Conversión:** `convert_vio_dataset_to_xarm5.py` (conversor v3) con `--start-poses`, SIN `--drop-flagged`: se conservan todas las demos, por decisión del usuario.
- Resultado de esta primera conversión (v1, muñeca a ±149°; borrada, ver §7): 114 demos y 53 208 frames.
- 26 demos marcadas:
  - **Alcance:** 20. La mano avanza ALTA antes de bajar, y algunos finales de celdas lejanas quedan en el borde. En v8, el 64 % de las demos pasa el alcance del puente (v7: 20 %), con una extensión máxima mediana del 101 % del codo recto (v7: 88 %).
  - **Giro de muñeca:** 10, todas con las hojas a 180°. Con la cámara alineada se piden ±180° de J5, y el puente acota el giro a ±2,6 rad (±149°) para no cruzar ±π. El conversor lo recorta, y la orientación se desvía ~31° (mediana) frente a 9-14° en los otros ángulos.
- La cadena está verificada: en 49/49 demos el eje óptico final cae en la celda asignada (4,2 cm del centro en la mediana) y la cámara termina a 26-33 cm de la mesa.

**Verificación** (`verify_training_dataset.py`) contra `demonstrations_real_v7_xarm5_v2.zarr`, el dataset del modelo probado en el xArm5 (`train_real_v7x5v2_relative_ee_e300`, también el desplegado en la Jetson). Veredicto: LISTO.
- **Igual a la referencia:**
  - mismos arreglos de entrenamiento: imagen 256×256×3 uint8, estado 8 y acción 9;
  - mismas convenciones: coordenadas relativas a HOME, hemisferio del cuaternión `home` y `arm_plane`;
  - el mismo lote de `ZarrDataset` ([8,2,3,256,256], [8,2,8], [8,16,9]) y el mismo desfase acción/estado (0);
  - un paso forward/backward de `DiffusionPolicy` en la GPU, con pérdida y gradiente finitos.
- **Mejoras deliberadas:**
  - extrínseca medida (v7_v2 tenía traslación 0);
  - inicio medido;
  - poses de DROID, más suaves: salto mediano por frame de 4,6 mm frente a 6,5 mm.
- **Diferencias de contenido:**
  - 2,2× más ventanas de entrenamiento (53 094);
  - episodios más largos (22,5 s de mediana, frente a 15,3 s);
  - la zona de trabajo es más amplia: el TCP final llega hasta x = 0,74 m (v7: 0,60 m).
- **Arreglos extra** (sellos, inliers, pérdidas): el cargador los ignora.

## 5. Para entrenar y para seguir sumando

- **Entrenamiento:** con todas las demos (114) o solo con las factibles (94, sin las 20 marcadas por alcance tras la corrección de la muñeca, §6; lista en `datasets/demonstrations_real_v8_xarm5_conversion_report.json`), y comparar (ablación para el paper). Antes, congelar una copia (§7).
- **No mezclar con `v7_xarm5_v2`:** usa la extrínseca vieja. Si se combina con v7, usar `v7_xarm5_v3`.
- **Demos nuevas:** `tools/xarm5/make_v8_training_set.sh`, sin argumentos (ver §7), rehace el dataset de entrenamiento completo:
  1. DROID en las nuevas;
  2. poses de inicio;
  3. informe;
  4. armado (se detiene si a una demo válida le falta la trayectoria de DROID);
  5. conversión;
  6. verificación (poses de DROID obligatorias);
  7. reemplazo del dataset oficial, solo si la verificación pasa.
  - Las REPETIR no se marcan solas: hace falta el OK del usuario.
- **Monitor de alcance en vivo** (`reach_monitor.py`): avisa por voz sobre la posición (caja y alcance) y el giro de muñeca (±177° desde §6, con 5° de tolerancia): "baja primero y después avanza", "gira menos la cámara".
- **Borrado (2026-11-26, a pedido del usuario):** `datasets/demonstrations_real_v8_droid.zarr` (7,8 GB), el armado v1 con el error de la referencia (borraba movimiento real al inicio). No se usó para entrenar; se reemplazó por `demonstrations_real_v8_droid_v2.zarr` (armado v2).

## 6. Giro de muñeca a ±177° (decisión del usuario, 2026-11-26)

El proyecto es SOLO aproximación (sin pinza), y las demos con las hojas a 180° deben terminar con las hojas hacia ARRIBA en la imagen (cámara alineada). Esa vista pide ~±180° de J5, y el puente la acotaba a ±2,6 rad (±149°).
- **Cambios:**
  - `yaw_min/max` pasa a ±3,0892 rad (±177°, con 3° de margen a ±π) en `xarm5_real_params.yaml` y en los valores por defecto del puente;
  - nueva función `xarm5_interface.clamp_yaw_near`: si el yaw pedido queda FUERA del rango, se usa el límite más cercano al yaw ACTUAL (el comandado, o el del frame anterior), no el del lado del pedido.
- **Por qué hacía falta:** cerca de ±π el pedido salta entre +π y −π. Con el recorte anterior, el objetivo alternaba entre los dos límites, y como el yaw se interpola sin cruzar ±π, la muñeca intentaba casi una vuelta entera ida y vuelta. Dentro del rango nada cambia.
- **Dónde se usa:** en el puente (modo `arm_plane` y los demás) y en el conversor, que ahora proyecta secuencialmente recordando el yaw del frame anterior.
- **Monitor en vivo:** solo avisa del giro si pasa el límite en más de `reach_yaw_tolerance_deg` (5°). Con ±177° no avisa al alinear a 180°, porque el robot queda 3° corto.
- **Pruebas:** `test_clamp_yaw_near`, `YawNearPiTest` (el yaw oscilando a ±179,5° ya no salta) y la del monitor.
- **Despliegue:** el puente corre en la Jetson. Hay que llevarle este cambio (sincronizar el repo y recompilar/reconstruir la imagen) antes de ejecutar políticas entrenadas con demos de 180°.
- **Reconversión** (`demonstrations_real_v8_xarm5_v2.zarr`, que en §7 pasó al nombre fijo `demonstrations_real_v8_xarm5.zarr`):
  - Pasan de 26 a 20 demos marcadas, todas por alcance.
  - En las 15 de 180°, la desviación de orientación baja de 31,3° a 12,0° (mediana; máx. 21,5°), como en los demás ángulos.
  - Verificado contra `v7_xarm5_v2`: LISTO, con un paso de entrenamiento de pérdida 1,11.
  - La conversión v1 (a ±149°) quedó obsoleta y se borró (§7).
- **Nota para entrenar:** las demos de 180° terminan en +177° o −177°, según el lado desde el que se giró la cámara. Físicamente están a 6° una de la otra, pero en la convención `home` sus cuaterniones quedan casi opuestos (producto punto mínimo con home: 0,009 ≥ 0). Con `relative_ee`, la acción relativa al efector no se afecta; el estado de entrada tiene esos dos grupos.
- **Pendiente aparte (no relacionado):** `tests/test_hand_eye.py::test_solve_command_end_to_end` espera que `solve` se niegue con el tablero "sin medir", pero `charuco_board.yaml` ya está medido (`measured: true`). Las otras 4 fallas de la batería completa son del simulador Isaac Sim y de `analyze_convergence.py` (argumento `language`).

## 7. Organización de los datasets (decisión del usuario, 2026-11-26)

La regla literal del repo ("nunca sobrescribir ni borrar un dataset; cada corrección va a una copia nueva") dejó, además del crudo, tres carpetas v8: `_droid_v2`, `_xarm5` y `_xarm5_v2`. El usuario no sabía cuál era la oficial. Aclaró que la regla protege los crudos y los funcionales, no lo que rehace un script; quedó escrito en AGENTS.md §2.3.

- **Crudo** (la fuente; solo crece):
  - `datasets/demonstrations_real_v8.zarr`: lo que guarda el recolector (imágenes de 256 px, odometría en vivo, programa y marcas de descarte).
  - `datasets/raw_bags/demonstrations_real_v8/`: un bag MCAP por demo, con color y profundidad a resolución completa e IMU.
  - Cada sesión agrega sus demos a este mismo crudo, y el programa sigue donde quedó.
- **Entrenamiento** (derivado, nombre fijo): `datasets/demonstrations_real_v8_xarm5.zarr`, con su `_conversion_report.json`.
  - `tools/xarm5/make_v8_training_set.sh`, sin argumentos, lo rehace COMPLETO desde el crudo después de cada sesión. Así todas las demos pasan por el mismo código: agregando las nuevas a la versión anterior, las viejas habrían quedado con la muñeca a ±149° y las nuevas a ±177°.
  - Arma en `datasets/_work_demonstrations_real_v8/`, ignorada por git, y reemplaza el oficial solo si la verificación pasa. Si no pasa, el oficial queda intacto y el nuevo queda en la carpeta de trabajo para revisarlo.
  - El intermedio con las poses de DROID no se guarda: se rehace en ~25 s.
  - Primera corrida (2026-11-26, sin demos nuevas para DROID): 1 min 36 s de punta a punta. Cada demo nueva suma ~1 min de DROID.
- **Poses de DROID obligatorias** (pedido del usuario): cada episodio del dataset que se entrena lleva las poses de DROID-SLAM.
  - `build_dataset_from_bags.py` se detiene sin escribir nada si a una demo no descartada le falta la trayectoria. Antes la omitía en silencio; `--allow-missing-droid` conserva ese comportamiento.
  - `verify_training_dataset.py --require-pose-source droid_rgbd` no da LISTO si algún episodio no tiene `pose_source: droid_rgbd`. Con el dataset v7 (odometría en vivo) da REVISAR en 76 de 76.
  - El ancla de inicio en el marco del robot sigue saliendo de los ArUco y el IMU: DROID solo da el movimiento relativo de la cámara.
- **Funcional** (intocable): para entrenar, `tools/xarm5/freeze_training_set.sh` copia el oficial con fecha (`demonstrations_real_v8_xarm5_<AAAA-MM-DD>.zarr`, de solo lectura) y se entrena con esa copia. Así el `training_meta.json` del modelo nombra un dataset que no cambia. Si un entrenamiento usó el nombre fijo y su contenido no está congelado, `make_v8_training_set.sh` se niega a reemplazarlo.
- **Caché de DROID** (se conserva):
  - `datasets/vio_frames/demonstrations_real_v8/` (8,4 GB): cuadros, calibración e IMU exportados de cada bag, que usan DROID, la pose de inicio y el armado (`calib.json`);
  - `experiments/v8_validation/demonstrations_real_v8/droid/`: trayectorias TUM y registros.
- **Trazabilidad:** cada episodio de entrenamiento guarda `raw_dataset`, `raw_episode` (su clave en el crudo) y `raw_bag`.
- **Borrados a pedido del usuario (2026-11-26):**
  - `demonstrations_real_v8_droid.zarr`: armado v1, con el error de la referencia.
  - `demonstrations_real_v8_droid_v2.zarr`: intermedio.
  - La conversión v1, con la muñeca a ±149°.
  - `demonstrations_real_v8_xarm5_v2.zarr`, después de comprobar que el de nombre fijo es idéntico. Sus 161 336 archivos coinciden byte a byte, salvo los 115 `.zattrs`, que solo cambian la procedencia. Los episodios del informe de conversión también son iguales.
  - Se liberaron 15 GB. Los registros de esos armados quedan en `experiments/v8_validation/demonstrations_real_v8/historial_2026-11-26/`.
