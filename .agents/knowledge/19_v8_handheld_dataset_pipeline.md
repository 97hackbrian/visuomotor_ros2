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

- **Entrenamiento:** con todas las demos o solo con las que quedan dentro del alcance del robot, y comparar (ablación para el paper). Desde el 2026-11-26 son 230 demos; las que están dentro del alcance salen del criterio físico de las métricas (§8: 137 con ≤ 5 % de frames fuera), no de las marcas del conversor, que solo marca más del 50 % fuera. Antes de entrenar, congelar una copia (§7).
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

## 8. Segunda sesión de recolección (2026-11-26) y dataset final

**Recolección:**
- **Programa v3**, empezando por las 4 pendientes de v2.
- **Meta de la sesión: 112, pedida por el usuario.** La voz dice "N de 112" (`V8_GOAL=112 tools/hardware/v8_tmux.sh collect`) y avisa una sola vez al cumplirla. Se llegó a 116 de 220: unas 1,1 min por demo y ~2 h 40 min en total, con 3 reinicios del recolector para aplicar arreglos.
- **Se guardaron 124 demos** (`episode_128` a `episode_251`):
  - 5 descartadas: 164, 177 y 197 por la validación, con el OK del usuario, y 187 y 188 con [U];
  - 3 que DROID mandó a repetir y que no se marcaron: 237 (#106), 248 (#66, segunda vez) y 249 (#116). Quedan fuera del entrenamiento solas.
- **Las 6 demos que DROID mandó a repetir en la sesión son a 180°** (las 3 descartadas y las 3 sin marcar). Al girar media vuelta, la cámara mira el mantel blanco y DROID salta (22–187 mm). La #66 (celda 4) falló dos veces, a los 8,6 s y a los 8,8 s.
  - Sugerencia para la próxima: hacer el giro arriba, al principio, con los ArUco a la vista, y acercarse después.

**Cambios hechos durante la sesión:**
- **Límites del puente ampliados, a pedido del usuario:** x ≤ 0,70, |y| ≤ 0,35, radio ≤ 0,70 y alcance al 97 % del codo recto (antes 0,65 / ±0,30 / 0,65 / 95 %). Con los nuevos, 98 de las 114 demos de ayer terminan dentro (antes 80). Hay que llevarlos a la Jetson junto con el giro de ±177° y comprobar en la celda los bordes nuevos.
- **Aviso de alcance callado con la cámara quieta** (`reach_mute_when_still`). En celdas lejanas el final queda fuera por geometría, y el aviso repetido empujaba a mover la cámara justo en la quietud final.
- **El guardado automático espera también a que el GIRO esté quieto** (modo `imu`), y la llegada empieza cuando lo están giro y traslación. Antes se guardaba con la traslación quieta y el control del giróscopo rechazaba la demo completa: 4 rechazos en 46 intentos. Después del arreglo: 0 rechazos en 82 demos, con 8 esperas por giro en las que la voz dice "quieto, sin girar".
- **Inicio:** las primeras demos del día arrancaron a 8–11 cm de HOME3. La guía aceptaba el error en píxeles con escala 0,98–1,10, o sea, la cámara ~5–7 cm más baja. Se corrigió pidiendo una escala de ~1,00 al arrancar. Mediana del día: 57 mm (ayer 44).

**Dataset final** (`make_v8_training_set.sh`, 4 min 50 s):
- `datasets/demonstrations_real_v8_xarm5.zarr`: 230 demos (114 + 116) y 95 429 frames, todas con poses de DROID.
- 25 demos marcadas por el conversor; se conservan.
- Verificado LISTO contra `v7_xarm5_v2`. `verify_zarr_dataset.py` pasa en 230/230, con las 95 429 imágenes y un paso de DiffusionPolicy (pérdida 1,242).
- Cambio en el verificador: con `start_anchor: measured`, el paso 0 es el desfase medido respecto de HOME3, no [0,0,0]. Solo es error si pasa de `--max-start-offset-m`, 0,20 m por defecto.

**Métricas como las de v7** (`datasets/evaluations_real_v8_xarm5/README.md`, tabla v7 vs v8):
- **Poses de DROID frente a la odometría en vivo** (mismo código, mismas demos): 0 saltos > 30 mm y 0 discontinuidades > 20 mm, contra 709 y 652 en vivo (v7: 926 y 784).
- **Ruido que quita el filtro:** 0,9 mm (v7: 8,8). Llegada en movimiento: 11,9 % (v7: 21 %).
- **Altura final del TCP:** p5–p95 de 0,305 a 0,365 m, IQR 2,2 cm (v7: 9,7 cm).
- **Hallazgo principal, el alcance:**
  - El 21 % de los frames pasa el alcance del puente y el 15 % pasa el codo recto, es decir, es inalcanzable. Lo hacen 80 de 230 demos. Corrección del 2026-11-27 (§9): no es "casi todo al comienzo". Por fase, quedan fuera el 10 % de los frames del primer tercio, el 26 % del medio y el 21 % del último. La mano avanza alta antes de bajar y, en las celdas lejanas o laterales con la cámara girada, el final mismo queda fuera.
  - v7, con la misma extrínseca y los mismos límites: 3,3 % y 0,8 %.
  - Solo 12 finales quedan fuera de la caja.
  - El conversor marca una demo solo si más del 50 % de sus frames está fuera, así que marca solo 25.
  - Solo 129 demos quedan enteras dentro del alcance (137 con ≤ 5 % de frames fuera), y les faltan sobre todo 180° y ±90° en las celdas lejanas y laterales (figura 5).
  - Antes de entrenar hay que decidir: con todas, solo con las que quedan dentro del alcance, o regrabar bajando primero.
- **Herramientas:**
  - `tools/xarm5/analyze_v8_dataset_quality.py`: versión v8 de la auditoría de v7 (figuras 1–3 y 5). Compara odometría en vivo, DROID y DROID filtrado, y pone v7 y v8 en los mismos ejes.
  - `analyze_vio_dataset_quality.py --raw-diagnostics`: corre también sobre el armado con DROID. Se arregló el informe cuando no hay discontinuidades.

## 9. Revisión de entrenabilidad (2026-11-27, pedido del usuario)

**Datos para entrenar:** `datasets/demonstrations_real_v8_xarm5_2026-11-27.zarr`, copia congelada y de solo lectura del armado versión 3.
- La de ayer (`_2026-11-26`, versión 2, con las etiquetas de llegada sin corregir) quedó superada y **se borró el 2026-11-27 a pedido del usuario** (15 GB). Ningún `training_meta.json` la usaba. Queda como registro su informe `datasets/demonstrations_real_v8_xarm5_2026-11-26_conversion_report.json`.

**Revisión demo por demo** (`tools/xarm5/review_v8_trainability.py`):
- Usa la pose REAL de la cámara al final: la pose de inicio (ArUco + IMU) compuesta con DROID.
- Salidas en `datasets/evaluations_real_v8_xarm5/`: `trainability_review.json` y figuras 6–7 (README, sección "Trainability review").

| Control | Resultado |
|---|---|
| Destino | En 215 de 220 demos el eje óptico final cae en la celda asignada (±3 cm). Las otras 5 se pasan 3–5 cm del borde. Mediana: 4,8 cm del centro de la celda. Las etiquetas de celda y la cadena inicio + DROID son coherentes. |
| Altura final | De 26 a 32 cm sobre la mesa (p5–p95), como pide el protocolo. |
| Hojas arriba | Error mediano de 3,9° (p90: 17°). El peor caso es +45°, con 13° de mediana: la mano gira de menos. Los 6 desvíos grandes son de la primera tanda (programa v1). |
| Vistas finales | En las 230 (hoja de contactos, figura 6) la herramienta sale entera, centrada y con las hojas arriba, incluida la primera tanda. O sea, en v1 lo que no coincide es la etiqueta de orientación, y esa etiqueta no entra al entrenamiento. Por eso NO hay que excluir esas demos. |
| Llegada (done) | En el armado v2, en 44 demos el TCP se movía más de 3 cm dentro del done (33 con la llegada del recolector; peor caso 30 cm). El armado v3 (`trim_done_to_radius`) recorta el comienzo del done para dejar solo los frames a ≤ 3 cm de la pose final de la cámara; nunca lo alarga. Quedó con ≤ 4,1 cm en el TCP (p90 3,0) y llegada en movimiento del 6,5 % (antes 11,9 %). Hay 23 demos con 1–2 s de llegada (mínimo 0,9 s); `make_v8_training_set.sh` verifica con `V8_MIN_DONE_FRAMES` = 15. Solo cambió la columna done de esas 44 demos; el resto es idéntico bit a bit. |
| Imágenes | Ninguna borrosa. Son más nítidas que las de v7: 134 frente a 106 al final, y 142 frente a 99 en el frame más rápido. Brillo parejo. |
| Giro a 180° | 35 demos: 24 terminan en +177° y 11 en −177°. Son dos grupos en el estado; el robot llega a ambos. |
| Cuaternión | La acción relativa es inmune al signo (`rotations.py:121` normaliza el delta a qw ≥ 0). 15 demos a 180° cruzan la frontera del hemisferio 'home' en el ESTADO (1–2 veces cada una). Solo afecta a la entrada; mejora futura del modelo: una representación continua de la rotación. |
| Alcance del final | 187 de 228 finales son alcanzables y 27 están más allá del codo recto. 6 combinaciones tienen menos de 2 finales alcanzables: celda 4 a 180°; celda 7 a +90° y a 180°; celda 8 a 180°; celda 9 a −90° y a 180°. Más débiles: 6·−90, 7·−90, 7·−45, 8·−90 y 9·−45, con 2 cada una. |

**Veredicto:**
- **Se puede entrenar.** Con la llegada corregida no hace falta regrabar para empezar.
- **Regrabar sirve solo si esas 6 combinaciones deben funcionar en el robot.** Habría que hacerlas con un final alcanzable (en diagonal, desde más cerca del robot, como ya sugiere la guía), sumando las 3 demos a 180° que DROID rechazó: unas 25–30 demos.
- **Cuántas demos hacen falta** no se puede saber antes de entrenar. Hay que entrenar y comparar con la mitad de los datos.
