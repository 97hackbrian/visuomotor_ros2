---
name: v8-handheld-demo-collection
description: Procedimiento que funcionó el 2026-11-25 y el 2026-11-26 (230 demos válidas) para grabar demostraciones A MANO con la D435i para el xArm5. Cubre la sesión en tmux con voz y meta, la guía de inicio a HOME3, el programa de celdas × orientaciones, el bag crudo, el cierre, DROID, el armado del dataset de entrenamiento, las métricas y el congelado. Usar cuando el usuario quiera tomar más demos (v8 o un dataset nuevo), regrabar demos marcadas o preparar el dataset después de una sesión.
---

# Skill: demostraciones a mano con la D435i (v1.0, 2026-11-27)

**Fuente:** dos sesiones reales con 114 + 116 demos válidas, a ~1,1 min por demo. Detalle y cifras en
`.agents/knowledge/19_v8_handheld_dataset_pipeline.md`.

**Otras skills:**
- `realsense-vio-data-collection`: fondo general (red local, modos AUTO/AUMENTO, odometría en vivo).
- `droid-slam-offline-vio`: las poses offline.

## 0. Acordar con el usuario ANTES de lanzar

1. **Meta de la sesión** (número de demos), que va en `V8_GOAL`.
   - La voz dice "Llevan N de META", donde N son las entradas HECHAS del programa, acumuladas también de días
     anteriores. Ejemplo: si el programa tiene 116 hechas y se quieren 50 más, `V8_GOAL=166`.
   - Sin `V8_GOAL`, la voz dice el largo del programa (220), y el usuario no sabía cuánto le faltaba.
2. **El crudo se acumula** (modo AUMENTO): las demos del día se suman a `datasets/demonstrations_real_v8.zarr`.
   - "Guardada. Demo N" usa el número de episodio total, no el del día.
   - Decirlo al empezar: el 2026-11-26 el usuario creyó que se mezclaban dos datasets.
3. **Programa** (`experiments/collection_schedules/v8_schedule_v3.csv`): tiene 220 entradas, de las que 116 estaban
   hechas al 2026-11-26.
   - Para uno nuevo, continuar el anterior. Sin programa, el recolector graba sin celda ni orientación asignadas.
     ```bash
     docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/hardware/make_collection_schedule.py \
         --out experiments/collection_schedules/<nuevo>.csv --reps <n> --seed <s> \
         --continue-from experiments/collection_schedules/<anterior>.csv --dataset datasets/demonstrations_real_v8.zarr
     ```
   - El nuevo se elige con `V8_SCHEDULE=<csv>`.
4. **Regrabaciones dirigidas:** hacer un programa solo con esas entradas. Ejemplos:
   - las 6 combinaciones sin final alcanzable (4·180, 7·+90, 7·180, 8·180, 9·−90 y 9·180), con el final en diagonal;
   - las REPETIR.

## 1. Checklist física (`run_v8_session.sh` la imprime)

- Tablero ChArUco FUERA de la mesa: solo se usa para medir.
- Mantel y ArUco 12 y 14 en su lugar, sin moverlos: están calibrados con el robot (`table_markers.yaml`).
- Base del robot (Husky) donde estaba: sale en la vista de HOME3, como en v7 y en la inferencia.
- Luz fija encendida.
- D435i en un puerto USB 3. Las reglas udev (`tools/hardware/setup_realsense_udev.sh`) se instalan una sola vez.
- ≥ 30 GB libres: cada bag ocupa ~0,1–0,25 GB.
- `install/` al día. Si cambió el recolector o su YAML, recompilar:
  ```bash
  colcon build --symlink-install --packages-select visuomotor_data_collection
  ```
  `run_v8_session.sh` lo comprueba y da el comando.

## 2. Lanzar (agente, desde el host)

```bash
tools/hardware/v8_tmux.sh camera                  # sesión cam_v8: D435i + RTAB-Map (registro en experiments/collection_sessions/<fecha>/)
V8_GOAL=<meta> tools/hardware/v8_tmux.sh collect  # sesión v8: ventanas "voz" y "recolector" (chequeo previo + recolector)
tools/hardware/v8_tmux.sh guide                   # ventana "guia": rqt_image_view de /vo_collector_node/start_guide
tools/hardware/v8_tmux.sh show                    # abre una terminal enganchada para que el usuario teclee
tools/hardware/v8_tmux.sh log recolector 60       # el agente lee el recolector (también camara o voz)
tools/hardware/v8_announcer.sh --test             # probar los parlantes
tools/hardware/v8_tmux.sh grid                    # opcional: ajustar la grilla (reiniciar el recolector al guardar)
```

La cámara corre en una sesión aparte (`cam_v8`): unos Ctrl-C en la terminal compartida la apagaron una vez.

## 3. Cada demo (lo hace el usuario; la voz lo guía)

1. **Herramienta en la celda AMARILLA** de la grilla numerada que la guía dibuja sobre la mesa.
   - La flecha marca hacia dónde apuntan las HOJAS; el punto negro es el mango.
   - Orientaciones: 0 = de frente (lejos de la base), +90 = DERECHA, −90 = IZQUIERDA, 180 = hacia la base.
   - No tapar nunca los ArUco; las celdas 7 y 9 los contienen.
2. **Cámara a la vista de HOME3** hasta "ALINEADA": 20 px de tolerancia y escala ~1,00, o sea, no más baja. Con 1 s
   quieta empieza sola.
3. **"Quieta: reiniciando odometría": NO moverse.** Esos frames dan la pose real de inicio (ArUco + IMU).
4. **"Grabando": bajar suave**, a ≤ 15 cm/s; si no, la voz dice "Más lento".
5. **A 180°, girar arriba y al principio**, con los ArUco a la vista, y después acercarse. Girar media vuelta mirando
   el mantel blanco es el caso en que DROID falla: las 6 REPETIR del 2026-11-26 fueron a 180°.
6. **Terminar a 25–30 cm SOBRE la herramienta**, centrada y con las HOJAS hacia ARRIBA en la imagen.
   - En celdas lejanas o de borde, la voz pide "final en diagonal": terminar desde más cerca del robot, porque así
     llega el brazo.
7. **Quedarse quieta ~2,5 s y SIN GIRAR**; si gira, la voz dice "Quieto, sin girar". Guarda sola y dice "Guardada.
   Demo N".
8. **Teclas:**
   - [S] guarda siempre: es el respaldo si no guarda en ~5 s.
   - [D] descarta y repite.
   - [U] descarta la última guardada: la marca, no la borra, y solo funciona entre demos.
   - [N] salta la entrada del programa.
   - [I] empieza ignorando la guía (queda registrado).
   - [H] marca la llegada a mano.
   - [F] fuerza el guardado.
   - [G] cambia el gripper.
   - [1–9] elige la celda.
   - [Q] sale.
9. **Si la odometría en vivo se pierde cerca de la mesa**, es solo un aviso y la demo vale: las poses del dataset
   salen después de DROID.

## 4. Durante la sesión (agente)

- Leer el recolector cada pocas demos (`v8_tmux.sh log`). Buscar:
  - rechazos;
  - "[ALCANCE] FUERA";
  - cancelaciones;
  - demos sin programa.
- Para cambiar código o parámetros: `v8_tmux.sh stop recolector`, recompilar y volver a `collect`. El programa
  continúa solo y la cámara puede seguir corriendo.
- No descartar nada por cuenta propia. Proponerlo y esperar el OK del usuario; [U] y [D] son decisiones suyas.

## 5. Cierre y dataset (agente)

```bash
tools/hardware/v8_tmux.sh stop                              # todas las ventanas; la cámara, la última
pgrep -af "[v]o_collector|[r]tabmap|[r]ealsense"; docker ps  # comprobar que no quedó nada (sin visuomotor_vo_active)
tools/xarm5/make_v8_training_set.sh                         # DROID en las nuevas → inicio → informe → armado → conversión → verificación → reemplazo
```

- **REPETIR y REVISAR:** salen en el informe (paso 3 del script). Las REPETIR no entran al dataset.
  - Para regrabarlas, darle al usuario la celda, la orientación y el número de programa.
  - Marcarlas en el crudo requiere su OK.
- **Métricas como las de v7:** sección "Regenerate" de `datasets/evaluations_real_v8_xarm5/README.md`.
- **Revisión de entrenabilidad:** `tools/xarm5/review_v8_trainability.py`.
- **Antes de entrenar:** `tools/xarm5/freeze_training_set.sh demonstrations_real_v8_xarm5`, y entrenar con la copia
  con fecha.
- **Copias congeladas nunca usadas:** borrarlas dejando registro, por regla del usuario.
- **Al cerrar:** skill `update-knowledge`, paso 0 (`.agents/STATE.md`).

## 6. Cifras de referencia

| | 2026-11-25 | 2026-11-26 |
|---|---|---|
| Demos válidas | 114 | 116 (meta 112) |
| Tiempo | ~2 h 20 min para 126 grabadas | ~2 h 40 min para 124 guardadas |
| Descartadas | 14 | 5, más 3 REPETIR sin marcar |
| Rechazos del guardado automático | 4 de 46 intentos antes del arreglo del giro | 0 de 82 después |
| Inicio: mediana de la distancia a HOME3 | 44 mm | 57 mm |
| DROID: saltos > 30 mm o discontinuidades > 20 mm | 0 | 0 |

## 7. Errores vistos y cómo se detectan

- **La voz decía el largo del programa (220), no la meta:** usar `V8_GOAL`.
- **Ventanas en la sesión equivocada:** tmux compara sesiones por prefijo ("v8" encontraba "v8_cam"). `v8_tmux.sh`
  usa objetivos exactos (`=nombre`).
- **Un log de rclpy que cambia de severidad en la misma línea** lanza `ValueError` y tumba el ejecutor (AGENTS §17).
- **`install/` desactualizado:** `run_v8_session.sh` se detiene y da el comando de recompilación.
- **Sin el servicio `reset_odom`:** no arranca, porque cada demo empezaría sin reiniciar la odometría.
- **Inicios 5–7 cm más bajos:** la guía aceptaba escalas de 0,98 a 1,10. Ahora pide escala ~1,00.
- **Guardados con la cámara girando**, que el control del giróscopo rechazaba: el guardado espera también a que el
  giro esté quieto (v3.2).
- **El aviso de alcance, repetido en la quietud final, empujaba a mover la cámara:** `reach_mute_when_still`.
- **Alcance del puente demasiado estricto para las demos:** se amplió a pedido del usuario (x ≤ 0,70, |y| ≤ 0,35,
  r ≤ 0,70, 97 %). Llevarlo a la Jetson antes de ejecutar políticas.
