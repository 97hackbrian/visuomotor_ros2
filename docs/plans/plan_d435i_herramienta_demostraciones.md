# Plan de mejora: la D435i como herramienta confiable de demostraciones expertas (aproximación)

Aprobado el 2026-11-23. El plan original está abajo, sin cambios; esta tabla lleva el avance.

## Estado de avance (2026-11-24)

| Fase | Hecho (con pruebas) | Pendiente |
|---|---|---|
| 0. Diagnóstico y documentos | `--raw-diagnostics` en `tools/xarm5/analyze_vio_dataset_quality.py` (`tests/test_vio_raw_diagnostics.py`), con resultados en `datasets/evaluations_real_v7_xarm5_v2/raw_diagnostics.*`. Corregidos AGENTS §17/§20 y comandos, K15 (§0 nuevo), skill v1.4 y `context.md` (lección 43). Quitado `ultra_precision` de los launch de la D435i; el YAML `_best_umi` queda marcado como histórico | H4 (distancia final cámara–mesa): requiere medir el lado de los ArUco de la mesa (`--aruco-marker-m`) |
| 1. Escena y cámara | `charuco_board.yaml` (**A3 9×7, cuadro de 36 mm**, elegido con datos el 2026-11-24), `visuomotor_bringup/charuco.py`, `tools/hardware/make_charuco_board.py` (`tests/test_charuco.py`); `tools/hardware/read_camera_auto_settings.py` (exposición/ganancia/balance automáticos -> manuales) | Imprimir en A3 (PDF corregido el 2026-11-24), medir con calibre y colocarlo para medir (centro de la mesa; las demos van sin tablero); luz fija; escribir la exposición y el balance manuales en `d435i_camera_params.yaml` |
| 2. Medición con referencia | Métricas `trajectory_metrics.py` (+ `evaluate_paired`, RPE por tiempo con huecos); `tools/hardware/record_vio_bag.sh`; `tools/hardware/evaluate_pose_estimates.py` (referencia del tablero, pares por sello, resumen por estimador); bag sintético con verdad `tools/hardware/make_synthetic_vio_bag.py`. Pruebas: `tests/test_trajectory_metrics.py`, `tests/test_vio_measurement_pipeline.py`. Calibración `tools/xarm5/calibrate_hand_eye.py` (`tests/test_hand_eye.py`) | Corridas a mano contra el tablero (sin robot); con el robot: captura de la calibración cámara-brida y trayectorias reproducidas por el xArm5 (`record_vio_ground_truth.py`) |
| 3. Elección del estimador | `tools/hardware/replay_odometry_candidates.py`: reprocesa los mismos bags con cada candidato de `vio_measurement.yaml` (base, ajuste de haces, GuessMotion, F2F, flujo óptico, `Vis/MinDepth` 0,15, sin Kalman, sin IMU) y llama al evaluador. Probado de punta a punta con el bag sintético | Correr sobre los bags reales y decidir con la regla de la Fase 3 |
| 4. Recolector | **v3:** guía de inicio a la vista de HOME3 (ArUco de la mesa, referencia de los 14 intentos reales), inicio y guardado automáticos, aviso de velocidad, programa de celdas × orientaciones, imagen de guía `~/start_guide`. Corregidos: caída del ejecutor por cambiar la severidad de un log en la misma línea (v2), `/tf_static` no se grababa en los bags crudos, `run_data_collection.sh` no cargaba el YAML. Pruebas: `tests/test_vo_start_guide.py`, `tests/test_vo_schedule.py`, `tests/test_vo_collector_node_sim.py` (v2 y v3) | Probar con la cámara real; control de ejecutabilidad en vivo (núcleo de `Converter`); `build_dataset_from_bags.py` (tras la Fase 3) |
| 5. Protocolo y piloto | Programa de recolección (`tools/hardware/make_collection_schedule.py`); protocolo abajo ("Trabajo sin el robot") | Piloto de 10–15 demos y comparación con v7 |
| 6. Documentación | Fase 0; recolector v3 y cadena de medición (2026-11-24) | Skill v2.0 y K15 con las mediciones reales |

**Hallazgos de la Fase 0 (v7 crudo):**
- Los saltos de 0,51 m de los episodios 23 y 24 son poses nulas de RTAB-Map guardadas como poses (4 frames).
- Hay 784 discontinuidades persistentes > 2 cm (mediana 33 mm) en 71/83 episodios.
- 72/83 episodios tienen done solo en el piso forzado de 40 frames, y el ~21 % de esos frames está en movimiento.
- No hay sellos de tiempo.
- Solo hay 4 estados y 0 imágenes consecutivas duplicadas. La hipótesis de duplicados al pasar de 30 a 20 Hz queda
  descartada.

**Hallazgo de la prueba del recolector:**
- Dos callbacks de imagen que terminan en orden inverso hacían perder la imagen que llegaba tarde. Corregido.
- Con FastDDS, el suscriptor en Python perdió ráfagas de imágenes; con CycloneDDS, que usan los scripts, no.

## Trabajo sin el robot (2026-11-24)

El robot no está disponible; solo la cámara. Todo lo de abajo funciona sin el xArm5. Lo que requiere el robot (la
calibración cámara-brida y las trayectorias reproducidas por el brazo) queda para después.

### Lo que dejaron los 14 intentos en el xArm5 (2026-11-23) para v8
- **Inicio:** el robot arranca siempre desde la misma vista (HOME3; los ArUco de la mesa se repiten con < 1 px). Las
  demos v7 empezaban dispersas: escala 0,88–1,26 y corrimiento de hasta ~80 px (en 640) respecto a esa vista.
  -> Guía de inicio del recolector v3.
- **Cobertura:** los fallos fueron con las tijeras giradas (intentos 7, 9 y 10) y junto a la base (8 y 11), donde v7
  no tenía demos. -> Programa de celdas × orientaciones, en orden aleatorio.
- **Parada:** la altura a la que la persona se detiene sobre un mismo lugar varió ±7 cm. En el intento 12, la red
  reconoció el final solo un instante y siguió bajando. -> Llegar siempre a la misma vista final: la herramienta
  centrada y llenando el recuadro central de la imagen de guía, y quieta al menos 2 s.
- **Escena:** con el mantel corrido y sin ArUco (intento 14) la señal de término no pasó de 0,11. -> La escena de las
  demos (mantel y los dos ArUco, como en v7 y en los ensayos) debe ser la misma en la inferencia.
- **Velocidad:** en v7, la velocidad máxima por episodio tuvo mediana 0,11 m/s (p90 0,15). En los ensayos el puente
  limitó a 0,05 m/s. -> Aviso por encima de 0,15 m/s.

### Hallazgo con el bag sintético: el tamaño del tablero
Error de posición de la referencia del tablero contra la verdad, en una aproximación sintética (cámara ideal, solo
compresión JPEG). Mediana en mm:

| Tablero | < 0,45 m | 0,45–0,60 m | 0,60–0,75 m | 0,75–1,2 m (HOME3) |
|---|---|---|---|---|
| A4 7×5, cuadro de 33 mm (el anterior) | 0,35 | 1,2 | 2,3 | 14,0 |
| A4 9×5, cuadro de 30 mm (alternativa) | 0,33 | 0,9 | 2,3 | 6,1 |
| **A3 9×7, cuadro de 36 mm (el elegido)** | 0,11 | 0,4 | 0,7 | 1,7 |

- Con el A4 anterior, la referencia no servía a la distancia de HOME3: el primer frame salía ~1° girado y eso
  propagaba ~9 mm al anclar.
- Pesa más el número de esquinas (48 interiores frente a 24) que el tamaño del cuadro.
- Con una cámara real el error será mayor. Por eso la primera corrida de cada sesión es la cámara quieta, para
  medirlo.

### Cadena de medición, probada sin cámara
Usa el bag sintético con verdad conocida (`tests/test_vio_measurement_pipeline.py`):
- el evaluador recupera exactamente la escala (+3,00 %) y las pérdidas (3,9 %) inyectadas en la odometría simulada;
- el reprocesado reproduce el bag:
  - descomprime JPEG y compressedDepth;
  - pasa por Madgwick y `rgbd_odometry`;
  - graba y evalúa.
- RTAB-Map base sobre la escena sintética: 4,5 mm de mediana y 9,3 mm al final.

Es una escena fácil: esto valida las herramientas, no la odometría real.

### Pasos con la cámara conectada (en orden)
1. **Recompilar** el recolector si cambió (la instalación de `visuomotor_data_collection` es una copia, no
   enlaces): `colcon build --symlink-install --packages-select visuomotor_data_collection visuomotor_bringup`.
2. **Tablero** (solo para las corridas de medición; ver "Dónde va el tablero"):
   - Generarlo con `python3 tools/hardware/make_charuco_board.py`, que crea un A3 en `docs/hardware/`.
   - Imprimirlo al 100 % y pegarlo sobre una placa plana.
   - Medir varios cuadros y marcadores con calibre y escribir la media en `charuco_board.yaml` con `measured: true`.
3. **Luz y cámara:**
   - Con la luz definitiva, correr `read_camera_auto_settings.py` en el contenedor de la cámara.
   - Escribir la exposición, la ganancia y el balance manuales en `d435i_camera_params.yaml`.
   - Llevar el mismo cambio a la Jetson.
4. **Medición base:**
   - Con la cámara y RTAB-Map corriendo, poner el tablero en el centro de la mesa (ver "Dónde va el tablero").
   - Dejar `check_board_view.py` abierto en otra terminal: muestra si la vista coincide con HOME3 y cuántas
     esquinas se ven.
   - Grabar con `record_vio_bag.sh`, que reinicia la odometría al empezar, como el recolector.
     - `reposo_home3` 15 s: la cámara quieta en la vista de HOME3. Da el ruido de la referencia y de la odometría.
     - `reposo_cerca` 15 s: lo mismo, a ~0,4 m del tablero.
     - ≥ 10 aproximaciones tipo demo (`aprox_NN`):
       - empezar en la vista de HOME3 y quedarse quieta ~2 s;
       - bajar hasta ~25–30 cm sobre el tablero, como en una demo y a su velocidad;
       - quedarse quieta ~2 s y detener con Ctrl+C.
       - Variar la trayectoria entre corridas: llegar desde un lado, con algo de giro, más rápido o más lento.
5. **Evaluar y elegir el estimador:**
   - `evaluate_pose_estimates.py` mide la odometría en vivo.
   - `replay_odometry_candidates.py` reprocesa con todos los candidatos.
   - Regla de la Fase 3: gana la menor mediana de ATE anclado que cumpla el p90 y no tenga pérdidas. Si dos están a
     menos del 10 %, se elige el más simple.
   - Pasar el ganador a `rtabmap_vio_params.yaml`.
6. **Piloto v8 (10–15 demos):**
   - Con la base móvil del robot estacionada frente a la mesa, como en v7 (ver "Riesgos nuevos").
   - Lanzar con `COLLECTOR_ARGS="-p auto_start:=true -p auto_save:=true -p schedule_file:=<csv> -p
     raw_bag_enabled:=true" ./tools/hardware/run_data_collection.sh demonstrations_real_v8`.
   - Mirar la guía con `rqt_image_view` en `/vo_collector_node/start_guide`.
   - Después: control de calidad, conversión (`convert_vio_dataset_to_xarm5.py`), 20 épocas y comparación con v7
     contra las líneas base exigentes.
7. **Toma completa de v8** solo si el piloto pasa. Los bags crudos permiten recalcular las poses si después gana
   otro estimador.

### Dónde va el tablero (cambio respecto a la Fase 1 original, 2026-11-24)
El plan original dejaba el tablero fijo en la escena también para las demos y la inferencia. Se cambia, por dos
razones:
- La guía de inicio usa los dos ArUco de la mesa; ya no necesita el tablero.
- Si el tablero queda en la escena, cambia lo que ve la política: habría que ponerlo también en la inferencia, y v8 ya
  no se parecería a v7 ni a los ensayos.

**Uso del tablero:**
- **Corridas de medición:** plano sobre la mesa, en la zona central donde va la herramienta, sin tapar los dos ArUco.
  Las aproximaciones terminan sobre él, como si fuera la herramienta.
  - Así se ve desde la vista de HOME3 (~0,85 m) y también cerca al final.
  - Bien sujeto con cinta en las esquinas, que no se mueva durante la sesión.
  - Comprobar con `tools/hardware/check_board_view.py`: desde la vista de HOME3 deben verse las 48 esquinas.
- **Demos v8:** sin tablero. La misma escena que en v7 y en los ensayos: mantel, los dos ArUco, la herramienta y la
  base móvil del robot frente a la mesa.
- **Con el robot:** el tablero sirve para la calibración cámara-brida. Después se retira, y las trayectorias de
  referencia se miden con el robot sin tablero en la escena.

**Preparación:**
- Imprimirlo al 100 % ("tamaño real", sin "ajustar a la página"), en papel mate y sin modo de ahorro de tóner.
- Comprobar la regla de 100 mm impresa.
- Pegarlo sin burbujas sobre una placa rígida y plana (cartón pluma, MDF o acrílico de ≥ 3 mm). Un tablero curvo
  sesga la pose.
- Medir con calibre el largo total de los 9 cuadros (324 mm nominal) y de los 7 (252 mm), y dividir.
- Medir 3–4 marcadores (27 mm nominal).
- Escribir las medias en `charuco_board.yaml` con `measured: true`.

### Riesgos nuevos
- **Vista de HOME3:** la referencia sale de imágenes tomadas con la cámara en el robot y usa solo los dos ArUco de la
  mesa. No despegar ni mover los ArUco sobre el mantel. Cuando vuelva el robot, estacionarlo en el mismo lugar
  respecto de la mesa: con el brazo en HOME3, `check_board_view.py` debe dar ALINEADA. Si no se puede, regenerar la
  referencia con `make_home3_view_reference.py` e imágenes nuevas en HOME3.
- **La base del robot en la imagen** (aclarado por el usuario, 2026-11-24): el xArm5 va montado sobre una base móvil
  tipo Husky A200. En HOME3, la cámara de la brida ve abajo el frente amarillo de la base, una Kinect 360 encima y los
  salientes de las ruedas, ~18 % de la imagen (`experiments/xarm5_real_trials/2026-11-23/trial_001/pre.png`).
  - Sale en toda inferencia, porque es el propio robot, y en el primer frame de las demos v7, grabadas con la base
    frente a la mesa.
  - No sale en la vista final sobre la herramienta, ni siquiera junto a la base (ensayos 8 y 11). La guía de inicio
    no la usa.
  - Demos v8: grabarlas con la base en su lugar. Sin ella cambia la franja inferior de los primeros frames, justo
    cuando la política decide hacia dónde ir. Si hay que grabar sin la base, entrenar con un aumento que tape o
    reemplace esa franja.
  - Las corridas de medición con el tablero no la necesitan.
- **Medición optimista:** en las corridas de medición el tablero agrega textura, y la odometría rastrea mejor que en
  las demos (mantel blanco). Sirve para ordenar a los candidatos. La exactitud absoluta en la escena real se mide
  después con el robot como referencia (Fase 2.2), sin tablero en la escena.
- **Disco:** con los bags crudos, ~10 MB/s mientras se graba un episodio (100 demos de 20 s ≈ 20 GB).

## Contexto

Las demostraciones reales de v7 se grabaron a mano con la D435i y la odometría RGB-D de RTAB-Map. La Fase 22 tuvo
que rescatar ese dataset a posteriori: la cámara estaba invertida 180°, el cuaternión cambiaba de signo, el ruido en
movimiento era de unos 8,8 mm RMS y hubo pérdidas de rastreo. Además, los `.md` presentan como "certificado" un
rendimiento que los propios registros del repo no sostienen.

El objetivo es una herramienta **confiable** (no guarda datos malos sin avisar), **precisa** (error medido contra una
referencia, no estimado) y **útil** (lo grabado es ejecutable por el xArm5 y da mejores políticas).

**Alcance fijado por el usuario:**
- Solo aproximación y solo la cámara. No aparece ninguna garra en la imagen, ni en la demostración ni en la inferencia.
- La referencia de medición es el xArm5 más un tablero ChArUco.
- Cambios físicos aprobados: tablero ChArUco fijo e iluminación fija. No se hará un mango nuevo ni se usará pedal.
- El estimador de pose se elige con datos: RTAB-Map puede reemplazarse.

## Diagnóstico crítico (afirmaciones de los .md frente a la evidencia del repo)

| # | Afirmación (dónde) | Evidencia | Veredicto |
|---|---|---|---|
| 1 | "Certificación VIO grado UMI: retorno 2,307 mm" (AGENTS §20, K15 §10.D, skill §2) | `experiments/vio_benchmark_history.json`, 9 corridas dinámicas de la conf 4: 2,3 / 14,4 / 16,2 / 16,9 / 21,6 / 28,3 / 39,3 / 59,0 / 59,7 mm. Mediana 21,6 mm; solo 1 de 9 cumple ≤ 3 mm, y es la trayectoria más corta (279 cm) | Es el mejor caso presentado como certificación. No está certificado |
| 2 | El error de retorno como criterio de aceptación | Un error de escala no mueve el punto de retorno. F2M con `OdomF2M/MaxSize 5000`, ajustado "para el cierre contra la cuna", se relocaliza contra los puntos de t=0 | No mide ni la trayectoria ni la escala. Criterio inválido |
| 3 | Jitter de 1,61 mm (y 0,62 mm en K15 §9) | Conf 4 en reposo: mediana 3,0 mm (rango 1,6–6,1). En movimiento: ~8,8 mm RMS (K17 §1) | Es el mejor caso; en movimiento es ~5 veces peor |
| 4 | "0 % de pérdida de rastreo" | v7: saltos de 3–9 cm por frame y de 0,45–0,68 m (episodios 23 y 24) | El banco de prueba no representa la toma real |
| 5 | "VIO; el eje Z queda estrictamente vertical" (K15 §9) | El IMU solo inicializa la gravedad (`wait_imu_to_init`). Con el ajuste de haces apagado no hay restricción de gravedad. La vertical de odom derivó 50–60° (K17 §1.4). K15 §9 dice `AlignWithGround: true` y después `false` | Es odometría visual RGB-D con el marco inicial alineado a la gravedad, no VIO |
| 6 | "GFTT/ORB se ancla a la trama IR" (K15 §6) | El rastreo usa la imagen RGB (OV2740, rolling shutter). Un patrón proyectado se mueve con la cámara: si se viera, sesgaría el movimiento estimado hacia cero | Sin verificar; de ser cierto sería un defecto, no una ventaja |
| 7 | "Apagar el ajuste de haces evita la deriva de escala" (K15 §10.B; docs/research §2.5) | Las confs 0–3 tienen 1 o 2 corridas cada una, con trayectorias distintas (279–1360 cm) | Hipótesis sin experimento controlado |
| 8 | "20 Hz sincrónicos" y la "auditoría de frecuencia" (skill §5) | El recolector no guarda sellos de tiempo y `verify_dataset_frequency.py` asume dt = 0,050. El timer toma la última imagen y la última pose por separado, sin emparejarlas; pasar de 30 a 20 Hz puede duplicar o saltar frames | Afirmación circular, imposible de verificar |
| 9 | `min_inliers_tracking: 10` (YAML del recolector) | Se declara y nunca se usa en `vo_collector_node.py` | No hay ningún control de rastreo al grabar |
| 10 | Pose nula al perder el rastreo | RTAB-Map publica por defecto una odometría nula cuando se pierde. El recolector convierte el cuaternión nulo en identidad y la posición 0, lo que da un salto igual a −pose inicial. Como odom no se reinicia entre episodios, ese salto crece con la sesión | Hipótesis fuerte para los saltos de 0,45–0,68 m: verificar |
| 11 | "Piso de done garantizado" | `_save_episode` pone done=1 en los últimos 40 frames aunque la cámara siga moviéndose | Ruido en las etiquetas |
| 12 | "Inicio = HOME3" | El primer frame se asigna a HOME3. La inclinación inicial medida fue 32° ± 6° frente a 24,6°, y la posición inicial nunca se midió | Error sistemático de anclaje |
| 13 | Extrínseca cámara→TCP marcada `extrinsic_calibrated: true` | La rotación se infirió de la gravedad y los ArUco; la traslación es [0,0,0] sin medir | Calibración incompleta |
| 14 | Deudas menores | Exposición y balance de blancos automáticos; `qw ≥ 0` al grabar; metadatos mínimos (`'sensor': 'Intel RealSense'`); sin datos crudos para reprocesar; `ultra_precision` no hace nada en el launch de la D435i; `rtabmap_vio_params_best_umi.yaml` es una copia idéntica del activo; AGENTS §17 dice subpíxel 3×3/10 y el YAML usa 5×5/15 | Corregir |

**Hallazgo adyacente (afecta la prueba del robot):** la imagen de la Jetson (`deploy/jetson/Dockerfile.jetson`) no
instala `realsense2_camera`/librealsense, y ningún launch del robot arranca la cámara. Hay que confirmarlo en la Jetson
con `ros2 pkg list | grep realsense`.

## Objetivo y criterios de aceptación

Los umbrales son propuestos y se fijan tras la medición base de la Fase 2. Siempre se reportan como distribución
(mediana, p90 y máximo de al menos 10 corridas), nunca como el mejor caso.

- **Precisión** frente al xArm5, en trayectorias de tipo demostración:
  - ATE anclado al primer frame (es como el pipeline usa las poses): mediana ≤ 5 mm y p90 ≤ 10 mm;
  - RPE a 0,5 s ≤ 3 mm;
  - error de escala ≤ 1 %;
  - error de orientación ≤ 1° (mediana).
- **Fiabilidad:** ningún frame perdido o nulo en un episodio guardado, y ≥ 95 % de episodios aceptados por el control de calidad (QA) en cada sesión.
- **Tiempo:** sello de tiempo por frame, imagen y pose del mismo instante, muestreo de 50 ± 2 ms y ningún frame duplicado.
- **Inicio:** a ≤ 10 mm y ≤ 2° de la vista de HOME3, medido con el tablero.
- **Etiquetas:** todos los frames con done=1 son estáticos (velocidad bajo el umbral).
- **Ejecutabilidad:** ningún frame fuera del alcance o de la caja del puente, y orientación y velocidad dentro de sus límites.
- **Utilidad:**
  - error de reproducción en el robot ≤ 10 mm (mediana);
  - el piloto bate a la velocidad constante en FDE, y su FDE sube claramente con imágenes cruzadas.

**Decisión de arquitectura:** la odometría en vivo solo sirve para guiar al operador y para el control de calidad. Las
poses del dataset se calculan después, offline, a partir de la grabación cruda de cada episodio y con el estimador que
gane la Fase 3. Así la precisión no depende del tiempo real y siempre se puede reprocesar.

## Fases

### Fase 0: sin hardware (se puede empezar ya)
1. **Diagnóstico de v7 crudo** (solo lectura). Ampliar `tools/xarm5/analyze_vio_dataset_quality.py` con
   `--raw-diagnostics`, reutilizando `episode_keys` y `aruco_axis_tilt`. Las salidas van con nombres nuevos en
   `datasets/evaluations_real_v7_xarm5_v2/`. Comprobar:
   - H1, poses nulas: cuaternión exactamente identidad y posición ≈ −pose inicial.
   - H2, duplicados al pasar de 30 a 20 Hz: estados o imágenes consecutivas idénticas.
   - H3, frames con done=1 en movimiento.
   - H4, distancia final cámara–mesa frente a `Vis/MinDepth` y al mínimo de profundidad de la D435i (requiere medir el lado de los ArUco).
2. **Corregir los .md ahora:** AGENTS §17 y §20, K15 §6, §9 y §10, la tabla §2 y §5 de la skill, y `context.md`.
   - Quitar "certificado" y "grado UMI".
   - Dar la distribución real de las corridas.
   - Llamar al método "odometría RGB-D con marco inicial alineado a la gravedad".
   - Marcar las hipótesis 6 y 7 como no verificadas.
   - Advertir que la auditoría de frecuencia es circular.
3. **Limpieza:** eliminar o implementar `ultra_precision` en `realsense_d435i_vio.launch.py`, quitar el YAML duplicado
   (queda en el historial de git) y alinear los valores de subpíxel en la documentación.

### Fase 1: escena y cámara (hardware aprobado)
- **Tablero ChArUco:**
  - `tools/hardware/make_charuco_board.py` genera un PDF a escala 1:1 con `cv2.aruco.CharucoBoard`.
  - Se imprime, se pega a una placa rígida y plana, y se miden los cuadros con calibre. Las medidas van a
    `visuomotor_bringup/config/realsense/charuco_board.yaml`.
  - Se fija en la celda de modo que se vea desde HOME3 y durante la aproximación.
  - **Se queda también durante la inferencia**: la escena tiene que ser igual en las demos y en el robot.
- **Luz fija y cámara manual:** en `d435i_camera_params.yaml`, exposición corta (contra el desenfoque y el rolling
  shutter), ganancia y balance de blancos fijos. Se miden una sola vez con la luz nueva y se guardan en los atributos
  de cada episodio.

### Fase 2: medición con referencia (xArm5 y tablero)
- **Requisito:** driver de la D435i en la imagen de la Jetson (librealsense con backend RSUSB y `realsense2_camera`).
  Sirve también para el despliegue. Se graba en la Jetson, con un solo reloj, usando `ros2 bag record` (MCAP) y la
  lista de tópicos en YAML: color, profundidad, infra1 e infra2, IMU, `camera_info`, estado del robot y TF. El
  procesamiento se hace en el portátil.
1. **Calibración cámara–brida** (`tools/xarm5/calibrate_hand_eye.py`):
   - El usuario mueve el brazo con el marcador de RViz que ya existe, a 15–25 poses.
   - En cada parada la herramienta toma la TF `link_base→xarm_tcp` y detecta el ChArUco.
   - Calcula con `cv2.calibrateHandEye` (Park y Tsai, comparados entre sí) e informa el residuo.
   - Entrega tres resultados:
     - la rotación y la traslación medidas para `xarm5_dataset_alignment.yaml`, que se escriben solo tras revisarlas;
     - `T_base_tablero`;
     - la vista de referencia `T_tablero_cámara` en HOME3.
2. **Trayectorias con referencia** (`tools/xarm5/record_vio_ground_truth.py`):
   - El robot reproduce demos de v7 ya convertidas (están en el marco del robot y son ejecutables) publicando
     `target_frame` al puente, con su caja y sus límites de velocidad.
   - Se añaden tramos estáticos y de solo giro.
   - Al menos 10 corridas, con la exposición fija y un subconjunto con exposición automática.
3. **Corridas a mano** (al menos 10) con el tablero como referencia, para medir la dinámica real de la mano. Antes se
   valida esa referencia contra el robot con las corridas del punto 2.
4. **Métricas:**
   - Módulo `visuomotor_core/visuomotor_core/analysis/trajectory_metrics.py`, en NumPy y reutilizando
     `visuomotor_core/utils/rotations.py`: ATE anclado al primer frame, ATE con alineación Umeyama SE(3) y Sim(3)
     (de donde sale la escala), RPE a 0,25, 0,5 y 1 s, porcentaje de pérdidas y latencia.
   - Herramienta `tools/hardware/evaluate_pose_estimates.py`: JSON y figura en inglés, y registro en `experiments.db`.

### Fase 3: elección del estimador (offline, sobre los mismos bags)
- **Candidatos:**
  - A: RTAB-Map actual (línea base).
  - B: variantes de A: ajuste de haces, `GuessMotion` y flujo óptico.
  - C: RTAB-Map con la IR izquierda (global shutter) y profundidad, con el emisor apagado en un subconjunto grabado.
  - D: localización por tablero (PnP) cuando se ve, con odometría para los huecos.
  - E: VIO acoplado (OpenVINS u ORB-SLAM3 estéreo-inercial). Solo si A–D no alcanzan el objetivo; requiere
    `rs-imu-calibration` y calibrar cámara–IMU con Kalibr.
- **Medir también** cuánto aporta el filtro offline actual (`visuomotor_core/dataset/trajectory_filters.py`): si
  reduce el error o solo el ruido.
- **Regla de decisión:** gana la menor mediana de ATE anclado que cumpla el p90 y tenga 0 pérdidas. Si dos están a
  menos del 10 %, se elige el más simple.

### Fase 4: recolector v2 (`vo_collector_node.py` y sus YAML)
La lógica pura va a `visuomotor_data_collection/vo_episode_logic.py`, que se puede probar sin ROS.
1. **Grabación cruda por episodio** con rosbag2 en MCAP:
   - Tópicos definidos en YAML.
   - Color comprimido en JPEG de alta calidad y profundidad PNG sin pérdida.
   - Se borra al descartar el episodio, y su ruta queda en los atributos.
2. **Emparejamiento por sello de tiempo:** RTAB-Map publica la odometría con el sello de la imagen.
   - Muestreo a 20 Hz por tiempo: pose interpolada (SLERP) e imagen a ≤ 1/60 s.
   - Se guardan los arrays `timestamp`, `image_stamp` y `pose_stamp`, y se rechazan los duplicados.
3. **Calidad de rastreo:**
   - Suscripción a `odom_info`; se usa por fin `min_inliers_tracking`.
   - Una pose nula se marca como perdida y nunca se convierte en identidad.
   - Se guardan `inliers` y `lost` por frame; si hay pérdidas, el episodio no se guarda salvo confirmación (parámetro).
4. **Reinicio de odometría al inicio** de cada episodio con el servicio `reset_odom`, con la cámara quieta. Se guarda
   la gravedad media del IMU, que da la inclinación real de inicio.
5. **Guía de inicio con el tablero:**
   - Una línea en la terminal muestra Δx/Δy/Δz y Δángulo respecto a la vista de HOME3 de la Fase 2.1.
   - START solo se acepta dentro de la tolerancia.
   - `auto_start` opcional: arranca solo cuando la cámara lleva 1 s alineada y quieta. Así no hace falta tocar el
     teclado, que fue el problema que motivaba el pedal.
6. **Etiqueta done:** empieza en HOLD (manual o detectado por quietud, `auto_hold`) y se verifica que esos frames
   sean estáticos durante al menos `min_hold_duration_s`. No se fuerza done en frames en movimiento.
7. **Signo del cuaternión** continuo con el frame anterior, en vez de `qw ≥ 0`.
8. **Control de calidad en vivo y al guardar:**
   - Se extrae el núcleo de `Converter` (`tools/xarm5/convert_vio_dataset_to_xarm5.py`: `project`, `to_base`, `speeds`)
     a un módulo importable junto a `visuomotor_bringup/visuomotor_bringup/xarm5_interface.py`. Ahí se reutilizan
     `WorkspaceLimits`, `ReachLimit` y `clamp_to_workspace`.
   - Durante la grabación se avisa de velocidad, alcance, caja, inclinación lateral y giro de muñeca > ±140°.
   - Al guardar se muestra el mismo resumen que `--report-only`.
9. **Metadatos por episodio:**
   - serie y firmware de la cámara, commit de git, sha256 de los YAML y versión de RTAB-Map;
   - ajustes de exposición;
   - celda de la grilla del objeto (teclas 1–9) y estadísticas de rastreo.
10. **Almacenamiento** (`utils/zarr_storage.py`): `state`, `action` y los arrays nuevos en un solo bloque por episodio,
    con compresión. Las imágenes siguen en un bloque por frame. `zarr_dataset.py` ya lee `[:]`, así que es compatible.
11. **Dataset final** con `tools/hardware/build_dataset_from_bags.py`: recalcula las poses con el estimador ganador y
    conserva imágenes, límites de episodio y etiquetas. Escribe siempre un dataset nuevo, nunca sobrescribe.

### Fase 5: protocolo, control de calidad y piloto v8
- **Lista de verificación de cada sesión:**
  - luz encendida y exposición verificada;
  - tablero visible;
  - 10 s estáticos con ruido bajo el umbral;
  - una pasada de control contra el tablero.
- **Grilla de objetos:** a ambos lados de la base y a varias profundidades, en orden aleatorio, con su celda guardada.
  Desde HOME3 bajar antes de alejarse, y a velocidad compatible con el puente.
- **Piloto:**
  - 10–15 episodios, que pasan por control de calidad, conversión y 20 épocas de entrenamiento;
  - comparación con v7 frente a las líneas base exigentes (`tools/diagnose_policy_baselines.py`);
  - el robot reproduce 5 demos con la cámara en la brida, y se mide el error de la pose del tablero entre la demo y la
    reproducción, frame a frame. Esto valida juntos la odometría, la extrínseca y el anclaje a HOME3.
- Solo si el piloto pasa, se hace la toma completa de v8.

### Fase 6: documentación
Skill `realsense-vio-data-collection` v2.0, K15 reescrito (método y medición con referencia), K17 y AGENTS §17/§20
con los números medidos, y una lección en `context.md`. Nunca se publica el mejor caso como certificación.

## Verificación
- **Pruebas unitarias** (contenedor, `pytest`):
  - `tests/test_trajectory_metrics.py`: una Sim(3) conocida se recupera; trayectorias idénticas dan RPE = 0.
  - `tests/test_vo_episode_logic.py`: emparejamiento, pose nula, continuidad del signo, done estático y remuestreo.
  - Las pruebas existentes siguen pasando: `test_convert_vio_dataset.py` y `test_relative_coordinates.py`.
  - Tras extraer el núcleo del conversor, reconvertir v7 reproduce `demonstrations_real_v7_xarm5_v2.zarr` bit a bit.
- **Reproducción de un bag:** `ros2 bag play` de una corrida de la Fase 2, pasada por el recolector v2, da sellos
  monótonos a 50 ± 2 ms, ningún duplicado y ningún frame perdido sin marcar.
- **Medición:** informe por candidato con mediana, p90 y máximo frente a los criterios. La decisión de la Fase 3
  queda documentada.
- **De extremo a extremo:** error de reproducción en el robot y piloto frente a las líneas base exigentes.

## Riesgos
- El mantel blanco tiene poca textura. El tablero ayuda, pero cualquier cambio de escena tiene que repetirse igual en
  la inferencia.
- El robot se mueve más suave que una mano; por eso hay corridas a mano contra el tablero.
- Con el emisor apagado, la profundidad empeora en superficies lisas (afecta al candidato C).
- Disco: unos 10 MB/s comprimido mientras se graba, solo durante los episodios.
- Compilar librealsense para L4T r35 en la Jetson puede tomar tiempo.
