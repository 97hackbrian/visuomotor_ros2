# 20. Entrenamiento de Diffusion Policy con el dataset v8 (2026-11-27)

Pedido del usuario (2026-11-27): "entrena aún con más épocas que la anterior vez", con el proceso monitorizado de
forma continua y la hora estimada de fin visible, y que el resultado sea un modelo exitoso.

## 1. Qué se entrena

- **Corrida:** `visuomotor_core/outputs/train_real_v8x5_relative_ee_e400`. El comando exacto está en su `README.md`
  y los argumentos en `visuomotor_core/outputs/train_real_v8x5_relative_ee_e400.launch`.
- **Inicio:** 2026-11-27 09:24, hora local (−03). Los CSV y el log usan el reloj del contenedor, que va en UTC: 12:24.
- **Datos:** `datasets/demonstrations_real_v8_xarm5_2026-11-27.zarr`, la copia congelada del armado v3.
  - 230 demos con poses de DROID y la llegada recortada a 3 cm.
  - Partición por episodio de 207 para entrenar y 23 para validar (`--val_ratio 0.1`, semilla 42). La validación
    está repartida a lo largo del orden de grabación: `episode_7`, `episode_17`, `episode_27`…
- **Tamaño:** 84 991 ventanas de entrenamiento, 1 327 pasos por época y 400 épocas, o sea 530 800 pasos.
- **Receta:** la del modelo que funcionó en el robot (época 200 de `train_real_v7x5v2_relative_ee_e300`):
  - `relative_ee`;
  - EMA de pesos (potencia 0,75, máx. 0,9999);
  - warmup de 500 pasos;
  - LR 1e-4 con batch 64, en coseno hasta 1e-6;
  - `color_jitter`;
  - `proprio_dropout` 0,15;
  - hemisferio `home`, con el mismo cuaternión de referencia que v7 (HOME3).
- **Cambios frente a la corrida de v7:**
  - 400 épocas en vez de 300, a pedido del usuario;
  - checkpoint y evaluación cada 5 épocas en vez de 10;
  - todas las ventanas de validación en cada evaluación;
  - `--plot_freq 0`: las curvas se dibujan solo en las épocas de evaluación, porque dibujarlas a menudo hacía crecer
    la RAM.
- **Recursos:** `--num_workers 6 --prefetch_factor 2 --max_rss_gb 12`, un entrenamiento pesado a la vez.
  - Al arrancar: RAM del host 12/31 GB, proceso principal 4,1 GB.
  - GPU al 99–100 %, con 2,6 de 8 GB en uso y ~57 °C.

## 2. Lectura crítica de "más épocas"

- **Una época de v8 no equivale a una de v7.** Tiene 1 327 pasos contra 329, 4,0 veces más.
  - Las 300 épocas de v7 fueron 98 700 pasos. Las 400 de v8 son 530 800, 5,4 veces más optimización.
- **En v7, las épocas de más no aportaron.** Fuente: `train_real_v7x5v2_relative_ee_e300/eval_metrics.csv`.
  - La validación se estabilizó hacia la época 230: ADE 11,6 mm y FDE 18,4 mm sin cambios hasta la 300.
  - El F1 del done llegó a su máximo en la época 220 (0,708) y bajó hasta 0,665 en la 300.
  - Las últimas 70–80 épocas no mejoraron el comportamiento y empeoraron un poco la señal de término.
- **Consecuencia: la última época no tiene por qué ser la mejor.** La época se elige por comportamiento:
  - métricas de muestreo en validación;
  - líneas base exigentes;
  - reproducción de la ruta de despliegue (§5).
  - Los checkpoints cada 5 épocas permiten esa elección.
- **Parar antes:** si la validación queda plana durante ≥ 50 épocas, conviene parar (`STOP`) y ahorrar horas. Hay un
  costo: el LR va en coseno sobre las 400 épocas, así que al cortar a la mitad queda en ~50 % del máximo, sin
  recocido. La EMA lo atenúa. Lo decide el usuario.
- **Tiempo:** 0,178 s por paso, igual que en v7 (0,180).
  - Cada época tarda 3 min 57 s (236,5 s medidos) y cada evaluación de ~10 500 ventanas, 48 s.
  - En total, ~26 h de pasos más 80 evaluaciones (~1 h): fin estimado el 2026-11-28 hacia las 12:45, hora local.
  - Corrección del monitor: la primera estimación decía 0,200 s por paso y fin a las 14:53. Los sellos tienen
    resolución de 1 s y las filas van cada ~1,8 s, así que la mediana de los intervalos quedaba cuantizada. Ahora
    se usa la duración medida de las épocas.

## 3. Riesgos para que el modelo funcione en el robot, y cómo se revisan

- **Alcance:** el 21 % de los frames pasa el alcance del puente.
  - En las celdas lejanas y laterales, la política pedirá poses que el brazo no alcanza, y el puente las recorta.
  - En los ensayos, empezar por las celdas alcanzables (`datasets/evaluations_real_v8_xarm5/README.md`).
  - Después, la ablación con solo las 137 factibles.
- **Uso de la visión:** v8 empieza siempre en la misma vista y termina en 9 celdas × 6 orientaciones, así que la
  política está obligada a mirar. En v7 esto apenas pasaba: acertaba la dirección desde el reposo en el 64 % de los
  casos, contra el 55 % de la dirección media.
  - Cómo se comprueba: con la ablación de imagen de `tools/diagnose_policy_baselines.py`. El FDE con imágenes de
    otro episodio tiene que subir claramente, y la dirección desde el reposo tiene que ganarle a la dirección media.
- **Señal de término:** las etiquetas ya vienen recortadas en el armado v3.
  - Revisar el F1 del done en validación.
  - Calibrar el umbral × paciencia del auto-stop con la reproducción offline. En v7 quedó en 0,60 × 5.
- **Qué mide la validación:** las mismas combinaciones celda × orientación están en entrenamiento y en validación
  (3–6 demos de cada una). La validación mide interpolación dentro de lo cubierto, no la generalización a lugares
  nuevos; esa solo la mide el robot.
- **Escena:** en los ensayos, el mismo mantel y los mismos ArUco, la base en su lugar y sin el tablero ChArUco, como
  en las demos.
- **Cuaternión:** 15 demos a 180° cruzan el límite del hemisferio `home` en el estado. Afecta solo la entrada de
  condicionamiento: las acciones relativas no cambian de signo.
- **Jetson:** antes de un ensayo hay que sincronizar el giro de ±177° con `clamp_yaw_near` y la caja 0,70 / ±0,35 /
  0,70 al 97 %. Sin eso, las demos a 180° piden giros que el puente viejo recorta a ±149°.

## 4. Monitoreo

- **Lanzador:** `tools/train_with_monitor.sh <run> -- <argumentos de train.py>`.
  - Lanza el entrenamiento desacoplado (`docker exec -d`): sobrevive a tmux, a la terminal y al agente.
  - Se niega a correr si ya hay otro `train.py`.
  - Guarda los argumentos en `<run>.launch` y escribe `<run>/README.md`.
  - Crea la sesión tmux con tres ventanas: `monitor`, `log` y `sin_suspender`. Esta última bloquea la suspensión,
    también al cerrar la tapa, mientras el entrenamiento corre (`systemd-inhibit`).
  - Otros modos: `--resume`, `--monitor-only` y `--show` (abre una terminal del escritorio).
- **Monitor:** `tools/training_monitor.py --run-dir <run> [--once|--watch 30|--json] --tz-offset "$(date +%z)"`.
  - Muestra el progreso, la velocidad y la hora estimada de fin.
  - La hora estimada se calcula así: pasos restantes × mediana de segundos por paso, más evaluaciones restantes ×
    duración media de las pausas largas.
  - Muestra también la pérdida, la tabla de validación con las líneas base ("quieto" y velocidad constante), la
    mejor época por FDE y por F1 del done, la RAM y la GPU, y las carpetas de las gráficas.
  - Solo lee.
  - Comprobado con la corrida de v7: reproduce 5 h 04 min, 0,200 s por paso y 16 s por evaluación.
- **Gráficas**, refrescadas en cada evaluación:
  - `losses/training_curves.png`;
  - `evaluation/offline_eval_curves.png`;
  - al terminar, `convergence/` (criterios de parada) y `final_report/`.
- **Vigilante del agente (Claude Code):** una tarea Monitor sobre `training.log`. Avisa en cada `[Eval val]`, ante
  ERROR, Traceback, la guardia de RAM, una pérdida NaN o el fin, y si el proceso deja de existir. Dura 30 min y hay
  que volver a armarla.

## 5. Cómo elegir la época (al terminar, o antes si la validación se estanca)

1. **Candidatas, desde `eval_metrics.csv` (split `val`):**
   - FDE y ADE ejecutado más bajos;
   - F1 del done alto;
   - ganarle a la velocidad constante en FDE (`cv_fde_mm`); en v7, la época 200 ganaba desde ~0,5 s.
2. **Ablación visual y líneas base en varias épocas:**
   ```bash
   docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/diagnose_policy_baselines.py \
       --run-dir visuomotor_core/outputs/train_real_v8x5_relative_ee_e400 --epochs 100,150,200,250,300,350,400 \
       --highlight <candidata> --out-dir visuomotor_core/outputs/train_real_v8x5_relative_ee_e400/baseline_diagnostics
   ```
3. **Ruta de despliegue y auto-stop:**
   - `tools/xarm5/replay_policy_offline.py --episodes val`, para calibrar el auto-stop;
   - `tools/xarm5/check_policy_closed_loop_sim.py`.
4. **Llevarla a la Jetson**, con el OK del usuario:
   - copiar a `deploy/jetson/model/`;
   - sincronizar el puente (§3);
   - reconstruir la imagen.
5. **Ensayo en el xArm5:** solo con el pedido explícito del usuario para cada acción, y con él en la parada de
   emergencia.

## 6. Resultados

**Validación, primeras épocas** (`eval_metrics.csv`, split `val`, 23 episodios, todas las ventanas):

| Época | ADE | ADE ejecutado | FDE | Rotación | F1 del done | Progreso |
|---|---|---|---|---|---|---|
| 1 | 21,1 mm | 16,0 mm | 32,3 mm | 4,49° | 0,227 | 1,24 |
| 5 | 11,7 mm | 7,2 mm | 20,2 mm | 2,05° | 0,627 | 1,04 |

Líneas base en los mismos episodios:
- quedarse quieto: ADE 14,8 mm y ADE ejecutado 8,5 mm;
- velocidad constante: ADE 5,7 mm, ADE ejecutado 2,3 mm y FDE 13,0 mm.

**Hallazgo: en v8, la velocidad constante es una vara mucho más alta que en v7.** Su FDE de validación es 13,0 mm,
contra 21,3 mm en v7, porque las trayectorias de DROID son suaves (0,9 mm de ruido contra 8,8).
- En v7, la política ganaba a la velocidad constante en FDE en parte porque extrapolar el ruido salía caro.
- En v8, extrapolar el movimiento a 0,8 s acierta en casi todas las ventanas en movimiento. Perder con ella en el
  promedio no prueba que la política sea mala: lo que la política tiene que hacer bien son las decisiones (arrancar
  desde el reposo hacia la celda correcta, dónde detenerse, cuándo marcar el final). Eso no lo mide el promedio de
  ADE/FDE.
- Para elegir la época, además del promedio:
  - la dirección desde el reposo y la ablación de imagen (`diagnose_policy_baselines.py`);
  - el error de la posición final y el auto-stop en la reproducción offline;
  - el F1 del done.

**Validación hasta la época 50:** mejora lenta y casi plana desde la época 30, con el LR todavía al ~98 %.

| Época | ADE | ADE ejecutado | FDE | Rotación | F1 del done |
|---|---|---|---|---|---|
| 10 | 11,1 mm | 6,7 mm | 19,2 mm | 1,90° | 0,692 |
| 20 | 10,7 mm | 6,5 mm | 18,6 mm | 1,81° | 0,713 |
| 30 | 10,6 mm | 6,4 mm | 18,3 mm | 1,79° | 0,733 |
| 40 | 10,5 mm | 6,3 mm | 18,3 mm | 1,78° | 0,731 |
| 50 | 10,4 mm | 6,3 mm | 18,1 mm | 1,76° | 0,734 |

**Diagnóstico de visión en la época 50** (`tools/diagnose_policy_baselines.py`, carpeta
`baseline_diagnostics_e50/` de la corrida):
- 23 episodios de validación y 10 208 ventanas.
- Tardó 1,5 min con el entrenamiento corriendo.
- La referencia de v7 es su época 200, medida con la misma herramienta.

| Medida | v7, época 200 | v8, época 50 |
|---|---|---|
| FDE de la política | 18,5 mm | 18,1 mm |
| FDE con imágenes de otro episodio (ablación) | 29,2 mm | 29,9 mm (+65 %) |
| FDE de la velocidad constante | 21,3 mm | 13,0 mm |
| Dirección desde el reposo, coseno > 0,9: política / dirección media de entrenamiento / imágenes cruzadas | 64 % / 55 % (n = 22) | 47 % / 22 % / 16 % (n = 221) |
| Coseno mediano de la dirección desde el reposo: política / dirección media | — | 0,887 / 0,706 |
| Ventanas en movimiento: dirección con coseno > 0,9, política / dirección media | — | 54 % / 18 % |

**Lectura:**
- **La política usa la imagen para decidir:**
  - Con imágenes de otro episodio, el FDE sube un 65 %.
  - Desde el reposo, acierta la dirección 2,2 veces más que "ir siempre hacia la dirección media" (47 % contra 22 %).
  - En v7, en cambio, la dirección media ya acertaba el 55 %, porque las demos iban casi todas al mismo lado. v8 sí
    obliga a mirar, y la política lo hace.
- **Lo que falta:**
  - Desde el reposo, la mitad de los arranques se desvía más de 26° de la demo.
  - En las ventanas en movimiento, la velocidad constante sigue ganando en FDE (14,6 contra 20,1 mm).
  - En lazo cerrado la imagen se actualiza y corrige el rumbo. Eso lo miden la reproducción offline y el robot,
    no este promedio.

**Meseta desde la época 65:** ADE 10,2 mm, ADE ejecutado 6,1 mm, FDE 17,7 mm, rotación 1,72° y F1 del done
0,74–0,75, con el LR al ~92 %.
- **Se deja terminar hasta la 400** (decisión explicada al usuario el 2026-11-27).
- En v7, la fase de bajada del LR (épocas 50 → 200) dejó la trayectoria casi igual (−3 % en FDE), pero subió el F1
  del done de 0,614 a 0,698. Ese F1 es justo lo que falló en el robot.
- Después bajó: 0,665 en la época 300.

**Reproducción offline de la ruta de la Jetson en la época 75** (`tools/xarm5/replay_policy_offline.py`, carpeta
`replay_e75/`):
- Parámetros de `jetson_policy_params.yaml` y auto-stop 0,60 × 5.
- Teacher forcing sobre los 23 episodios de validación; tardó 4,6 min con el entrenamiento corriendo.
- Error de comando: 10,3 mm de media (p95 20,0) y 1,79° de rotación.
- Auto-stop: se detiene en 20 de 23 episodios, con 3 paradas prematuras (> 1 s antes) y 16,7 mm de mediana entre la
  parada y la meta.

El total esconde dos grupos:

| Episodios de validación | Se detiene | Prematuras (> 1 s) | Parada → meta (mediana) |
|---|---|---|---|
| Dentro del alcance (18; celdas 1–7 y 9) | **18/18** | 1 (`episode_127`, celda 6: −2,6 s, 62 mm) | 17,0 mm |
| Marcados fuera del alcance (5; celdas 6, 8 y 9) | 2/5: no paran `episode_27`, `episode_47` ni `episode_157` | 2: las dos paradas que hubo | 43,0 mm |

- **Los fallos del auto-stop se concentran en las demos que terminan fuera del alcance del brazo.** Son las celdas
  que el robot no puede completar igual.
- **En las alcanzables, el auto-stop ya funciona en la época 75**, aunque todavía peor que en v7 (100 %, 0
  prematuras, ~8 mm en 8 episodios).
- **Barrido de umbral × paciencia** (los 23 episodios): subir la paciencia a 10 baja las prematuras a 2 y la
  distancia parada → meta a 11–13 mm, con 0,35–0,55 s de demora. Con umbral 0,8, la tasa de parada cae a 0,78–0,87.
- **Pendiente:** calibrarlo al final, con el checkpoint elegido y sobre los episodios dentro del alcance
  (`--episodes` con la lista).

*(se sigue completando: época elegida y por qué)*
