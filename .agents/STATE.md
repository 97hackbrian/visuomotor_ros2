# Estado vivo del proyecto (leer justo después de AGENTS.md)

> **Última actualización:** 2026-11-27, Claude Code (Opus 5.5).
>
> **Para qué sirve:** retomar el trabajo sin depender de la memoria del agente (compactación del contexto, sesión
> nueva u otro agente). En Claude Code, un hook `SessionStart` lo vuelve a cargar al iniciar, al reanudar y tras cada
> compactación (`.claude/settings.json`).
>
> **Qué no es:** no reemplaza las reglas (`AGENTS.md`) ni el detalle (`.agents/knowledge/`, `context.md`).
>
> **Mantenimiento:** corto, ≤ 150 líneas. Se reescribe lo superado en vez de acumular historia. Protocolo: skill
> `update-knowledge`, paso 0.

## 1. Qué corre ahora

- **Entrenamiento de DP con v8** (`train_real_v8x5_relative_ee_e400`):
  - empezó el 2026-11-27 a las 09:24 (−03); los logs van en UTC (12:24);
  - va a 0,184 s por paso (4 min 04 s por época, 50 s por evaluación) y debería terminar el **2026-11-28 hacia las 13:15**;
  - se mira con `tmux attach -t train_v8` (ventanas `monitor`, `log` y `sin_suspender`). `Ctrl-b d` sale sin detener
    nada. Si la sesión se cerró: `TRAIN_TMUX_SESSION=train_v8 tools/train_with_monitor.sh train_real_v8x5_relative_ee_e400 --monitor-only`;
  - la suspensión del equipo queda bloqueada mientras corre (`systemd-inhibit`);
  - al terminar, una espera en segundo plano del agente corre `tools/diagnose_policy_baselines.py` con las épocas
    100–400 (cada 25) en `<corrida>/baseline_diagnostics_e100-400/` (+ `.log`). Si esa sesión se perdió, lanzarlo a
    mano (AGENTS §7, comando 19).
- **Cámara y recolector:** cerrados desde el 2026-11-26.

## 2. Datasets (niveles de AGENTS §2.3)

| Nivel | Dataset | Estado |
|---|---|---|
| Crudo | `datasets/demonstrations_real_v8.zarr` + `datasets/raw_bags/demonstrations_real_v8/` | 252 episodios (0–251) en 2 sesiones. Solo crece |
| Derivado | `datasets/demonstrations_real_v8_xarm5.zarr` | Armado v3: 230 demos, 95 429 frames, poses de DROID. Lo rehace `tools/xarm5/make_v8_training_set.sh` |
| Funcional | `datasets/demonstrations_real_v8_xarm5_2026-11-27.zarr` | Copia congelada, solo lectura. **Es la que se entrena** |
| Funcional | `datasets/demonstrations_real_v7_xarm5_v2.zarr` | Entrenó el modelo desplegado. Para mezclar con v8 usar `v7_xarm5_v3` (extrínseca medida) |

- **Pendientes de repetir (REPETIR), sin marcar:** `episode_237`, `episode_248` y `episode_249`, todas a 180°. Quedan
  fuera del entrenamiento solas. Marcarlas como descartadas requiere el OK del usuario.
- **Veredicto: ENTRENABLE** (`datasets/evaluations_real_v8_xarm5/README.md`). Límite geométrico: 6 combinaciones
  celda × orientación no tienen un final alcanzable (4·180, 7·+90, 7·180, 8·180, 9·−90 y 9·180). El 21 % de los
  frames pasa el alcance del puente.

## 3. Modelos

- **Desplegado en la Jetson:** época 200 de `visuomotor_core/outputs/train_real_v7x5v2_relative_ee_e300`.
  - Ensayo real del 2026-11-23: 9/14 llegaron y se detuvieron solos.
  - Falla con la herramienta girada y junto a la base.
- **En entrenamiento:** `visuomotor_core/outputs/train_real_v8x5_relative_ee_e400`. Detalle del entrenamiento en
  `.agents/knowledge/20_v8_training.md`.
  - Datos: la copia congelada del 2026-11-27, con 207 demos de entrenamiento y 23 de validación (por episodio).
  - Configuración: `relative_ee`, EMA y `color_jitter`, 400 épocas. Checkpoint y evaluación por muestreo cada 5 épocas.
  - Estado de un vistazo: `docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/training_monitor.py --run-dir visuomotor_core/outputs/train_real_v8x5_relative_ee_e400 --once --tz-offset "$(date +%z)"`.
  - Parar limpio: `touch visuomotor_core/outputs/train_real_v8x5_relative_ee_e400/STOP`. Reanudar:
    `TRAIN_TMUX_SESSION=train_v8 tools/train_with_monitor.sh train_real_v8x5_relative_ee_e400 --resume`.
  - Hasta la época 75: meseta en la validación (FDE 17,7 mm, F1 del done 0,75). La política usa la imagen: la
    ablación sube el FDE un 65 %. Reproducción de la ruta de la Jetson con la época 75: en las 18 demos de
    validación dentro del alcance, el auto-stop para 18/18 (1 prematura); los fallos se concentran en las 5 que
    terminan fuera del alcance.
  - Épocas 100–200: meseta (FDE 17,5 → 17,3 mm). F1 del done: máximo 0,761 en la 125 y 0,743 en la 200 (con 23
    demos de validación puede ser ruido). La pérdida de denoising de validación sube desde la época 10 (0,013 →
    0,044) y en el split de entrenamiento el FDE es 3,0 mm: memoriza, aunque el muestreo de validación no empeora.
    Sigue perdiendo con la velocidad constante en los promedios (FDE 13,0 mm), como desde el inicio. Candidatas: la
    125–135 y las de menor FDE.
  - La última época no tiene por qué ser la mejor: en v7 las métricas se estancaron hacia la época 230 y el F1 del
    done bajó después. La época se elige con el protocolo de `.agents/knowledge/20_v8_training.md` §5.
  - Gráficas, refrescadas en cada evaluación:
    - `losses/training_curves.png`;
    - `evaluation/offline_eval_curves.png`;
    - al terminar, `convergence/` y `final_report/`.

## 4. Pendientes, en orden

1. **Elegir la época de v8 por comportamiento**, no por la pérdida:
   - FDE de validación contra la línea base de velocidad constante;
   - F1 del done;
   - `tools/diagnose_policy_baselines.py` en varias épocas, que incluye la ablación de imagen;
   - `tools/xarm5/replay_policy_offline.py` y `tools/xarm5/check_policy_closed_loop_sim.py`;
   - calibrar el auto-stop (umbral × paciencia) con la reproducción offline.
2. **Sincronizar la Jetson antes de cualquier ensayo:**
   - giro de muñeca ±177° con `clamp_yaw_near`;
   - caja 0,70 / ±0,35 / 0,70 y alcance al 97 % (`xarm5_real_params.yaml`);
   - el modelo nuevo;
   - reconstruir la imagen.
   - Probar los bordes nuevos despacio en la celda.
3. **Ensayo en el xArm5:** solo con un pedido explícito del usuario para cada acción y con el usuario en la parada de
   emergencia. Herramienta: `tools/xarm5/real_trials/run_trial.sh`. El piso del vigilante (brida a z ≥ 0,15 m) es
   obligatorio.
4. **Regrabación (opcional):**
   - ~25–30 demos con el final en diagonal desde más cerca del robot, para las 6 combinaciones inalcanzables;
   - las 3 REPETIR.
   - Procedimiento: skill `v8-handheld-demo-collection`.
5. **Ablación para el paper:** las 230 demos contra solo las factibles (137, con ≤ 5 % de frames fuera del alcance).
   Un entrenamiento pesado a la vez.
6. **Próxima sesión con el robot:** grabar también infra1/infra2 (con el emisor apagado) y calibrar con Kalibr, para
   probar VIO estéreo-inercial.

## 5. Reglas del usuario vigentes (además de AGENTS.md)

- **Brazo:** nunca mover ni armar el xArm5 sin un pedido explícito para ESA acción. "Continúa" no basta.
- **Entrenamiento:** uno pesado a la vez. Hay 31 GB de RAM, y tres a la vez cerraron la sesión gráfica.
  - Usar `--num_workers 6 --prefetch_factor 2 --max_rss_gb 12`.
- **Git:** commit y push directo a `main`; una rama solo para resolver merges. Git solo en el host, nunca en el contenedor.
- **Datos:**
  - Lo que no se usa o es de prueba se borra, pero dejando registro (informe, conocimiento y memoria).
  - Antes de borrar una copia congelada, comprobar que ningún `training_meta.json` la nombra.
  - Nunca se borran los experimentos de VIO (`datasets/vio_bags`, `datasets/vio_frames`, `replay*`), porque son para el paper.
- **Recolección:**
  - La voz dice la meta oficial de la sesión que fija el usuario (`V8_GOAL`).
  - El crudo v8 se acumula entre días, y así se le explica al usuario al empezar.
- **Alcance:** el xArm5 llega a 700 mm (±0,47 m laterales a la altura de la tarea), lo que no es lo mismo que la caja
  del puente. Siempre se dan las dos cifras, y la caja la decide el usuario.
- **Respaldo:** `/mnt/wdc/backup_bc/respaldo`, verificado por SHA-256. Copiar con rsync, no con Thunar.
- **Diapositivas INFONOR:**
  - Se editan solo en `presentacion_infonor2026/` y se compilan solo cuando el usuario lo pide.
  - Términos técnicos en inglés y tono académico; nunca "esta tesis".
- **Comunicación:** en español, técnica y concisa. Nunca dar el mejor caso como certificación: dar la distribución
  (mediana, p90 y máximo).

## 6. Dónde está el detalle

- **v8 de punta a punta:** `.agents/knowledge/19_v8_handheld_dataset_pipeline.md` (§1–9).
  - Métricas y revisión: `datasets/evaluations_real_v8_xarm5/README.md`.
- **Entrenamiento v8:** `.agents/knowledge/20_v8_training.md` y el `README.md` de su carpeta de salida.
- **Recolección a mano (procedimiento que funcionó):** `.agents/skills/v8-handheld-demo-collection/SKILL.md`.
- **xArm5 real:** `.agents/knowledge/16_xarm5_real_hardware_bringup.md`.
- **Curación de v7 y acciones relativas:** `.agents/knowledge/17_real_vio_dataset_curation_and_relative_actions.md`.
- **DROID-SLAM:** `.agents/knowledge/18_vio_estimator_selection_droid_slam.md` y la skill `droid-slam-offline-vio`.
- **Lecciones y errores:** `context.md`.

## 7. Retomar en 2 minutos

1. `git log --oneline -8` y `git status`, en el host.
2. `tmux ls` y `docker ps`: ¿sigue el entrenamiento (`train_v8`)? Monitor del §3.
3. Leer el §4 y la última instrucción del usuario. Si no está escrita aquí, preguntarla antes de actuar.
