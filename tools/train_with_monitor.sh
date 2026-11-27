#!/usr/bin/env bash
# Lanza un entrenamiento de visuomotor_core/train.py en el contenedor y una sesión tmux para mirarlo.
#
# El entrenamiento corre DESACOPLADO (docker exec -d): sobrevive a tmux, a la terminal y a la sesión del agente.
# La sesión tmux tiene tres ventanas:
#   monitor       tools/training_monitor.py: progreso, hora estimada de fin, validación con líneas base, GPU y RAM,
#                 carpetas de las gráficas.
#   log           training.log en vivo.
#   sin_suspender bloqueo de suspensión del equipo (también al cerrar la tapa) mientras el entrenamiento corre.
# Guarda los argumentos en visuomotor_core/outputs/<run>.launch y escribe <run>/README.md con el comando exacto.
#
#   tools/train_with_monitor.sh <run> -- <argumentos de train.py, SIN --out_dir>
#   tools/train_with_monitor.sh <run> --resume          # reanuda con los mismos argumentos (+ --resume)
#   tools/train_with_monitor.sh <run> --monitor-only    # solo la sesión tmux (si se cerró)
#   tools/train_with_monitor.sh <run> --show            # abre una terminal del escritorio enganchada a la sesión
#
# Variables: TRAIN_TMUX_SESSION (por defecto train), TRAIN_ALLOW_CONCURRENT=1 (saltar la regla de uno a la vez).
# Reglas (AGENTS.md §22): un entrenamiento pesado a la vez; --num_workers 6 --prefetch_factor 2 --max_rss_gb 12.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
RUN=${1:?"uso: tools/train_with_monitor.sh <run> -- <argumentos de train.py>"}
shift
MODE=new
case "${1:-}" in
  --) shift ;;
  --resume) MODE=resume; shift ;;
  --monitor-only) MODE=monitor; shift ;;
  --show) MODE=show; shift ;;
  *) echo "uso: tools/train_with_monitor.sh <run> -- <argumentos de train.py> | --resume | --monitor-only | --show"; exit 1 ;;
esac
S=${TRAIN_TMUX_SESSION:-train}
C=visuomotor_ros2
OUT_REL=visuomotor_core/outputs/$RUN
LAUNCH=visuomotor_core/outputs/$RUN.launch
STDOUT=visuomotor_core/outputs/$RUN.stdout
docker ps --format '{{.Names}}' | grep -qx "$C" || { echo "ERROR: el contenedor $C no corre (docker start $C)"; exit 1; }
running() { docker exec "$C" pgrep -f "[t]rain.py.*$RUN" >/dev/null 2>&1; }

show() {
  if command -v gnome-terminal >/dev/null; then
    setsid gnome-terminal --title "entrenamiento $RUN" -- tmux attach -t "=$S" >/dev/null 2>&1 &
  else
    setsid xterm -T "entrenamiento $RUN" -e tmux attach -t "=$S" >/dev/null 2>&1 &
  fi
  echo "terminal abierta: Ctrl-b n / Ctrl-b p cambian de ventana; Ctrl-b d se desengancha sin detener nada"
}

monitor_session() {
  tmux has-session -t "=$S" 2>/dev/null && { echo "la sesión tmux $S ya existe (tmux attach -t $S)"; return 0; }
  tmux new-session -d -s "$S" -n monitor -c "$REPO" -x 200 -y 55
  tmux send-keys -t "=$S:monitor" "docker exec -it -u 1000 -w /ros $C python3 tools/training_monitor.py --run-dir $OUT_REL --watch 30 --tz-offset \"\$(date +%z)\" --host-root \"$REPO\"" Enter
  tmux new-window -d -t "=$S" -n log -c "$REPO"
  tmux send-keys -t "=$S:log" "tail -n 40 -F $OUT_REL/training.log" Enter
  if command -v systemd-inhibit >/dev/null; then
    tmux new-window -d -t "=$S" -n sin_suspender -c "$REPO"
    tmux send-keys -t "=$S:sin_suspender" "systemd-inhibit --what=sleep:idle:handle-lid-switch --who=visuomotor --why='Entrenamiento $RUN' --mode=block bash -c 'echo Suspension bloqueada mientras corre $RUN; while docker exec $C pgrep -f \"[t]rain.py.*$RUN\" >/dev/null 2>&1; do sleep 60; done; echo Entrenamiento terminado: bloqueo liberado'" Enter
  fi
  tmux select-window -t "=$S:monitor"
  echo "sesión tmux: $S (tmux attach -t $S)"
}

write_readme() {   # cuando train.py ya creó la carpeta
  local args=$1
  for _ in $(seq 1 60); do [ -d "$OUT_REL" ] && break; sleep 2; done
  [ -d "$OUT_REL" ] || { echo "AVISO: $OUT_REL no apareció; ver $STDOUT"; return 0; }
  [ -f "$OUT_REL/README.md" ] && return 0
  cat > "$OUT_REL/README.md" <<EOF
# $RUN

Lanzado el $(date '+%Y-%m-%d %H:%M %z') con \`tools/train_with_monitor.sh\` (commit $(git rev-parse --short HEAD)).

## Comando (contenedor $C, usuario 1000, en /ros)

\`\`\`bash
python3 visuomotor_core/train.py $args --out_dir /ros/$OUT_REL > /ros/$STDOUT 2>&1
\`\`\`

## Mirar, parar y reanudar (desde el host, en la raíz del repo)

- Sesión tmux: \`tmux attach -t $S\` (ventanas monitor, log y sin_suspender; \`Ctrl-b d\` sale sin detener nada).
  Si se cerró: \`tools/train_with_monitor.sh $RUN --monitor-only\`.
- Una vez: \`docker exec -u 1000 -w /ros $C python3 tools/training_monitor.py --run-dir $OUT_REL --once --tz-offset "\$(date +%z)"\`
- Parar limpio (termina la época, guarda el estado y el modelo): \`touch $OUT_REL/STOP\`
- Reanudar: \`tools/train_with_monitor.sh $RUN --resume\`

## Gráficas (se refrescan en cada evaluación)

- \`losses/training_curves.png\` (pérdida por paso y por época, con doble eje pasos/épocas)
- \`evaluation/offline_eval_curves.png\` (ADE/FDE/rotación/done F1 de train y validación, con líneas base)
- Al terminar: \`convergence/\` y \`final_report/\`
- Datos: \`training_metrics.csv\` (cada 10 pasos), \`epoch_metrics.csv\`, \`eval_metrics.csv\`
EOF
  echo "README: $OUT_REL/README.md"
}

case "$MODE" in
  show) show; exit 0 ;;
  monitor) monitor_session; exit 0 ;;
  new)
    [ $# -gt 0 ] || { echo "ERROR: faltan los argumentos de train.py después de --"; exit 1; }
    for a in "$@"; do case "$a" in --out_dir*|--resume) echo "ERROR: no pasar $a (lo pone este script)"; exit 1 ;; esac; done
    [ ! -e "$OUT_REL" ] || { echo "ERROR: $OUT_REL ya existe (otro nombre, o --resume)"; exit 1; }
    ARGS=$(printf '%q ' "$@") ;;
  resume)
    [ -f "$LAUNCH" ] || { echo "ERROR: no existe $LAUNCH con los argumentos originales"; exit 1; }
    [ -f "$OUT_REL/training_state.pt" ] || { echo "ERROR: no hay $OUT_REL/training_state.pt para reanudar"; exit 1; }
    ARGS="$(cat "$LAUNCH") --resume"
    rm -f "$OUT_REL/STOP" ;;
esac
if running; then echo "ERROR: $RUN ya está corriendo"; exit 1; fi
if [ "${TRAIN_ALLOW_CONCURRENT:-0}" != 1 ] && docker exec "$C" pgrep -af "[t]rain.py" >/dev/null 2>&1; then
  echo "ERROR: ya hay otro entrenamiento corriendo (regla: uno pesado a la vez; host de 31 GB):"
  docker exec "$C" pgrep -af "[t]rain.py" | head -3
  exit 1
fi
[ "$MODE" = new ] && printf '%s\n' "$ARGS" > "$LAUNCH"
docker exec -d -u 1000 -w /ros "$C" bash -c "python3 visuomotor_core/train.py $ARGS --out_dir /ros/$OUT_REL >> /ros/$STDOUT 2>&1"
echo "lanzado: $RUN ($MODE)  ·  salida: $STDOUT"
sleep 15
running || { echo "ERROR: el entrenamiento no siguió corriendo; últimas líneas de $STDOUT:"; tail -20 "$STDOUT"; exit 1; }
[ "$MODE" = new ] && write_readme "$(cat "$LAUNCH")"
monitor_session
