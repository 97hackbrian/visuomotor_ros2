#!/usr/bin/env bash
# Un intento de la política en el xArm5 real, orquestado desde el PC (etapa C de deploy/jetson/README.md).
#   tools/xarm5/real_trials/run_trial.sh N "posición de las tijeras"       # intento N en modo real
#   tools/xarm5/real_trials/run_trial.sh --dry-run                          # etapa B: puente en dry-run, no mueve
# Requisitos: brazo en HOME3, puente lanzado por el usuario (modo real y desarmado para la etapa C; dry-run para la
# B), cámara publicando y el usuario en la parada de emergencia. El chequeo previo aborta SIN armar si algo falla.
# Variables: JETSON (jetson@172.16.35.107), CONTAINER (visuomotor_orin_inference), SESSION (carpeta de la sesión).
# Pitido corto al empezar y doble al terminar. Métricas: tools/xarm5/real_trial_metrics.py.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
JETSON=${JETSON:-jetson@172.16.35.107}; CONTAINER=${CONTAINER:-visuomotor_orin_inference}
SESSION=${SESSION:-$REPO/experiments/xarm5_real_trials/$(date +%F)}
J="ssh -o BatchMode=yes $JETSON"
BEEP=$HERE/.beeps; [ -f "$BEEP/inicio.wav" ] || python3 "$HERE/make_beeps.py" "$BEEP"
play() { paplay "$1" 2>/dev/null || aplay -q "$1" 2>/dev/null || printf '\a'; }
# Scripts del lado de la Jetson al contenedor (sin reconstruir la imagen)
scp -q -o BatchMode=yes "$HERE"/jetson/* "$JETSON":/tmp/ || { echo "sin conexión con $JETSON"; exit 1; }
$J "docker exec $CONTAINER mkdir -p /tmp/vtrials && for f in preflight.py grab_home3.py monitor.py record_dry_run.py run_attempt.sh run_dry_run.sh; do docker cp /tmp/\$f $CONTAINER:/tmp/vtrials/\$f; done"
DX="docker exec $CONTAINER /ros_ws/entrypoint.sh"
if [ "${1:-}" = "--dry-run" ]; then
  play "$BEEP/inicio.wav"; timeout 240 $J "$DX bash /tmp/vtrials/run_dry_run.sh" 2>&1 | grep -vE "UserWarning|warn\(|^$"
  play "$BEEP/fin.wav"; exit 0
fi
N=$1; LABEL=$2; T=$SESSION/trial_$(printf %03d "$N"); mkdir -p "$T"
echo "== intento $N ($LABEL): chequeo previo"
# El código de salida es el del chequeo (no el de un filtro de la salida): si falla, no se arma nada.
PRE=$(timeout 60 $J "$DX python3 /tmp/vtrials/preflight.py" 2>&1); RC=$?
echo "$PRE" | grep -vE "UserWarning|warn\("
if [ "$RC" -ne 0 ]; then
  echo "ABORTO: el chequeo previo falló (código $RC): no se arma nada"; play "$BEEP/fin.wav"; exit 1; fi
timeout 60 $J "$DX python3 /tmp/vtrials/grab_home3.py >/dev/null 2>&1; docker cp $CONTAINER:/tmp/vtrials/pre.png /tmp/vtrials_pre.png" \
  && scp -q -o BatchMode=yes "$JETSON":/tmp/vtrials_pre.png "$T/pre.png"
play "$BEEP/inicio.wav"; date "+INICIO intento $N %H:%M:%S"
timeout 240 $J "$DX bash /tmp/vtrials/run_attempt.sh" 2>&1 | grep -E "arm:|vigilante|ABORTO|forzando|publicadores|armed=|Auto-Stop|ERROR" | cut -c1-180
date "+FIN intento $N %H:%M:%S"; play "$BEEP/fin.wav"
for f in final.png tcp.npy policy.log monitor.json; do
  $J "docker cp $CONTAINER:/tmp/vtrials/$f /tmp/vtrials_$f" && scp -q -o BatchMode=yes "$JETSON:/tmp/vtrials_$f" "$T/$f"; done
docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "python3 tools/xarm5/real_trial_metrics.py --session '${SESSION#$REPO/}' --trial $N --label '$LABEL' 2>&1 | tail -2"
