#!/usr/bin/env bash
# Vigilante de una sesión de recolección v8: revisa cada pocos segundos el registro del recolector y el de la cámara,
# y TERMINA (código de salida = motivo) en cuanto pasa algo que requiere atención. Pensado para correr en segundo
# plano y despertar a quien lo lanzó (agente o usuario).
#
#   tools/hardware/v8_watch.sh <registro_recolector> <registro_camara> [demos_por_control=10] [latido_min=25]
#
# Salidas: 0 = control cada N demos guardadas | 2 = problema | 3 = latido (nada raro en latido_min minutos)
set -u
LOG=$1
CAMLOG=$2
EVERY=${3:-10}
HEARTBEAT_MIN=${4:-25}
CAM=visuomotor_vo_active
MIN_FREE_GB=${V8_MIN_FREE_GB:-20}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

touch "$LOG" "$CAMLOG"
pos_log=$(wc -l <"$LOG")
pos_cam=$(wc -l <"$CAMLOG")
saves=0
consec_cancel=0
consec_reject=0
t0=$(date +%s)
say_alert() { command -v spd-say >/dev/null && spd-say -l es -r 10 "Atención: $1" 2>/dev/null; }
problem() { echo "PROBLEMA: $*"; say_alert "$1"; exit 2; }

while true; do
  sleep 3
  # Procesos vivos
  docker ps --format '{{.Names}}' | grep -qx "$CAM" || problem "se detuvo el contenedor de la cámara"
  docker exec "$CAM" pgrep -f vo_collector >/dev/null 2>&1 || problem "el recolector no está corriendo"
  # Disco
  free_gb=$(df -BG --output=avail "$REPO/datasets" | tail -1 | tr -dc '0-9')
  [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] || problem "quedan solo ${free_gb} GB de disco"
  # Líneas nuevas del recolector
  n=$(wc -l <"$LOG")
  if [ "$n" -gt "$pos_log" ]; then
    while IFS= read -r line; do
      line=${line//$'\r'/}
      case "$line" in
        *"[SAVE] episode_"*)
          saves=$((saves + 1)); consec_cancel=0; consec_reject=0
          echo "guardada ($saves en este tramo): $(sed -E 's/.*\[SAVE\] //' <<<"$line" | cut -c1-120)" ;;
        *"episodio cancelado"*|*"no entregó poses válidas"*)
          consec_cancel=$((consec_cancel + 1))
          [ "$consec_cancel" -ge 3 ] && problem "tres demos canceladas seguidas al reiniciar la odometría" ;;
        *"[AUTO] No se guardó"*)
          consec_reject=$((consec_reject + 1))
          [ "$consec_reject" -ge 2 ] && problem "dos demos seguidas rechazadas por el control de calidad" ;;
        *Traceback*|*"Exception"*|*"exception"*)
          problem "excepción en el recolector: $(cut -c1-160 <<<"$line")" ;;
        *"[PROGRAMA] Completo"*)
          echo "PROGRAMA COMPLETO"; exit 0 ;;
      esac
    done < <(sed -n "$((pos_log + 1)),${n}p" "$LOG")
    pos_log=$n
  fi
  # Líneas nuevas de la cámara
  n=$(wc -l <"$CAMLOG")
  if [ "$n" -gt "$pos_cam" ]; then
    bad=$(sed -n "$((pos_cam + 1)),${n}p" "$CAMLOG" | tr -d '\r' | grep -E "Hardware Error|Hardware Notification|Frames didn't arrive|No RealSense devices|process has died|terminate called" | head -1)
    pos_cam=$n
    [ -n "$bad" ] && problem "error de la cámara: $(cut -c1-160 <<<"$bad")"
  fi
  [ "$saves" -ge "$EVERY" ] && { echo "CONTROL: $saves demos guardadas"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge $((HEARTBEAT_MIN * 60)) ] && { echo "LATIDO: $saves demos guardadas en ${HEARTBEAT_MIN} min, sin problemas"; exit 3; }
done
