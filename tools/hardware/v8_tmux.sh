#!/usr/bin/env bash
# Sesión de recolección v8 en tmux: cámara, revisión previa, guía y recolector en ventanas de UNA sesión. Así un
# agente puede lanzarlos y mirarlos (capture-pane), y el usuario ve la misma terminal y puede teclear [S]/[D]/[N].
#
#   tools/hardware/v8_tmux.sh camera    # ventana "camara": D435i + RTAB-Map (realsense_d435i_vio.launch.py)
#   tools/hardware/v8_tmux.sh check     # ventana "chequeo": guía de HOME3 sin grabar (/board_check/image)
#   tools/hardware/v8_tmux.sh collect   # ventana "recolector": run_v8_session.sh -> EMPIEZA A GRABAR DEMOS
#   tools/hardware/v8_tmux.sh guide     # ventana "guia": rqt_image_view de la guía del recolector
#   tools/hardware/v8_tmux.sh grid      # ventana "grilla": ajustar la grilla a mano (adjust_v8_grid.py; en vivo si
#                                          la cámara corre). Reiniciar el recolector después de guardar.
#   tools/hardware/v8_tmux.sh show      # abre una terminal en el escritorio enganchada a la sesión
#   tools/hardware/v8_tmux.sh log [ventana] [líneas]   # últimas líneas de una ventana (por defecto el recolector)
#   tools/hardware/v8_tmux.sh stop [ventana]           # Ctrl-C a una ventana, o a todas (cámara la última)
#
# Variables: V8_TMUX_SESSION (por defecto v8), V8_DATASET (por defecto demonstrations_real_v8).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
S=${V8_TMUX_SESSION:-v8}
# La cámara va en una sesión APARTE que la terminal compartida no muestra (2026-11-25: al cerrarse la ventana del
# recolector, la terminal pasó a la de la cámara y unos Ctrl-C la apagaron).
CAMS=cam_${S}
# Objetivos EXACTOS (=nombre): tmux compara sesiones por prefijo y "v8" encontraba "v8_cam" (2026-11-25: las ventanas
# del recolector terminaron en la sesión de la cámara y la terminal compartida volvió a mostrarla).
DATASET=${V8_DATASET:-demonstrations_real_v8}
CAM=visuomotor_vo_active
ROS_ENV="export ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR=/tmp/.ros_log_user \
&& mkdir -p /tmp/.ros_log_user && source /opt/ros/jazzy/setup.bash && source install/setup.bash"

has_win() { tmux list-windows -t "=$S" -F '#W' 2>/dev/null | grep -qx "$1"; }
new_win() {  # nombre, comando (la ventana queda abierta con un shell al terminar, para leer el error)
  local name=$1; shift
  if has_win "$name"; then echo "ya existe la ventana $name (tools/hardware/v8_tmux.sh stop $name)"; return 1; fi
  if tmux has-session -t "=$S" 2>/dev/null; then
    tmux new-window -d -t "=$S" -n "$name" -c "$REPO"
  else
    tmux new-session -d -s "$S" -n "$name" -c "$REPO" -x 200 -y 50
  fi
  tmux send-keys -t "=$S:$name" "$*" Enter
}
cam_running() { docker ps --format '{{.Names}}' | grep -qx "$CAM"; }

case "${1:-}" in
  camera)
    if tmux has-session -t "=$CAMS" 2>/dev/null; then echo "la cámara ya corre (sesión $CAMS)"; exit 1; fi
    tmux new-session -d -s "$CAMS" -n camara -c "$REPO" -x 200 -y 50
    tmux send-keys -t "=$CAMS:camara" "./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false" Enter
    LOGDIR="$REPO/experiments/collection_sessions/$(date +%F)"; mkdir -p "$LOGDIR"
    CAMLOG="$LOGDIR/camera_$(date +%H%M%S).log"
    tmux pipe-pane -o -t "=$CAMS:camara" "cat >> '$CAMLOG'"
    echo "registro de la cámara: $CAMLOG" ;;
  check)
    cam_running || { echo "la cámara no está corriendo (v8_tmux.sh camera)"; exit 1; }
    new_win chequeo "docker exec -it -u 1000 -w /ros $CAM bash -c '$ROS_ENV && python3 tools/hardware/check_board_view.py'"
    sleep 3
    new_win guia_chequeo "docker exec -it -u 1000 -w /ros $CAM bash -c '$ROS_ENV && ros2 run rqt_image_view rqt_image_view /board_check/image'" ;;
  collect)
    cam_running || { echo "la cámara no está corriendo (v8_tmux.sh camera)"; exit 1; }
    # Registro de la sesión (trazabilidad) + avisos por los parlantes (v8_announcer.sh) en la ventana "voz"
    LOGDIR="$REPO/experiments/collection_sessions/$(date +%F)"; mkdir -p "$LOGDIR"
    LOG="$LOGDIR/collector_${DATASET}_$(date +%H%M%S).log"
    has_win voz || new_win voz "./tools/hardware/v8_announcer.sh '$LOG'"
    new_win recolector "./tools/hardware/run_v8_session.sh $DATASET" || exit 1
    tmux pipe-pane -o -t "=$S:recolector" "cat >> '$LOG'"
    tmux select-window -t "=$S:recolector"
    echo "registro: $LOG" ;;
  guide)
    cam_running || { echo "la cámara no está corriendo (v8_tmux.sh camera)"; exit 1; }
    new_win guia "docker exec -it -u 1000 -w /ros $CAM bash -c '$ROS_ENV && ros2 run rqt_image_view rqt_image_view /vo_collector_node/start_guide'" ;;
  grid)
    if cam_running; then
      new_win grilla "docker exec -it -u 1000 -w /ros $CAM bash -c '$ROS_ENV && python3 tools/hardware/adjust_v8_grid.py --live'"
    else
      new_win grilla "docker exec -it -u 1000 -w /ros visuomotor_ros2 bash -c 'source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 tools/hardware/adjust_v8_grid.py'"
    fi ;;
  show)
    tmux has-session -t "=$S" 2>/dev/null || { echo "no hay sesión $S"; exit 1; }
    if command -v gnome-terminal >/dev/null; then
      setsid gnome-terminal --title "v8 ($S)" -- tmux attach -t "=$S" >/dev/null 2>&1 &
    else
      setsid xterm -T "v8 ($S)" -e tmux attach -t "=$S" >/dev/null 2>&1 &
    fi
    echo "terminal abierta (Ctrl-b n / Ctrl-b p cambia de ventana; Ctrl-b d se desengancha sin detener nada)" ;;
  log)
    w=${2:-recolector}
    if [ "$w" = camara ]; then t="=$CAMS:camara"; else t="=$S:$w"; fi
    tmux capture-pane -p -J -t "$t" -S "-${3:-40}" ;;
  stop)
    if [ "${2:-}" = camara ]; then
      tmux has-session -t "=$CAMS" 2>/dev/null && { tmux send-keys -t "=$CAMS:camara" C-c; sleep 5; tmux kill-session -t "=$CAMS"; }
    elif [ -n "${2:-}" ]; then
      has_win "$2" && { tmux send-keys -t "=$S:$2" C-c; sleep 1; tmux kill-window -t "=$S:$2"; }
    else
      for w in grilla guia recolector voz guia_chequeo chequeo; do has_win "$w" && { tmux send-keys -t "=$S:$w" C-c; sleep 2; tmux kill-window -t "=$S:$w"; }; done
      tmux kill-session -t "=$S" 2>/dev/null
      tmux has-session -t "=$CAMS" 2>/dev/null && { tmux send-keys -t "=$CAMS:camara" C-c; sleep 5; tmux kill-session -t "=$CAMS"; }
    fi ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
