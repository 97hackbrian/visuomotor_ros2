#!/usr/bin/env bash
# Reprocesa los bags de las corridas con el robot con cada candidato de RTAB-Map (vio_measurement.yaml) y evalúa cada
# odometría contra el xArm5 (evaluate_vio_gt.py, etiqueta = candidato). No necesita el robot ni la cámara.
#   tools/xarm5/vio_gt/replay_candidates_vs_robot.sh <sesión> <static_run> "<cand1,cand2,...>" <run1> [<run2> ...]
# Orden: por corrida, todos los candidatos (si se corta, las corridas ya hechas tienen la comparación completa).
# Reanudable: salta lo que ya tiene vio_vs_robot_<cand>.json. Un reprocesado a la vez (dominio DDS fijo).
set -u
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
SESSION=$1; STATIC=$2; CANDS=$3; shift 3
for RUN in "$@"; do
  BAG=$(cat "$RUN/bag_path.txt")
  for C in ${CANDS//,/ }; do
    [ -f "$RUN/vio_vs_robot_$C.json" ] && { echo "[$(basename $RUN)] $C: ya evaluado"; continue; }
    OUTB=$SESSION/replay/$(basename "$BAG")/$C/odom
    if [ ! -d "$OUTB" ]; then
      docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && \
        python3 tools/hardware/replay_odometry_candidates.py --run $BAG --candidates $C --out-dir $SESSION/replay --no-eval" \
        2>&1 | grep -E "^\[|Error|error" | tail -2
    fi
    docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && \
      python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --run $RUN --hand-eye $STATIC/hand_eye.json --odom-bag $OUTB \
      --odom-topic /replay/odom --label $C" 2>&1 | grep -E "ATE anclada|pérdidas|Error" | sed "s|^|[$(basename $RUN)] $C: |"
  done
done
