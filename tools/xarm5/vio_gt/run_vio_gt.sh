#!/usr/bin/env bash
# Una corrida de medición de la odometría con el xArm5 como referencia (Fase 2.2), orquestada desde el PC.
#   tools/xarm5/vio_gt/run_vio_gt.sh <plan> [--no-arm]
#   tools/xarm5/vio_gt/run_vio_gt.sh static_board --no-arm     # solo chequeo previo en la Jetson, no mueve
# Requisitos: cámara D435i en ESTE equipo con la odometría corriendo
#   (./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false),
# puente del xArm5 lanzado en la Jetson (modo real, desarmado), brazo en HOME3 y el usuario en la parada de emergencia.
# Planes: tools/xarm5/vio_gt/plan_vio_gt.py -> $SESSION/plans/<plan>.json
# Pasos: copia plan y script a la Jetson, mide el desfase de relojes (PC - Jetson), reinicia la odometría con el brazo
# quieto, graba el bag (tópicos de vio_measurement.yaml), ejecuta el plan con vigilante y trae el registro del TCP.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
JETSON=${JETSON:-jetson@172.16.35.107}; CONTAINER=${CONTAINER:-visuomotor_orin_inference}
CAM=${CAM:-visuomotor_vo_active}
SESSION=${SESSION:-$REPO/experiments/vio_gt/$(date +%F)}
PLAN=${1:?uso: run_vio_gt.sh <plan> [--no-arm]}; NOARM=${2:-}
PJ=$SESSION/plans/$PLAN.json; [ -f "$PJ" ] || { echo "no existe $PJ"; exit 1; }
J="ssh -o BatchMode=yes $JETSON"
BEEP=$REPO/tools/xarm5/real_trials/.beeps
play() { paplay "$1" 2>/dev/null || aplay -q "$1" 2>/dev/null || printf '\a'; }
DX="docker exec $CONTAINER /ros_ws/entrypoint.sh"
scp -q -o BatchMode=yes "$PJ" "$HERE/jetson/vio_gt_motion.py" "$JETSON":/tmp/ || { echo "sin conexión con $JETSON"; exit 1; }
$J "docker exec $CONTAINER mkdir -p /tmp/vio_gt && docker cp /tmp/$PLAN.json $CONTAINER:/tmp/vio_gt/ && docker cp /tmp/vio_gt_motion.py $CONTAINER:/tmp/vio_gt/"
if [ "$NOARM" = "--no-arm" ]; then
  timeout 60 $J "$DX python3 /tmp/vio_gt/vio_gt_motion.py /tmp/vio_gt/$PLAN.json --no-arm" 2>&1 | grep "\[vio_gt\]"; exit 0
fi
docker ps --format '{{.Names}}' | grep -q "^$CAM$" || { echo "la cámara ($CAM) no está corriendo"; exit 1; }
OUT=$SESSION/runs/$(date +%H%M%S)_$PLAN; mkdir -p "$OUT"
# Desfase de relojes: la Jetson no tiene NTP. Cada ssh nuevo gasta ~0,7 s en el saludo ANTES de ejecutar date, así
# que el reloj de la Jetson se lee al final de la ida y vuelta: desfase = b - j (no el punto medio). El evaluador lo
# refina con el giróscopo de la cámara (evaluate_vio_gt.py).
for k in $(seq 1 7); do
  a=$(date +%s.%N); j=$($J "date +%s.%N"); b=$(date +%s.%N); echo "$a $j $b"; done > "$OUT/clock_samples.txt"
awk '{rtt=$3-$1; off=$3-$2; if (NR==1||rtt<best){best=rtt; o=off}} END {printf "{\"offset_pc_minus_jetson_s\": %.6f, \"rtt_s\": %.6f}\n", o, best}' \
  "$OUT/clock_samples.txt" > "$OUT/clock.json"; cat "$OUT/clock.json"
BAG=datasets/vio_bags/$(date +%Y%m%d_%H%M%S)_vio_gt_$PLAN
echo "$BAG" > "$OUT/bag_path.txt"
# Odometría reiniciada con el brazo quieto en HOME3, luego el bag (sin TTY: --disable-keyboard-controls)
docker exec -u 1000 -w /ros "$CAM" bash -c "
export ROS_LOG_DIR=/tmp/.ros_log_user; mkdir -p \$ROS_LOG_DIR; source /opt/ros/jazzy/setup.bash; source install/setup.bash
timeout 5 ros2 service call /reset_odom std_srvs/srv/Empty > /dev/null || echo 'AVISO: /reset_odom no respondió'
sleep 1.5
TOPICS=\$(python3 -c \"import yaml; print(' '.join(yaml.safe_load(open('visuomotor_bringup/config/realsense/vio_measurement.yaml'))['vio_measurement']['record_topics']))\")
exec ros2 bag record -s mcap --disable-keyboard-controls -o $BAG --topics \$TOPICS /camera/camera/gyro/sample /camera/camera/accel/sample
" > "$OUT/bag_record.log" 2>&1 &
RECPID=$!
sleep 4
grep -q "Recording" "$OUT/bag_record.log" || sleep 3
play "$BEEP/inicio.wav"; date "+INICIO $PLAN %H:%M:%S"
timeout $(( $(grep -o '"duration_s": [0-9]*' "$PJ" | awk '{print $2}') + 90 )) \
  $J "$DX python3 /tmp/vio_gt/vio_gt_motion.py /tmp/vio_gt/$PLAN.json" 2>&1 | grep "\[vio_gt\]\|{" | tee "$OUT/motion.log"
date "+FIN $PLAN %H:%M:%S"; play "$BEEP/fin.wav"
sleep 2
docker exec "$CAM" pkill -INT -f "ros2 bag record" ; wait $RECPID 2>/dev/null
for f in npz json; do
  $J "docker cp $CONTAINER:/tmp/vio_gt/$PLAN.$f /tmp/vio_gt_$PLAN.$f" && scp -q -o BatchMode=yes "$JETSON:/tmp/vio_gt_$PLAN.$f" "$OUT/robot.$f"; done
cp "$PJ" "$OUT/plan.json"
docker exec -u 1000 -w /ros "$CAM" bash -c "source /opt/ros/jazzy/setup.bash && ros2 bag info $BAG | sed -n '1,8p'"
echo "corrida: ${OUT#$REPO/}"
