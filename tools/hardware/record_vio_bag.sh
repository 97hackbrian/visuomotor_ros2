#!/usr/bin/env bash
# Graba una corrida de medición de la odometría (bag MCAP) desde el contenedor activo de la cámara.
#
# Requisitos: la cámara y RTAB-Map corriendo en otra terminal
#   ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false
# y el tablero ChArUco (charuco_board.yaml) a la vista durante la corrida.
#
# Uso:
#   ./tools/hardware/record_vio_bag.sh <nombre> [duración_s]
#   ./tools/hardware/record_vio_bag.sh reposo_home3 15        # 15 s quieta (ruido de la referencia y de la odometría)
#   ./tools/hardware/record_vio_bag.sh aprox_01                # Ctrl+C para terminar
#
# Tópicos y carpeta: visuomotor_bringup/config/realsense/vio_measurement.yaml (record_topics, bag_dir).
# Después: tools/hardware/evaluate_pose_estimates.py y tools/hardware/replay_odometry_candidates.py.
set -euo pipefail
NAME="${1:?uso: record_vio_bag.sh <nombre> [duración_s]}"
DUR="${2:-0}"
CONTAINER="${CONTAINER:-visuomotor_vo_active}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "El contenedor '${CONTAINER}' (cámara + RTAB-Map) no está corriendo. Lanzarlo antes:"
    echo "  ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false"
    exit 1
fi

docker exec -it -u 1000 -w /ros "${CONTAINER}" bash -c "
set -e
export ROS_LOG_DIR=/tmp/.ros_log_user ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash && source install/setup.bash
CFG=visuomotor_bringup/config/realsense/vio_measurement.yaml
DIR=\$(python3 -c \"import yaml; print(yaml.safe_load(open('\$CFG'))['vio_measurement']['bag_dir'])\")
TOPICS=\$(python3 -c \"import yaml; print(' '.join(yaml.safe_load(open('\$CFG'))['vio_measurement']['record_topics']))\")
OUT=\$DIR/${STAMP}_${NAME}
mkdir -p \$DIR
echo \"Grabando \$OUT\"
echo \"Tópicos: \$TOPICS\"
if [ '${DUR}' != '0' ]; then
    timeout -s INT ${DUR} ros2 bag record -s mcap -o \$OUT \$TOPICS || true
else
    ros2 bag record -s mcap -o \$OUT \$TOPICS || true
fi
ros2 bag info \$OUT | sed -n '1,8p;/Topic information/,\$p'
"
printf '\a'
