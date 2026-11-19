#!/usr/bin/env bash
# tools/hardware/run_benchmark.sh
# Lanza el benchmark interactivo conectándose al contenedor activo de VIO

LABEL="${1:-conf 0 (Baseline)}"
CONTAINER_NAME="visuomotor_vo_active"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "=========================================================================="
    echo " [ERROR] El contenedor '$CONTAINER_NAME' NO está corriendo."
    echo " Primero ejecuta en otra terminal:"
    echo "   ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=true"
    echo "=========================================================================="
    exit 1
fi

docker exec -it "$CONTAINER_NAME" bash -c \
    "export ROS_LOCALHOST_ONLY=1 && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST && source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 tools/benchmark_vio_pose.py --label '$LABEL'"
