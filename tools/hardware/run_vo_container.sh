#!/usr/bin/env bash
# Lanzador de Docker con passthrough USB dinámico completo para RealSense D415 + RTAB-Map VO
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="visuomotor_ros2_vo:jazzy"
CONTAINER_NAME="visuomotor_vo_active"

echo "=== Lanzador RTAB-Map Visual Odometry (Docker USB Passthrough) ==="
echo "Workspace: $WORKSPACE_DIR"
echo "Imagen:    $IMAGE_NAME"

# Si el contenedor ya existe, removerlo para iniciar limpio
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Configurar permisos X11 en Hyprland si DISPLAY está activo
if [ -n "$DISPLAY" ]; then
    xhost +local:root 2>/dev/null || true
    xhost +local:1000 2>/dev/null || true
fi

# Flags interactivos si hay TTY
INTERACTIVE_FLAGS=""
if [ -t 0 ]; then
    INTERACTIVE_FLAGS="-it"
fi

# Ejecutar contenedor con /dev y /dev/bus/usb mapeados en vivo
docker run $INTERACTIVE_FLAGS --rm \
    --name $CONTAINER_NAME \
    --net=host \
    --ipc=host \
    --privileged \
    --gpus all \
    --group-add 44 \
    --group-add 46 \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ROS_DOMAIN_ID=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev:/dev:rw \
    -v /dev/bus/usb:/dev/bus/usb:rw \
    -v "$WORKSPACE_DIR":/ros:rw \
    -w /ros \
    -u 1000:1000 \
    $IMAGE_NAME \
    bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && $*"
