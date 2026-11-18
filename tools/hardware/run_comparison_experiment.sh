#!/usr/bin/env bash
# Experimento comparativo automatizado: Baseline (ORB) vs Ultra-Precisión (SIFT + Subpíxel)
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="visuomotor_ros2_vo:jazzy"
CONTAINER_NAME="vo_benchmark_runner"

echo "=========================================================================="
echo " EXPERIMENTO COMPARATIVO DE PRECISIÓN Y FRECUENCIA (VO REAL)"
echo " Baseline (ORB) vs Ultra-Precisión (SIFT + Subpíxel + g2o BA)"
echo "=========================================================================="

cleanup() {
    echo "Limpiando contenedores del benchmark..."
    docker stop -t 2 $CONTAINER_NAME 2>/dev/null || true
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cleanup

# -----------------------------------------------------------------------------
# FASE 1: BASELINE (ORB)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/2] Evaluando Configuración BASELINE (ORB, Vis/FeatureType=8)..."

docker run -d --rm \
    --name $CONTAINER_NAME \
    --net=host \
    --ipc=host \
    --privileged \
    --gpus all \
    --group-add 44 \
    --group-add 46 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ROS_DOMAIN_ID=0 \
    -v /dev:/dev:rw \
    -v /dev/bus/usb:/dev/bus/usb:rw \
    -v "$WORKSPACE_DIR":/ros:rw \
    -w /ros \
    -u 1000:1000 \
    $IMAGE_NAME \
    bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch visuomotor_bringup realsense_d415_vo.launch.py ultra_precision:=false"

echo "Esperando 6s para estabilización del stream óptico y mapa inicial..."
sleep 6

echo "Ejecutando muestreo estadístico de odometría (6.0s)..."
BASELINE_OUT=$(docker exec -u 1000 $CONTAINER_NAME bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 tools/hardware/benchmark_vo.py --duration 6.0 --label Baseline_ORB")
echo "$BASELINE_OUT"

echo "Deteniendo nodo baseline..."
docker stop -t 2 $CONTAINER_NAME 2>/dev/null || true
sleep 4

# -----------------------------------------------------------------------------
# FASE 2: ULTRA-PRECISIÓN (SIFT + Subpixel + g2o BA)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [2/2] Evaluando Configuración ULTRA-PRECISIÓN (SIFT, Vis/FeatureType=1 + Subpíxel + BA)..."

docker run -d --rm \
    --name $CONTAINER_NAME \
    --net=host \
    --ipc=host \
    --privileged \
    --gpus all \
    --group-add 44 \
    --group-add 46 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ROS_DOMAIN_ID=0 \
    -v /dev:/dev:rw \
    -v /dev/bus/usb:/dev/bus/usb:rw \
    -v "$WORKSPACE_DIR":/ros:rw \
    -w /ros \
    -u 1000:1000 \
    $IMAGE_NAME \
    bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch visuomotor_bringup realsense_d415_vo.launch.py ultra_precision:=true"

echo "Esperando 6s para estabilización del stream óptico y mapa inicial..."
sleep 6

echo "Ejecutando muestreo estadístico de odometría (6.0s)..."
ULTRA_OUT=$(docker exec -u 1000 $CONTAINER_NAME bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 tools/hardware/benchmark_vo.py --duration 6.0 --label Ultra_Precision_SIFT")
echo "$ULTRA_OUT"

echo "Deteniendo nodo ultra-precisión..."
docker stop -t 2 $CONTAINER_NAME 2>/dev/null || true

echo ""
echo "=========================================================================="
echo " EXPERIMENTO COMPLETADO CON ÉXITO"
echo "=========================================================================="
