#!/usr/bin/env bash
# tools/hardware/run_data_collection.sh
# Lanza el recolector interactivo de demostraciones conectándose al contenedor activo de VIO.
# Soporta modo AUTO (demonstrations_real_vX.zarr) o continuar/aumentar un dataset existente.

DATASET_ARG="${1:-AUTO}"
CONTAINER_NAME="visuomotor_vo_active"

echo "=========================================================================="
echo "    RECOLECTOR DE DEMOSTRACIONES EXPERTAS REALES (VIO UMI-GRADE)"
echo "=========================================================================="
if [ "$DATASET_ARG" = "AUTO" ]; then
    echo " Modo: AUTO -> Se creará automáticamente un nuevo demonstrations_real_vX.zarr"
else
    echo " Modo: ESPECÍFICO -> Continuando/guardando en: $DATASET_ARG"
fi
echo "=========================================================================="

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo " [AVISO] El contenedor '$CONTAINER_NAME' con la cámara y VIO no está corriendo."
    echo " Para recolectar demostraciones, primero debes iniciar el stream en otra terminal:"
    echo "   ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false"
    echo ""
    echo " Alternativamente, puedes lanzar todo en un único comando integrado con:"
    echo "   ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio_collection.launch.py dataset_path:=$DATASET_ARG"
    echo "=========================================================================="
    exit 1
fi

docker exec -it -u 1000 "$CONTAINER_NAME" bash -c \
    "export ROS_LOCALHOST_ONLY=1 && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 run visuomotor_data_collection vo_collector --ros-args -p dataset_path:='$DATASET_ARG'"
