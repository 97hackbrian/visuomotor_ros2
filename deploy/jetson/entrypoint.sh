#!/usr/bin/env bash
set -e

# Cargar el entorno de ROS 2 (en la imagen base de Jetson vive en /opt/ros/<distro>/install)
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO}/install/setup.bash}"
if [ ! -f "${ROS_SETUP}" ]; then
    ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
fi
if [ -f "${ROS_SETUP}" ]; then
    source "${ROS_SETUP}"
else
    echo "[entrypoint] ERROR: no se encontró setup.bash de ROS (${ROS_SETUP})" >&2
    exit 1
fi

# Driver del xArm5 (workspace subyacente; /ros_ws también lo encadena, esto es por robustez)
if [ -n "${XARM_WS}" ] && [ -f "${XARM_WS}/install/setup.bash" ]; then
    source "${XARM_WS}/install/setup.bash"
fi

# Cargar los paquetes compilados del workspace visuomotor
if [ -f "/ros_ws/install/setup.bash" ]; then
    source "/ros_ws/install/setup.bash"
fi

# Driver de la RealSense (solo en la imagen construida sobre Dockerfile.realsense_base). Se carga al final para que
# quede en AMENT_PREFIX_PATH aunque /ros_ws no lo haya encadenado al compilar.
if [ -f "/opt/realsense_ws/install/setup.bash" ]; then
    source "/opt/realsense_ws/install/setup.bash"
fi

# Configuración de DDS de baja latencia para comunicación robot/cámara
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}  # dominio del robot real (decisión del usuario, 2026-11-18)

# Ejecutar el comando pasado como argumento
exec "$@"
