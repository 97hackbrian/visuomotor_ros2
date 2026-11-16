#!/usr/bin/env bash
set -e

# Cargar el entorno de ROS 2 instalado en el contenedor
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# Cargar los paquetes compilados del workspace visuomotor
if [ -f "/ros_ws/install/setup.bash" ]; then
    source "/ros_ws/install/setup.bash"
fi

# Configuración de DDS de baja latencia para comunicación robot/cámara
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

# Ejecutar el comando pasado como argumento
exec "$@"
