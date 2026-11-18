#!/usr/bin/env bash
set -e

# Descubrimiento dinámico de la Jetson (cero IP hardcodeada en el repositorio)
WIFI_IFACE=$(ip -4 route show default 2>/dev/null | awk '{print $5}')
WIFI_IFACE="${WIFI_IFACE:-wlo1}"

# 1. Si el usuario pasa la IP como argumento, se usa directamente:
if [ -n "$1" ]; then
    JETSON_IP="$1"
# 2. Si no, autodetectar en la subred Wi-Fi activa (excluyendo el gateway del router):
else
    # Buscar vecino activo con IP en la subred 172.16.x.x
    JETSON_IP=$(ip -4 neigh show dev "$WIFI_IFACE" nud reachable 2>/dev/null | grep -E "172\.16\." | grep -v "172.16.32.1" | head -n1 | awk '{print $1}')
    
    # Si aún no está en caché ARP, intentar resolver por mDNS
    if [ -z "$JETSON_IP" ]; then
        JETSON_IP=$(getent ahostsv4 ubuntu.local 2>/dev/null | grep -E "172\.16\." | head -n1 | awk '{print $1}')
    fi
fi

if [ -z "$JETSON_IP" ]; then
    echo "⚠️  No se detectó automáticamente la IP de la Jetson en la red Wi-Fi ($WIFI_IFACE)."
    echo "   Por favor especifícala manualmente: $0 <IP_JETSON>"
    exit 1
fi

echo "================================================================="
echo "🤖 xArm5 Remote Teleoperation — RViz2"
echo "🌐 Jetson detectada dinámicamente en: ${JETSON_IP} (vía ${WIFI_IFACE})"
echo "📡 Canal DDS: Dominio 42 (CycloneDDS Unicast Peer)"
echo "================================================================="

# Autorizar display local para RViz2 en Docker
xhost +local:root >/dev/null 2>&1 || true
xhost +local:hackbrian >/dev/null 2>&1 || true

docker exec -it \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}" \
    -e DDS_PEER="${JETSON_IP}" \
    visuomotor_ros2 bash -c "\
        source /ros/.bashrc && \
        export DDS_PEER=${JETSON_IP} && \
        ros2 launch visuomotor_bringup xarm5_rviz.launch.py"
