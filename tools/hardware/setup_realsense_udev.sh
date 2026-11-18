#!/usr/bin/env bash
# Script para configurar reglas udev de Intel RealSense en el Host Ubuntu 24.04
set -e

echo "=== Configuración de Reglas udev para Intel RealSense en Host ==="

if [ -f /etc/udev/rules.d/99-realsense-libusb.rules.bak ]; then
    echo "Restaurando /etc/udev/rules.d/99-realsense-libusb.rules desde .bak..."
    sudo cp /etc/udev/rules.d/99-realsense-libusb.rules.bak /etc/udev/rules.d/99-realsense-libusb.rules
fi

# Desactivar iio-sensor-proxy para evitar bloqueo del chip IMU (Bosch BMI085) en laptops
if systemctl is-active --quiet iio-sensor-proxy.service; then
    echo "Desactivando iio-sensor-proxy para liberar el IMU de RealSense..."
    sudo systemctl stop iio-sensor-proxy.service || true
    sudo systemctl mask iio-sensor-proxy.service || true
fi

if [ ! -f /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    echo "Descargando reglas udev oficiales de Intel RealSense..."
    sudo wget -q https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules -O /etc/udev/rules.d/99-realsense-libusb.rules
fi

echo "Recargando reglas de udev y disparando eventos..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Agregando usuario $USER a grupos video y plugdev..."
sudo usermod -aG video,plugdev $USER

echo "✔ Reglas udev configuradas correctamente."
