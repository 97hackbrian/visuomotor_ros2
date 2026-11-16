#!/usr/bin/env bash
# ==============================================================================
# setup_host.sh — Puesta a Punto del Host NVIDIA Jetson Orin Nano
# ==============================================================================
# Este script se ejecuta UNA SOLA VEZ directamente en la terminal de la Jetson:
#   chmod +x setup_host.sh && ./setup_host.sh
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Iniciando Configuración de Rendimiento para Jetson Orin Nano ===${NC}"

# 1. Configuración de Modo de Energía (15W Máximo Rendimiento)
echo -e "\n${YELLOW}[1/6] Configurando perfil de energía a 15W (Modo 0)...${NC}"
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    echo "Modo de energía actual:"
    sudo nvpmodel -q
else
    echo -e "${RED}nvpmodel no encontrado. Omitiendo...${NC}"
fi

# 2. Fijación de Relojes de CPU y GPU al Máximo (Jetson Clocks)
echo -e "\n${YELLOW}[2/6] Fijando relojes de CPU, GPU y memoria al máximo (jetson_clocks)...${NC}"
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    echo -e "${GREEN}Relojes fijados exitosamente.${NC}"
else
    echo -e "${RED}jetson_clocks no encontrado. Omitiendo...${NC}"
fi

# 3. Configuración de Memoria SWAP de 8 GB (Crítico para evitar OOM Killer)
echo -e "\n${YELLOW}[3/6] Verificando memoria de intercambio (SWAP)...${NC}"
CURRENT_SWAP=$(free -g | awk '/^Swap:/ {print $2}')
if [ "$CURRENT_SWAP" -lt 6 ]; then
    echo "SWAP actual ($CURRENT_SWAP GB) es menor a 8 GB. Creando swapfile de 8 GB..."
    if [ ! -f /swapfile_8g ]; then
        sudo fallocate -l 8G /swapfile_8g
        sudo chmod 600 /swapfile_8g
        sudo mkswap /swapfile_8g
    fi
    sudo swapon /swapfile_8g || true
    if ! grep -q "/swapfile_8g" /etc/fstab; then
        echo "/swapfile_8g none swap sw 0 0" | sudo tee -a /etc/fstab
    fi
    echo -e "${GREEN}SWAP de 8 GB configurado y activado.${NC}"
else
    echo -e "${GREEN}SWAP suficiente detectado ($CURRENT_SWAP GB).${NC}"
fi

# 4. Asegurar Runtime de NVIDIA por defecto en Docker
echo -e "\n${YELLOW}[4/6] Verificando runtime nvidia en /etc/docker/daemon.json...${NC}"
DAEMON_JSON="/etc/docker/daemon.json"
if [ -f "$DAEMON_JSON" ]; then
    if ! grep -q '"default-runtime": "nvidia"' "$DAEMON_JSON"; then
        echo "Añadiendo default-runtime: nvidia a $DAEMON_JSON..."
        sudo cp "$DAEMON_JSON" "${DAEMON_JSON}.bak"
        sudo python3 -c '
import json
p = "/etc/docker/daemon.json"
with open(p, "r") as f:
    d = json.load(f)
d["default-runtime"] = "nvidia"
with open(p, "w") as f:
    json.dump(d, f, indent=4)
'
        sudo systemctl restart docker
    else
        echo -e "${GREEN}default-runtime nvidia ya está configurado.${NC}"
    fi
else
    echo "Creando $DAEMON_JSON con default-runtime nvidia..."
    sudo mkdir -p /etc/docker
    sudo bash -c 'cat > /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF'
    sudo systemctl restart docker
fi

# 5. Agregar usuario actual al grupo docker
echo -e "\n${YELLOW}[5/6] Asegurando permisos de usuario en grupo docker...${NC}"
sudo usermod -aG docker "$USER" || true

# 6. Reducción de Latencia en Puertos USB Serie (1ms para control reactivo de robot)
echo -e "\n${YELLOW}[6/6] Configurando temporizador de latencia USB serial a 1ms...${NC}"
for dev in /sys/bus/usb-serial/devices/ttyUSB*; do
    if [ -f "$dev/latency_timer" ]; then
        echo 1 | sudo tee "$dev/latency_timer" > /dev/null || true
        echo "Latencia reducida a 1ms en: $(basename $dev)"
    fi
done

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN} ¡Configuración del Host completada con éxito!${NC}"
echo -e "${GREEN} Si modificaste el grupo docker, cierra sesión o ejecuta: newgrp docker${NC}"
echo -e "${GREEN}================================================================${NC}"
