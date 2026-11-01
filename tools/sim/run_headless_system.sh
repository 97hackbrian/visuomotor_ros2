#!/usr/bin/env bash
# ==============================================================================
# run_headless_system.sh — Lanzador Maestro Headless: Isaac Sim + ROS 2 Jazzy
# ==============================================================================
# Orquesta el ciclo de vida completo de simulación headless acelerada (3x a 7x)
# sin ventanas gráficas ni sobrecarga de VRAM en Display X11/Wayland.
#
# Uso:
#   ./tools/sim/run_headless_system.sh [sim-only|collect|rl|benchmark|inference|shell] [--stage ESCENARIO]
# ==============================================================================

set -uo pipefail

# Colores terminales
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # Sin color

ISAAC_CONTAINER="isaac-sim-6.0-wayland"
ROS_CONTAINER="visuomotor_ros2"
DEFAULT_STAGE="new_world_3.usda"
STAGE="$DEFAULT_STAGE"
MODE=""
EXTRA_ARGS=()
USER_FLAGS="-u $(id -u):$(id -g)"
DOCKER_IT="-i"
if [ -t 0 ]; then
    DOCKER_IT="-it"
fi
xhost +local:root >/dev/null 2>&1 || true

show_help() {
    local code="${1:-0}"
    echo -e "${BOLD}${CYAN}Lanzador Maestro Headless de Isaac Sim + ROS 2 Jazzy${NC}"
    echo -e "Uso: $0 [MODO] [--stage ARCHIVO.usda] [OPCIONES...]\n"
    echo -e "Modos disponibles:"
    echo -e "  ${GREEN}sim-only${NC}    Arranca Isaac Sim Headless + lite6_cartesian y se mantiene a la espera."
    echo -e "  ${GREEN}collect${NC}     Lanza recolección autónoma de datos (pickup_approximate.launch.py)."
    echo -e "  ${GREEN}rl${NC}          Lanza entrenamiento online por refuerzo (rl_training.launch.py)."
    echo -e "  ${GREEN}benchmark${NC}   Ejecuta campaña de benchmarking pareado (compare_models.py)."
    echo -e "  ${GREEN}inference${NC}   Lanza nodo de inferencia de política visuomotora (inference.launch.py)."
    echo -e "  ${GREEN}shell${NC}       Abre una consola interactiva bash en visuomotor_ros2."
    echo -e "  ${GREEN}stop${NC}        Detiene de inmediato todas las simulaciones y controladores activos, liberando VRAM."
    echo -e "\nOpciones generales y por modo:"
    echo -e "  --stage ARCHIVO         Escenario USD (defecto: new_world_3.usda)"
    echo -e "  --episodes, -n NUM      Número de episodios a recolectar o evaluar (en collect y benchmark)"
    echo -e "  --dataset, -d RUTA      Ruta del dataset Zarr (en modo collect)"
    echo -e "  --checkpoint, -c RUTA   Ruta del checkpoint (en modo inference o rl)"
    echo -e "  --dp_ckpt RUTA          Ruta del checkpoint Solo DP (en modo benchmark)"
    echo -e "  --rl_ckpt RUTA          Ruta del checkpoint DP+RL (en modo benchmark)\n"
    exit "$code"
}

EPISODES_ARG=""
DATASET_ARG=""
CHECKPOINT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        sim-only|collect|rl|benchmark|inference|shell|stop)
            MODE="$1"
            shift
            ;;
        --stage)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción --stage requiere un argumento de archivo.${NC}"
                exit 1
            fi
            STAGE="$2"
            shift 2
            ;;
        --episodes|-n)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción $1 requiere un número de episodios.${NC}"
                exit 1
            fi
            EPISODES_ARG="$2"
            shift 2
            ;;
        --dataset|-d)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción $1 requiere la ruta del dataset.${NC}"
                exit 1
            fi
            DATASET_ARG="$2"
            shift 2
            ;;
        --checkpoint|-c)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción $1 requiere la ruta del checkpoint.${NC}"
                exit 1
            fi
            CHECKPOINT_OVERRIDE="$2"
            shift 2
            ;;
        help|-h|--help)
            show_help 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        echo -e "${RED}[ERROR FATAL] Modo no reconocido o ausente. Argumentos desconocidos: ${EXTRA_ARGS[*]}${NC}\n"
        show_help 1
    fi
    show_help 0
fi

# Manejo inmediato del modo STOP sin iniciar simulación ni esperar DDS
if [[ "$MODE" == "stop" ]]; then
    echo -e "${BOLD}${BLUE}======================================================================"
    echo -e "      Deteniendo Sistema: Isaac Sim + ROS 2 Jazzy                     "
    echo -e "======================================================================${NC}"
    if docker ps -q -f name="^/${ISAAC_CONTAINER}$" | grep -q .; then
        echo -e "  [1/2] Deteniendo simulación headless en ${ISAAC_CONTAINER}..."
        docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "headless_autorun" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "kit" 2>/dev/null || true
        sleep 1
        docker exec "${ISAAC_CONTAINER}" pkill -9 -f "kit" 2>/dev/null || true
    fi
    if docker ps -q -f name="^/${ROS_CONTAINER}$" | grep -q .; then
        echo -e "  [2/2] Deteniendo nodos y controladores en ${ROS_CONTAINER}..."
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "lite6_cartesian" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "ros2_control_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "spawner" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "pickup_approximate" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "rl_train_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "policy_node" 2>/dev/null || true
        sleep 1
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -9 -f "ros2_control_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null && ros2 daemon stop 2>/dev/null" || true
    fi
    echo -e "\n${BOLD}${GREEN}✔ Todos los procesos han sido detenidos exitosamente. VRAM liberada.${NC}"
    exit 0
fi

echo -e "${BOLD}${BLUE}======================================================================"
echo -e "      Iniciando Sistema Headless: Isaac Sim + ROS 2 Jazzy             "
echo -e "======================================================================${NC}"
echo -e "Modo de ejecución:  ${BOLD}${GREEN}${MODE}${NC}"
echo -e "Stage seleccionado: ${BOLD}${YELLOW}${STAGE}${NC}"
echo -e "Red / Middleware:   ${BOLD}CycloneDDS | ROS_DOMAIN_ID=42 | NetworkMode=host${NC}"
echo -e "----------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# 1. Rutina de Limpieza y Recuperación de Recursos (trap cleanup)
# ------------------------------------------------------------------------------
cleanup() {
    trap - EXIT INT TERM
    local exit_code=$?
    echo -e "\n${BOLD}${YELLOW}[CLEANUP] Deteniendo procesos y restaurando GPU...${NC}"

    # Detener controladores ROS 2 en visuomotor_ros2
    if docker ps -q -f name="^/${ROS_CONTAINER}$" | grep -q .; then
        echo -e "[CLEANUP] Finalizando nodos en ${ROS_CONTAINER}..."
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "lite6_cartesian" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "ros2_control_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "spawner" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "pickup_approximate" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "rl_train_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "policy_node" 2>/dev/null || true
        sleep 1
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -9 -f "ros2_control_node" 2>/dev/null || true
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null && ros2 daemon stop 2>/dev/null" || true
    fi

    # Detener Isaac Sim Headless en isaac-sim-6.0-wayland
    if docker ps -q -f name="^/${ISAAC_CONTAINER}$" | grep -q .; then
        echo -e "[CLEANUP] Finalizando simulación headless en ${ISAAC_CONTAINER}..."
        docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "headless_autorun" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "kit" 2>/dev/null || true
        sleep 1
        docker exec "${ISAAC_CONTAINER}" pkill -9 -f "kit" 2>/dev/null || true
    fi

    echo -e "${BOLD}${GREEN}[CLEANUP] Limpieza concluida. VRAM liberada.${NC}"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# 2. Verificación y Arranque de Contenedores Docker
# ------------------------------------------------------------------------------
echo -e "${CYAN}[1/5] Verificando estado de contenedores Docker...${NC}"

if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}[ERROR FATAL] El comando 'docker' no está instalado en el Host.${NC}"
    exit 1
fi

# Contenedor Isaac Sim
if ! docker inspect -f '{{.State.Running}}' "${ISAAC_CONTAINER}" 2>/dev/null | grep -q "true"; then
    echo -e "${YELLOW}Arrancando contenedor ${ISAAC_CONTAINER}...${NC}"
    docker start "${ISAAC_CONTAINER}" >/dev/null
    sleep 2
fi
echo -e "  ✔ ${ISAAC_CONTAINER}: ${GREEN}En ejecución${NC}"

# Contenedor ROS 2 Visuomotor
if ! docker inspect -f '{{.State.Running}}' "${ROS_CONTAINER}" 2>/dev/null | grep -q "true"; then
    echo -e "${YELLOW}Arrancando contenedor ${ROS_CONTAINER}...${NC}"
    docker start "${ROS_CONTAINER}" >/dev/null
    sleep 2
fi
echo -e "  ✔ ${ROS_CONTAINER}: ${GREEN}En ejecución${NC}"

# ------------------------------------------------------------------------------
# 3. Lanzamiento de Isaac Sim Headless
# ------------------------------------------------------------------------------
echo -e "${CYAN}[2/5] Verificando estado de Isaac Sim Headless en ${ISAAC_CONTAINER}...${NC}"

# Comprobar si Isaac Sim ya está activo y publicando tópicos en CycloneDDS
EXISTING_TOPICS=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_LOCALHOST_ONLY=1
    export ROS_DOMAIN_ID=42
    ros2 topic list 2>/dev/null
" 2>/dev/null || true)

if echo "$EXISTING_TOPICS" | grep -E '^/rgb$' >/dev/null && echo "$EXISTING_TOPICS" | grep -E '^/isaac/joint_states$' >/dev/null && echo "$EXISTING_TOPICS" | grep -E '^/clock$' >/dev/null; then
    echo -e "  ✔ Isaac Sim Headless ya se encuentra activo y publicando /rgb, /isaac/joint_states y /clock en CycloneDDS. Reutilizando instancia."
else
    # Matar instancias previas si existieran
    docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
    docker exec "${ISAAC_CONTAINER}" pkill -f "kit" 2>/dev/null || true
    sleep 1

    # Lanzar ejecutor headless en segundo plano
    docker exec -d "${ISAAC_CONTAINER}" bash -c "
        cd /isaac-sim 2>/dev/null || true
        if [ -f /isaac-sim/setup_ros_env.sh ]; then
            source /isaac-sim/setup_ros_env.sh
        elif [ -f ./setup_ros_env.sh ]; then
            source ./setup_ros_env.sh
        fi
        export ROS_DISTRO=jazzy
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export ROS_LOCALHOST_ONLY=1
        export ROS_DOMAIN_ID=42
        export LD_LIBRARY_PATH=\"/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\${LD_LIBRARY_PATH:-}\"
        export OMNI_KIT_ACCEPT_EULA=YES
        export CAMERA_FRAME_SKIP="${CAMERA_FRAME_SKIP:-1}"
        /isaac-sim/isaac-sim.streaming.sh --no-window \
            --/app/omni.graph.scriptnode/enable_opt_in=false \
            --/app/omni.graph.scriptnode/opt_in=true \
            --/app/runLoops/main/rateLimitEnabled=false \
            --/app/runLoops/rendering/rateLimitEnabled=false \
            --/app/player/playEveryFrame=true \
            --/persistent/simulation/minFrameRate=0 \
            --/rtx/ecoMode/enabled=true \
            --/rtx/rt/ecoMode/enabled=true \
            --/rtx/post/aa/op=3 \
            --/rtx/post/dlss/execMode=0 \
            --/rtx/dlss/frameGeneration=true \
            --/rtx-transient/dlssg/enabled=true \
            --/rtx-transient/internal/dlssg/rtpt/interpolatedFrameCount=1 \
            --exec /workspace/robo_imitate/xarm_bringup/isaac/headless_autorun.py > /tmp/isaac_headless.log 2>&1
    "
    echo -e "  ✔ Isaac Sim Headless lanzado. Registro en /tmp/isaac_headless.log"
fi

# ------------------------------------------------------------------------------
# 4. Sondeo DDS: Esperar tópicos /rgb e /isaac/joint_states
# ------------------------------------------------------------------------------
echo -e "${CYAN}[3/5] Sondeando bus DDS para tópicos /rgb e /isaac/joint_states...${NC}"

MAX_WAIT=180
ELAPSED=0
TOPICS_FOUND=false

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    TOPIC_LIST=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
        source /opt/ros/jazzy/setup.bash 2>/dev/null || true
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export ROS_LOCALHOST_ONLY=1
        export ROS_DOMAIN_ID=42
        ros2 topic list 2>/dev/null
    " 2>/dev/null || true)

    HAS_RGB=$(echo "$TOPIC_LIST" | grep -E '^/rgb$' || true)
    HAS_JS=$(echo "$TOPIC_LIST" | grep -E '^/isaac/joint_states$' || true)
    HAS_CLOCK=$(echo "$TOPIC_LIST" | grep -E '^/clock$' || true)

    if [[ -n "$HAS_RGB" && -n "$HAS_JS" && -n "$HAS_CLOCK" ]]; then
        TOPICS_FOUND=true
        break
    fi

    echo -ne "  ⏳ Esperando inicialización de stage y ROS 2 bridge... (${ELAPSED}/${MAX_WAIT}s)\r"
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done
echo ""

if [ "$TOPICS_FOUND" = false ]; then
    echo -e "${RED}[ERROR FATAL] Timeout esperando tópicos de Isaac Sim tras ${MAX_WAIT} segundos.${NC}"
    echo -e "${YELLOW}Últimas líneas del registro en ${ISAAC_CONTAINER}:/tmp/isaac_headless.log :${NC}"
    docker exec "${ISAAC_CONTAINER}" tail -n 25 /tmp/isaac_headless.log 2>/dev/null || true
    exit 1
fi

echo -e "  ✔ Tópicos sensoriales confirmados en CycloneDDS:"
echo -e "    - /rgb: ${GREEN}ACTIVO${NC}"
echo -e "    - /isaac/joint_states: ${GREEN}ACTIVO${NC}"
echo -e "    - /clock: ${GREEN}ACTIVO${NC}"

# ------------------------------------------------------------------------------
# 5. Lanzamiento en Segundo Plano de lite6_cartesian.launch.py
# ------------------------------------------------------------------------------
echo -e "${CYAN}[4/5] Lanzando pila cinemática cartesiana en ${ROS_CONTAINER}...${NC}"

# Limpiar instancias previas de controladores
docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "ros2_control_node" 2>/dev/null || true
docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "spawner" 2>/dev/null || true
docker exec ${USER_FLAGS} "${ROS_CONTAINER}" pkill -f "lite6_cartesian" 2>/dev/null || true
sleep 1

docker exec -d ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
    source /ros/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_LOCALHOST_ONLY=1
    export ROS_DOMAIN_ID=42
    ros2 launch visuomotor_bringup lite6_cartesian.launch.py rviz:=false sim:=true > /tmp/lite6_cartesian.log 2>&1
"

echo -e "  Esperando estabilización de controladores cartesianos (6s)..."
sleep 6
echo -e "  ✔ Pila cinemática cartesiana activa (rviz=false, sim=true)."

# ------------------------------------------------------------------------------
# 6. Despacho Modular de Tareas
# ------------------------------------------------------------------------------
echo -e "${CYAN}[5/5] Ejecutando tarea solicitada: ${BOLD}${GREEN}${MODE}${NC}..."
echo -e "----------------------------------------------------------------------"

# Enlazar banderas ergonómicas a los argumentos de cada modo
case "$MODE" in
    collect)
        if [[ -n "${EPISODES_ARG:-}" ]]; then
            EXTRA_ARGS+=("max_episodes:=${EPISODES_ARG}" "auto_start:=true")
        fi
        if [[ -n "${DATASET_ARG:-}" ]]; then
            EXTRA_ARGS+=("dataset_path:=${DATASET_ARG}")
        fi
        ;;
    benchmark)
        if [[ -n "${EPISODES_ARG:-}" ]]; then
            EXTRA_ARGS+=("--episodes" "${EPISODES_ARG}")
        fi
        ;;
    inference|rl)
        if [[ -n "${CHECKPOINT_OVERRIDE:-}" ]]; then
            EXTRA_ARGS+=("checkpoint_path:=${CHECKPOINT_OVERRIDE}")
        fi
        ;;
esac

case "$MODE" in
    sim-only)
        echo -e "${GREEN}Simulación Headless y Control Cartesiano operativos.${NC}"
        echo -e "Presiona ${BOLD}Ctrl+C${NC} para detener la sesión y liberar los recursos."
        echo -e "Monitoreando telemetría de Isaac Sim (/tmp/isaac_headless.log):"
        docker exec ${DOCKER_IT} "${ISAAC_CONTAINER}" tail -f /tmp/isaac_headless.log
        ;;

    collect)
        echo -e "${BOLD}Iniciando Recolección Autónoma de Demostraciones Experto...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            ros2 launch visuomotor_data_collection pickup_approximate.launch.py ${EXTRA_ARGS[*]:-}
        "
        ;;

    rl)
        echo -e "${BOLD}Iniciando Entrenamiento por Refuerzo (DPPO Fine-Tuning)...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            ros2 launch visuomotor_ros2 rl_training.launch.py ${EXTRA_ARGS[*]:-}
        "
        ;;

    benchmark)
        echo -e "${BOLD}Iniciando Benchmarking Comparativo Pareado (Solo DP vs DP+RL)...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            python3 tools/compare_models.py ${EXTRA_ARGS[*]:-}
        "
        ;;

    inference)
        echo -e "${BOLD}Iniciando Inferencia en Vivo de Política Visuomotora...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            ros2 launch visuomotor_ros2 inference.launch.py ${EXTRA_ARGS[*]:-}
        "
        ;;

    shell)
        echo -e "${BOLD}Abriendo shell interactiva en ${ROS_CONTAINER} bajo usuario hackbrian...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            bash
        "
        ;;

    *)
        echo -e "${RED}[ERROR] Modo desconocido: '${MODE}'. Consulta con --help.${NC}"
        exit 1
        ;;
esac
