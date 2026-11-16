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
    echo -e "  ${GREEN}eval-checkpoints${NC} Evalúa checkpoints de entrenamiento (epoch_10, 20, 30, latest) en Isaac Sim."
    echo -e "  ${GREEN}inference${NC}   Lanza nodo de inferencia de política visuomotora (inference.launch.py)."
    echo -e "  ${GREEN}shell${NC}       Abre una consola interactiva bash en visuomotor_ros2."
    echo -e "  ${GREEN}stop${NC}        Detiene de inmediato todas las simulaciones y controladores activos, liberando VRAM."
    echo -e "\nOpciones generales y por modo:"
    echo -e "  --instances, -i NUM     Número de instancias paralelas para recolección autónoma (defecto: 1)"
    echo -e "  --stage ARCHIVO         Escenario USD (defecto: new_world_3.usda)"
    echo -e "  --episodes, -n NUM      Número de episodios a recolectar o evaluar (en collect y benchmark)"
    echo -e "  --dataset, -d RUTA      Ruta del dataset Zarr (en modo collect)"
    echo -e "  --checkpoint, -c RUTA   Ruta del checkpoint (en modo inference o rl)"
    echo -e "  --dp_ckpt RUTA          Ruta del checkpoint Solo DP (en modo benchmark)"
    echo -e "  --rl_ckpt RUTA          Ruta del checkpoint DP+RL (en modo benchmark)"
    echo -e "  --max_steps, --steps NUM Pasos máximos por episodio en evaluación/benchmark (defecto: 2400 / 120s)\n"
    exit "$code"
}

INSTANCES=1
EPISODES_ARG=""
MAX_STEPS_ARG=""
DATASET_ARG=""
CHECKPOINT_OVERRIDE=""
TARGET_DATASET=""
BATCH_DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        sim-only|collect|rl|benchmark|eval-checkpoints|inference|shell|stop)
            MODE="$1"
            shift
            ;;
        --instances|-i)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción $1 requiere un número de instancias.${NC}"
                exit 1
            fi
            INSTANCES="$2"
            shift 2
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
        --max_steps|--steps)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR FATAL] La opción $1 requiere un número de pasos.${NC}"
                exit 1
            fi
            MAX_STEPS_ARG="$2"
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

if ! [[ "$INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR FATAL] El número de instancias (--instances) debe ser un entero >= 1. Recibido: '${INSTANCES}'${NC}" >&2
    exit 1
fi

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
        echo -e "  [1/2] Deteniendo simulación headless y contenedor ${ISAAC_CONTAINER} (docker stop)..."
        docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "headless_autorun" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "kit" 2>/dev/null || true
        sleep 1
        docker exec "${ISAAC_CONTAINER}" pkill -9 -f "kit" 2>/dev/null || true
        docker stop "${ISAAC_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if docker ps -q -f name="^/${ROS_CONTAINER}$" | grep -q .; then
        echo -e "  [2/2] Deteniendo nodos y controladores en ${ROS_CONTAINER}..."
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
            pkill -9 -f 'lite6_cartesian' 2>/dev/null || true
            pkill -9 -f 'ros2_control_node' 2>/dev/null || true
            pkill -9 -f 'pose_smoother' 2>/dev/null || true
            pkill -9 -f 'spawner' 2>/dev/null || true
            pkill -9 -f 'pickup_approximate' 2>/dev/null || true
            pkill -9 -f 'rl_train_node' 2>/dev/null || true
            pkill -9 -f 'policy_node' 2>/dev/null || true
            pkill -9 -f 'robot_state_publisher' 2>/dev/null || true
            pkill -9 -f 'republish' 2>/dev/null || true
            rm -f /tmp/pickup_inst_*.done /tmp/*.pid 2>/dev/null || true
            source /opt/ros/jazzy/setup.bash 2>/dev/null && ros2 daemon stop 2>/dev/null || true
        " 2>/dev/null || true
    fi
    echo -e "\n${BOLD}${GREEN}✔ Todos los procesos han sido detenidos exitosamente. VRAM liberada.${NC}"
    exit 0
fi

echo -e "${BOLD}${BLUE}======================================================================"
echo -e "      Iniciando Sistema Headless: Isaac Sim + ROS 2 Jazzy             "
echo -e "======================================================================${NC}"
echo -e "Modo de ejecución:  ${BOLD}${GREEN}${MODE}${NC}"
echo -e "Instancias:         ${BOLD}${CYAN}${INSTANCES}${NC}"
echo -e "Stage seleccionado: ${BOLD}${YELLOW}${STAGE}${NC}"
if [ "$INSTANCES" -eq 1 ]; then
    echo -e "Red / Middleware:   ${BOLD}CycloneDDS | ROS_DOMAIN_ID=42 | NetworkMode=host${NC}"
else
    echo -e "Red / Middleware:   ${BOLD}CycloneDDS | ROS_DOMAIN_ID=42..$((42 + INSTANCES - 1)) (Aislado por instancia) | NetworkMode=host${NC}"
fi
echo -e "----------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# 1. Rutina de Limpieza y Recuperación de Recursos (trap cleanup)
# ------------------------------------------------------------------------------
cleanup() {
    trap - EXIT INT TERM
    local exit_code=$?
    echo -e "\n${BOLD}${YELLOW}[CLEANUP] Deteniendo procesos y restaurando GPU...${NC}"

    # 1. Finalizar primero los nodos y recolectores en visuomotor_ros2 para evitar carreras al consolidar
    if docker ps -q -f name="^/${ROS_CONTAINER}$" | grep -q .; then
        echo -e "[CLEANUP] Finalizando nodos en ${ROS_CONTAINER}..."
        docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
            pkill -9 -f 'pickup_approximate' 2>/dev/null || true
            pkill -9 -f 'rl_train_node' 2>/dev/null || true
            pkill -9 -f 'policy_node' 2>/dev/null || true
            pkill -9 -f 'lite6_cartesian' 2>/dev/null || true
            pkill -9 -f 'ros2_control_node' 2>/dev/null || true
            pkill -9 -f 'pose_smoother' 2>/dev/null || true
            pkill -9 -f 'spawner' 2>/dev/null || true
            pkill -9 -f 'robot_state_publisher' 2>/dev/null || true
            pkill -9 -f 'republish' 2>/dev/null || true
            rm -f /tmp/pickup_inst_*.done /tmp/*.pid 2>/dev/null || true
            source /opt/ros/jazzy/setup.bash 2>/dev/null && ros2 daemon stop 2>/dev/null || true
        " 2>/dev/null || true
    fi

    # 2. Si estábamos recolectando en paralelo y existen batches temporales, consolidar episodios capturados
    if [[ "${MODE:-}" == "collect" && "${INSTANCES:-1}" -gt 1 && -n "${TARGET_DATASET:-}" && ${#BATCH_DATASETS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}[CLEANUP] Consolidando episodios capturados en lotes temporales hacia ${TARGET_DATASET}...${NC}"
        docker exec ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" python3 /ros/tools/merge_zarr_batches.py \
            --inputs "${BATCH_DATASETS[@]}" \
            --output "${TARGET_DATASET}" \
            --remove-sources 2>/dev/null || true
    fi

    # 3. Detener Isaac Sim Headless en isaac-sim-6.0-wayland
    if docker ps -q -f name="^/${ISAAC_CONTAINER}$" | grep -q .; then
        echo -e "[CLEANUP] Finalizando simulación headless en ${ISAAC_CONTAINER}..."
        docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "headless_autorun" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "kit" 2>/dev/null || true
        sleep 1
        docker exec "${ISAAC_CONTAINER}" pkill -9 -f "kit" 2>/dev/null || true
        docker stop "${ISAAC_CONTAINER}" >/dev/null 2>&1 || true
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

# Contenedor Isaac Sim: Purgar estado residual y asegurar arranque limpio mediante stop + start (sin eliminar)
echo -e "  Verificando contenedor ${ISAAC_CONTAINER}..."
if docker inspect -f '{{.State.Running}}' "${ISAAC_CONTAINER}" 2>/dev/null | grep -q "true"; then
    echo -e "  ${YELLOW}Reiniciando limpiamente contenedor ${ISAAC_CONTAINER} (docker stop + docker start)...${NC}"
    docker stop "${ISAAC_CONTAINER}" >/dev/null 2>&1 || true
    sleep 1
    docker start "${ISAAC_CONTAINER}" >/dev/null
    sleep 2
else
    echo -e "  ${YELLOW}Arrancando contenedor ${ISAAC_CONTAINER} (docker start)...${NC}"
    docker start "${ISAAC_CONTAINER}" >/dev/null
    sleep 2
fi

if docker inspect -f '{{.State.Running}}' "${ISAAC_CONTAINER}" 2>/dev/null | grep -q "true"; then
    echo -e "  ✔ ${ISAAC_CONTAINER}: ${GREEN}En ejecución (ciclo stop/start verificado)${NC}"
else
    echo -e "${RED}[ERROR FATAL] No se pudo inicializar el contenedor ${ISAAC_CONTAINER}.${NC}"
    exit 1
fi

# Contenedor ROS 2 Visuomotor
if ! docker inspect -f '{{.State.Running}}' "${ROS_CONTAINER}" 2>/dev/null | grep -q "true"; then
    echo -e "${YELLOW}Arrancando contenedor ${ROS_CONTAINER}...${NC}"
    docker start "${ROS_CONTAINER}" >/dev/null
    sleep 2
fi
echo -e "  ✔ ${ROS_CONTAINER}: ${GREEN}En ejecución${NC}"

# Purgar caché del daemon de ROS 2 en ROS_CONTAINER para evitar lecturas de tópicos fantasma
docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash 2>/dev/null && ros2 daemon stop 2>/dev/null || true"

# ------------------------------------------------------------------------------
# 3. Lanzamiento de Isaac Sim Headless
# ------------------------------------------------------------------------------
echo -e "${CYAN}[2/5] Verificando estado de Isaac Sim Headless (${INSTANCES} instancia(s)) en ${ISAAC_CONTAINER}...${NC}"

if [ "$INSTANCES" -eq 1 ]; then
    KIT_PID=$(docker exec "${ISAAC_CONTAINER}" pgrep -f "kit" 2>/dev/null || true)
    if [[ -n "$KIT_PID" ]]; then
        echo -e "  ✔ Isaac Sim Headless ya se encuentra activo en ${ISAAC_CONTAINER} (PID ${KIT_PID}). Reutilizando instancia."
    else
        # Matar instancias previas si existieran
        docker exec "${ISAAC_CONTAINER}" pkill -f "run_headless_stage.py" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -f "headless_autorun" 2>/dev/null || true
        docker exec "${ISAAC_CONTAINER}" pkill -9 -f "kit" 2>/dev/null || true
        sleep 1

        # Lanzar ejecutor headless en segundo plano
        docker exec -d "${ISAAC_CONTAINER}" bash -c "
            ulimit -n 524288 2>/dev/null || ulimit -n 65536 2>/dev/null || true
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
            export STAGE_NAME=\"${STAGE}\"
            export LD_LIBRARY_PATH=\"/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\${LD_LIBRARY_PATH:-}\"
            export OMNI_KIT_ACCEPT_EULA=YES
            export CAMERA_FRAME_SKIP=\"${CAMERA_FRAME_SKIP:-1}\"
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
        echo -e "  ✔ Isaac Sim Headless lanzado (Domain 42). Registro en /tmp/isaac_headless.log"
    fi
else
    # Múltiples instancias aisladas por ROS_DOMAIN_ID
    for ((k=1; k<=INSTANCES; k++)); do
        CURR_DOMAIN=$((42 + k - 1))
        CURR_PORT=$((8211 + k - 1))
        CURR_HTTP=$((8011 + k - 1))
        CURR_LOG="/tmp/isaac_headless_${k}.log"

        KIT_PID=$(docker exec "${ISAAC_CONTAINER}" pgrep -f "port=${CURR_PORT}" 2>/dev/null || true)
        if [[ -n "$KIT_PID" ]]; then
            echo -e "  ✔ Instancia #${k} ya activa en Isaac Sim (PID ${KIT_PID}, Puerto ${CURR_PORT}, Domain ${CURR_DOMAIN}). Reutilizando instancia."
        else
            echo -e "  Lanzando Instancia #${k} en Domain ${CURR_DOMAIN} (Puerto ${CURR_PORT})..."
            docker exec -d "${ISAAC_CONTAINER}" bash -c "
                ulimit -n 524288 2>/dev/null || ulimit -n 65536 2>/dev/null || true
                cd /isaac-sim 2>/dev/null || true
                if [ -f /isaac-sim/setup_ros_env.sh ]; then
                    source /isaac-sim/setup_ros_env.sh
                elif [ -f ./setup_ros_env.sh ]; then
                    source ./setup_ros_env.sh
                fi
                export ROS_DISTRO=jazzy
                export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                export ROS_LOCALHOST_ONLY=1
                    export ROS_DOMAIN_ID=${CURR_DOMAIN}
                export STAGE_NAME=\"${STAGE}\"
                export LD_LIBRARY_PATH=\"/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\${LD_LIBRARY_PATH:-}\"
                export OMNI_KIT_ACCEPT_EULA=YES
                export CAMERA_FRAME_SKIP=\"${CAMERA_FRAME_SKIP:-1}\"
                /isaac-sim/isaac-sim.streaming.sh --no-window \
                    --/app/livestream/port=$((8211 + k - 1)) \
                    --/exts/omni.services.transport.server.http/port=$((8011 + k - 1)) \
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
                    --exec /workspace/robo_imitate/xarm_bringup/isaac/headless_autorun.py > ${CURR_LOG} 2>&1
            "
            echo -e "  ✔ Instancia #${k} lanzada (Domain ${CURR_DOMAIN}). Registro en ${CURR_LOG}"
        fi
    done
fi

# ------------------------------------------------------------------------------
# 4. Sondeo DDS: Esperar tópicos /rgb e /isaac/joint_states
# ------------------------------------------------------------------------------
echo -e "${CYAN}[3/5] Sondeando bus DDS para tópicos /rgb e /isaac/joint_states (${INSTANCES} instancia(s))...${NC}"

MAX_WAIT=300

for ((k=1; k<=INSTANCES; k++)); do
    CURR_DOMAIN=$((42 + k - 1))
    CURR_LOG="/tmp/isaac_headless_${k}.log"
    if [ "$INSTANCES" -eq 1 ]; then
        CURR_LOG="/tmp/isaac_headless.log"
    fi

    ELAPSED=0
    TOPICS_FOUND=false

    echo -e "  Sondeando Instancia #${k} en ROS_DOMAIN_ID=${CURR_DOMAIN}..."

    while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
        TOPIC_LIST=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash 2>/dev/null || true
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=${CURR_DOMAIN}
            ros2 topic list --no-daemon 2>/dev/null || ros2 topic list 2>/dev/null
        " 2>/dev/null || true)

        HAS_RGB=$(echo "$TOPIC_LIST" | grep -E '^/rgb$' || true)
        HAS_JS=$(echo "$TOPIC_LIST" | grep -E '^/isaac/joint_states$' || true)
        HAS_CLOCK=$(echo "$TOPIC_LIST" | grep -E '^/clock$' || true)

        if [[ -n "$HAS_RGB" && -n "$HAS_JS" && -n "$HAS_CLOCK" ]]; then
            TOPICS_FOUND=true
            break
        fi

        echo -ne "  ⏳ Esperando tópicos en Domain ${CURR_DOMAIN}... (${ELAPSED}/${MAX_WAIT}s)\r"
        sleep 3
        ELAPSED=$((ELAPSED + 3))
    done
    echo ""

    if [ "$TOPICS_FOUND" = false ]; then
        echo -e "${RED}[ERROR FATAL] Timeout esperando tópicos de Isaac Sim (Instancia #${k}, Domain ${CURR_DOMAIN}) tras ${MAX_WAIT} segundos.${NC}"
        echo -e "${YELLOW}Últimas líneas del registro en ${ISAAC_CONTAINER}:${CURR_LOG} :${NC}"
        docker exec "${ISAAC_CONTAINER}" tail -n 25 "${CURR_LOG}" 2>/dev/null || true
        exit 1
    fi

    echo -e "  ✔ Tópicos sensoriales confirmados en CycloneDDS (Instancia #${k}, Domain ${CURR_DOMAIN}):"
    echo -e "    - /rgb: ${GREEN}ACTIVO${NC}"
    echo -e "    - /isaac/joint_states: ${GREEN}ACTIVO${NC}"
    echo -e "    - /clock: ${GREEN}ACTIVO${NC}"
done

# ------------------------------------------------------------------------------
# 5. Lanzamiento en Segundo Plano de lite6_cartesian.launch.py
# ------------------------------------------------------------------------------
echo -e "${CYAN}[4/5] Lanzando pila cinemática cartesiana en ${ROS_CONTAINER} (${INSTANCES} instancia(s))...${NC}"

# Limpiar exhaustivamente instancias previas de controladores y nodos en ROS_CONTAINER
docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
    pkill -9 -f 'ros2_control_node' 2>/dev/null || true
    pkill -9 -f 'pose_smoother' 2>/dev/null || true
    pkill -9 -f 'spawner' 2>/dev/null || true
    pkill -9 -f 'pickup_approximate' 2>/dev/null || true
    pkill -9 -f 'lite6_cartesian' 2>/dev/null || true
    rm -f /tmp/pickup_inst_*.done /tmp/*.pid 2>/dev/null || true
" 2>/dev/null || true
sleep 1

for ((k=1; k<=INSTANCES; k++)); do
    CURR_DOMAIN=$((42 + k - 1))
    CARTESIAN_LOG="/tmp/lite6_cartesian_${k}.log"
    if [ "$INSTANCES" -eq 1 ]; then
        CARTESIAN_LOG="/tmp/lite6_cartesian.log"
    fi

    docker exec -d ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
        # TAG_CART_${k}
        echo \$\$ > /tmp/lite6_cartesian_${k}.pid
        source /opt/ros/jazzy/setup.bash
        if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
        source /ros/install/setup.bash
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export ROS_LOCALHOST_ONLY=1
        export ROS_DOMAIN_ID=${CURR_DOMAIN}
        ros2 launch visuomotor_bringup lite6_cartesian.launch.py rviz:=false sim:=true use_sim_time:=true > ${CARTESIAN_LOG} 2>&1
    "
    echo -e "  ✔ Pila cinemática cartesiana lanzada para Instancia #${k} (Domain ${CURR_DOMAIN}, Log: ${CARTESIAN_LOG})"
done

echo -e "  Esperando estabilización de controladores cartesianos (/joint_states en cada instancia)..."
for ((k=1; k<=INSTANCES; k++)); do
    CURR_DOMAIN=$((42 + k - 1))
    cw=0
    while [ "$cw" -lt 30 ]; do
        sleep 2
        cw=$((cw + 2))
        if docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash 2>/dev/null
            export ROS_DOMAIN_ID=${CURR_DOMAIN}
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            ros2 topic list 2>/dev/null | grep -E '^/joint_states$' >/dev/null
        " 2>/dev/null; then
            echo -e "  ✔ Instancia #${k} controladores activos (Domain ${CURR_DOMAIN}, ${cw}s)."
            break
        fi
    done
done
echo -e "  ✔ Pila(s) cinemática(s) cartesiana(s) activa(s) (rviz=false, sim=true)."

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
        if [[ -n "${MAX_STEPS_ARG:-}" ]]; then
            EXTRA_ARGS+=("--max_steps" "${MAX_STEPS_ARG}")
        fi
        ;;
    eval-checkpoints)
        if [[ -n "${EPISODES_ARG:-}" ]]; then
            EXTRA_ARGS+=("--episodes" "${EPISODES_ARG}")
        fi
        if [[ -n "${MAX_STEPS_ARG:-}" ]]; then
            EXTRA_ARGS+=("--max_steps" "${MAX_STEPS_ARG}")
        fi
        if [[ -n "${CHECKPOINT_OVERRIDE:-}" ]]; then
            EXTRA_ARGS+=("--run_dir" "${CHECKPOINT_OVERRIDE}")
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
        echo -e "${GREEN}Simulación Headless y Control Cartesiano operativos (${INSTANCES} instancia(s)).${NC}"
        echo -e "Presiona ${BOLD}Ctrl+C${NC} para detener la sesión y liberar los recursos."
        if [ "$INSTANCES" -eq 1 ]; then
            echo -e "Monitoreando telemetría de Isaac Sim (/tmp/isaac_headless.log):"
            docker exec ${DOCKER_IT} "${ISAAC_CONTAINER}" tail -f /tmp/isaac_headless.log
        else
            echo -e "Monitoreando telemetría de Isaac Sim Instancia 1 (/tmp/isaac_headless_1.log):"
            docker exec ${DOCKER_IT} "${ISAAC_CONTAINER}" tail -f /tmp/isaac_headless_1.log
        fi
        ;;

    collect)
        if [ "$INSTANCES" -eq 1 ]; then
            echo -e "${BOLD}Iniciando Recolección Autónoma de Demostraciones Experto (Instancia Única)...${NC}"
            docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
                ulimit -n 524288 2>/dev/null || true
                source /opt/ros/jazzy/setup.bash
                if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
                source /ros/install/setup.bash
                export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                export ROS_LOCALHOST_ONLY=1
                export ROS_DOMAIN_ID=42
                ros2 launch visuomotor_data_collection pickup_approximate.launch.py use_sim_time:=true ${EXTRA_ARGS[*]:-}
            "
        else
            echo -e "${BOLD}Iniciando Recolección Autónoma Paralela (${INSTANCES} instancias con Batches Temporales)...${NC}"

            # 1. Determinar Dataset Canónico Destino
            if [[ -n "${DATASET_ARG:-}" ]]; then
                TARGET_DATASET="${DATASET_ARG}"
            else
                TARGET_DATASET=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" python3 -c "
import glob, os, re
base_dir = '/ros/datasets'
existing = glob.glob(os.path.join(base_dir, 'demonstrations_v*.zarr'))
max_v = 1
for path in existing:
    m = re.search(r'demonstrations_v(\d+)\.zarr', path)
    if m:
        max_v = max(max_v, int(m.group(1)))
print(f'/ros/datasets/demonstrations_v{max_v + 1}.zarr')
")
            fi

            # Normalizar TARGET_DATASET para que siempre sea una ruta canónica absoluta dentro de /ros
            if [[ "$TARGET_DATASET" == /ros/* ]]; then
                TARGET_DATASET="${TARGET_DATASET}"
            elif [[ "$TARGET_DATASET" == /* ]]; then
                REPO_DIR_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
                if [[ "$TARGET_DATASET" == "$REPO_DIR_HOST"/* ]]; then
                    TARGET_DATASET="/ros/${TARGET_DATASET#$REPO_DIR_HOST/}"
                else
                    TARGET_DATASET="/ros/${TARGET_DATASET#/}"
                fi
            else
                CLEAN_REL="${TARGET_DATASET#./}"
                TARGET_DATASET="/ros/${CLEAN_REL}"
            fi
            echo -e "  Dataset canónico destino: ${BOLD}${GREEN}${TARGET_DATASET}${NC}"

            # 2. Asignar Shards / Batches Temporales
            BASE_NAME="${TARGET_DATASET%.zarr}"
            BATCH_DATASETS=()
            for ((k=1; k<=INSTANCES; k++)); do
                BATCH_DATASETS+=("${BASE_NAME}_batch_${k}.zarr")
            done

            # 3. Particionar meta de episodios equitativamente
            TOTAL_EPS=0
            if [[ -n "${EPISODES_ARG:-}" && "${EPISODES_ARG}" -gt 0 ]] 2>/dev/null; then
                TOTAL_EPS="${EPISODES_ARG}"
            fi

            INST_EPS_LIST=()
            if [ "$TOTAL_EPS" -gt 0 ]; then
                BASE_EPS=$((TOTAL_EPS / INSTANCES))
                REM_EPS=$((TOTAL_EPS % INSTANCES))
                for ((k=1; k<=INSTANCES; k++)); do
                    if [ "$k" -le "$REM_EPS" ]; then
                        INST_EPS_LIST+=($((BASE_EPS + 1)))
                    else
                        INST_EPS_LIST+=(${BASE_EPS})
                    fi
                done
            else
                for ((k=1; k<=INSTANCES; k++)); do
                    INST_EPS_LIST+=(0)
                done
            fi

            echo -e "  Plan de recolección paralelo:"
            for ((k=1; k<=INSTANCES; k++)); do
                CURR_DOMAIN=$((42 + k - 1))
                CURR_BATCH="${BATCH_DATASETS[$((k - 1))]}"
                CURR_PLAN_EPS="${INST_EPS_LIST[$((k - 1))]}"
                echo -e "    - Instancia #${k}: Domain=${CURR_DOMAIN} | Meta=${CURR_PLAN_EPS} eps | Shard=${CURR_BATCH}"
            done

            # 4. Lanzamiento concurrente de colectores en segundo plano
            for ((k=1; k<=INSTANCES; k++)); do
                CURR_DOMAIN=$((42 + k - 1))
                CURR_BATCH="${BATCH_DATASETS[$((k - 1))]}"
                CURR_PLAN_EPS="${INST_EPS_LIST[$((k - 1))]}"
                SEED=$((1000 * k + 42))
                LOG_PATH="/tmp/pickup_approximate_${k}.log"
                DONE_MARKER="/tmp/pickup_inst_${k}.done"

                # Limpiar marcador de término previo
                docker exec ${USER_FLAGS} "${ROS_CONTAINER}" rm -f "${DONE_MARKER}" 2>/dev/null || true

                docker exec -d ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
                    # TAG_INST_${k}
                    echo \$\$ > /tmp/pickup_inst_${k}.pid
                    rm -f ${DONE_MARKER}
                    ulimit -n 524288 2>/dev/null || true
                    source /opt/ros/jazzy/setup.bash
                    if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
                    source /ros/install/setup.bash
                    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                    export ROS_LOCALHOST_ONLY=1
                    export ROS_DOMAIN_ID=${CURR_DOMAIN}
                    ros2 launch visuomotor_data_collection pickup_approximate.launch.py \
                        dataset_path:=\"${CURR_BATCH}\" \
                        max_episodes:=\"${CURR_PLAN_EPS}\" \
                        auto_start:=true \
                        seed:=\"${SEED}\" \
                        use_sim_time:=true > \"${LOG_PATH}\" 2>&1
                    echo \$? > ${DONE_MARKER}
                "
                echo -e "  ✔ Colector #${k} activo (PID en segundo plano, Domain ${CURR_DOMAIN}, Log: ${LOG_PATH})"
            done

            # 4.5. Función Watchdog Auto-Heal para recuperación en caliente de instancias individuales
            recover_instance() {
                local inst_idx=$1
                local c_dom=$((42 + inst_idx - 1))
                local c_port=$((8210 + inst_idx))
                local c_http=$((8010 + inst_idx))
                local c_batch="${BATCH_DATASETS[$((inst_idx - 1))]}"
                local c_plan_eps="${INST_EPS_LIST[$((inst_idx - 1))]}"
                local c_seed=$((1000 * inst_idx + 42 + RANDOM % 1000))
                local c_log="/tmp/pickup_approximate_${inst_idx}.log"
                local c_done="/tmp/pickup_inst_${inst_idx}.done"
                local c_kit_log="/tmp/isaac_headless_${inst_idx}.log"
                local c_cart_log="/tmp/lite6_cartesian_${inst_idx}.log"

                echo -e "\n${BOLD}${RED}══════════════════════════════════════════════════════════════════════${NC}"
                echo -e "${BOLD}${YELLOW}⚠️  [WATCHDOG AUTO-HEAL] Reiniciando en caliente Instancia #${inst_idx} (Domain ${c_dom}, Puerto ${c_port})...${NC}"
                echo -e "${BOLD}${RED}══════════════════════════════════════════════════════════════════════${NC}"

                # 1. Matar exhaustivamente todos los nodos y procesos de esa instancia por ROS_DOMAIN_ID
                docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
                    for pid in \$(pgrep -u 1000); do
                        if [ -f \"/proc/\$pid/environ\" ] && grep -qz \"^ROS_DOMAIN_ID=${c_dom}\$\" \"/proc/\$pid/environ\" 2>/dev/null; then
                            kill -9 \"\$pid\" 2>/dev/null || true
                        fi
                    done
                    rm -f '${c_done}' '/tmp/lite6_cartesian_${inst_idx}.pid' '/tmp/pickup_inst_${inst_idx}.pid'
                " 2>/dev/null || true

                # 3. Matar el proceso Kit de Isaac Sim en el puerto de esa instancia
                docker exec "${ISAAC_CONTAINER}" bash -c "
                    pkill -9 -f 'port=${c_port}' 2>/dev/null || true
                " 2>/dev/null || true
                sleep 2

                # 4. Relanzar proceso Kit de Isaac Sim para esa instancia
                echo -e "  [Watchdog] Relanzando Isaac Sim Kit (Puerto ${c_port}, Domain ${c_dom})..."
                docker exec -d "${ISAAC_CONTAINER}" bash -c "
                    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                    export ROS_LOCALHOST_ONLY=1
                    export ROS_DOMAIN_ID=${c_dom}
                    export STAGE_NAME=\"${STAGE}\"
                    export LD_LIBRARY_PATH=\"/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:\${LD_LIBRARY_PATH:-}\"
                    export OMNI_KIT_ACCEPT_EULA=YES
                    export CAMERA_FRAME_SKIP=\"${CAMERA_FRAME_SKIP:-1}\"
                    /isaac-sim/isaac-sim.streaming.sh --no-window \
                        --/app/livestream/port=${c_port} \
                        --/exts/omni.services.transport.server.http/port=${c_http} \
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
                        --exec /workspace/robo_imitate/xarm_bringup/isaac/headless_autorun.py > \"${c_kit_log}\" 2>&1
                "

                # 5. Esperar confirmación DDS de los tópicos
                echo -e "  [Watchdog] Esperando tópicos DDS en Domain ${c_dom}..."
                local w=0
                while [ "$w" -lt 180 ]; do
                    sleep 3
                    w=$((w + 3))
                    local top
                    top=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
                        source /opt/ros/jazzy/setup.bash 2>/dev/null || true
                        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                        export ROS_LOCALHOST_ONLY=1
                        export ROS_DOMAIN_ID=${c_dom}
                        ros2 topic list 2>/dev/null
                    " 2>/dev/null || true)
                    if echo "$top" | grep -E '^/rgb$' >/dev/null && echo "$top" | grep -E '^/isaac/joint_states$' >/dev/null; then
                        echo -e "  [Watchdog] ✔ Tópicos DDS reanudados en Domain ${c_dom} (${w}s)."
                        break
                    fi
                done

                # 6. Relanzar controladores cartesianos en c_dom
                echo -e "  [Watchdog] Relanzando controladores cartesianos en Domain ${c_dom}..."
                docker exec -d ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
                    # TAG_CART_${inst_idx}
                    echo \$\$ > /tmp/lite6_cartesian_${inst_idx}.pid
                    source /opt/ros/jazzy/setup.bash
                    if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
                    source /ros/install/setup.bash
                    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                    export ROS_LOCALHOST_ONLY=1
                    export ROS_DOMAIN_ID=${c_dom}
                    ros2 launch visuomotor_bringup lite6_cartesian.launch.py \
                        rviz:=false sim:=true use_sim_time:=true > \"${c_cart_log}\" 2>&1
                "
                # Esperar activamente confirmación de /joint_states antes de lanzar el colector
                echo -e "  [Watchdog] Esperando controladores cartesianos (/joint_states) en Domain ${c_dom}..."
                local cw=0
                while [ "$cw" -lt 30 ]; do
                    sleep 2
                    cw=$((cw + 2))
                    if docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
                        source /opt/ros/jazzy/setup.bash 2>/dev/null
                        export ROS_DOMAIN_ID=${c_dom}
                        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                        export ROS_LOCALHOST_ONLY=1
                        ros2 topic list 2>/dev/null | grep -E '^/joint_states$' >/dev/null
                    " 2>/dev/null; then
                        echo -e "  [Watchdog] ✔ Controladores cartesianos activos en Domain ${c_dom} (${cw}s)."
                        break
                    fi
                done

                # 7. Relanzar colector pickup_approximate
                echo -e "  [Watchdog] Relanzando colector pickup_approximate en Domain ${c_dom}..."
                docker exec -d ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
                    # TAG_INST_${inst_idx}
                    echo \$\$ > /tmp/pickup_inst_${inst_idx}.pid
                    rm -f \"${c_done}\"
                    ulimit -n 524288 2>/dev/null || true
                    source /opt/ros/jazzy/setup.bash
                    if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
                    source /ros/install/setup.bash
                    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                    export ROS_LOCALHOST_ONLY=1
                    export ROS_DOMAIN_ID=${c_dom}
                    ros2 launch visuomotor_data_collection pickup_approximate.launch.py \
                        dataset_path:=\"${c_batch}\" \
                        max_episodes:=\"${c_plan_eps}\" \
                        auto_start:=true \
                        seed:=\"${c_seed}\" \
                        use_sim_time:=true > \"${c_log}\" 2>&1
                    echo \$? > \"${c_done}\"
                "
                echo -e "${BOLD}${GREEN}  ✔ [Watchdog] Instancia #${inst_idx} reanudada exitosamente sin afectar al resto.${NC}\n"
            }

            echo -e "\n${BOLD}${CYAN}Monitoreando progreso concurrente con Watchdog Auto-Heal (Ctrl+C para finalizar y aplanar):${NC}"

            # 5. Bucle de supervisión y telemetría de episodios con Watchdog
            ALL_FINISHED=false
            FAILED_COUNT=0
            WATCHDOG_TIMEOUT=140
            LOOP_TICK=0

            LAST_COUNT=()
            LAST_PROGRESS_TIME=()
            NOW_INIT=$(date +%s)
            for ((k=1; k<=INSTANCES; k++)); do
                LAST_COUNT+=(0)
                LAST_PROGRESS_TIME+=("${NOW_INIT}")
            done

            while [ "$ALL_FINISHED" = false ]; do
                sleep 2
                LOOP_TICK=$((LOOP_TICK + 1))
                STATUS_PARTS=()
                STILL_RUNNING=0

                for ((k=1; k<=INSTANCES; k++)); do
                    CURR_BATCH="${BATCH_DATASETS[$((k - 1))]}"
                    DONE_MARKER="/tmp/pickup_inst_${k}.done"
                    CURR_PLAN_EPS="${INST_EPS_LIST[$((k - 1))]}"
                    CURR_DOMAIN=$((42 + k - 1))

                    COUNT=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" python3 -c "
import zarr, os
p = '${CURR_BATCH}'
if os.path.exists(p):
    try:
        z = zarr.open_group(p, mode='r')
        print(len([k for k in z.group_keys() if k.startswith('episode_')]))
    except Exception:
        print(0)
else:
    print(0)
" 2>/dev/null || echo "0")

                    STATUS_CODE=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "if [ -f '${DONE_MARKER}' ]; then cat '${DONE_MARKER}'; else echo 'RUNNING'; fi" 2>/dev/null || echo "RUNNING")
                    STATUS_CODE=$(echo "$STATUS_CODE" | tr -d '[:space:]')

                    NOW_SEC=$(date +%s)
                    COUNT_INT=${COUNT:-0}
                    PREV_COUNT=${LAST_COUNT[$((k - 1))]:-0}

                    if [ "$COUNT_INT" -gt "$PREV_COUNT" ]; then
                        LAST_COUNT[$((k - 1))]="$COUNT_INT"
                        LAST_PROGRESS_TIME[$((k - 1))]="$NOW_SEC"
                    fi

                    TIME_NO_PROGRESS=$(( NOW_SEC - LAST_PROGRESS_TIME[$((k - 1))] ))

                    # Chequeo Watchdog 1: Estancamiento por inactividad (>140s)
                    STUCK=false
                    if [ "$STATUS_CODE" = "RUNNING" ] && [ "$COUNT_INT" -lt "$CURR_PLAN_EPS" ] && [ "$TIME_NO_PROGRESS" -gt "$WATCHDOG_TIMEOUT" ]; then
                        echo -e "\n${BOLD}${RED}[WATCHDOG TRIGGER] Instancia #${k} estancada durante ${TIME_NO_PROGRESS}s sin nuevos episodios (Umbral: ${WATCHDOG_TIMEOUT}s).${NC}"
                        STUCK=true
                    fi

                    # Chequeo Watchdog 2: Detección activa de divergencia articular PhysX cada 10s (5 ticks)
                    if [ "$STATUS_CODE" = "RUNNING" ] && [ "$COUNT_INT" -lt "$CURR_PLAN_EPS" ] && [ "$((LOOP_TICK % 5))" -eq 0 ] && [ "$STUCK" = false ]; then
                        DIVERGED=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "
                            source /opt/ros/jazzy/setup.bash 2>/dev/null || true
                            export ROS_DOMAIN_ID=${CURR_DOMAIN}
                            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
                            export ROS_LOCALHOST_ONLY=1
                            python3 -c \"
import rclpy, sys
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init()
n = Node('div_chk')
msg = None
def cb(m): global msg; msg = m
sub = n.create_subscription(JointState, '/joint_states', cb, 1)
import time; t0 = time.time()
while time.time() - t0 < 0.6 and msg is None:
    rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node()
rclpy.shutdown()
if msg:
    max_p = max([abs(p) for p in msg.position]) if msg.position else 0.0
    max_e = max([abs(e) for e in msg.effort if e == e]) if msg.effort else 0.0
    if max_p > 35.0 or max_e > 120.0:
        print('DIVERGED')
        sys.exit(0)
print('OK')
\" 2>/dev/null || echo 'OK'
                        " 2>/dev/null || echo "OK")
                        if [[ "$DIVERGED" == *"DIVERGED"* ]]; then
                            echo -e "\n${BOLD}${RED}[WATCHDOG TRIGGER] Divergencia articular PhysX detectada en Instancia #${k} (esfuerzo >120Nm o ángulo >35rad).${NC}"
                            STUCK=true
                        fi
                    fi

                    # Chequeo Watchdog 0: Salida prematura antes de completar la meta
                    if [ "$COUNT_INT" -lt "$CURR_PLAN_EPS" ] && [ "$STATUS_CODE" != "RUNNING" ]; then
                        echo -e "\n${BOLD}${RED}[WATCHDOG TRIGGER] Instancia #${k} finalizó prematuramente con ${COUNT_INT}/${CURR_PLAN_EPS} eps (Código: ${STATUS_CODE}). Recuperando...${NC}"
                        STUCK=true
                    fi

                    if [ "$STUCK" = true ]; then
                        recover_instance "$k"
                        LAST_PROGRESS_TIME[$((k - 1))]=$(date +%s)
                        STATUS_CODE="RUNNING"
                    fi

                    if [ "$COUNT_INT" -ge "$CURR_PLAN_EPS" ]; then
                        STATUS_PARTS+=("Instancia #${k}: ${COUNT}/${CURR_PLAN_EPS} [FINALIZADO]")
                    elif [ "$STATUS_CODE" = "RUNNING" ]; then
                        STATUS_PARTS+=("Instancia #${k}: ${COUNT}/${CURR_PLAN_EPS} [EN CURSO]")
                        STILL_RUNNING=$((STILL_RUNNING + 1))
                    else
                        STATUS_PARTS+=("Instancia #${k}: ${COUNT}/${CURR_PLAN_EPS} [REINTENTANDO]")
                        STILL_RUNNING=$((STILL_RUNNING + 1))
                    fi
                done

                echo -ne "  ⏳ $(IFS=' | '; echo "${STATUS_PARTS[*]}")\r"

                if [ "$STILL_RUNNING" -eq 0 ]; then
                    ALL_FINISHED=true
                    echo ""
                    break
                fi
            done

            # Verificar si alguna instancia finalizó con fallo
            for ((k=1; k<=INSTANCES; k++)); do
                DONE_MARKER="/tmp/pickup_inst_${k}.done"
                LOG_PATH="/tmp/pickup_approximate_${k}.log"
                STATUS_CODE=$(docker exec ${USER_FLAGS} "${ROS_CONTAINER}" bash -c "if [ -f '${DONE_MARKER}' ]; then cat '${DONE_MARKER}'; else echo 'UNKNOWN'; fi" 2>/dev/null || echo "UNKNOWN")
                STATUS_CODE=$(echo "$STATUS_CODE" | tr -d '[:space:]')
                if [[ "$STATUS_CODE" != "0" ]]; then
                    echo -e "\n${RED}[ALERTA] Instancia #${k} finalizó con código no-cero (${STATUS_CODE}). Últimas líneas de ${LOG_PATH}:${NC}"
                    docker exec ${USER_FLAGS} "${ROS_CONTAINER}" tail -n 25 "${LOG_PATH}" 2>/dev/null || true
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi
            done

            if [ "$FAILED_COUNT" -eq 0 ]; then
                echo -e "\n${BOLD}${GREEN}✔ Todas las instancias de recolección han finalizado sus tareas exitosamente.${NC}"
            else
                echo -e "\n${BOLD}${YELLOW}[AVISO] ${FAILED_COUNT} instancia(s) reportaron advertencias/fallos. Consolidando episodios válidos...${NC}"
            fi

            # 6. Cierre y Aplanado Automático de Lotes Zarr
            echo -e "\n${BOLD}${BLUE}======================================================================"
            echo -e "      Aplanando Lotes Temporales -> Dataset Canónico                  "
            echo -e "======================================================================${NC}"
            echo -e "Ejecutando merge_zarr_batches.py hacia: ${TARGET_DATASET}..."

            docker exec ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" python3 /ros/tools/merge_zarr_batches.py \
                --inputs "${BATCH_DATASETS[@]}" \
                --output "${TARGET_DATASET}" \
                --remove-sources

            MERGE_EXIT=$?
            if [ "$MERGE_EXIT" -eq 0 ]; then
                echo -e "\n${BOLD}${GREEN}✔ Consolidación y aplanado completado con éxito.${NC}"
                echo -e "Dataset listo para entrenamiento: ${BOLD}${GREEN}${TARGET_DATASET}${NC}"
                BATCH_DATASETS=()
            else
                echo -e "\n${BOLD}${RED}[ERROR] Error durante la fusión de lotes (código ${MERGE_EXIT}). Lotes temporales preservados.${NC}"
                exit 1
            fi
        fi
        ;;

    rl)
        if [ "$INSTANCES" -gt 1 ]; then
            echo -e "${YELLOW}[AVISO] Modo 'rl' opera sobre la instancia principal (Domain 42).${NC}"
        fi
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
        if [ "$INSTANCES" -gt 1 ]; then
            echo -e "${YELLOW}[AVISO] Modo 'benchmark' opera sobre la instancia principal (Domain 42).${NC}"
        fi
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

    eval-checkpoints)
        if [ "$INSTANCES" -gt 1 ]; then
            echo -e "${YELLOW}[AVISO] Modo 'eval-checkpoints' opera sobre la instancia principal (Domain 42).${NC}"
        fi
        echo -e "${BOLD}Iniciando Evaluación Multi-Checkpoint de Convergencia Físico-Sensorial...${NC}"
        docker exec ${DOCKER_IT} ${USER_FLAGS} -w /ros "${ROS_CONTAINER}" bash -c "
            source /opt/ros/jazzy/setup.bash
            if [ -f /tmp/xarm_ws/install/setup.bash ]; then source /tmp/xarm_ws/install/setup.bash; fi
            source /ros/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export ROS_LOCALHOST_ONLY=1
            export ROS_DOMAIN_ID=42
            python3 tools/evaluate_convergence_checkpoints.py ${EXTRA_ARGS[*]:-}
        "
        ;;

    inference)
        if [ "$INSTANCES" -gt 1 ]; then
            echo -e "${YELLOW}[AVISO] Modo 'inference' opera sobre la instancia principal (Domain 42).${NC}"
        fi
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
