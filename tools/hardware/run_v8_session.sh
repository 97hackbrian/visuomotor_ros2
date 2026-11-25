#!/usr/bin/env bash
# Sesión de recolección del dataset v8: demostraciones A MANO con la D435i. Se puede repetir cada día: continúa el
# mismo dataset (modo AUMENTO) y el mismo programa (salta las entradas ya hechas).
#
#   Terminal 1: ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false
#   Terminal 2: ./tools/hardware/run_v8_session.sh                     # [dataset, por defecto demonstrations_real_v8]
#   Terminal 3 (guía): docker exec -it -u 1000 visuomotor_vo_active bash -c \
#     "source /opt/ros/jazzy/setup.bash && ros2 run rqt_image_view rqt_image_view /vo_collector_node/start_guide"
#
# Qué graba cada demo:
# * el episodio Zarr: imagen de 256 px, pose de la odometría en vivo (solo guía y control de calidad), sellos y atributos;
# * el bag crudo MCAP desde ARMING (recolector v3.1): color JPEG 640x480, profundidad alineada en PNG de 16 bits, IMU,
#   camera_info, odometría y tf_static. De ahí salen después las poses del dataset (DROID-SLAM RGB-D) y la pose real
#   de inicio (ArUco + IMU). SIN la profundidad, DROID no tiene escala métrica: es imprescindible.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
DATASET=${1:-demonstrations_real_v8}
SCHED=experiments/collection_schedules/v8_schedule.csv
CAM=visuomotor_vo_active
fail() { echo "NO LISTO: $*"; exit 1; }
docker ps --format '{{.Names}}' | grep -q "^$CAM$" || fail "la cámara no está corriendo. Terminal 1:
  ./tools/hardware/run_vo_container.sh ros2 launch visuomotor_bringup realsense_d435i_vio.launch.py gui:=false"
for f in "$SCHED" visuomotor_bringup/config/realsense/home3_view_reference.yaml \
         visuomotor_bringup/config/realsense/v8_cells_home3.yaml visuomotor_bringup/config/realsense/table_markers.yaml; do
  [ -f "$f" ] || fail "falta $f"
done
FREE_GB=$(df -BG --output=avail datasets | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 30 ] || fail "solo ${FREE_GB} GB libres (cada demo ocupa ~0,1-0,2 GB en el bag crudo)"
cmp -s visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py \
  install/visuomotor_data_collection/lib/python3.12/site-packages/visuomotor_data_collection/vo_collector_node.py \
  || fail "install/ desactualizado: docker exec -u 1000 -w /ros visuomotor_ros2 bash -c 'source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --packages-select visuomotor_data_collection'"
cat <<'EOF'
==========================================================================================
 SESIÓN v8 (demos a mano). Antes de empezar:
  [ ] Tablero ChArUco FUERA de la mesa (solo se usa para medir).
  [ ] Mantel y los dos ArUco (12 y 14) en su lugar, sin moverlos (calibrados con el robot).
  [ ] Base del robot (Husky) estacionada donde estaba, como en v7 y en la inferencia.
  [ ] Luz fija encendida.
 Cada demo:
  1. Poner la herramienta en la celda AMARILLA de la guía; la flecha = hacia dónde apuntan las HOJAS.
     (Celdas 7 y 9: la herramienta hacia el centro de la celda. Nunca tapar los ArUco.)
  2. Llevar la cámara a la vista de HOME3 hasta "ALINEADA"; quieta 1 s -> empieza sola.
  3. "QUIETA: reiniciando odometría": NO moverse (estos frames dan la pose real de inicio).
  4. Pitido "Grabando": bajar suave (<= 15 cm/s) hacia la herramienta.
  5. Terminar a 25-30 cm SOBRE la herramienta, centrada en el recuadro, y quedarse quieta ~2,5 s -> guarda sola.
     Respaldo: si no guarda en ~5 s, UNA tecla [S] (detecta sola el tramo quieto; no hace falta [H]).
  6. Si el control de calidad la rechaza: [D] y repetir. [N] salta la entrada del programa.
     (Si la odometría en vivo se pierde cerca de la mesa: es un AVISO, la demo se guarda igual; el final quieto
      lo detecta el giróscopo. Las poses del dataset salen después de DROID desde el bag crudo.)
==========================================================================================
EOF
COLLECTOR_ARGS="-p auto_start:=true -p auto_save:=true -p schedule_file:=$SCHED -p raw_bag_enabled:=true \
-p raw_bag_preroll_arming:=true -p raw_bag_dir:=datasets/raw_bags -p qa_tracking_blocking:=false -p auto_hold:=true" \
  exec ./tools/hardware/run_data_collection.sh "$DATASET"
