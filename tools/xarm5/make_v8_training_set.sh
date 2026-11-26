#!/usr/bin/env bash
# Del dataset crudo v8 (el del recolector) al dataset de ENTRENAMIENTO del xArm5, de punta a punta y reanudable.
# Para correr después de cada sesión de recolección: las demos nuevas se suman y nada se sobrescribe.
#   1. Valida con DROID-SLAM las demos que aún no tienen trayectoria (validate_v8_batch.sh; GPU, reanudable).
#   2. Pose real de inicio de todas (estimate_start_poses.py: ArUco + IMU en ARMING).
#   3. Informe de validación (validate_v8_episodes.py report): OK, REVISAR o REPETIR.
#   4. Dataset con las poses de DROID (build_dataset_from_bags.py). Omite las descartadas y las REPETIR. Las REPETIR
#      no se marcan solas: marcarlas descartadas en el dataset crudo requiere el OK del usuario.
#   5. Conversión al xArm5 (convert_vio_dataset_to_xarm5.py, conversor v3) con la pose real de inicio. Conserva las
#      demos marcadas por alcance (decisión del 2026-11-25: entrenar con todas o solo con las factibles y comparar).
#   6. Verificación contra el dataset de referencia (verify_training_dataset.py, con un paso de entrenamiento en la GPU).
#
#   tools/xarm5/make_v8_training_set.sh <salida_droid> <salida_xarm5> [crudo=demonstrations_real_v8] \
#       [referencia=demonstrations_real_v7_xarm5_v2]
#   p. ej. (siguiente toma): ... demonstrations_real_v8_droid_v3 demonstrations_real_v8_xarm5_v3
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
DROID_OUT=${1:?nombre del dataset con poses de DROID (sin .zarr)}
XARM5_OUT=${2:?nombre del dataset de entrenamiento xArm5 (sin .zarr)}
RAW=${3:-demonstrations_real_v8}
REF=${4:-demonstrations_real_v7_xarm5_v2}
V=experiments/v8_validation/$RAW
for d in "$DROID_OUT" "$XARM5_OUT"; do
  [ ! -e "datasets/$d.zarr" ] || { echo "ERROR: datasets/$d.zarr ya existe (nunca se sobrescribe)"; exit 1; }
done
docker ps --format '{{.Names}}' | grep -qx visuomotor_ros2 || { echo "ERROR: el contenedor visuomotor_ros2 no corre"; exit 1; }
run() { docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && $*"; }

echo "== 1/6 DROID en las demos nuevas"
tools/xarm5/validate_v8_batch.sh 0 100000 "$RAW" | grep -E "DROID en|ERROR|TERMINADO"
echo "== 2/6 Poses reales de inicio"
run "python3 tools/xarm5/estimate_start_poses.py --zarr datasets/$RAW.zarr --out $V/start_poses.json" | tail -1
echo "== 3/6 Informe de validación"
run "python3 tools/xarm5/validate_v8_episodes.py report --dataset datasets/$RAW.zarr --start-poses $V/start_poses.json" \
  | grep -vE " OK $" | tail -30
echo "== 4/6 Dataset con poses de DROID -> datasets/$DROID_OUT.zarr"
run "python3 tools/xarm5/build_dataset_from_bags.py --src datasets/$RAW.zarr --dst datasets/$DROID_OUT.zarr" \
  > "$V/build_$DROID_OUT.log" 2>&1
tail -1 "$V/build_$DROID_OUT.log" | cut -c1-200
echo "== 5/6 Conversión al xArm5 -> datasets/$XARM5_OUT.zarr"
run "python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/$DROID_OUT.zarr --dst datasets/$XARM5_OUT.zarr \
  --start-poses $V/start_poses.json" | tail -2
echo "== 6/6 Verificación contra datasets/$REF.zarr"
run "python3 tools/xarm5/verify_training_dataset.py --dataset datasets/$XARM5_OUT.zarr --reference datasets/$REF.zarr \
  --forward --out $V/verify_$XARM5_OUT.json"
