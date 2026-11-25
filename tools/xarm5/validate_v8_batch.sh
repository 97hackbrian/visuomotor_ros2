#!/usr/bin/env bash
# Lote de validación offline de demos v8: por cada episodio no descartado, exporta su bag crudo (carpeta tipo TUM en
# datasets/vio_frames/<dataset>/<episodio>, la misma que usa estimate_start_poses.py) y corre DROID-SLAM RGB-D
# (visuomotor_learned_vo:v1, GPU, CPU limitada para no frenar una toma en curso). Reanudable: salta lo ya hecho.
# Después: python3 tools/xarm5/validate_v8_episodes.py report
#
#   tools/xarm5/validate_v8_batch.sh <primero> <último> [dataset=demonstrations_real_v8]
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
FIRST=${1:?primero}
LAST=${2:?último}
DS=${3:-demonstrations_real_v8}
OUT=experiments/v8_validation/$DS/droid
FR=datasets/vio_frames/$DS
CPUS=${V8_VALIDATE_CPUS:-6}
mkdir -p "$OUT" "$FR"
list=$(docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/validate_v8_episodes.py list \
       --dataset "datasets/$DS.zarr" --first "$FIRST" --last "$LAST") || { echo "ERROR: no se pudo listar"; exit 1; }
n_ok=0
while read -r key bag; do
  [ -n "$key" ] || continue
  if [ -s "$OUT/$key.tum" ]; then echo "$key: ya validada"; continue; fi
  if [ ! -f "$FR/$key/associations.txt" ]; then
    docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && nice -n 10 python3 \
      tools/xarm5/vio_gt/export_bag_frames.py --bag '$bag' --out '$FR/$key'" >/dev/null 2>&1 \
      || { echo "$key: ERROR al exportar el bag"; continue; }
  fi
  t0=$(date +%s)
  docker run --rm --gpus all --cpus "$CPUS" -u 1000 -e HOME=/tmp -v "$REPO":/ros -w /ros visuomotor_learned_vo:v1 \
    python3 tools/xarm5/vio_gt/frameworks/learned/run_droid.py "$FR/$key" "$OUT/$key.tum" --filter-thresh 1.0 \
    >"$OUT/$key.log" 2>&1 || { echo "$key: ERROR en DROID (ver $OUT/$key.log)"; continue; }
  n_ok=$((n_ok + 1))
  echo "$key: DROID en $(( $(date +%s) - t0 )) s"
done <<<"$list"
echo "LOTE TERMINADO: $n_ok episodios nuevos con DROID"
