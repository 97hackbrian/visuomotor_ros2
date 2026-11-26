#!/usr/bin/env bash
# Dataset de ENTRENAMIENTO v8 desde el CRUDO, de punta a punta. Correr después de cada sesión de recolección: rehace
# el dataset COMPLETO con las demos de todos los días (DROID corre solo en las nuevas) y, si pasa la verificación,
# reemplaza al anterior. Así todas las demos pasan por el mismo código.
#
# Organización de los datasets (AGENTS.md §2.3, decisión del usuario del 2026-11-26):
#   - CRUDO: datasets/raw_bags/<crudo>/ + datasets/<crudo>.zarr. Es la fuente y solo crece; este script no lo toca.
#   - ENTRENAMIENTO: datasets/<crudo>_xarm5.zarr, con su informe de conversión. Es derivado y tiene nombre fijo: se rehace
#     aquí y solo se reemplaza si pasa la verificación. Las poses salen SIEMPRE de DROID-SLAM. El armado se detiene si
#     a una demo válida le falta la trayectoria, y la verificación exige las poses de DROID en cada episodio.
#   - FUNCIONAL: para entrenar se congela una copia con fecha (tools/xarm5/freeze_training_set.sh) y se entrena con
#     ella. Si un entrenamiento usó el nombre fijo y su contenido no está congelado, este script no lo reemplaza.
#   - Caché de DROID (se conserva): datasets/vio_frames/<crudo>/ (cuadros de cada bag) y
#     experiments/v8_validation/<crudo>/droid/ (trayectorias). El intermedio con las poses de DROID se arma en
#     datasets/_work_<crudo>/ y se borra al terminar (se rehace en segundos).
#
# Pasos:
#   1. DROID-SLAM en las demos que aún no tienen trayectoria (validate_v8_batch.sh; GPU, reanudable).
#   2. Pose real de inicio de todas (estimate_start_poses.py: ArUco + IMU en ARMING).
#   3. Informe de validación (validate_v8_episodes.py report): OK, REVISAR o REPETIR. Las REPETIR no entran; marcarlas
#      descartadas en el crudo requiere el OK del usuario.
#   4. Armado con las poses de DROID (build_dataset_from_bags.py).
#   5. Conversión al xArm5 (conversor v3) con la pose real de inicio. Conserva las demos marcadas por alcance (decisión
#      del 2026-11-25: entrenar con todas o solo con las factibles y comparar).
#   6. Verificación contra la referencia (verify_training_dataset.py: poses de DROID obligatorias, paso en la GPU).
#   7. Reemplazo del dataset de entrenamiento y limpieza de la carpeta de trabajo.
#
#   tools/xarm5/make_v8_training_set.sh [crudo=demonstrations_real_v8] [referencia=demonstrations_real_v7_xarm5_v2]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
RAW=${1:-demonstrations_real_v8}
REF=${2:-demonstrations_real_v7_xarm5_v2}
OUT=${RAW}_xarm5
V=experiments/v8_validation/$RAW
WORK=datasets/_work_$RAW
[ -d "datasets/$RAW.zarr" ] || { echo "ERROR: no existe datasets/$RAW.zarr"; exit 1; }
[ -d "datasets/$REF.zarr" ] || { echo "ERROR: no existe la referencia datasets/$REF.zarr"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx visuomotor_ros2 || { echo "ERROR: el contenedor visuomotor_ros2 no corre"; exit 1; }
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/make_v8_training_set_$RAW.lock"
flock -n 9 || { echo "ERROR: ya hay otro armado de $RAW corriendo"; exit 1; }
run() { docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && $*"; }

# Un entrenamiento que usó el nombre fijo perdería su dataset al reemplazarlo: congelarlo antes y entrenar con la copia
check_frozen() {
  docker exec -i -u 1000 -w /ros visuomotor_ros2 python3 - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]

def stamp(path):
    try:
        return json.load(open(path)).get("xarm5_conversion", {}).get("timestamp")
    except (OSError, ValueError):
        return None

cur = stamp(f"datasets/{out}.zarr/.zattrs")
if cur is None:
    sys.exit(0)
users = []
for meta in glob.glob("visuomotor_core/outputs/*/training_meta.json"):
    try:
        name = json.load(open(meta)).get("dataset_name")
    except (OSError, ValueError):
        continue
    if name == out and os.path.getmtime(meta) >= cur:          # entrenó con el contenido ACTUAL
        users.append(os.path.dirname(meta))
if users and not any(stamp(p) == cur for p in glob.glob(f"datasets/{out}_*.zarr/.zattrs")):
    print(f"ERROR: estos entrenamientos usaron datasets/{out}.zarr (nombre fijo) y lo perderían al reemplazarlo:")
    print("\n".join("  " + u for u in users))
    print(f"Congelarlo antes: tools/xarm5/freeze_training_set.sh {out}")
    sys.exit(1)
EOF
}
check_frozen

rm -rf "$WORK"
mkdir -p "$WORK"
echo "== 1/7 DROID en las demos nuevas"
tools/xarm5/validate_v8_batch.sh 0 100000 "$RAW" | { grep -E "DROID en|ERROR|TERMINADO" || true; }
echo "== 2/7 Poses reales de inicio"
run "python3 tools/xarm5/estimate_start_poses.py --zarr datasets/$RAW.zarr --out $V/start_poses.json" | tail -1
echo "== 3/7 Informe de validación"
run "python3 tools/xarm5/validate_v8_episodes.py report --dataset datasets/$RAW.zarr --start-poses $V/start_poses.json" \
  | { grep -vE " OK $" || true; } | tail -30
echo "== 4/7 Armado con las poses de DROID"
run "python3 tools/xarm5/build_dataset_from_bags.py --src datasets/$RAW.zarr --dst $WORK/${RAW}_droid.zarr" \
  > "$V/build.log" 2>&1 || { tail -3 "$V/build.log"; echo "ERROR en el armado (ver $V/build.log)"; exit 1; }
tail -1 "$V/build.log" | cut -c1-250
echo "== 5/7 Conversión al xArm5"
run "python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src $WORK/${RAW}_droid.zarr --dst $WORK/$OUT.zarr \
  --start-poses $V/start_poses.json" > "$V/convert.log" 2>&1 \
  || { tail -3 "$V/convert.log"; echo "ERROR en la conversión (ver $V/convert.log)"; exit 1; }
tail -1 "$V/convert.log" | cut -c1-250
echo "== 6/7 Verificación contra datasets/$REF.zarr"
if ! run "python3 tools/xarm5/verify_training_dataset.py --dataset $WORK/$OUT.zarr --reference datasets/$REF.zarr \
     --require-pose-source droid_rgbd --forward --out $V/verify.json" > "$V/verify.log" 2>&1; then
  grep -A20 "^PROBLEMAS" "$V/verify.log" || tail -5 "$V/verify.log"
  echo "NO se reemplaza datasets/$OUT.zarr: el nuevo queda en $WORK para revisarlo"
  exit 1
fi
grep -E "^Origen de las poses|^Marcadas|^VEREDICTO" "$V/verify.log" || true
echo "== 7/7 Reemplazo de datasets/$OUT.zarr"
check_frozen                                      # por si alguien empezó a entrenar con él mientras tanto
mkdir -p "$WORK/anterior"
[ ! -e "datasets/$OUT.zarr" ] || mv "datasets/$OUT.zarr" "$WORK/anterior/"
[ ! -e "datasets/${OUT}_conversion_report.json" ] || mv "datasets/${OUT}_conversion_report.json" "$WORK/anterior/"
mv "$WORK/$OUT.zarr" "datasets/$OUT.zarr"
mv "$WORK/${OUT}_conversion_report.json" "datasets/${OUT}_conversion_report.json"
rm -rf "$WORK"
n=$(tail -1 "$V/convert.log" | grep -oE "^[0-9]+ episodios" || true)
echo "LISTO en $((SECONDS / 60)) min $((SECONDS % 60)) s: datasets/$OUT.zarr ($n, poses de DROID)."
echo "Para entrenar, congelar antes una copia: tools/xarm5/freeze_training_set.sh $OUT"
