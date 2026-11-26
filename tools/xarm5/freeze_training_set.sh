#!/usr/bin/env bash
# Congela un dataset de entrenamiento para entrenar con él. Crea una copia con fecha, de SOLO LECTURA, que ya no se
# toca (dataset FUNCIONAL, AGENTS.md §2.3, decisión del usuario del 2026-11-26). Se entrena SIEMPRE con la copia, y así
# el training_meta.json del modelo la nombra. El nombre fijo no sirve: make_v8_training_set.sh lo rehace cada vez que
# se suman demos.
#
#   tools/xarm5/freeze_training_set.sh [dataset=demonstrations_real_v8_xarm5] [etiqueta=AAAA-MM-DD]
#   -> datasets/<dataset>_<etiqueta>.zarr y su informe de conversión.
#   Borrar una copia congelada (solo a pedido del usuario): chmod -R u+w <copia> && rm -rf <copia>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
D=${DATASETS_DIR:-datasets}
SRC=${1:-demonstrations_real_v8_xarm5}
TAG=${2:-$(date +%F)}
DST=${SRC}_${TAG}
[ -d "$D/$SRC.zarr" ] || { echo "ERROR: no existe $D/$SRC.zarr"; exit 1; }
[ ! -e "$D/$DST.zarr" ] || { echo "ERROR: $D/$DST.zarr ya existe (una copia congelada no se sobrescribe): usa otra etiqueta"; exit 1; }
tmp="$D/.$DST.zarr"                               # oculta hasta que la copia está completa y protegida
rm -rf "$tmp"
cp -a "$D/$SRC.zarr" "$tmp"
if [ -f "$D/${SRC}_conversion_report.json" ]; then
  cp -a "$D/${SRC}_conversion_report.json" "$D/${DST}_conversion_report.json"
  chmod a-w "$D/${DST}_conversion_report.json"
fi
chmod -R a-w "$tmp"
mv "$tmp" "$D/$DST.zarr"
echo "Congelado (solo lectura): $D/$DST.zarr ($(du -sh "$D/$DST.zarr" | cut -f1))"
echo "Entrenar con: --root /ros/$D/$DST.zarr"
