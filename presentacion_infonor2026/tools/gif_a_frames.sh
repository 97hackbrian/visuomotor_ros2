#!/usr/bin/env bash
# =============================================================================
# Convierte un GIF o un video en una secuencia de cuadros JPG para el paquete
# LaTeX `animate` (\animategraphics), que reproduce animaciones dentro del PDF.
#
# Uso:
#   tools/gif_a_frames.sh ENTRADA DIR_SALIDA [FPS] [ANCHO] [VELOCIDAD] [INICIO_S] [DURACION_S] [FILTRO_PREVIO] [CALIDAD]
#     FPS           cuadros por segundo de salida (por defecto 10)
#     ANCHO         ancho en píxeles; el alto conserva la proporción (por defecto 400)
#     VELOCIDAD     factor de aceleración: 2 = velocidad x2 (por defecto 1)
#     INICIO_S      segundo inicial del recorte (por defecto 0)
#     DURACION_S    duración del recorte en segundos de la fuente; 0 = completo
#     FILTRO_PREVIO filtergraph de ffmpeg aplicado antes (por ejemplo, un crop)
#     CALIDAD       calidad JPEG de ffmpeg, 2 (mejor) a 31 (peor); por defecto 4
#
# Ejemplo:
#   tools/gif_a_frames.sh demo.gif assets/gifs/demo 10 256
#   -> \animategraphics[autoplay,loop,width=30mm]{10}{assets/gifs/demo/f-}{0}{N}
# =============================================================================
set -euo pipefail
if [ "$#" -lt 2 ]; then
  sed -n '2,19p' "$0"; exit 1
fi
in="$1"; out="$2"; fps="${3:-10}"; width="${4:-400}"; speed="${5:-1}"
ss="${6:-0}"; dur="${7:-0}"; pre="${8:-}"; q="${9:-4}"

mkdir -p "$out"
rm -f "$out"/f-*.jpg

vf="setpts=PTS/${speed},fps=${fps},scale=${width}:-2"
if [ -n "$pre" ]; then vf="${pre},${vf}"; fi

args=(-v error -y -ss "$ss")
if [ "$dur" != "0" ]; then args+=(-t "$dur"); fi
ffmpeg "${args[@]}" -i "$in" -vf "$vf" -q:v "$q" -start_number 0 "$out/f-%d.jpg"

n=$(find "$out" -maxdepth 1 -name 'f-*.jpg' | wc -l)
echo "Cuadros: $n  ->  \\animategraphics[autoplay,loop,width=...]{${fps}}{${out}/f-}{0}{$((n - 1))}"
