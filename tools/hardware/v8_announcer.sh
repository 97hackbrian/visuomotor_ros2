#!/usr/bin/env bash
# Avisos por los parlantes durante la recolección a mano: con la cámara en la mano no se mira la pantalla.
# Lee el registro del recolector (tmux pipe-pane, lo prepara v8_tmux.sh collect) y, en cada evento, pita y dice en voz
# alta qué pasa: celda siguiente, "quieta", "grabando", "más lento", "guardada", "rechazada", "cancelada".
#
#   tools/hardware/v8_announcer.sh <registro_del_recolector>
#   tools/hardware/v8_announcer.sh --countdown 15      # cuenta regresiva con pitidos y termina
#   tools/hardware/v8_announcer.sh --test               # un pitido y una frase
set -u
SND=/usr/share/sounds/freedesktop/stereo
VOICE_LANG=${V8_VOICE_LANG:-es}
SLOW_EVERY_S=${V8_SLOW_EVERY_S:-3}      # "más lento" como mucho cada tantos segundos
# Meta de la SESIÓN (2026-11-26, pedido del usuario): con V8_GOAL=112 la voz dice "Llevan N de 112" (N = entradas
# hechas del programa) en vez del largo del programa (220), y avisa al cumplirla. Vacío = el largo del programa.
GOAL=${V8_GOAL:-}

play() { local f="$SND/$1.oga"; [ -f "$f" ] || return 0
  if command -v paplay >/dev/null; then paplay "$f" 2>/dev/null & elif command -v pw-play >/dev/null; then pw-play "$f" 2>/dev/null & fi; }
say() { if command -v spd-say >/dev/null; then spd-say -l "$VOICE_LANG" -r 10 "$1" 2>/dev/null
        elif command -v espeak >/dev/null; then espeak -v "$VOICE_LANG" "$1" 2>/dev/null & fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

case "${1:-}" in
  --test) play bell; sleep 0.4; say "Avisos de la recolección activados"; exit 0 ;;
  --countdown)
    n=${2:-15}; say "La recolección empieza en $n segundos"; sleep 3
    for ((i = n - 3; i > 0; i--)); do
      if [ "$i" -le 5 ]; then play bell; say "$i"; elif [ $((i % 5)) -eq 0 ]; then play bell; fi
      sleep 1
    done
    play complete; exit 0 ;;
  "") echo "uso: $0 <registro> | --countdown N | --test"; exit 1 ;;
esac

LOG=$1
touch "$LOG"
last_slow=0
rejected=0                                  # "Rechazada" una sola vez por demo ([S] repetida no la repite)
stamp "escuchando $LOG"
tail -n 0 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  line=${line//$'\r'/}
  case "$line" in
    *"[PROGRAMA] Siguiente #"*)
      cell=$(sed -nE 's/.*celda ([0-9]+).*/\1/p' <<<"$line")
      yaw=$(sed -nE 's/.*hojas (.*) \(([0-9]+)\/([0-9]+) hechas.*/\1/p' <<<"$line"); yaw=${yaw//°/ grados}
      cnt=$(sed -nE 's/.*\(([0-9]+)\/([0-9]+) hechas.*/\1 de \2/p' <<<"$line")
      if [ -n "$GOAL" ]; then
        n=$(sed -nE 's/.*\(([0-9]+)\/[0-9]+ hechas.*/\1/p' <<<"$line"); cnt="$n de $GOAL"
        if [ -n "$n" ] && [ "$n" -ge "$GOAL" ] && [ "${goal_said:-0}" -eq 0 ]; then goal_said=1
          stamp "meta de la sesión cumplida ($cnt)"; play complete; say "Meta de la sesión cumplida. $n de $GOAL"; fi
      fi
      hint=$(sed -nE 's/.*Ojo: (.*)\.$/\1/p' <<<"$line"); hint=${hint//°/ grados}; hint=${hint//\~/}
      stamp "siguiente: celda $cell, hojas $yaw${hint:+ ($hint)}"; play dialog-information
      say "Siguiente: celda $cell. Hojas $yaw.${hint:+ Ojo: $hint.} Llevan $cnt" ;;
    *"[PROGRAMA] Completo"*) stamp "programa completo"; play complete; say "Programa completo" ;;
    *"Vista de HOME3 alineada y quieta"*) stamp "alineada: quieta"; play bell; say "Quieta" ;;
    *">>> [REC] Grabando"*) rejected=0; stamp "grabando"; play camera-shutter; say "Grabando" ;;
    *"deje de girar"*) stamp "sin girar"; say "Quieto, sin girar" ;;
    *"MÁS LENTO"*)
      now=$(date +%s); if [ $((now - last_slow)) -ge "$SLOW_EVERY_S" ]; then last_slow=$now; stamp "más lento"; say "Más lento"; fi ;;
    *"[SAVE] episode_"*)
      ep=$(sed -nE 's/.*\[SAVE\] episode_([0-9]+).*/\1/p' <<<"$line"); stamp "guardada episode_$ep"; play complete
      say "Guardada. Demo $((ep + 1))" ;;
    *"Episodio NO guardado"*|*"[AUTO] No se guardó"*)
      if [ "$rejected" -eq 0 ]; then rejected=1; stamp "rechazada"; play dialog-error; say "Rechazada. S para guardar igual, D para descartar"; fi ;;
    *"episodio cancelado"*|*"no entregó poses válidas"*) stamp "cancelada"; play dialog-warning; say "Cancelada. Volver a la vista inicial" ;;
    *"[DISCARD]"*) stamp "descartada"; play dialog-warning; say "Descartada" ;;
    *"[ALCANCE] FUERA"*) stamp "fuera del alcance del robot"; play dialog-warning; say "Fuera del alcance del robot. $(sed -nE 's/.*\): (.*)\.$/\1/p' <<<"$line")" ;;
    *"[ALCANCE] cerca"*) stamp "cerca del límite del robot"; say "Cerca del límite del robot" ;;
    *"[UNDO]"*) stamp "última guardada descartada"; play dialog-warning; say "Última demo descartada. Se vuelve a pedir" ;;
  esac
done
