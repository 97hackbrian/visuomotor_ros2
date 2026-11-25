#!/usr/bin/env python3
"""Genera el programa de recolección (celda × orientación de la herramienta) para v8, por rondas balanceadas.

Cada ronda tiene las 9 celdas una vez y las orientaciones van en cuadrado latino (``make_balanced_schedule``):
cortar la sesión en cualquier punto deja las celdas parejas, y cada 2 rondas (18 demos) cada orientación sale 3 veces.
Al retomar, el recolector sigue desde la primera entrada pendiente (no repite las hechas).

Ver ``visuomotor_data_collection/vo_schedule.py`` (definición de celdas y yaw). El recolector lo usa con
``-p schedule_file:=<csv>``: muestra la siguiente posición y la guarda en cada episodio.

Ejemplos (desde la raíz del repo):
    python3 tools/hardware/make_collection_schedule.py --out experiments/collection_schedules/v8.csv
    python3 tools/hardware/make_collection_schedule.py --out experiments/collection_schedules/v8_schedule_v2.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_schedule import (  # noqa: E402
    CELL_NAMES, YAW_CONVENTION, make_balanced_schedule, save_csv, yaw_words)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--yaws", default="", help="grados (" + YAW_CONVENTION + "); vacío = tool_yaws_deg de --cells-config")
    ap.add_argument("--cells-config", default=str(REPO / "visuomotor_bringup/config/realsense/v8_cells.yaml"))
    ap.add_argument("--reps", type=int, default=2, help="ciclos: cada uno tiene todas las celdas × orientaciones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/collection_schedules/v8_schedule_v2.csv")
    a = ap.parse_args(argv)
    cells = [int(c) for c in a.cells.split(",") if c.strip()]
    if a.yaws.strip():
        yaws = [float(y) for y in a.yaws.split(",") if y.strip()]
    else:
        import yaml
        yaws = [float(y) for y in yaml.safe_load(open(a.cells_config))["v8_cells"]["tool_yaws_deg"]]
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    if out.exists():
        print(f"ERROR: {out} ya existe (no se sobrescribe un programa en uso). Usar otro --out.")
        return 1
    entries = make_balanced_schedule(cells, yaws, a.reps, a.seed)
    save_csv(entries, out)
    print(f"{len(entries)} demos = {len(cells)} celdas × {len(yaws)} orientaciones × {a.reps} repeticiones -> {out}")
    print("Celdas (teclado numérico visto en la imagen de HOME3): " +
          ", ".join(f"{c} {CELL_NAMES[c]}" for c in sorted(cells)))
    print("Primeras 10:")
    for e in entries[:10]:
        print("  " + e.describe())
    per_cell = Counter(e.cell for e in entries)
    print("Por celda: " + ", ".join(f"{c}:{n}" for c, n in sorted(per_cell.items())))
    print("Por orientación: " + ", ".join(f"{yaw_words(y)}: {n}" for y, n in sorted(Counter(e.yaw_deg for e in entries).items())))
    worst_c = max(max(Counter(e.cell for e in entries[:k]).values()) - min(Counter(e.cell for e in entries[:k]).get(c, 0) for c in cells)
                  for k in range(1, len(entries) + 1))
    print(f"Balance al cortar en cualquier punto: las celdas difieren en a lo más {worst_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
