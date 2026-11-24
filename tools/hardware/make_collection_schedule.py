#!/usr/bin/env python3
"""Genera el programa de recolección (celda × orientación de la herramienta, en orden aleatorio) para v8.

Ver ``visuomotor_data_collection/vo_schedule.py`` (definición de celdas y yaw). El recolector lo usa con
``-p schedule_file:=<csv>``: muestra la siguiente posición y la guarda en cada episodio.

Ejemplos (desde la raíz del repo):
    python3 tools/hardware/make_collection_schedule.py --out experiments/collection_schedules/v8.csv
    python3 tools/hardware/make_collection_schedule.py --cells 1,2,3,4,5,6,7,8,9 --yaws 0,45,-45,90 --reps 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_schedule import CELL_NAMES, make_schedule, save_csv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--yaws", default="0,45,-45,90", help="grados; 0 = como en v7, positivo = antihorario")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/collection_schedules/v8_schedule.csv")
    a = ap.parse_args(argv)
    cells = [int(c) for c in a.cells.split(",") if c.strip()]
    yaws = [float(y) for y in a.yaws.split(",") if y.strip()]
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    if out.exists():
        print(f"ERROR: {out} ya existe (no se sobrescribe un programa en uso). Usar otro --out.")
        return 1
    entries = make_schedule(cells, yaws, a.reps, a.seed)
    save_csv(entries, out)
    print(f"{len(entries)} demos = {len(cells)} celdas × {len(yaws)} orientaciones × {a.reps} repeticiones -> {out}")
    print("Celdas (teclado numérico visto en la imagen de HOME3): " +
          ", ".join(f"{c} {CELL_NAMES[c]}" for c in sorted(cells)))
    print("Primeras 10:")
    for e in entries[:10]:
        print("  " + e.describe())
    per_cell = Counter(e.cell for e in entries)
    print("Por celda: " + ", ".join(f"{c}:{n}" for c, n in sorted(per_cell.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
