"""Programa de recolección: lista al azar de posiciones (celda) y orientaciones (yaw) de la herramienta.

Los intentos en el xArm5 (2026-11-23) fallaron donde v7 no tenía demos: tijeras giradas (intentos 7, 9 y 10) y
tijeras junto a la base (8 y 11). El programa reparte las demos de v8 por igual entre celdas y orientaciones, en orden
aleatorio, y el recolector guarda en cada episodio cuál le tocó.

Celdas: teclado numérico tal como se ve en la imagen de HOME3, definidas EN METROS en link_base
(visuomotor_bringup/config/realsense/v8_cells.yaml; generador y alcanzabilidad: tools/hardware/make_v8_cells.py). No son
tercios de la imagen: el tercio superior queda fuera del alcance del robot. La guía del recolector dibuja la celda
pedida y la orientación de las hojas.
    7 8 9   lejos de la base (arriba de la imagen)
    4 5 6   medio
    1 2 3   cerca de la base (abajo de la imagen)
Yaw: 0° = como en v7 (hojas hacia arriba de la imagen, mangos hacia la base); positivo = giro antihorario visto en
la imagen de HOME3.

Sin ROS: lo usan ``vo_collector_node`` (parámetro ``schedule_file``) y ``tools/hardware/make_collection_schedule.py``.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

CELL_NAMES = {7: "lejos-izquierda", 8: "lejos-centro", 9: "lejos-derecha",
              4: "medio-izquierda", 5: "centro", 6: "medio-derecha",
              1: "cerca-izquierda, junto a la base", 2: "cerca-centro, junto a la base",
              3: "cerca-derecha, junto a la base"}


@dataclass
class ScheduleEntry:
    index: int
    cell: int
    yaw_deg: float
    rep: int = 0

    def describe(self) -> str:
        return (f"#{self.index}: celda {self.cell} ({CELL_NAMES.get(self.cell, '?')}), herramienta a "
                f"{self.yaw_deg:+.0f}°")

    def as_attrs(self) -> dict:
        return {"index": int(self.index), "cell": int(self.cell), "yaw_deg": float(self.yaw_deg), "rep": int(self.rep)}


def make_schedule(cells: Sequence[int], yaws: Sequence[float], reps: int, seed: int = 0) -> List[ScheduleEntry]:
    """Todas las combinaciones celda × yaw, ``reps`` veces. Cada repetición completa va barajada antes de la
    siguiente, para que una sesión cortada a la mitad igual quede balanceada."""
    for c in cells:
        if c not in CELL_NAMES:
            raise ValueError(f"celda {c} fuera de 1-9")
    rng = random.Random(seed)
    out: List[ScheduleEntry] = []
    for rep in range(reps):
        combo = [(c, float(y)) for c in cells for y in yaws]
        rng.shuffle(combo)
        base = len(out)                                       # fijo: extend con un generador cambia len(out)
        out.extend([ScheduleEntry(index=base + k, cell=c, yaw_deg=y, rep=rep) for k, (c, y) in enumerate(combo)])
    return out


def save_csv(entries: Iterable[ScheduleEntry], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "cell", "yaw_deg", "rep", "description"])
        for e in entries:
            w.writerow([e.index, e.cell, f"{e.yaw_deg:g}", e.rep, e.describe()])


def load_csv(path) -> List[ScheduleEntry]:
    with Path(path).open(newline="") as f:
        return [ScheduleEntry(index=int(r["index"]), cell=int(r["cell"]), yaw_deg=float(r["yaw_deg"]),
                              rep=int(r.get("rep") or 0)) for r in csv.DictReader(f)]


def next_pending(entries: Sequence[ScheduleEntry], done: Iterable[int], skipped: Iterable[int] = ()
                 ) -> Optional[ScheduleEntry]:
    """Primera entrada que no está hecha ni saltada."""
    taken = set(int(i) for i in done) | set(int(i) for i in skipped)
    for e in entries:
        if e.index not in taken:
            return e
    return None
