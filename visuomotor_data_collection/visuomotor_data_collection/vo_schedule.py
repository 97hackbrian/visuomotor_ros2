"""Programa de recolección: posiciones (celda) y orientaciones (yaw) de la herramienta, repartidas por igual.

Los intentos en el xArm5 (2026-11-23) fallaron donde v7 no tenía demos: tijeras giradas (intentos 7, 9 y 10) y
tijeras junto a la base (8 y 11). El programa reparte las demos de v8 por igual entre celdas y orientaciones, y el
recolector guarda en cada episodio cuál le tocó.

Celdas: teclado numérico tal como se ve en la imagen de HOME3, definidas EN METROS en link_base
(visuomotor_bringup/config/realsense/v8_cells.yaml; alcanzabilidad: tools/hardware/make_v8_cells.py). La guía del
recolector dibuja la grilla numerada SOBRE LA MESA (homografía con los ArUco detectados) y resalta la celda pedida.
    7 8 9   lejos de la base (arriba de la imagen)
    4 5 6   medio
    1 2 3   cerca de la base (abajo de la imagen)

Yaw = hacia dónde apuntan las HOJAS de la herramienta, visto desde atrás del robot (convención ``YAW_CONVENTION``,
definida por el usuario el 2026-11-25):
      0° = de frente (hojas lejos de la base, +x)        +90° = a la DERECHA (-y)
    -90° = a la IZQUIERDA (+y)                           180° = hacia la base (-x)
El programa v8 original (v8_schedule.csv, 4 yaws) usaba "positivo = antihorario"; sus episodios lo dicen en sus
atributos (sin ``yaw_convention``).

Sin ROS: lo usan ``vo_collector_node`` (parámetro ``schedule_file``) y ``tools/hardware/make_collection_schedule.py``.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

YAW_CONVENTION = "hojas: 0=frente(+x), +90=derecha(-y), -90=izquierda(+y), 180=hacia la base"

CELL_NAMES = {7: "lejos-izquierda", 8: "lejos-centro", 9: "lejos-derecha",
              4: "medio-izquierda", 5: "centro", 6: "medio-derecha",
              1: "cerca-izquierda, junto a la base", 2: "cerca-centro, junto a la base",
              3: "cerca-derecha, junto a la base"}


def yaw_words(yaw_deg: float) -> str:
    """'0° de frente', '45° a la derecha', '90° a la izquierda', '180° hacia la base'."""
    y = ((float(yaw_deg) + 180.0) % 360.0) - 180.0
    if abs(y) < 1e-6:
        return "0° de frente"
    if abs(abs(y) - 180.0) < 1e-6:
        return "180° hacia la base"
    return f"{abs(y):.0f}° a la {'derecha' if y > 0 else 'izquierda'}"


def blade_direction_xy(yaw_deg: float) -> Tuple[float, float]:
    """Dirección de las hojas en el plano de la mesa (x, y de link_base) según ``YAW_CONVENTION``."""
    t = math.radians(float(yaw_deg))
    return math.cos(t), -math.sin(t)


def cells_from_edges(x_edges: Sequence[float], y_edges: Sequence[float]) -> Dict[int, Tuple[float, float, float, float]]:
    """Celdas 1-9 en metros: ``{celda: (x_lo, x_hi, y_lo, y_hi)}``. Filas por x (1-3 cerca de la base), columnas de
    izquierda (+y) a derecha (-y), como el teclado numérico visto en la imagen de HOME3."""
    if len(x_edges) != 4 or len(y_edges) != 4:
        raise ValueError("se esperan 4 bordes en x y 4 en y (3x3 celdas)")
    xe, ye = sorted(float(v) for v in x_edges), sorted(float(v) for v in y_edges)
    out = {}
    for ri, ids in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9))):
        for ci, cid in enumerate(ids):
            out[cid] = (xe[ri], xe[ri + 1], ye[2 - ci], ye[3 - ci])
    return out


def placement_hint(cell_info: Optional[dict]) -> str:
    """Pista para la vista final de una celda, desde ``v8_cells_home3.yaml`` (``make_v8_cells.py``):
    * ``tilt_needed_deg`` > 0: el robot solo llega mirando EN DIAGONAL desde más cerca de la base;
    * ``reachable_fraction`` < 1 con el centro alcanzable: no llega a alguna esquina, herramienta hacia el centro.
    Vacía si basta con la cámara justo encima de cualquier punto de la celda."""
    if not cell_info:
        return ""
    out = []
    if float(cell_info.get("reachable_fraction", 1.0)) < 1.0 and float(cell_info.get("center_reachable_fraction", 1.0)) >= 1.0:
        out.append("herramienta hacia el centro de la celda")
    tilt = float(cell_info.get("tilt_needed_deg", 0.0))
    if tilt > 0:
        out.append(f"final en diagonal (~{tilt:.0f}°) desde más cerca del robot")
    return "; ".join(out)


@dataclass
class ScheduleEntry:
    index: int
    cell: int
    yaw_deg: float
    rep: int = 0

    def describe(self) -> str:
        return f"#{self.index}: celda {self.cell} ({CELL_NAMES.get(self.cell, '?')}), hojas {yaw_words(self.yaw_deg)}"

    def as_attrs(self) -> dict:
        return {"index": int(self.index), "cell": int(self.cell), "yaw_deg": float(self.yaw_deg), "rep": int(self.rep)}


def make_schedule(cells: Sequence[int], yaws: Sequence[float], reps: int, seed: int = 0) -> List[ScheduleEntry]:
    """(Programa v8 original) todas las combinaciones celda × yaw, ``reps`` veces, cada repetición barajada."""
    _check_cells(cells)
    rng = random.Random(seed)
    out: List[ScheduleEntry] = []
    for rep in range(reps):
        combo = [(c, float(y)) for c in cells for y in yaws]
        rng.shuffle(combo)
        base = len(out)                                       # fijo: extend con un generador cambia len(out)
        out.extend([ScheduleEntry(index=base + k, cell=c, yaw_deg=y, rep=rep) for k, (c, y) in enumerate(combo)])
    return out


def make_balanced_schedule(cells: Sequence[int], yaws: Sequence[float], reps: int, seed: int = 0
                           ) -> List[ScheduleEntry]:
    """Programa por RONDAS: cada ronda tiene todas las celdas una vez, así que cortar la sesión en cualquier punto
    deja las celdas parejas (difieren en a lo más 1) y no hay tendencia de posición al retomar.

    Las orientaciones van en un cuadrado latino: en ``len(yaws)`` rondas (un ciclo) cada celda pasa por todas las
    orientaciones una vez, es decir, cada combinación celda × yaw sale una vez por ciclo. Con 9 celdas y 6 yaws, el
    orden de rondas (0, 3, 1, 4, 2, 5) deja además cada yaw exactamente 3 veces en cada par de rondas (18 demos).
    ``reps`` = ciclos. Dentro de una ronda el orden es al azar y no repite la celda de la demo anterior."""
    _check_cells(cells)
    cells, yaws = list(cells), [float(y) for y in yaws]
    m = len(yaws)
    if m == 0:
        raise ValueError("sin orientaciones")
    rng = random.Random(seed)
    half = (m + 1) // 2
    order = [b for pair in zip(range(half), range(half, m)) for b in pair] + ([half - 1] if m % 2 else [])
    order = list(dict.fromkeys(order))                        # m impar: sin duplicados
    out: List[ScheduleEntry] = []
    for rep in range(reps):
        perm = cells[:]
        rng.shuffle(perm)                                    # qué celda empieza en qué yaw, distinto por ciclo
        shift = rng.randrange(m)
        for b in order:
            rnd = [(c, yaws[(i + b + shift) % m]) for i, c in enumerate(perm)]
            rng.shuffle(rnd)
            if out and len(rnd) > 1 and rnd[0][0] == out[-1].cell:
                rnd[0], rnd[1] = rnd[1], rnd[0]
            base = len(out)                                   # fijo: con un generador, len(out) cambia al extender
            out.extend([ScheduleEntry(index=base + k, cell=c, yaw_deg=y, rep=rep) for k, (c, y) in enumerate(rnd)])
    return out


def _check_cells(cells: Sequence[int]) -> None:
    for c in cells:
        if c not in CELL_NAMES:
            raise ValueError(f"celda {c} fuera de 1-9")


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


def schedule_id(path) -> str:
    """Identidad del programa (nombre del archivo sin extensión). Un episodio cuenta como hecho solo en el programa
    con el que se grabó: si no, al cambiar de programa, los índices viejos marcarían como hechas entradas nuevas."""
    return Path(path).stem


def done_indices(episode_attrs: Iterable[dict], sched_id: str, legacy_id: Optional[str] = None) -> set:
    """Índices hechos del programa ``sched_id`` según los atributos de los episodios. Los episodios sin
    ``schedule_id`` (recolector anterior al 2026-11-25) cuentan solo si ``legacy_id == sched_id``."""
    out = set()
    for a in episode_attrs:
        sch = (a or {}).get("schedule") or {}
        if "index" not in sch:
            continue
        if sch.get("schedule_id", legacy_id) == sched_id:
            out.add(int(sch["index"]))
    return out


def next_pending(entries: Sequence[ScheduleEntry], done: Iterable[int], skipped: Iterable[int] = ()
                 ) -> Optional[ScheduleEntry]:
    """Primera entrada que no está hecha ni saltada."""
    taken = set(int(i) for i in done) | set(int(i) for i in skipped)
    for e in entries:
        if e.index not in taken:
            return e
    return None
