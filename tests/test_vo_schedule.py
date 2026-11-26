"""Pruebas del programa de recolección (visuomotor_data_collection.vo_schedule), sin ROS."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_schedule import (  # noqa: E402
    ScheduleEntry, blade_direction_xy, cells_from_edges, continue_schedule, done_indices, load_csv, make_balanced_schedule,
    make_schedule, next_pending, placement_hint, save_csv, schedule_id, yaw_words)


class ScheduleTest(unittest.TestCase):
    def test_balanced_unique_and_reproducible(self):
        e = make_schedule(list(range(1, 10)), [0, 45, -45, 90], reps=3, seed=0)
        self.assertEqual(len(e), 108)
        self.assertEqual([x.index for x in e], list(range(108)))           # índices consecutivos y únicos
        self.assertEqual(set(Counter(x.cell for x in e).values()), {12})   # 4 orientaciones x 3 repeticiones
        self.assertEqual(set(Counter((x.cell, x.yaw_deg) for x in e).values()), {3})
        for rep in range(3):                                               # cada repetición completa y balanceada
            self.assertEqual(len({(x.cell, x.yaw_deg) for x in e if x.rep == rep}), 36)
        e2 = make_schedule(list(range(1, 10)), [0, 45, -45, 90], reps=3, seed=0)
        self.assertEqual([(x.cell, x.yaw_deg) for x in e], [(x.cell, x.yaw_deg) for x in e2])

    def test_csv_and_next_pending(self):
        e = make_schedule([1, 5], [0.0, 90.0], reps=1, seed=2)
        path = Path(tempfile.mkdtemp()) / "s.csv"
        save_csv(e, path)
        back = load_csv(path)
        self.assertEqual([(x.index, x.cell, x.yaw_deg) for x in back], [(x.index, x.cell, x.yaw_deg) for x in e])
        self.assertEqual(next_pending(back, done=[]).index, 0)
        self.assertEqual(next_pending(back, done=[0], skipped=[1]).index, 2)
        self.assertIsNone(next_pending(back, done=[0, 1, 2, 3]))
        with self.assertRaises(ValueError):
            make_schedule([0], [0.0], 1)



class BalancedScheduleTest(unittest.TestCase):
    """Programa v2 (2026-11-25): 6 orientaciones definidas por el usuario y rondas balanceadas, para que cortar la
    sesión en cualquier punto no deje tendencia de posición ni de orientación."""
    CELLS, YAWS = list(range(1, 10)), [-90.0, -45.0, 0.0, 45.0, 90.0, 180.0]

    def test_rounds_latin_and_prefix_balance(self):
        e = make_balanced_schedule(self.CELLS, self.YAWS, reps=2, seed=0)
        self.assertEqual(len(e), 108)
        self.assertEqual([x.index for x in e], list(range(108)))
        for cyc in (e[:54], e[54:]):                                   # cada combinación una vez por ciclo
            self.assertEqual(len({(x.cell, x.yaw_deg) for x in cyc}), 54)
        for r in range(12):                                            # cada ronda: todas las celdas una vez
            self.assertEqual(sorted(x.cell for x in e[9 * r:9 * r + 9]), self.CELLS)
        for b in range(6):                                             # cada 18 demos: cada orientación 3 veces
            self.assertEqual(set(Counter(x.yaw_deg for x in e[18 * b:18 * b + 18]).values()), {3})
        for k in range(1, 109):                                        # cortar en cualquier punto
            c = Counter(x.cell for x in e[:k])
            self.assertLessEqual(max(c.values()) - min(c.get(i, 0) for i in self.CELLS), 1)
        self.assertTrue(all(a.cell != b.cell for a, b in zip(e, e[1:])))
        e2 = make_balanced_schedule(self.CELLS, self.YAWS, reps=2, seed=0)
        self.assertEqual([(x.cell, x.yaw_deg) for x in e], [(x.cell, x.yaw_deg) for x in e2])

    def test_yaw_convention(self):
        self.assertEqual(yaw_words(0), "0° de frente")
        self.assertEqual(yaw_words(90), "90° a la derecha")
        self.assertEqual(yaw_words(-90), "90° a la izquierda")
        self.assertEqual(yaw_words(180), "180° hacia la base")
        for y, d in ((0, (1, 0)), (90, (0, -1)), (-90, (0, 1)), (180, (-1, 0))):   # +x lejos, +y izquierda
            self.assertAlmostEqual(blade_direction_xy(y)[0], d[0], places=9)
            self.assertAlmostEqual(blade_direction_xy(y)[1], d[1], places=9)

    def test_cells_numbering(self):
        c = cells_from_edges([0.42, 0.49, 0.555, 0.615], [-0.21, -0.07, 0.07, 0.21])
        self.assertEqual(c[1], (0.42, 0.49, 0.07, 0.21))                # cerca de la base, izquierda (+y)
        self.assertEqual(c[3], (0.42, 0.49, -0.21, -0.07))              # cerca, derecha
        self.assertEqual(c[8], (0.555, 0.615, -0.07, 0.07))             # lejos, centro

    def test_placement_hint(self):
        self.assertEqual(placement_hint(None), "")
        self.assertEqual(placement_hint({"reachable_fraction": 1.0, "tilt_needed_deg": 0.0}), "")
        self.assertIn("diagonal", placement_hint({"reachable_fraction": 1.0, "tilt_needed_deg": 15.0}))
        h = placement_hint({"reachable_fraction": 0.5, "center_reachable_fraction": 1.0, "tilt_needed_deg": 45.0})
        self.assertIn("centro de la celda", h)
        self.assertIn("45", h)
        by = {"reachable_fraction": 0.5, "center_reachable_fraction": 1.0, "tilt_needed_deg": 45.0,
              "by_yaw": {0: {"reachable_fraction": 1.0, "tilt_needed_deg": 0.0}, 180: {"reachable_fraction": 1.0,
                                                                                  "tilt_needed_deg": 15.0}}}
        self.assertEqual(placement_hint(by, 0.0), "")                       # esa orientación: justo encima
        self.assertIn("15", placement_hint(by, 180.0))
        self.assertIn("45", placement_hint(by))                             # sin orientación: el peor caso

    def test_done_only_counts_same_schedule(self):
        attrs = [{"schedule": {"index": 0, "cell": 1}},                          # programa v1, sin schedule_id
                 {"schedule": {"index": 3, "cell": 2, "schedule_id": "v8_schedule_v2"}}, {"schedule": None}, {}]
        self.assertEqual(done_indices(attrs, "v8_schedule_v2", "v8_schedule"), {3})
        self.assertEqual(done_indices(attrs, "v8_schedule", "v8_schedule"), {0})
        attrs.append({"schedule": {"index": 5, "schedule_id": "v8_schedule_v2"}, "discarded": True})   # [U]
        self.assertEqual(done_indices(attrs, "v8_schedule_v2", "v8_schedule"), {3})
        self.assertEqual(schedule_id("experiments/collection_schedules/v8_schedule_v2.csv"), "v8_schedule_v2")


class ContinueScheduleTest(unittest.TestCase):
    """2026-11-26: el programa v2 quedó con 4 pendientes (las demos descartadas). El programa siguiente las pone
    primero, con la misma celda y orientación, y después rondas balanceadas nuevas."""

    def test_pending_first_then_new_reindexed(self):
        pending = [ScheduleEntry(26, 3, 45.0, 0), ScheduleEntry(54, 8, 180.0, 0), ScheduleEntry(99, 9, 180.0, 1)]
        new = make_balanced_schedule(list(range(1, 10)), [-90, -45, 0, 45, 90, 180], reps=2, seed=1)
        out = continue_schedule(pending, new)
        self.assertEqual(len(out), len(pending) + len(new))
        self.assertEqual([e.index for e in out], list(range(len(out))))
        self.assertEqual([(e.cell, e.yaw_deg, e.rep) for e in out[:3]], [(3, 45.0, -1), (8, 180.0, -1), (9, 180.0, -1)])
        self.assertEqual(Counter((e.cell, e.yaw_deg) for e in out[3:]), Counter((e.cell, e.yaw_deg) for e in new))
        self.assertTrue(all(out[i].cell != out[i + 1].cell for i in range(len(out) - 1)))

    def test_generator_takes_pending_from_the_dataset(self):
        import zarr
        sys.path.insert(0, str(ROOT / "tools" / "hardware"))
        from make_collection_schedule import main
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prev = d / "prog_a.csv"
            save_csv([ScheduleEntry(i, c, y, 0) for i, (c, y) in enumerate([(1, 0.0), (2, 90.0), (3, 0.0), (1, 90.0)])],
                     prev)
            g = zarr.open_group(str(d / "raw.zarr"), mode="w")
            for k, (idx, discarded) in enumerate([(0, False), (1, True), (2, False)]):
                at = g.create_group(f"episode_{k}").attrs
                at["schedule"] = {"schedule_id": "prog_a", "index": idx}
                at["discarded"] = discarded
            out = d / "prog_b.csv"
            self.assertEqual(main(["--out", str(out), "--cells", "1,2,3", "--yaws", "0,90", "--reps", "1",
                                   "--continue-from", str(prev), "--dataset", str(d / "raw.zarr")]), 0)
            e = load_csv(out)
        self.assertEqual(len(e), 2 + 6)                       # #1 (descartada) y #3 (sin grabar) + 3 celdas × 2
        self.assertEqual([(x.cell, x.yaw_deg, x.rep) for x in e[:2]], [(2, 90.0, -1), (1, 90.0, -1)])


if __name__ == "__main__":
    unittest.main()
