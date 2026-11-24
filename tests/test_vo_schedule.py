"""Pruebas del programa de recolección (visuomotor_data_collection.vo_schedule), sin ROS."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_schedule import load_csv, make_schedule, next_pending, save_csv  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
