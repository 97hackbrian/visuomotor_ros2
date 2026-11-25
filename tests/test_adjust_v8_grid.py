"""Pruebas de la herramienta de ajuste de la grilla v8 (tools/hardware/adjust_v8_grid.py), sin interfaz gráfica."""

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

try:
    from adjust_v8_grid import (check_limits, draw_overlay, edges_from_limits, limits_from_edges,  # noqa: E402
                                marker_limit_x, write_edges_yaml)
    from make_v8_cells import ReachModel, default_paths  # noqa: E402
    HAVE = True
except Exception as e:  # noqa: BLE001 (xarm5_interface u otras dependencias del contenedor)
    HAVE, ERR = False, e

CELLS_YAML = ROOT / "visuomotor_bringup" / "config" / "realsense" / "v8_cells.yaml"


@unittest.skipUnless(HAVE, "dependencias del contenedor")
class GridToolTest(unittest.TestCase):
    def test_edges_roundtrip_and_equal_thirds(self):
        xe, ye = edges_from_limits(0.405, 0.633, 0.31, 0.27)
        self.assertEqual(len(xe), 4)
        np.testing.assert_allclose(np.diff(xe), [0.076] * 3, atol=1e-4)
        np.testing.assert_allclose(np.diff(ye), [0.58 / 3] * 3, atol=1e-4)
        self.assertAlmostEqual(ye[0], -0.27)
        self.assertAlmostEqual(ye[-1], 0.31)
        np.testing.assert_allclose(limits_from_edges(xe, ye), (0.405, 0.633, 0.31, 0.27), atol=1e-4)
        self.assertIsNone(check_limits(0.405, 0.633, 0.31, 0.27))
        self.assertIsNotNone(check_limits(0.50, 0.52, 0.31, 0.27))       # demasiado baja
        self.assertIsNotNone(check_limits(0.40, 0.60, 0.02, 0.02))       # demasiado angosta

    def test_write_edges_keeps_the_rest_of_the_yaml(self):
        text = CELLS_YAML.read_text()
        new = write_edges_yaml(text, [0.4, 0.48, 0.56, 0.64], [-0.3, -0.1, 0.1, 0.3], "nota de prueba")
        old_cfg, new_cfg = yaml.safe_load(text)["v8_cells"], yaml.safe_load(new)["v8_cells"]
        self.assertEqual(new_cfg["x_edges_m"], [0.4, 0.48, 0.56, 0.64])
        self.assertEqual(new_cfg["y_edges_m"], [-0.3, -0.1, 0.1, 0.3])
        for k in old_cfg:                                         # el resto, igual
            if k not in ("x_edges_m", "y_edges_m"):
                self.assertEqual(new_cfg[k], old_cfg[k], k)
        old_comments = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
        new_comments = [ln for ln in new.splitlines() if ln.lstrip().startswith("#")]
        self.assertEqual(old_comments, new_comments)
        self.assertEqual(new.count("nota de prueba"), 2)

    def test_reach_model_matches_the_current_grid(self):
        d = default_paths()
        if not Path(d["calib"]).is_file():
            self.skipTest("falta la calibración de la cámara (datasets/vio_frames)")
        cfg = yaml.safe_load(CELLS_YAML.read_text())["v8_cells"]
        m = ReachModel(d["bridge"], d["alignment"], d["calib"], cfg["table_z_m"], cfg["check_heights_m"],
                       cfg.get("check_camera_yaws_deg", [0.0]))
        self.assertEqual(m.cell(0.42, 0.49, -0.07, 0.07)[0], 1.0)          # cerca, al centro: llega
        frac, clamp = m.cell(0.56, 0.63, -0.40, -0.30)                      # fuera de la caja |y| <= 0,30
        self.assertLess(frac, 1.0)
        self.assertGreater(clamp, 0.01)
        mk = yaml.safe_load((ROOT / "visuomotor_bringup/config/realsense/table_markers.yaml").read_text())
        txy = {int(i): np.asarray(c, float)[:, :2] for i, c in mk["table_markers"]["corners_base_m"].items()}
        self.assertAlmostEqual(marker_limit_x(txy, 0.03), 0.6335, places=3)

    def test_draw_overlay_colors(self):
        img = np.full((480, 640, 3), 200, np.uint8)
        H = np.array([[0.0, -1000.0, 320.0], [-1000.0, 0.0, 800.0], [0.0, 0.0, 1.0]])   # mesa vista desde arriba
        cells = {1: (0.40, 0.48, 0.05, 0.15), 3: (0.40, 0.48, -0.15, -0.05)}
        out = draw_overlay(img, H, cells, {1: (1.0, 0.0), 3: (0.0, 0.05)}, 0.63, ["linea"])
        # u = 320 - 1000 y, v = 800 - 1000 x: un punto del relleno (no el centro, donde va el número)
        v, u1, u3 = 800 - 420, 320 - 130, 320 + 130
        self.assertGreater(int(out[v, u1, 1]) - int(out[v, u1, 2]), 20)                # celda 1 verde
        v3 = 800 - 465                                                 # arriba del texto "no llega"
        self.assertGreater(int(out[v3, u3, 2]) - int(out[v3, u3, 1]), 20)              # celda 3 roja


if __name__ == "__main__":
    unittest.main()
