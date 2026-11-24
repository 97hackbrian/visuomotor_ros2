"""Pruebas del tablero ChArUco (visuomotor_bringup.charuco): se renderiza el tablero con una cámara virtual de pose
conocida y se comprueba que la pose estimada la recupera."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_bringup"))

import cv2  # noqa: E402

from visuomotor_bringup.charuco import (  # noqa: E402
    DEFAULT_CONFIG, CharucoConfig, board_image, chessboard_corners, detect_corners, estimate_pose, make_board)

K = np.array([[605.0, 0, 320.0], [0, 605.0, 240.0], [0, 0, 1.0]])


def _render(cfg, board, r_cb, t_cb, ppm=4000.0):
    """Imagen 640x480 del tablero visto con ``p_cam = r_cb p_board + t_cb``.

    La relación entre el marco del tablero y los píxeles de su imagen plana se obtiene detectando las esquinas en
    esa imagen (independiente de la convención de ejes de cada versión de OpenCV)."""
    w_m, h_m = cfg.size_m
    flat = board_image(board, (int(w_m * ppm), int(h_m * ppm)))
    px, ids = detect_corners(flat, board, cfg)
    obj = chessboard_corners(board)[ids][:, :2]
    a, _ = cv2.estimateAffine2D(obj.astype(np.float32), px.astype(np.float32))   # tablero (m) -> píxel plano
    a3 = np.vstack([a, [0, 0, 1]])
    h = K @ np.c_[r_cb[:, 0], r_cb[:, 1], t_cb] @ np.linalg.inv(a3)              # píxel plano -> cámara
    return cv2.warpPerspective(flat, h, (640, 480), flags=cv2.INTER_AREA, borderValue=200)


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


class CharucoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = CharucoConfig.load(DEFAULT_CONFIG)
        cls.board = make_board(cls.cfg)

    def test_config_rejects_even_rows(self):
        d = {"dictionary": "DICT_5X5_100", "squares_x": 7, "squares_y": 6, "square_length_m": 0.03,
             "marker_length_m": 0.02}
        with self.assertRaises(ValueError):
            CharucoConfig.from_dict(d)

    def _front_facing(self):
        """Rotación base con la que la cámara ve la CARA impresa del tablero. Según la versión de OpenCV el marco del
        tablero tiene la y hacia abajo (z hacia dentro del papel) o hacia arriba (z hacia fuera)."""
        w, h = self.cfg.size_m
        flat = board_image(self.board, (int(w * 4000), int(h * 4000)))
        px, ids = detect_corners(flat, self.board, self.cfg)
        obj = chessboard_corners(self.board)[ids][:, :2]
        a, _ = cv2.estimateAffine2D(obj.astype(np.float32), px.astype(np.float32))
        return np.eye(3) if np.linalg.det(a[:, :2]) > 0 else _rot(math.pi, 0.0, 0.0)

    def test_pose_recovered_from_synthetic_view(self):
        w, h = self.cfg.size_m
        base = self._front_facing()
        for (rx, ry, rz, dist) in ((0.35, 0.1, 0.2, 0.55), (-0.2, -0.25, -0.4, 0.8)):
            r = base @ _rot(rx, ry, rz)
            # centro del tablero delante de la cámara a la distancia dada
            t = np.array([0.0, 0.0, dist]) - r @ np.array([w / 2, h / 2, 0.0])
            img = _render(self.cfg, self.board, r, t)
            pose = estimate_pose(img, self.board, self.cfg, K)
            self.assertIsNotNone(pose)
            self.assertGreaterEqual(pose.n_corners, 12)
            self.assertLess(pose.reprojection_error_px, 1.0)
            ang = math.degrees(math.acos(np.clip((np.trace(pose.rotation.T @ r) - 1) / 2, -1, 1)))
            self.assertLess(ang, 0.5)
            self.assertLess(np.linalg.norm(pose.translation - t), 0.003)   # < 3 mm a 0.55-0.8 m

    def test_no_board_returns_none(self):
        self.assertIsNone(estimate_pose(np.full((480, 640), 200, np.uint8), self.board, self.cfg, K))


class BoardViewCheckTest(unittest.TestCase):
    """tools/hardware/check_board_view.py: esquinas, distancia y ángulo con una vista sintética de pose conocida."""

    def test_stats_on_synthetic_view(self):
        sys.path.insert(0, str(ROOT / "tools" / "hardware"))
        from check_board_view import board_view_stats, summary
        cfg = CharucoConfig.load(DEFAULT_CONFIG)
        board = make_board(cfg)
        base = CharucoTest._front_facing(type("T", (), {"cfg": cfg, "board": board})())
        center = chessboard_corners(board).mean(0)
        r = base @ _rot(math.radians(25), 0.0, 0.0)
        t = np.array([0.0, 0.0, 0.75]) - r @ center                    # centro del tablero a 0,75 m, 25° de visión
        img = _render(cfg, board, r, t)
        st = board_view_stats(img, board, cfg, K)
        self.assertEqual(st["corners"], st["total"])
        self.assertAlmostEqual(st["distance_m"], 0.75, delta=0.01)
        self.assertAlmostEqual(st["view_angle_deg"], 25.0, delta=1.0)
        self.assertIn("OK", summary(st, {"aligned": True, "error_px": 2.0}))

class BoardPdfTest(unittest.TestCase):
    """El PDF para imprimir: estructura válida, tamaño de página y del tablero exactos, e imagen idéntica al tablero
    (el backend PDF de matplotlib 3.6 rellenaba mal las filas de las imágenes de 1 bit y el tablero salía corrupto)."""

    def test_pdf_board_is_exact(self):
        import re
        import tempfile
        import zlib
        sys.path.insert(0, str(ROOT / "tools" / "hardware"))
        import make_charuco_board
        out = Path(tempfile.mkdtemp())
        self.assertEqual(make_charuco_board.main(["--paper", "A3", "--dpi", "150", "--out-dir", str(out)]), 0)
        pdf = next(out.glob("*.pdf")).read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF"))
        mb = [float(v) for v in re.search(rb"/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]", pdf).groups()]
        np.testing.assert_allclose(mb, [420 / 25.4 * 72, 297 / 25.4 * 72], atol=0.01)      # A3 apaisado
        cfg = CharucoConfig.load(DEFAULT_CONFIG)
        w_mm, h_mm = (1000 * v for v in cfg.size_m)
        cm = [float(v) for v in re.search(rb"q ([\d.]+) 0 0 ([\d.]+) [\d.]+ [\d.]+ cm /Im0 Do", pdf).groups()]
        np.testing.assert_allclose(cm, [w_mm / 25.4 * 72, h_mm / 25.4 * 72], atol=0.01)    # tamaño impreso exacto
        m = re.search(rb"/Width (\d+) /Height (\d+) .*?/Length (\d+) >>\nstream\n", pdf, re.S)
        w, h, n = (int(v) for v in m.groups())
        data = zlib.decompress(pdf[m.end():m.end() + n])
        self.assertEqual(len(data), h * ((w + 7) // 8))                                     # filas completas al byte
        bits = np.unpackbits(np.frombuffer(data, np.uint8).reshape(h, -1), axis=1)[:, :w]
        ref = board_image(make_board(cfg), (w, h)) > 127
        self.assertTrue(np.array_equal(bits.astype(bool), ref))
        # El tablero se detecta completo en la imagen del PDF
        px, ids = detect_corners(np.where(bits > 0, 255, 0).astype(np.uint8), make_board(cfg), cfg)
        self.assertEqual(len(ids), (cfg.squares_x - 1) * (cfg.squares_y - 1))


if __name__ == "__main__":
    unittest.main()
