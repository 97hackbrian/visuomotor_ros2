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


if __name__ == "__main__":
    unittest.main()
