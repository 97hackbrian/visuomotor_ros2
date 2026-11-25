"""Pruebas de la guía de inicio a HOME3 (visuomotor_data_collection.vo_start_guide), sin ROS.

Se dibujan los dos ArUco de la mesa donde los ve la cámara en HOME3 y se comprueba que:
* la imagen de referencia da error ~0 y "alineada";
* un corrimiento, una escala o un giro conocidos se recuperan, y la pista apunta en la dirección correcta;
* la misma vista a otra resolución sigue alineada;
* la referencia que genera ``build_reference`` coincide con las esquinas dibujadas.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))

import cv2  # noqa: E402

from visuomotor_data_collection.vo_start_guide import (  # noqa: E402
    ViewReference, build_reference, detect_markers, draw_guide, fit_similarity, summary_line, view_alignment)

REF_YAML = ROOT / "visuomotor_bringup" / "config" / "realsense" / "home3_view_reference.yaml"


def _marker(mid, n=120):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(d, mid, n)
    return cv2.aruco.drawMarker(d, mid, n)


def _render(markers, size=(640, 480)):
    """Mantel blanco con cada marcador deformado a sus 4 esquinas (orden de OpenCV: sup-izq, sup-der, inf-der,
    inf-izq del marcador)."""
    w, h = size
    img = np.full((h, w), 235, np.uint8)
    for mid, c in markers.items():
        m = _marker(mid)
        n = m.shape[0]
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        hm = cv2.getPerspectiveTransform(src, np.float32(c))
        warped = cv2.warpPerspective(m, hm, (w, h), flags=cv2.INTER_AREA, borderValue=255)
        mask = cv2.warpPerspective(np.full_like(m, 255), hm, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        img[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _transform(markers, s=1.0, th_deg=0.0, t=(0.0, 0.0)):
    allc = np.concatenate(list(markers.values()))
    c0 = allc.mean(0)
    th = math.radians(th_deg)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return {i: (s * (c - c0) @ r.T + c0 + np.asarray(t)) for i, c in markers.items()}


class StartGuideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ViewReference.load(REF_YAML)

    def test_reference_file(self):
        self.assertEqual(self.ref.image_size, (640, 480))
        self.assertEqual(sorted(self.ref.markers), [12, 14])
        self.assertLess(self.ref.repeatability_px, 2.0)          # el robot repite HOME3 con < 1 px

    def test_fit_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(100, 500, (8, 2))
        th = math.radians(4.0)
        r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        dst = 1.07 * src @ r.T + np.array([12.0, -5.0])
        s, th_est, t, res = fit_similarity(src, dst)
        self.assertAlmostEqual(s, 1.07, places=6)
        self.assertAlmostEqual(math.degrees(th_est), 4.0, places=5)
        np.testing.assert_allclose(t, [12.0, -5.0], atol=1e-6)
        self.assertLess(res, 1e-6)

    def test_rendered_reference_is_aligned(self):
        img = _render(self.ref.markers)
        live = detect_markers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.ref.dictionary)
        self.assertEqual(sorted(live), [12, 14])
        res = view_alignment(live, (640, 480), self.ref, tol_px=8.0)
        self.assertTrue(res["aligned"], res)
        self.assertLess(res["error_px"], 1.0)
        self.assertEqual(res["hints"], [])
        out = draw_guide(img, live, self.ref, res)
        self.assertEqual(out.shape, img.shape)
        self.assertIn("ALINEADA", summary_line(res))
        # Celda del programa (v8): polígono amarillo y flecha de la orientación dentro de la celda
        poly = [[200, 300], [300, 300], [300, 360], [200, 360]]
        out2 = draw_guide(img, live, self.ref, res, {"polygon_px": poly, "label": "celda 5", "yaw_deg": 90.0})
        self.assertEqual(out2.shape, img.shape)
        yellow = (out2[:, :, 2] > 200) & (out2[:, :, 1] > 180) & (out2[:, :, 0] < 60)
        self.assertGreater(int(yellow[290:370, 190:310].sum()), 200)

    def test_other_resolution(self):
        half = {i: c * 0.5 for i, c in self.ref.markers.items()}
        img = _render(half, size=(320, 240))
        live = detect_markers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.ref.dictionary)
        res = view_alignment(live, (320, 240), self.ref, tol_px=8.0)
        self.assertTrue(res["aligned"], res)

    def test_shift_scale_roll_hints(self):
        cases = [
            (dict(t=(30.0, 0.0)), "DERECHA"),            # los marcadores se ven a la derecha -> mover a la derecha
            (dict(t=(-30.0, 0.0)), "IZQUIERDA"),
            (dict(t=(0.0, 25.0)), "ABAJO"),
            (dict(s=1.15), "ALEJAR"),                     # la escena se ve más grande -> cámara demasiado cerca
            (dict(s=0.85), "ACERCAR"),
            (dict(th_deg=6.0), "HORARIO"),                # giro horario en la imagen -> rotar la cámara en horario
        ]
        for kw, word in cases:
            moved = _transform(self.ref.markers, **kw)
            img = _render(moved)
            live = detect_markers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.ref.dictionary)
            res = view_alignment(live, (640, 480), self.ref, tol_px=8.0)
            self.assertFalse(res["aligned"], (kw, res))
            self.assertTrue(any(word in h for h in res["hints"]), (kw, res["hints"]))
            if "s" in kw:
                self.assertAlmostEqual(res["scale"], kw["s"], delta=0.01)
            if "th_deg" in kw:
                self.assertAlmostEqual(res["roll_deg"], kw["th_deg"], delta=0.3)
            if "t" in kw:
                np.testing.assert_allclose(res["shift_px"], kw["t"], atol=1.0)

    def test_no_markers(self):
        res = view_alignment({}, (640, 480), self.ref)
        self.assertFalse(res["aligned"])
        self.assertIsNone(res["error_px"])
        self.assertTrue(res["hints"])

    def test_table_grid_homography(self):
        """Guía v8 (2026-11-25): grilla numerada SOBRE LA MESA con la homografía de los ArUco (esquinas en link_base de
        table_markers.yaml). En la vista de HOME3: reproduce los marcadores, +x (lejos de la base) va hacia arriba de la
        imagen, -y (derecha, +90°) hacia la derecha, y la celda pedida queda en amarillo con su flecha."""
        import yaml
        from visuomotor_data_collection.vo_schedule import blade_direction_xy, cells_from_edges
        from visuomotor_data_collection.vo_start_guide import project_table_xy, table_homography
        cfg_dir = ROOT / "visuomotor_bringup" / "config" / "realsense"
        mk = yaml.safe_load((cfg_dir / "table_markers.yaml").read_text())["table_markers"]["corners_base_m"]
        table_xy = {int(i): np.asarray(c, float)[:, :2] for i, c in mk.items()}
        cfg = yaml.safe_load((cfg_dir / "v8_cells.yaml").read_text())["v8_cells"]
        cells_m = cells_from_edges(cfg["x_edges_m"], cfg["y_edges_m"])
        H = table_homography(self.ref.markers, table_xy)
        self.assertIsNotNone(H)
        for i in table_xy:                                    # mismo orden de esquinas que OpenCV
            self.assertLess(np.abs(project_table_xy(H, table_xy[i]) - self.ref.markers[i]).max(), 3.0)
        self.assertIsNone(table_homography({12: self.ref.markers[12]}, table_xy))      # un solo marcador: no
        x0, x1, y0, y1 = cells_m[5]
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        uv = project_table_xy(H, [c, c + 0.05 * np.array(blade_direction_xy(0)), c + 0.05 * np.array(blade_direction_xy(90))])
        self.assertLess(uv[1, 1], uv[0, 1] - 5)               # 0° (lejos de la base) = hacia arriba en la imagen
        self.assertGreater(uv[2, 0], uv[0, 0] + 5)            # +90° = hacia la DERECHA en la imagen
        for cid in (1, 3, 7, 9):                              # todas las celdas dentro de la imagen
            q = project_table_xy(H, [(cells_m[cid][0], cells_m[cid][2]), (cells_m[cid][1], cells_m[cid][3])])
            self.assertTrue(((q >= 0) & (q < [640, 480])).all(), (cid, q))
        img = _render(self.ref.markers)
        live = detect_markers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.ref.dictionary)
        res = view_alignment(live, (640, 480), self.ref, tol_px=8.0)
        target = {"cells_m": cells_m, "table_xy": table_xy, "cell": 5, "yaw_deg": 90.0, "label": "COLOCAR: celda 5"}
        u, v = project_table_xy(H, [c])[0].astype(int)
        for lv, expect in ((live, True), ({}, False)):      # sin ArUco en vivo no se dibuja (quedaría flotando)
            out = draw_guide(img, lv, self.ref, res, target)
            yellow = (out[:, :, 2] > 200) & (out[:, :, 1] > 180) & (out[:, :, 0] < 90)
            self.assertEqual(int(yellow[v - 25:v + 25, u - 40:u + 40].sum()) > 150, expect)
            self.assertGreater(int(yellow[28:52].sum()), 50)                     # la etiqueta siempre

    def test_build_reference(self):
        rng = np.random.default_rng(1)
        imgs = []
        for _ in range(5):
            jitter = {i: c + rng.normal(0, 0.3, c.shape) for i, c in self.ref.markers.items()}
            imgs.append(_render(jitter))
        ref2 = build_reference(imgs)
        self.assertEqual(sorted(ref2.markers), [12, 14])
        for i in ref2.markers:
            self.assertLess(np.abs(ref2.markers[i] - self.ref.markers[i]).max(), 1.0)
        rt = ViewReference.from_dict(ref2.to_dict())
        self.assertEqual(sorted(rt.markers), [12, 14])


if __name__ == "__main__":
    unittest.main()
