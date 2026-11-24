"""Cadena de medición de la odometría sin cámara: bag sintético -> referencia del tablero -> métricas -> reprocesado
con RTAB-Map (tools/hardware/make_synthetic_vio_bag.py, evaluate_pose_estimates.py, replay_odometry_candidates.py).

El bag sintético tiene verdad conocida y una odometría "en vivo" con +3 % de escala y un tramo perdido. Se comprueba:
* la referencia del tablero coincide con la verdad (mm);
* el evaluador recupera la escala y las pérdidas inyectadas;
* el reprocesado con RTAB-Map produce una odometría completa y razonable (prueba lenta, ~30 s).
Requiere ROS 2 (``source /opt/ros/jazzy/setup.bash``); el reprocesado requiere además ``rtabmap_odom``.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
try:
    import rosbag2_py  # noqa: F401
    import rclpy  # noqa: F401
    HAVE_ROS = True
except Exception:  # noqa: BLE001
    HAVE_ROS = False


def _have_rtabmap() -> bool:
    if not shutil.which("ros2"):
        return False
    r = subprocess.run(["ros2", "pkg", "prefix", "rtabmap_odom"], capture_output=True, text=True)
    return r.returncode == 0


@unittest.skipUnless(HAVE_ROS, "ROS 2 no disponible (source /opt/ros/jazzy/setup.bash)")
class VioMeasurementPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import make_synthetic_vio_bag
        cls.tmp = Path(tempfile.mkdtemp())
        cls.bag = cls.tmp / "synthetic_run"
        assert make_synthetic_vio_bag.main(["--out", str(cls.bag), "--depth", "compressedDepth"]) == 0

    def test_board_reference_and_live_metrics(self):
        import evaluate_pose_estimates
        out = self.tmp / "eval"
        rc = evaluate_pose_estimates.main(["--run", str(self.bag), "--ground-truth-topic", "/ground_truth/optical_pose",
                                           "--allow-nominal", "--out-dir", str(out)])
        self.assertEqual(rc, 0)
        d = json.loads((out / "pose_evaluation.json").read_text())
        run = d["runs"]["synthetic_run"]
        # La referencia del tablero (A3 9x7) contra la verdad: pocos mm
        self.assertLess(run["board_vs_truth"]["ate_anchored_mm"]["median"], 3.0)
        self.assertLess(abs(run["board_vs_truth"]["scale_error_pct"]), 0.5)
        # La odometría en vivo simulada: +3 % de escala, 10 de 255 frames nulos
        self.assertAlmostEqual(run["live"]["scale_error_pct"], 3.0, delta=0.6)
        self.assertAlmostEqual(run["live"]["est_lost_pct_all"], 100.0 * 10 / 255, delta=0.5)
        self.assertAlmostEqual(run["live_vs_truth"]["scale_error_pct"], 3.0, delta=0.3)
        self.assertTrue((out / "pose_evaluation.png").exists())
        self.assertIn("live", d["summary"])

    @unittest.skipUnless(_have_rtabmap(), "rtabmap_odom no disponible")
    def test_replay_baseline_with_rtabmap(self):
        import replay_odometry_candidates
        out = self.tmp / "replay"
        rc = replay_odometry_candidates.main(["--run", str(self.bag), "--candidates", "baseline", "--out-dir", str(out),
                                              "--allow-nominal", "--ground-truth-topic", "/ground_truth/optical_pose"])
        self.assertEqual(rc, 0)
        d = json.loads((out / "evaluation" / "pose_evaluation.json").read_text())
        base = d["runs"]["synthetic_run"]["baseline_vs_truth"]
        self.assertGreater(base["pairs"], 0.9 * 255)                 # procesó casi todos los frames
        self.assertLess(base["est_lost_pct_all"], 5.0)
        self.assertLess(base["ate_anchored_mm"]["median"], 20.0)     # escena sintética fácil; holgado
        self.assertLess(abs(base["scale_error_pct"]), 2.0)


if __name__ == "__main__":
    unittest.main()
