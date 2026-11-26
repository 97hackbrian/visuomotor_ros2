"""Pruebas del verificador de datasets de entrenamiento xArm5 (tools/xarm5/verify_training_dataset.py), sin GPU."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "xarm5"))

from verify_training_dataset import pose_sources  # noqa: E402


class PoseSourcesTest(unittest.TestCase):
    """2026-11-26, a pedido del usuario: el dataset que se entrena pasa por DROID. ``--require-pose-source droid_rgbd``
    usa este conteo: un episodio sin el atributo trae las poses de la odometría en vivo."""

    def test_groups_episodes_by_pose_source(self):
        import zarr
        with tempfile.TemporaryDirectory() as d:
            g = zarr.open_group(str(Path(d) / "ds.zarr"), mode="w")
            g.create_group("episode_0").attrs["pose_source"] = "droid_rgbd"
            g.create_group("episode_1").attrs["pose_source"] = "droid_rgbd"
            g.create_group("episode_2")
            by = pose_sources(g, ["episode_0", "episode_1", "episode_2"])
        self.assertEqual(by, {"droid_rgbd": ["episode_0", "episode_1"], "sin pose_source": ["episode_2"]})


if __name__ == "__main__":
    unittest.main()
