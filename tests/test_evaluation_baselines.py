"""Pruebas de las líneas base de la evaluación offline (visuomotor_core.training.evaluation).

Ejecutar (contenedor):  python3 -m pytest tests/test_evaluation_baselines.py -q
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_core"))

from visuomotor_core.training.evaluation import (  # noqa: E402
    constant_velocity_prediction, estimate_action_offset, evaluate_sampling)

H = 16


class _WindowDataset(torch.utils.data.Dataset):
    """Ventanas sintéticas (2 observaciones + horizonte de 16 acciones absolutas de 9 dims) de un movimiento recto."""

    def __init__(self, n=10, speed=0.002, accel=0.0001):
        self.items = []
        rng = np.random.default_rng(0)
        for i in range(n):
            p0 = rng.normal(scale=0.1, size=3)
            v = rng.normal(size=3)
            v *= speed / np.linalg.norm(v)
            t = np.arange(-1, H + 1, dtype=np.float64)          # estados t-1, t y acciones t .. t+H-1
            pos = p0[None] + t[:, None] * v[None] + 0.5 * accel * (t[:, None] ** 2)  # con aceleración: CV no es exacta
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
            full = np.concatenate([pos, quat, np.zeros((len(t), 1))], axis=1)
            state = full[:2]
            action = np.concatenate([full[1:1 + H], np.zeros((H, 1))], axis=1)   # action[t] = state[t] (offset 0)
            self.items.append({"observation.image": torch.zeros(2, 3, 8, 8),
                               "observation.state": torch.tensor(state, dtype=torch.float32),
                               "action": torch.tensor(action, dtype=torch.float32)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _ConstantVelocityPolicy:
    """Política falsa que predice exactamente la línea base de velocidad constante (pasos 1..H-1)."""
    training = False

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def predict_chunk(self, obs):
        st = obs["observation.state"]
        pos = constant_velocity_prediction(st, H - 1, 0)
        b = pos.shape[0]
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pos.device).expand(b, H - 1, 4)
        return torch.cat([pos, quat, torch.zeros(b, H - 1, 2, device=pos.device)], dim=-1)


class TestConstantVelocity(unittest.TestCase):
    def test_prediction_offsets(self):
        st = torch.zeros(1, 2, 8)
        st[0, 1, :3] = torch.tensor([0.01, -0.02, 0.03])            # v = (0.01, -0.02, 0.03) por paso
        p0 = constant_velocity_prediction(st, 4, 0)[0]
        p1 = constant_velocity_prediction(st, 4, 1)[0]
        v = st[0, 1, :3]
        for j in range(4):
            torch.testing.assert_close(p0[j], v + (j + 1) * v)
            torch.testing.assert_close(p1[j], v + (j + 2) * v)

    def test_single_observation_is_hold(self):
        st = torch.ones(3, 1, 8)
        torch.testing.assert_close(constant_velocity_prediction(st, 5, 0), torch.ones(3, 5, 3))

    def test_estimate_action_offset(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            g = zarr.open_group(str(Path(tmp) / "d.zarr"), mode="w")
            for name, shift in (("same", 0), ("next", 1)):
                s = np.cumsum(rng.normal(scale=0.01, size=(50, 8)), axis=0)
                a = np.concatenate([s[shift:], np.repeat(s[-1:], shift, axis=0)], axis=0)[:, :8]
                a = np.concatenate([a, np.zeros((50, 1))], axis=1)
                ep = g.create_group(f"episode_{name}")
                ep.create_dataset("state", data=s)
                ep.create_dataset("action", data=a)
            self.assertEqual(estimate_action_offset(g, ["episode_same"]), 0)
            self.assertEqual(estimate_action_offset(g, ["episode_next"]), 1)


class TestEvaluateSamplingBaselines(unittest.TestCase):
    def test_policy_equal_to_baseline_matches_cv_metrics(self):
        ds = _WindowDataset()
        m = evaluate_sampling(_ConstantVelocityPolicy(), ds, list(range(len(ds))), torch.device("cpu"),
                              n_action_steps=3, batch_size=4, num_workers=0)
        for k in ("ade_mm", "ade_exec_mm", "fde_mm"):
            self.assertAlmostEqual(m[k], m[f"cv_{k}"], places=4)
        self.assertGreater(m["cv_fde_mm"], 0.0)                      # la aceleración hace que la CV no sea exacta
        self.assertGreater(m["hold_fde_mm"], m["cv_fde_mm"])         # pero mejor que quedarse quieto
        self.assertTrue(np.isfinite(m["hold_fde_mm"]))

    def test_exact_constant_velocity_motion_has_zero_cv_error(self):
        ds = _WindowDataset(accel=0.0)
        m = evaluate_sampling(_ConstantVelocityPolicy(), ds, list(range(len(ds))), torch.device("cpu"),
                              n_action_steps=3, batch_size=5, num_workers=0)
        self.assertLess(m["cv_ade_mm"], 1e-3)
        self.assertGreater(m["hold_ade_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
