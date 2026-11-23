"""Inferencia con CUDA Graphs (DiffusionModel.enable_cuda_graph): mismo resultado que la ejecución normal.

Usa el modelo desplegado (deploy/jetson/model, pesos fuera de git) y una GPU; si falta alguno, se omite.
"""

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visuomotor_core"))
MODEL = ROOT / "deploy" / "jetson" / "model"

from visuomotor_core.models.diffusion_policy import DiffusionPolicy  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available() and (MODEL / "model.safetensors").exists(),
                     "requiere CUDA y deploy/jetson/model/model.safetensors")
class CudaGraphInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = DiffusionPolicy.from_pretrained(MODEL).cuda().eval()
        g = torch.Generator(device="cuda").manual_seed(0)
        state = torch.zeros((1, 2, 8), device="cuda")
        state[..., 0:3] = torch.tensor([0.05, -0.02, -0.10], device="cuda")
        state[..., 3:7] = torch.tensor([0.0, 0.7071068, 0.0, 0.7071068], device="cuda")
        cls.batch = {"observation.image": torch.rand((1, 2, 3, 256, 256), device="cuda", generator=g),
                     "observation.state": state}

    def _run(self, steps):
        self.policy.diffusion.num_inference_steps = steps
        torch.manual_seed(123)
        with torch.no_grad():
            return self.policy.predict_chunk(dict(self.batch))

    def test_graph_matches_eager_and_recaptures(self):
        p = self.policy
        p.diffusion.disable_cuda_graph()
        eager5 = self._run(5)
        eager3 = self._run(3)
        self.assertTrue(p.enable_cuda_graph(batch_size=1))
        graph5 = self._run(5)
        self.assertEqual(p.diffusion._cuda_graph["steps"], 5)
        graph3 = self._run(3)                                  # otro número de pasos: recaptura
        self.assertEqual(p.diffusion._cuda_graph["steps"], 3)
        torch.testing.assert_close(graph5, eager5, rtol=0, atol=1e-5)
        torch.testing.assert_close(graph3, eager3, rtol=0, atol=1e-5)
        # el ruido se genera fuera del grafo: otra semilla da otra muestra (sigue siendo estocástico)
        torch.manual_seed(7)
        with torch.no_grad():
            other = p.predict_chunk(dict(self.batch))
        self.assertGreater((other - graph3).abs().max().item(), 1e-4)
        p.diffusion.disable_cuda_graph()

    def test_training_mode_and_deepcopy_unaffected(self):
        p = self.policy
        p.enable_cuda_graph(batch_size=1)
        p.diffusion.disable_cuda_graph()
        copy.deepcopy(p)                                       # el EMA del entrenamiento copia el modelo
        p.diffusion.train()
        self.assertIsNone(p.diffusion._sample_with_cuda_graph(self.batch))   # en entrenamiento nunca usa el grafo
        p.diffusion.eval()


if __name__ == "__main__":
    unittest.main()
