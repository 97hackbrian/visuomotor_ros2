# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Pruebas unitarias para el módulo de análisis de convergencia y criterios de parada.
Verifica robustez numérica, casos de borde, detección sobre señales sintéticas y reales,
conversiones bidireccionales de unidades (Pasos <-> Épocas) y generación de gráficos.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from visuomotor_core.analysis.stopping_criteria import (
    ConvergenceAnalyzer,
    EMAGapDetector,
    GradientFlatnessDetector,
    KneedleDetector,
    PrecheltProgressDetector,
    RelativeImprovementDetector,
    VarianceStabilizationDetector,
    epoch_to_step,
    smooth_curve_ema,
    step_to_epoch,
)


class TestStoppingCriteria(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_conv_"))
        np.random.seed(42)

        # Generar señal de entrenamiento sintética verosímil:
        # Caída exponencial inicial (1.2 -> 0.05) + Plateau ruidoso alrededor de 0.03
        self.num_steps = 1000
        self.steps_per_epoch = 100
        self.steps = np.arange(self.num_steps)

        # Pérdida sintética realista con decaimiento rápido y plateau
        decay = 1.15 * np.exp(-self.steps / 120.0)
        plateau = 0.035 + 0.005 * np.random.randn(self.num_steps)
        self.losses = np.maximum(0.001, decay + plateau)

        self.step_history = [
            {
                "step": int(s),
                "epoch": int(s // self.steps_per_epoch) + 1,
                "loss": float(self.losses[s]),
                "loss_ema": float(self.losses[s]),
                "lr": 1e-4,
                "vram_peak_mb": 1500.0,
            }
            for s in range(self.num_steps)
        ]

        self.epoch_history = [
            {
                "epoch": ep + 1,
                "avg_loss": float(np.mean(self.losses[ep * 100 : (ep + 1) * 100])),
                "final_lr": 1e-4,
                "duration_s": 25.0,
            }
            for ep in range(10)
        ]

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_unit_conversions(self):
        steps_per_epoch = 250
        # Paso 0 -> Época 0.0 (inicio)
        self.assertAlmostEqual(step_to_epoch(0, steps_per_epoch), 0.0)
        # Paso 250 -> Época 1.0 (exactamente 1 época completada)
        self.assertAlmostEqual(step_to_epoch(250, steps_per_epoch), 1.0)
        # Época 0.0 -> Paso 0
        self.assertEqual(epoch_to_step(0.0, steps_per_epoch), 0)
        # Época 1.0 -> Paso 250
        self.assertEqual(epoch_to_step(1.0, steps_per_epoch), 250)
        # Época 2.0 -> Paso 500
        self.assertEqual(epoch_to_step(2.0, steps_per_epoch), 500)

        # Arrays
        st_arr = np.array([0, 250, 500])
        ep_arr = step_to_epoch(st_arr, steps_per_epoch)
        np.testing.assert_allclose(ep_arr, [0.0, 1.0, 2.0])

        # Caso límite: steps_per_epoch = 0
        self.assertEqual(step_to_epoch(10, 0), 0.0)
        self.assertEqual(epoch_to_step(2.0, 0), 0)

    def test_smooth_curve_ema(self):
        raw = np.array([1.0, 2.0, 3.0, 4.0])
        smoothed = smooth_curve_ema(raw, alpha=0.5)
        self.assertEqual(len(smoothed), 4)
        self.assertAlmostEqual(smoothed[0], 1.0)
        self.assertAlmostEqual(smoothed[1], 1.5)

        # Vacío
        self.assertEqual(len(smooth_curve_ema(np.array([]))), 0)

    def test_kneedle_detector(self):
        detector = KneedleDetector()
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "kneedle_elbow")
        # El codo en exp(-x/120) debería encontrarse entre los pasos 80 y 400
        self.assertTrue(50 <= rec.recommended_step <= 400, f"Paso detectado: {rec.recommended_step}")
        self.assertTrue(0.5 <= rec.recommended_epoch <= 4.0)
        self.assertGreater(rec.confidence, 0.5)

    def test_gradient_flatness_detector(self):
        detector = GradientFlatnessDetector(threshold=1e-4, window_size=20, patience=10)
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "gradient_flatness")
        self.assertGreater(rec.recommended_step, 100)

    def test_variance_stabilization_detector(self):
        detector = VarianceStabilizationDetector(threshold_rsd=0.15, window_size=30, patience=10)
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "variance_stabilization")
        self.assertGreater(rec.recommended_step, 50)

    def test_relative_improvement_detector(self):
        detector = RelativeImprovementDetector(threshold=0.01, patience=15)
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "relative_improvement")

    def test_prechelt_progress_detector(self):
        detector = PrecheltProgressDetector(strip_length=20, threshold_pq=20.0, patience=5)
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "prechelt_progress")

    def test_ema_gap_detector(self):
        detector = EMAGapDetector(fast_alpha=0.10, slow_alpha=0.02, threshold_gap=0.05, patience=10)
        rec = detector.detect(self.steps, self.losses, self.steps_per_epoch)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method_id, "ema_gap")
        self.assertGreater(rec.recommended_step, 100)

    def test_convergence_analyzer_full_pipeline(self):
        analyzer = ConvergenceAnalyzer(
            steps_per_epoch=self.steps_per_epoch,
            step_history=self.step_history,
            epoch_history=self.epoch_history,
        )
        recs = analyzer.analyze()
        self.assertIn("kneedle_elbow", recs)
        self.assertIn("gradient_flatness", recs)
        self.assertIn("variance_stabilization", recs)
        self.assertIn("relative_improvement", recs)
        self.assertIn("prechelt_progress", recs)
        self.assertIn("ema_gap", recs)
        self.assertIn("consensus", recs)

        # Verificar tabla de resumen
        summary = analyzer.generate_summary_table()
        self.assertIn("EVALUACIÓN COMPARATIVA", summary)
        self.assertIn("Consenso", summary)

        # Guardar reportes
        report_json = analyzer.save_report(self.tmp_dir)
        self.assertTrue(report_json.exists())
        self.assertTrue((self.tmp_dir / "convergence_summary.csv").exists())

        with open(report_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["steps_per_epoch"], self.steps_per_epoch)
            self.assertIn("consensus", data["recommendations"])

        # Generar gráficos dual axis
        png_path, pdf_path = analyzer.plot_dual_axis(self.tmp_dir, dpi=100, show_diagnostics=True)
        self.assertTrue(png_path.exists())
        self.assertTrue(pdf_path.exists())
        self.assertTrue((self.tmp_dir / "convergence_diagnostics_4panel.png").exists())
    def test_cli_analyze_convergence(self):
        # Escribir CSVs sintéticos en tmp_dir
        import csv
        import subprocess
        step_csv = self.tmp_dir / "training_metrics.csv"
        with open(step_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "loss", "loss_ema", "lr", "vram_peak_mb", "timestamp"])
            for r in self.step_history:
                w.writerow([r["step"], r["epoch"], r["loss"], r["loss_ema"], r["lr"], r["vram_peak_mb"], "2026-11-02 12:00:00"])

        cmd = [
            "python3",
            "tools/analyze_convergence.py",
            "--run_dir",
            str(self.tmp_dir),
            "--steps_per_epoch",
            str(self.steps_per_epoch),
            "--dpi",
            "100",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("EVALUACIÓN COMPARATIVA", res.stdout)
        self.assertTrue((self.tmp_dir / "convergence_analysis" / "convergence_report.json").exists())
        self.assertTrue((self.tmp_dir / "convergence_analysis" / "convergence_stopping_criteria_dual_axis.png").exists())
        self.assertTrue((self.tmp_dir / "convergence_analysis" / "convergence_stopping_criteria_dual_axis.pdf").exists())


if __name__ == "__main__":
    unittest.main()
