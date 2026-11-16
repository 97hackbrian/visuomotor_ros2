#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
Herramienta de análisis de convergencia y evaluación de criterios de parada (Stopping Criteria)
para modelos de Diffusion Policy entrenados en el framework visuomotor_ros2.

Permite analizar cualquier corrida de entrenamiento pasada o presente desde sus métricas CSV:
1. Evalúa los 5 métodos SOTA (Kneedle, Flatness |dL/dt|, Varianza RSD, Early Stop ΔL/L, Prechelt PQ).
2. Reporta la convergencia sincrónica en Épocas y Pasos de Optimización (Steps).
3. Genera figuras dual-axis de calidad de publicación a 300 DPI (PNG y PDF vectorial).
4. Exporta resumen en JSON estructurado y CSV tabular.
"""

import argparse
import sys
from pathlib import Path

# Añadir visuomotor_core al path de Python si no está instalado como paquete editable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "visuomotor_core"))

from visuomotor_core.analysis.stopping_criteria import ConvergenceAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa convergencia y criterios de parada sobre curvas de entrenamiento de Diffusion Policy."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directorio de salida del entrenamiento (contiene training_metrics.csv y/o epoch_metrics.csv).",
    )
    parser.add_argument(
        "--step_csv",
        type=str,
        default=None,
        help="Ruta explícita al archivo training_metrics.csv.",
    )
    parser.add_argument(
        "--epoch_csv",
        type=str,
        default=None,
        help="Ruta explícita al archivo epoch_metrics.csv.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Cantidad de pasos de optimización que equivalen a 1 época (si no se especifica, se infiere).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directorio destino para las figuras y reportes (por defecto el mismo que run_dir).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución de las figuras exportadas (por defecto 300 DPI).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Omitir la renderización de figuras de Matplotlib (solo imprimir tabla y JSON).",
    )

    args = parser.parse_args()

    # Resolver rutas de entrada
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        step_csv = Path(args.step_csv) if args.step_csv else run_dir / "training_metrics.csv"
        epoch_csv = Path(args.epoch_csv) if args.epoch_csv else run_dir / "epoch_metrics.csv"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "convergence_analysis"
    else:
        if not args.step_csv:
            print("[ERROR] Debe especificarse --run_dir o --step_csv.")
            sys.exit(1)
        step_csv = Path(args.step_csv)
        epoch_csv = Path(args.epoch_csv) if args.epoch_csv else None
        out_dir = Path(args.out_dir) if args.out_dir else step_csv.parent / "convergence_analysis"

    if not step_csv.exists():
        print(f"[ERROR] No se encontró el archivo de métricas por paso: {step_csv}")
        sys.exit(1)

    print(f"\n📂 Analizando métricas de entrenamiento desde: {step_csv}")
    if epoch_csv and epoch_csv.exists():
        print(f"📊 Métricas de época encontradas: {epoch_csv}")

    analyzer = ConvergenceAnalyzer.from_csv_files(
        step_csv=step_csv,
        epoch_csv=epoch_csv if (epoch_csv and epoch_csv.exists()) else None,
        steps_per_epoch=args.steps_per_epoch,
    )

    recs = analyzer.analyze()
    if not recs:
        print("[WARN] No se pudieron generar recomendaciones de parada (datos insuficientes).")
        sys.exit(0)

    # Imprimir tabla comparativa
    summary_table = analyzer.generate_summary_table()
    print("\n" + summary_table + "\n")

    # Guardar reportes JSON y CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = analyzer.save_report(out_dir)
    print(f"💾 Reportes exportados en:\n  - JSON: {report_json}\n  - CSV:  {out_dir / 'convergence_summary.csv'}")

    # Renderizar gráficos de calidad de publicación
    if not args.no_plots:
        print(f"🎨 Renderizando figuras científicas dual-axis a {args.dpi} DPI...")
        png_path, pdf_path = analyzer.plot_dual_axis(out_dir, dpi=args.dpi, show_diagnostics=True)
        print(f"✅ Figuras generadas exitosamente:")
        print(f"  - Canónica PNG: {png_path}")
        print(f"  - Canónica PDF: {pdf_path}")
        print(f"  - Diagnósticos 4-Panel PNG: {out_dir / 'convergence_diagnostics_4panel.png'}")
        print(f"  - Diagnósticos 4-Panel PDF: {out_dir / 'convergence_diagnostics_4panel.pdf'}")


if __name__ == "__main__":
    main()
