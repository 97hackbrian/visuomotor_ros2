#!/usr/bin/env python3
"""Genera el tablero ChArUco de la celda como PDF (y PNG) a escala 1:1, listo para imprimir.

Lee la geometría de ``visuomotor_bringup/config/realsense/charuco_board.yaml`` (fuente única) y la centra en una
hoja A4 o A3 apaisada. Añade una regla de 100 mm para comprobar la escala de la impresión. Después de imprimir hay
que medir con calibre y escribir las medidas reales en el YAML (``measured: true``).

Uso (contenedor):
    python3 tools/hardware/make_charuco_board.py                    # A4 -> docs/hardware/charuco_board.{pdf,png}
    python3 tools/hardware/make_charuco_board.py --paper A3 --out-dir /tmp/board
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visuomotor_bringup.charuco import DEFAULT_CONFIG, CharucoConfig, board_image, make_board  # noqa: E402

PAPERS_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0)}           # apaisado (ancho, alto)
INK = "#52514e"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--paper", choices=sorted(PAPERS_MM), default="A4")
    ap.add_argument("--dpi", type=int, default=600, help="resolución del tablero dentro del PDF")
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "hardware"))
    args = ap.parse_args(argv)

    cfg = CharucoConfig.load(args.config)
    board = make_board(cfg)
    w_mm, h_mm = (1000.0 * v for v in cfg.size_m)
    pw, ph = PAPERS_MM[args.paper]
    if round(w_mm, 3) > pw - 20.0 or round(h_mm, 3) > ph - 35.0:
        print(f"El tablero ({w_mm:.0f} x {h_mm:.0f} mm) no cabe en {args.paper} con márgenes: usar --paper A3 o "
              f"reducir square_length_m.", file=sys.stderr)
        return 2
    px = (int(round(w_mm / 25.4 * args.dpi)), int(round(h_mm / 25.4 * args.dpi)))
    img = board_image(board, px, margin_px=0)

    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4))
    left, bottom = (pw - w_mm) / 2.0, (ph - h_mm) / 2.0 + 8.0
    ax = fig.add_axes([left / pw, bottom / ph, w_mm / pw, h_mm / ph])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(0, w_mm, 0, h_mm))
    ax.set_axis_off()
    # Regla de 100 mm (comprobar la escala de impresión)
    rx, ry = left / pw, (bottom - 10.0) / ph
    fig.add_artist(plt.Line2D([rx, rx + 100.0 / pw], [ry, ry], color="black", lw=1.2, transform=fig.transFigure))
    for xx in (rx, rx + 100.0 / pw):
        fig.add_artist(plt.Line2D([xx, xx], [ry - 1.5 / ph, ry + 1.5 / ph], color="black", lw=1.2,
                                  transform=fig.transFigure))
    fig.text(rx + 102.0 / pw, ry, "100 mm (print at 100 %, no fit-to-page)", va="center", fontsize=8, color=INK)
    fig.text(left / pw, (bottom - 18.0) / ph,
             f"ChArUco {cfg.squares_x}x{cfg.squares_y}, {cfg.dictionary}, square {cfg.square_length_m * 1000:.1f} mm, "
             f"marker {cfg.marker_length_m * 1000:.1f} mm (nominal). Measure with calipers and write the measured "
             f"values in charuco_board.yaml (measured: true).", fontsize=7.5, color=INK)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_board_{cfg.squares_x}x{cfg.squares_y}_{cfg.square_length_m * 1000:.0f}mm_{args.paper}"
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"Tablero {cfg.squares_x}x{cfg.squares_y} ({w_mm:.0f} x {h_mm:.0f} mm) en {args.paper}: "
          f"{out / (stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
